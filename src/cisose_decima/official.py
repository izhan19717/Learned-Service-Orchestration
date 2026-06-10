"""Official Decima simulator adapter for the PyTorch port.

This module is intentionally explicit about the boundary between our PyTorch
actor and the authors' simulator. The simulator under `external/decima-sim` is
kept as the operational ground truth for arrivals, task durations, rewards, and
baseline heuristics; only the TensorFlow actor is replaced.
"""

from __future__ import annotations

import bisect
import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import torch

from cisose_decima.config import DEFAULT_CONFIG, DecimaConfig
from cisose_decima.model import DecimaPolicy
from cisose_decima.training import AveragePerStepReward, actor_loss, discount


ActionMode = Literal["sample", "greedy"]
Scheme = Literal["dynamic_partition", "spark_fifo", "pytorch_decima"]


@dataclass(frozen=True)
class OfficialModules:
    param: ModuleType
    environment: type
    dynamic_partition_agent: type
    spark_agent: type


@dataclass(frozen=True)
class OfficialActorTensors:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    adj_mats: tuple[torch.Tensor, ...]
    masks: tuple[torch.Tensor, ...]
    job_features: torch.Tensor
    node_valid_mask: torch.Tensor
    job_valid_mask: torch.Tensor
    dag_summ_backward_map: torch.Tensor
    dag_summary_mat: torch.Tensor
    running_dag_mat: torch.Tensor


@dataclass(frozen=True)
class OfficialStateTensors(OfficialActorTensors):
    job_dags: tuple[Any, ...]
    source_job: Any
    num_source_exec: int
    exec_map: dict[Any, int]
    action_map: Any
    frontier_nodes: Any
    exec_commit: Any
    moving_executors: Any


@dataclass(frozen=True)
class OfficialActionRecord:
    state: OfficialActorTensors
    node_action: int
    job_index: int
    executor_level_index: int


@dataclass(frozen=True)
class OfficialActionDecision:
    node: Any
    use_exec: int
    record: OfficialActionRecord | None


@dataclass
class OfficialExperienceStep:
    record: OfficialActionRecord
    reward: float
    wall_time: float


@dataclass(frozen=True)
class OfficialEpisodeResult:
    scheme: str
    seed: int
    num_stream_dags: int
    total_reward: float
    mean_job_completion_time: float
    median_job_completion_time: float
    num_finished_jobs: int
    makespan: float
    decisions: int
    elapsed_seconds: float


@dataclass(frozen=True)
class OfficialRolloutResult:
    seed: int
    steps: tuple[OfficialExperienceStep, ...]
    rewards: np.ndarray
    wall_times: np.ndarray
    total_reward: float
    mean_job_completion_time: float
    num_finished_jobs: int
    reset_hit: bool


@dataclass(frozen=True)
class OfficialTrainResult:
    checkpoint_path: Path
    final_epoch: int
    mean_total_reward: float
    mean_job_completion_time: float
    entropy_weight: float
    reset_prob: float


class DensePostman:
    """TensorFlow-free message-passing path builder matching Decima's Postman."""

    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.job_dags: tuple[Any, ...] = ()
        self.msg_mats: tuple[np.ndarray, ...] = ()
        self.msg_masks: tuple[np.ndarray, ...] = ()
        self.dag_summ_backward_map: np.ndarray | None = None
        self.running_dag_mat: np.ndarray | None = None

    def get_msg_path(
        self,
        job_dags: tuple[Any, ...],
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray, bool]:
        changed = len(self.job_dags) != len(job_dags) or not all(
            old is new for old, new in zip(self.job_dags, job_dags, strict=False)
        )
        if changed:
            self.msg_mats, self.msg_masks = _get_msg_path(job_dags, self.max_depth)
            self.dag_summ_backward_map = _get_dag_summ_backward_map(job_dags)
            self.running_dag_mat = _get_running_dag_mat(job_dags)
            self.job_dags = tuple(job_dags)
        assert self.dag_summ_backward_map is not None
        assert self.running_dag_mat is not None
        return (
            self.msg_mats,
            self.msg_masks,
            self.dag_summ_backward_map,
            self.running_dag_mat,
            changed,
        )


class OfficialDecimaAgent:
    """PyTorch actor with the official Decima simulator action contract."""

    def __init__(
        self,
        policy: DecimaPolicy,
        *,
        config: DecimaConfig = DEFAULT_CONFIG,
        device: str | torch.device = "cpu",
        action_mode: ActionMode = "sample",
        seed: int = 0,
    ):
        self.policy = policy.to(device)
        self.config = config
        self.device = torch.device(device)
        self.action_mode = action_mode
        self.executor_levels = tuple(policy.executor_levels)
        self.postman = DensePostman(config.max_depth)
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(seed))

    def get_action(self, obs: tuple[Any, ...]) -> tuple[Any, int]:
        decision = self.get_action_record(obs)
        return decision.node, decision.use_exec

    def get_action_record(self, obs: tuple[Any, ...]) -> OfficialActionDecision:
        job_dags, source_job, num_source_exec, frontier_nodes, _, exec_commit, moving_executors, action_map = obs
        if len(frontier_nodes) == 0:
            return OfficialActionDecision(None, int(num_source_exec), None)

        state = translate_state(
            obs,
            config=self.config,
            executor_levels=self.executor_levels,
            postman=self.postman,
            device=self.device,
        )
        if float(state.node_valid_mask.sum().item()) == 0.0:
            return OfficialActionDecision(None, int(num_source_exec), None)

        with torch.no_grad():
            output = self.policy.predict(
                state.node_features,
                state.adjacency,
                job_features=state.job_features,
                node_valid_mask=state.node_valid_mask,
                job_valid_mask=state.job_valid_mask,
                dag_summ_backward_map=state.dag_summ_backward_map,
                dag_summary_mat=state.dag_summary_mat,
                running_dag_mat=state.running_dag_mat,
                adj_mats=state.adj_mats,
                masks=state.masks,
            )
        node_action = _choose_index(output.node_probs.detach().cpu(), self.action_mode, self.rng)
        assert int(state.node_valid_mask[node_action].item()) == 1
        node = action_map[node_action]
        job_index = job_dags.index(node.job_dag)
        executor_level_index = _choose_index(
            output.job_probs[job_index].detach().cpu(),
            self.action_mode,
            self.rng,
        )
        assert int(state.job_valid_mask[job_index, executor_level_index].item()) == 1

        if node.job_dag is source_job:
            agent_exec_act = (
                self.executor_levels[executor_level_index]
                - state.exec_map[node.job_dag]
                + int(num_source_exec)
            )
        else:
            agent_exec_act = self.executor_levels[executor_level_index] - state.exec_map[node.job_dag]

        use_exec = min(
            node.num_tasks
            - node.next_task_idx
            - exec_commit.node_commit[node]
            - moving_executors.count(node),
            agent_exec_act,
            int(num_source_exec),
        )
        if use_exec <= 0:
            raise AssertionError("PyTorch Decima selected a non-positive executor allocation")
        record = OfficialActionRecord(
            state=_actor_tensors(state),
            node_action=node_action,
            job_index=job_index,
            executor_level_index=executor_level_index,
        )
        return OfficialActionDecision(node, int(use_exec), record)


def configure_official_simulator(
    root: Path,
    *,
    config: DecimaConfig = DEFAULT_CONFIG,
    num_stream_dags: int | None = None,
    seed: int = 42,
) -> OfficialModules:
    """Import and configure the official simulator under controlled argv."""

    _patch_numpy_compat()
    decima_root = root / "external" / "decima-sim"
    if not decima_root.exists():
        raise FileNotFoundError(f"official decima-sim clone not found: {decima_root}")
    decima_root_str = str(decima_root)
    if decima_root_str not in sys.path:
        sys.path.insert(0, decima_root_str)

    argv = _official_argv(root, config=config, num_stream_dags=num_stream_dags, seed=seed)
    if "param" not in sys.modules:
        old_argv = sys.argv[:]
        sys.argv = argv
        try:
            param = importlib.import_module("param")
        finally:
            sys.argv = old_argv
    else:
        param = sys.modules["param"]
    _set_official_args(param.args, root, config=config, num_stream_dags=num_stream_dags, seed=seed)

    env_mod = importlib.import_module("spark_env.env")
    heuristic_mod = importlib.import_module("heuristic_agent")
    spark_mod = importlib.import_module("spark_agent")
    return OfficialModules(
        param=param,
        environment=env_mod.Environment,
        dynamic_partition_agent=heuristic_mod.DynamicPartitionAgent,
        spark_agent=spark_mod.SparkAgent,
    )


def translate_state(
    obs: tuple[Any, ...],
    *,
    config: DecimaConfig = DEFAULT_CONFIG,
    executor_levels: tuple[int, ...] | None = None,
    postman: DensePostman | None = None,
    device: str | torch.device = "cpu",
) -> OfficialStateTensors:
    """Translate an official simulator observation into PyTorch actor tensors."""

    executor_levels = tuple(executor_levels or config.actor_executor_levels)
    job_dags, source_job, num_source_exec, frontier_nodes, _, exec_commit, moving_executors, action_map = obs
    job_dags = tuple(job_dags)
    total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))
    node_inputs = np.zeros([total_num_nodes, config.node_input_dim], dtype=np.float32)
    job_inputs = np.zeros([len(job_dags), config.job_input_dim], dtype=np.float32)

    exec_map = _executor_map(job_dags, exec_commit, moving_executors)
    for job_idx, job_dag in enumerate(job_dags):
        job_inputs[job_idx, 0] = exec_map[job_dag] / 20.0
        job_inputs[job_idx, 1] = 2.0 if job_dag is source_job else -2.0
        job_inputs[job_idx, 2] = int(num_source_exec) / 20.0

    node_idx = 0
    for job_idx, job_dag in enumerate(job_dags):
        for node in job_dag.nodes:
            node_inputs[node_idx, :3] = job_inputs[job_idx, :3]
            node_inputs[node_idx, 3] = (
                (node.num_tasks - node.next_task_idx) * node.tasks[-1].duration / 100000.0
            )
            node_inputs[node_idx, 4] = (node.num_tasks - node.next_task_idx) / 200.0
            node_idx += 1

    node_valid_mask, job_valid_mask = _valid_masks(
        job_dags,
        frontier_nodes,
        source_job,
        int(num_source_exec),
        exec_map,
        action_map,
        executor_levels,
    )
    postman = postman or DensePostman(config.max_depth)
    msg_mats, msg_masks, dag_summ_backward_map, running_dag_mat, _ = postman.get_msg_path(job_dags)
    dag_summary_mat = _get_unfinished_nodes_summ_mat(job_dags)
    adjacency = np.zeros((total_num_nodes, total_num_nodes), dtype=np.float32)

    torch_device = torch.device(device)
    return OfficialStateTensors(
        node_features=torch.as_tensor(node_inputs, dtype=torch.float32, device=torch_device),
        adjacency=torch.as_tensor(adjacency, dtype=torch.float32, device=torch_device),
        adj_mats=tuple(torch.as_tensor(mat, dtype=torch.float32, device=torch_device) for mat in msg_mats),
        masks=tuple(torch.as_tensor(mask, dtype=torch.float32, device=torch_device) for mask in msg_masks),
        job_features=torch.as_tensor(job_inputs, dtype=torch.float32, device=torch_device),
        node_valid_mask=torch.as_tensor(node_valid_mask, dtype=torch.float32, device=torch_device),
        job_valid_mask=torch.as_tensor(job_valid_mask, dtype=torch.float32, device=torch_device),
        dag_summ_backward_map=torch.as_tensor(dag_summ_backward_map, dtype=torch.float32, device=torch_device),
        dag_summary_mat=torch.as_tensor(dag_summary_mat, dtype=torch.float32, device=torch_device),
        running_dag_mat=torch.as_tensor(running_dag_mat, dtype=torch.float32, device=torch_device),
        job_dags=job_dags,
        source_job=source_job,
        num_source_exec=int(num_source_exec),
        exec_map=exec_map,
        action_map=action_map,
        frontier_nodes=frontier_nodes,
        exec_commit=exec_commit,
        moving_executors=moving_executors,
    )


def run_official_episode(
    root: Path,
    *,
    scheme: Scheme,
    seed: int,
    num_stream_dags: int,
    config: DecimaConfig = DEFAULT_CONFIG,
    policy: DecimaPolicy | None = None,
    action_mode: ActionMode = "sample",
    device: str | torch.device = "cpu",
    max_decisions: int = 2_000_000,
) -> OfficialEpisodeResult:
    modules = configure_official_simulator(
        root,
        config=config,
        num_stream_dags=num_stream_dags,
        seed=seed,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = modules.environment()
    env.seed(seed)
    env.reset()
    if scheme == "dynamic_partition":
        agent = modules.dynamic_partition_agent()
    elif scheme == "spark_fifo":
        agent = modules.spark_agent(exec_cap=config.exec_cap)
    elif scheme == "pytorch_decima":
        if policy is None:
            policy = DecimaPolicy(config=config)
        policy.eval()
        agent = OfficialDecimaAgent(
            policy,
            config=config,
            device=device,
            action_mode=action_mode,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown Decima scheme: {scheme}")

    obs = env.observe()
    total_reward = 0.0
    done = False
    decisions = 0
    t0 = time.time()
    while not done:
        node, use_exec = agent.get_action(obs)
        obs, reward, done = env.step(node, use_exec)
        total_reward += float(reward)
        decisions += 1
        if decisions > max_decisions:
            raise RuntimeError(f"Decima official episode exceeded {max_decisions} decisions")

    durations = _job_completion_times(env.finished_job_dags)
    return OfficialEpisodeResult(
        scheme=scheme,
        seed=seed,
        num_stream_dags=num_stream_dags,
        total_reward=float(total_reward),
        mean_job_completion_time=float(np.mean(durations)) if len(durations) else math.nan,
        median_job_completion_time=float(np.median(durations)) if len(durations) else math.nan,
        num_finished_jobs=int(len(env.finished_job_dags)),
        makespan=float(env.wall_time.curr_time),
        decisions=decisions,
        elapsed_seconds=float(time.time() - t0),
    )


def collect_pytorch_rollout(
    root: Path,
    policy: DecimaPolicy,
    *,
    seed: int,
    action_seed: int,
    max_time: float,
    num_stream_dags: int,
    config: DecimaConfig = DEFAULT_CONFIG,
    device: str | torch.device = "cpu",
    max_decisions: int = 2_000_000,
) -> OfficialRolloutResult:
    modules = configure_official_simulator(
        root,
        config=config,
        num_stream_dags=num_stream_dags,
        seed=seed,
    )
    np.random.seed(seed + action_seed)
    torch.manual_seed(seed + action_seed)
    env = modules.environment()
    env.seed(seed)
    env.reset(max_time=max_time)
    agent = OfficialDecimaAgent(
        policy,
        config=config,
        device=device,
        action_mode="sample",
        seed=action_seed,
    )
    obs = env.observe()
    steps: list[OfficialExperienceStep] = []
    rewards: list[float] = []
    wall_times: list[float] = [float(env.wall_time.curr_time)]
    done = False
    decisions = 0

    while not done:
        decision = agent.get_action_record(obs)
        obs, reward, done = env.step(decision.node, decision.use_exec)
        reward = float(reward)
        decisions += 1
        if decision.record is not None:
            rewards.append(reward)
            wall_times.append(float(env.wall_time.curr_time))
            steps.append(
                OfficialExperienceStep(
                    record=decision.record,
                    reward=reward,
                    wall_time=float(env.wall_time.curr_time),
                )
            )
        elif rewards:
            rewards[-1] += reward
            wall_times[-1] = float(env.wall_time.curr_time)
            steps[-1].reward += reward
            steps[-1].wall_time = float(env.wall_time.curr_time)
        if decisions > max_decisions:
            raise RuntimeError(f"Decima rollout exceeded {max_decisions} decisions")

    durations = _job_completion_times(env.finished_job_dags)
    return OfficialRolloutResult(
        seed=seed,
        steps=tuple(steps),
        rewards=np.asarray(rewards, dtype=np.float64),
        wall_times=np.asarray(wall_times, dtype=np.float64),
        total_reward=float(np.sum(rewards)),
        mean_job_completion_time=float(np.mean(durations)) if len(durations) else math.nan,
        num_finished_jobs=int(len(env.finished_job_dags)),
        reset_hit=bool(env.wall_time.curr_time >= env.max_time),
    )


def train_pytorch_official(
    root: Path,
    *,
    config: DecimaConfig = DEFAULT_CONFIG,
    epochs: int,
    num_agents: int,
    num_stream_dags: int,
    checkpoint_dir: Path,
    seed: int = 42,
    lr: float = 0.001,
    device: str | torch.device = "cpu",
    checkpoint_interval: int = 100,
) -> OfficialTrainResult:
    """Train the PyTorch actor with the official simulator/reward loop."""

    policy = DecimaPolicy(config=config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    avg_reward = AveragePerStepReward(size=100000)
    entropy_weight = 1.0
    reset_prob = config.reset_prob
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_rollouts: list[OfficialRolloutResult] = []

    for epoch in range(1, epochs + 1):
        max_time = _generate_coin_flips(reset_prob, rng=np.random.default_rng(seed + epoch))
        rollouts = [
            collect_pytorch_rollout(
                root,
                policy,
                seed=seed + epoch,
                action_seed=agent_id,
                max_time=max_time,
                num_stream_dags=num_stream_dags,
                config=config,
                device=device,
            )
            for agent_id in range(num_agents)
        ]
        last_rollouts = rollouts
        active_rollouts: list[OfficialRolloutResult] = []
        all_times: list[np.ndarray] = []
        all_cum_rewards: list[np.ndarray] = []
        for rollout in rollouts:
            if len(rollout.rewards) == 0:
                continue
            diff_time = np.diff(rollout.wall_times)
            avg_reward.add_list_filter_zero(rollout.rewards, diff_time)
            active_rollouts.append(rollout)
            all_times.append(rollout.wall_times[1:])

        avg_per_step_reward = avg_reward.get_avg_per_step_reward()
        for rollout in active_rollouts:
            diff_time = np.diff(rollout.wall_times)
            rewards = rollout.rewards - avg_per_step_reward * diff_time
            all_cum_rewards.append(discount(rewards, gamma=1.0))

        baselines = _piecewise_linear_baseline(all_cum_rewards, all_times)
        optimizer.zero_grad()
        loss_value = 0.0
        loss_count = 0
        for rollout, cum_reward, baseline in zip(active_rollouts, all_cum_rewards, baselines, strict=True):
            advantages = cum_reward - baseline
            for step, advantage in zip(rollout.steps, advantages, strict=False):
                output = policy.predict(
                    step.record.state.node_features,
                    step.record.state.adjacency,
                    job_features=step.record.state.job_features,
                    node_valid_mask=step.record.state.node_valid_mask,
                    job_valid_mask=step.record.state.job_valid_mask,
                    dag_summ_backward_map=step.record.state.dag_summ_backward_map,
                    dag_summary_mat=step.record.state.dag_summary_mat,
                    running_dag_mat=step.record.state.running_dag_mat,
                    adj_mats=step.record.state.adj_mats,
                    masks=step.record.state.masks,
                )
                loss, _ = actor_loss(
                    node_probs=output.node_probs,
                    job_probs=output.job_probs,
                    node_action=step.record.node_action,
                    job_index=step.record.job_index,
                    executor_level_index=step.record.executor_level_index,
                    advantage=torch.tensor(float(advantage), dtype=torch.float32, device=device),
                    entropy_weight=entropy_weight,
                )
                loss.backward()
                loss_value += float(loss.detach().cpu().item())
                loss_count += 1
        optimizer.step()
        entropy_weight = max(0.0001, entropy_weight - 1e-3)
        reset_prob = max(config.reset_prob_min, reset_prob - config.reset_prob_decay)
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            _save_checkpoint(
                checkpoint_dir / f"model_ep_{epoch}.pt",
                policy,
                optimizer,
                config=config,
                epoch=epoch,
                entropy_weight=entropy_weight,
                reset_prob=reset_prob,
                loss=loss_value / max(loss_count, 1),
            )

    final_path = checkpoint_dir / f"model_ep_{epochs}.pt"
    if not final_path.exists():
        _save_checkpoint(
            final_path,
            policy,
            optimizer,
            config=config,
            epoch=epochs,
            entropy_weight=entropy_weight,
            reset_prob=reset_prob,
            loss=math.nan,
        )
    return OfficialTrainResult(
        checkpoint_path=final_path,
        final_epoch=epochs,
        mean_total_reward=float(np.mean([r.total_reward for r in last_rollouts])) if last_rollouts else math.nan,
        mean_job_completion_time=float(np.mean([r.mean_job_completion_time for r in last_rollouts]))
        if last_rollouts
        else math.nan,
        entropy_weight=entropy_weight,
        reset_prob=reset_prob,
    )


def load_pytorch_checkpoint(path: Path, *, config: DecimaConfig = DEFAULT_CONFIG, device: str | torch.device = "cpu") -> DecimaPolicy:
    payload = torch.load(path, map_location=device)
    policy = DecimaPolicy(config=config)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


def _save_checkpoint(
    path: Path,
    policy: DecimaPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    config: DecimaConfig,
    epoch: int,
    entropy_weight: float,
    reset_prob: float,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config.__dict__,
            "executor_levels": tuple(policy.executor_levels),
            "entropy_weight": entropy_weight,
            "reset_prob": reset_prob,
            "loss": loss,
        },
        path,
    )


def _official_argv(
    root: Path,
    *,
    config: DecimaConfig,
    num_stream_dags: int | None,
    seed: int,
) -> list[str]:
    argv = ["decima-pytorch-port"]
    for key, values in _official_arg_values(root, config=config, num_stream_dags=num_stream_dags, seed=seed).items():
        argv.append(f"--{key}")
        if isinstance(values, tuple):
            argv.extend(str(value) for value in values)
        else:
            argv.append(str(values))
    return argv


def _official_arg_values(
    root: Path,
    *,
    config: DecimaConfig,
    num_stream_dags: int | None,
    seed: int,
) -> dict[str, object]:
    job_folder = root / "external" / "decima-sim" / "spark_env" / "tpch"
    return {
        "seed": seed,
        "exec_cap": config.exec_cap,
        "num_init_dags": config.num_init_dags,
        "num_stream_dags": num_stream_dags if num_stream_dags is not None else config.train_num_stream_dags,
        "stream_interval": config.stream_interval_ms,
        "executor_data_point": tuple(config.executor_data_points),
        "job_folder": str(job_folder) + "/",
        "tpch_size": tuple(config.tpch_sizes),
        "tpch_num": config.tpch_num_queries,
        "node_input_dim": config.node_input_dim,
        "job_input_dim": config.job_input_dim,
        "hid_dims": tuple(config.hidden_dims),
        "output_dim": config.output_dim,
        "max_depth": config.max_depth,
        "diff_reward_enabled": config.diff_reward_enabled,
        "reset_prob": config.reset_prob,
        "reset_prob_min": config.reset_prob_min,
        "reset_prob_decay": config.reset_prob_decay,
        "num_agents": config.num_agents,
        "model_save_interval": config.model_save_interval,
        "model_folder": config.model_folder,
        "canvs_visualization": 0,
    }


def _set_official_args(
    args: Any,
    root: Path,
    *,
    config: DecimaConfig,
    num_stream_dags: int | None,
    seed: int,
) -> None:
    for key, value in _official_arg_values(
        root,
        config=config,
        num_stream_dags=num_stream_dags,
        seed=seed,
    ).items():
        setattr(args, key, list(value) if isinstance(value, tuple) and key in {"executor_data_point", "tpch_size", "hid_dims"} else value)


def _executor_map(job_dags: tuple[Any, ...], exec_commit: Any, moving_executors: Any) -> dict[Any, int]:
    job_dag_cls = _class_from_first(job_dags)
    node_cls = _node_class_from_jobs(job_dags)
    exec_map = {job_dag: len(job_dag.executors) for job_dag in job_dags}
    for node in moving_executors.moving_executors.values():
        exec_map[node.job_dag] += 1
    for source in exec_commit.commit:
        if isinstance(source, job_dag_cls):
            source_job = source
        elif isinstance(source, node_cls):
            source_job = source.job_dag
        elif source is None:
            source_job = None
        else:
            raise TypeError(f"unknown executor-commit source: {source!r}")
        for node in exec_commit.commit[source]:
            if node is not None and node.job_dag != source_job:
                exec_map[node.job_dag] += exec_commit.commit[source][node]
    return exec_map


def _valid_masks(
    job_dags: tuple[Any, ...],
    frontier_nodes: Any,
    source_job: Any,
    num_source_exec: int,
    exec_map: dict[Any, int],
    action_map: Any,
    executor_levels: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    job_valid_mask = np.zeros([len(job_dags), len(executor_levels)], dtype=np.float32)
    job_valid: dict[Any, bool] = {}
    for job_idx, job_dag in enumerate(job_dags):
        if job_dag is source_job:
            least_exec_amount = exec_map[job_dag] - num_source_exec + 1
        else:
            least_exec_amount = exec_map[job_dag] + 1
        assert least_exec_amount > 0
        assert least_exec_amount <= executor_levels[-1] + 1
        exec_level_idx = bisect.bisect_left(executor_levels, least_exec_amount)
        job_valid[job_dag] = exec_level_idx < len(executor_levels)
        if job_valid[job_dag]:
            job_valid_mask[job_idx, exec_level_idx:] = 1.0

    total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))
    node_valid_mask = np.zeros(total_num_nodes, dtype=np.float32)
    for node in frontier_nodes:
        if job_valid[node.job_dag]:
            node_valid_mask[action_map.inverse_map[node]] = 1.0
    return node_valid_mask, job_valid_mask


def _get_msg_path(job_dags: tuple[Any, ...], max_depth: int) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    total_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))
    merged_mats = [np.zeros((total_nodes, total_nodes), dtype=np.float32) for _ in range(max_depth)]
    merged_masks = [np.zeros((total_nodes, 1), dtype=np.float32) for _ in range(max_depth)]
    base = 0
    for job_dag in job_dags:
        local_mats, local_masks = _get_bottom_up_paths(job_dag, max_depth)
        end = base + job_dag.num_nodes
        for depth in range(max_depth):
            merged_mats[depth][base:end, base:end] = local_mats[depth]
            merged_masks[depth][base:end, 0] = local_masks[depth]
        base = end
    return tuple(merged_mats), tuple(merged_masks)


def _get_bottom_up_paths(job_dag: Any, max_depth: int) -> tuple[list[np.ndarray], np.ndarray]:
    num_nodes = job_dag.num_nodes
    msg_mats = [np.zeros((num_nodes, num_nodes), dtype=np.float32) for _ in range(max_depth)]
    msg_masks = np.zeros((max_depth, num_nodes), dtype=np.float32)
    frontier = _get_init_frontier(job_dag, max_depth)
    msg_level = {node: 0 for node in frontier}
    for depth in range(max_depth):
        new_frontier = set()
        parent_visited = set()
        for node in frontier:
            for parent in node.parent_nodes:
                if parent in parent_visited:
                    continue
                curr_level = 0
                children_all_in_frontier = True
                for child in parent.child_nodes:
                    if child not in frontier:
                        children_all_in_frontier = False
                        break
                    curr_level = max(curr_level, msg_level[child])
                if children_all_in_frontier and (
                    parent not in msg_level or curr_level + 1 > msg_level[parent]
                ):
                    new_frontier.add(parent)
                    msg_level[parent] = curr_level + 1
                parent_visited.add(parent)
        if len(new_frontier) == 0:
            break
        for node in new_frontier:
            for child in node.child_nodes:
                msg_mats[depth][node.idx, child.idx] = 1.0
            msg_masks[depth, node.idx] = 1.0
        for node in frontier:
            parents_all_in_frontier = all(parent in msg_level for parent in node.parent_nodes)
            if not parents_all_in_frontier:
                new_frontier.add(node)
        frontier = new_frontier
    return msg_mats, msg_masks


def _get_init_frontier(job_dag: Any, depth: int) -> set[Any]:
    sources = set(job_dag.nodes)
    for _ in range(depth):
        new_sources = set()
        for node in sources:
            if len(node.child_nodes) == 0:
                new_sources.add(node)
            else:
                new_sources.update(node.child_nodes)
        sources = new_sources
    return sources


def _get_dag_summ_backward_map(job_dags: tuple[Any, ...]) -> np.ndarray:
    total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))
    out = np.zeros([total_num_nodes, len(job_dags)], dtype=np.float32)
    base = 0
    for job_idx, job_dag in enumerate(job_dags):
        for node in job_dag.nodes:
            out[base + node.idx, job_idx] = 1.0
        base += job_dag.num_nodes
    return out


def _get_running_dag_mat(job_dags: tuple[Any, ...]) -> np.ndarray:
    out = np.zeros((1, len(job_dags)), dtype=np.float32)
    for job_idx, job_dag in enumerate(job_dags):
        if not job_dag.completed:
            out[0, job_idx] = 1.0
    return out


def _get_unfinished_nodes_summ_mat(job_dags: tuple[Any, ...]) -> np.ndarray:
    total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))
    out = np.zeros((len(job_dags), total_num_nodes), dtype=np.float32)
    base = 0
    for job_idx, job_dag in enumerate(job_dags):
        for node in job_dag.nodes:
            if not node.tasks_all_done:
                out[job_idx, base + node.idx] = 1.0
        base += job_dag.num_nodes
    return out


def _choose_index(probs: torch.Tensor, mode: ActionMode, rng: torch.Generator) -> int:
    if mode == "greedy":
        return int(torch.argmax(probs).item())
    clean = probs.clamp_min(0)
    total = float(clean.sum().item())
    if total <= 0 or not math.isfinite(total):
        return int(torch.argmax(probs).item())
    clean = clean / total
    return int(torch.multinomial(clean, num_samples=1, generator=rng).item())


def _actor_tensors(state: OfficialStateTensors) -> OfficialActorTensors:
    return OfficialActorTensors(
        node_features=state.node_features.detach(),
        adjacency=state.adjacency.detach(),
        adj_mats=tuple(mat.detach() for mat in state.adj_mats),
        masks=tuple(mask.detach() for mask in state.masks),
        job_features=state.job_features.detach(),
        node_valid_mask=state.node_valid_mask.detach(),
        job_valid_mask=state.job_valid_mask.detach(),
        dag_summ_backward_map=state.dag_summ_backward_map.detach(),
        dag_summary_mat=state.dag_summary_mat.detach(),
        running_dag_mat=state.running_dag_mat.detach(),
    )


def _job_completion_times(finished_job_dags: Any) -> np.ndarray:
    return np.asarray(
        [job.completion_time - job.start_time for job in finished_job_dags],
        dtype=np.float64,
    )


def _generate_coin_flips(p: float, *, rng: np.random.Generator) -> float:
    if p == 0:
        return np.inf
    return float(rng.geometric(p))


def _piecewise_linear_baseline(
    all_cum_rewards: list[np.ndarray],
    all_wall_time: list[np.ndarray],
) -> list[np.ndarray]:
    if len(all_cum_rewards) == 0:
        return []
    unique_wall_time = np.unique(np.hstack(all_wall_time))
    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0.0
        for rewards, wall_time in zip(all_cum_rewards, all_wall_time, strict=True):
            idx = bisect.bisect_left(wall_time, t)
            if idx == 0:
                baseline += rewards[idx]
            elif idx == len(rewards):
                baseline += rewards[-1]
            elif wall_time[idx] == t:
                baseline += rewards[idx]
            else:
                baseline += (
                    (rewards[idx] - rewards[idx - 1])
                    / (wall_time[idx] - wall_time[idx - 1])
                    * (t - wall_time[idx])
                    + rewards[idx]
                )
        baseline_values[t] = baseline / float(len(all_wall_time))
    return [np.asarray([baseline_values[t] for t in wall_time], dtype=np.float64) for wall_time in all_wall_time]


def _class_from_first(job_dags: tuple[Any, ...]) -> type:
    if not job_dags:
        return object
    return type(job_dags[0])


def _node_class_from_jobs(job_dags: tuple[Any, ...]) -> type:
    for job_dag in job_dags:
        if job_dag.nodes:
            return type(job_dag.nodes[0])
    return object


def _patch_numpy_compat() -> None:
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]
