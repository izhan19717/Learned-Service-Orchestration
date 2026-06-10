"""REINFORCE training for the DeepRM policy."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import optim

from cisose_deeprm.model import DeepRMPolicy, save_checkpoint
from cisose_deeprm.protocol import (
    TRAIN_ITERATIONS,
    TRAIN_JOBSETS,
    TRAIN_LEARNING_RATE,
    TRAIN_ROLLOUTS_PER_JOBSET,
    DeepRMConfig,
)
from cisose_deeprm.simulator import DeepRMEnv
from cisose_deeprm.workload import WorkloadTrace, generate_time_trace


@dataclass(frozen=True)
class TrainConfig:
    load: float
    run_label: str = "clean"
    master_seed: int = 20260514
    iterations: int = TRAIN_ITERATIONS
    num_jobsets: int = TRAIN_JOBSETS
    rollouts_per_jobset: int = TRAIN_ROLLOUTS_PER_JOBSET
    learning_rate: float = TRAIN_LEARNING_RATE
    episode_horizon: int = 50
    hidden_dim: int = 20
    checkpoint_interval: int = 50
    eval_interval: int = 10
    max_episode_steps: int = 2000
    drain: bool = True
    resume_from_checkpoint: str | None = None
    rollout_workers: int = 1


@dataclass(frozen=True)
class TrainingSummary:
    checkpoint_path: str
    final_mean_reward: float
    final_mean_slowdown: float | None
    iterations: int
    load: float
    resumed_from_checkpoint: str | None = None
    resume_iteration: int = 0


def make_training_jobsets(config: TrainConfig, env_config: DeepRMConfig) -> tuple[WorkloadTrace, ...]:
    seed_seq = np.random.SeedSequence(config.master_seed)
    child_seeds = seed_seq.spawn(config.num_jobsets)
    return tuple(
        generate_time_trace(
            horizon=config.episode_horizon,
            rate=config.load,
            seed=int(child.generate_state(1)[0]),
            config=env_config,
        )
        for child in child_seeds
    )


def rollout(
    policy: DeepRMPolicy,
    trace: WorkloadTrace,
    *,
    env_config: DeepRMConfig,
    max_steps: int,
    drain: bool,
) -> tuple[list[torch.Tensor], list[float], int, bool]:
    env = DeepRMEnv(trace, config=env_config, drain=drain)
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    steps = 0
    while not env.done and steps < max_steps:
        state = torch.from_numpy(env.observe()).unsqueeze(0).float()
        logits = policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        _, reward, _, _ = env.step(int(action.item()))
        log_probs.append(dist.log_prob(action).squeeze())
        rewards.append(float(reward))
        steps += 1
    capped = not env.done
    return log_probs, rewards, steps, capped


def _jobset_seed(master_seed: int, iteration: int, jobset_idx: int) -> int:
    return int(np.random.SeedSequence([master_seed, iteration, jobset_idx]).generate_state(1)[0])


def _jobset_gradient_worker(payload: tuple[dict[str, torch.Tensor], DeepRMConfig, int, WorkloadTrace, int, bool, int, int]) -> dict[str, object]:
    (
        policy_state,
        env_config,
        hidden_dim,
        trace,
        rollouts_per_jobset,
        drain,
        max_episode_steps,
        seed,
    ) = payload
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = DeepRMPolicy(config=env_config, hidden_dim=hidden_dim)
    policy.load_state_dict(policy_state)
    policy.train()

    rollout_data = []
    for rollout_idx in range(rollouts_per_jobset):
        rollout_seed = (seed + rollout_idx * 1_000_003) % (2**32 - 1)
        torch.manual_seed(rollout_seed)
        np.random.seed(rollout_seed)
        rollout_data.append(
            rollout(
                policy,
                trace,
                env_config=env_config,
                max_steps=max_episode_steps,
                drain=drain,
            )
        )

    returns = [discounted_returns(rewards, env_config.discount) for _, rewards, _, _ in rollout_data]
    max_len = max((len(ret) for ret in returns), default=0)
    if max_len == 0:
        raise RuntimeError("parallel worker produced no training rollout")
    padded_returns = torch.zeros((len(returns), max_len), dtype=torch.float32)
    masks = torch.zeros((len(returns), max_len), dtype=torch.float32)
    for ridx, ret in enumerate(returns):
        padded_returns[ridx, : len(ret)] = ret
        masks[ridx, : len(ret)] = 1.0
    counts = torch.clamp(masks.sum(dim=0), min=1.0)
    baseline = padded_returns.sum(dim=0) / counts

    policy.zero_grad(set_to_none=True)
    losses: list[torch.Tensor] = []
    rewards_for_log: list[float] = []
    steps_for_log: list[int] = []
    capped_for_log: list[bool] = []
    for ridx, (log_probs, rewards, steps, capped) in enumerate(rollout_data):
        if not log_probs:
            continue
        ret = padded_returns[ridx, : len(log_probs)]
        advantage = ret - baseline[: len(log_probs)]
        log_prob_tensor = torch.stack(log_probs)
        losses.append(-(log_prob_tensor * advantage.detach()).sum())
        rewards_for_log.append(float(sum(rewards)))
        steps_for_log.append(steps)
        capped_for_log.append(capped)
    if not losses:
        raise RuntimeError("parallel worker produced no gradient-bearing rollout")
    jobset_loss = torch.stack(losses).mean()
    jobset_loss.backward()
    gradients = [
        param.grad.detach().cpu().numpy() if param.grad is not None else np.zeros(param.shape, dtype=np.float32)
        for param in policy.parameters()
    ]
    return {
        "gradients": gradients,
        "loss": float(jobset_loss.detach().item()),
        "rewards": rewards_for_log,
        "steps": steps_for_log,
        "capped": capped_for_log,
    }


def discounted_returns(rewards: list[float], discount: float = 1.0) -> torch.Tensor:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + discount * running
        returns[idx] = running
    return torch.from_numpy(returns)


def train_policy(
    config: TrainConfig,
    *,
    root: Path,
    env_config: DeepRMConfig | None = None,
) -> TrainingSummary:
    env_config = env_config or DeepRMConfig(primary_load=config.load)
    torch.manual_seed(config.master_seed)
    np.random.seed(config.master_seed)

    policy = DeepRMPolicy(config=env_config, hidden_dim=config.hidden_dim)
    expected_parameters = env_config.state_dim * config.hidden_dim + config.hidden_dim
    expected_parameters += config.hidden_dim * env_config.action_dim + env_config.action_dim
    if policy.num_parameters != expected_parameters:
        raise RuntimeError(
            f"DeepRM parameter count mismatch: {policy.num_parameters} != {expected_parameters}"
        )

    resume_iteration = 0
    resume_metadata: dict[str, object] = {}
    resume_optimizer_state: dict[str, object] | None = None
    if config.resume_from_checkpoint:
        checkpoint_path = Path(config.resume_from_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        policy.load_state_dict(checkpoint["state_dict"])
        resume_optimizer_state = checkpoint.get("optimizer_state")
        resume_metadata = dict(checkpoint.get("metadata", {}))
        resume_iteration = int(resume_metadata.get("iteration", 0))
        if resume_iteration <= 0:
            raise RuntimeError(f"resume checkpoint does not record a positive iteration: {checkpoint_path}")
        if resume_iteration >= config.iterations:
            raise RuntimeError(
                f"resume checkpoint iteration {resume_iteration} is not before target iterations {config.iterations}"
            )

    optimizer = optim.RMSprop(policy.parameters(), lr=config.learning_rate, alpha=0.9, eps=1e-9)
    resume_optimizer_note = "reset_unavailable_in_checkpoint"
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        resume_optimizer_note = "restored_from_checkpoint"
    jobsets = make_training_jobsets(config, env_config)
    mlflow.log_params({f"train.{k}": _mlflow_param(v) for k, v in asdict(config).items()})
    mlflow.log_params({"model.num_parameters": policy.num_parameters, "model.hidden_dim": config.hidden_dim})
    if config.resume_from_checkpoint:
        mlflow.log_params(
            {
                "train.resume_iteration": resume_iteration,
                "train.resume_optimizer_state": resume_optimizer_note,
                "train.resume_source_run_id": _mlflow_param(resume_metadata.get("mlflow_run_id")),
            }
        )

    checkpoint_dir = root / "results" / "checkpoints" / config.run_label / f"load_{config.load}"
    curve_suffix = f"_resume_from_{resume_iteration}" if config.resume_from_checkpoint else ""
    curve_path = root / "results" / "training" / config.run_label / f"load_{config.load}_curve{curve_suffix}.jsonl"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint = checkpoint_dir / "policy_final.pt"
    final_mean_reward = float("nan")
    final_mean_slowdown: float | None = None

    executor = (
        ProcessPoolExecutor(
            max_workers=config.rollout_workers,
            mp_context=mp.get_context("spawn"),
        )
        if config.rollout_workers and config.rollout_workers > 1
        else None
    )
    try:
        for iteration in range(resume_iteration + 1, config.iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            jobset_loss_values: list[float] = []
            rewards_for_log: list[float] = []
            steps_for_log: list[int] = []
            capped_for_log: list[bool] = []

            if executor is None:
                _accumulate_serial_iteration(
                    policy=policy,
                    jobsets=jobsets,
                    config=config,
                    env_config=env_config,
                    jobset_loss_values=jobset_loss_values,
                    rewards_for_log=rewards_for_log,
                    steps_for_log=steps_for_log,
                    capped_for_log=capped_for_log,
                )
            else:
                _accumulate_parallel_iteration(
                    executor=executor,
                    policy=policy,
                    jobsets=jobsets,
                    config=config,
                    env_config=env_config,
                    iteration=iteration,
                    jobset_loss_values=jobset_loss_values,
                    rewards_for_log=rewards_for_log,
                    steps_for_log=steps_for_log,
                    capped_for_log=capped_for_log,
                )

            if not jobset_loss_values:
                raise RuntimeError("no training rollouts produced gradients")
            optimizer.step()

            final_mean_reward = float(np.mean(rewards_for_log))
            record = {
                "iteration": iteration,
                "loss": float(np.mean(jobset_loss_values)),
                "mean_episode_reward": final_mean_reward,
                "mean_episode_steps": float(np.mean(steps_for_log)),
                "capped_episode_fraction": float(np.mean(capped_for_log)) if capped_for_log else 0.0,
            }
            with curve_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            if iteration == 1 or iteration % config.eval_interval == 0 or iteration == config.iterations:
                mlflow.log_metrics(
                    {
                        "train.loss": record["loss"],
                        "train.mean_episode_reward": record["mean_episode_reward"],
                        "train.mean_episode_steps": record["mean_episode_steps"],
                        "train.capped_episode_fraction": record["capped_episode_fraction"],
                    },
                    step=iteration,
                )
                print(
                    "iteration={iteration} loss={loss:.6f} mean_episode_reward={reward:.6f} "
                    "mean_episode_steps={steps:.2f}".format(
                        iteration=iteration,
                        loss=record["loss"],
                        reward=record["mean_episode_reward"],
                        steps=record["mean_episode_steps"],
                    ),
                    flush=True,
                )
            if iteration % config.checkpoint_interval == 0 or iteration == config.iterations:
                ckpt = checkpoint_dir / f"policy_iter_{iteration}.pt"
                save_checkpoint(
                    policy,
                    ckpt,
                    metadata={
                        "iteration": iteration,
                        "train_config": asdict(config),
                        "env_config": asdict(env_config),
                        "resume_iteration": resume_iteration,
                        "resumed_from_checkpoint": config.resume_from_checkpoint,
                        "resume_optimizer_state": resume_optimizer_note if config.resume_from_checkpoint else None,
                        "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
                    },
                    optimizer_state=optimizer.state_dict(),
                )
                mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    save_checkpoint(
        policy,
        final_checkpoint,
        metadata={
            "iteration": config.iterations,
            "train_config": asdict(config),
            "env_config": asdict(env_config),
            "resume_iteration": resume_iteration,
            "resumed_from_checkpoint": config.resume_from_checkpoint,
            "resume_optimizer_state": resume_optimizer_note if config.resume_from_checkpoint else None,
            "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
        },
        optimizer_state=optimizer.state_dict(),
    )
    mlflow.log_artifact(str(final_checkpoint), artifact_path="checkpoints")
    mlflow.log_artifact(str(curve_path), artifact_path="training")
    return TrainingSummary(
        checkpoint_path=str(final_checkpoint),
        final_mean_reward=final_mean_reward,
        final_mean_slowdown=final_mean_slowdown,
        iterations=config.iterations,
        load=config.load,
        resumed_from_checkpoint=config.resume_from_checkpoint,
        resume_iteration=resume_iteration,
    )


def _accumulate_serial_iteration(
    *,
    policy: DeepRMPolicy,
    jobsets: tuple[WorkloadTrace, ...],
    config: TrainConfig,
    env_config: DeepRMConfig,
    jobset_loss_values: list[float],
    rewards_for_log: list[float],
    steps_for_log: list[int],
    capped_for_log: list[bool],
) -> None:
    for trace in jobsets:
        rollout_data = [
            rollout(policy, trace, env_config=env_config, max_steps=config.max_episode_steps, drain=config.drain)
            for _ in range(config.rollouts_per_jobset)
        ]
        returns = [discounted_returns(rewards, env_config.discount) for _, rewards, _, _ in rollout_data]
        max_len = max((len(ret) for ret in returns), default=0)
        if max_len == 0:
            continue
        padded_returns = torch.zeros((len(returns), max_len), dtype=torch.float32)
        masks = torch.zeros((len(returns), max_len), dtype=torch.float32)
        for ridx, ret in enumerate(returns):
            padded_returns[ridx, : len(ret)] = ret
            masks[ridx, : len(ret)] = 1.0
        counts = torch.clamp(masks.sum(dim=0), min=1.0)
        baseline = padded_returns.sum(dim=0) / counts

        losses: list[torch.Tensor] = []
        for ridx, (log_probs, rewards, steps, capped) in enumerate(rollout_data):
            if not log_probs:
                continue
            ret = padded_returns[ridx, : len(log_probs)]
            advantage = ret - baseline[: len(log_probs)]
            log_prob_tensor = torch.stack(log_probs)
            losses.append(-(log_prob_tensor * advantage.detach()).sum())
            rewards_for_log.append(float(sum(rewards)))
            steps_for_log.append(steps)
            capped_for_log.append(capped)
        if losses:
            jobset_loss = torch.stack(losses).mean()
            jobset_loss_values.append(float(jobset_loss.detach().item()))
            (jobset_loss / len(jobsets)).backward()


def _accumulate_parallel_iteration(
    *,
    executor: ProcessPoolExecutor,
    policy: DeepRMPolicy,
    jobsets: tuple[WorkloadTrace, ...],
    config: TrainConfig,
    env_config: DeepRMConfig,
    iteration: int,
    jobset_loss_values: list[float],
    rewards_for_log: list[float],
    steps_for_log: list[int],
    capped_for_log: list[bool],
) -> None:
    policy_state = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    payloads = [
        (
            policy_state,
            env_config,
            config.hidden_dim,
            trace,
            config.rollouts_per_jobset,
            config.drain,
            config.max_episode_steps,
            _jobset_seed(config.master_seed, iteration, jobset_idx),
        )
        for jobset_idx, trace in enumerate(jobsets)
    ]
    gradients_sum: list[torch.Tensor] | None = None
    completed = 0
    for result in executor.map(_jobset_gradient_worker, payloads):
        jobset_loss_values.append(float(result["loss"]))
        rewards_for_log.extend(float(value) for value in result["rewards"])
        steps_for_log.extend(int(value) for value in result["steps"])
        capped_for_log.extend(bool(value) for value in result["capped"])
        gradients = result["gradients"]
        if gradients_sum is None:
            gradients_sum = [torch.from_numpy(grad).float() for grad in gradients]
        else:
            for idx, grad in enumerate(gradients):
                gradients_sum[idx] += torch.from_numpy(grad).float()
        completed += 1
    if gradients_sum is None or completed == 0:
        raise RuntimeError("parallel training produced no gradients")
    for param, grad_sum in zip(policy.parameters(), gradients_sum, strict=True):
        param.grad = grad_sum.to(param.device) / float(completed)


def _mlflow_param(value: object) -> object:
    return "" if value is None else value
