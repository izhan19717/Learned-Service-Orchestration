import mlflow
import torch

from cisose_deeprm.model import DeepRMPolicy
from cisose_deeprm.protocol import DeepRMConfig
from cisose_deeprm.evaluation import clean_evaluation_payload, evaluate_clean_policy, generate_eval_traces
from cisose_deeprm.training import TrainConfig, discounted_returns, rollout, train_policy
from cisose_deeprm.workload import Job, trace_from_jobs


def test_policy_parameter_count_matches_paper():
    policy = DeepRMPolicy(DeepRMConfig())
    assert policy.num_parameters == 89_451


def test_discounted_returns_with_unit_discount():
    returns = discounted_returns([1.0, 2.0, 3.0], discount=1.0)
    assert returns.tolist() == [6.0, 5.0, 3.0]


def test_rollout_truncates_at_episode_cap_instead_of_crashing():
    config = DeepRMConfig(time_horizon=2)
    policy = DeepRMPolicy(config)
    with torch.no_grad():
        for param in policy.parameters():
            param.zero_()
        policy.net[3].bias[0] = 10.0
    trace = trace_from_jobs(
        [Job(id=0, arrival_time=0, duration=3, demand=(0.1, 0.1))],
        rate=1.0,
        tail_alpha=float("inf"),
        seed=22,
    )
    _, rewards, steps, capped = rollout(
        policy,
        trace,
        env_config=config,
        max_steps=5,
        drain=True,
    )
    assert steps == 5
    assert len(rewards) == 5
    assert capped is True


def test_train_policy_tiny_smoke(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_name = "unit_train_smoke"
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        summary = train_policy(
            TrainConfig(
                load=0.7,
                run_label="unit",
                master_seed=3,
                iterations=1,
                num_jobsets=1,
                rollouts_per_jobset=1,
                checkpoint_interval=1,
                eval_interval=1,
            ),
            root=tmp_path,
        )
    assert summary.iterations == 1
    assert (tmp_path / "results" / "checkpoints" / "unit" / "load_0.7" / "policy_final.pt").exists()


def test_train_policy_tiny_parallel_smoke(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_name = "unit_train_parallel_smoke"
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        summary = train_policy(
            TrainConfig(
                load=0.7,
                run_label="parallel_unit",
                master_seed=13,
                iterations=1,
                num_jobsets=2,
                rollouts_per_jobset=2,
                checkpoint_interval=1,
                eval_interval=1,
                rollout_workers=2,
            ),
            root=tmp_path,
        )
    assert summary.iterations == 1
    assert (tmp_path / "results" / "checkpoints" / "parallel_unit" / "load_0.7" / "policy_final.pt").exists()


def test_train_policy_can_resume_from_checkpoint(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_name = "unit_train_resume"
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        train_policy(
            TrainConfig(
                load=0.7,
                run_label="resume",
                master_seed=5,
                iterations=1,
                num_jobsets=1,
                rollouts_per_jobset=1,
                checkpoint_interval=1,
                eval_interval=1,
            ),
            root=tmp_path,
        )
    checkpoint = tmp_path / "results" / "checkpoints" / "resume" / "load_0.7" / "policy_iter_1.pt"
    with mlflow.start_run():
        summary = train_policy(
            TrainConfig(
                load=0.7,
                run_label="resume",
                master_seed=5,
                iterations=2,
                num_jobsets=1,
                rollouts_per_jobset=1,
                checkpoint_interval=1,
                eval_interval=1,
                resume_from_checkpoint=str(checkpoint),
            ),
            root=tmp_path,
        )
    final = torch.load(tmp_path / "results" / "checkpoints" / "resume" / "load_0.7" / "policy_final.pt")
    assert summary.resume_iteration == 1
    assert summary.resumed_from_checkpoint == str(checkpoint)
    assert final["metadata"]["iteration"] == 2
    assert final["metadata"]["resume_iteration"] == 1
    assert (tmp_path / "results" / "training" / "resume" / "load_0.7_curve_resume_from_1.jsonl").exists()


def test_clean_evaluation_payload_smoke():
    policy = DeepRMPolicy(DeepRMConfig())
    with torch.no_grad():
        for param in policy.parameters():
            param.zero_()
        policy.net[3].bias[0] = 1.0
    traces = generate_eval_traces(load=0.7, num_seeds=1, trace_jobs=5, seed=11)
    summary, metrics = evaluate_clean_policy(
        policy,
        traces,
        checkpoint_path="unit.pt",
        load=0.7,
        seed=11,
        policy_deterministic=True,
        max_steps=1000,
    )
    payload = clean_evaluation_payload(summary, metrics)
    assert summary.num_seeds == 1
    assert summary.policy_mode == "deterministic_argmax"
    assert "DeepRM" in payload["per_seed_metrics"]
    assert "Tetris*" in payload["summary"]["comparisons"]
