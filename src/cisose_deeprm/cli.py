"""Command-line entry points for the experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import mlflow
import torch

from cisose_deeprm.evaluation import (
    clean_evaluation_payload,
    evaluate_clean_policy,
    generate_eval_traces,
    holm_bonferroni,
    paired_result,
    run_adversarial_policy_episode,
    run_episode,
    run_lagged_policy_episode,
    run_lagged_scheduler_episode,
)
from cisose_deeprm.model import DeepRMPolicy, DeepRMScheduler, load_checkpoint
from cisose_deeprm.protocol import (
    ADVERSARIAL_EPS_SWEEP,
    ANCHOR_EPSILON,
    ANCHOR_LAG,
    ANCHOR_TAIL_ALPHA,
    LAG_SWEEP,
    LOAD_SWEEP,
    PRIMARY_LOAD,
    TAIL_SWEEP,
    EVAL_NUM_SEEDS,
    EVAL_TRACE_JOBS,
    DeepRMConfig,
    author_source_config,
)
from cisose_deeprm.schedulers import RandomScheduler, PackerScheduler, SJFScheduler, SourceTetrisScheduler, TetrisScheduler
from cisose_deeprm.simulator import DeepRMEnv
from cisose_deeprm.training import TrainConfig, train_policy
from cisose_deeprm.tracking import protocol_manifest, start_tracked_run, write_json_with_run_id
from cisose_deeprm.workload import Job, WorkloadTrace, generate_time_trace, sample_demand, sample_duration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cisose-deeprm")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("manifest", help="Log protocol/source manifest to MLflow")
    sub.add_parser("protocol", help="Print locked protocol constants")
    sub.add_parser("test", help="Run pytest and log the verification result to MLflow")
    sub.add_parser("gpu-check", help="Check CUDA/GPU visibility and log it to MLflow")
    sub.add_parser("author-preflight", help="Run cheap DeepRM author-source alignment checks")
    author_eval = sub.add_parser(
        "author-eval-smoke",
        help="Run a cheap source-style DeepRM evaluation smoke, separate from strict v2.2 evaluation",
    )
    author_eval.add_argument("--checkpoint", type=Path, required=True)
    author_eval.add_argument("--load", type=float, default=PRIMARY_LOAD)
    author_eval.add_argument("--trace-horizon", type=int, default=200)
    author_eval.add_argument("--seed", type=int, default=20260518)
    author_eval.add_argument("--max-steps", type=int, default=10_000)
    author_eval.add_argument(
        "--policy-mode",
        choices=("stochastic", "deterministic"),
        default="stochastic",
    )
    author_eval.add_argument("--run-name", default="deeprm-author-source-eval-smoke")
    author_full_eval = sub.add_parser(
        "author-evaluate",
        help="Run source-style DeepRM clean evaluation over multiple traces",
    )
    author_full_eval.add_argument("--checkpoint", type=Path, required=True)
    author_full_eval.add_argument("--load", type=float, default=PRIMARY_LOAD)
    author_full_eval.add_argument("--num-seeds", type=int, default=100)
    author_full_eval.add_argument("--trace-horizon", type=int, default=200)
    author_full_eval.add_argument("--seed", type=int, default=20260520)
    author_full_eval.add_argument("--max-steps", type=int, default=20_000)
    author_full_eval.add_argument(
        "--policy-mode",
        choices=("stochastic", "deterministic"),
        default="stochastic",
    )
    author_full_eval.add_argument("--run-name", default="deeprm-author-source-clean-evaluation")
    train = sub.add_parser("train", help="Train a clean DeepRM policy")
    train.add_argument("--load", type=float, default=PRIMARY_LOAD)
    train.add_argument("--iterations", type=int, default=None)
    train.add_argument("--num-jobsets", type=int, default=None)
    train.add_argument("--rollouts-per-jobset", type=int, default=None)
    train.add_argument("--master-seed", type=int, default=20260514)
    train.add_argument("--checkpoint-interval", type=int, default=50)
    train.add_argument("--eval-interval", type=int, default=10)
    train.add_argument("--max-episode-steps", type=int, default=2000)
    train.add_argument("--episode-horizon", type=int, default=50)
    train.add_argument("--rollout-workers", type=int, default=1, help="Parallel jobset rollout workers")
    train.add_argument("--run-label", default=None, help="Checkpoint/result namespace for this training run")
    train.add_argument("--author-source", action="store_true", help="Use source-aligned DeepRM config")
    train.add_argument(
        "--visible-slots",
        type=int,
        default=None,
        help="Override DeepRM visible queue slots M for action-space ablation experiments.",
    )
    train.add_argument(
        "--train-end",
        choices=("all-done", "no-new-job"),
        default="all-done",
        help="Official DeepRM training uses all-done drain semantics.",
    )
    train.add_argument("--run-name", default=None)
    train.add_argument("--resume-from", type=Path, default=None, help="Resume policy weights from a checkpoint")
    train.add_argument("--smoke", action="store_true", help="Use tiny settings for pipeline verification")
    clean_eval = sub.add_parser("evaluate-clean", help="Evaluate a clean DeepRM checkpoint against classical baselines")
    clean_eval.add_argument("--checkpoint", type=Path, required=True)
    clean_eval.add_argument("--load", type=float, default=PRIMARY_LOAD)
    clean_eval.add_argument("--num-seeds", type=int, default=EVAL_NUM_SEEDS)
    clean_eval.add_argument("--trace-jobs", type=int, default=EVAL_TRACE_JOBS)
    clean_eval.add_argument("--seed", type=int, default=20260516)
    clean_eval.add_argument("--policy-seed", type=int, default=None)
    clean_eval.add_argument("--max-steps", type=int, default=100_000)
    clean_eval.add_argument(
        "--strict-admission",
        action="store_true",
        help="Override checkpoint env config to keep all submitted jobs queued instead of dropping overflow.",
    )
    clean_eval.add_argument(
        "--policy-mode",
        choices=("stochastic", "deterministic"),
        default="stochastic",
        help="Official DeepRM reproduction uses stochastic sampling; deterministic argmax is diagnostic.",
    )
    clean_eval.add_argument("--run-name", default=None)
    perturb_eval = sub.add_parser(
        "evaluate-perturbations",
        help="Evaluate DeepRM P1/P2/P3 perturbation sweeps from a clean checkpoint",
    )
    perturb_eval.add_argument("--checkpoint", type=Path, required=True)
    perturb_eval.add_argument("--load", type=float, default=PRIMARY_LOAD)
    perturb_eval.add_argument("--num-seeds", type=int, default=EVAL_NUM_SEEDS)
    perturb_eval.add_argument("--trace-jobs", type=int, default=EVAL_TRACE_JOBS)
    perturb_eval.add_argument("--seed", type=int, default=20260520)
    perturb_eval.add_argument("--policy-seed", type=int, default=None)
    perturb_eval.add_argument("--max-steps", type=int, default=100_000)
    perturb_eval.add_argument(
        "--policy-mode",
        choices=("stochastic", "deterministic"),
        default="stochastic",
        help="Use stochastic sampling for the primary official-mode DeepRM evaluation.",
    )
    perturb_eval.add_argument("--run-name", default="deeprm-v2.2-perturbation-sweeps")
    return parser


def cmd_manifest(root: Path) -> None:
    with start_tracked_run(run_name="manifest", role="provenance", root=root) as run:
        payload = {"manifest": protocol_manifest(root)}
        out = root / "results" / "provenance" / "manifest.json"
        write_json_with_run_id(out, payload, run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(f"Artifact: {out}")


def cmd_protocol() -> None:
    payload = {
        "primary_load": PRIMARY_LOAD,
        "load_sweep": LOAD_SWEEP,
        "lag_sweep": LAG_SWEEP,
        "tail_sweep": ["inf" if x == float("inf") else x for x in TAIL_SWEEP],
        "adversarial_eps_sweep": ADVERSARIAL_EPS_SWEEP,
        "anchors": {
            "lag": ANCHOR_LAG,
            "tail_alpha": ANCHOR_TAIL_ALPHA,
            "epsilon": ANCHOR_EPSILON,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_test(root: Path) -> None:
    junit = root / "logs" / "tests" / "junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    with start_tracked_run(run_name="unit-tests", role="verification", root=root) as run:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", f"--junitxml={junit}"],
            cwd=root,
            check=False,
            text=True,
        )
        mlflow.log_metric("pytest_returncode", result.returncode)
        if junit.exists():
            mlflow.log_artifact(str(junit), artifact_path="tests")
        print(f"MLflow run: {run.info.run_id}")
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def cmd_gpu_check(root: Path) -> None:
    import subprocess
    import torch

    with start_tracked_run(run_name="gpu-check", role="hardware-check", root=root) as run:
        report: dict[str, object] = {
            "torch_version": torch.__version__,
            "torch_cuda_built": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            report["device_name_0"] = torch.cuda.get_device_name(0)
            report["device_capability_0"] = torch.cuda.get_device_capability(0)
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            report["cuda_matmul_probe"] = float(y[0, 0].detach().cpu())
        try:
            smi = subprocess.run(
                ["/usr/lib/wsl/lib/nvidia-smi"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            report["nvidia_smi_returncode"] = smi.returncode
            report["nvidia_smi"] = smi.stdout
        except Exception as exc:
            report["nvidia_smi_error"] = str(exc)
        out = root / "results" / "provenance" / "gpu_check.json"
        write_json_with_run_id(out, report, run.info.run_id)
        mlflow.log_metric("cuda_available", 1.0 if report["cuda_available"] else 0.0)
        mlflow.log_metric("cuda_device_count", float(report["device_count"]))
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(report, indent=2, sort_keys=True, default=str))


def cmd_author_preflight(root: Path) -> None:
    import numpy as np

    paper_config = DeepRMConfig()
    source_config = author_source_config(PRIMARY_LOAD)
    rng = np.random.default_rng(123)
    source_demands = [sample_demand(rng, source_config) for _ in range(200)]
    rng = np.random.default_rng(123)
    source_durations = [sample_duration(rng, source_config, float("inf")) for _ in range(200)]
    source_policy = DeepRMPolicy(source_config)
    paper_policy = DeepRMPolicy(paper_config)
    trace = generate_time_trace(horizon=20, rate=0.7, seed=1234, config=source_config)
    baseline_metrics = {}
    for scheduler in (SJFScheduler(), PackerScheduler(source_dot=True), SourceTetrisScheduler()):
        env = DeepRMEnv(trace, config=source_config, drain=True)
        steps = 0
        while not env.done and steps < 1000:
            env.step(scheduler.act(env))
            steps += 1
        baseline_metrics[scheduler.name] = {
            "done": env.done,
            "steps": steps,
            "completed": len(env.completed),
            "dropped": len(env.dropped),
            "mean_slowdown": env.mean_slowdown() if env.completed else None,
        }

    reward_trace = WorkloadTrace(
        jobs=(Job(id=0, arrival_time=0, duration=2, demand=(5.0, 1.0)),),
        rate=1.0,
        tail_alpha=float("inf"),
        seed=1,
        horizon=1,
    )
    reward_env = DeepRMEnv(reward_trace, config=source_config, drain=True)
    _, allocate_reward, _, allocate_info = reward_env.step(0)
    _, move_reward, _, move_info = reward_env.step(source_config.visible_slots)
    payload = {
        "source_repo": "https://github.com/hongzimao/deeprm",
        "source_commit": _git_head(root / "external" / "deeprm_official"),
        "paper_config": {
            "state_shape": paper_config.state_shape,
            "parameter_count": paper_policy.num_parameters,
            "include_extra_info": paper_config.include_extra_info,
            "demand_mode": paper_config.demand_mode,
        },
        "source_config": {
            "state_shape": source_config.state_shape,
            "parameter_count": source_policy.num_parameters,
            "include_extra_info": source_config.include_extra_info,
            "demand_mode": source_config.demand_mode,
            "resource_capacity": source_config.resource_capacity,
            "dominant_demand_min": source_config.dominant_demand_min,
            "dominant_demand_max": source_config.dominant_demand_max,
            "nondominant_demand_min": source_config.nondominant_demand_min,
            "nondominant_demand_max": source_config.nondominant_demand_max,
            "reward_on_allocate": source_config.reward_on_allocate,
            "external_admission": source_config.external_admission,
            "max_start_inclusive": source_config.max_start_inclusive,
        },
        "source_demand_minmax": {
            "res0": [float(min(d[0] for d in source_demands)), float(max(d[0] for d in source_demands))],
            "res1": [float(min(d[1] for d in source_demands)), float(max(d[1] for d in source_demands))],
        },
        "source_duration_values": sorted(set(source_durations)),
        "baseline_smoke": baseline_metrics,
        "reward_timing_check": {
            "allocate_status": allocate_info.status,
            "allocate_reward": allocate_reward,
            "move_status": move_info.status,
            "move_reward": move_reward,
        },
        "status": "passed",
    }
    checks = [
        source_config.state_shape == (20, 224, 1),
        source_policy.num_parameters == 89_851,
        paper_policy.num_parameters == 89_451,
        payload["reward_timing_check"]["allocate_reward"] == 0.0,
        all(row["done"] for row in baseline_metrics.values()),
    ]
    if not all(checks):
        payload["status"] = "failed"
        payload["checks"] = checks
    with start_tracked_run(
        run_name="deeprm-author-source-preflight",
        role="author-alignment-preflight",
        root=root,
        params={"source_commit": payload["source_commit"], "load": PRIMARY_LOAD},
    ) as run:
        mlflow.log_metric("author_preflight_passed", 1.0 if payload["status"] == "passed" else 0.0)
        mlflow.log_metric("source_policy_parameter_count", source_policy.num_parameters)
        mlflow.log_metric("paper_policy_parameter_count", paper_policy.num_parameters)
        out = root / "results" / "provenance" / "deeprm_author_preflight.json"
        write_json_with_run_id(out, payload, run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if payload["status"] != "passed":
        raise SystemExit(1)


def cmd_author_eval_smoke(args: argparse.Namespace, root: Path) -> None:
    checkpoint = root / args.checkpoint
    raw = torch.load(checkpoint, map_location="cpu")
    metadata = raw.get("metadata", {})
    env_config = DeepRMConfig(**metadata.get("env_config", {})) if metadata.get("env_config") else author_source_config(args.load)
    if env_config.demand_mode != "source_discrete" or not env_config.include_extra_info:
        raise SystemExit("author-eval-smoke requires an author-source checkpoint/config")
    policy = load_checkpoint(checkpoint, config=env_config)
    trace = generate_time_trace(
        horizon=args.trace_horizon,
        rate=args.load,
        seed=args.seed,
        config=env_config,
    )
    generator = torch.Generator().manual_seed(args.seed)
    schedulers = (
        DeepRMScheduler(
            policy=policy,
            deterministic=args.policy_mode == "deterministic",
            generator=None if args.policy_mode == "deterministic" else generator,
        ),
        SJFScheduler(),
        PackerScheduler(source_dot=True),
        SourceTetrisScheduler(),
    )
    metrics = {}
    for scheduler in schedulers:
        env = DeepRMEnv(trace, config=env_config, drain=True)
        steps = 0
        while not env.done and steps < args.max_steps:
            env.step(scheduler.act(env))
            steps += 1
        completed_slowdowns = [job.slowdown for job in env.completed]
        completed_response_times = [
            (job.finish_time or 0) - job.arrival_time for job in env.completed
        ]
        metrics[scheduler.name] = {
            "done": env.done,
            "steps": steps,
            "submitted_jobs": len(trace.jobs),
            "completed_jobs": len(env.completed),
            "dropped_jobs": len(env.dropped),
            "drop_fraction": len(env.dropped) / float(max(1, len(trace.jobs))),
            "mean_slowdown_finished": float(sum(completed_slowdowns) / len(completed_slowdowns))
            if completed_slowdowns
            else None,
            "p95_completion_time_finished": float(torch.quantile(torch.tensor(completed_response_times, dtype=torch.float32), 0.95).item())
            if completed_response_times
            else None,
        }
    payload = {
        "status": "passed" if all(row["done"] for row in metrics.values()) else "failed",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "source_commit": _git_head(root / "external" / "deeprm_official"),
        "load": args.load,
        "trace_horizon": args.trace_horizon,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "policy_mode": args.policy_mode,
        "evaluation_semantics": "author_source_finished_jobs_drop_accounting",
        "metrics": metrics,
    }
    with start_tracked_run(
        run_name=args.run_name,
        role="author-source-evaluation-smoke",
        root=root,
        params={
            "checkpoint": str(args.checkpoint),
            "load": args.load,
            "trace_horizon": args.trace_horizon,
            "seed": args.seed,
            "policy_mode": args.policy_mode,
        },
    ) as run:
        mlflow.log_metric("author_eval_smoke_passed", 1.0 if payload["status"] == "passed" else 0.0)
        for method, row in metrics.items():
            key = _metric_name(method)
            mlflow.log_metric(f"author_eval.completed_jobs.{key}", row["completed_jobs"])
            mlflow.log_metric(f"author_eval.dropped_jobs.{key}", row["dropped_jobs"])
            if row["mean_slowdown_finished"] is not None:
                mlflow.log_metric(f"author_eval.mean_slowdown_finished.{key}", row["mean_slowdown_finished"])
        out = root / "results" / "evaluation" / "deeprm" / "author_source_eval_smoke.json"
        write_json_with_run_id(out, payload, run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if payload["status"] != "passed":
        raise SystemExit(1)


def _finished_metrics(env: DeepRMEnv, trace: WorkloadTrace, steps: int) -> dict[str, object]:
    completed_slowdowns = [job.slowdown for job in env.completed]
    completed_response_times = [
        (job.finish_time or 0) - job.arrival_time for job in env.completed
    ]
    return {
        "done": env.done,
        "steps": steps,
        "submitted_jobs": len(trace.jobs),
        "completed_jobs": len(env.completed),
        "dropped_jobs": len(env.dropped),
        "visible_jobs_remaining": sum(job is not None for job in env.visible_slots),
        "backlog_jobs_remaining": len(env.backlog),
        "external_jobs_remaining": len(env.external),
        "running_jobs_remaining": len(env.machine.running),
        "completed_fraction": len(env.completed) / float(max(1, len(trace.jobs))),
        "drop_fraction": len(env.dropped) / float(max(1, len(trace.jobs))),
        "mean_slowdown_finished": float(sum(completed_slowdowns) / len(completed_slowdowns))
        if completed_slowdowns
        else None,
        "p95_completion_time_finished": float(torch.quantile(torch.tensor(completed_response_times, dtype=torch.float32), 0.95).item())
        if completed_response_times
        else None,
    }


def _run_source_style_episode(scheduler, trace: WorkloadTrace, env_config: DeepRMConfig, max_steps: int) -> dict[str, object]:
    env = DeepRMEnv(trace, config=env_config, drain=True)
    steps = 0
    while not env.done and steps < max_steps:
        env.step(scheduler.act(env))
        steps += 1
    return _finished_metrics(env, trace, steps)


def cmd_author_evaluate(args: argparse.Namespace, root: Path) -> None:
    import numpy as np

    checkpoint = root / args.checkpoint
    raw = torch.load(checkpoint, map_location="cpu")
    metadata = raw.get("metadata", {})
    env_config = DeepRMConfig(**metadata.get("env_config", {})) if metadata.get("env_config") else author_source_config(args.load)
    if env_config.demand_mode != "source_discrete" or not env_config.include_extra_info:
        raise SystemExit("author-evaluate requires an author-source checkpoint/config")
    policy = load_checkpoint(checkpoint, config=env_config)
    seed_seq = np.random.SeedSequence(args.seed)
    trace_seeds = [int(child.generate_state(1)[0]) for child in seed_seq.spawn(args.num_seeds)]
    traces = tuple(
        generate_time_trace(
            horizon=args.trace_horizon,
            rate=args.load,
            seed=trace_seed,
            config=env_config,
        )
        for trace_seed in trace_seeds
    )
    policy_generator = torch.Generator().manual_seed(args.seed)
    methods = {
        "DeepRM": DeepRMScheduler(
            policy=policy,
            deterministic=args.policy_mode == "deterministic",
            generator=None if args.policy_mode == "deterministic" else policy_generator,
        ),
        "SJF": SJFScheduler(),
        "Packer": PackerScheduler(source_dot=True),
        "SourceTetris": SourceTetrisScheduler(),
        "Random": RandomScheduler(seed=args.seed),
    }
    metrics_by_method: dict[str, list[dict[str, object]]] = {name: [] for name in methods}
    for trace in traces:
        for name, scheduler in methods.items():
            metrics_by_method[name].append(
                _run_source_style_episode(scheduler, trace, env_config, args.max_steps)
            )

    summary_by_method = {}
    for name, rows in metrics_by_method.items():
        slowdowns = [row["mean_slowdown_finished"] for row in rows if row["mean_slowdown_finished"] is not None]
        summary_by_method[name] = {
            "mean_slowdown_finished": float(np.mean(slowdowns)) if slowdowns else None,
            "median_slowdown_finished": float(np.median(slowdowns)) if slowdowns else None,
            "mean_completed_fraction": float(np.mean([row["completed_fraction"] for row in rows])),
            "mean_drop_fraction": float(np.mean([row["drop_fraction"] for row in rows])),
            "all_episodes_done": all(bool(row["done"]) for row in rows),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
        }
    deep_rows = metrics_by_method["DeepRM"]
    comparisons = {}
    for name, rows in metrics_by_method.items():
        if name == "DeepRM":
            continue
        paired = [
            float(row["mean_slowdown_finished"]) - float(deep["mean_slowdown_finished"])
            for row, deep in zip(rows, deep_rows, strict=True)
            if row["mean_slowdown_finished"] is not None and deep["mean_slowdown_finished"] is not None
        ]
        if paired:
            comparisons[f"{name}_minus_DeepRM"] = {
                "mean_difference": float(np.mean(paired)),
                "num_pairs": len(paired),
                "differences": paired,
            }
    source_tetris_delta = comparisons.get("SourceTetris_minus_DeepRM", {}).get("mean_difference")
    sjf_delta = comparisons.get("SJF_minus_DeepRM", {}).get("mean_difference")
    gate_passed = bool(
        source_tetris_delta is not None
        and sjf_delta is not None
        and source_tetris_delta > 0.0
        and sjf_delta > 0.0
        and summary_by_method["DeepRM"]["all_episodes_done"]
        and summary_by_method["DeepRM"]["mean_drop_fraction"] <= 0.01
    )
    payload = {
        "status": "passed",
        "gate_passed": gate_passed,
        "gate_definition": (
            "DeepRM source-style finished-job slowdown lower than SourceTetris and SJF, "
            "DeepRM episodes all done, and DeepRM mean drop fraction <= 0.01."
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "source_commit": _git_head(root / "external" / "deeprm_official"),
        "load": args.load,
        "num_seeds": args.num_seeds,
        "trace_horizon": args.trace_horizon,
        "trace_seeds": trace_seeds,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "policy_mode": args.policy_mode,
        "evaluation_semantics": "author_source_finished_jobs_drop_accounting",
        "summary": summary_by_method,
        "comparisons": comparisons,
        "per_seed_metrics": metrics_by_method,
    }
    with start_tracked_run(
        run_name=args.run_name,
        role="author-source-clean-evaluation",
        root=root,
        params={
            "checkpoint": str(args.checkpoint),
            "load": args.load,
            "num_seeds": args.num_seeds,
            "trace_horizon": args.trace_horizon,
            "seed": args.seed,
            "policy_mode": args.policy_mode,
            "max_steps": args.max_steps,
        },
    ) as run:
        mlflow.log_metric("author_eval.gate_passed", 1.0 if gate_passed else 0.0)
        for method, row in summary_by_method.items():
            key = _metric_name(method)
            if row["mean_slowdown_finished"] is not None:
                mlflow.log_metric(f"author_eval.mean_slowdown_finished.{key}", row["mean_slowdown_finished"])
            mlflow.log_metric(f"author_eval.mean_completed_fraction.{key}", row["mean_completed_fraction"])
            mlflow.log_metric(f"author_eval.mean_drop_fraction.{key}", row["mean_drop_fraction"])
        for name, row in comparisons.items():
            mlflow.log_metric(f"author_eval.delta.{_metric_name(name)}", row["mean_difference"])
        out = root / "results" / "evaluation" / "deeprm" / "author_source_clean_load_0.7.json"
        write_json_with_run_id(out, payload, run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload["summary"], indent=2, sort_keys=True, default=str))
        print(json.dumps({"gate_passed": gate_passed, "comparisons": {k: v["mean_difference"] for k, v in comparisons.items()}}, indent=2, sort_keys=True))


def cmd_train(args: argparse.Namespace, root: Path) -> None:
    if args.smoke:
        iterations = args.iterations if args.iterations is not None else 2
        num_jobsets = args.num_jobsets if args.num_jobsets is not None else 2
        rollouts = args.rollouts_per_jobset if args.rollouts_per_jobset is not None else 2
        checkpoint_interval = min(args.checkpoint_interval, iterations)
        eval_interval = 1
        role = "training-smoke"
        run_label = args.run_label or "smoke"
    else:
        iterations = args.iterations if args.iterations is not None else 1000
        num_jobsets = args.num_jobsets if args.num_jobsets is not None else 100
        rollouts = args.rollouts_per_jobset if args.rollouts_per_jobset is not None else 20
        checkpoint_interval = args.checkpoint_interval
        eval_interval = args.eval_interval
        role = "training"
        run_label = args.run_label or "clean"
    env_config = author_source_config(args.load) if args.author_source else DeepRMConfig(primary_load=args.load)
    if args.visible_slots is not None:
        if args.visible_slots < 1:
            raise SystemExit("--visible-slots must be >= 1")
        env_config = DeepRMConfig(**{**env_config.__dict__, "visible_slots": args.visible_slots})
    config = TrainConfig(
        load=args.load,
        run_label=run_label,
        master_seed=args.master_seed,
        iterations=iterations,
        num_jobsets=num_jobsets,
        rollouts_per_jobset=rollouts,
        episode_horizon=args.episode_horizon,
        checkpoint_interval=checkpoint_interval,
        eval_interval=eval_interval,
        max_episode_steps=args.max_episode_steps,
        drain=args.train_end == "all-done",
        resume_from_checkpoint=str(args.resume_from) if args.resume_from else None,
        rollout_workers=args.rollout_workers,
    )
    if args.resume_from and role == "training":
        role = "training-resume"
    run_name = args.run_name or f"{role}-load-{args.load}"
    with start_tracked_run(
        run_name=run_name,
        role=role,
        root=root,
        params={
            "load": args.load,
            "smoke": args.smoke,
            "iterations": iterations,
            "num_jobsets": num_jobsets,
            "rollouts_per_jobset": rollouts,
            "max_episode_steps": args.max_episode_steps,
            "train_end": args.train_end,
            "run_label": run_label,
            "episode_horizon": args.episode_horizon,
            "author_source": args.author_source,
            "visible_slots": env_config.visible_slots,
            "action_dim": env_config.action_dim,
            "rollout_workers": args.rollout_workers,
            "resume_from_checkpoint": str(args.resume_from) if args.resume_from else None,
        },
    ) as run:
        print(f"MLflow run started: {run.info.run_id}", flush=True)
        summary = train_policy(config, root=root, env_config=env_config)
        payload = {
            "summary": summary.__dict__,
            "train_config": config.__dict__,
        }
        out = root / "results" / "training" / f"{role}_load_{args.load}.json"
        write_json_with_run_id(out, payload, run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(f"Checkpoint: {summary.checkpoint_path}")


def cmd_evaluate_clean(args: argparse.Namespace, root: Path) -> None:
    checkpoint = root / args.checkpoint
    raw = torch.load(checkpoint, map_location="cpu")
    metadata = raw.get("metadata", {})
    env_config = DeepRMConfig(**metadata.get("env_config", {})) if metadata.get("env_config") else DeepRMConfig(primary_load=args.load)
    if args.strict_admission:
        env_config = DeepRMConfig(**{**env_config.__dict__, "external_admission": True})
    policy = load_checkpoint(checkpoint, config=env_config)
    traces = generate_eval_traces(
        load=args.load,
        num_seeds=args.num_seeds,
        trace_jobs=args.trace_jobs,
        seed=args.seed,
        config=env_config,
    )
    checkpoint_sha256 = _sha256(checkpoint)
    run_name = args.run_name or f"clean-eval-load-{args.load}"
    with start_tracked_run(
        run_name=run_name,
        role="clean-evaluation",
        root=root,
        params={
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "load": args.load,
            "num_seeds": args.num_seeds,
            "trace_jobs": args.trace_jobs,
            "seed": args.seed,
            "policy_seed": args.policy_seed if args.policy_seed is not None else args.seed,
            "policy_mode": args.policy_mode,
            "max_steps": args.max_steps,
            "strict_admission": args.strict_admission,
            "eval_external_admission": env_config.external_admission,
            "resume_iteration": metadata.get("resume_iteration"),
            "resume_optimizer_state": metadata.get("resume_optimizer_state"),
            "source_training_run_id": metadata.get("mlflow_run_id"),
        },
    ) as run:
        print(f"MLflow run started: {run.info.run_id}", flush=True)
        try:
            summary, metrics = evaluate_clean_policy(
                policy,
                traces,
                checkpoint_path=str(args.checkpoint),
                load=args.load,
                seed=args.seed,
                config=env_config,
                policy_deterministic=args.policy_mode == "deterministic",
                policy_seed=args.policy_seed,
                max_steps=args.max_steps,
            )
        except RuntimeError as exc:
            mlflow.log_metric("clean.evaluation_failed", 1.0)
            mlflow.log_metric("clean.strict_gate_passed", 0.0)
            mlflow.log_metric("clean.mean_gate_passed", 0.0)
            failure_payload = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "checkpoint": str(args.checkpoint),
                "checkpoint_metadata": metadata,
                "checkpoint_sha256": checkpoint_sha256,
                "load": args.load,
                "num_seeds": args.num_seeds,
                "trace_jobs": args.trace_jobs,
                "seed": args.seed,
                "policy_seed": args.policy_seed if args.policy_seed is not None else args.seed,
                "policy_mode": args.policy_mode,
                "max_steps": args.max_steps,
                "strict_admission": args.strict_admission,
                "eval_external_admission": env_config.external_admission,
            }
            out = (
                root
                / "results"
                / "evaluation"
                / "deeprm"
                / f"clean_load_{args.load}_{args.policy_mode}_failure.json"
            )
            write_json_with_run_id(out, failure_payload, run.info.run_id)
            print(f"MLflow run: {run.info.run_id}")
            print(json.dumps(failure_payload, indent=2, sort_keys=True, default=str))
            raise
        payload = clean_evaluation_payload(summary, metrics)
        payload["checkpoint_metadata"] = metadata
        payload["checkpoint_sha256"] = checkpoint_sha256
        payload["eval_env_config"] = env_config.__dict__
        payload["strict_admission"] = args.strict_admission
        for method, value in summary.method_means.items():
            mlflow.log_metric(f"clean.mean_slowdown.{_metric_name(method)}", value)
        for name, result in summary.comparisons.items():
            key = _metric_name(name)
            mlflow.log_metric(f"clean.delta.{key}_minus_deeprm", result.mean_difference)
            mlflow.log_metric(f"clean.ci_low.{key}_minus_deeprm", result.ci_low)
            mlflow.log_metric(f"clean.ci_high.{key}_minus_deeprm", result.ci_high)
            mlflow.log_metric(f"clean.p_greater.{key}_minus_deeprm", result.p_greater_than_zero)
        mlflow.log_metric("clean.strict_gate_passed", 1.0 if summary.strict_gate_passed else 0.0)
        mlflow.log_metric("clean.mean_gate_passed", 1.0 if summary.mean_gate_passed else 0.0)
        out = root / "results" / "evaluation" / "deeprm" / f"clean_load_{args.load}.json"
        write_json_with_run_id(out, payload, run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload["summary"], indent=2, sort_keys=True, default=str))


def cmd_evaluate_perturbations(args: argparse.Namespace, root: Path) -> None:
    import numpy as np

    checkpoint = root / args.checkpoint
    raw = torch.load(checkpoint, map_location="cpu")
    metadata = raw.get("metadata", {})
    env_config = DeepRMConfig(**metadata.get("env_config", {})) if metadata.get("env_config") else DeepRMConfig(primary_load=args.load)
    base_eval_config = DeepRMConfig(**{**env_config.__dict__, "external_admission": True})
    policy = load_checkpoint(checkpoint, config=base_eval_config)
    tetris = TetrisScheduler(source_dot=True)
    trace_seeds = _trace_seeds(args.seed, args.num_seeds)
    policy_seed = args.seed if args.policy_seed is None else args.policy_seed
    policy_deterministic = args.policy_mode == "deterministic"

    cells: dict[str, dict[str, object]] = {}

    base_traces = tuple(
        generate_trace_for_seed(args.load, args.trace_jobs, seed, base_eval_config, float("inf"))
        for seed in trace_seeds
    )

    for idx, lag in enumerate(LAG_SWEEP):
        generator = _cell_generator(policy_seed, 1_000 + idx)
        deep_metrics = tuple(
            run_lagged_policy_episode(
                policy,
                trace,
                lag=lag,
                config=base_eval_config,
                policy_deterministic=policy_deterministic,
                policy_generator=None if policy_deterministic else generator,
                max_steps=args.max_steps,
            )
            for trace in base_traces
        )
        tetris_metrics = tuple(
            run_lagged_scheduler_episode(
                tetris,
                trace,
                lag=lag,
                config=base_eval_config,
                max_steps=args.max_steps,
            )
            for trace in base_traces
        )
        cells[f"P1_lag_{lag}"] = _perturbation_cell_payload(
            deep_metrics,
            tetris_metrics,
            statistic_seed=args.seed + 10_000 + idx,
            parameter_name="lag",
            parameter_value=lag,
        )

    for idx, alpha in enumerate(TAIL_SWEEP):
        tail_config = _tail_eval_config(base_eval_config, alpha)
        tail_traces = tuple(
            generate_trace_for_seed(args.load, args.trace_jobs, seed, tail_config, alpha)
            for seed in trace_seeds
        )
        generator = _cell_generator(policy_seed, 2_000 + idx)
        deep_scheduler = DeepRMScheduler(
            policy=policy,
            deterministic=policy_deterministic,
            generator=None if policy_deterministic else generator,
        )
        deep_metrics = tuple(
            run_episode(deep_scheduler, trace, config=tail_config, max_steps=args.max_steps)
            for trace in tail_traces
        )
        tetris_metrics = tuple(
            run_episode(tetris, trace, config=tail_config, max_steps=args.max_steps)
            for trace in tail_traces
        )
        label = "inf" if alpha == float("inf") else str(alpha)
        cells[f"P2_tail_{label}"] = _perturbation_cell_payload(
            deep_metrics,
            tetris_metrics,
            statistic_seed=args.seed + 20_000 + idx,
            parameter_name="tail_alpha",
            parameter_value=label,
            eval_env_config=tail_config.__dict__,
        )

    for idx, epsilon in enumerate(ADVERSARIAL_EPS_SWEEP):
        generator = _cell_generator(policy_seed, 3_000 + idx)
        deep_metrics = tuple(
            run_adversarial_policy_episode(
                policy,
                trace,
                epsilon=epsilon,
                config=base_eval_config,
                policy_deterministic=policy_deterministic,
                policy_generator=None if policy_deterministic else generator,
                max_steps=args.max_steps,
            )
            for trace in base_traces
        )
        tetris_metrics = tuple(
            run_episode(tetris, trace, config=base_eval_config, max_steps=args.max_steps)
            for trace in base_traces
        )
        cells[f"P3_epsilon_{epsilon}"] = _perturbation_cell_payload(
            deep_metrics,
            tetris_metrics,
            statistic_seed=args.seed + 30_000 + idx,
            parameter_name="epsilon",
            parameter_value=epsilon,
        )

    anchors = {
        "P1-DeepRM": cells[f"P1_lag_{ANCHOR_LAG}"]["comparison"],
        "P2-DeepRM": cells[f"P2_tail_{ANCHOR_TAIL_ALPHA}"]["comparison"],
        "P3-DeepRM": cells[f"P3_epsilon_{ANCHOR_EPSILON}"]["comparison"],
    }
    confirmation_raw_p = {
        name: float(row["p_less_than_zero"])
        for name, row in anchors.items()
    }
    falsification_raw_p = {
        name: float(row["p_greater_than_zero"])
        for name, row in anchors.items()
    }
    confirmation_holm = holm_bonferroni(confirmation_raw_p)
    falsification_holm = holm_bonferroni(falsification_raw_p)
    anchor_verdicts = {
        name: _directional_verdict(
            comparison,
            confirmation_p=confirmation_holm[name],
            falsification_p=falsification_holm[name],
        )
        for name, comparison in anchors.items()
    }
    summary = {
        "policy_mode": "deterministic_argmax" if policy_deterministic else "stochastic_sample",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "load": args.load,
        "num_seeds": args.num_seeds,
        "trace_jobs": args.trace_jobs,
        "seed": args.seed,
        "policy_seed": policy_seed,
        "max_steps": args.max_steps,
        "strict_admission": True,
        "comparator": "Tetris*(alpha=0.5,dot_product_packing)",
        "anchor_verdicts_deeprm_only_holm": anchor_verdicts,
        "cross_method_holm_status": "pending_decima_and_rossi_pvalues",
    }
    payload = {
        "summary": summary,
        "anchor_confirmation_raw_p": confirmation_raw_p,
        "anchor_confirmation_holm_deeprm_only": confirmation_holm,
        "anchor_falsification_raw_p": falsification_raw_p,
        "anchor_falsification_holm_deeprm_only": falsification_holm,
        "checkpoint_metadata": metadata,
        "eval_env_config": base_eval_config.__dict__,
        "trace_seeds": trace_seeds,
        "cells": cells,
    }

    with start_tracked_run(
        run_name=args.run_name,
        role="deeprm-perturbation-evaluation",
        root=root,
        params={
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "load": args.load,
            "num_seeds": args.num_seeds,
            "trace_jobs": args.trace_jobs,
            "seed": args.seed,
            "policy_seed": policy_seed,
            "policy_mode": args.policy_mode,
            "max_steps": args.max_steps,
            "strict_admission": True,
        },
    ) as run:
        for name, cell in cells.items():
            key = _metric_name(name)
            mlflow.log_metric(f"perturb.deep_mean_slowdown.{key}", cell["deep_rm_mean_slowdown"])
            mlflow.log_metric(f"perturb.tetris_mean_slowdown.{key}", cell["tetris_mean_slowdown"])
            mlflow.log_metric(f"perturb.delta_mean.{key}", cell["comparison"]["mean_difference"])
            mlflow.log_metric(f"perturb.delta_ci_low.{key}", cell["comparison"]["ci_low"])
            mlflow.log_metric(f"perturb.delta_ci_high.{key}", cell["comparison"]["ci_high"])
        for name in anchors:
            key = _metric_name(name)
            mlflow.log_metric(f"perturb.anchor_confirmation_p.{key}", confirmation_raw_p[name])
            mlflow.log_metric(f"perturb.anchor_confirmation_holm_deeprm_only.{key}", confirmation_holm[name])
            mlflow.log_metric(f"perturb.anchor_falsification_p.{key}", falsification_raw_p[name])
            mlflow.log_metric(f"perturb.anchor_falsification_holm_deeprm_only.{key}", falsification_holm[name])
        out = root / "results" / "evaluation" / "deeprm" / "perturbation_sweeps_v2_2.json"
        write_json_with_run_id(out, payload, run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))


def generate_trace_for_seed(
    load: float,
    trace_jobs: int,
    seed: int,
    config: DeepRMConfig,
    tail_alpha: float,
):
    from cisose_deeprm.workload import generate_trace

    return generate_trace(
        num_jobs=trace_jobs,
        rate=load,
        seed=seed,
        config=config,
        tail_alpha=tail_alpha,
    )


def _trace_seeds(seed: int, num_seeds: int) -> list[int]:
    import numpy as np

    seed_seq = np.random.SeedSequence(seed)
    return [int(child.generate_state(1)[0]) for child in seed_seq.spawn(num_seeds)]


def _cell_generator(policy_seed: int, cell_offset: int) -> torch.Generator:
    import numpy as np

    seed = int(np.random.SeedSequence([policy_seed, cell_offset]).generate_state(1)[0])
    return torch.Generator().manual_seed(seed)


def _tail_eval_config(config: DeepRMConfig, alpha: float) -> DeepRMConfig:
    if alpha == float("inf"):
        return config
    planning_horizon = max(
        config.time_horizon,
        config.tail_x_max + (0 if config.max_start_inclusive else 1),
    )
    return DeepRMConfig(**{**config.__dict__, "planning_horizon": planning_horizon})


def _perturbation_cell_payload(
    deep_metrics,
    tetris_metrics,
    *,
    statistic_seed: int,
    parameter_name: str,
    parameter_value,
    eval_env_config: dict[str, object] | None = None,
) -> dict[str, object]:
    result = paired_result(tetris_metrics, deep_metrics, seed=statistic_seed)
    payload = {
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "deep_rm_mean_slowdown": float(
            sum(metric.mean_slowdown for metric in deep_metrics) / len(deep_metrics)
        ),
        "tetris_mean_slowdown": float(
            sum(metric.mean_slowdown for metric in tetris_metrics) / len(tetris_metrics)
        ),
        "comparison": asdict(result),
        "deep_rm_metrics": [asdict(metric) for metric in deep_metrics],
        "tetris_metrics": [asdict(metric) for metric in tetris_metrics],
    }
    if eval_env_config is not None:
        payload["eval_env_config"] = eval_env_config
    return payload


def _directional_verdict(
    comparison: dict[str, object],
    *,
    confirmation_p: float,
    falsification_p: float,
) -> str:
    ci_low = float(comparison["ci_low"])
    ci_high = float(comparison["ci_high"])
    if ci_high < 0.0 and confirmation_p < 0.05:
        return "confirmed_deeprm_loses"
    if ci_low > 0.0 and falsification_p < 0.05:
        return "falsified_deeprm_still_wins"
    return "inconclusive"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metric_name(name: str) -> str:
    return name.lower().replace("*", "star").replace(" ", "_")


def _git_head(path: Path) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "manifest":
        cmd_manifest(root)
    elif args.command == "protocol":
        cmd_protocol()
    elif args.command == "test":
        cmd_test(root)
    elif args.command == "gpu-check":
        cmd_gpu_check(root)
    elif args.command == "author-preflight":
        cmd_author_preflight(root)
    elif args.command == "author-eval-smoke":
        cmd_author_eval_smoke(args, root)
    elif args.command == "author-evaluate":
        cmd_author_evaluate(args, root)
    elif args.command == "train":
        cmd_train(args, root)
    elif args.command == "evaluate-clean":
        cmd_evaluate_clean(args, root)
    elif args.command == "evaluate-perturbations":
        cmd_evaluate_perturbations(args, root)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
