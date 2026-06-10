"""CLI for Decima implementation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlflow
import torch

from cisose_common.tracking import start_run, write_json_artifact
from cisose_decima.config import DECIMA_COMMIT, DECIMA_REPO_URL, DEFAULT_CONFIG
from cisose_decima.gates import current_readiness
from cisose_decima.graphene import GrapheneStyleComparator
from cisose_decima.model import DecimaPolicy, parameter_count
from cisose_decima.official import run_official_episode, train_pytorch_official
from cisose_decima.preflight import decima_preflight_report
from cisose_decima.reproduction import reference_command_payload
from cisose_decima.rollout import rollout_smoke
from cisose_decima.sampling import sample_templates, sampling_probabilities
from cisose_decima.tpch import load_tpch_templates, template_summary


EXPERIMENT_NAME = "cisose_decima_v2_2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cisose-decima")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol", help="Print official Decima reference configuration")
    sub.add_parser("readiness", help="Print Decima readiness gates")
    sub.add_parser("preflight", help="Print pre-execution Decima readiness checks")
    sub.add_parser("reference-commands", help="Print official README reproduction command record")
    summary = sub.add_parser("tpch-summary", help="Summarize official TPC-H DAG templates")
    summary.add_argument("--tpch-root", type=Path, default=Path("external/decima-sim/spark_env/tpch"))
    graphene = sub.add_parser("graphene-smoke", help="Run an MLflow-tracked Graphene-style comparator smoke check")
    graphene.add_argument("--tpch-root", type=Path, default=Path("external/decima-sim/spark_env/tpch"))
    smoke = sub.add_parser("smoke-run", help="Run an MLflow-tracked Decima model/data smoke check")
    smoke.add_argument("--tpch-root", type=Path, default=Path("external/decima-sim/spark_env/tpch"))
    smoke.add_argument("--seed", type=int, default=20260515)
    smoke.add_argument("--w", type=float, default=0.5)
    official_smoke = sub.add_parser(
        "official-smoke",
        help="Run a small MLflow-tracked official-simulator smoke evaluation",
    )
    official_smoke.add_argument("--seed", type=int, default=42)
    official_smoke.add_argument("--num-stream-dags", type=int, default=5)
    official_smoke.add_argument("--action-mode", choices=("sample", "greedy"), default="greedy")
    official_smoke.add_argument("--device", default="cpu")
    official_train = sub.add_parser(
        "official-train-smoke",
        help="Run a tiny PyTorch training smoke on the official simulator",
    )
    official_train.add_argument("--seed", type=int, default=42)
    official_train.add_argument("--epochs", type=int, default=1)
    official_train.add_argument("--num-agents", type=int, default=1)
    official_train.add_argument("--num-stream-dags", type=int, default=2)
    official_train.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints/decima/official_smoke"))
    official_train.add_argument("--device", default="cpu")
    return parser


def cmd_protocol() -> None:
    readiness = current_readiness()
    payload = {
        "repo_url": DECIMA_REPO_URL,
        "commit": DECIMA_COMMIT,
        "config": DEFAULT_CONFIG.__dict__,
        "readiness": {
            "ready_for_perturbations": readiness.ready_for_perturbations,
            "official_readme_reproduction": readiness.official_readme_reproduction.__dict__,
            "graphene_validation": readiness.graphene_validation.__dict__,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_readiness() -> None:
    readiness = current_readiness()
    payload = {
        "ready_for_perturbations": readiness.ready_for_perturbations,
        "official_readme_reproduction": readiness.official_readme_reproduction.__dict__,
        "graphene_validation": readiness.graphene_validation.__dict__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_tpch_summary(args: argparse.Namespace, root: Path) -> None:
    templates = load_tpch_templates(root / args.tpch_root)
    print(json.dumps(template_summary(templates), indent=2, sort_keys=True))


def cmd_graphene_smoke(args: argparse.Namespace, root: Path) -> None:
    templates = load_tpch_templates(root / args.tpch_root)
    template = templates[0]
    comparator = GrapheneStyleComparator()
    schedule = comparator.preferred_schedule(template)
    payload = {
        "validation_status": "graphene_style_scaffold_only_not_paper_evidence",
        "template": {
            "size": template.size,
            "query_id": template.query_id,
            "num_nodes": template.num_nodes,
            "num_edges": template.num_edges,
            "total_work": template.total_work,
        },
        "first_nodes": list(schedule.node_order[: min(10, len(schedule.node_order))]),
        "score_count": len(schedule.scores),
    }
    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name="decima-graphene-smoke",
        role="smoke",
        params={
            "method": "decima",
            "component": "graphene_style_comparator_scaffold",
            "decima_repo_url": DECIMA_REPO_URL,
            "decima_commit": DECIMA_COMMIT,
        },
    ) as run:
        mlflow.log_metric("graphene_smoke_num_nodes", template.num_nodes)
        mlflow.log_metric("graphene_smoke_total_work", template.total_work)
        out = root / "results" / "decima" / "graphene_smoke.json"
        write_json_artifact(out, payload, run_id=run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_smoke(args: argparse.Namespace, root: Path) -> None:
    templates = load_tpch_templates(root / args.tpch_root)
    sampled = sample_templates(templates, count=8, w=args.w, seed=args.seed)
    template = sampled[0]
    torch.manual_seed(args.seed)
    policy = DecimaPolicy()
    node_features = torch.rand(template.num_nodes, DEFAULT_CONFIG.node_input_dim)
    adjacency = torch.as_tensor(template.adjacency, dtype=torch.float32)
    probs = policy(node_features, adjacency)
    rollout = rollout_smoke(policy, sampled, count=3)
    probabilities = sampling_probabilities(templates, w=args.w)
    payload = {
        "tpch_summary": template_summary(templates),
        "sampled_first": {
            "size": template.size,
            "query_id": template.query_id,
            "num_nodes": template.num_nodes,
            "num_edges": template.num_edges,
            "depth": template.depth,
        },
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "policy_prob_sum": float(probs.sum().item()),
        "policy_argmax": int(torch.argmax(probs).item()),
        "policy_parameter_count": int(parameter_count(policy)),
        "rollout_smoke_steps": len(rollout.steps),
        "rollout_smoke_total_reward": rollout.total_reward,
    }
    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name="decima-smoke",
        role="smoke",
        params={
            "method": "decima",
            "decima_repo_url": DECIMA_REPO_URL,
            "decima_commit": DECIMA_COMMIT,
            "exec_cap": DEFAULT_CONFIG.exec_cap,
            "w": args.w,
            "seed": args.seed,
        },
    ) as run:
        mlflow.log_metric("tpch_template_count", payload["tpch_summary"]["count"])
        mlflow.log_metric("policy_prob_sum", payload["policy_prob_sum"])
        mlflow.log_metric("policy_parameter_count", payload["policy_parameter_count"])
        mlflow.log_metric("rollout_smoke_total_reward", payload["rollout_smoke_total_reward"])
        out = root / "results" / "decima" / "smoke.json"
        write_json_artifact(out, payload, run_id=run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_official_smoke(args: argparse.Namespace, root: Path) -> None:
    torch.manual_seed(args.seed)
    policy = DecimaPolicy()
    results = [
        run_official_episode(
            root,
            scheme="dynamic_partition",
            seed=args.seed,
            num_stream_dags=args.num_stream_dags,
        ),
        run_official_episode(
            root,
            scheme="pytorch_decima",
            seed=args.seed,
            num_stream_dags=args.num_stream_dags,
            policy=policy,
            action_mode=args.action_mode,
            device=args.device,
        ),
    ]
    payload = {
        "scope": "official_simulator_smoke_not_reproduction_gate",
        "official_readme_scale": False,
        "executor_action_levels": list(policy.executor_levels),
        "num_executor_action_levels": len(policy.executor_levels),
        "results": [asdict(result) for result in results],
    }
    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name="decima-official-simulator-smoke",
        role="smoke",
        params={
            "method": "decima",
            "component": "official_simulator_adapter",
            "decima_repo_url": DECIMA_REPO_URL,
            "decima_commit": DECIMA_COMMIT,
            "seed": args.seed,
            "num_stream_dags": args.num_stream_dags,
            "action_mode": args.action_mode,
            "device": args.device,
        },
    ) as run:
        for result in results:
            mlflow.log_metric(f"{result.scheme}_total_reward", result.total_reward)
            mlflow.log_metric(f"{result.scheme}_mean_jct", result.mean_job_completion_time)
            mlflow.log_metric(f"{result.scheme}_num_finished_jobs", result.num_finished_jobs)
            mlflow.log_metric(f"{result.scheme}_decisions", result.decisions)
        out = root / "results" / "decima" / "official_smoke.json"
        write_json_artifact(out, payload, run_id=run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_official_train_smoke(args: argparse.Namespace, root: Path) -> None:
    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name="decima-official-training-smoke",
        role="smoke",
        params={
            "method": "decima",
            "component": "pytorch_official_training_loop",
            "decima_repo_url": DECIMA_REPO_URL,
            "decima_commit": DECIMA_COMMIT,
            "seed": args.seed,
            "epochs": args.epochs,
            "num_agents": args.num_agents,
            "num_stream_dags": args.num_stream_dags,
            "device": args.device,
        },
    ) as run:
        result = train_pytorch_official(
            root,
            epochs=args.epochs,
            num_agents=args.num_agents,
            num_stream_dags=args.num_stream_dags,
            checkpoint_dir=root / args.checkpoint_dir,
            seed=args.seed,
            device=args.device,
            checkpoint_interval=max(args.epochs, 1),
        )
        payload = {
            "scope": "official_training_smoke_not_reproduction_gate",
            "official_readme_scale": False,
            "result": asdict(result),
        }
        mlflow.log_metric("final_epoch", result.final_epoch)
        mlflow.log_metric("mean_total_reward", result.mean_total_reward)
        mlflow.log_metric("mean_job_completion_time", result.mean_job_completion_time)
        mlflow.log_artifact(str(result.checkpoint_path), artifact_path="checkpoints")
        out = root / "results" / "decima" / "official_train_smoke.json"
        write_json_artifact(out, payload, run_id=run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "protocol":
        cmd_protocol()
    elif args.command == "readiness":
        cmd_readiness()
    elif args.command == "preflight":
        print(json.dumps(decima_preflight_report(root), indent=2, sort_keys=True))
    elif args.command == "tpch-summary":
        cmd_tpch_summary(args, root)
    elif args.command == "graphene-smoke":
        cmd_graphene_smoke(args, root)
    elif args.command == "reference-commands":
        print(json.dumps(reference_command_payload(), indent=2, sort_keys=True))
    elif args.command == "smoke-run":
        cmd_smoke(args, root)
    elif args.command == "official-smoke":
        cmd_official_smoke(args, root)
    elif args.command == "official-train-smoke":
        cmd_official_train_smoke(args, root)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
