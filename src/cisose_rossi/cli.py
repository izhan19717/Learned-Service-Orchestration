"""CLI for Rossi/RLAD implementation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import mlflow

from cisose_common.tracking import sha256_file, start_run, write_json_artifact
from cisose_rossi.checkpointing import save_model_based_checkpoint
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import ModelBasedController, ThresholdHPAController
from cisose_rossi.evaluation import reproduction_gate_report, smoke_compare, table_i_metrics
from cisose_rossi.simulator import RladSimulator
from cisose_rossi.workload import java_slow_profile_sequence, load_profile, profile_summary


EXPERIMENT_NAME = "cisose_rossi_v2_2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cisose-rossi")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol", help="Print source-derived Rossi/RLAD protocol")
    smoke = sub.add_parser("smoke-run", help="Run a short MLflow-tracked Rossi smoke comparison")
    smoke.add_argument("--horizon", type=int, default=200)
    smoke.add_argument("--seed", type=int, default=20260515)
    smoke.add_argument(
        "--agent",
        choices=("model_based", "dynaq2"),
        default="model_based",
        help="Rossi agent to smoke-test; model_based is the paper-primary method",
    )
    smoke.add_argument(
        "--profile",
        type=Path,
        default=Path("external/rlad-core-simulator/data/profile.dat"),
    )
    reproduce = sub.add_parser(
        "reproduce-table-i",
        help="Run the Rossi Table I model-based reproduction gate",
    )
    reproduce.add_argument(
        "--profile",
        type=Path,
        default=Path("external/rlad-core-simulator/data/profile.dat"),
    )
    reproduce.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_CONFIG.time_limit + 1,
        help="Decision ticks to replay; Java default is time_limit + 1 = 4001",
    )
    reproduce.add_argument("--tolerance", type=float, default=0.15)
    reproduce.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/rossi/checkpoints/model_based_table_i_clean.npz"),
        help="Where to save the trained clean model-based controller",
    )
    return parser


def cmd_protocol() -> None:
    payload = {
        "repo_url": RLAD_REPO_URL,
        "commit": RLAD_COMMIT,
        "profile_sha256": PROFILE_SHA256,
        "primary_agent": "AGENT_RLMB",
        "simulator_default_agent": "AGENT_DYNAQ2",
        "config": DEFAULT_CONFIG.__dict__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_smoke(args: argparse.Namespace, root: Path) -> None:
    profile = load_profile(root / args.profile)
    sequence = java_slow_profile_sequence(profile, steps=args.horizon)
    params = {
        "method": "rossi_rlad",
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "agent_type": "AGENT_RLMB" if args.agent == "model_based" else "AGENT_DYNAQ2",
        "agent_role": "paper_primary" if args.agent == "model_based" else "simulator_default_sensitivity",
        "comparator": "source_threshold_hpa",
        "horizon": args.horizon,
        "seed": args.seed,
    }
    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name=f"rossi-smoke-{args.horizon}",
        role="smoke",
        params=params,
    ) as run:
        result = smoke_compare(sequence, seed=args.seed, horizon=args.horizon, agent_type=args.agent)
        result["profile_summary"] = profile_summary(profile)
        result["profile_sha256"] = PROFILE_SHA256
        mlflow.log_metric("delta_hpa_minus_rossi", float(result["delta_hpa_minus_rossi"]))
        mlflow.log_metric("rossi_total_cost", float(result["rossi"]["total_cost"]))
        mlflow.log_metric("hpa_total_cost", float(result["hpa"]["total_cost"]))
        out = root / "results" / "rossi" / f"smoke_h{args.horizon}.json"
        write_json_artifact(out, result, run_id=run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(result, indent=2, sort_keys=True))


def cmd_reproduce_table_i(args: argparse.Namespace, root: Path) -> None:
    profile = load_profile(root / args.profile)
    sequence = java_slow_profile_sequence(profile, steps=args.horizon)
    params = {
        "method": "rossi_rlad",
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "agent_type": "AGENT_RLMB",
        "agent_role": "paper_primary",
        "scaling_mode": "HORIZONTAL_OR_VERTICAL",
        "reproduction_gate": "Rossi 2019 Table I performance-weighted 5-action Model-based",
        "horizon": args.horizon,
        "tolerance": args.tolerance,
        "profile_sha256": PROFILE_SHA256,
        "w_resources": DEFAULT_CONFIG.w_resources,
        "w_reconfiguration": DEFAULT_CONFIG.w_reconfiguration,
        "w_sla": DEFAULT_CONFIG.w_sla,
        "gamma": DEFAULT_CONFIG.gamma,
        "alpha": DEFAULT_CONFIG.alpha,
    }
    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name=f"rossi-table-i-reproduction-h{args.horizon}",
        role="reproduction_gate",
        params=params,
    ) as run:
        controller = ModelBasedController(DEFAULT_CONFIG)
        records = RladSimulator(DEFAULT_CONFIG).run(controller, sequence, horizon=args.horizon)
        report = reproduction_gate_report(records, tolerance=args.tolerance)
        hpa_records = RladSimulator(DEFAULT_CONFIG).run(
            ThresholdHPAController(DEFAULT_CONFIG),
            sequence,
            horizon=args.horizon,
        )
        report["hpa_table_i_metrics"] = table_i_metrics(hpa_records).__dict__
        report["profile_summary"] = profile_summary(profile)
        report["profile_sha256"] = PROFILE_SHA256
        report["toolchain"] = {
            "maven_available": shutil.which("mvn") is not None,
            "javac_available": shutil.which("javac") is not None,
        }
        report["notes"] = [
            "Python port is source-aligned against the inspected Java simulator commit.",
            "No Java compiler/build tool was available in the current environment for direct Java execution.",
        ]
        checkpoint_path = root / args.checkpoint
        save_model_based_checkpoint(
            checkpoint_path,
            controller,
            metadata={
                "mlflow_run_id": run.info.run_id,
                "gate_name": report["gate_name"],
                "gate_passed": report["passed"],
                "horizon": args.horizon,
                "profile_sha256": PROFILE_SHA256,
                "rlad_commit": RLAD_COMMIT,
            },
        )
        report["checkpoint"] = {
            "path": str(checkpoint_path.relative_to(root)),
            "sha256": sha256_file(checkpoint_path),
            "frozen_for_perturbation_evaluation": True,
        }
        mlflow.log_metric("gate_passed", 1.0 if report["passed"] else 0.0)
        for row in report["rows"]:
            metric = row["metric"]
            mlflow.log_metric(f"{metric}_observed", float(row["observed"]))
            mlflow.log_metric(f"{metric}_target", float(row["target"]))
            mlflow.log_metric(f"{metric}_relative_error", float(row["relative_error"]))
            mlflow.log_metric(f"{metric}_within_15pct", 1.0 if row["within_15pct"] else 0.0)
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

        out = root / "results" / "rossi" / "reproduction_table_i_model_based.json"
        write_json_artifact(out, report, run_id=run.info.run_id)
        _write_reproduction_tables(root, report, run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(report, indent=2, sort_keys=True))


def _write_reproduction_tables(root: Path, report: dict[str, object], run_id: str) -> None:
    table_dir = root / "results" / "paper" / "rossi" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = list(report["rows"])
    csv_path = table_dir / "rossi_reproduction_table_i.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "observed", "target", "relative_error", "within_15pct"],
        )
        writer.writeheader()
        writer.writerows(rows)
    md_path = table_dir / "rossi_reproduction_table_i.md"
    lines = [
        "# Rossi Table I Reproduction Gate",
        "",
        f"MLflow run: `{run_id}`",
        "",
        f"Gate passed: `{report['passed']}`",
        "",
        "| Metric | Observed | Target | Relative error | Within 15% |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {observed:.6g} | {target:.6g} | {relative_error:.3%} | {within_15pct} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(csv_path), artifact_path="paper/tables")
    mlflow.log_artifact(str(md_path), artifact_path="paper/tables")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "protocol":
        cmd_protocol()
    elif args.command == "smoke-run":
        cmd_smoke(args, root)
    elif args.command == "reproduce-table-i":
        cmd_reproduce_table_i(args, root)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
