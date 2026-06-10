#!/usr/bin/env python3
"""Run the Decima official README test gate inside the TF1 Docker image."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import mlflow

from cisose_common.tracking import start_run, write_json_artifact
from cisose_decima.config import DECIMA_COMMIT, DECIMA_REPO_URL, DEFAULT_CONFIG


EXPERIMENT_NAME = "cisose_decima_v2_2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="cisose-decima-tf1:1.15.5")
    parser.add_argument("--container-name", default="cisose_decima_tf1_readme_test_gate")
    parser.add_argument(
        "--saved-model",
        type=Path,
        default=Path("results/checkpoints/decima/official_tf1_readme/model_ep_10000"),
    )
    parser.add_argument("--num-exp", type=int, default=1)
    parser.add_argument("--num-stream-dags", type=int, default=5000)
    parser.add_argument("--write-visualizations", action="store_true")
    parser.add_argument(
        "--result-folder",
        type=Path,
        default=Path("results/paper/decima/official_readme_test_raw"),
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("results/paper/decima/tables/decima_official_readme_gate.json"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("results/paper/decima/tables/decima_official_readme_gate_per_exp.csv"),
    )
    parser.add_argument(
        "--metrics-md",
        type=Path,
        default=Path("results/paper/decima/tables/decima_official_readme_gate.md"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/training/decima_tf1_readme_test_gate.log"),
    )
    parser.add_argument("--force-remove-container", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path.cwd()
    saved_model = (root / args.saved_model).resolve()
    if not saved_model.with_suffix(".index").exists():
        raise FileNotFoundError(saved_model.with_suffix(".index"))
    for path in (args.result_folder, args.metrics_json.parent, args.log_file.parent):
        (root / path).mkdir(parents=True, exist_ok=True)

    command = _docker_command(root, args, saved_model)
    if args.force_remove_container:
        subprocess.run(["docker", "rm", "-f", args.container_name], check=False, capture_output=True)

    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name="decima-official-tf1-readme-test-gate",
        role="evaluation",
        params={
            "method": "decima",
            "component": "official_tf1_readme_test_gate",
            "decima_repo_url": DECIMA_REPO_URL,
            "decima_commit": DECIMA_COMMIT,
            "docker_image": args.image,
            "container_name": args.container_name,
            "saved_model": str(args.saved_model),
            "exec_cap": DEFAULT_CONFIG.exec_cap,
            "num_init_dags": DEFAULT_CONFIG.num_init_dags,
            "num_stream_dags": args.num_stream_dags,
            "num_exp": args.num_exp,
            "test_schemes": "dynamic_partition,learn",
            "write_visualizations": args.write_visualizations,
        },
        tags={
            "decima.backend": "official_tf1_docker",
            "decima.reproduction_gate": "readme_test",
            "decima.perturbations_allowed": "false",
        },
    ) as run:
        launch_payload = {
            "mlflow_run_id": run.info.run_id,
            "scope": "official_tf1_readme_test_gate",
            "not_perturbation_result": True,
            "command": command,
            "saved_model": str(args.saved_model),
            "metrics_json": str((root / args.metrics_json).resolve()),
            "metrics_csv": str((root / args.metrics_csv).resolve()),
            "metrics_md": str((root / args.metrics_md).resolve()),
            "log_file": str((root / args.log_file).resolve()),
        }
        state_path = root / "results" / "decima" / "tf1_readme_test_gate_state.json"
        write_json_artifact(state_path, launch_payload, run_id=run.info.run_id)

        start = time.time()
        with (root / args.log_file).open("ab") as log:
            log.write((json.dumps(launch_payload, sort_keys=True) + "\n").encode("utf-8"))
            log.flush()
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            _write_state(
                state_path,
                launch_payload
                | {
                    "wrapper_pid": os.getpid(),
                    "docker_client_pid": process.pid,
                    "status": "running",
                },
            )
            code = process.wait()

        elapsed = time.time() - start
        status = {
            "status": "completed" if code == 0 else "failed",
            "exit_code": code,
            "elapsed_seconds": elapsed,
        }
        payload = launch_payload | status
        if code == 0:
            metrics = json.loads((root / args.metrics_json).read_text())
            payload["gate"] = metrics.get("gate", {})
            payload["aggregate"] = metrics.get("aggregate", {})
            _write_markdown(root / args.metrics_md, metrics, run.info.run_id)
            _log_metrics(metrics)
            mlflow.log_artifact(str(root / args.metrics_json), artifact_path="results")
            mlflow.log_artifact(str(root / args.metrics_csv), artifact_path="results")
            mlflow.log_artifact(str(root / args.metrics_md), artifact_path="results")
            mlflow.set_tag(
                "decima.readme_test_gate_passed",
                str(metrics.get("gate", {}).get("gate_passed", False)),
            )
        _write_state(state_path, payload)
        mlflow.log_metric("elapsed_seconds", elapsed)
        mlflow.log_artifact(str(root / args.log_file), artifact_path="logs")
        mlflow.log_artifact(str(state_path), artifact_path="results")
        if code != 0:
            raise RuntimeError(f"Decima README test gate failed: {payload}")


def _docker_command(root: Path, args: argparse.Namespace, saved_model: Path) -> list[str]:
    uid = os.getuid()
    gid = os.getgid()
    metrics_json = "/workspace/" + str((root / args.metrics_json).resolve().relative_to(root))
    metrics_csv = "/workspace/" + str((root / args.metrics_csv).resolve().relative_to(root))
    result_folder = "/workspace/" + str((root / args.result_folder).resolve().relative_to(root)) + "/"
    saved_model_arg = "/workspace/" + str(saved_model.relative_to(root))
    eval_args = [
        "python",
        "-u",
        "/workspace/scripts/decima_tf1_readme_test_eval.py",
        "--exec_cap",
        str(DEFAULT_CONFIG.exec_cap),
        "--num_init_dags",
        str(DEFAULT_CONFIG.num_init_dags),
        "--num_stream_dags",
        str(args.num_stream_dags),
        "--canvs_visualization",
        "0",
        "--test_schemes",
        "dynamic_partition",
        "learn",
        "--num_exp",
        str(args.num_exp),
        "--saved_model",
        saved_model_arg,
        "--result_folder",
        result_folder,
    ]
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        args.container_name,
        "--user",
        f"{uid}:{gid}",
        "-e",
        f"DECIMA_METRICS_OUTPUT={metrics_json}",
        "-e",
        f"DECIMA_METRICS_CSV={metrics_csv}",
        "-e",
        f"DECIMA_WRITE_VISUALIZATIONS={1 if args.write_visualizations else 0}",
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "-v",
        f"{root}:/workspace",
        "-w",
        "/workspace/external/decima-sim",
        args.image,
        "bash",
        "-lc",
        " ".join(eval_args),
    ]


def _log_metrics(metrics: dict[str, object]) -> None:
    gate = metrics.get("gate", {})
    aggregate = metrics.get("aggregate", {})
    for scheme, scheme_metrics in aggregate.items():
        if isinstance(scheme_metrics, dict):
            for key, value in scheme_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{scheme}_{key}", value)
    if isinstance(gate, dict):
        for key, value in gate.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(f"gate_{key}", value)


def _write_markdown(path: Path, metrics: dict[str, object], run_id: str) -> None:
    aggregate = metrics.get("aggregate", {})
    gate = metrics.get("gate", {})
    rows = [
        "# Decima Official README Test Gate",
        "",
        f"MLflow run: `{run_id}`",
        "",
        f"Gate passed: `{gate.get('gate_passed')}`",
        "",
        "| Scheme | Mean JCT | Total reward | Jobs | Decision steps | Runtime seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scheme in ("dynamic_partition", "learn"):
        vals = aggregate.get(scheme, {})
        rows.append(
            "| {} | {:.6g} | {:.6g} | {:.6g} | {:.6g} | {:.6g} |".format(
                scheme,
                vals.get("mean_jct", float("nan")),
                vals.get("mean_total_reward", float("nan")),
                vals.get("mean_num_jobs", float("nan")),
                vals.get("mean_decision_steps", float("nan")),
                vals.get("mean_elapsed_seconds", float("nan")),
            )
        )
    rows.extend(
        [
            "",
            "## Gate",
            "",
            "| Metric | Value |",
            "|---|---:|",
            "| Observed mean-JCT improvement (%) | {:.6g} |".format(
                gate.get("observed_improvement_pct", float("nan"))
            ),
            "| Target improvement (%) | {:.6g} |".format(
                gate.get("target_improvement_pct", float("nan"))
            ),
            "| Relative error to target | {:.6g} |".format(
                gate.get("relative_error_to_target", float("nan"))
            ),
            "| Within 15% relative target tolerance | {} |".format(
                gate.get("within_15pct_relative_of_target")
            ),
            "| Learn beats dynamic partition | {} |".format(
                gate.get("learn_beats_dynamic_partition")
            ),
            "",
            "Note: visualization side effects from the stock README test are disabled by default here; the simulator loop, seeds, agents, saved model, and test scale match the README gate.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
