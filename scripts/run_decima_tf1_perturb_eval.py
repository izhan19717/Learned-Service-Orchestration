#!/usr/bin/env python3
"""MLflow/Docker wrapper for amended Decima simulator perturbation cells."""

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
    parser.add_argument("--container-name", default="cisose_decima_tf1_p2_tail_w_0_5")
    parser.add_argument("--perturbation", choices=["tail", "lag", "fgsm"], default="tail")
    parser.add_argument("--tail-weight", type=float, default=0.5)
    parser.add_argument("--lag-lambda", type=float, default=1.0)
    parser.add_argument("--fgsm-epsilon", type=float, default=0.05)
    parser.add_argument("--num-exp", type=int, default=30)
    parser.add_argument("--num-stream-dags", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=99017)
    parser.add_argument(
        "--saved-model",
        type=Path,
        default=Path("results/checkpoints/decima/official_tf1_readme/model_ep_10000"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/paper/decima/tables/decima_p2_tail_w_0_5.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/paper/decima/tables/decima_p2_tail_w_0_5_raw.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/paper/decima/tables/decima_p2_tail_w_0_5.md"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/training/decima_tf1_p2_tail_w_0_5.log"),
    )
    parser.add_argument("--force-remove-container", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path.cwd()
    saved_model = (root / args.saved_model).resolve()
    if not saved_model.with_suffix(".index").exists():
        raise FileNotFoundError(saved_model.with_suffix(".index"))
    for path in (args.output_json.parent, args.log_file.parent):
        (root / path).mkdir(parents=True, exist_ok=True)

    command = _docker_command(root, args, saved_model)
    if args.force_remove_container:
        subprocess.run(["docker", "rm", "-f", args.container_name], check=False, capture_output=True)

    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name=f"decima-amended-{args.perturbation}-w-{args.tail_weight}",
        role="evaluation",
        params={
            "method": "decima",
            "component": "amended_tf1_official_simulator_perturbation",
            "protocol_amendment": "PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md",
            "decima_repo_url": DECIMA_REPO_URL,
            "decima_commit": DECIMA_COMMIT,
            "docker_image": args.image,
            "container_name": args.container_name,
            "perturbation": args.perturbation,
            "tail_weight": args.tail_weight,
            "lag_lambda": args.lag_lambda,
            "fgsm_epsilon": args.fgsm_epsilon,
            "saved_model": str(args.saved_model),
            "exec_cap": DEFAULT_CONFIG.exec_cap,
            "num_init_dags": DEFAULT_CONFIG.num_init_dags,
            "num_stream_dags": args.num_stream_dags,
            "num_exp": args.num_exp,
            "comparator": "dynamic_partition",
        },
        tags={
            "decima.backend": "official_tf1_docker",
            "decima.protocol": "amended_official_simulator_dynamic_partition",
            "decima.perturbation": args.perturbation,
        },
    ) as run:
        launch_payload = {
            "mlflow_run_id": run.info.run_id,
            "scope": "decima_amended_perturbation_eval",
            "command": command,
            "perturbation": args.perturbation,
            "tail_weight": args.tail_weight,
            "lag_lambda": args.lag_lambda,
            "fgsm_epsilon": args.fgsm_epsilon,
            "saved_model": str(args.saved_model),
            "output_json": str((root / args.output_json).resolve()),
            "output_csv": str((root / args.output_csv).resolve()),
            "output_md": str((root / args.output_md).resolve()),
            "log_file": str((root / args.log_file).resolve()),
        }
        state_path = root / "results" / "decima" / _state_file_name(args)
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

        status = {"status": "completed" if code == 0 else "failed", "exit_code": code, "elapsed_seconds": time.time() - start}
        payload = launch_payload | status
        if code == 0:
            metrics = json.loads((root / args.output_json).read_text())
            payload["aggregate"] = metrics.get("aggregate", {})
            _write_markdown(root / args.output_md, metrics, run.info.run_id)
            _log_metrics(metrics)
            for artifact in (args.output_json, args.output_csv, args.output_md):
                mlflow.log_artifact(str(root / artifact), artifact_path="results")
        _write_state(state_path, payload)
        mlflow.log_metric("elapsed_seconds", status["elapsed_seconds"])
        mlflow.log_artifact(str(root / args.log_file), artifact_path="logs")
        mlflow.log_artifact(str(state_path), artifact_path="results")
        if code != 0:
            raise RuntimeError(f"Decima perturbation eval failed: {payload}")


def _docker_command(root: Path, args: argparse.Namespace, saved_model: Path) -> list[str]:
    uid = os.getuid()
    gid = os.getgid()
    output_json = "/workspace/" + str((root / args.output_json).resolve().relative_to(root))
    output_csv = "/workspace/" + str((root / args.output_csv).resolve().relative_to(root))
    saved_model_arg = "/workspace/" + str(saved_model.relative_to(root))
    eval_args = [
        "python",
        "-u",
        "/workspace/scripts/decima_tf1_perturb_eval.py",
        "--exec_cap",
        str(DEFAULT_CONFIG.exec_cap),
        "--num_init_dags",
        str(DEFAULT_CONFIG.num_init_dags),
        "--num_stream_dags",
        str(args.num_stream_dags),
        "--test_schemes",
        "dynamic_partition",
        "learn",
        "--num_exp",
        str(args.num_exp),
        "--saved_model",
        saved_model_arg,
    ]
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        args.container_name,
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "-e",
        f"DECIMA_PERTURBATION={args.perturbation}",
        "-e",
        f"DECIMA_TAIL_WEIGHT={args.tail_weight}",
        "-e",
        f"DECIMA_LAG_LAMBDA={args.lag_lambda}",
        "-e",
        f"DECIMA_FGSM_EPSILON={args.fgsm_epsilon}",
        "-e",
        f"DECIMA_BOOTSTRAP_SEED={args.bootstrap_seed}",
        "-e",
        f"DECIMA_PERTURB_OUTPUT_JSON={output_json}",
        "-e",
        f"DECIMA_PERTURB_OUTPUT_CSV={output_csv}",
        "-v",
        f"{root}:/workspace",
        "-w",
        "/workspace/external/decima-sim",
        args.image,
        "bash",
        "-lc",
        " ".join(eval_args),
    ]
    if not _rootless_docker_socket():
        command[5:5] = ["--user", f"{uid}:{gid}"]
    return command


def _rootless_docker_socket() -> bool:
    docker_host = os.environ.get("DOCKER_HOST", "")
    return docker_host.startswith("unix:///run/user/")


def _log_metrics(metrics: dict[str, object]) -> None:
    aggregate = metrics.get("aggregate", {})
    if isinstance(aggregate, dict):
        for key, value in aggregate.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, value)


def _write_markdown(path: Path, metrics: dict[str, object], run_id: str) -> None:
    aggregate = metrics["aggregate"]
    rows = [
        _title(metrics),
        "",
        f"MLflow run: `{run_id}`",
        "",
        _magnitude_line(metrics),
        "",
        "Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| dynamic_partition mean JCT | {aggregate['dynamic_partition_mean_jct']:.6g} |",
        f"| Decima mean JCT | {aggregate['learn_mean_jct']:.6g} |",
        f"| Delta mean | {aggregate['delta_mean']:.6g} |",
        f"| 95% CI low | {aggregate['delta_ci_low']:.6g} |",
        f"| 95% CI high | {aggregate['delta_ci_high']:.6g} |",
        f"| p_less | {aggregate['p_less']:.6g} |",
        f"| p_greater | {aggregate['p_greater']:.6g} |",
        f"| Prediction confirmed | {aggregate['prediction_confirmed']} |",
        "",
        "Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def _title(metrics: dict[str, object]) -> str:
    perturbation = metrics.get("perturbation")
    if perturbation == "lag":
        return "# Decima P1 Observation Lag"
    if perturbation == "fgsm":
        return "# Decima P3 Adversarial Node Features"
    return "# Decima P2 Tail Shift"


def _magnitude_line(metrics: dict[str, object]) -> str:
    perturbation = metrics.get("perturbation")
    if perturbation == "lag":
        return f"Lag lambda: `{metrics['lag_lambda']}`"
    if perturbation == "fgsm":
        return f"FGSM epsilon: `{metrics['fgsm_epsilon']}`"
    return f"Tail weight: `{metrics['tail_weight']}`"


def _state_file_name(args: argparse.Namespace) -> str:
    if args.perturbation == "lag":
        return f"tf1_lag_lambda_{args.lag_lambda}_state.json"
    if args.perturbation == "fgsm":
        return f"tf1_fgsm_epsilon_{args.fgsm_epsilon}_state.json"
    return f"tf1_tail_w_{args.tail_weight}_state.json"


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
