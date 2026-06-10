#!/usr/bin/env python3
"""Launch and monitor the official Decima TF1 README training in Docker."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
    parser.add_argument("--container-name", default="cisose_decima_tf1_model_ep_10000")
    parser.add_argument("--target-epoch", type=int, default=10_000)
    parser.add_argument("--resume-from-epoch", type=int, default=0)
    parser.add_argument("--saved-model", type=Path, default=None)
    parser.add_argument("--num-agents", type=int, default=16)
    parser.add_argument("--num-stream-dags", type=int, default=200)
    parser.add_argument("--model-save-interval", type=int, default=100)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("results/checkpoints/decima/official_tf1_readme"),
    )
    parser.add_argument(
        "--final-alias-folder",
        type=Path,
        default=None,
        help="Optional folder where the completed checkpoint is copied as model_ep_<target_epoch>.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/training/decima_tf1_model_ep_10000_docker.log"),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--force-remove-container", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path.cwd()
    model_folder = (root / args.model_folder).resolve()
    log_file = (root / args.log_file).resolve()
    model_folder.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    target_prefix = model_folder / f"model_ep_{args.target_epoch}"
    local_target_epoch = args.target_epoch - args.resume_from_epoch
    if args.resume_from_epoch < 0:
        raise ValueError("--resume-from-epoch must be non-negative")
    if local_target_epoch <= 0:
        raise ValueError("--target-epoch must be greater than --resume-from-epoch")
    if args.saved_model is not None and not (root / args.saved_model).with_suffix(".index").exists():
        raise FileNotFoundError(f"saved model checkpoint is incomplete: {root / args.saved_model}")
    resume_entropy_weight = max(0.0001, 1.0 - args.resume_from_epoch * 1e-3)
    resume_reset_prob = max(
        DEFAULT_CONFIG.reset_prob_min,
        DEFAULT_CONFIG.reset_prob - args.resume_from_epoch * DEFAULT_CONFIG.reset_prob_decay,
    )
    local_target_prefix = model_folder / f"model_ep_{local_target_epoch}"
    final_alias_prefix = (
        (root / args.final_alias_folder).resolve() / f"model_ep_{args.target_epoch}"
        if args.final_alias_folder is not None
        else None
    )
    command = _docker_command(root, args, model_folder)

    if args.force_remove_container:
        subprocess.run(["docker", "rm", "-f", args.container_name], check=False, capture_output=True)

    with start_run(
        root=root,
        experiment_name=EXPERIMENT_NAME,
        run_name=f"decima-official-tf1-model-ep-{args.target_epoch}-training",
        role="training",
        params={
            "method": "decima",
            "component": "official_tf1_author_training",
            "decima_repo_url": DECIMA_REPO_URL,
            "decima_commit": DECIMA_COMMIT,
            "docker_image": args.image,
            "container_name": args.container_name,
            "target_epoch": args.target_epoch,
            "resume_from_epoch": args.resume_from_epoch,
            "local_target_epoch": local_target_epoch,
            "num_ep_arg": local_target_epoch + 1,
            "saved_model": str(args.saved_model) if args.saved_model is not None else "",
            "exec_cap": DEFAULT_CONFIG.exec_cap,
            "num_init_dags": DEFAULT_CONFIG.num_init_dags,
            "num_stream_dags": args.num_stream_dags,
            "reset_prob": DEFAULT_CONFIG.reset_prob,
            "effective_reset_prob": resume_reset_prob,
            "reset_prob_min": DEFAULT_CONFIG.reset_prob_min,
            "reset_prob_decay": DEFAULT_CONFIG.reset_prob_decay,
            "effective_entropy_weight_init": resume_entropy_weight,
            "diff_reward_enabled": DEFAULT_CONFIG.diff_reward_enabled,
            "num_agents": args.num_agents,
            "model_save_interval": args.model_save_interval,
            "model_folder": str(model_folder),
        },
        tags={
            "decima.backend": "official_tf1_docker",
            "decima.reproduction_gate": "readme_training",
            "decima.perturbations_allowed": "false",
        },
    ) as run:
        launch_payload = {
            "mlflow_run_id": run.info.run_id,
            "scope": "official_tf1_readme_training",
            "not_perturbation_result": True,
            "command": command,
            "target_checkpoint_prefix": str(local_target_prefix),
            "cumulative_target_checkpoint_prefix": str(final_alias_prefix or target_prefix),
            "resume_from_epoch": args.resume_from_epoch,
            "saved_model": str(args.saved_model) if args.saved_model is not None else None,
            "effective_reset_prob": resume_reset_prob,
            "effective_entropy_weight_init": resume_entropy_weight,
            "final_alias_prefix": str(final_alias_prefix) if final_alias_prefix is not None else None,
            "log_file": str(log_file),
            "watchdog_behavior": "stop_container_after_target_checkpoint_appears",
        }
        state_path = root / "results" / "decima" / "tf1_reproduction_launch_state.json"
        write_json_artifact(state_path, launch_payload, run_id=run.info.run_id)

        with log_file.open("ab") as log:
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
            mlflow.log_param("wrapper_pid", os.getpid())
            mlflow.log_param("docker_client_pid", process.pid)
            status = _monitor(
                process,
                args.container_name,
                local_target_prefix,
                log_file,
                poll_seconds=args.poll_seconds,
            )
            if status["status"] == "completed" and final_alias_prefix is not None:
                _copy_checkpoint_alias(local_target_prefix, final_alias_prefix)
                status["final_alias_prefix"] = str(final_alias_prefix)
                status["final_alias_index_exists"] = final_alias_prefix.with_suffix(".index").exists()
        _write_state(state_path, launch_payload | status)
        mlflow.set_tag("decima.tf1_training_status", status["status"])
        for key, value in status.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        mlflow.log_artifact(str(log_file), artifact_path="logs")
        mlflow.log_artifact(str(state_path), artifact_path="results")
        if status["status"] != "completed":
            raise RuntimeError(f"Decima TF1 training did not complete: {status}")
        for path in _checkpoint_files(local_target_prefix):
            mlflow.log_artifact(str(path), artifact_path=f"checkpoints/model_ep_{local_target_epoch}")
        if final_alias_prefix is not None:
            for path in _checkpoint_files(final_alias_prefix):
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path=f"checkpoints/model_ep_{args.target_epoch}_alias")


def _docker_command(root: Path, args: argparse.Namespace, model_folder: Path) -> list[str]:
    workspace_model_folder = "/workspace/" + str(model_folder.relative_to(root))
    local_target_epoch = args.target_epoch - args.resume_from_epoch
    resume_entropy_weight = max(0.0001, 1.0 - args.resume_from_epoch * 1e-3)
    resume_reset_prob = max(
        DEFAULT_CONFIG.reset_prob_min,
        DEFAULT_CONFIG.reset_prob - args.resume_from_epoch * DEFAULT_CONFIG.reset_prob_decay,
    )
    train_args = [
        "python",
        "-u",
        "train.py",
        "--exec_cap",
        str(DEFAULT_CONFIG.exec_cap),
        "--num_init_dags",
        str(DEFAULT_CONFIG.num_init_dags),
        "--num_stream_dags",
        str(args.num_stream_dags),
        "--reset_prob",
        str(resume_reset_prob),
        "--reset_prob_min",
        str(DEFAULT_CONFIG.reset_prob_min),
        "--reset_prob_decay",
        str(DEFAULT_CONFIG.reset_prob_decay),
        "--entropy_weight_init",
        str(resume_entropy_weight),
        "--diff_reward_enabled",
        str(DEFAULT_CONFIG.diff_reward_enabled),
        "--num_agents",
        str(args.num_agents),
        "--model_save_interval",
        str(args.model_save_interval),
        "--num_ep",
        str(local_target_epoch + 1),
        "--model_folder",
        workspace_model_folder + "/",
    ]
    if args.saved_model is not None:
        train_args.extend(["--saved_model", "/workspace/" + str((root / args.saved_model).resolve().relative_to(root))])
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        args.container_name,
        "-v",
        f"{root}:/workspace",
        "-w",
        "/workspace/external/decima-sim",
        args.image,
        "bash",
        "-lc",
        " ".join(train_args),
    ]


def _monitor(
    process: subprocess.Popen[bytes],
    container_name: str,
    target_prefix: Path,
    log_file: Path,
    *,
    poll_seconds: float,
) -> dict[str, object]:
    start = time.time()
    last_offset = 0
    latest_epoch = 0
    last_metrics: dict[str, float] = {}
    while True:
        latest_epoch, last_metrics, last_offset = _parse_new_log_lines(
            log_file,
            last_offset=last_offset,
            latest_epoch=latest_epoch,
            last_metrics=last_metrics,
        )
        mlflow.log_metric("latest_epoch", latest_epoch)
        for key, value in last_metrics.items():
            mlflow.log_metric(key, value)

        if _checkpoint_complete(target_prefix):
            subprocess.run(["docker", "stop", container_name], check=False, capture_output=True)
            process.wait(timeout=120)
            return {
                "status": "completed",
                "latest_epoch": latest_epoch,
                "elapsed_seconds": time.time() - start,
                "target_checkpoint_index_exists": (target_prefix.with_suffix(".index")).exists(),
            }

        code = process.poll()
        if code is not None:
            return {
                "status": "failed",
                "exit_code": code,
                "latest_epoch": latest_epoch,
                "elapsed_seconds": time.time() - start,
                "target_checkpoint_index_exists": (target_prefix.with_suffix(".index")).exists(),
            }
        time.sleep(poll_seconds)


def _parse_new_log_lines(
    log_file: Path,
    *,
    last_offset: int,
    latest_epoch: int,
    last_metrics: dict[str, float],
) -> tuple[int, dict[str, float], int]:
    if not log_file.exists():
        return latest_epoch, last_metrics, last_offset
    with log_file.open("rb") as f:
        f.seek(last_offset)
        data = f.read()
        offset = f.tell()
    for raw in data.decode("utf-8", errors="replace").splitlines():
        epoch = re.search(r"training epoch\s+(\d+)", raw)
        if epoch:
            latest_epoch = int(epoch.group(1))
        timing = re.search(r"^(got reward from workers|advantage ready|worker send back gradients|apply gradient)\s+([0-9.]+)", raw)
        if timing:
            metric = timing.group(1).replace(" ", "_")
            last_metrics[f"last_{metric}_seconds"] = float(timing.group(2))
    return latest_epoch, last_metrics, offset


def _checkpoint_files(prefix: Path) -> list[Path]:
    return [
        prefix.with_suffix(".index"),
        prefix.with_suffix(".meta"),
        prefix.with_name(prefix.name + ".data-00000-of-00001"),
        prefix.parent / "checkpoint",
    ]


def _checkpoint_complete(prefix: Path) -> bool:
    files = _checkpoint_files(prefix)
    return all(path.exists() and path.stat().st_size > 0 for path in files[:3])


def _copy_checkpoint_alias(source_prefix: Path, dest_prefix: Path) -> None:
    dest_prefix.parent.mkdir(parents=True, exist_ok=True)
    mapping = [
        (source_prefix.with_suffix(".index"), dest_prefix.with_suffix(".index")),
        (source_prefix.with_suffix(".meta"), dest_prefix.with_suffix(".meta")),
        (
            source_prefix.with_name(source_prefix.name + ".data-00000-of-00001"),
            dest_prefix.with_name(dest_prefix.name + ".data-00000-of-00001"),
        ),
    ]
    for src, dst in mapping:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
    checkpoint = dest_prefix.parent / "checkpoint"
    checkpoint.write_text(
        'model_checkpoint_path: "{}"\nall_model_checkpoint_paths: "{}"\n'.format(
            dest_prefix.name,
            dest_prefix.name,
        )
    )


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
