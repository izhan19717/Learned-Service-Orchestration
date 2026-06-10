#!/usr/bin/env python3
"""Run Decima SRTF-style comparator sensitivity in the TF1 Docker image."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_common.tracking import start_run
from cisose_decima.config import DECIMA_COMMIT, DECIMA_REPO_URL, DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_decima_v2_2"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "decima_srtf_comparator"
TABLE_DIR = OUT_DIR / "tables"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
ROOT_FIG_DIR = ROOT / "figures"
ROOT_DATA_DIR = ROOT / "data"
BOOTSTRAP_SEED = 20260602


@dataclass(frozen=True)
class Cell:
    key: str
    label: str
    perturbation: str
    tail_weight: float = 0.5
    lag_lambda: float = 1.0
    fgsm_epsilon: float = 0.05


CELLS = (
    Cell("clean", "Clean", "clean", tail_weight=0.0, lag_lambda=0.0, fgsm_epsilon=0.0),
    Cell("p1_lag_lambda_1_0", "P1 lag lambda=1.0", "lag"),
    Cell("p2_tail_w_0_5", "P2 tail w=0.5", "tail"),
    Cell("p3_fgsm_epsilon_0_05", "P3 FGSM epsilon=0.05", "fgsm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="cisose-decima-tf1:1.15.5")
    parser.add_argument("--num-exp", type=int, default=30)
    parser.add_argument("--num-stream-dags", type=int, default=200)
    parser.add_argument("--saved-model", type=Path, default=Path("results/checkpoints/decima/official_tf1_readme/model_ep_10000"))
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--force-remove-container", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    docker_bin = resolve_docker(args.docker_bin)
    saved_model = (ROOT / args.saved_model).resolve()
    if not saved_model.with_suffix(".index").exists():
        raise FileNotFoundError(saved_model.with_suffix(".index"))

    payload = {
        "status": "running",
        "experiment": "decima_srtf_style_comparator_sensitivity",
        "scope": "post-hoc sensitivity, not official Graphene evidence and not a replacement for dynamic_partition results",
        "method": "Decima",
        "comparator": "SRTF-style dependency-aware shortest remaining work",
        "delta_definition": "mean_JCT(SRTF)-mean_JCT(Decima)",
        "delta_interpretation": "negative means SRTF beats Decima; positive means Decima beats SRTF",
        "docker_image": args.image,
        "docker_bin": str(docker_bin),
        "saved_model": str(args.saved_model),
        "num_exp": args.num_exp,
        "num_stream_dags": args.num_stream_dags,
        "exec_cap": DEFAULT_CONFIG.exec_cap,
        "num_init_dags": DEFAULT_CONFIG.num_init_dags,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "decima_repo_url": DECIMA_REPO_URL,
        "decima_commit": DECIMA_COMMIT,
        "cells": {},
    }

    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="decima-srtf-style-comparator-sensitivity",
        role="post_hoc_sensitivity",
        params={
            "method": "decima",
            "experiment": "srtf_style_comparator_sensitivity",
            "scope": "post_hoc_sensitivity_not_graphene",
            "docker_image": args.image,
            "saved_model": str(args.saved_model),
            "num_exp": args.num_exp,
            "num_stream_dags": args.num_stream_dags,
            "exec_cap": DEFAULT_CONFIG.exec_cap,
            "num_init_dags": DEFAULT_CONFIG.num_init_dags,
        },
        tags={"method": "decima", "sensitivity": "srtf_style_comparator"},
    ) as run:
        payload["mlflow_run_id"] = run.info.run_id
        for idx, cell in enumerate(CELLS):
            print(f"Running Decima SRTF cell {cell.label}", flush=True)
            cell_payload = run_cell(args, docker_bin=docker_bin, saved_model=saved_model, cell=cell, index=idx)
            payload["cells"][cell.key] = cell_payload
            aggregate = cell_payload["aggregate"]
            metric_key = cell.key.replace(".", "_")
            for key, value in aggregate.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(f"{metric_key}.{key}", float(value))
            mlflow.log_artifact(str(Path(cell_payload["output_json"])), artifact_path="results/cells")
            mlflow.log_artifact(str(Path(cell_payload["output_csv"])), artifact_path="results/cells")
            mlflow.log_artifact(str(Path(cell_payload["log_file"])), artifact_path="logs")

        payload["status"] = "completed"
        payload["summary"] = summarize(payload)
        paths = write_outputs(payload)
        for path in paths + [Path(__file__), ROOT / "scripts" / "decima_tf1_srtf_eval.py"]:
            mlflow.log_artifact(str(path), artifact_path=artifact_group(path))
        print(f"MLflow run: {run.info.run_id}")
        print(str((OUT_DIR / "decima_srtf_comparator_results.md").relative_to(ROOT)))
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def run_cell(args: argparse.Namespace, *, docker_bin: Path, saved_model: Path, cell: Cell, index: int) -> dict[str, object]:
    output_json = TABLE_DIR / f"decima_srtf_{cell.key}.json"
    output_csv = DATA_DIR / f"decima_srtf_{cell.key}_raw.csv"
    log_file = ROOT / "logs" / "training" / f"decima_srtf_{cell.key}.log"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    container_name = f"cisose_decima_srtf_{cell.key}".replace("_", "-")[:60]
    if args.force_remove_container:
        subprocess.run([str(docker_bin), "rm", "-f", container_name], check=False, capture_output=True)
    command = docker_command(
        args,
        docker_bin=docker_bin,
        saved_model=saved_model,
        cell=cell,
        output_json=output_json,
        output_csv=output_csv,
        container_name=container_name,
        bootstrap_seed=BOOTSTRAP_SEED + index,
    )
    launch = {
        "cell": cell.key,
        "label": cell.label,
        "command": command,
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "log_file": str(log_file),
    }
    started = time.time()
    with log_file.open("ab") as log:
        log.write((json.dumps(launch, sort_keys=True) + "\n").encode("utf-8"))
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Decima SRTF cell failed: {cell.key}; see {log_file}")
    metrics = json.loads(output_json.read_text(encoding="utf-8"))
    return {
        **launch,
        "status": "completed",
        "exit_code": code,
        "elapsed_seconds_wrapper": time.time() - started,
        "aggregate": metrics["aggregate"],
        "args": metrics["args"],
        "perturbation_metadata": metrics.get("perturbation_metadata", {}),
        "paired": metrics["paired"],
    }


def docker_command(
    args: argparse.Namespace,
    *,
    docker_bin: Path,
    saved_model: Path,
    cell: Cell,
    output_json: Path,
    output_csv: Path,
    container_name: str,
    bootstrap_seed: int,
) -> list[str]:
    uid = os.getuid()
    gid = os.getgid()
    output_json_arg = "/workspace/" + str(output_json.resolve().relative_to(ROOT))
    output_csv_arg = "/workspace/" + str(output_csv.resolve().relative_to(ROOT))
    saved_model_arg = "/workspace/" + str(saved_model.relative_to(ROOT))
    eval_args = [
        "python",
        "-u",
        "/workspace/scripts/decima_tf1_srtf_eval.py",
        "--exec_cap",
        str(DEFAULT_CONFIG.exec_cap),
        "--num_init_dags",
        str(DEFAULT_CONFIG.num_init_dags),
        "--num_stream_dags",
        str(args.num_stream_dags),
        "--test_schemes",
        "srtf",
        "learn",
        "--num_exp",
        str(args.num_exp),
        "--saved_model",
        saved_model_arg,
    ]
    return [
        str(docker_bin),
        "run",
        "--rm",
        "--name",
        container_name,
        "--user",
        f"{uid}:{gid}",
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "-e",
        f"DECIMA_PERTURBATION={cell.perturbation}",
        "-e",
        f"DECIMA_TAIL_WEIGHT={cell.tail_weight}",
        "-e",
        f"DECIMA_LAG_LAMBDA={cell.lag_lambda}",
        "-e",
        f"DECIMA_FGSM_EPSILON={cell.fgsm_epsilon}",
        "-e",
        f"DECIMA_BOOTSTRAP_SEED={bootstrap_seed}",
        "-e",
        f"DECIMA_SRTF_OUTPUT_JSON={output_json_arg}",
        "-e",
        f"DECIMA_SRTF_OUTPUT_CSV={output_csv_arg}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace/external/decima-sim",
        args.image,
        "bash",
        "-lc",
        " ".join(eval_args),
    ]


def summarize(payload: dict[str, object]) -> dict[str, object]:
    rows = []
    for cell in CELLS:
        aggregate = payload["cells"][cell.key]["aggregate"]
        rows.append(
            {
                "cell": cell.key,
                "label": cell.label,
                "srtf_mean_jct": aggregate["srtf_mean_jct"],
                "learn_mean_jct": aggregate["learn_mean_jct"],
                "delta_mean": aggregate["delta_mean"],
                "delta_ci_low": aggregate["delta_ci_low"],
                "delta_ci_high": aggregate["delta_ci_high"],
                "srtf_beats_decima": aggregate["srtf_beats_decima"],
                "decima_beats_srtf": aggregate["decima_beats_srtf"],
            }
        )
    return {
        "scope": "post_hoc_sensitivity_not_graphene",
        "rows": rows,
        "srtf_beats_decima_cells": [row["cell"] for row in rows if row["srtf_beats_decima"]],
        "decima_beats_srtf_cells": [row["cell"] for row in rows if row["decima_beats_srtf"]],
    }


def write_outputs(payload: dict[str, object]) -> list[Path]:
    json_path = OUT_DIR / "decima_srtf_comparator_sensitivity.json"
    report = OUT_DIR / "decima_srtf_comparator_results.md"
    root_report = ROOT / "decima_srtf_comparator_results.md"
    summary_csv = TABLE_DIR / "decima_srtf_comparator_summary.csv"
    paired_csv = DATA_DIR / "decima_srtf_comparator_paired.csv"
    root_paired_csv = ROOT_DATA_DIR / "decima_srtf_comparator_paired.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(summary_csv, payload["summary"]["rows"])
    paired_rows = paired_output_rows(payload)
    write_csv(paired_csv, paired_rows)
    write_csv(root_paired_csv, paired_rows)
    figures = write_figures(payload)
    text = report_text(payload, figures)
    report.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    return [json_path, report, root_report, summary_csv, paired_csv, root_paired_csv, *figures]


def paired_output_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for cell in CELLS:
        for pair in payload["cells"][cell.key]["paired"]:
            rows.append({"cell": cell.key, "label": cell.label, **pair})
    return rows


def write_figures(payload: dict[str, object]) -> list[Path]:
    rows = payload["summary"]["rows"]
    labels = [row["label"].replace(" ", "\n") for row in rows]
    means = np.asarray([row["delta_mean"] for row in rows], dtype=np.float64)
    lows = np.asarray([row["delta_ci_low"] for row in rows], dtype=np.float64)
    highs = np.asarray([row["delta_ci_high"] for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = ["#4C78A8" if mean >= 0 else "#F58518" for mean in means]
    ax.bar(x, means, color=colors, width=0.62)
    ax.errorbar(x, means, yerr=np.vstack([means - lows, highs - means]), fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Delta mean JCT: SRTF - Decima")
    ax.set_title("Decima versus SRTF-style comparator sensitivity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    pdf = FIG_DIR / "decima_srtf_delta_by_cell.pdf"
    png = FIG_DIR / "decima_srtf_delta_by_cell.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    for path in (pdf, png):
        shutil.copyfile(path, ROOT_FIG_DIR / path.name)
    return [pdf, png, ROOT_FIG_DIR / pdf.name, ROOT_FIG_DIR / png.name]


def report_text(payload: dict[str, object], figures: list[Path]) -> str:
    lines = [
        "# Decima SRTF-Style Comparator Sensitivity",
        "",
        f"MLflow run: `{payload.get('mlflow_run_id', 'pending')}`",
        "",
        "Scope: post-hoc sensitivity analysis. This is not Graphene evidence and does not replace the amended official `dynamic_partition` Decima results.",
        "",
        "Comparator: dependency-aware SRTF-style shortest remaining work. It chooses the arrived job DAG with the smallest estimated remaining work, then the ready node with the smallest estimated remaining node work, and allocates currently available source executors work-conservatively.",
        "",
        "Delta is `mean_JCT(SRTF) - mean_JCT(Decima)`. Negative values favor SRTF; positive values favor Decima.",
        "",
        "## Results",
        "",
        "| Cell | SRTF mean JCT | Decima mean JCT | Delta | 95% CI | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["summary"]["rows"]:
        if row["srtf_beats_decima"]:
            verdict = "SRTF beats Decima"
        elif row["decima_beats_srtf"]:
            verdict = "Decima beats SRTF"
        else:
            verdict = "Inconclusive"
        lines.append(
            f"| {row['label']} | {row['srtf_mean_jct']:.6g} | {row['learn_mean_jct']:.6g} | "
            f"{row['delta_mean']:+.6g} | [{row['delta_ci_low']:+.6g}, {row['delta_ci_high']:+.6g}] | {verdict} |"
        )
    lines.extend(["", "## Figures", ""])
    for path in figures:
        if path.parent == FIG_DIR:
            lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## Scientific Reading",
            "",
            "This analysis answers whether the Decima conclusions are sensitive to replacing the README-exposed `dynamic_partition` comparator with a simple shortest-remaining-work classical scheduler. Because SRTF is reconstructed locally and is not an official Decima README comparator, it should be reported only as an alternative-comparator sensitivity.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_docker(requested: str | None) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested))
    path_docker = shutil.which("docker")
    if path_docker:
        candidates.append(Path(path_docker))
    candidates.append(Path("/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"))
    for candidate in candidates:
        if candidate.exists() or str(candidate) == "docker":
            result = subprocess.run([str(candidate), "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True)
            if result.returncode == 0:
                return candidate
    raise RuntimeError("Docker daemon is not reachable; start Docker Desktop and rerun.")


def artifact_group(path: Path) -> str:
    if "figures" in path.parts:
        return "paper/figures"
    if "tables" in path.parts:
        return "paper/tables"
    if "data" in path.parts:
        return "paper/data"
    if path.suffix == ".md":
        return "paper/reports"
    if path.name.endswith(".py"):
        return "source"
    return "results"


def ensure_dirs() -> None:
    for directory in (OUT_DIR, TABLE_DIR, DATA_DIR, FIG_DIR, ROOT_FIG_DIR, ROOT_DATA_DIR, ROOT / "logs" / "training"):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
