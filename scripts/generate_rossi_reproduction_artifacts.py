#!/usr/bin/env python3
"""Generate paper artifacts for the Rossi Table I reproduction gate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_common.tracking import start_run


EXPERIMENT_NAME = "cisose_rossi_v2_2"
ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "rossi" / "reproduction_table_i_model_based.json"


LABELS = {
    "rmax_violations_pct": "Rmax violations",
    "avg_cpu_utilization_pct": "CPU utilization",
    "avg_cpu_share_pct": "CPU share",
    "avg_containers": "Containers",
    "median_response_ms": "Median response",
    "adaptations_pct": "Adaptations",
}


def main() -> None:
    report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    rows = report["rows"]
    errors_pct = np.asarray([100.0 * row["relative_error"] for row in rows], dtype=float)
    labels = [LABELS[row["metric"]] for row in rows]
    observed = [float(row["observed"]) for row in rows]
    targets = [float(row["target"]) for row in rows]

    fig_dir = ROOT / "results" / "paper" / "rossi" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / "rossi_reproduction_gate.pdf"
    png_path = fig_dir / "rossi_reproduction_gate.png"

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    order = np.argsort(errors_pct)[::-1]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.axvline(15.0, color="#8f2d2d", linestyle="--", linewidth=1.0)
    ax.text(
        15.0,
        -0.7,
        "15% gate",
        color="#8f2d2d",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    ax.hlines(y, 0.0003, errors_pct[order], color="#b8c4cc", linewidth=1.0)
    ax.scatter(errors_pct[order], y, color="#2f6278", s=30, zorder=3)
    for idx, row_idx in enumerate(order):
        ax.text(
            errors_pct[row_idx] * 1.28,
            idx,
            f"{errors_pct[row_idx]:.3f}%",
            va="center",
            fontsize=7,
            color="#333333",
        )
    ax.set_xscale("log")
    ax.set_xlim(0.0003, 30.0)
    ax.set_ylim(-0.8, len(rows) - 0.35)
    ax.set_xlabel("Absolute relative error from Table I target (%)")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[idx] for idx in order])
    ax.set_title("Rossi Table I Reproduction Gate")
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.06,
        f"Gate passed: max error {np.max(errors_pct):.3f}%",
        transform=ax.transAxes,
        fontsize=7,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="rossi-reproduction-paper-artifacts",
        role="paper_artifacts",
        params={
            "source_run_id": report["mlflow_run_id"],
            "gate_passed": report["passed"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "max_relative_error_pct": float(np.max(errors_pct)),
            "gate_threshold_pct": 15.0,
        },
    ) as run:
        mlflow.log_metric("max_relative_error_pct", float(np.max(errors_pct)))
        mlflow.log_metric("gate_threshold_pct", 15.0)
        for label, error, obs, target in zip(labels, errors_pct, observed, targets, strict=True):
            key = label.lower().replace(" ", "_")
            mlflow.log_metric(f"{key}.relative_error_pct", float(error))
            mlflow.log_metric(f"{key}.observed", float(obs))
            mlflow.log_metric(f"{key}.target", float(target))
        mlflow.log_artifact(str(pdf_path), artifact_path="paper/figures")
        mlflow.log_artifact(str(png_path), artifact_path="paper/figures")
        mlflow.log_artifact(str(Path(__file__)), artifact_path="paper/scripts")
        mlflow.log_artifact(str(RESULT_PATH), artifact_path="results")
        print(f"MLflow run: {run.info.run_id}")
        print(str(pdf_path.relative_to(ROOT)))
        print(str(png_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
