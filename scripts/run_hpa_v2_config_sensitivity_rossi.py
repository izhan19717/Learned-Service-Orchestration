#!/usr/bin/env python3
"""Post-hoc Rossi HPA-v2 configuration sensitivity."""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_common.stats import paired_bootstrap_ci, sign_flip_pvalues
from cisose_common.tracking import start_run
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import HPAv2Controller
from cisose_rossi.evaluation import metrics
from cisose_rossi.perturbations import capped_pareto_cv2
from cisose_rossi.simulator import RladSimulator
from cisose_rossi.workload import java_slow_profile_sequence, load_profile
from run_experiment_b_hpa_v2_rossi import (
    BASE_SEED,
    DEFAULT_HORIZON,
    PROFILE_PATH,
)

try:
    from run_experiment_b_hpa_v2_rossi import load_locked_rossi_costs, profile_offsets
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Experiment B helpers are required for paired offsets and locked Rossi costs") from exc


EXPERIMENT_NAME = "cisose_rossi_v2_2"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "hpa_v2_config_sensitivity"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
ROOT_FIG_DIR = ROOT / "figures"
ROOT_DATA_DIR = ROOT / "data"
BOOTSTRAP_SEED = 20260602
BOOTSTRAP_RESAMPLES = 5000


@dataclass(frozen=True)
class ConfigSpec:
    key: str
    target_utilization: float
    scale_down_stabilization_seconds: int
    label: str


@dataclass(frozen=True)
class CellSpec:
    key: str
    label: str


CONFIGS = tuple(
    ConfigSpec(
        key=f"target_{int(target * 100)}_down_{down}",
        target_utilization=target,
        scale_down_stabilization_seconds=down,
        label=f"target={int(target * 100)}%, down={down}s",
    )
    for down in (300, 0)
    for target in (0.40, 0.50, 0.60, 0.70)
)

CELLS = (
    CellSpec("clean", "Clean"),
    CellSpec("p1", "P1 lag k=10"),
    CellSpec("p2", "P2 tail alpha=1.5"),
    CellSpec("p3", "P3 bucket-flip epsilon=0.05"),
)


def main() -> None:
    _ensure_dirs()
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    offsets = profile_offsets(len(sequence), horizon=DEFAULT_HORIZON, n=30, seed=BASE_SEED)
    locked_rossi = load_locked_rossi_costs(offsets)
    raw = run_grid(sequence=sequence, offsets=offsets, locked_rossi=locked_rossi)
    summaries = summarize(raw)
    payload = {
        "status": "completed",
        "experiment": "post_hoc_hpa_v2_config_sensitivity",
        "method": "Rossi",
        "scope": "post-hoc sensitivity, not replacement for Experiment B",
        "grid": [spec.__dict__ for spec in CONFIGS],
        "cells": [spec.__dict__ for spec in CELLS],
        "fixed_hpa_parameters": {
            "sync_period_seconds": 15,
            "tolerance": 0.10,
            "scale_up_stabilization_seconds": 0,
            "min_replicas": 1,
            "max_replicas": DEFAULT_CONFIG.max_replication,
        },
        "offsets": offsets,
        "horizon": DEFAULT_HORIZON,
        "base_seed": BASE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "profile_sha256": PROFILE_SHA256,
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "raw": raw,
        "summaries": summaries,
        "interpretation": interpret(summaries),
    }
    paths = write_outputs(payload)
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="hpa-v2-config-sensitivity-rossi",
        role="post_hoc_sensitivity",
        params={
            "method": "rossi",
            "experiment": "hpa_v2_config_sensitivity",
            "scope": "post_hoc_sensitivity_not_replacement",
            "targets": [0.40, 0.50, 0.60, 0.70],
            "scale_down_stabilization_seconds": [300, 0],
            "horizon": DEFAULT_HORIZON,
            "n_seeds": 30,
            "base_seed": BASE_SEED,
            "profile_sha256": PROFILE_SHA256,
        },
        tags={"method": "rossi", "sensitivity": "hpa_v2_config"},
    ) as run:
        payload["mlflow_run_id"] = run.info.run_id
        for row in summaries:
            key = f"{row['config_key']}.{row['cell']}"
            mlflow.log_metric(f"{key}.delta_hpa_minus_rossi", row["delta_mean"])
            mlflow.log_metric(f"{key}.ci_low", row["ci_low"])
            mlflow.log_metric(f"{key}.ci_high", row["ci_high"])
            mlflow.log_metric(f"{key}.hpa_mean_total_cost", row["hpa_mean_total_cost"])
            mlflow.log_metric(f"{key}.rossi_mean_total_cost", row["rossi_mean_total_cost"])
            mlflow.log_metric(f"{key}.hpa_wins", row["hpa_wins"])
        paths = write_outputs(payload)
        for path in paths + [Path(__file__)]:
            mlflow.log_artifact(str(path), artifact_path=artifact_group(path))
        print(f"MLflow run: {run.info.run_id}")
        print(str((OUT_DIR / "hpa_v2_config_sensitivity_results.md").relative_to(ROOT)))
        print(json.dumps(payload["interpretation"], indent=2, sort_keys=True))


def run_grid(*, sequence: tuple[float, ...], offsets: tuple[int, ...], locked_rossi: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    tasks = {}
    raw: list[dict[str, object]] = []
    max_workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for config in CONFIGS:
            for cell in CELLS:
                for seed_index, offset in enumerate(offsets):
                    rates = tuple(sequence[offset : offset + DEFAULT_HORIZON])
                    future = pool.submit(run_cell, config, cell, rates)
                    tasks[future] = (config, cell, seed_index, offset)
        total = len(tasks)
        for completed, future in enumerate(as_completed(tasks), start=1):
            config, cell, seed_index, offset = tasks[future]
            result = future.result()
            locked = locked_rossi[cell.key][seed_index]
            result.update(
                {
                    "config_key": config.key,
                    "config_label": config.label,
                    "target_utilization": config.target_utilization,
                    "scale_down_stabilization_seconds": config.scale_down_stabilization_seconds,
                    "cell": cell.key,
                    "cell_label": cell.label,
                    "seed_index": seed_index,
                    "offset": offset,
                    "rossi_total_cost": float(locked["rossi_total_cost"]),
                    "delta_hpa_minus_rossi": float(result["hpa_total_cost"]) - float(locked["rossi_total_cost"]),
                }
            )
            raw.append(result)
            if completed % 40 == 0 or completed == total:
                print(f"hpa_config_sensitivity {completed}/{total}", flush=True)
    raw.sort(key=lambda row: (str(row["config_key"]), str(row["cell"]), int(row["seed_index"])))
    return raw


def run_cell(config: ConfigSpec, cell: CellSpec, rates: tuple[float, ...]) -> dict[str, object]:
    controller = HPAv2Controller(
        DEFAULT_CONFIG,
        target_utilization=config.target_utilization,
        scale_down_stabilization_seconds=config.scale_down_stabilization_seconds,
    )
    if cell.key == "clean":
        records = RladSimulator(DEFAULT_CONFIG).run(controller, rates, horizon=len(rates))
    elif cell.key == "p1":
        records = RladSimulator(DEFAULT_CONFIG).run(controller, rates, horizon=len(rates), observation_lag_steps=10)
    elif cell.key == "p2":
        records = RladSimulator(DEFAULT_CONFIG, service_time_cv2=capped_pareto_cv2(1.5)).run(
            controller,
            rates,
            horizon=len(rates),
        )
    elif cell.key == "p3":
        records = RladSimulator(DEFAULT_CONFIG).run(controller, rates, horizon=len(rates))
    else:
        raise ValueError(cell.key)
    m = metrics(records)
    replicas = np.asarray([record.replicas_before for record in records], dtype=np.float64)
    return {
        "hpa_total_cost": float(m.total_cost),
        "hpa_sla_violation_rate": float(m.sla_violation_rate),
        "hpa_action_churn": float(m.action_churn),
        "hpa_mean_replicas": float(np.mean(replicas)),
        "hpa_replica_std": float(np.std(replicas)),
        "hpa_replica_min": float(np.min(replicas)),
        "hpa_replica_max": float(np.max(replicas)),
    }


def summarize(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for config in CONFIGS:
        for cell in CELLS:
            rows = [row for row in raw if row["config_key"] == config.key and row["cell"] == cell.key]
            diffs = np.asarray([float(row["delta_hpa_minus_rossi"]) for row in rows], dtype=np.float64)
            ci_low, ci_high = paired_bootstrap_ci(diffs, seed=BOOTSTRAP_SEED + len(summaries))
            p_less, p_greater = sign_flip_pvalues(diffs, seed=BOOTSTRAP_SEED + 100 + len(summaries))
            summaries.append(
                {
                    "config_key": config.key,
                    "config_label": config.label,
                    "target_utilization": config.target_utilization,
                    "scale_down_stabilization_seconds": config.scale_down_stabilization_seconds,
                    "cell": cell.key,
                    "cell_label": cell.label,
                    "rossi_mean_total_cost": float(np.mean([float(row["rossi_total_cost"]) for row in rows])),
                    "hpa_mean_total_cost": float(np.mean([float(row["hpa_total_cost"]) for row in rows])),
                    "delta_mean": float(np.mean(diffs)),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "p_less_than_zero": float(p_less),
                    "p_greater_than_zero": float(p_greater),
                    "hpa_wins": int(np.sum(diffs < 0.0)),
                    "rossi_wins": int(np.sum(diffs > 0.0)),
                    "mean_replicas": float(np.mean([float(row["hpa_mean_replicas"]) for row in rows])),
                    "mean_replica_std": float(np.mean([float(row["hpa_replica_std"]) for row in rows])),
                    "mean_sla_violation_rate": float(np.mean([float(row["hpa_sla_violation_rate"]) for row in rows])),
                }
            )
    return summaries


def interpret(summaries: list[dict[str, object]]) -> dict[str, object]:
    by_cell = {cell.key: [row for row in summaries if row["cell"] == cell.key] for cell in CELLS}
    p1 = by_cell["p1"]
    clean = by_cell["clean"]
    p2 = by_cell["p2"]
    p3 = by_cell["p3"]
    return {
        "scope": "post_hoc_sensitivity_not_replacement",
        "p1_delta_range": [float(min(row["delta_mean"] for row in p1)), float(max(row["delta_mean"] for row in p1))],
        "p1_all_configs_far_below_bundled_threshold_collapse": bool(max(row["delta_mean"] for row in p1) < 0.25 * 965.0),
        "clean_hpa_dominates_all_configs": bool(all(row["delta_mean"] < 0.0 for row in clean)),
        "p2_hpa_dominates_all_configs": bool(all(row["delta_mean"] < 0.0 for row in p2)),
        "p3_hpa_dominates_all_configs": bool(all(row["delta_mean"] < 0.0 for row in p3)),
        "best_p1_config_for_rossi": max(p1, key=lambda row: row["delta_mean"])["config_key"],
        "best_clean_config_for_hpa": min(clean, key=lambda row: row["delta_mean"])["config_key"],
        "main_reading": (
            "The Experiment B conclusion is not an artifact of the single 50% target-utilization choice "
            "if P1 remains far below the bundled-threshold collapse and clean/P2/P3 remain HPA-favorable "
            "across the grid."
        ),
    }


def write_outputs(payload: dict[str, object]) -> list[Path]:
    summary_csv = TABLE_DIR / "hpa_v2_config_sensitivity_summary.csv"
    raw_csv = DATA_DIR / "hpa_v2_config_sensitivity_raw.csv"
    root_raw_csv = ROOT_DATA_DIR / "hpa_v2_config_sensitivity_raw.csv"
    json_path = OUT_DIR / "hpa_v2_config_sensitivity.json"
    report_path = OUT_DIR / "hpa_v2_config_sensitivity_results.md"
    root_report = ROOT / "hpa_v2_config_sensitivity_results.md"
    write_csv(summary_csv, payload["summaries"])
    write_csv(raw_csv, payload["raw"])
    write_csv(root_raw_csv, payload["raw"])
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figures = write_figures(payload)
    text = report_text(payload, figures)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    return [summary_csv, raw_csv, root_raw_csv, json_path, report_path, root_report, *figures]


def write_figures(payload: dict[str, object]) -> list[Path]:
    summaries = payload["summaries"]
    p1 = [row for row in summaries if row["cell"] == "p1"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for down, marker in ((300, "o"), (0, "s")):
        rows = sorted([row for row in p1 if row["scale_down_stabilization_seconds"] == down], key=lambda row: row["target_utilization"])
        xs = [100 * row["target_utilization"] for row in rows]
        ys = [row["delta_mean"] for row in rows]
        yerr = np.vstack(
            [
                np.asarray(ys) - np.asarray([row["ci_low"] for row in rows]),
                np.asarray([row["ci_high"] for row in rows]) - np.asarray(ys),
            ]
        )
        ax.errorbar(xs, ys, yerr=yerr, marker=marker, linewidth=1.4, capsize=3, label=f"scale-down {down}s")
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.axhline(965, color="#999999", linewidth=0.9, linestyle="--", label="bundled-threshold P1 anchor")
    ax.set_xlabel("HPA target utilization (%)")
    ax.set_ylabel("Delta cost: HPA-v2 - Rossi")
    ax.set_title("Rossi P1 under HPA-v2 configuration sensitivity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p1_pdf = FIG_DIR / "hpa_v2_config_p1_delta.pdf"
    p1_png = FIG_DIR / "hpa_v2_config_p1_delta.png"
    fig.savefig(p1_pdf, bbox_inches="tight")
    fig.savefig(p1_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    cells = [cell.key for cell in CELLS]
    labels = [cell.label for cell in CELLS]
    configs = [config.key for config in CONFIGS]
    matrix = np.array(
        [[next(row["delta_mean"] for row in summaries if row["config_key"] == config and row["cell"] == cell) for cell in cells] for config in configs],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    vmax = max(1.0, float(np.max(np.abs(matrix))))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(cells)), labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(configs)), [config.label for config in CONFIGS])
    ax.set_title("Delta cost across HPA-v2 configurations")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("HPA-v2 - Rossi cost")
    fig.tight_layout()
    heat_pdf = FIG_DIR / "hpa_v2_config_delta_heatmap.pdf"
    heat_png = FIG_DIR / "hpa_v2_config_delta_heatmap.png"
    fig.savefig(heat_pdf, bbox_inches="tight")
    fig.savefig(heat_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    paths = [p1_pdf, p1_png, heat_pdf, heat_png]
    for path in paths:
        target = ROOT_FIG_DIR / path.name
        target.write_bytes(path.read_bytes())
    return paths + [ROOT_FIG_DIR / path.name for path in paths]


def report_text(payload: dict[str, object], figures: list[Path]) -> str:
    interp = payload["interpretation"]
    rows = payload["summaries"]
    lines = [
        "# HPA-v2 Configuration Sensitivity for Rossi",
        "",
        f"MLflow run: `{payload.get('mlflow_run_id', 'pending')}`",
        "",
        "Scope: post-hoc sensitivity analysis, not a replacement for Experiment B.",
        "",
        "Grid: HPA-v2 target utilization `{40%, 50%, 60%, 70%}` crossed with scale-down stabilization `{300s, 0s}`. Sync period, tolerance, scale-up stabilization, and min/max replica bounds are held fixed.",
        "",
        "Delta is `cost(HPA-v2) - cost(Rossi)`. Negative values favor HPA-v2; positive values favor Rossi.",
        "",
        "## Interpretation",
        "",
        f"- P1 delta range across configs: `{interp['p1_delta_range'][0]:.3f}` to `{interp['p1_delta_range'][1]:.3f}`.",
        f"- P1 remains far below the bundled-threshold collapse anchor: `{interp['p1_all_configs_far_below_bundled_threshold_collapse']}`.",
        f"- Clean HPA-v2 dominates all configs: `{interp['clean_hpa_dominates_all_configs']}`.",
        f"- P2 HPA-v2 dominates all configs: `{interp['p2_hpa_dominates_all_configs']}`.",
        f"- P3 HPA-v2 dominates all configs: `{interp['p3_hpa_dominates_all_configs']}`.",
        "",
        "## Summary Table",
        "",
        "| Config | Cell | Rossi cost | HPA-v2 cost | Delta | 95% CI | HPA wins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config_label']} | {row['cell_label']} | {row['rossi_mean_total_cost']:.3f} | "
            f"{row['hpa_mean_total_cost']:.3f} | {row['delta_mean']:+.3f} | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {row['hpa_wins']}/30 |"
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
            "The result is a configuration sensitivity around Experiment B. It should be cited only to show whether the HPA-v2 finding is robust to representative HPA target/stabilization choices; it does not alter the locked Rossi reproduction or perturbation protocol.",
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


def artifact_group(path: Path) -> str:
    if "figures" in path.parts:
        return "paper/figures"
    if "tables" in path.parts:
        return "paper/tables"
    if "data" in path.parts:
        return "paper/data"
    if path.suffix == ".md":
        return "paper/reports"
    return "results"


def _ensure_dirs() -> None:
    for directory in (OUT_DIR, TABLE_DIR, FIG_DIR, DATA_DIR, ROOT_FIG_DIR, ROOT_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
