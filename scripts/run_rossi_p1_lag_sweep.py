#!/usr/bin/env python3
"""Run Rossi P1 observation-lag sweep with the frozen clean controller."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_common.tracking import sha256_file, start_run, write_json_artifact
from cisose_rossi.checkpointing import load_model_based_checkpoint
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import ThresholdHPAController
from cisose_rossi.evaluation import RossiMetrics, metrics, paired_result
from cisose_rossi.simulator import RladSimulator
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_rossi_v2_2"
PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
CHECKPOINT_PATH = ROOT / "results" / "rossi" / "checkpoints" / "model_based_table_i_clean.npz"
LAGS = (0, 1, 2, 5, 10, 30)
HORIZON = DEFAULT_CONFIG.time_limit + 1
N_SEEDS = 30
BASE_SEED = 20260520
BOOTSTRAP_SEED = 20260521


def profile_offsets(sequence_len: int, *, horizon: int, n: int, seed: int) -> tuple[int, ...]:
    max_segments = sequence_len // horizon
    if max_segments < n:
        raise ValueError(f"need {n} non-overlapping windows, found {max_segments}")
    rng = np.random.default_rng(seed)
    segments = rng.choice(max_segments, size=n, replace=False)
    return tuple(int(segment * horizon) for segment in sorted(segments))


def run_cell(rates: tuple[float, ...], lag: int) -> tuple[float, float]:
    rossi, _ = load_model_based_checkpoint(CHECKPOINT_PATH, freeze=True)
    rossi_records = RladSimulator(DEFAULT_CONFIG).run(
        rossi,
        rates,
        horizon=HORIZON,
        observation_lag_steps=lag,
    )
    hpa_records = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=HORIZON,
        observation_lag_steps=lag,
    )
    return metrics(rossi_records).total_cost, metrics(hpa_records).total_cost


def outcome(ci_low: float, ci_high: float) -> str:
    if ci_high < 0.0:
        return "confirmed"
    if ci_low > 0.0:
        return "falsified"
    return "inconclusive"


def write_tables(cells: list[dict[str, object]], run_id: str) -> None:
    table_dir = ROOT / "results" / "paper" / "rossi" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "rossi_p1_lag_sweep.csv"
    fieldnames = [
        "lag",
        "rossi_mean_total_cost",
        "hpa_mean_total_cost",
        "delta_hpa_minus_rossi",
        "ci_low",
        "ci_high",
        "p_less_than_zero",
        "p_greater_than_zero",
        "outcome",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow({key: cell[key] for key in fieldnames})
    md_path = table_dir / "rossi_p1_lag_sweep.md"
    lines = [
        "# Rossi P1 Observation-Lag Sweep",
        "",
        f"MLflow run: `{run_id}`",
        "",
        "| Lag (s) | Rossi cost | HPA cost | Delta HPA-Rossi | 95% CI | Outcome |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        lines.append(
            "| {lag} | {rossi_mean_total_cost:.6g} | {hpa_mean_total_cost:.6g} | "
            "{delta_hpa_minus_rossi:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | {outcome} |".format(
                **cell
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(csv_path), artifact_path="paper/tables")
    mlflow.log_artifact(str(md_path), artifact_path="paper/tables")


def write_figure(cells: list[dict[str, object]]) -> None:
    fig_dir = ROOT / "results" / "paper" / "rossi" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / "rossi_p1_observation_lag.pdf"
    png_path = fig_dir / "rossi_p1_observation_lag.png"
    lags = np.asarray([cell["lag"] for cell in cells], dtype=float)
    deltas = np.asarray([cell["delta_hpa_minus_rossi"] for cell in cells], dtype=float)
    ci_low = np.asarray([cell["ci_low"] for cell in cells], dtype=float)
    ci_high = np.asarray([cell["ci_high"] for cell in cells], dtype=float)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.axvline(10.0, color="#b00020", linewidth=0.9, linestyle="--")
    ax.fill_between(lags, ci_low, ci_high, color="#9ecae9", alpha=0.45, linewidth=0)
    ax.plot(lags, deltas, marker="o", color="#1f77b4", linewidth=1.4)
    ax.set_xlabel("Observation lag k (s)")
    ax.set_ylabel("Total cost(HPA) - total cost(Rossi)")
    ax.set_title("Rossi P1 observation lag")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    mlflow.log_artifact(str(pdf_path), artifact_path="paper/figures")
    mlflow.log_artifact(str(png_path), artifact_path="paper/figures")


def main() -> None:
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    offsets = profile_offsets(len(sequence), horizon=HORIZON, n=N_SEEDS, seed=BASE_SEED)
    params = {
        "method": "rossi_rlad",
        "prediction": "P1-Rossi",
        "perturbation": "observation_lag",
        "p1_anchor_lag_seconds": 10,
        "lags": list(LAGS),
        "n_seeds": N_SEEDS,
        "seed_definition": "non-overlapping official slow-profile start offsets",
        "base_seed": BASE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "horizon": HORIZON,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "profile_sha256": PROFILE_SHA256,
        "comparator": "source_threshold_hpa",
        "p1_comparator_semantics": "Option B: both Rossi and HPA observe lagged utilization",
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="rossi-p1-observation-lag-sweep",
        role="perturbation_sweep",
        params=params,
    ) as run:
        cells = []
        for lag in LAGS:
            rossi_values = []
            hpa_values = []
            for offset in offsets:
                rates = tuple(sequence[offset : offset + HORIZON])
                rossi_cost, hpa_cost = run_cell(rates, lag)
                rossi_values.append(rossi_cost)
                hpa_values.append(hpa_cost)
            rossi_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in rossi_values)
            hpa_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in hpa_values)
            comparison = paired_result(hpa_metrics, rossi_metrics, seed=BOOTSTRAP_SEED + lag)
            cell = {
                "lag": lag,
                "rossi_mean_total_cost": float(np.mean(rossi_values)),
                "hpa_mean_total_cost": float(np.mean(hpa_values)),
                "delta_hpa_minus_rossi": comparison.mean_difference,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "p_less_than_zero": comparison.p_less_than_zero,
                "p_greater_than_zero": comparison.p_greater_than_zero,
                "outcome": outcome(comparison.ci_low, comparison.ci_high),
                "rossi_total_costs": rossi_values,
                "hpa_total_costs": hpa_values,
            }
            cells.append(cell)
            mlflow.log_metric(f"lag_{lag}_delta_hpa_minus_rossi", cell["delta_hpa_minus_rossi"])
            mlflow.log_metric(f"lag_{lag}_ci_low", cell["ci_low"])
            mlflow.log_metric(f"lag_{lag}_ci_high", cell["ci_high"])
        anchor = next(cell for cell in cells if cell["lag"] == 10)
        result = {
            "mlflow_run_id": run.info.run_id,
            "prediction": "P1-Rossi",
            "anchor_lag_seconds": 10,
            "anchor_outcome": anchor["outcome"],
            "offsets": offsets,
            "cells": cells,
            "params": params,
        }
        mlflow.log_metric("anchor_delta_hpa_minus_rossi", anchor["delta_hpa_minus_rossi"])
        mlflow.log_metric("anchor_ci_low", anchor["ci_low"])
        mlflow.log_metric("anchor_ci_high", anchor["ci_high"])
        mlflow.log_metric("anchor_confirmed", 1.0 if anchor["outcome"] == "confirmed" else 0.0)
        out = ROOT / "results" / "rossi" / "p1_observation_lag_sweep.json"
        write_json_artifact(out, result, run_id=run.info.run_id)
        write_tables(cells, run.info.run_id)
        write_figure(cells)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
