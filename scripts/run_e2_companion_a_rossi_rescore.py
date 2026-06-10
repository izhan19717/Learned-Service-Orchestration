#!/usr/bin/env python3
"""E2 Companion Analysis A: deterministic Rossi reward-weight rescoring."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_common.stats import holm_bonferroni, paired_bootstrap_ci, sign_flip_pvalues
from cisose_common.tracking import start_run, write_json_artifact
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import ModelBasedController, ThresholdHPAController
from cisose_rossi.simulator import RladSimulator, StepRecord
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_e2_objective_native"
PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "e2_objective_native"
TABLE_DIR = OUT_DIR / "tables"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

CHURN_WEIGHTS = (0.01, 0.05, 0.10, 0.20, 0.30)
CHURN_VARIANTS = ("adaptation_nonnoop", "source_vertical", "action_change")
CELLS = {"clean": 0, "p1_lag_k10": 10}
BASE_SEED = 20260520
BOOTSTRAP_SEED = 20260603
DEFAULT_HORIZON = DEFAULT_CONFIG.time_limit + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-windows", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--max-workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


def profile_offsets(sequence_len: int, *, horizon: int, n: int, seed: int) -> tuple[int, ...]:
    max_segments = sequence_len // horizon
    if max_segments < n:
        raise ValueError(f"need {n} non-overlapping windows, found {max_segments}")
    rng = np.random.default_rng(seed)
    segments = rng.choice(max_segments, size=n, replace=False)
    return tuple(int(segment * horizon) for segment in sorted(segments))


def renormalized_weights(churn_weight: float) -> dict[str, float]:
    if not 0.0 < churn_weight < 1.0:
        raise ValueError("churn_weight must lie in (0,1)")
    non_churn = 1.0 - churn_weight
    return {
        "sla": non_churn * DEFAULT_CONFIG.w_sla / (DEFAULT_CONFIG.w_sla + DEFAULT_CONFIG.w_resources),
        "resource": non_churn
        * DEFAULT_CONFIG.w_resources
        / (DEFAULT_CONFIG.w_sla + DEFAULT_CONFIG.w_resources),
        "churn": churn_weight,
    }


def run_window(cell: str, rates: tuple[float, ...]) -> dict[str, object]:
    lag = CELLS[cell]
    simulator_kwargs = {
        "horizon": len(rates),
        "observation_lag_steps": lag,
        "observation_applies_to_update": lag > 0,
    }
    rossi_records = RladSimulator(DEFAULT_CONFIG).run(
        ModelBasedController(DEFAULT_CONFIG),
        rates,
        **simulator_kwargs,
    )
    threshold_records = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        **simulator_kwargs,
    )
    return {
        "cell": cell,
        "rossi_components": component_totals(rossi_records),
        "threshold_components": component_totals(threshold_records),
    }


def component_totals(records: tuple[StepRecord, ...]) -> dict[str, float]:
    resource = np.asarray(
        [
            record.replicas_before
            * (record.cpu_before / 100.0)
            / DEFAULT_CONFIG.max_replication
            for record in records
        ],
        dtype=np.float64,
    )
    sla = np.asarray([1.0 if record.sla_violated else 0.0 for record in records], dtype=np.float64)
    nonnoop = np.asarray([1.0 if record.action_index != 1 else 0.0 for record in records], dtype=np.float64)
    vertical = np.asarray(
        [1.0 if record.action_label.startswith("vertical_") else 0.0 for record in records],
        dtype=np.float64,
    )
    action_indices = [record.action_index for record in records]
    action_change = np.asarray(
        [1.0 if prev != curr else 0.0 for prev, curr in zip(action_indices, action_indices[1:], strict=False)],
        dtype=np.float64,
    )
    return {
        "n_steps": float(len(records)),
        "sla_sum": float(np.sum(sla)),
        "resource_sum": float(np.sum(resource)),
        "adaptation_nonnoop_sum": float(np.sum(nonnoop)),
        "source_vertical_sum": float(np.sum(vertical)),
        "action_change_sum": float(np.sum(action_change)),
        "sla_rate": float(np.mean(sla)),
        "resource_mean": float(np.mean(resource)),
        "adaptation_nonnoop_rate": float(np.mean(nonnoop)),
        "source_vertical_rate": float(np.mean(vertical)),
        "action_change_rate": float(np.mean(action_change)) if len(action_change) else 0.0,
    }


def rescore(components: dict[str, float], weights: dict[str, float], churn_variant: str) -> float:
    return float(
        weights["sla"] * components["sla_sum"]
        + weights["resource"] * components["resource_sum"]
        + weights["churn"] * components[f"{churn_variant}_sum"]
    )


def summarize(raw: list[dict[str, object]], *, bootstrap_seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {cell: [] for cell in CELLS}
    for row in raw:
        grouped[str(row["cell"])].append(row)
    for entries in grouped.values():
        entries.sort(key=lambda item: int(item["window_index"]))

    for cell_idx, (cell, entries) in enumerate(grouped.items()):
        for churn_variant in CHURN_VARIANTS:
            p_less: dict[str, float] = {}
            p_greater: dict[str, float] = {}
            pending_rows = []
            for weight_idx, churn_weight in enumerate(CHURN_WEIGHTS):
                weights = renormalized_weights(churn_weight)
                rossi_scores = [
                    rescore(entry["rossi_components"], weights, churn_variant)  # type: ignore[arg-type]
                    for entry in entries
                ]
                threshold_scores = [
                    rescore(entry["threshold_components"], weights, churn_variant)  # type: ignore[arg-type]
                    for entry in entries
                ]
                diffs = [threshold - rossi for threshold, rossi in zip(threshold_scores, rossi_scores, strict=True)]
                seed = bootstrap_seed + 10_000 * cell_idx + 100 * CHURN_VARIANTS.index(churn_variant) + weight_idx
                ci_low, ci_high = paired_bootstrap_ci(diffs, seed=seed)
                less, greater = sign_flip_pvalues(diffs, seed=seed + 1)
                key = f"{cell}:{churn_variant}:w={churn_weight}"
                p_less[key] = less
                p_greater[key] = greater
                row = {
                    "cell": cell,
                    "churn_variant": churn_variant,
                    "churn_weight": churn_weight,
                    "w_sla": weights["sla"],
                    "w_resource": weights["resource"],
                    "w_churn": weights["churn"],
                    "rossi_mean_score": float(np.mean(rossi_scores)),
                    "threshold_mean_score": float(np.mean(threshold_scores)),
                    "delta_threshold_minus_rossi": float(np.mean(diffs)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_less_than_zero": less,
                    "p_greater_than_zero": greater,
                    "rossi_wins": int(np.sum(np.asarray(diffs) > 0.0)),
                    "threshold_wins": int(np.sum(np.asarray(diffs) < 0.0)),
                    "differences": diffs,
                }
                pending_rows.append((key, row))
            holm_less = holm_bonferroni(p_less)
            holm_greater = holm_bonferroni(p_greater)
            for key, row in pending_rows:
                row["holm_less_than_zero_within_cell_variant"] = holm_less[key]
                row["holm_greater_than_zero_within_cell_variant"] = holm_greater[key]
                summary_rows.append(row)

        for window_idx, entry in enumerate(entries):
            for method_key in ("rossi_components", "threshold_components"):
                components = entry[method_key]
                method = method_key.replace("_components", "")
                raw_rows.append(
                    {
                        "cell": cell,
                        "window_index": window_idx,
                        "offset": entry["offset"],
                        "method": method,
                        **components,  # type: ignore[arg-type]
                    }
                )
    return summary_rows, raw_rows


def write_outputs(summary_rows: list[dict[str, object]], raw_rows: list[dict[str, object]], run_id: str, params: dict[str, object]) -> None:
    for directory in (OUT_DIR, TABLE_DIR, DATA_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    summary_csv = TABLE_DIR / "e2_companion_a_weight_rescore.csv"
    raw_csv = DATA_DIR / "e2_companion_a_components.csv"
    json_path = OUT_DIR / "e2_companion_a_weight_rescore.json"
    md_path = OUT_DIR / "e2_companion_a_results.md"

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [k for k in summary_rows[0].keys() if k != "differences"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    payload = {
        "experiment": "E2 Companion Analysis A",
        "mlflow_run_id": run_id,
        "params": params,
        "summary": summary_rows,
        "raw_components_csv": str(raw_csv.relative_to(ROOT)),
        "summary_csv": str(summary_csv.relative_to(ROOT)),
    }
    write_json_artifact(json_path, payload, run_id=run_id)

    primary = [
        row
        for row in summary_rows
        if row["churn_variant"] == "adaptation_nonnoop" and row["cell"] in ("clean", "p1_lag_k10")
    ]
    lines = [
        "# E2 Companion Analysis A",
        "",
        f"MLflow run: `{run_id}`",
        "",
        "This is deterministic rescoring of fixed Rossi/threshold rollouts; no policy training is performed.",
        "",
        "Primary paper-facing churn variant: `adaptation_nonnoop`.",
        "",
        "| Cell | w_churn | Rossi score | Threshold score | Delta threshold-Rossi | 95% CI | Holm p(Delta<0) | Holm p(Delta>0) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {cell} | {churn_weight:.2f} | {rossi_mean_score:.6g} | {threshold_mean_score:.6g} | "
            "{delta_threshold_minus_rossi:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | "
            "{holm_less_than_zero_within_cell_variant:.6g} | {holm_greater_than_zero_within_cell_variant:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Delta is `score(threshold) - score(Rossi)`. Positive values mean Rossi has the lower rescored scalar objective.",
            "",
            f"- Summary CSV: `{summary_csv.relative_to(ROOT)}`",
            f"- Raw component CSV: `{raw_csv.relative_to(ROOT)}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(summary_csv), artifact_path="paper/tables")
    mlflow.log_artifact(str(raw_csv), artifact_path="paper/data")
    mlflow.log_artifact(str(md_path), artifact_path="paper")

    write_figure(summary_rows)


def write_figure(summary_rows: list[dict[str, object]]) -> None:
    primary = [
        row for row in summary_rows if row["churn_variant"] == "adaptation_nonnoop"
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for cell, color in (("clean", "#4c78a8"), ("p1_lag_k10", "#f58518")):
        rows = [row for row in primary if row["cell"] == cell]
        xs = np.asarray([row["churn_weight"] for row in rows], dtype=float)
        ys = np.asarray([row["delta_threshold_minus_rossi"] for row in rows], dtype=float)
        lows = np.asarray([row["ci_low"] for row in rows], dtype=float)
        highs = np.asarray([row["ci_high"] for row in rows], dtype=float)
        ax.plot(xs, ys, marker="o", linewidth=1.3, color=color, label=cell.replace("_", " "))
        ax.fill_between(xs, lows, highs, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.axvline(0.01, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Churn weight")
    ax.set_ylabel("Score(threshold) - score(Rossi)")
    ax.set_title("E2 Companion A rescoring")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    pdf = FIG_DIR / "e2_companion_a_rescore.pdf"
    png = FIG_DIR / "e2_companion_a_rescore.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    mlflow.log_artifact(str(pdf), artifact_path="paper/figures")
    mlflow.log_artifact(str(png), artifact_path="paper/figures")


def main() -> None:
    args = parse_args()
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    offsets = profile_offsets(
        len(sequence), horizon=args.horizon, n=args.n_windows, seed=args.base_seed
    )
    params = {
        "protocol": "PREREG_E2_objective_native Companion Analysis A",
        "status": "zero_new_training_rescore",
        "n_windows": args.n_windows,
        "horizon": args.horizon,
        "base_seed": args.base_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "churn_weights": list(CHURN_WEIGHTS),
        "churn_variants": list(CHURN_VARIANTS),
        "cells": CELLS,
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "profile_sha256": PROFILE_SHA256,
        "compute_policy": "canonical on remote workstation; local workstation only for smoke",
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="e2-companion-a-rossi-weight-rescore",
        role="e2_companion_a",
        params=params,
        tags={"experiment": "E2", "method": "rossi", "analysis": "companion_a"},
    ) as run:
        futures = {}
        raw: list[dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            for cell in CELLS:
                for window_index, offset in enumerate(offsets):
                    rates = tuple(sequence[offset : offset + args.horizon])
                    future = pool.submit(run_window, cell, rates)
                    futures[future] = (cell, window_index, offset)
            for done, future in enumerate(as_completed(futures), start=1):
                cell, window_index, offset = futures[future]
                row = future.result()
                row["window_index"] = window_index
                row["offset"] = offset
                raw.append(row)
                print(f"E2A {done}/{len(futures)} cell={cell} window={window_index}", flush=True)
        summary_rows, raw_rows = summarize(raw, bootstrap_seed=args.bootstrap_seed)
        write_outputs(summary_rows, raw_rows, run.info.run_id, params)
        for row in summary_rows:
            if row["churn_variant"] != "adaptation_nonnoop":
                continue
            key = f"{row['cell']}.w{row['churn_weight']}"
            mlflow.log_metric(f"{key}.delta_threshold_minus_rossi", row["delta_threshold_minus_rossi"])
            mlflow.log_metric(f"{key}.ci_low", row["ci_low"])
            mlflow.log_metric(f"{key}.ci_high", row["ci_high"])
        print(f"MLflow run: {run.info.run_id}")
        print(str((OUT_DIR / "e2_companion_a_results.md").relative_to(ROOT)))


if __name__ == "__main__":
    main()
