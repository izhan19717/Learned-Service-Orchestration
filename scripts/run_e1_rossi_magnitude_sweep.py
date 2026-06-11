#!/usr/bin/env python3
"""E1 Rossi perturbation-magnitude sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from cisose_rossi.controllers import HPAv2Controller, ModelBasedController, ThresholdHPAController
from cisose_rossi.evaluation import RossiMetrics, metrics, paired_result
from cisose_rossi.perturbations import capped_pareto_cv2, minimum_bucket_flip_utilization
from cisose_rossi.simulator import RladSimulator
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_e1_magnitude_sweep"
PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "e1_magnitude_sweep" / "rossi"
TABLE_DIR = OUT_DIR / "tables"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

LAGS = (0, 1, 2, 5, 10, 20, 50)
ALPHAS = (2.5, 2.0, 1.75, 1.5, 1.3, 1.1)
EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
BLOCK_LENGTHS = (5, 10)
BASE_SEED = 20260520
BOOTSTRAP_SEED = 20260603
DEFAULT_HORIZON = DEFAULT_CONFIG.time_limit + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-windows", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--max-workers", type=int, default=min(24, os.cpu_count() or 1))
    return parser.parse_args()


def profile_offsets(sequence_len: int, *, horizon: int, n: int, seed: int) -> tuple[int, ...]:
    max_segments = sequence_len // horizon
    if max_segments < n:
        raise ValueError(f"need {n} non-overlapping windows, found {max_segments}")
    rng = np.random.default_rng(seed)
    segments = rng.choice(max_segments, size=n, replace=False)
    return tuple(int(segment * horizon) for segment in sorted(segments))


def run_task(curve: str, magnitude: float | int, rates: tuple[float, ...]) -> dict[str, object]:
    if curve == "p1_threshold":
        return run_p1(lag=int(magnitude), rates=rates, comparator="threshold")
    if curve == "p1_hpa_v2":
        return run_p1(lag=int(magnitude), rates=rates, comparator="hpa_v2")
    if curve == "p2_threshold":
        return run_p2(alpha=float(magnitude), rates=rates)
    if curve == "p3_threshold":
        return run_p3(epsilon=float(magnitude), rates=rates)
    raise ValueError(curve)


def run_p1(*, lag: int, rates: tuple[float, ...], comparator: str) -> dict[str, object]:
    kwargs = {
        "horizon": len(rates),
        "observation_lag_steps": lag,
        "observation_applies_to_update": lag > 0,
    }
    rossi_records = RladSimulator(DEFAULT_CONFIG).run(
        ModelBasedController(DEFAULT_CONFIG),
        rates,
        **kwargs,
    )
    if comparator == "threshold":
        comp = ThresholdHPAController(DEFAULT_CONFIG)
    elif comparator == "hpa_v2":
        comp = HPAv2Controller(DEFAULT_CONFIG)
    else:
        raise ValueError(comparator)
    comp_records = RladSimulator(DEFAULT_CONFIG).run(comp, rates, **kwargs)
    return metrics_payload(rossi_records, comp_records)


def run_p2(*, alpha: float, rates: tuple[float, ...]) -> dict[str, object]:
    cv2 = capped_pareto_cv2(alpha)
    rossi_records = RladSimulator(DEFAULT_CONFIG, service_time_cv2=cv2).run(
        ModelBasedController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
    )
    comp_records = RladSimulator(DEFAULT_CONFIG, service_time_cv2=cv2).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
    )
    payload = metrics_payload(rossi_records, comp_records)
    payload["service_time_cv2"] = cv2
    return payload


def run_p3(*, epsilon: float, rates: tuple[float, ...]) -> dict[str, object]:
    if epsilon <= 0.0:
        rossi_records = RladSimulator(DEFAULT_CONFIG).run(
            ModelBasedController(DEFAULT_CONFIG),
            rates,
            horizon=len(rates),
        )
        comp_records = RladSimulator(DEFAULT_CONFIG).run(
            ThresholdHPAController(DEFAULT_CONFIG),
            rates,
            horizon=len(rates),
        )
        payload = metrics_payload(rossi_records, comp_records)
        payload.update({"attack_fraction": 0.0, "mean_abs_delta": 0.0, "max_abs_delta": 0.0})
        return payload

    def transform(controller, service, util: float) -> float:
        delta = minimum_bucket_flip_utilization(
            controller=controller,
            replicas=service.replicas,
            cpu=service.cpu,
            utilization=util,
            epsilon=epsilon,
            config=DEFAULT_CONFIG,
        )
        return util + delta

    rossi_records = RladSimulator(DEFAULT_CONFIG).run(
        ModelBasedController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
        observation_transform=transform,
    )
    comp_records = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
    )
    payload = metrics_payload(rossi_records, comp_records)
    abs_deltas = np.abs([record.observation_delta for record in rossi_records])
    payload.update(
        {
            "attack_fraction": float(np.mean(abs_deltas > 1e-12)),
            "mean_abs_delta": float(np.mean(abs_deltas)),
            "max_abs_delta": float(np.max(abs_deltas)),
        }
    )
    return payload


def metrics_payload(rossi_records, comparator_records) -> dict[str, object]:
    rossi = metrics(rossi_records)
    comp = metrics(comparator_records)
    return {
        "rossi_total_cost": rossi.total_cost,
        "comparator_total_cost": comp.total_cost,
        "rossi_sla_violation_rate": rossi.sla_violation_rate,
        "comparator_sla_violation_rate": comp.sla_violation_rate,
        "rossi_action_churn": rossi.action_churn,
        "comparator_action_churn": comp.action_churn,
        "rossi_mean_replicas": float(np.mean([record.replicas_before for record in rossi_records])),
        "comparator_mean_replicas": float(np.mean([record.replicas_before for record in comparator_records])),
    }


def summarize(raw: dict[str, dict[float | int, list[dict[str, object]]]], *, bootstrap_seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for curve_idx, (curve, by_mag) in enumerate(raw.items()):
        p_less: dict[str, float] = {}
        p_greater: dict[str, float] = {}
        pending = []
        for mag_idx, (magnitude, entries) in enumerate(by_mag.items()):
            entries.sort(key=lambda item: int(item["window_index"]))
            rossi_values = [float(entry["rossi_total_cost"]) for entry in entries]
            comp_values = [float(entry["comparator_total_cost"]) for entry in entries]
            rossi_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in rossi_values)
            comp_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in comp_values)
            seed = bootstrap_seed + 10_000 * curve_idx + mag_idx
            comparison = paired_result(comp_metrics, rossi_metrics, seed=seed)
            diffs = list(comparison.differences)
            row = {
                "curve": curve,
                "magnitude": magnitude,
                "rossi_mean_total_cost": float(np.mean(rossi_values)),
                "comparator_mean_total_cost": float(np.mean(comp_values)),
                "delta_comparator_minus_rossi": comparison.mean_difference,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "p_less_than_zero": comparison.p_less_than_zero,
                "p_greater_than_zero": comparison.p_greater_than_zero,
                "rossi_sla_violation_rate": float(np.mean([float(e["rossi_sla_violation_rate"]) for e in entries])),
                "comparator_sla_violation_rate": float(np.mean([float(e["comparator_sla_violation_rate"]) for e in entries])),
                "rossi_action_churn": float(np.mean([float(e["rossi_action_churn"]) for e in entries])),
                "comparator_action_churn": float(np.mean([float(e["comparator_action_churn"]) for e in entries])),
                "differences": diffs,
            }
            for optional in ("service_time_cv2", "attack_fraction", "mean_abs_delta", "max_abs_delta"):
                if optional in entries[0]:
                    row[optional] = float(np.mean([float(e[optional]) for e in entries]))
            for block_len in BLOCK_LENGTHS:
                block = block_analysis(diffs, block_len=block_len, seed=seed + 100 + block_len)
                row[f"block_L{block_len}_ci_low"] = block["ci_low"]
                row[f"block_L{block_len}_ci_high"] = block["ci_high"]
                row[f"block_L{block_len}_p_less"] = block["p_less"]
                row[f"block_L{block_len}_p_greater"] = block["p_greater"]
            key = f"{curve}:{magnitude}"
            p_less[key] = comparison.p_less_than_zero
            p_greater[key] = comparison.p_greater_than_zero
            pending.append((key, row))
        holm_less = holm_bonferroni(p_less)
        holm_greater = holm_bonferroni(p_greater)
        for key, row in pending:
            row["holm_less_curve"] = holm_less[key]
            row["holm_greater_curve"] = holm_greater[key]
            rows.append(row)
    return rows


def block_analysis(differences: list[float], *, block_len: int, seed: int) -> dict[str, float]:
    diffs = np.asarray(differences, dtype=np.float64)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    blocks = np.asarray([diffs[start : start + block_len] for start in range(0, n - block_len + 1)])
    sampled_blocks = math.ceil(n / block_len)
    idx = rng.integers(0, len(blocks), size=(5000, sampled_blocks))
    means = np.empty(5000, dtype=np.float64)
    for i in range(5000):
        means[i] = blocks[idx[i]].reshape(-1)[:n].mean()
    nonoverlap = [diffs[start : min(start + block_len, n)] for start in range(0, n, block_len)]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(100_000, len(nonoverlap)))
    null = np.empty(100_000, dtype=np.float64)
    for i in range(100_000):
        total = 0.0
        count = 0
        for sign, block in zip(signs[i], nonoverlap, strict=True):
            total += float(sign) * float(np.sum(block))
            count += len(block)
        null[i] = total / count
    observed = float(np.mean(diffs))
    return {
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_less": float((np.count_nonzero(null <= observed) + 1.0) / 100_001.0),
        "p_greater": float((np.count_nonzero(null >= observed) + 1.0) / 100_001.0),
    }


def write_outputs(rows: list[dict[str, object]], raw: dict[str, dict[float | int, list[dict[str, object]]]], run_id: str, params: dict[str, object]) -> None:
    for directory in (OUT_DIR, TABLE_DIR, DATA_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "e1_rossi_magnitude_sweep.csv"
    json_path = OUT_DIR / "e1_rossi_magnitude_sweep.json"
    md_path = OUT_DIR / "e1_rossi_magnitude_sweep.md"
    raw_csv = DATA_DIR / "e1_rossi_raw_windows.csv"

    fieldnames = [key for key in rows[0].keys() if key != "differences"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames_raw = [
            "curve",
            "magnitude",
            "window_index",
            "offset",
            "rossi_total_cost",
            "comparator_total_cost",
            "delta_comparator_minus_rossi",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames_raw, extrasaction="ignore")
        writer.writeheader()
        for curve, by_mag in raw.items():
            for magnitude, entries in by_mag.items():
                for entry in entries:
                    writer.writerow(
                        {
                            **entry,
                            "curve": curve,
                            "magnitude": magnitude,
                            "delta_comparator_minus_rossi": float(entry["comparator_total_cost"])
                            - float(entry["rossi_total_cost"]),
                        }
                    )
    payload = {"experiment": "E1", "method": "rossi", "params": params, "summary": rows}
    write_json_artifact(json_path, payload, run_id=run_id)
    mlflow.log_artifact(str(csv_path), artifact_path="paper/tables")
    mlflow.log_artifact(str(raw_csv), artifact_path="paper/data")

    threshold_k10 = next(row for row in rows if row["curve"] == "p1_threshold" and int(row["magnitude"]) == 10)
    bundled_collapse = abs(float(threshold_k10["delta_comparator_minus_rossi"]))
    hpa_rows = [row for row in rows if row["curve"] == "p1_hpa_v2"]
    h2_all_below_quarter = all(
        abs(float(row["delta_comparator_minus_rossi"])) < 0.25 * bundled_collapse for row in hpa_rows
    )
    h2_any_above_half = any(
        abs(float(row["delta_comparator_minus_rossi"])) >= 0.5 * bundled_collapse for row in hpa_rows
    )
    lines = [
        "# E1 Rossi Magnitude Sweep",
        "",
        f"MLflow run: `{run_id}`",
        "",
        f"Bundled-threshold collapse reference at k=10: `{bundled_collapse:.6g}`.",
        f"H2 all HPA-v2 gaps below 25% of bundled collapse: `{h2_all_below_quarter}`.",
        f"H2 any HPA-v2 gap at/above 50% of bundled collapse: `{h2_any_above_half}`.",
        "",
        "| Curve | Magnitude | Delta comparator-Rossi | 95% CI | Holm p(Delta<0) | Holm p(Delta>0) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {curve} | {magnitude} | {delta_comparator_minus_rossi:.6g} | "
            "[{ci_low:.6g}, {ci_high:.6g}] | {holm_less_curve:.6g} | {holm_greater_curve:.6g} |".format(
                **row
            )
        )
    lines.extend(["", f"- CSV: `{csv_path.relative_to(ROOT)}`", f"- Raw: `{raw_csv.relative_to(ROOT)}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(md_path), artifact_path="paper")
    write_figure(rows, bundled_collapse)


def write_figure(rows: list[dict[str, object]], bundled_collapse: float) -> None:
    specs = [
        ("p1_threshold", "P1 bundled threshold", "Lag k"),
        ("p1_hpa_v2", "P1 HPA-v2", "Lag k"),
        ("p2_threshold", "P2 tail", "Pareto alpha"),
        ("p3_threshold", "P3 bucket flip", "epsilon"),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.9,
            "axes.labelsize": 9.1,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(5.65, 4.35))
    for ax, (curve, title, xlabel) in zip(axes.ravel(), specs, strict=True):
        subset = sorted([row for row in rows if row["curve"] == curve], key=lambda row: float(row["magnitude"]))
        xs = np.asarray([float(row["magnitude"]) for row in subset], dtype=float)
        ys = np.asarray([float(row["delta_comparator_minus_rossi"]) for row in subset], dtype=float)
        lows = np.asarray([float(row["ci_low"]) for row in subset], dtype=float)
        highs = np.asarray([float(row["ci_high"]) for row in subset], dtype=float)
        ax.axhline(0.0, color="#333333", linewidth=0.85)
        if curve == "p1_hpa_v2":
            ax.axhline(0.25 * bundled_collapse, color="#999999", linestyle=":", linewidth=0.95)
            ax.axhline(-0.25 * bundled_collapse, color="#999999", linestyle=":", linewidth=0.95)
        ax.fill_between(xs, lows, highs, color="#9ecae9", alpha=0.34, linewidth=0)
        ax.plot(xs, ys, color="#1f77b4", marker="o", markersize=4.2, linewidth=1.45)
        ax.set_title(title, pad=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Delta")
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if curve.startswith("p1"):
            ax.set_xticks([0, 10, 20, 50])
        elif curve == "p2_threshold":
            ax.set_xticks([1.5, 2.0, 2.5])
        else:
            ax.set_xticks([0.00, 0.05, 0.10, 0.20])
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.91, hspace=0.54, wspace=0.36)
    pdf = FIG_DIR / "e1_rossi_magnitude_sweep.pdf"
    png = FIG_DIR / "e1_rossi_magnitude_sweep.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(pdf), artifact_path="paper/figures")
        mlflow.log_artifact(str(png), artifact_path="paper/figures")


def main() -> None:
    args = parse_args()
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    offsets = profile_offsets(len(sequence), horizon=args.horizon, n=args.n_windows, seed=args.base_seed)
    curve_values: dict[str, tuple[float | int, ...]] = {
        "p1_threshold": LAGS,
        "p1_hpa_v2": LAGS,
        "p2_threshold": ALPHAS,
        "p3_threshold": EPSILONS,
    }
    params = {
        "protocol": "PREREG_E1_magnitude_sweep",
        "method": "rossi",
        "metric": "total_cost",
        "delta": "metric(comparator)-metric(Rossi)",
        "curves": {key: list(value) for key, value in curve_values.items()},
        "n_windows": args.n_windows,
        "horizon": args.horizon,
        "base_seed": args.base_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "max_workers": args.max_workers,
        "block_lengths": list(BLOCK_LENGTHS),
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "profile_sha256": PROFILE_SHA256,
        "compute_policy": "canonical on remote workstation; local workstation only for smoke",
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="e1-rossi-magnitude-sweep",
        role="e1_magnitude_sweep",
        params=params,
        tags={"experiment": "E1", "method": "rossi"},
    ) as run:
        raw: dict[str, dict[float | int, list[dict[str, object]]]] = {
            curve: {magnitude: [] for magnitude in magnitudes}
            for curve, magnitudes in curve_values.items()
        }
        futures = {}
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            for curve, magnitudes in curve_values.items():
                for magnitude in magnitudes:
                    for window_index, offset in enumerate(offsets):
                        rates = tuple(sequence[offset : offset + args.horizon])
                        future = pool.submit(run_task, curve, magnitude, rates)
                        futures[future] = (curve, magnitude, window_index, offset)
            for done, future in enumerate(as_completed(futures), start=1):
                curve, magnitude, window_index, offset = futures[future]
                row = future.result()
                row["window_index"] = window_index
                row["offset"] = offset
                raw[curve][magnitude].append(row)
                print(
                    f"E1 Rossi {done}/{len(futures)} curve={curve} magnitude={magnitude} window={window_index}",
                    flush=True,
                )
        rows = summarize(raw, bootstrap_seed=args.bootstrap_seed)
        write_outputs(rows, raw, run.info.run_id, params)
        for row in rows:
            key = f"{row['curve']}.{row['magnitude']}"
            mlflow.log_metric(f"{key}.delta", row["delta_comparator_minus_rossi"])
            mlflow.log_metric(f"{key}.ci_low", row["ci_low"])
            mlflow.log_metric(f"{key}.ci_high", row["ci_high"])
        print(f"MLflow run: {run.info.run_id}")
        print(str((OUT_DIR / "e1_rossi_magnitude_sweep.md").relative_to(ROOT)))


if __name__ == "__main__":
    main()
