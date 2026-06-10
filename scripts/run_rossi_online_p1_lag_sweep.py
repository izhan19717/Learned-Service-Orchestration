#!/usr/bin/env python3
"""Run canonical online-adaptive Rossi P1 observation-lag sweep."""

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

from cisose_common.tracking import start_run, write_json_artifact
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import ModelBasedController, ThresholdHPAController
from cisose_rossi.evaluation import RossiMetrics, metrics, paired_result
from cisose_rossi.simulator import RladSimulator
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_rossi_v2_2"
PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
P2_RESULT_PATH = ROOT / "results" / "rossi" / "p2_online_service_tail.json"
DEFAULT_LAGS = (0, 1, 2, 5, 10, 30)
DEFAULT_HORIZON = DEFAULT_CONFIG.time_limit + 1
BASE_SEED = 20260520
BOOTSTRAP_SEED = 20260521


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", choices=("anchor", "anchor_plus_clean", "full"), default="full")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--reuse-p2-clean",
        action="store_true",
        help=(
            "For anchor_plus_clean, reuse the completed P2 alpha=inf online clean cell "
            "as P1 k=0."
        ),
    )
    parser.add_argument(
        "--update-on-lagged-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the delayed utilization observation to Rossi's online state update as well as "
            "its action selection. Enabled for canonical P1 environmental observation lag."
        ),
    )
    return parser.parse_args()


def profile_offsets(sequence_len: int, *, horizon: int, n: int, seed: int) -> tuple[int, ...]:
    max_segments = sequence_len // horizon
    if max_segments < n:
        raise ValueError(f"need {n} non-overlapping windows, found {max_segments}")
    rng = np.random.default_rng(seed)
    segments = rng.choice(max_segments, size=n, replace=False)
    return tuple(int(segment * horizon) for segment in sorted(segments))


def selected_lags(values: str) -> tuple[int, ...]:
    if values == "anchor":
        return (10,)
    if values == "anchor_plus_clean":
        return (0, 10)
    return DEFAULT_LAGS


def outcome(ci_low: float, ci_high: float) -> str:
    if ci_high < 0.0:
        return "confirmed"
    if ci_low > 0.0:
        return "falsified"
    return "inconclusive"


def run_p1_task(
    lag: int,
    rates: tuple[float, ...],
    *,
    update_on_lagged_observation: bool,
) -> dict[str, object]:
    rossi_records = RladSimulator(DEFAULT_CONFIG).run(
        ModelBasedController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
        observation_lag_steps=lag,
        observation_applies_to_update=update_on_lagged_observation,
    )
    hpa_records = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
        observation_lag_steps=lag,
        observation_applies_to_update=update_on_lagged_observation,
    )
    rossi_metrics = metrics(rossi_records)
    hpa_metrics = metrics(hpa_records)
    rossi_abs_delta = np.abs([record.observation_delta for record in rossi_records])
    hpa_abs_delta = np.abs([record.observation_delta for record in hpa_records])
    return {
        "rossi_total_cost": rossi_metrics.total_cost,
        "hpa_total_cost": hpa_metrics.total_cost,
        "rossi_sla_violation_rate": rossi_metrics.sla_violation_rate,
        "hpa_sla_violation_rate": hpa_metrics.sla_violation_rate,
        "rossi_action_churn": rossi_metrics.action_churn,
        "hpa_action_churn": hpa_metrics.action_churn,
        "rossi_mean_abs_observation_delta": float(np.mean(rossi_abs_delta)),
        "hpa_mean_abs_observation_delta": float(np.mean(hpa_abs_delta)),
    }


def run_tasks(
    *,
    lags: tuple[int, ...],
    sequence: tuple[float, ...],
    offsets: tuple[int, ...],
    horizon: int,
    max_workers: int,
    update_on_lagged_observation: bool,
) -> dict[int, list[dict[str, object]]]:
    raw: dict[int, list[dict[str, object]]] = {lag: [] for lag in lags}
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for lag in lags:
            for seed_idx, offset in enumerate(offsets):
                rates = tuple(sequence[offset : offset + horizon])
                future = pool.submit(
                    run_p1_task,
                    lag,
                    rates,
                    update_on_lagged_observation=update_on_lagged_observation,
                )
                futures[future] = (lag, seed_idx)
        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            lag, seed_idx = futures[future]
            result = future.result()
            result["seed_index"] = seed_idx
            result["offset"] = offsets[seed_idx]
            raw[lag].append(result)
            completed += 1
            print(f"p1 {completed}/{total} lag={lag} seed={seed_idx}", flush=True)
    for lag in lags:
        raw[lag].sort(key=lambda item: int(item["seed_index"]))
    return raw


def summarize_cells(
    raw: dict[int, list[dict[str, object]]],
    *,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    cells = []
    for idx, (lag, entries) in enumerate(raw.items()):
        rossi_values = [float(entry["rossi_total_cost"]) for entry in entries]
        hpa_values = [float(entry["hpa_total_cost"]) for entry in entries]
        rossi_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in rossi_values)
        hpa_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in hpa_values)
        comparison = paired_result(hpa_metrics, rossi_metrics, seed=bootstrap_seed + idx)
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
            "rossi_sla_violation_rate": float(
                np.mean([float(entry["rossi_sla_violation_rate"]) for entry in entries])
            ),
            "hpa_sla_violation_rate": float(
                np.mean([float(entry["hpa_sla_violation_rate"]) for entry in entries])
            ),
            "rossi_action_churn": float(
                np.mean([float(entry["rossi_action_churn"]) for entry in entries])
            ),
            "hpa_action_churn": float(
                np.mean([float(entry["hpa_action_churn"]) for entry in entries])
            ),
            "rossi_mean_abs_observation_delta": float(
                np.mean(
                    [float(entry["rossi_mean_abs_observation_delta"]) for entry in entries]
                )
            ),
            "hpa_mean_abs_observation_delta": float(
                np.mean([float(entry["hpa_mean_abs_observation_delta"]) for entry in entries])
            ),
            "rossi_total_costs": rossi_values,
            "hpa_total_costs": hpa_values,
        }
        cells.append(cell)
    return cells


def p2_clean_cell_as_p1_clean(*, bootstrap_seed: int) -> dict[str, object]:
    data = json.loads(P2_RESULT_PATH.read_text(encoding="utf-8"))
    clean = next(cell for cell in data["p2"]["cells"] if cell["value"] == "inf")
    rossi_values = [float(value) for value in clean["rossi_total_costs"]]
    hpa_values = [float(value) for value in clean["hpa_total_costs"]]
    rossi_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in rossi_values)
    hpa_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in hpa_values)
    comparison = paired_result(hpa_metrics, rossi_metrics, seed=bootstrap_seed)
    return {
        "lag": 0,
        "rossi_mean_total_cost": float(np.mean(rossi_values)),
        "hpa_mean_total_cost": float(np.mean(hpa_values)),
        "delta_hpa_minus_rossi": comparison.mean_difference,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "p_less_than_zero": comparison.p_less_than_zero,
        "p_greater_than_zero": comparison.p_greater_than_zero,
        "outcome": outcome(comparison.ci_low, comparison.ci_high),
        "rossi_sla_violation_rate": float(clean["rossi_sla_violation_rate"]),
        "hpa_sla_violation_rate": float(clean["hpa_sla_violation_rate"]),
        "rossi_action_churn": None,
        "hpa_action_churn": None,
        "rossi_mean_abs_observation_delta": 0.0,
        "hpa_mean_abs_observation_delta": 0.0,
        "rossi_total_costs": rossi_values,
        "hpa_total_costs": hpa_values,
        "reused_from_prediction": "P2-Rossi alpha=inf clean online cell",
        "reused_from_mlflow_run_id": data["mlflow_run_id"],
        "reused_from_path": str(P2_RESULT_PATH.relative_to(ROOT)),
    }


def write_table(cells: list[dict[str, object]], run_id: str) -> None:
    table_dir = ROOT / "results" / "paper" / "rossi" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "rossi_p1_online_lag_sweep.csv"
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
        "rossi_sla_violation_rate",
        "hpa_sla_violation_rate",
        "rossi_action_churn",
        "hpa_action_churn",
        "rossi_mean_abs_observation_delta",
        "hpa_mean_abs_observation_delta",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)
    md_path = table_dir / "rossi_p1_online_lag_sweep.md"
    lines = [
        "# Rossi P1 Online Adaptive Observation-Lag Sweep",
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
    pdf_path = fig_dir / "rossi_p1_online_observation_lag.pdf"
    png_path = fig_dir / "rossi_p1_online_observation_lag.png"
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
    ax.set_title("Rossi P1 online observation lag")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    mlflow.log_artifact(str(pdf_path), artifact_path="paper/figures")
    mlflow.log_artifact(str(png_path), artifact_path="paper/figures")


def main() -> None:
    args = parse_args()
    lags = selected_lags(args.values)
    lags_to_run = tuple(lag for lag in lags if not (args.reuse_p2_clean and lag == 0))
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    offsets = profile_offsets(
        len(sequence),
        horizon=args.horizon,
        n=args.n_seeds,
        seed=args.base_seed,
    )
    params = {
        "method": "rossi_rlad",
        "prediction": "P1-Rossi",
        "protocol": "online_adaptive_model_based",
        "perturbation": "observation_lag",
        "values": args.values,
        "p1_anchor_lag_seconds": 10,
        "lags": list(lags),
        "lags_run": list(lags_to_run),
        "n_seeds": args.n_seeds,
        "seed_definition": "non-overlapping official slow-profile start offsets",
        "base_seed": args.base_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "horizon": args.horizon,
        "max_workers": args.max_workers,
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "profile_sha256": PROFILE_SHA256,
        "comparator": "source_threshold_hpa",
        "p1_comparator_semantics": "Option B: both Rossi and HPA observe lagged utilization",
        "online_update_semantics": (
            "Rossi online update and action selection use the delayed utilization bucket; "
            "replica count, CPU allocation, input rate, and realized cost remain current."
        ),
        "update_on_lagged_observation": args.update_on_lagged_observation,
        "reuse_p2_clean": args.reuse_p2_clean,
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name=f"rossi-online-p1-{args.values}",
        role="perturbation_sweep",
        params=params,
    ) as run:
        raw = run_tasks(
            lags=lags_to_run,
            sequence=sequence,
            offsets=offsets,
            horizon=args.horizon,
            max_workers=args.max_workers,
            update_on_lagged_observation=args.update_on_lagged_observation,
        )
        cells = summarize_cells(raw, bootstrap_seed=args.bootstrap_seed)
        if args.reuse_p2_clean:
            cells = [
                p2_clean_cell_as_p1_clean(bootstrap_seed=args.bootstrap_seed),
                *cells,
            ]
        cells.sort(key=lambda cell: int(cell["lag"]))
        anchor = next(cell for cell in cells if cell["lag"] == 10)
        result = {
            "mlflow_run_id": run.info.run_id,
            "prediction": "P1-Rossi",
            "anchor_lag_seconds": 10,
            "anchor_outcome": anchor["outcome"],
            "offsets": offsets,
            "p1": {"cells": cells},
            "params": params,
        }
        for cell in cells:
            lag = cell["lag"]
            mlflow.log_metric(f"lag_{lag}_delta_hpa_minus_rossi", cell["delta_hpa_minus_rossi"])
            mlflow.log_metric(f"lag_{lag}_ci_low", cell["ci_low"])
            mlflow.log_metric(f"lag_{lag}_ci_high", cell["ci_high"])
            mlflow.log_metric(f"lag_{lag}_rossi_sla_violation_rate", cell["rossi_sla_violation_rate"])
            mlflow.log_metric(f"lag_{lag}_hpa_sla_violation_rate", cell["hpa_sla_violation_rate"])
        mlflow.log_metric("anchor_delta_hpa_minus_rossi", anchor["delta_hpa_minus_rossi"])
        mlflow.log_metric("anchor_ci_low", anchor["ci_low"])
        mlflow.log_metric("anchor_ci_high", anchor["ci_high"])
        mlflow.log_metric("anchor_confirmed", 1.0 if anchor["outcome"] == "confirmed" else 0.0)
        out = ROOT / "results" / "rossi" / "p1_online_observation_lag_sweep.json"
        write_json_artifact(out, result, run_id=run.info.run_id)
        write_table(cells, run.info.run_id)
        write_figure(cells)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
