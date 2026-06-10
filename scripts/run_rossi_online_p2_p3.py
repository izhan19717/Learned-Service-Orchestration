#!/usr/bin/env python3
"""Run paper-faithful online Rossi P2/P3 perturbation sweeps."""

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

from cisose_common.tracking import start_run, write_json_artifact
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import ModelBasedController, ThresholdHPAController
from cisose_rossi.evaluation import RossiMetrics, metrics, paired_result
from cisose_rossi.perturbations import capped_pareto_cv2, minimum_bucket_flip_utilization
from cisose_rossi.simulator import RladSimulator
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_rossi_v2_2"
PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
P2_RESULT_PATH = ROOT / "results" / "rossi" / "p2_online_service_tail.json"
P2_ALPHAS = ("inf", "3.0", "2.0", "1.5", "1.2")
P3_EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.10)
DEFAULT_HORIZON = DEFAULT_CONFIG.time_limit + 1
BASE_SEED = 20260520
BOOTSTRAP_SEED = 20260521


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=("p2", "p3", "both"), default="both")
    parser.add_argument("--values", choices=("anchor", "anchor_plus_clean", "full"), default="full")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--reuse-p2-clean",
        action="store_true",
        help="For P3 anchor_plus_clean, reuse the completed P2 alpha=inf clean cell.",
    )
    return parser.parse_args()


def alpha_to_float(alpha: str) -> float:
    return float("inf") if alpha == "inf" else float(alpha)


def profile_offsets(sequence_len: int, *, horizon: int, n: int, seed: int) -> tuple[int, ...]:
    max_segments = sequence_len // horizon
    if max_segments < n:
        raise ValueError(f"need {n} non-overlapping windows, found {max_segments}")
    rng = np.random.default_rng(seed)
    segments = rng.choice(max_segments, size=n, replace=False)
    return tuple(int(segment * horizon) for segment in sorted(segments))


def outcome(ci_low: float, ci_high: float) -> str:
    if ci_high < 0.0:
        return "confirmed"
    if ci_low > 0.0:
        return "falsified"
    return "inconclusive"


def run_p2_task(alpha: str, rates: tuple[float, ...]) -> dict[str, object]:
    cv2 = capped_pareto_cv2(alpha_to_float(alpha))
    rossi_records = RladSimulator(DEFAULT_CONFIG, service_time_cv2=cv2).run(
        ModelBasedController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
    )
    hpa_records = RladSimulator(DEFAULT_CONFIG, service_time_cv2=cv2).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
    )
    return {
        "rossi_total_cost": metrics(rossi_records).total_cost,
        "hpa_total_cost": metrics(hpa_records).total_cost,
        "rossi_sla_violation_rate": metrics(rossi_records).sla_violation_rate,
        "hpa_sla_violation_rate": metrics(hpa_records).sla_violation_rate,
        "service_time_cv2": cv2,
    }


def run_p3_task(epsilon: float, rates: tuple[float, ...]) -> dict[str, object]:
    if epsilon == 0.0:
        rossi_records = RladSimulator(DEFAULT_CONFIG).run(
            ModelBasedController(DEFAULT_CONFIG),
            rates,
            horizon=len(rates),
        )
        hpa_records = RladSimulator(DEFAULT_CONFIG).run(
            ThresholdHPAController(DEFAULT_CONFIG),
            rates,
            horizon=len(rates),
        )
        return {
            "rossi_total_cost": metrics(rossi_records).total_cost,
            "hpa_total_cost": metrics(hpa_records).total_cost,
            "rossi_sla_violation_rate": metrics(rossi_records).sla_violation_rate,
            "hpa_sla_violation_rate": metrics(hpa_records).sla_violation_rate,
            "attack_fraction": 0.0,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }

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
    hpa_records = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
    )
    deltas = np.asarray([record.observation_delta for record in rossi_records], dtype=np.float64)
    abs_deltas = np.abs(deltas)
    return {
        "rossi_total_cost": metrics(rossi_records).total_cost,
        "hpa_total_cost": metrics(hpa_records).total_cost,
        "rossi_sla_violation_rate": metrics(rossi_records).sla_violation_rate,
        "hpa_sla_violation_rate": metrics(hpa_records).sla_violation_rate,
        "attack_fraction": float(np.mean(abs_deltas > 1e-12)),
        "mean_abs_delta": float(np.mean(abs_deltas)),
        "max_abs_delta": float(np.max(abs_deltas)),
    }


def summarize_cells(
    raw: dict[object, list[dict[str, object]]],
    *,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    cells = []
    for idx, (value, entries) in enumerate(raw.items()):
        rossi_values = [float(entry["rossi_total_cost"]) for entry in entries]
        hpa_values = [float(entry["hpa_total_cost"]) for entry in entries]
        rossi_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in rossi_values)
        hpa_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in hpa_values)
        comparison = paired_result(hpa_metrics, rossi_metrics, seed=bootstrap_seed + idx)
        cell = {
            "value": value,
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
            "rossi_total_costs": rossi_values,
            "hpa_total_costs": hpa_values,
        }
        for key in ("service_time_cv2", "attack_fraction", "mean_abs_delta", "max_abs_delta"):
            if key in entries[0]:
                cell[key] = float(np.mean([float(entry[key]) for entry in entries]))
        cells.append(cell)
    return cells


def p2_clean_cell_as_p3_clean(*, bootstrap_seed: int) -> dict[str, object]:
    data = json.loads(P2_RESULT_PATH.read_text(encoding="utf-8"))
    clean = next(cell for cell in data["p2"]["cells"] if cell["value"] == "inf")
    rossi_values = [float(value) for value in clean["rossi_total_costs"]]
    hpa_values = [float(value) for value in clean["hpa_total_costs"]]
    rossi_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in rossi_values)
    hpa_metrics = tuple(RossiMetrics(value, 0.0, 0.0, 0) for value in hpa_values)
    comparison = paired_result(hpa_metrics, rossi_metrics, seed=bootstrap_seed)
    return {
        "value": 0.0,
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
        "attack_fraction": 0.0,
        "mean_abs_delta": 0.0,
        "max_abs_delta": 0.0,
        "rossi_total_costs": rossi_values,
        "hpa_total_costs": hpa_values,
        "reused_from_prediction": "P2-Rossi alpha=inf clean online cell",
        "reused_from_mlflow_run_id": data["mlflow_run_id"],
        "reused_from_path": str(P2_RESULT_PATH.relative_to(ROOT)),
    }


def run_tasks(
    *,
    task_name: str,
    values: tuple[object, ...],
    sequence: tuple[float, ...],
    offsets: tuple[int, ...],
    horizon: int,
    max_workers: int,
) -> dict[object, list[dict[str, object]]]:
    raw: dict[object, list[dict[str, object]]] = {value: [] for value in values}
    fn = run_p2_task if task_name == "p2" else run_p3_task
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for value in values:
            for seed_idx, offset in enumerate(offsets):
                rates = tuple(sequence[offset : offset + horizon])
                future = pool.submit(fn, value, rates)
                futures[future] = (value, seed_idx)
        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            value, seed_idx = futures[future]
            result = future.result()
            result["seed_index"] = seed_idx
            result["offset"] = offsets[seed_idx]
            raw[value].append(result)
            completed += 1
            print(f"{task_name} {completed}/{total} value={value} seed={seed_idx}", flush=True)
    for value in values:
        raw[value].sort(key=lambda item: int(item["seed_index"]))
    return raw


def write_table(name: str, cells: list[dict[str, object]], run_id: str) -> None:
    table_dir = ROOT / "results" / "paper" / "rossi" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"rossi_{name}_online.csv"
    fieldnames = [
        "value",
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
        "service_time_cv2",
        "attack_fraction",
        "mean_abs_delta",
        "max_abs_delta",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)
    md_path = table_dir / f"rossi_{name}_online.md"
    lines = [
        f"# Rossi {name.upper()} Online Adaptive Sweep",
        "",
        f"MLflow run: `{run_id}`",
        "",
        "| Value | Rossi cost | HPA cost | Delta HPA-Rossi | 95% CI | Outcome |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        lines.append(
            "| {value} | {rossi_mean_total_cost:.6g} | {hpa_mean_total_cost:.6g} | "
            "{delta_hpa_minus_rossi:.6g} | [{ci_low:.6g}, {ci_high:.6g}] | {outcome} |".format(
                **cell
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(csv_path), artifact_path="paper/tables")
    mlflow.log_artifact(str(md_path), artifact_path="paper/tables")


def write_figure(name: str, cells: list[dict[str, object]]) -> None:
    fig_dir = ROOT / "results" / "paper" / "rossi" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f"rossi_{name}_online.pdf"
    png_path = fig_dir / f"rossi_{name}_online.png"
    x = np.arange(len(cells), dtype=float)
    labels = [str(cell["value"]) for cell in cells]
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
    if name == "p2":
        anchor_label = "1.5"
        ax.set_xlabel("Pareto service-time alpha")
        ax.set_title("Rossi P2 service-time tail")
    else:
        anchor_label = "0.05"
        ax.set_xlabel("Bucket-flip epsilon")
        ax.set_title("Rossi P3 bucket flip")
    if anchor_label in labels:
        ax.axvline(labels.index(anchor_label), color="#b00020", linewidth=0.9, linestyle="--")
    ax.fill_between(x, ci_low, ci_high, color="#9ecae9", alpha=0.45, linewidth=0)
    ax.plot(x, deltas, marker="o", color="#1f77b4", linewidth=1.4)
    ax.set_ylabel("Total cost(HPA) - total cost(Rossi)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
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
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    offsets = profile_offsets(
        len(sequence),
        horizon=args.horizon,
        n=args.n_seeds,
        seed=args.base_seed,
    )
    if args.values == "anchor":
        p2_values = ("1.5",)
        p3_values = (0.05,)
    elif args.values == "anchor_plus_clean":
        p2_values = ("inf", "1.5")
        p3_values = (0.05,) if args.reuse_p2_clean else (0.0, 0.05)
    else:
        p2_values = P2_ALPHAS
        p3_values = P3_EPSILONS
    params = {
        "method": "rossi_rlad",
        "protocol": "online_adaptive_model_based",
        "which": args.which,
        "values": args.values,
        "n_seeds": args.n_seeds,
        "horizon": args.horizon,
        "max_workers": args.max_workers,
        "seed_definition": "non-overlapping official slow-profile start offsets",
        "base_seed": args.base_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "profile_sha256": PROFILE_SHA256,
        "p2_tail_mapping": "mean-preserving capped Pareto CV^2, cap_ratio=100",
        "p3_comparator_semantics": "Option A: Rossi perturbed, HPA true utilization",
        "reuse_p2_clean": args.reuse_p2_clean,
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name=f"rossi-online-{args.which}-{args.values}",
        role="perturbation_sweep",
        params=params,
    ) as run:
        result: dict[str, object] = {
            "mlflow_run_id": run.info.run_id,
            "params": params,
            "offsets": offsets,
        }
        if args.which in ("p2", "both"):
            raw = run_tasks(
                task_name="p2",
                values=p2_values,
                sequence=sequence,
                offsets=offsets,
                horizon=args.horizon,
                max_workers=args.max_workers,
            )
            cells = summarize_cells(raw, bootstrap_seed=args.bootstrap_seed)
            result["p2"] = {"cells": cells}
            write_json_artifact(ROOT / "results" / "rossi" / "p2_online_service_tail.json", result, run_id=run.info.run_id)
            write_table("p2", cells, run.info.run_id)
            write_figure("p2", cells)
            anchor = next(cell for cell in cells if cell["value"] == "1.5")
            mlflow.log_metric("p2_anchor_delta_hpa_minus_rossi", anchor["delta_hpa_minus_rossi"])
            mlflow.log_metric("p2_anchor_ci_low", anchor["ci_low"])
            mlflow.log_metric("p2_anchor_ci_high", anchor["ci_high"])
        if args.which in ("p3", "both"):
            raw = run_tasks(
                task_name="p3",
                values=p3_values,
                sequence=sequence,
                offsets=offsets,
                horizon=args.horizon,
                max_workers=args.max_workers,
            )
            cells = summarize_cells(raw, bootstrap_seed=args.bootstrap_seed + 1000)
            if args.reuse_p2_clean:
                cells = [
                    p2_clean_cell_as_p3_clean(bootstrap_seed=args.bootstrap_seed + 1000),
                    *cells,
                ]
            result["p3"] = {"cells": cells}
            write_json_artifact(ROOT / "results" / "rossi" / "p3_online_bucket_flip.json", result, run_id=run.info.run_id)
            write_table("p3", cells, run.info.run_id)
            write_figure("p3", cells)
            anchor = next(cell for cell in cells if math.isclose(float(cell["value"]), 0.05))
            mlflow.log_metric("p3_anchor_delta_hpa_minus_rossi", anchor["delta_hpa_minus_rossi"])
            mlflow.log_metric("p3_anchor_ci_low", anchor["ci_low"])
            mlflow.log_metric("p3_anchor_ci_high", anchor["ci_high"])
            mlflow.log_metric("p3_anchor_attack_fraction", anchor.get("attack_fraction", 0.0))
        write_json_artifact(
            ROOT / "results" / "rossi" / f"online_{args.which}_{args.values}_combined.json",
            result,
            run_id=run.info.run_id,
        )
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
