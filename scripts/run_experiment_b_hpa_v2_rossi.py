#!/usr/bin/env python3
"""Experiment B: production-grade HPA-v2 comparator for Rossi cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_common.stats import holm_bonferroni, paired_bootstrap_ci, sign_flip_pvalues
from cisose_common.tracking import start_run
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import HPAv2Controller, ThresholdHPAController
from cisose_rossi.evaluation import metrics
from cisose_rossi.perturbations import capped_pareto_cv2
from cisose_rossi.simulator import RladSimulator, ServiceState, StepRecord
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


EXPERIMENT_NAME = "cisose_rossi_v2_2"
PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "experiment_b"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
ROOT_FIG_DIR = ROOT / "figures"
ROOT_DATA_DIR = ROOT / "data"
ROSSI_P1_JSON = ROOT / "results" / "rossi" / "p1_online_observation_lag_sweep.json"
ROSSI_P2_JSON = ROOT / "results" / "rossi" / "p2_online_service_tail.json"
ROSSI_P3_JSON = ROOT / "results" / "rossi" / "p3_online_bucket_flip.json"

DEFAULT_HORIZON = DEFAULT_CONFIG.time_limit + 1
BASE_SEED = 20260520
BOOTSTRAP_SEED = 20260529
BLOCK_LENGTHS = (5, 10)
BOOTSTRAP_REPLICATES = 5000
SIGN_FLIP_REPLICATES = 100_000


@dataclass(frozen=True)
class CellSpec:
    key: str
    label: str
    perturbation: str
    anchor_value: str


CELLS = (
    CellSpec("clean", "Clean", "clean", "none"),
    CellSpec("p1", "P1 lag k=10", "p1_lag", "10"),
    CellSpec("p2", "P2 tail alpha=1.5", "p2_tail", "1.5"),
    CellSpec("p3", "P3 bucket-flip epsilon=0.05", "p3_bucket_flip", "0.05"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs()
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    offsets = profile_offsets(len(sequence), horizon=args.horizon, n=args.n_seeds, seed=args.base_seed)
    locked_rossi = load_locked_rossi_costs(offsets)
    sanity = run_sanity_checks(sequence, offsets[0], args.horizon)
    if not all(item["passed"] for item in sanity.values()):
        report_path = write_report(
            sanity=sanity,
            cells=[],
            family_tables={},
            costs_path=None,
            fig_paths=[],
            run_id=None,
            halted=True,
        )
        print("Experiment B halted: sanity check failure")
        print(str(report_path.relative_to(ROOT)))
        return

    if args.sanity_only:
        report_path = write_report(
            sanity=sanity,
            cells=[],
            family_tables={},
            costs_path=None,
            fig_paths=[],
            run_id=None,
            halted=False,
        )
        print("Experiment B sanity checks passed")
        print(str(report_path.relative_to(ROOT)))
        return

    params = {
        "method": "rossi_rlad",
        "experiment": "B",
        "protocol": "hpa_v2_comparator_sensitivity",
        "controller": "HPAv2Controller",
        "target_utilization": 0.50,
        "target_utilization_note": "representative production-grade user-specified target, not Kubernetes universal default",
        "sync_period_seconds": 15,
        "tolerance": 0.10,
        "scale_down_stabilization_seconds": 300,
        "scale_up_stabilization_seconds": 0,
        "n_seeds": args.n_seeds,
        "horizon": args.horizon,
        "base_seed": args.base_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "block_lengths": list(BLOCK_LENGTHS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sign_flip_replicates": SIGN_FLIP_REPLICATES,
        "rlad_repo_url": RLAD_REPO_URL,
        "rlad_commit": RLAD_COMMIT,
        "profile_sha256": PROFILE_SHA256,
        "p1_semantics": "Option B shared telemetry lag",
        "p3_semantics": "Option A Rossi bucket-flipped representation; HPA-v2 true continuous utilization",
        "rossi_cost_source": "locked canonical online Rossi runs; HPA-v2 newly evaluated on same offsets",
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="experiment-b-hpa-v2-rossi",
        role="experiment_b_hpa_v2_comparator",
        params=params,
        tags={"experiment": "B", "method": "rossi", "comparator": "hpa_v2"},
    ) as run:
        raw = run_tasks(
            sequence=sequence,
            offsets=offsets,
            horizon=args.horizon,
            max_workers=args.max_workers,
            locked_rossi=locked_rossi,
        )
        cells = summarize_cells(raw, bootstrap_seed=args.bootstrap_seed)
        family_tables = familywise(cells)
        costs_path = write_costs(raw)
        table_paths = write_tables(cells, family_tables)
        fig_paths = write_figures(sequence, offsets, raw)
        report_path = write_report(
            sanity=sanity,
            cells=cells,
            family_tables=family_tables,
            costs_path=costs_path,
            fig_paths=fig_paths,
            run_id=run.info.run_id,
            halted=False,
        )
        manifest_path = write_manifest(sanity, cells, family_tables, [costs_path, *table_paths], fig_paths, run.info.run_id)

        for name, check in sanity.items():
            mlflow.log_metric(f"sanity.{name}.passed", 1.0 if check["passed"] else 0.0)
            for metric_name, value in check["metrics"].items():
                mlflow.log_metric(f"sanity.{name}.{metric_name}", float(value))
        for cell in cells:
            key = cell["key"]
            mlflow.log_metric(f"{key}.delta_hpa_v2_minus_rossi", float(cell["delta_hpa_v2_minus_rossi"]))
            mlflow.log_metric(f"{key}.ci_low", float(cell["ci_low"]))
            mlflow.log_metric(f"{key}.ci_high", float(cell["ci_high"]))
            mlflow.log_metric(f"{key}.p_two_sided", float(cell["p_two_sided"]))
            mlflow.log_metric(f"{key}.p_one_sided_observed", float(cell["p_one_sided_observed"]))
            mlflow.log_metric(f"{key}.hpa_v2_mean_total_cost", float(cell["hpa_v2_mean_total_cost"]))
            mlflow.log_metric(f"{key}.rossi_mean_total_cost", float(cell["rossi_mean_total_cost"]))
        for block_len, tables in family_tables.items():
            for name, pvalue in tables["holm_two_sided"].items():
                mlflow.log_metric(f"L{block_len}.holm_two_sided.{name}", float(pvalue))
        for path in [costs_path, *table_paths, *fig_paths, report_path, manifest_path, Path(__file__), ROOT / "src" / "cisose_rossi" / "controllers.py"]:
            mlflow.log_artifact(str(path), artifact_path=_artifact_group(path))
        for protocol in (
            ROOT / "EXPERIMENT_B_hpa_baseline_rossi.md",
            ROOT / "EXPERIMENT_A_block_bootstrap_rossi.md",
            ROOT / "00_MASTER_coordination.md",
        ):
            if protocol.exists():
                mlflow.log_artifact(str(protocol), artifact_path="protocol/new_experiments")
        print(f"MLflow run: {run.info.run_id}")
        print(str(report_path.relative_to(ROOT)))
        print(str(costs_path.relative_to(ROOT)))


def _ensure_dirs() -> None:
    for directory in (OUT_DIR, TABLE_DIR, FIG_DIR, DATA_DIR, ROOT_FIG_DIR, ROOT_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def profile_offsets(sequence_len: int, *, horizon: int, n: int, seed: int) -> tuple[int, ...]:
    max_segments = sequence_len // horizon
    if max_segments < n:
        raise ValueError(f"need {n} non-overlapping windows, found {max_segments}")
    rng = np.random.default_rng(seed)
    segments = rng.choice(max_segments, size=n, replace=False)
    return tuple(int(segment * horizon) for segment in sorted(segments))


def load_locked_rossi_costs(offsets: tuple[int, ...]) -> dict[str, list[dict[str, object]]]:
    p1 = json.loads(ROSSI_P1_JSON.read_text(encoding="utf-8"))
    p2 = json.loads(ROSSI_P2_JSON.read_text(encoding="utf-8"))
    p3 = json.loads(ROSSI_P3_JSON.read_text(encoding="utf-8"))
    for path, payload in ((ROSSI_P1_JSON, p1), (ROSSI_P2_JSON, p2), (ROSSI_P3_JSON, p3)):
        locked_offsets = tuple(int(value) for value in payload["offsets"])
        if locked_offsets != offsets:
            raise ValueError(f"offset mismatch for {path}: locked offsets do not match Experiment B")
    return {
        "clean": _locked_cell(p2["p2"]["cells"], "value", "inf"),
        "p1": _locked_cell(p1["p1"]["cells"], "lag", 10),
        "p2": _locked_cell(p2["p2"]["cells"], "value", "1.5"),
        "p3": _locked_cell(p3["p3"]["cells"], "value", 0.05),
    }


def _locked_cell(cells: list[dict[str, object]], selector_key: str, selector_value: object) -> list[dict[str, object]]:
    cell = next(item for item in cells if str(item[selector_key]) == str(selector_value))
    rows = []
    for cost in cell["rossi_total_costs"]:
        rows.append(
            {
                "rossi_total_cost": float(cost),
                "rossi_sla_violation_rate": float(cell.get("rossi_sla_violation_rate", np.nan)),
                "rossi_action_churn": float(cell.get("rossi_action_churn", np.nan) or np.nan),
                "rossi_mean_replicas": float("nan"),
                "rossi_replica_std": float("nan"),
                "locked_rossi_source": "canonical_online_rossi_json",
            }
        )
    return rows


def run_sanity_checks(sequence: tuple[float, ...], offset: int, horizon: int) -> dict[str, dict[str, object]]:
    rates = tuple(sequence[offset : offset + horizon])
    hpa_clean = RladSimulator(DEFAULT_CONFIG).run(HPAv2Controller(DEFAULT_CONFIG), rates, horizon=horizon)
    threshold_clean = RladSimulator(DEFAULT_CONFIG).run(ThresholdHPAController(DEFAULT_CONFIG), rates, horizon=horizon)
    hpa_reps = np.asarray([record.replicas_before for record in hpa_clean], dtype=np.float64)
    threshold_reps = np.asarray([record.replicas_before for record in threshold_clean], dtype=np.float64)
    s1_std = float(np.std(hpa_reps))
    s1 = {
        "passed": s1_std < 1.0,
        "metrics": {
            "hpa_v2_replica_std": s1_std,
            "bundled_threshold_replica_std": float(np.std(threshold_reps)),
            "hpa_v2_replica_min": float(np.min(hpa_reps)),
            "hpa_v2_replica_max": float(np.max(hpa_reps)),
        },
    }

    max_distinct = 0
    for start in range(0, max(0, len(hpa_reps) - 14)):
        max_distinct = max(max_distinct, len(set(int(value) for value in hpa_reps[start : start + 15])))
    s2 = {"passed": max_distinct <= 2, "metrics": {"max_distinct_replicas_in_15_tick_window": float(max_distinct)}}

    synthetic_reps = synthetic_step_response()
    max_overload = max(synthetic_reps[:60])
    replica_at_250 = synthetic_reps[60 + 250]
    s3 = {
        "passed": replica_at_250 >= max_overload,
        "metrics": {
            "max_replicas_during_overload": float(max_overload),
            "replicas_250_ticks_after_underload": float(replica_at_250),
            "final_replicas": float(synthetic_reps[-1]),
        },
    }
    return {"S1_clean_stability": s1, "S2_sync_cadence": s2, "S3_stabilization": s3}


def synthetic_step_response() -> list[int]:
    controller = HPAv2Controller(DEFAULT_CONFIG)
    service = ServiceState(replicas=1, cpu=DEFAULT_CONFIG.initial_cpu, utilization=0.0)
    replicas = []
    for t in range(420):
        util = 0.9 if t < 60 else 0.2
        service.utilization = util
        controller.update(service, 0.0, 0.0)
        action = controller.pick_action(service, util)
        service.replicas += action.replica_delta
        replicas.append(service.replicas)
    return replicas


def run_tasks(
    *,
    sequence: tuple[float, ...],
    offsets: tuple[int, ...],
    horizon: int,
    max_workers: int,
    locked_rossi: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    raw = {cell.key: [] for cell in CELLS}
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for cell in CELLS:
            for seed_idx, offset in enumerate(offsets):
                rates = tuple(sequence[offset : offset + horizon])
                future = pool.submit(run_cell_task, cell.key, rates)
                futures[future] = (cell.key, seed_idx, offset)
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            key, seed_idx, offset = futures[future]
            result = future.result()
            result.update(locked_rossi[key][seed_idx])
            result["delta_hpa_v2_minus_rossi"] = (
                float(result["hpa_v2_total_cost"]) - float(result["rossi_total_cost"])
            )
            result["cell"] = key
            result["seed_index"] = seed_idx
            result["offset"] = offset
            raw[key].append(result)
            completed += 1
            print(f"experiment_b {completed}/{total} cell={key} seed={seed_idx}", flush=True)
    for key in raw:
        raw[key].sort(key=lambda item: int(item["seed_index"]))
    return raw


def run_cell_task(cell_key: str, rates: tuple[float, ...]) -> dict[str, object]:
    if cell_key == "clean":
        hpa_records = RladSimulator(DEFAULT_CONFIG).run(HPAv2Controller(DEFAULT_CONFIG), rates, horizon=len(rates))
    elif cell_key == "p1":
        hpa_records = RladSimulator(DEFAULT_CONFIG).run(
            HPAv2Controller(DEFAULT_CONFIG),
            rates,
            horizon=len(rates),
            observation_lag_steps=10,
        )
    elif cell_key == "p2":
        cv2 = capped_pareto_cv2(1.5)
        hpa_records = RladSimulator(DEFAULT_CONFIG, service_time_cv2=cv2).run(
            HPAv2Controller(DEFAULT_CONFIG),
            rates,
            horizon=len(rates),
        )
    elif cell_key == "p3":
        hpa_records = RladSimulator(DEFAULT_CONFIG).run(HPAv2Controller(DEFAULT_CONFIG), rates, horizon=len(rates))
    else:
        raise ValueError(cell_key)

    hpa_metrics = metrics(hpa_records)
    return {
        "hpa_v2_total_cost": hpa_metrics.total_cost,
        "hpa_v2_sla_violation_rate": hpa_metrics.sla_violation_rate,
        "hpa_v2_action_churn": hpa_metrics.action_churn,
        "hpa_v2_replica_std": float(np.std([record.replicas_before for record in hpa_records])),
        "hpa_v2_mean_replicas": float(np.mean([record.replicas_before for record in hpa_records])),
    }


def summarize_cells(raw: dict[str, list[dict[str, object]]], *, bootstrap_seed: int) -> list[dict[str, object]]:
    cells = []
    for idx, spec in enumerate(CELLS):
        entries = raw[spec.key]
        diffs = np.asarray([float(entry["delta_hpa_v2_minus_rossi"]) for entry in entries], dtype=np.float64)
        ci_low, ci_high = paired_bootstrap_ci(diffs, seed=bootstrap_seed + idx)
        p_less, p_greater = sign_flip_pvalues(diffs, seed=bootstrap_seed + 100 + idx)
        block = {
            block_len: block_analysis(diffs, block_len=block_len, seed=bootstrap_seed + 1000 * idx + block_len)
            for block_len in BLOCK_LENGTHS
        }
        cells.append(
            {
                "key": spec.key,
                "label": spec.label,
                "anchor_value": spec.anchor_value,
                "rossi_mean_total_cost": float(np.mean([float(entry["rossi_total_cost"]) for entry in entries])),
                "hpa_v2_mean_total_cost": float(np.mean([float(entry["hpa_v2_total_cost"]) for entry in entries])),
                "delta_hpa_v2_minus_rossi": float(np.mean(diffs)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_less_than_zero": float(p_less),
                "p_greater_than_zero": float(p_greater),
                "p_two_sided": min(1.0, 2.0 * min(float(p_less), float(p_greater))),
                "p_one_sided_observed": float(p_greater if float(np.mean(diffs)) >= 0 else p_less),
                "block": block,
                "hpa_v2_mean_replicas": float(np.mean([float(entry["hpa_v2_mean_replicas"]) for entry in entries])),
                "hpa_v2_replica_std": float(np.mean([float(entry["hpa_v2_replica_std"]) for entry in entries])),
                "rossi_mean_replicas": float(np.mean([float(entry["rossi_mean_replicas"]) for entry in entries])),
                "rossi_sla_violation_rate": float(np.mean([float(entry["rossi_sla_violation_rate"]) for entry in entries])),
                "hpa_v2_sla_violation_rate": float(np.mean([float(entry["hpa_v2_sla_violation_rate"]) for entry in entries])),
                "hpa_v2_wins": int(np.sum(diffs < 0.0)),
                "rossi_wins": int(np.sum(diffs > 0.0)),
            }
        )
    return cells


def block_analysis(diffs: np.ndarray, *, block_len: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    blocks = np.asarray([diffs[start : start + block_len] for start in range(0, n - block_len + 1)])
    n_blocks_sampled = math.ceil(n / block_len)
    sample_idx = rng.integers(0, len(blocks), size=(BOOTSTRAP_REPLICATES, n_blocks_sampled))
    boot_means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        boot_means[i] = float(np.mean(blocks[sample_idx[i]].reshape(-1)[:n]))
    nonoverlap = [diffs[start : min(start + block_len, n)] for start in range(0, n, block_len)]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(SIGN_FLIP_REPLICATES, len(nonoverlap)))
    null_means = np.empty(SIGN_FLIP_REPLICATES, dtype=np.float64)
    for i in range(SIGN_FLIP_REPLICATES):
        null_means[i] = sum(float(sign) * float(np.sum(block)) for sign, block in zip(signs[i], nonoverlap, strict=True)) / n
    observed = float(np.mean(diffs))
    tolerance = 1e-12 * max(1.0, abs(observed))
    p_two = (np.count_nonzero(np.abs(null_means) + tolerance >= abs(observed)) + 1.0) / (
        SIGN_FLIP_REPLICATES + 1.0
    )
    if observed >= 0.0:
        p_one = (np.count_nonzero(null_means + tolerance >= observed) + 1.0) / (
            SIGN_FLIP_REPLICATES + 1.0
        )
    else:
        p_one = (np.count_nonzero(null_means - tolerance <= observed) + 1.0) / (
            SIGN_FLIP_REPLICATES + 1.0
        )
    return {
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "p_two_sided": float(p_two),
        "p_one_sided_observed": float(p_one),
    }


def familywise(cells: list[dict[str, object]]) -> dict[int, dict[str, dict[str, float]]]:
    tables = {}
    for block_len in BLOCK_LENGTHS:
        p_two = {cell["key"]: float(cell["block"][block_len]["p_two_sided"]) for cell in cells}
        p_one = {cell["key"]: float(cell["block"][block_len]["p_one_sided_observed"]) for cell in cells}
        tables[block_len] = {
            "unadjusted_two_sided": p_two,
            "holm_two_sided": holm_bonferroni(p_two),
            "unadjusted_one_sided_observed": p_one,
            "holm_one_sided_observed": holm_bonferroni(p_one),
        }
    return tables


def write_costs(raw: dict[str, list[dict[str, object]]]) -> Path:
    path = DATA_DIR / "experiment_b_costs.csv"
    root_path = ROOT_DATA_DIR / "experiment_b_costs.csv"
    fields = [
        "cell",
        "seed_index",
        "offset",
        "rossi_total_cost",
        "hpa_v2_total_cost",
        "delta_hpa_v2_minus_rossi",
        "rossi_sla_violation_rate",
        "hpa_v2_sla_violation_rate",
        "rossi_action_churn",
        "hpa_v2_action_churn",
        "rossi_mean_replicas",
        "hpa_v2_mean_replicas",
        "rossi_replica_std",
        "hpa_v2_replica_std",
    ]
    rows = [entry for entries in raw.values() for entry in entries]
    for out in (path, root_path):
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return path


def write_tables(cells: list[dict[str, object]], family_tables: dict[int, dict[str, dict[str, float]]]) -> list[Path]:
    csv_path = TABLE_DIR / "experiment_b_results.csv"
    md_path = OUT_DIR / "experiment_b_results_table.md"
    fields = [
        "cell",
        "rossi_mean_total_cost",
        "hpa_v2_mean_total_cost",
        "delta_hpa_v2_minus_rossi",
        "ci_low",
        "ci_high",
        "p_two_sided",
        "p_one_sided_observed",
        "block_L10_ci_low",
        "block_L10_ci_high",
        "block_L10_p_two_sided",
        "block_L10_holm_two_sided",
        "block_L10_holm_one_sided_observed",
        "outcome",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            b10 = cell["block"][10]
            writer.writerow(
                {
                    "cell": cell["label"],
                    "rossi_mean_total_cost": cell["rossi_mean_total_cost"],
                    "hpa_v2_mean_total_cost": cell["hpa_v2_mean_total_cost"],
                    "delta_hpa_v2_minus_rossi": cell["delta_hpa_v2_minus_rossi"],
                    "ci_low": cell["ci_low"],
                    "ci_high": cell["ci_high"],
                    "p_two_sided": cell["p_two_sided"],
                    "p_one_sided_observed": cell["p_one_sided_observed"],
                    "block_L10_ci_low": b10["ci_low"],
                    "block_L10_ci_high": b10["ci_high"],
                    "block_L10_p_two_sided": b10["p_two_sided"],
                    "block_L10_holm_two_sided": family_tables[10]["holm_two_sided"][cell["key"]],
                    "block_L10_holm_one_sided_observed": family_tables[10]["holm_one_sided_observed"][cell["key"]],
                    "outcome": cell_outcome(cell),
                }
            )
    lines = [
        "# Experiment B HPA-v2 Results Table",
        "",
        "| Cell | Rossi cost | HPA-v2 cost | Δ HPA-v2-Rossi | iid 95% CI | block L=10 95% CI | Holm p2s L=10 | Outcome |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        b10 = cell["block"][10]
        lines.append(
            "| {label} | {rossi:.3f} | {hpa:.3f} | {delta:.3f} | [{lo:.3f}, {hi:.3f}] | "
            "[{blo:.3f}, {bhi:.3f}] | {holm:.4g} | {outcome} |".format(
                label=cell["label"],
                rossi=float(cell["rossi_mean_total_cost"]),
                hpa=float(cell["hpa_v2_mean_total_cost"]),
                delta=float(cell["delta_hpa_v2_minus_rossi"]),
                lo=float(cell["ci_low"]),
                hi=float(cell["ci_high"]),
                blo=float(b10["ci_low"]),
                bhi=float(b10["ci_high"]),
                holm=float(family_tables[10]["holm_two_sided"][cell["key"]]),
                outcome=cell_outcome(cell),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def write_figures(
    sequence: tuple[float, ...],
    offsets: tuple[int, ...],
    raw: dict[str, list[dict[str, object]]],
) -> list[Path]:
    p1_entries = raw["p1"]
    median_entry = sorted(p1_entries, key=lambda item: float(item["delta_hpa_v2_minus_rossi"]))[len(p1_entries) // 2]
    offset = int(median_entry["offset"])
    rates = tuple(sequence[offset : offset + DEFAULT_HORIZON])
    paths = []
    paths.extend(plot_hpa_clean_vs_lag(rates))
    paths.extend(plot_threshold_vs_hpa_lag(rates))
    return paths


def plot_hpa_clean_vs_lag(rates: tuple[float, ...]) -> list[Path]:
    clean = RladSimulator(DEFAULT_CONFIG).run(HPAv2Controller(DEFAULT_CONFIG), rates, horizon=len(rates))
    lag = RladSimulator(DEFAULT_CONFIG).run(
        HPAv2Controller(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
        observation_lag_steps=10,
    )
    return _plot_two_traces(
        clean,
        lag,
        titles=("HPA-v2 clean", "HPA-v2 lag k=10"),
        filename="hpa_v2_clean_vs_k10_failure_trace",
    )


def plot_threshold_vs_hpa_lag(rates: tuple[float, ...]) -> list[Path]:
    threshold = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
        observation_lag_steps=10,
        observation_applies_to_update=True,
    )
    hpa = RladSimulator(DEFAULT_CONFIG).run(
        HPAv2Controller(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
        observation_lag_steps=10,
    )
    return _plot_two_traces(
        threshold,
        hpa,
        titles=("Bundled threshold lag k=10", "HPA-v2 lag k=10"),
        filename="threshold_vs_hpa_v2_under_lag",
    )


def _plot_two_traces(
    left: tuple[StepRecord, ...],
    right: tuple[StepRecord, ...],
    *,
    titles: tuple[str, str],
    filename: str,
) -> list[Path]:
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
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 3.9), sharex=True)
    for col, (records, title) in enumerate(((left, titles[0]), (right, titles[1]))):
        t = np.asarray([record.time for record in records], dtype=float)
        reps = np.asarray([record.replicas_before for record in records], dtype=float)
        true_util = np.asarray([record.utilization for record in records], dtype=float)
        obs_util = np.asarray([record.observed_utilization for record in records], dtype=float)
        axes[0, col].plot(t, reps, color="#2f6278", linewidth=0.9)
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Replicas")
        axes[0, col].set_ylim(0.5, DEFAULT_CONFIG.max_replication + 0.5)
        axes[1, col].plot(t, true_util, color="#333333", linewidth=0.8, label="true")
        axes[1, col].plot(t, obs_util, color="#b00020", linewidth=0.7, alpha=0.8, label="observed")
        axes[1, col].set_ylabel("Utilization")
        axes[1, col].set_xlabel("Tick")
        axes[1, col].set_ylim(0.0, max(1.2, float(np.max(true_util) * 1.05)))
        axes[1, col].legend(frameon=False, loc="upper right", fontsize=7)
    for ax in axes.reshape(-1):
        ax.grid(True, color="#dddddd", linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    paths = []
    for directory in (FIG_DIR, ROOT_FIG_DIR):
        pdf = directory / f"{filename}.pdf"
        png = directory / f"{filename}.png"
        fig.savefig(pdf)
        fig.savefig(png, dpi=300)
        paths.extend([pdf, png])
    plt.close(fig)
    return paths


def write_report(
    *,
    sanity: dict[str, dict[str, object]],
    cells: list[dict[str, object]],
    family_tables: dict[int, dict[str, dict[str, float]]],
    costs_path: Path | None,
    fig_paths: list[Path],
    run_id: str | None,
    halted: bool,
) -> Path:
    path = OUT_DIR / "experiment_b_results.md"
    root_copy = ROOT / "experiment_b_results.md"
    lines = [
        "# Experiment B Results — HPA-v2 Comparator for Rossi",
        "",
        f"MLflow run: `{run_id or 'not-started'}`",
        "",
        "HPA-v2 target utilisation is 50%, treated as a representative production-grade user-specified configuration, not a Kubernetes universal default.",
        "P1 applies shared telemetry lag to Rossi and HPA-v2. P3 applies bucket flipping only to Rossi's discretised Q-table representation; HPA-v2 reads true continuous utilisation.",
        "Rossi costs are loaded from the locked canonical online Rossi runs on the same 30 offsets; only the new HPA-v2 comparator is newly evaluated here.",
        "",
        "## Sanity Checks",
        "",
        "| Check | Passed | Key metrics |",
        "|---|---:|---|",
    ]
    for name, check in sanity.items():
        metric_text = ", ".join(f"{k}={float(v):.4g}" for k, v in check["metrics"].items())
        lines.append(f"| {name} | {check['passed']} | {metric_text} |")
    if halted or not cells:
        lines.append("")
        lines.append("Experiment B did not proceed beyond sanity checks.")
    else:
        lines.extend(
            [
                "",
                "## Results",
                "",
                "| Cell | Rossi cost | HPA-v2 cost | Δ HPA-v2-Rossi | iid 95% CI | block L=10 95% CI | Holm p2s L=10 | Verdict |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for cell in cells:
            b10 = cell["block"][10]
            lines.append(
                "| {label} | {rossi:.3f} | {hpa:.3f} | {delta:.3f} | [{lo:.3f}, {hi:.3f}] | "
                "[{blo:.3f}, {bhi:.3f}] | {holm:.4g} | {outcome} |".format(
                    label=cell["label"],
                    rossi=float(cell["rossi_mean_total_cost"]),
                    hpa=float(cell["hpa_v2_mean_total_cost"]),
                    delta=float(cell["delta_hpa_v2_minus_rossi"]),
                    lo=float(cell["ci_low"]),
                    hi=float(cell["ci_high"]),
                    blo=float(b10["ci_low"]),
                    bhi=float(b10["ci_high"]),
                    holm=float(family_tables[10]["holm_two_sided"][cell["key"]]),
                    outcome=cell_outcome(cell),
                )
            )
        lines.extend(["", "## Interpretation", "", experiment_b_interpretation(cells)])
    lines.extend(["", "## Artifact Paths", ""])
    if costs_path is not None:
        lines.append(f"- Costs CSV: `{costs_path.relative_to(ROOT)}`")
        lines.append(f"- Costs CSV copy: `data/experiment_b_costs.csv`")
    for fig_path in fig_paths:
        lines.append(f"- Figure: `{fig_path.relative_to(ROOT)}`")
    report = "\n".join(lines) + "\n"
    path.write_text(report, encoding="utf-8")
    root_copy.write_text(report, encoding="utf-8")
    return path


def experiment_b_interpretation(cells: list[dict[str, object]]) -> str:
    by_key = {cell["key"]: cell for cell in cells}
    p1 = by_key["p1"]
    p1_delta = float(p1["delta_hpa_v2_minus_rossi"])
    if -200 <= p1_delta <= 400:
        p1_case = "H_P1a prevails: HPA-v2 does not reproduce the bundled-threshold lag collapse."
    elif 300 <= p1_delta <= 900:
        p1_case = "H_P1b prevails: HPA-v2 remains substantially worse than Rossi under lag, though by less than the bundled threshold."
    else:
        p1_case = "P1 falls outside both pre-registered magnitude bands and should be described as an out-of-band sensitivity result."
    clean = cell_outcome(by_key["clean"])
    p2 = cell_outcome(by_key["p2"])
    p3 = cell_outcome(by_key["p3"])
    return (
        f"{p1_case} Clean verdict: {clean}. P2 verdict: {p2}. P3 verdict: {p3}. "
        "The sign convention is Δ = cost(HPA-v2) - cost(Rossi); negative values favor HPA-v2 and positive values favor Rossi. "
        "The original bundled-threshold P1 anchor was approximately +965, so the HPA-v2 P1 value should be interpreted as strong evidence that the dramatic bundled-threshold lag collapse does not generalise to a stabilization-windowed comparator. "
        "The block-sign Holm values are conservative because L=10 supplies only three sign blocks; the paired iid CIs and point estimates are the more informative magnitude summaries for this follow-up sensitivity."
    )


def cell_outcome(cell: dict[str, object]) -> str:
    low = float(cell["ci_low"])
    high = float(cell["ci_high"])
    key = str(cell["key"])
    if key == "p1":
        delta = float(cell["delta_hpa_v2_minus_rossi"])
        if -200 <= delta <= 400:
            return "H_P1a"
        if 300 <= delta <= 900:
            return "H_P1b"
        return "out_of_band"
    if high < 0.0:
        return "HPA-v2 dominates Rossi"
    if low > 0.0:
        return "Rossi dominates HPA-v2"
    return "inconclusive"


def write_manifest(
    sanity: dict[str, dict[str, object]],
    cells: list[dict[str, object]],
    family_tables: dict[int, dict[str, dict[str, float]]],
    artifact_paths: Iterable[Path],
    fig_paths: Iterable[Path],
    run_id: str,
) -> Path:
    path = OUT_DIR / "experiment_b_manifest.json"
    payload = {
        "experiment": "B",
        "mlflow_run_id": run_id,
        "sanity": sanity,
        "cells": _strip_blocks_for_json(cells),
        "family_tables": family_tables,
        "artifacts": [str(path.relative_to(ROOT)) for path in artifact_paths],
        "figures": [str(path.relative_to(ROOT)) for path in fig_paths],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _strip_blocks_for_json(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    cleaned = []
    for cell in cells:
        out = dict(cell)
        out["block"] = {str(k): v for k, v in out["block"].items()}
        cleaned.append(out)
    return cleaned


def _artifact_group(path: Path) -> str:
    if path.suffix == ".csv":
        return "experiment_b/tables" if "tables" in path.parts else "experiment_b/data"
    if path.suffix in {".pdf", ".png"}:
        return "experiment_b/figures"
    if path.suffix == ".py":
        return "experiment_b/code"
    return "experiment_b"


if __name__ == "__main__":
    main()
