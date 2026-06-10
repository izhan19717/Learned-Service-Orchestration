#!/usr/bin/env python3
"""E1 DeepRM perturbation-magnitude sweep."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch

from cisose_common.tracking import start_run, write_json_artifact
from cisose_deeprm.evaluation import (
    paired_bootstrap_ci,
    paired_result,
    run_adversarial_policy_episode,
    run_episode,
    run_lagged_policy_episode,
    run_lagged_scheduler_episode,
    sign_flip_pvalues,
)
from cisose_deeprm.model import load_checkpoint
from cisose_deeprm.protocol import DeepRMConfig, PRIMARY_LOAD
from cisose_deeprm.schedulers import SJFScheduler, TetrisScheduler
from cisose_deeprm.workload import generate_trace


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_e1_magnitude_sweep"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "e1_magnitude_sweep" / "deeprm"
TABLE_DIR = OUT_DIR / "tables"
DATA_DIR = OUT_DIR / "data"
CELL_DIR = DATA_DIR / "cells"

LAGS = (0, 1, 2, 5, 10, 20, 50)
ALPHAS = (2.5, 2.0, 1.75, 1.5, 1.3, 1.1)
EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
COMPARATORS = (TetrisScheduler(source_dot=True), SJFScheduler())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/checkpoints/author_source_rescue/load_0.7/policy_final.pt"),
    )
    parser.add_argument("--load", type=float, default=PRIMARY_LOAD)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--trace-jobs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--policy-seed", type=int, default=20260520)
    parser.add_argument("--policy-mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--censor-max-step-failures", action="store_true")
    return parser.parse_args()


def trace_seeds(seed: int, num_seeds: int) -> list[int]:
    seq = np.random.SeedSequence(seed)
    return [int(child.generate_state(1)[0]) for child in seq.spawn(num_seeds)]


def cell_generator(policy_seed: int, cell_offset: int) -> torch.Generator:
    seed = int(np.random.SeedSequence([policy_seed, cell_offset]).generate_state(1)[0])
    return torch.Generator().manual_seed(seed)


def tail_config(config: DeepRMConfig, alpha: float) -> DeepRMConfig:
    planning_horizon = max(config.time_horizon, config.tail_x_max + (0 if config.max_start_inclusive else 1))
    return DeepRMConfig(**{**config.__dict__, "planning_horizon": planning_horizon})


def make_traces(*, load: float, jobs: int, seeds: list[int], config: DeepRMConfig, alpha: float) -> tuple:
    return tuple(generate_trace(num_jobs=jobs, rate=load, seed=seed, config=config, tail_alpha=alpha) for seed in seeds)


def summarize_cell(deep_metrics, comparator_metrics, *, seed: int) -> dict[str, object]:
    comparison = paired_result(comparator_metrics, deep_metrics, seed=seed)
    return {
        "status": "complete",
        "deep_rm_mean_slowdown": float(np.mean([m.mean_slowdown for m in deep_metrics])),
        "comparator_mean_slowdown": float(np.mean([m.mean_slowdown for m in comparator_metrics])),
        "comparison": asdict(comparison),
        "deep_rm_metrics": [asdict(m) for m in deep_metrics],
        "comparator_metrics": [asdict(m) for m in comparator_metrics],
    }


def instability_cell(
    *,
    reason: str,
    max_steps: int,
    comparator_metrics=None,
    deep_failure: dict[str, object] | None = None,
    comparator_failure: dict[str, object] | None = None,
) -> dict[str, object]:
    cell: dict[str, object] = {
        "status": "noncompletion",
        "reason": reason,
        "max_steps": max_steps,
        "deep_failure": deep_failure,
        "comparator_failure": comparator_failure,
    }
    if comparator_metrics is not None:
        cell["comparator_mean_slowdown"] = float(np.mean([m.mean_slowdown for m in comparator_metrics]))
        cell["comparator_metrics"] = [asdict(m) for m in comparator_metrics]
    return cell


def cell_path(key: str) -> Path:
    safe = key.replace(":", "__").replace(".", "_")
    return CELL_DIR / f"{safe}.json"


def load_cell(key: str) -> dict[str, object] | None:
    path = cell_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cell(key: str, cell: dict[str, object]) -> None:
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    cell_path(key).write_text(json.dumps(cell, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def metric_key(key: str) -> str:
    return key.replace("*", "star")


def statistic_seed_for_key(key: str, base_seed: int) -> int:
    _, curve, _, magnitude = key.split(":", 3)
    if curve == "p1_lag":
        return base_seed + 10_000 + LAGS.index(int(magnitude))
    if curve == "p2_tail":
        return base_seed + 20_000 + _float_index(ALPHAS, float(magnitude))
    if curve == "p3_fgsm":
        return base_seed + 30_000 + _float_index(EPSILONS, float(magnitude))
    raise ValueError(f"unknown DeepRM E1 curve in key {key!r}")


def _float_index(values: tuple[float, ...], target: float) -> int:
    for idx, value in enumerate(values):
        if np.isclose(value, target, rtol=0.0, atol=1e-12):
            return idx
    raise ValueError(f"{target!r} not in {values!r}")


def refresh_comparison_seed(cell: dict[str, object], *, seed: int) -> dict[str, object]:
    if cell.get("status") != "complete" or "comparison" not in cell:
        return cell
    differences = tuple(float(value) for value in cell["comparison"]["differences"])
    ci_low, ci_high = paired_bootstrap_ci(differences, seed=seed)
    p_less, p_greater = sign_flip_pvalues(differences, seed=seed + 1)
    cell["comparison"] = {
        "differences": differences,
        "mean_difference": float(np.mean(np.asarray(differences, dtype=np.float64))),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_less_than_zero": p_less,
        "p_greater_than_zero": p_greater,
    }
    return cell


def evaluate_many(label: str, traces, evaluator, *, censor: bool, max_steps: int):
    metrics = []
    for idx, trace in enumerate(traces):
        try:
            metrics.append(evaluator(trace))
        except RuntimeError as exc:
            if censor and "exceeded max_steps" in str(exc):
                return None, {
                    "label": label,
                    "seed_index": idx,
                    "reason": str(exc),
                    "max_steps": max_steps,
                }
            raise
    return tuple(metrics), None


def holm_by_curve(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    from cisose_common.stats import holm_bonferroni

    grouped_less: dict[str, dict[str, float]] = {}
    grouped_greater: dict[str, dict[str, float]] = {}
    for key, cell in cells.items():
        if cell.get("status") != "complete" or "comparison" not in cell:
            continue
        curve = ":".join(key.split(":")[:3])
        grouped_less.setdefault(curve, {})[key] = float(cell["comparison"]["p_less_than_zero"])
        grouped_greater.setdefault(curve, {})[key] = float(cell["comparison"]["p_greater_than_zero"])
    return {
        "less_than_zero": {curve: holm_bonferroni(vals) for curve, vals in grouped_less.items()},
        "greater_than_zero": {curve: holm_bonferroni(vals) for curve, vals in grouped_greater.items()},
    }


def write_outputs(cells: dict[str, dict[str, object]], holm: dict[str, object], run_id: str, params: dict[str, object]) -> None:
    for directory in (OUT_DIR, TABLE_DIR, DATA_DIR, CELL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "e1_deeprm_magnitude_sweep.json"
    csv_path = TABLE_DIR / "e1_deeprm_magnitude_sweep.csv"
    raw_csv = DATA_DIR / "e1_deeprm_per_seed.csv"
    md_path = OUT_DIR / "e1_deeprm_magnitude_sweep.md"

    rows = []
    raw_rows = []
    for key, cell in cells.items():
        method, curve, comparator, magnitude = key.split(":", 3)
        comparison = cell.get("comparison", {})
        holm_key = f"{method}:{curve}:{comparator}"
        rows.append(
            {
                "method": method,
                "curve": curve,
                "comparator": comparator,
                "magnitude": magnitude,
                "status": cell.get("status", "complete"),
                "deep_rm_mean_slowdown": cell.get("deep_rm_mean_slowdown"),
                "comparator_mean_slowdown": cell.get("comparator_mean_slowdown"),
                "delta_comparator_minus_deeprm": comparison.get("mean_difference"),
                "ci_low": comparison.get("ci_low"),
                "ci_high": comparison.get("ci_high"),
                "p_less_than_zero": comparison.get("p_less_than_zero"),
                "p_greater_than_zero": comparison.get("p_greater_than_zero"),
                "holm_less_curve": holm["less_than_zero"].get(holm_key, {}).get(key),
                "holm_greater_curve": holm["greater_than_zero"].get(holm_key, {}).get(key),
                "reason": cell.get("reason"),
            }
        )
        if cell.get("status") != "complete" or "comparison" not in cell:
            continue
        for idx, (deep, comp, diff) in enumerate(
            zip(cell["deep_rm_metrics"], cell["comparator_metrics"], comparison["differences"], strict=True)
        ):
            raw_rows.append(
                {
                    "key": key,
                    "seed_index": idx,
                    "deep_rm_mean_slowdown": deep["mean_slowdown"],
                    "comparator_mean_slowdown": comp["mean_slowdown"],
                    "delta": diff,
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    payload = {"experiment": "E1", "method": "deeprm", "params": params, "holm": holm, "cells": cells}
    write_json_artifact(json_path, payload, run_id=run_id)
    mlflow.log_artifact(str(csv_path), artifact_path="paper/tables")
    mlflow.log_artifact(str(raw_csv), artifact_path="paper/data")

    lines = [
        "# E1 DeepRM Magnitude Sweep",
        "",
        f"MLflow run: `{run_id}`",
        "",
        "| Curve | Comparator | Magnitude | Status | Delta comparator-DeepRM | 95% CI | Holm p(Delta<0) | Holm p(Delta>0) |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] == "complete":
            lines.append(
                "| {curve} | {comparator} | {magnitude} | complete | {delta_comparator_minus_deeprm:.6g} | "
                "[{ci_low:.6g}, {ci_high:.6g}] | {holm_less_curve:.6g} | {holm_greater_curve:.6g} |".format(
                    **row
                )
            )
        else:
            lines.append(
                f"| {row['curve']} | {row['comparator']} | {row['magnitude']} | "
                f"{row['status']} | NA | NA | NA | NA |"
            )
    lines.extend(["", f"- CSV: `{csv_path.relative_to(ROOT)}`", f"- Raw: `{raw_csv.relative_to(ROOT)}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(md_path), artifact_path="paper")


def main() -> None:
    args = parse_args()
    checkpoint = ROOT / args.checkpoint
    raw = torch.load(checkpoint, map_location="cpu")
    metadata = raw.get("metadata", {})
    env_config = DeepRMConfig(**metadata.get("env_config", {}))
    eval_config = DeepRMConfig(**{**env_config.__dict__, "external_admission": True})
    policy = load_checkpoint(checkpoint, config=eval_config)
    deterministic = args.policy_mode == "deterministic"
    seeds = trace_seeds(args.seed, args.num_seeds)
    base_traces = make_traces(load=args.load, jobs=args.trace_jobs, seeds=seeds, config=eval_config, alpha=float("inf"))
    params = {
        "protocol": "PREREG_E1_magnitude_sweep",
        "method": "deeprm",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": metadata,
        "load": args.load,
        "num_seeds": args.num_seeds,
        "trace_jobs": args.trace_jobs,
        "seed": args.seed,
        "policy_seed": args.policy_seed,
        "policy_mode": args.policy_mode,
        "max_steps": args.max_steps,
        "censor_max_step_failures": args.censor_max_step_failures,
        "lags": list(LAGS),
        "alphas": list(ALPHAS),
        "epsilons": list(EPSILONS),
        "comparators": [comp.name for comp in COMPARATORS],
        "compute_policy": "canonical on remote workstation; local workstation only for smoke",
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="e1-deeprm-magnitude-sweep",
        role="e1_magnitude_sweep",
        params=params,
        tags={"experiment": "E1", "method": "deeprm"},
    ) as run:
        cells: dict[str, dict[str, object]] = {}
        for lag_idx, lag in enumerate(LAGS):
            lag_keys = [f"deeprm:p1_lag:{comparator.name}:{lag}" for comparator in COMPARATORS]
            if args.resume_existing and all(load_cell(key) is not None for key in lag_keys):
                for key in lag_keys:
                    cells[key] = load_cell(key) or {}
                    print(f"E1 DeepRM skip existing {key}", flush=True)
                continue
            generator = cell_generator(args.policy_seed, 1_000 + lag_idx)
            deep_metrics, deep_failure = evaluate_many(
                f"p1_lag:{lag}:deeprm",
                base_traces,
                lambda trace: run_lagged_policy_episode(
                    policy,
                    trace,
                    lag=lag,
                    config=eval_config,
                    policy_deterministic=deterministic,
                    policy_generator=None if deterministic else generator,
                    max_steps=args.max_steps,
                ),
                censor=args.censor_max_step_failures,
                max_steps=args.max_steps,
            )
            for comparator in COMPARATORS:
                key = f"deeprm:p1_lag:{comparator.name}:{lag}"
                if args.resume_existing and (existing := load_cell(key)) is not None:
                    cells[key] = existing
                    print(f"E1 DeepRM skip existing {key}", flush=True)
                    continue
                comp_metrics, comp_failure = evaluate_many(
                    f"p1_lag:{lag}:{comparator.name}",
                    base_traces,
                    lambda trace: run_lagged_scheduler_episode(comparator, trace, lag=lag, config=eval_config, max_steps=args.max_steps),
                    censor=args.censor_max_step_failures,
                    max_steps=args.max_steps,
                )
                if deep_metrics is None or comp_metrics is None:
                    cells[key] = instability_cell(
                        reason=(deep_failure or comp_failure or {}).get("reason", "noncompletion"),
                        max_steps=args.max_steps,
                        comparator_metrics=comp_metrics,
                        deep_failure=deep_failure,
                        comparator_failure=comp_failure,
                    )
                else:
                    cells[key] = summarize_cell(deep_metrics, comp_metrics, seed=args.seed + 10_000 + lag_idx)
                save_cell(key, cells[key])
                print(f"E1 DeepRM done {key}", flush=True)

        for alpha_idx, alpha in enumerate(ALPHAS):
            alpha_keys = [f"deeprm:p2_tail:{comparator.name}:{alpha}" for comparator in COMPARATORS]
            if args.resume_existing and all(load_cell(key) is not None for key in alpha_keys):
                for key in alpha_keys:
                    cells[key] = load_cell(key) or {}
                    print(f"E1 DeepRM skip existing {key}", flush=True)
                continue
            config = tail_config(eval_config, alpha)
            traces = make_traces(load=args.load, jobs=args.trace_jobs, seeds=seeds, config=config, alpha=alpha)
            generator = cell_generator(args.policy_seed, 2_000 + alpha_idx)
            from cisose_deeprm.model import DeepRMScheduler

            deep_scheduler = DeepRMScheduler(policy=policy, deterministic=deterministic, generator=None if deterministic else generator)
            deep_metrics, deep_failure = evaluate_many(
                f"p2_tail:{alpha}:deeprm",
                traces,
                lambda trace: run_episode(deep_scheduler, trace, config=config, max_steps=args.max_steps),
                censor=args.censor_max_step_failures,
                max_steps=args.max_steps,
            )
            for comparator in COMPARATORS:
                key = f"deeprm:p2_tail:{comparator.name}:{alpha}"
                if args.resume_existing and (existing := load_cell(key)) is not None:
                    cells[key] = existing
                    print(f"E1 DeepRM skip existing {key}", flush=True)
                    continue
                comp_metrics, comp_failure = evaluate_many(
                    f"p2_tail:{alpha}:{comparator.name}",
                    traces,
                    lambda trace: run_episode(comparator, trace, config=config, max_steps=args.max_steps),
                    censor=args.censor_max_step_failures,
                    max_steps=args.max_steps,
                )
                if deep_metrics is None or comp_metrics is None:
                    cells[key] = instability_cell(
                        reason=(deep_failure or comp_failure or {}).get("reason", "noncompletion"),
                        max_steps=args.max_steps,
                        comparator_metrics=comp_metrics,
                        deep_failure=deep_failure,
                        comparator_failure=comp_failure,
                    )
                else:
                    cells[key] = summarize_cell(deep_metrics, comp_metrics, seed=args.seed + 20_000 + alpha_idx)
                save_cell(key, cells[key])
                print(f"E1 DeepRM done {key}", flush=True)

        for eps_idx, epsilon in enumerate(EPSILONS):
            eps_keys = [f"deeprm:p3_fgsm:{comparator.name}:{epsilon}" for comparator in COMPARATORS]
            if args.resume_existing and all(load_cell(key) is not None for key in eps_keys):
                for key in eps_keys:
                    cells[key] = load_cell(key) or {}
                    print(f"E1 DeepRM skip existing {key}", flush=True)
                continue
            generator = cell_generator(args.policy_seed, 3_000 + eps_idx)
            deep_metrics, deep_failure = evaluate_many(
                f"p3_fgsm:{epsilon}:deeprm",
                base_traces,
                lambda trace: run_adversarial_policy_episode(
                    policy,
                    trace,
                    epsilon=epsilon,
                    config=eval_config,
                    policy_deterministic=deterministic,
                    policy_generator=None if deterministic else generator,
                    max_steps=args.max_steps,
                ),
                censor=args.censor_max_step_failures,
                max_steps=args.max_steps,
            )
            for comparator in COMPARATORS:
                key = f"deeprm:p3_fgsm:{comparator.name}:{epsilon}"
                if args.resume_existing and (existing := load_cell(key)) is not None:
                    cells[key] = existing
                    print(f"E1 DeepRM skip existing {key}", flush=True)
                    continue
                comp_metrics, comp_failure = evaluate_many(
                    f"p3_fgsm:{epsilon}:{comparator.name}",
                    base_traces,
                    lambda trace: run_episode(comparator, trace, config=eval_config, max_steps=args.max_steps),
                    censor=args.censor_max_step_failures,
                    max_steps=args.max_steps,
                )
                if deep_metrics is None or comp_metrics is None:
                    cells[key] = instability_cell(
                        reason=(deep_failure or comp_failure or {}).get("reason", "noncompletion"),
                        max_steps=args.max_steps,
                        comparator_metrics=comp_metrics,
                        deep_failure=deep_failure,
                        comparator_failure=comp_failure,
                    )
                else:
                    cells[key] = summarize_cell(deep_metrics, comp_metrics, seed=args.seed + 30_000 + eps_idx)
                save_cell(key, cells[key])
                print(f"E1 DeepRM done {key}", flush=True)

        for key in list(cells):
            cells[key] = refresh_comparison_seed(cells[key], seed=statistic_seed_for_key(key, args.seed))
            save_cell(key, cells[key])
        holm = holm_by_curve(cells)
        write_outputs(cells, holm, run.info.run_id, params)
        for key, cell in cells.items():
            safe_key = metric_key(key)
            if cell.get("status") == "complete" and "comparison" in cell:
                mlflow.log_metric(f"{safe_key}.delta", cell["comparison"]["mean_difference"])
            else:
                mlflow.log_metric(f"{safe_key}.noncompletion", 1.0)
        print(f"MLflow run: {run.info.run_id}")
        print(str((OUT_DIR / "e1_deeprm_magnitude_sweep.md").relative_to(ROOT)))


if __name__ == "__main__":
    main()
