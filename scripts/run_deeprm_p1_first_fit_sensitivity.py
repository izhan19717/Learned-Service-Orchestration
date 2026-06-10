#!/usr/bin/env python3
"""Run DeepRM P1 lag sensitivity with first-fit stale-action fallback."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from cisose_deeprm.cli import _cell_generator, _sha256, _trace_seeds, generate_trace_for_seed
from cisose_deeprm.evaluation import EpisodeMetrics, paired_result
from cisose_deeprm.model import load_checkpoint
from cisose_deeprm.perturbations import (
    LagBuffer,
    heuristic_action_on_snapshot,
    policy_action_on_observation,
    step_with_stale_identity_first_fit_fallback,
)
from cisose_deeprm.protocol import LAG_SWEEP, DeepRMConfig
from cisose_deeprm.schedulers import TetrisScheduler
from cisose_deeprm.simulator import DeepRMEnv
from cisose_deeprm.tracking import start_tracked_run, write_json_with_run_id
from cisose_deeprm.workload import WorkloadTrace


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results" / "checkpoints" / "author_source_rescue" / "load_0.7" / "policy_final.pt"
LOCKED_SWEEP = ROOT / "results" / "evaluation" / "deeprm" / "perturbation_sweeps_v2_2.json"
OUT_JSON = ROOT / "results" / "evaluation" / "deeprm" / "p1_lag_first_fit_sensitivity.json"
PAPER_DIR = ROOT / "results" / "paper" / "deeprm"
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"

LOAD = 0.7
NUM_SEEDS = 30
TRACE_JOBS = 200
SEED = 20260520
POLICY_SEED = 20260520
MAX_STEPS = 100_000


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw = torch.load(CHECKPOINT, map_location="cpu")
    metadata = raw["metadata"]
    train_config = DeepRMConfig(**metadata["env_config"])
    eval_config = DeepRMConfig(**{**train_config.__dict__, "external_admission": True})
    policy = load_checkpoint(CHECKPOINT, config=eval_config)
    locked = _read_json(LOCKED_SWEEP)
    trace_seeds = _trace_seeds(SEED, NUM_SEEDS)
    traces = tuple(
        generate_trace_for_seed(LOAD, TRACE_JOBS, trace_seed, eval_config, float("inf"))
        for trace_seed in trace_seeds
    )

    cells: dict[str, dict[str, object]] = {}
    for idx, lag in enumerate(LAG_SWEEP):
        generator = _cell_generator(POLICY_SEED, 1_000 + idx)
        deep_metrics, deep_status = _evaluate_policy_lag_first_fit(
            policy,
            traces,
            lag=lag,
            config=eval_config,
            generator=generator,
        )
        tetris_metrics, tetris_status = _evaluate_scheduler_lag_first_fit(
            TetrisScheduler(source_dot=True),
            traces,
            lag=lag,
            config=eval_config,
        )
        comparison = paired_result(
            tetris_metrics,
            deep_metrics,
            seed=SEED + 50_000 + idx,
        )
        locked_cell = locked["cells"][f"P1_lag_{lag}"]
        cells[f"lag_{lag}"] = {
            "lag": lag,
            "deep_rm_mean_slowdown": _mean(metric.mean_slowdown for metric in deep_metrics),
            "tetris_mean_slowdown": _mean(metric.mean_slowdown for metric in tetris_metrics),
            "comparison": asdict(comparison),
            "deep_rm_metrics": [asdict(metric) for metric in deep_metrics],
            "tetris_metrics": [asdict(metric) for metric in tetris_metrics],
            "deep_rm_status_summary": _status_summary(deep_status),
            "tetris_status_summary": _status_summary(tetris_status),
            "locked_protocol": {
                "deep_rm_mean_slowdown": locked_cell["deep_rm_mean_slowdown"],
                "tetris_mean_slowdown": locked_cell["tetris_mean_slowdown"],
                "delta_tetris_minus_deeprm": locked_cell["comparison"]["mean_difference"],
                "ci_low": locked_cell["comparison"]["ci_low"],
                "ci_high": locked_cell["comparison"]["ci_high"],
            },
        }

    anchor = cells["lag_10"]["comparison"]
    payload = {
        "status": "completed",
        "experiment_type": "methodology_sensitivity_not_preregistered_replacement",
        "sensitivity_rule": (
            "When a stale slot action selected a job in the stale snapshot but "
            "that same job is no longer schedulable in the corresponding current "
            "slot, execute first-fit allocation on the current true visible "
            "state. Explicit void actions and stale-empty-slot actions remain void."
        ),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "locked_protocol_artifact": str(LOCKED_SWEEP.relative_to(ROOT)),
        "load": LOAD,
        "num_seeds": NUM_SEEDS,
        "trace_jobs": TRACE_JOBS,
        "seed": SEED,
        "policy_seed": POLICY_SEED,
        "policy_mode": "stochastic_sample",
        "max_steps": MAX_STEPS,
        "cells": cells,
        "anchor_lag_10_interpretation": _interpret_anchor(anchor),
    }

    table_paths = _write_tables(payload)
    figure_paths = _write_figures(payload, locked)

    with start_tracked_run(
        run_name="deeprm-p1-first-fit-lag-sensitivity",
        role="methodology-sensitivity",
        root=ROOT,
        params={
            "checkpoint": payload["checkpoint"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "load": LOAD,
            "num_seeds": NUM_SEEDS,
            "trace_jobs": TRACE_JOBS,
            "seed": SEED,
            "policy_seed": POLICY_SEED,
            "policy_mode": "stochastic_sample",
            "max_steps": MAX_STEPS,
            "sensitivity": "lag_first_fit_fallback",
        },
    ) as run:
        for key, cell in cells.items():
            metric_key = key.replace(".", "_")
            mlflow.log_metric(f"{metric_key}.deep_rm_mean_slowdown", cell["deep_rm_mean_slowdown"])
            mlflow.log_metric(f"{metric_key}.tetris_mean_slowdown", cell["tetris_mean_slowdown"])
            mlflow.log_metric(f"{metric_key}.delta", cell["comparison"]["mean_difference"])
            mlflow.log_metric(f"{metric_key}.ci_low", cell["comparison"]["ci_low"])
            mlflow.log_metric(f"{metric_key}.ci_high", cell["comparison"]["ci_high"])
            mlflow.log_metric(
                f"{metric_key}.tetris_first_fit_fallback_allocate_fraction",
                cell["tetris_status_summary"]["first_fit_fallback_allocate_fraction"],
            )
            mlflow.log_metric(
                f"{metric_key}.deep_rm_first_fit_fallback_allocate_fraction",
                cell["deep_rm_status_summary"]["first_fit_fallback_allocate_fraction"],
            )
        write_json_with_run_id(OUT_JSON, payload, run.info.run_id)
        for path in table_paths:
            mlflow.log_artifact(str(path), artifact_path="paper/tables")
        for path in figure_paths:
            mlflow.log_artifact(str(path), artifact_path="paper/figures")
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(_compact_summary(payload), indent=2, sort_keys=True))


def _evaluate_policy_lag_first_fit(
    policy,
    traces: tuple[WorkloadTrace, ...],
    *,
    lag: int,
    config: DeepRMConfig,
    generator: torch.Generator,
) -> tuple[tuple[EpisodeMetrics, ...], list[Counter]]:
    metrics = []
    statuses = []
    for trace in traces:
        env = DeepRMEnv(trace, config=config)
        lag_buffer = LagBuffer(env, lag)
        status = Counter()
        steps = 0
        while not env.done:
            snapshot = lag_buffer.current()
            action = policy_action_on_observation(
                policy,
                snapshot.observation,
                deterministic=False,
                generator=generator,
            )
            _, _, _, info, mode = step_with_stale_identity_first_fit_fallback(
                env,
                action,
                snapshot.slot_job_ids,
            )
            status[mode] += 1
            status[f"info_status_{info.status}"] += 1
            steps += 1
            lag_buffer.update(env)
            if steps > MAX_STEPS:
                raise RuntimeError(f"DeepRM first-fit lag episode exceeded max_steps={MAX_STEPS}")
        status["steps"] = steps
        metrics.append(_episode_metrics(env, steps))
        statuses.append(status)
    return tuple(metrics), statuses


def _evaluate_scheduler_lag_first_fit(
    scheduler,
    traces: tuple[WorkloadTrace, ...],
    *,
    lag: int,
    config: DeepRMConfig,
) -> tuple[tuple[EpisodeMetrics, ...], list[Counter]]:
    metrics = []
    statuses = []
    for trace in traces:
        env = DeepRMEnv(trace, config=config)
        lag_buffer = LagBuffer(env, lag)
        status = Counter()
        steps = 0
        while not env.done:
            snapshot = lag_buffer.current()
            action = heuristic_action_on_snapshot(scheduler, snapshot, env)
            _, _, _, info, mode = step_with_stale_identity_first_fit_fallback(
                env,
                action,
                snapshot.slot_job_ids,
            )
            status[mode] += 1
            status[f"info_status_{info.status}"] += 1
            steps += 1
            lag_buffer.update(env)
            if steps > MAX_STEPS:
                raise RuntimeError(f"Tetris first-fit lag episode exceeded max_steps={MAX_STEPS}")
        status["steps"] = steps
        metrics.append(_episode_metrics(env, steps))
        statuses.append(status)
    return tuple(metrics), statuses


def _episode_metrics(env: DeepRMEnv, steps: int) -> EpisodeMetrics:
    return EpisodeMetrics(
        mean_slowdown=env.mean_slowdown(),
        p95_completion_time=env.p95_completion_time(),
        makespan=env.makespan(),
        steps=steps,
    )


def _status_summary(statuses: list[Counter]) -> dict[str, float | int]:
    total = Counter()
    for status in statuses:
        total.update(status)
    steps = max(1, total["steps"])
    keys = [
        "same_identity_Allocate",
        "same_identity_MoveOn",
        "first_fit_fallback_allocate",
        "first_fit_fallback_no_fit",
        "stale_empty_slot_no_fallback",
        "explicit_void",
        "info_status_Allocate",
        "info_status_MoveOn",
    ]
    summary: dict[str, float | int] = {"steps": int(total["steps"])}
    for key in keys:
        summary[key] = int(total[key])
        summary[f"{key}_fraction"] = float(total[key] / steps)
    return summary


def _interpret_anchor(comparison: dict[str, object]) -> str:
    ci_low = float(comparison["ci_low"])
    ci_high = float(comparison["ci_high"])
    if ci_low > 0.0:
        return "p1_falsification_persists_under_first_fit_sensitivity"
    if ci_high < 0.0:
        return "p1_locked_falsification_likely_protocol_artifact_under_first_fit_sensitivity"
    return "p1_first_fit_sensitivity_inconclusive"


def _write_tables(payload: dict[str, object]) -> list[Path]:
    rows = []
    for key, cell in payload["cells"].items():
        rows.append(
            {
                "lag": cell["lag"],
                "deep_rm_mean": cell["deep_rm_mean_slowdown"],
                "tetris_mean": cell["tetris_mean_slowdown"],
                "delta_tetris_minus_deeprm": cell["comparison"]["mean_difference"],
                "ci_low": cell["comparison"]["ci_low"],
                "ci_high": cell["comparison"]["ci_high"],
                "locked_delta": cell["locked_protocol"]["delta_tetris_minus_deeprm"],
                "tetris_first_fit_fallback_allocate_fraction": cell["tetris_status_summary"][
                    "first_fit_fallback_allocate_fraction"
                ],
                "deep_rm_first_fit_fallback_allocate_fraction": cell["deep_rm_status_summary"][
                    "first_fit_fallback_allocate_fraction"
                ],
            }
        )
    csv_path = TABLE_DIR / "deeprm_p1_first_fit_sensitivity.csv"
    md_path = TABLE_DIR / "deeprm_p1_first_fit_sensitivity.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    return [csv_path, md_path]


def _write_figures(payload: dict[str, object], locked: dict[str, object]) -> list[Path]:
    lags = list(LAG_SWEEP)
    alt_delta = [payload["cells"][f"lag_{lag}"]["comparison"]["mean_difference"] for lag in lags]
    alt_low = [payload["cells"][f"lag_{lag}"]["comparison"]["ci_low"] for lag in lags]
    alt_high = [payload["cells"][f"lag_{lag}"]["comparison"]["ci_high"] for lag in lags]
    locked_delta = [locked["cells"][f"P1_lag_{lag}"]["comparison"]["mean_difference"] for lag in lags]
    locked_low = [locked["cells"][f"P1_lag_{lag}"]["comparison"]["ci_low"] for lag in lags]
    locked_high = [locked["cells"][f"P1_lag_{lag}"]["comparison"]["ci_high"] for lag in lags]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    _plot_ci_line(ax, lags, locked_delta, locked_low, locked_high, "#777777", "Locked no-op fallback")
    _plot_ci_line(ax, lags, alt_delta, alt_low, alt_high, "#1f4e79", "First-fit fallback sensitivity")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.axvline(10, color="#b04a1a", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Observation lag k")
    ax.set_ylabel("Delta slowdown: Tetris* - DeepRM")
    ax.set_title("DeepRM P1 Stale-Action Sensitivity")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png = FIG_DIR / "deeprm_p1_first_fit_sensitivity.png"
    pdf = FIG_DIR / "deeprm_p1_first_fit_sensitivity.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _plot_ci_line(ax, x, mean, low, high, color, label):
    mean_arr = np.asarray(mean, dtype=float)
    yerr = np.vstack([mean_arr - np.asarray(low), np.asarray(high) - mean_arr])
    ax.errorbar(x, mean, yerr=yerr, color=color, marker="o", linewidth=1.2, capsize=3, label=label)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compact_summary(payload: dict[str, object]) -> dict[str, object]:
    out = {
        "anchor_lag_10_interpretation": payload["anchor_lag_10_interpretation"],
        "cells": {},
    }
    for key, cell in payload["cells"].items():
        out["cells"][key] = {
            "deep_rm_mean_slowdown": cell["deep_rm_mean_slowdown"],
            "tetris_mean_slowdown": cell["tetris_mean_slowdown"],
            "delta": cell["comparison"]["mean_difference"],
            "ci_low": cell["comparison"]["ci_low"],
            "ci_high": cell["comparison"]["ci_high"],
            "locked_delta": cell["locked_protocol"]["delta_tetris_minus_deeprm"],
            "tetris_first_fit_fallback_allocate_fraction": cell["tetris_status_summary"][
                "first_fit_fallback_allocate_fraction"
            ],
        }
    return out


def _mean(values) -> float:
    return float(np.mean(list(values)))


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    main()
