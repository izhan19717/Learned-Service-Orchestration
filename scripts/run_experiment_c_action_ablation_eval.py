#!/usr/bin/env python3
"""Experiment C: DeepRM visible-action ablation under FGSM."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from dataclasses import asdict
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
import torch

from cisose_common.stats import paired_bootstrap_ci
from cisose_deeprm.cli import _cell_generator, _sha256, _trace_seeds, generate_trace_for_seed
from cisose_deeprm.evaluation import EpisodeMetrics, run_episode
from cisose_deeprm.model import load_checkpoint
from cisose_deeprm.perturbations import fgsm_observation
from cisose_deeprm.protocol import BOOTSTRAP_RESAMPLES, DeepRMConfig
from cisose_deeprm.schedulers import TetrisScheduler
from cisose_deeprm.simulator import DeepRMEnv
from cisose_deeprm.tracking import start_tracked_run, write_json_with_run_id
from cisose_deeprm.workload import WorkloadTrace


LOAD = 0.7
NUM_SEEDS = 30
TRACE_JOBS = 200
SEED = 20260520
POLICY_SEED = 20260520
MAX_STEPS = 100_000
EPSILON = 0.05

CHECKPOINTS = {
    10: ROOT / "results" / "checkpoints" / "author_source_aligned" / "load_0.7" / "policy_final.pt",
    3: ROOT / "results" / "checkpoints" / "experiment_c_m3_author_source" / "load_0.7" / "policy_final.pt",
    1: ROOT / "results" / "checkpoints" / "experiment_c_m1_author_source" / "load_0.7" / "policy_final.pt",
}

LOCKED_P3 = ROOT / "results" / "evaluation" / "deeprm" / "perturbation_sweeps_v2_2.json"
OUT_JSON = ROOT / "results" / "evaluation" / "deeprm" / "experiment_c_action_ablation.json"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "experiment_c"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
ROOT_FIG_DIR = ROOT / "figures"
ROOT_DATA_DIR = ROOT / "data"
ROOT_REPORT = ROOT / "experiment_c_results.md"


def main() -> None:
    _ensure_dirs()
    locked = _read_json(LOCKED_P3)
    trace_seeds = _trace_seeds(SEED, NUM_SEEDS)

    cells: dict[int, dict[str, object]] = {}
    for visible_slots in (10, 3, 1):
        print(f"Evaluating M={visible_slots} checkpoint...", flush=True)
        cells[visible_slots] = evaluate_visible_slots(
            visible_slots=visible_slots,
            checkpoint=CHECKPOINTS[visible_slots],
            trace_seeds=trace_seeds,
        )

    comparisons = compare_degradation(cells)
    verdict = interpret_verdict(cells, comparisons)
    m10_regression = m10_regression_check(cells[10], locked)

    payload = {
        "status": "completed",
        "experiment": "C",
        "method": "DeepRM",
        "purpose": "visible-action-space ablation under P3 FGSM",
        "load": LOAD,
        "num_seeds": NUM_SEEDS,
        "trace_jobs": TRACE_JOBS,
        "seed": SEED,
        "policy_seed": POLICY_SEED,
        "policy_mode": "stochastic_sample",
        "max_steps": MAX_STEPS,
        "epsilon": EPSILON,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "p3_semantics": "Option A: DeepRM observes FGSM-perturbed state; SourceTetris observes true state.",
        "competency_gate": "DeepRM_M must beat SourceTetris_M on clean in at least 20/30 paired seeds; one-sided exact binomial p<0.05.",
        "checkpoints": {
            str(m): {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for m, path in CHECKPOINTS.items()
        },
        "locked_p3_reference": str(LOCKED_P3.relative_to(ROOT)),
        "m10_regression_check": m10_regression,
        "cells": {str(k): v for k, v in cells.items()},
        "degradation_comparisons": comparisons,
        "verdict": verdict,
    }

    data_paths = write_data(payload)
    figure_paths = write_figures(payload)

    with start_tracked_run(
        run_name="experiment-c-deeprm-action-ablation",
        role="experiment_c_deeprm_action_ablation",
        root=ROOT,
        params={
            "experiment": "C",
            "method": "DeepRM",
            "load": LOAD,
            "num_seeds": NUM_SEEDS,
            "trace_jobs": TRACE_JOBS,
            "seed": SEED,
            "policy_seed": POLICY_SEED,
            "policy_mode": "stochastic_sample",
            "epsilon": EPSILON,
            "max_steps": MAX_STEPS,
            "visible_slots": [10, 3, 1],
        },
        tags={"experiment": "C", "method": "deeprm", "perturbation": "fgsm"},
    ) as run:
        payload["mlflow_run_id"] = run.info.run_id
        report_paths = write_report(payload, data_paths, figure_paths, run_id=run.info.run_id)
        manifest_path = write_manifest(payload, [*data_paths, *figure_paths, *report_paths])
        for m, cell in cells.items():
            key = f"m{m}"
            mlflow.log_metric(f"{key}.clean.deep_rm_mean_slowdown", cell["clean_deep_rm_mean_slowdown"])
            mlflow.log_metric(f"{key}.clean.source_tetris_mean_slowdown", cell["clean_source_tetris_mean_slowdown"])
            mlflow.log_metric(f"{key}.p3.deep_rm_mean_slowdown", cell["p3_deep_rm_mean_slowdown"])
            mlflow.log_metric(f"{key}.p3.source_tetris_mean_slowdown", cell["p3_source_tetris_mean_slowdown"])
            mlflow.log_metric(f"{key}.deg_mean", cell["deg_mean"])
            mlflow.log_metric(f"{key}.deg_ci_low", cell["deg_ci_low"])
            mlflow.log_metric(f"{key}.deg_ci_high", cell["deg_ci_high"])
            mlflow.log_metric(f"{key}.competency_wins", cell["competency_wins"])
            mlflow.log_metric(f"{key}.competency_gate_passed", 1.0 if cell["competency_gate_passed"] else 0.0)
            mlflow.log_metric(f"{key}.argmax_change_rate", cell["argmax_change_rate"])
            mlflow.log_metric(f"{key}.mean_total_variation", cell["mean_total_variation"])
            mlflow.log_metric(f"{key}.mean_clean_argmax_probability_drop", cell["mean_clean_argmax_probability_drop"])
            mlflow.log_metric(f"{key}.clean_action_entropy", cell["clean_action_entropy"])
        for name, value in comparisons.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"comparison.{name}", float(value))
        for name, value in m10_regression.items():
            if isinstance(value, (int, float, bool)):
                mlflow.log_metric(f"m10_regression.{name}", float(value))
        write_json_with_run_id(OUT_JSON, payload, run.info.run_id)
        for path in [*data_paths, *figure_paths, *report_paths, manifest_path, Path(__file__), ROOT / "EXPERIMENT_C_action_ablation_deeprm.md"]:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=_artifact_group(path))
        print(f"MLflow run: {run.info.run_id}")
        print(str(report_paths[0].relative_to(ROOT)))
        print(json.dumps(compact_summary(payload), indent=2, sort_keys=True))


def evaluate_visible_slots(
    *,
    visible_slots: int,
    checkpoint: Path,
    trace_seeds: list[int],
) -> dict[str, object]:
    raw = torch.load(checkpoint, map_location="cpu")
    metadata = raw.get("metadata", {})
    env_config_dict = metadata.get("env_config")
    if not env_config_dict:
        raise ValueError(f"checkpoint lacks env_config metadata: {checkpoint}")
    train_config = DeepRMConfig(**env_config_dict)
    if train_config.visible_slots != visible_slots:
        raise ValueError(
            f"checkpoint {checkpoint} has visible_slots={train_config.visible_slots}, expected {visible_slots}"
        )
    eval_config = DeepRMConfig(**{**train_config.__dict__, "external_admission": True})
    policy = load_checkpoint(checkpoint, config=eval_config)
    traces = tuple(
        generate_trace_for_seed(LOAD, TRACE_JOBS, seed, eval_config, float("inf"))
        for seed in trace_seeds
    )

    source_tetris = TetrisScheduler(source_dot=True)
    source_metrics = tuple(
        run_episode(source_tetris, trace, config=eval_config, max_steps=MAX_STEPS)
        for trace in traces
    )

    clean_generator = _cell_generator(POLICY_SEED, 3_000)
    p3_generator = _cell_generator(POLICY_SEED, 3_003)
    clean_runs = tuple(
        run_fgsm_episode_with_diagnostics(
            policy,
            trace,
            epsilon=0.0,
            config=eval_config,
            generator=clean_generator,
        )
        for trace in traces
    )
    p3_runs = tuple(
        run_fgsm_episode_with_diagnostics(
            policy,
            trace,
            epsilon=EPSILON,
            config=eval_config,
            generator=p3_generator,
        )
        for trace in traces
    )

    clean_deep = tuple(item["metrics"] for item in clean_runs)
    p3_deep = tuple(item["metrics"] for item in p3_runs)
    clean_deep_values = np.asarray([metric.mean_slowdown for metric in clean_deep], dtype=np.float64)
    p3_deep_values = np.asarray([metric.mean_slowdown for metric in p3_deep], dtype=np.float64)
    source_values = np.asarray([metric.mean_slowdown for metric in source_metrics], dtype=np.float64)
    deg = p3_deep_values - clean_deep_values
    deg_ci_low, deg_ci_high = paired_bootstrap_ci(deg, seed=SEED + 60_000 + visible_slots)

    delta_clean = source_values - clean_deep_values
    delta_p3 = source_values - p3_deep_values
    delta_clean_ci = paired_bootstrap_ci(delta_clean, seed=SEED + 61_000 + visible_slots)
    delta_p3_ci = paired_bootstrap_ci(delta_p3, seed=SEED + 62_000 + visible_slots)
    wins = int(np.count_nonzero(delta_clean > 0.0))
    competency_p = binomial_greater_equal_pvalue(wins, len(delta_clean))

    p3_diag = aggregate_diagnostics(p3_runs)
    clean_diag = aggregate_diagnostics(clean_runs)

    return {
        "visible_slots": visible_slots,
        "action_dim": visible_slots + 1,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "eval_env_config": eval_config.__dict__,
        "clean_deep_rm_mean_slowdown": float(np.mean(clean_deep_values)),
        "clean_source_tetris_mean_slowdown": float(np.mean(source_values)),
        "p3_deep_rm_mean_slowdown": float(np.mean(p3_deep_values)),
        "p3_source_tetris_mean_slowdown": float(np.mean(source_values)),
        "deg_per_seed": deg.tolist(),
        "deg_mean": float(np.mean(deg)),
        "deg_ci_low": deg_ci_low,
        "deg_ci_high": deg_ci_high,
        "delta_clean_per_seed": delta_clean.tolist(),
        "delta_clean_mean": float(np.mean(delta_clean)),
        "delta_clean_ci_low": delta_clean_ci[0],
        "delta_clean_ci_high": delta_clean_ci[1],
        "delta_p3_per_seed": delta_p3.tolist(),
        "delta_p3_mean": float(np.mean(delta_p3)),
        "delta_p3_ci_low": delta_p3_ci[0],
        "delta_p3_ci_high": delta_p3_ci[1],
        "competency_wins": wins,
        "competency_n": len(delta_clean),
        "competency_p_one_sided": competency_p,
        "competency_gate_passed": wins >= 20 and competency_p < 0.05,
        "argmax_change_rate": p3_diag["argmax_change_rate"],
        "mean_clean_argmax_probability_drop": p3_diag["mean_clean_argmax_probability_drop"],
        "mean_total_variation": p3_diag["mean_total_variation"],
        "clean_action_entropy": clean_diag["mean_entropy_clean"],
        "p3_clean_state_entropy": p3_diag["mean_entropy_clean"],
        "p3_adv_state_entropy": p3_diag["mean_entropy_adv"],
        "diagnostic_states": p3_diag["states"],
        "clean_deep_rm_metrics": [asdict(metric) for metric in clean_deep],
        "p3_deep_rm_metrics": [asdict(metric) for metric in p3_deep],
        "source_tetris_metrics": [asdict(metric) for metric in source_metrics],
    }


def run_fgsm_episode_with_diagnostics(
    policy,
    trace: WorkloadTrace,
    *,
    epsilon: float,
    config: DeepRMConfig,
    generator: torch.Generator,
) -> dict[str, object]:
    env = DeepRMEnv(trace, config=config)
    sums = {
        "states": 0,
        "argmax_change": 0.0,
        "clean_argmax_probability_drop": 0.0,
        "total_variation": 0.0,
        "entropy_clean": 0.0,
        "entropy_adv": 0.0,
    }
    steps = 0
    while not env.done:
        obs = env.observe()
        probs_clean = policy_probs(policy, obs)
        argmax_clean = int(torch.argmax(probs_clean).item())
        adv_obs = fgsm_observation(policy, obs, epsilon)
        probs_adv = policy_probs(policy, adv_obs)
        argmax_adv = int(torch.argmax(probs_adv).item())
        action = int(torch.multinomial(probs_adv, 1, generator=generator).item())

        sums["states"] += 1
        sums["argmax_change"] += 1.0 if argmax_clean != argmax_adv else 0.0
        sums["clean_argmax_probability_drop"] += float(probs_clean[argmax_clean] - probs_adv[argmax_clean])
        sums["total_variation"] += float(0.5 * torch.sum(torch.abs(probs_clean - probs_adv)).item())
        sums["entropy_clean"] += entropy(probs_clean)
        sums["entropy_adv"] += entropy(probs_adv)

        env.step(action)
        steps += 1
        if steps > MAX_STEPS:
            raise RuntimeError(f"FGSM episode exceeded max_steps={MAX_STEPS}")

    return {
        "metrics": EpisodeMetrics(
            mean_slowdown=env.mean_slowdown(),
            p95_completion_time=env.p95_completion_time(),
            makespan=env.makespan(),
            steps=steps,
        ),
        "diagnostics": sums,
    }


def policy_probs(policy, observation: np.ndarray) -> torch.Tensor:
    state = torch.from_numpy(observation).unsqueeze(0).float()
    with torch.no_grad():
        logits = policy(state).squeeze(0)
        return torch.softmax(logits, dim=-1)


def entropy(probs: torch.Tensor) -> float:
    clipped = torch.clamp(probs, min=1e-12)
    return float(-(clipped * torch.log(clipped)).sum().item())


def aggregate_diagnostics(runs: Iterable[dict[str, object]]) -> dict[str, float]:
    total = {
        "states": 0.0,
        "argmax_change": 0.0,
        "clean_argmax_probability_drop": 0.0,
        "total_variation": 0.0,
        "entropy_clean": 0.0,
        "entropy_adv": 0.0,
    }
    for run in runs:
        diagnostics = run["diagnostics"]
        for key in total:
            total[key] += float(diagnostics[key])
    states = max(total["states"], 1.0)
    return {
        "states": int(total["states"]),
        "argmax_change_rate": float(total["argmax_change"] / states),
        "mean_clean_argmax_probability_drop": float(total["clean_argmax_probability_drop"] / states),
        "mean_total_variation": float(total["total_variation"] / states),
        "mean_entropy_clean": float(total["entropy_clean"] / states),
        "mean_entropy_adv": float(total["entropy_adv"] / states),
    }


def compare_degradation(cells: dict[int, dict[str, object]]) -> dict[str, float]:
    deg10 = np.asarray(cells[10]["deg_per_seed"], dtype=np.float64)
    deg3 = np.asarray(cells[3]["deg_per_seed"], dtype=np.float64)
    deg1 = np.asarray(cells[1]["deg_per_seed"], dtype=np.float64)
    d3_minus_2d10 = deg3 - 2.0 * deg10
    d1_minus_4d10 = deg1 - 4.0 * deg10
    d3_minus_d10 = deg3 - deg10
    d1_minus_d3 = deg1 - deg3
    c_d3_2 = paired_bootstrap_ci(d3_minus_2d10, seed=SEED + 70_003)
    c_d1_4 = paired_bootstrap_ci(d1_minus_4d10, seed=SEED + 70_001)
    c_d3_10 = paired_bootstrap_ci(d3_minus_d10, seed=SEED + 71_003)
    c_d1_3 = paired_bootstrap_ci(d1_minus_d3, seed=SEED + 71_001)
    return {
        "deg_10": float(np.mean(deg10)),
        "deg_3": float(np.mean(deg3)),
        "deg_1": float(np.mean(deg1)),
        "ratio_deg3_to_deg10": safe_ratio(float(np.mean(deg3)), float(np.mean(deg10))),
        "ratio_deg1_to_deg10": safe_ratio(float(np.mean(deg1)), float(np.mean(deg10))),
        "mean_deg3_minus_2x_deg10": float(np.mean(d3_minus_2d10)),
        "ci_low_deg3_minus_2x_deg10": c_d3_2[0],
        "ci_high_deg3_minus_2x_deg10": c_d3_2[1],
        "mean_deg1_minus_4x_deg10": float(np.mean(d1_minus_4d10)),
        "ci_low_deg1_minus_4x_deg10": c_d1_4[0],
        "ci_high_deg1_minus_4x_deg10": c_d1_4[1],
        "mean_deg3_minus_deg10": float(np.mean(d3_minus_d10)),
        "ci_low_deg3_minus_deg10": c_d3_10[0],
        "ci_high_deg3_minus_deg10": c_d3_10[1],
        "mean_deg1_minus_deg3": float(np.mean(d1_minus_d3)),
        "ci_low_deg1_minus_deg3": c_d1_3[0],
        "ci_high_deg1_minus_deg3": c_d1_3[1],
    }


def interpret_verdict(cells: dict[int, dict[str, object]], comparisons: dict[str, float]) -> dict[str, object]:
    active_ms = [m for m in (10, 3, 1) if cells[m]["competency_gate_passed"]]
    if 1 not in active_ms:
        case = "M1_not_learnable_H_C1_not_supported_for_competent_conditions"
        summary = (
            "M=1 failed the clean competency gate, so it is not interpretable as a robustness test. "
            "Among competent conditions, reducing M from 10 to 3 did not increase FGSM aggregate "
            "degradation; H_C1 is not supported under the locked training budget."
        )
    elif comparisons["ci_low_deg3_minus_2x_deg10"] > 0.0 and comparisons["ci_low_deg1_minus_4x_deg10"] > 0.0:
        case = "H_C1"
        summary = "Action redundancy is supported: FGSM aggregate degradation grows substantially as M shrinks."
    elif (
        0.5 <= comparisons["ratio_deg3_to_deg10"] <= 2.0
        and 0.5 <= comparisons["ratio_deg1_to_deg10"] <= 2.0
    ):
        case = "H_C2"
        summary = "Action redundancy is falsified: degradation stays in the pre-registered constant-effect band."
    elif (
        0.5 <= comparisons["ratio_deg3_to_deg10"] <= 2.0
        and comparisons["ratio_deg1_to_deg10"] > 2.0
    ):
        case = "H_C3"
        summary = "Mixed result: moderate reduction is close to M=10, but M=1 is substantially larger."
    else:
        case = "unclassified"
        summary = "The effect pattern does not cleanly match H_C1, H_C2, or H_C3."
    return {
        "case": case,
        "summary": summary,
        "active_competent_conditions": active_ms,
        "all_competency_gates_passed": len(active_ms) == 3,
    }


def m10_regression_check(cell: dict[str, object], locked: dict[str, object]) -> dict[str, object]:
    clean = locked["cells"]["P3_epsilon_0.0"]
    p3 = locked["cells"]["P3_epsilon_0.05"]
    checks = {
        "clean_deep_rm_relative_error": relative_error(
            cell["clean_deep_rm_mean_slowdown"], clean["deep_rm_mean_slowdown"]
        ),
        "clean_source_tetris_relative_error": relative_error(
            cell["clean_source_tetris_mean_slowdown"], clean["tetris_mean_slowdown"]
        ),
        "p3_deep_rm_relative_error": relative_error(
            cell["p3_deep_rm_mean_slowdown"], p3["deep_rm_mean_slowdown"]
        ),
        "p3_source_tetris_relative_error": relative_error(
            cell["p3_source_tetris_mean_slowdown"], p3["tetris_mean_slowdown"]
        ),
    }
    checks["within_5pct"] = all(float(value) <= 0.05 for value in checks.values())
    checks["locked_values"] = {
        "clean_deep_rm_mean_slowdown": clean["deep_rm_mean_slowdown"],
        "clean_source_tetris_mean_slowdown": clean["tetris_mean_slowdown"],
        "p3_deep_rm_mean_slowdown": p3["deep_rm_mean_slowdown"],
        "p3_source_tetris_mean_slowdown": p3["tetris_mean_slowdown"],
    }
    return checks


def write_data(payload: dict[str, object]) -> list[Path]:
    rows = []
    for m, cell in sorted(payload["cells"].items(), key=lambda item: int(item[0]), reverse=True):
        visible_slots = int(m)
        clean_deep = cell["clean_deep_rm_metrics"]
        p3_deep = cell["p3_deep_rm_metrics"]
        source = cell["source_tetris_metrics"]
        deg = cell["deg_per_seed"]
        delta_clean = cell["delta_clean_per_seed"]
        delta_p3 = cell["delta_p3_per_seed"]
        for seed_idx in range(len(clean_deep)):
            rows.append(
                {
                    "M": visible_slots,
                    "seed_index": seed_idx,
                    "clean_deep_rm_slowdown": clean_deep[seed_idx]["mean_slowdown"],
                    "clean_source_tetris_slowdown": source[seed_idx]["mean_slowdown"],
                    "p3_deep_rm_slowdown": p3_deep[seed_idx]["mean_slowdown"],
                    "p3_source_tetris_slowdown": source[seed_idx]["mean_slowdown"],
                    "deg_M": deg[seed_idx],
                    "delta_clean_source_minus_deeprm": delta_clean[seed_idx],
                    "delta_p3_source_minus_deeprm": delta_p3[seed_idx],
                }
            )
    slowdowns_path = DATA_DIR / "experiment_c_slowdowns.csv"
    root_slowdowns_path = ROOT_DATA_DIR / "experiment_c_slowdowns.csv"
    _write_csv(slowdowns_path, rows)
    shutil.copyfile(slowdowns_path, root_slowdowns_path)

    summary_rows = []
    for m, cell in sorted(payload["cells"].items(), key=lambda item: int(item[0]), reverse=True):
        summary_rows.append(summary_row(int(m), cell))
    summary_path = TABLE_DIR / "experiment_c_results.csv"
    _write_csv(summary_path, summary_rows)
    return [slowdowns_path, root_slowdowns_path, summary_path]


def write_figures(payload: dict[str, object]) -> list[Path]:
    ms = [10, 3, 1]
    cells = {int(k): v for k, v in payload["cells"].items()}
    labels = [f"M={m}" for m in ms]
    x = np.arange(len(ms))

    means = np.asarray([cells[m]["deg_mean"] for m in ms], dtype=np.float64)
    lows = np.asarray([cells[m]["deg_ci_low"] for m in ms], dtype=np.float64)
    highs = np.asarray([cells[m]["deg_ci_high"] for m in ms], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(x, means, color=["#4C78A8", "#72B7B2", "#F58518"], width=0.62)
    ax.errorbar(x, means, yerr=np.vstack([means - lows, highs - means]), fmt="none", ecolor="#222222", elinewidth=1.2, capsize=4)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("P3 - clean mean slowdown")
    ax.set_title("DeepRM FGSM degradation by visible action set")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    degradation_pdf = FIG_DIR / "deeprm_ablation_degradation.pdf"
    degradation_png = FIG_DIR / "deeprm_ablation_degradation.png"
    fig.savefig(degradation_pdf, bbox_inches="tight")
    fig.savefig(degradation_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = [
        ("argmax_change_rate", "Argmax change rate"),
        ("mean_total_variation", "Total variation"),
        ("mean_clean_argmax_probability_drop", "Clean-argmax prob. drop"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True)
    for ax, (key, title) in zip(axes, metrics, strict=True):
        values = [cells[m][key] for m in ms]
        ax.bar(x, values, color="#4C78A8", width=0.62)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(bottom=0.0)
    fig.suptitle("FGSM action-level diagnostics", y=1.03)
    fig.tight_layout()
    diag_pdf = FIG_DIR / "deeprm_ablation_action_diagnostics.pdf"
    diag_png = FIG_DIR / "deeprm_ablation_action_diagnostics.png"
    fig.savefig(diag_pdf, bbox_inches="tight")
    fig.savefig(diag_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    training_paths = write_training_curve_figure()
    all_paths = [degradation_pdf, degradation_png, diag_pdf, diag_png, *training_paths]
    for path in all_paths:
        if path.parent == FIG_DIR:
            shutil.copyfile(path, ROOT_FIG_DIR / path.name)
    return all_paths + [ROOT_FIG_DIR / path.name for path in all_paths if path.parent == FIG_DIR]


def write_training_curve_figure() -> list[Path]:
    curves = {
        "M=10": ROOT / "results" / "training" / "author_source_aligned" / "load_0.7_curve.jsonl",
        "M=3": ROOT / "results" / "training" / "experiment_c_m3_author_source" / "load_0.7_curve.jsonl",
        "M=1": ROOT / "results" / "training" / "experiment_c_m1_author_source" / "load_0.7_curve.jsonl",
    }
    if not all(path.exists() for path in curves.values()):
        return []
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, path in curves.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        xs = np.asarray([row["iteration"] for row in rows], dtype=np.float64)
        ys = np.asarray([row["mean_episode_reward"] for row in rows], dtype=np.float64)
        window = 25
        if len(ys) >= window:
            smooth = np.convolve(ys, np.ones(window) / window, mode="valid")
            smooth_x = xs[window - 1 :]
        else:
            smooth = ys
            smooth_x = xs
        ax.plot(smooth_x, smooth, linewidth=1.4, label=label)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean episode reward, 25-iteration moving average")
    ax.set_title("Experiment C training curves")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    pdf = FIG_DIR / "deeprm_ablation_training_curves.pdf"
    png = FIG_DIR / "deeprm_ablation_training_curves.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def write_report(
    payload: dict[str, object],
    data_paths: list[Path],
    figure_paths: list[Path],
    *,
    run_id: str | None = None,
) -> list[Path]:
    rows = [summary_row(int(m), cell) for m, cell in sorted(payload["cells"].items(), key=lambda item: int(item[0]), reverse=True)]
    verdict = payload["verdict"]
    m10 = payload["m10_regression_check"]
    lines = [
        "# Experiment C Results - DeepRM Action-Space Ablation",
        "",
        f"MLflow run: `{run_id or 'pending until script logs artifact'}`",
        "",
        "## Protocol",
        "",
        f"- Evaluation uses load={LOAD}, {NUM_SEEDS} paired seeds, {TRACE_JOBS} jobs per trace, stochastic DeepRM policy sampling, and P3 epsilon={EPSILON}.",
        "- P3 follows Option A: DeepRM receives the FGSM-perturbed observation; SourceTetris receives the true current state.",
        "- The degradation statistic is within-pipeline: deg_M = DeepRM_M(P3) - DeepRM_M(clean).",
        "- The clean competency gate requires DeepRM_M to beat SourceTetris_M on at least 20/30 paired seeds with one-sided exact binomial p<0.05.",
        "",
        "## Training Provenance",
        "",
        "| M | checkpoint | iteration | action dim | MLflow training run |",
        "|---:|---|---:|---:|---|",
    ]
    for m, cell in sorted(payload["cells"].items(), key=lambda item: int(item[0]), reverse=True):
        meta = cell["checkpoint_metadata"]
        lines.append(
            f"| {m} | `{cell['checkpoint']}` | {meta.get('iteration')} | {cell['action_dim']} | `{meta.get('mlflow_run_id')}` |"
        )
    lines.extend(
        [
            "",
            "## M10 Regression Check",
            "",
            f"The M=10 evaluator was rerun through the Experiment C harness and compared against the locked main-paper P3 pipeline. All four reference means are within 5%: `{m10['within_5pct']}`.",
            "",
            "| Quantity | Relative error |",
            "|---|---:|",
            f"| Clean DeepRM | {100*m10['clean_deep_rm_relative_error']:.3f}% |",
            f"| Clean SourceTetris | {100*m10['clean_source_tetris_relative_error']:.3f}% |",
            f"| P3 DeepRM | {100*m10['p3_deep_rm_relative_error']:.3f}% |",
            f"| P3 SourceTetris | {100*m10['p3_source_tetris_relative_error']:.3f}% |",
            "",
            "## Results Table",
            "",
            "| M | Clean DeepRM | Clean SourceTetris | P3 DeepRM | P3 SourceTetris | deg_M (95% CI) | Argmax change | TV distance | Delta clean | Delta P3 | Competency |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['M']} | {row['clean_deep_rm_mean']:.3f} | {row['clean_source_tetris_mean']:.3f} | "
            f"{row['p3_deep_rm_mean']:.3f} | {row['p3_source_tetris_mean']:.3f} | "
            f"{row['deg_mean']:+.3f} [{row['deg_ci_low']:+.3f}, {row['deg_ci_high']:+.3f}] | "
            f"{row['argmax_change_rate']:.3f} | {row['mean_total_variation']:.3f} | "
            f"{row['delta_clean_mean']:+.3f} | {row['delta_p3_mean']:+.3f} | "
            f"{row['competency_wins']}/30, p={row['competency_p_one_sided']:.4g}, pass={row['competency_gate_passed']} |"
        )
    lines.extend(
        [
            "",
            "## Hypothesis Verdict",
            "",
            f"Case: `{verdict['case']}`.",
            "",
            verdict["summary"],
            "",
            "Key comparisons:",
            "",
            f"- deg_3 / deg_10 = {payload['degradation_comparisons']['ratio_deg3_to_deg10']:.3f}.",
            f"- deg_1 / deg_10 = {payload['degradation_comparisons']['ratio_deg1_to_deg10']:.3f}.",
            f"- deg_3 - 2*deg_10 = {payload['degradation_comparisons']['mean_deg3_minus_2x_deg10']:+.3f} "
            f"[{payload['degradation_comparisons']['ci_low_deg3_minus_2x_deg10']:+.3f}, {payload['degradation_comparisons']['ci_high_deg3_minus_2x_deg10']:+.3f}].",
            f"- deg_1 - 4*deg_10 = {payload['degradation_comparisons']['mean_deg1_minus_4x_deg10']:+.3f} "
            f"[{payload['degradation_comparisons']['ci_low_deg1_minus_4x_deg10']:+.3f}, {payload['degradation_comparisons']['ci_high_deg1_minus_4x_deg10']:+.3f}].",
            "",
            "## Interpretation",
            "",
            scientific_interpretation(verdict["case"]),
            "",
            "## Artifacts",
            "",
        ]
    )
    for path in data_paths:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    for path in figure_paths:
        if path.is_relative_to(OUT_DIR) or path.is_relative_to(ROOT_FIG_DIR):
            lines.append(f"- `{path.relative_to(ROOT)}`")
    text = "\n".join(lines) + "\n"
    report = OUT_DIR / "experiment_c_results.md"
    report.write_text(text, encoding="utf-8")
    ROOT_REPORT.write_text(text, encoding="utf-8")

    table_md = TABLE_DIR / "experiment_c_results_table.md"
    table_lines = lines[lines.index("| M | Clean DeepRM | Clean SourceTetris | P3 DeepRM | P3 SourceTetris | deg_M (95% CI) | Argmax change | TV distance | Delta clean | Delta P3 | Competency |") :]
    end = table_lines.index("") if "" in table_lines else len(table_lines)
    table_md.write_text("\n".join(table_lines[:end]) + "\n", encoding="utf-8")
    return [report, ROOT_REPORT, table_md]


def write_manifest(payload: dict[str, object], paths: list[Path]) -> Path:
    manifest = {
        "experiment": "C",
        "payload_json": str(OUT_JSON.relative_to(ROOT)),
        "artifacts": [str(path.relative_to(ROOT)) for path in paths if path.exists()],
        "checkpoints": payload["checkpoints"],
        "verdict": payload["verdict"],
    }
    path = OUT_DIR / "experiment_c_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def summary_row(m: int, cell: dict[str, object]) -> dict[str, object]:
    return {
        "M": m,
        "action_dim": cell["action_dim"],
        "clean_deep_rm_mean": cell["clean_deep_rm_mean_slowdown"],
        "clean_source_tetris_mean": cell["clean_source_tetris_mean_slowdown"],
        "p3_deep_rm_mean": cell["p3_deep_rm_mean_slowdown"],
        "p3_source_tetris_mean": cell["p3_source_tetris_mean_slowdown"],
        "deg_mean": cell["deg_mean"],
        "deg_ci_low": cell["deg_ci_low"],
        "deg_ci_high": cell["deg_ci_high"],
        "argmax_change_rate": cell["argmax_change_rate"],
        "mean_total_variation": cell["mean_total_variation"],
        "mean_clean_argmax_probability_drop": cell["mean_clean_argmax_probability_drop"],
        "clean_action_entropy": cell["clean_action_entropy"],
        "delta_clean_mean": cell["delta_clean_mean"],
        "delta_clean_ci_low": cell["delta_clean_ci_low"],
        "delta_clean_ci_high": cell["delta_clean_ci_high"],
        "delta_p3_mean": cell["delta_p3_mean"],
        "delta_p3_ci_low": cell["delta_p3_ci_low"],
        "delta_p3_ci_high": cell["delta_p3_ci_high"],
        "competency_wins": cell["competency_wins"],
        "competency_p_one_sided": cell["competency_p_one_sided"],
        "competency_gate_passed": cell["competency_gate_passed"],
    }


def scientific_interpretation(case: str) -> str:
    if case == "H_C1":
        return (
            "Action redundancy is directly supported for DeepRM if reducing the visible action set "
            "causes FGSM to translate from action-level disruption into aggregate slowdown degradation."
        )
    if case == "H_C2":
        return (
            "The action-redundancy explanation is not supported as the operative mechanism. "
            "DeepRM's small aggregate FGSM effect persists even when visible scheduling alternatives "
            "are reduced, so the mechanism is likely elsewhere."
        )
    if case == "H_C3":
        return (
            "The result is mixed: moderate action-set reduction remains close to M=10, while "
            "the minimum-action setting is qualitatively different. The redundancy mechanism is "
            "interpreted as thresholded rather than smoothly monotone."
        )
    if case == "M1_not_learnable_H_C1_not_supported_for_competent_conditions":
        return (
            "The M=1 condition is treated as a non-competent branch, not as evidence about FGSM "
            "robustness. The interpretable mechanism test is therefore M=10 versus M=3. Under that "
            "comparison, the moderate action-space reduction does not increase aggregate FGSM degradation. "
            "Experiment C does not confirm action redundancy as the operative defence for DeepRM; the binary "
            "visible-action condition shows that the locked source-aligned training budget no longer produces "
            "a competent clean scheduler when the action set is reduced to one visible job plus wait."
        )
    return (
        "The result does not support a definitive mechanism claim. Experiment C is interpreted as a "
        "mechanism sensitivity whose pattern does not map cleanly to the preregistered alternatives."
    )


def compact_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "status": payload["status"],
        "m10_regression_within_5pct": payload["m10_regression_check"]["within_5pct"],
        "verdict": payload["verdict"],
        "rows": [
            {
                "M": m,
                "deg_mean": payload["cells"][str(m)]["deg_mean"],
                "deg_ci_low": payload["cells"][str(m)]["deg_ci_low"],
                "deg_ci_high": payload["cells"][str(m)]["deg_ci_high"],
                "competency_wins": payload["cells"][str(m)]["competency_wins"],
                "competency_gate_passed": payload["cells"][str(m)]["competency_gate_passed"],
            }
            for m in ("10", "3", "1")
        ],
    }


def binomial_greater_equal_pvalue(wins: int, n: int) -> float:
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("inf") if numerator >= 0 else float("-inf")
    return float(numerator / denominator)


def relative_error(observed: float, target: float) -> float:
    return float(abs(float(observed) - float(target)) / max(abs(float(target)), 1e-12))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dirs() -> None:
    for directory in (OUT_DIR, TABLE_DIR, FIG_DIR, DATA_DIR, ROOT_FIG_DIR, ROOT_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _artifact_group(path: Path) -> str:
    if "figures" in path.parts:
        return "paper/figures"
    if "tables" in path.parts:
        return "paper/tables"
    if "data" in path.parts:
        return "paper/data"
    if path.name.endswith(".md"):
        return "paper/reports"
    if path.name.startswith("EXPERIMENT_"):
        return "protocol/new_experiments"
    return "experiment_c"


if __name__ == "__main__":
    main()
