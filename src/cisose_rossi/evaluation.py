"""Rossi/RLAD evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cisose_common.stats import paired_bootstrap_ci, sign_flip_pvalues
from cisose_rossi.config import DEFAULT_CONFIG, RossiConfig
from cisose_rossi.controllers import DynaQ2Controller, ModelBasedController, ThresholdHPAController
from cisose_rossi.simulator import RladSimulator, StepRecord, total_cost


TABLE_I_MODEL_BASED_TARGETS = {
    "rmax_violations_pct": 2.37,
    "avg_cpu_utilization_pct": 60.54,
    "avg_cpu_share_pct": 87.62,
    "avg_containers": 2.53,
    "median_response_ms": 10.39,
    "adaptations_pct": 39.67,
}


@dataclass(frozen=True)
class RossiMetrics:
    total_cost: float
    sla_violation_rate: float
    mean_response_time: float
    action_churn: int


@dataclass(frozen=True)
class TableIMetrics:
    rmax_violations_pct: float
    avg_cpu_utilization_pct: float
    avg_cpu_share_pct: float
    avg_containers: float
    median_response_ms: float
    adaptations_pct: float


@dataclass(frozen=True)
class RossiPairedResult:
    differences: tuple[float, ...]
    mean_difference: float
    ci_low: float
    ci_high: float
    p_less_than_zero: float
    p_greater_than_zero: float


def metrics(records: tuple[StepRecord, ...]) -> RossiMetrics:
    if not records:
        return RossiMetrics(0.0, 0.0, 0.0, 0)
    actions = [record.action_index for record in records]
    churn = sum(1 for prev, curr in zip(actions, actions[1:], strict=False) if prev != curr)
    return RossiMetrics(
        total_cost=total_cost(records),
        sla_violation_rate=float(np.mean([record.sla_violated for record in records])),
        mean_response_time=float(np.mean([record.response_time for record in records])),
        action_churn=churn,
    )


def table_i_metrics(records: tuple[StepRecord, ...]) -> TableIMetrics:
    """Compute Rossi 2019 Table I-style metrics.

    The Java simulator records SLA, response time, utilization, and container
    count before the newly selected action is applied. CPU share is therefore
    taken from `cpu_before`, matching the configuration that produced the
    measured interval. Adaptation percentage is the share of selected non-noop
    actions over the same decision ticks.
    """

    if not records:
        return TableIMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return TableIMetrics(
        rmax_violations_pct=float(np.mean([r.sla_violated for r in records]) * 100.0),
        avg_cpu_utilization_pct=float(np.mean([r.utilization for r in records]) * 100.0),
        avg_cpu_share_pct=float(np.mean([r.cpu_before for r in records])),
        avg_containers=float(np.mean([r.replicas_before for r in records])),
        median_response_ms=float(np.median([r.response_time for r in records]) * 1000.0),
        adaptations_pct=float(np.mean([r.action_index != 1 for r in records]) * 100.0),
    )


def reproduction_gate_report(
    records: tuple[StepRecord, ...],
    *,
    tolerance: float = 0.15,
    targets: dict[str, float] = TABLE_I_MODEL_BASED_TARGETS,
) -> dict[str, object]:
    observed = table_i_metrics(records).__dict__
    rows = []
    passed = True
    for metric, target in targets.items():
        value = float(observed[metric])
        relative_error = abs(value - target) / abs(target)
        within_tolerance = relative_error <= tolerance
        passed = passed and within_tolerance
        rows.append(
            {
                "metric": metric,
                "observed": value,
                "target": float(target),
                "relative_error": float(relative_error),
                "within_15pct": within_tolerance,
            }
        )
    return {
        "gate_name": "rossi_table_i_performance_weighted_5_action_model_based",
        "tolerance": tolerance,
        "passed": passed,
        "observed": observed,
        "targets": dict(targets),
        "rows": rows,
        "metric_definitions": {
            "rmax_violations_pct": "mean(response_time > 50 ms) * 100",
            "avg_cpu_utilization_pct": "mean(theoretical utilization before action) * 100",
            "avg_cpu_share_pct": "mean(cpu allocation percentage before action)",
            "avg_containers": "mean(container count before action)",
            "median_response_ms": "median(response_time) * 1000",
            "adaptations_pct": "mean(selected action is not no-op) * 100",
        },
    }


def smoke_compare(
    input_rates: tuple[float, ...],
    *,
    seed: int,
    horizon: int,
    agent_type: str = "model_based",
    config: RossiConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    if agent_type == "model_based":
        rl = ModelBasedController(config)
    elif agent_type == "dynaq2":
        rl = DynaQ2Controller(config, seed=seed)
    else:
        raise ValueError(f"unknown Rossi agent_type={agent_type!r}")
    hpa = ThresholdHPAController(config)
    rl_records = RladSimulator(config).run(rl, input_rates, horizon=horizon)
    hpa_records = RladSimulator(config).run(hpa, input_rates, horizon=horizon)
    rl_metrics = metrics(rl_records)
    hpa_metrics = metrics(hpa_records)
    return {
        "rossi": rl_metrics.__dict__,
        "hpa": hpa_metrics.__dict__,
        "agent_type": agent_type,
        "delta_hpa_minus_rossi": hpa_metrics.total_cost - rl_metrics.total_cost,
    }


def paired_result(
    baseline: tuple[RossiMetrics, ...],
    method: tuple[RossiMetrics, ...],
    *,
    seed: int,
) -> RossiPairedResult:
    diffs = tuple(b.total_cost - m.total_cost for b, m in zip(baseline, method, strict=True))
    ci_low, ci_high = paired_bootstrap_ci(diffs, seed=seed)
    p_less, p_greater = sign_flip_pvalues(diffs, seed=seed + 1)
    return RossiPairedResult(
        differences=diffs,
        mean_difference=float(np.mean(diffs)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_less_than_zero=p_less,
        p_greater_than_zero=p_greater,
    )
