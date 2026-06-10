"""Rossi/RLAD perturbation helpers."""

from __future__ import annotations

import math

from cisose_rossi.config import DEFAULT_CONFIG, RossiConfig
from cisose_rossi.state import RladState, discretize_util


def lagged_values(values: tuple[float, ...], lag_steps: int) -> tuple[float, ...]:
    if lag_steps <= 0:
        return values
    if not values:
        return values
    prefix = (values[0],) * lag_steps
    return (prefix + values)[: len(values)]


def minimum_bucket_flip_utilization(
    *,
    controller,
    replicas: int,
    cpu: int,
    utilization: float,
    epsilon: float,
    config: RossiConfig = DEFAULT_CONFIG,
) -> float:
    """Return the smallest signed utilization perturbation that flips action.

    If no boundary within the epsilon budget changes the greedy action, return
    0.0. Replica count and CPU allocation are held fixed by design.
    """

    clean_bucket = discretize_util(utilization, config)
    clean_state = RladState(replicas, clean_bucket, cpu)
    clean_action = controller.greedy_action(clean_state)
    candidates: list[tuple[float, float]] = []
    tiny = 1e-9
    for boundary_idx in range(1, config.util_states):
        boundary = boundary_idx * config.util_quantum
        for target in (boundary - tiny, boundary + tiny):
            if target < 0.0 or target > config.util_max:
                continue
            delta = target - utilization
            if abs(delta) <= epsilon and discretize_util(target, config) != clean_bucket:
                state = RladState(replicas, discretize_util(target, config), cpu)
                if controller.greedy_action(state).index != clean_action.index:
                    candidates.append((abs(delta), delta))
    if not candidates:
        return 0.0
    candidates.sort(key=lambda item: item[0])
    return float(candidates[0][1])


def capped_pareto_cv2(alpha: float, *, cap_ratio: float = 100.0) -> float:
    """Return CV^2 for a mean-normalized capped Pareto tail.

    RLAD's queueing model consumes a service-time second moment rather than
    sampled service times. For alpha <= 2 an uncapped Pareto has infinite
    variance, so the Rossi P2 perturbation uses the same directional idea as
    the DeepRM tail stressor but caps the relative service-time multiplier.
    """

    if math.isinf(alpha):
        return 0.0
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1 for a finite capped-Pareto mean")
    if cap_ratio <= 1.0:
        raise ValueError("cap_ratio must be > 1")
    first = _capped_pareto_raw_moment(alpha, cap_ratio, order=1)
    second = _capped_pareto_raw_moment(alpha, cap_ratio, order=2)
    return float(second / (first * first) - 1.0)


def _capped_pareto_raw_moment(alpha: float, cap_ratio: float, *, order: int) -> float:
    if math.isclose(alpha, float(order)):
        return float(alpha * math.log(cap_ratio) + 1.0)
    exponent = float(order) - alpha
    return float(alpha / exponent * (cap_ratio**exponent - 1.0) + cap_ratio**exponent)
