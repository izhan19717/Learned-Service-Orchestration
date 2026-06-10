"""Evaluation loops and statistical tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from cisose_deeprm.protocol import BOOTSTRAP_RESAMPLES, EVAL_NUM_SEEDS, EVAL_TRACE_JOBS, SIGN_FLIP_RESAMPLES, DeepRMConfig
from cisose_deeprm.model import DeepRMPolicy, DeepRMScheduler
from cisose_deeprm.perturbations import (
    LagBuffer,
    fgsm_observation,
    heuristic_action_on_snapshot,
    policy_action_on_observation,
)
from cisose_deeprm.schedulers import PackerScheduler, Scheduler, SJFScheduler, TetrisScheduler
from cisose_deeprm.simulator import DeepRMEnv
from cisose_deeprm.workload import WorkloadTrace, generate_trace


@dataclass(frozen=True)
class EpisodeMetrics:
    mean_slowdown: float
    p95_completion_time: float
    makespan: int
    steps: int


@dataclass(frozen=True)
class PairedResult:
    differences: tuple[float, ...]
    mean_difference: float
    ci_low: float
    ci_high: float
    p_less_than_zero: float
    p_greater_than_zero: float


@dataclass(frozen=True)
class CleanEvaluationSummary:
    checkpoint_path: str
    load: float
    num_seeds: int
    trace_jobs: int
    seed: int
    policy_mode: str
    policy_seed: int
    max_steps: int
    method_means: dict[str, float]
    comparisons: dict[str, PairedResult]
    strict_gate_passed: bool
    mean_gate_passed: bool


def run_episode(
    scheduler: Scheduler,
    trace: WorkloadTrace,
    *,
    config: DeepRMConfig | None = None,
    max_steps: int = 100_000,
) -> EpisodeMetrics:
    env = DeepRMEnv(trace, config=config)
    steps = 0
    while not env.done:
        action = scheduler.act(env)
        env.step(action)
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"episode exceeded max_steps={max_steps}")
    return EpisodeMetrics(
        mean_slowdown=env.mean_slowdown(),
        p95_completion_time=env.p95_completion_time(),
        makespan=env.makespan(),
        steps=steps,
    )


def run_lagged_scheduler_episode(
    scheduler: Scheduler,
    trace: WorkloadTrace,
    *,
    lag: int,
    config: DeepRMConfig | None = None,
    max_steps: int = 100_000,
) -> EpisodeMetrics:
    env = DeepRMEnv(trace, config=config)
    lag_buffer = LagBuffer(env, lag)
    steps = 0
    while not env.done:
        snapshot = lag_buffer.current()
        action = heuristic_action_on_snapshot(scheduler, snapshot, env)
        env.step_with_stale_identity(action, snapshot.slot_job_ids)
        lag_buffer.update(env)
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"lagged episode exceeded max_steps={max_steps}")
    return EpisodeMetrics(
        mean_slowdown=env.mean_slowdown(),
        p95_completion_time=env.p95_completion_time(),
        makespan=env.makespan(),
        steps=steps,
    )


def run_lagged_policy_episode(
    policy: DeepRMPolicy,
    trace: WorkloadTrace,
    *,
    lag: int,
    config: DeepRMConfig | None = None,
    policy_deterministic: bool = True,
    policy_generator: torch.Generator | None = None,
    max_steps: int = 100_000,
) -> EpisodeMetrics:
    env = DeepRMEnv(trace, config=config)
    lag_buffer = LagBuffer(env, lag)
    steps = 0
    while not env.done:
        snapshot = lag_buffer.current()
        action = policy_action_on_observation(
            policy,
            snapshot.observation,
            deterministic=policy_deterministic,
            generator=policy_generator,
        )
        env.step_with_stale_identity(action, snapshot.slot_job_ids)
        lag_buffer.update(env)
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"lagged policy episode exceeded max_steps={max_steps}")
    return EpisodeMetrics(
        mean_slowdown=env.mean_slowdown(),
        p95_completion_time=env.p95_completion_time(),
        makespan=env.makespan(),
        steps=steps,
    )


def run_adversarial_policy_episode(
    policy: DeepRMPolicy,
    trace: WorkloadTrace,
    *,
    epsilon: float,
    config: DeepRMConfig | None = None,
    policy_deterministic: bool = True,
    policy_generator: torch.Generator | None = None,
    max_steps: int = 100_000,
) -> EpisodeMetrics:
    env = DeepRMEnv(trace, config=config)
    steps = 0
    while not env.done:
        observation = fgsm_observation(policy, env.observe(), epsilon)
        action = policy_action_on_observation(
            policy,
            observation,
            deterministic=policy_deterministic,
            generator=policy_generator,
        )
        env.step(action)
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"adversarial policy episode exceeded max_steps={max_steps}")
    return EpisodeMetrics(
        mean_slowdown=env.mean_slowdown(),
        p95_completion_time=env.p95_completion_time(),
        makespan=env.makespan(),
        steps=steps,
    )


def evaluate_scheduler(
    scheduler: Scheduler,
    traces: Sequence[WorkloadTrace],
    *,
    config: DeepRMConfig | None = None,
    max_steps: int = 100_000,
) -> tuple[EpisodeMetrics, ...]:
    return tuple(run_episode(scheduler, trace, config=config, max_steps=max_steps) for trace in traces)


def generate_eval_traces(
    *,
    load: float,
    num_seeds: int = EVAL_NUM_SEEDS,
    trace_jobs: int = EVAL_TRACE_JOBS,
    seed: int,
    config: DeepRMConfig | None = None,
) -> tuple[WorkloadTrace, ...]:
    config = config or DeepRMConfig(primary_load=load)
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(num_seeds)
    return tuple(
        generate_trace(
            num_jobs=trace_jobs,
            rate=load,
            seed=int(child.generate_state(1)[0]),
            config=config,
        )
        for child in child_seeds
    )


def evaluate_clean_policy(
    policy: DeepRMPolicy,
    traces: Sequence[WorkloadTrace],
    *,
    checkpoint_path: str,
    load: float,
    seed: int,
    config: DeepRMConfig | None = None,
    policy_deterministic: bool = False,
    policy_seed: int | None = None,
    max_steps: int = 100_000,
) -> tuple[CleanEvaluationSummary, dict[str, tuple[EpisodeMetrics, ...]]]:
    config = config or DeepRMConfig(primary_load=load)
    resolved_policy_seed = seed if policy_seed is None else policy_seed
    policy_generator = torch.Generator().manual_seed(resolved_policy_seed)
    schedulers: tuple[Scheduler, ...] = (
        DeepRMScheduler(
            policy=policy,
            deterministic=policy_deterministic,
            generator=None if policy_deterministic else policy_generator,
        ),
        SJFScheduler(),
        PackerScheduler(source_dot=True),
        TetrisScheduler(source_dot=True),
    )
    metrics_by_method: dict[str, tuple[EpisodeMetrics, ...]] = {}
    for scheduler in schedulers:
        metrics_by_method[scheduler.name] = evaluate_scheduler(
            scheduler,
            traces,
            config=config,
            max_steps=max_steps,
        )

    deep_metrics = metrics_by_method["DeepRM"]
    comparisons = {
        name: paired_result(metrics, deep_metrics, seed=seed + idx)
        for idx, (name, metrics) in enumerate(metrics_by_method.items(), start=1)
        if name != "DeepRM"
    }
    method_means = {
        name: float(np.mean([metric.mean_slowdown for metric in metrics]))
        for name, metrics in metrics_by_method.items()
    }
    strict_gate_passed = all(result.ci_low > 0.0 for result in comparisons.values())
    mean_gate_passed = all(result.mean_difference > 0.0 for result in comparisons.values())
    summary = CleanEvaluationSummary(
        checkpoint_path=checkpoint_path,
        load=load,
        num_seeds=len(traces),
        trace_jobs=len(traces[0].jobs) if traces else 0,
        seed=seed,
        policy_mode="deterministic_argmax" if policy_deterministic else "stochastic_sample",
        policy_seed=resolved_policy_seed,
        max_steps=max_steps,
        method_means=method_means,
        comparisons=comparisons,
        strict_gate_passed=strict_gate_passed,
        mean_gate_passed=mean_gate_passed,
    )
    return summary, metrics_by_method


def clean_evaluation_payload(
    summary: CleanEvaluationSummary,
    metrics_by_method: dict[str, tuple[EpisodeMetrics, ...]],
) -> dict[str, object]:
    return {
        "summary": {
            **asdict(summary),
            "comparisons": {
                name: asdict(result) for name, result in summary.comparisons.items()
            },
        },
        "per_seed_metrics": {
            name: [asdict(metric) for metric in metrics]
            for name, metrics in metrics_by_method.items()
        },
    }


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    diffs = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def sign_flip_pvalues(
    differences: Sequence[float],
    *,
    seed: int,
    resamples: int = SIGN_FLIP_RESAMPLES,
) -> tuple[float, float]:
    diffs = np.asarray(differences, dtype=np.float64)
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, len(diffs)))
    null_means = (signs * diffs).mean(axis=1)
    p_less = (np.count_nonzero(null_means <= observed) + 1.0) / (resamples + 1.0)
    p_greater = (np.count_nonzero(null_means >= observed) + 1.0) / (resamples + 1.0)
    return (float(p_less), float(p_greater))


def paired_result(
    baseline_metrics: Sequence[EpisodeMetrics],
    method_metrics: Sequence[EpisodeMetrics],
    *,
    seed: int,
) -> PairedResult:
    if len(baseline_metrics) != len(method_metrics):
        raise ValueError("paired metrics must have the same length")
    differences = tuple(
        base.mean_slowdown - method.mean_slowdown
        for base, method in zip(baseline_metrics, method_metrics, strict=True)
    )
    ci_low, ci_high = paired_bootstrap_ci(differences, seed=seed)
    p_less, p_greater = sign_flip_pvalues(differences, seed=seed + 1)
    return PairedResult(
        differences=differences,
        mean_difference=float(np.mean(differences)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_less_than_zero=p_less,
        p_greater_than_zero=p_greater,
    )


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, float]:
    """Return Holm-adjusted p-values for a small family of tests."""

    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, pvalue) in enumerate(ordered, start=1):
        adj = min(1.0, (m - rank + 1) * pvalue)
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted
