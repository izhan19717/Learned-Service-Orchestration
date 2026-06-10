"""Workload generation for DeepRM-style traces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, isinf
from typing import Iterable

import numpy as np

from cisose_deeprm.protocol import DeepRMConfig


@dataclass(frozen=True)
class Job:
    id: int
    arrival_time: int
    duration: int
    demand: tuple[float, float]
    start_time: int | None = None
    finish_time: int | None = None

    @property
    def slowdown(self) -> float:
        if self.finish_time is None:
            raise ValueError("slowdown requested before job completed")
        return (self.finish_time - self.arrival_time) / float(self.duration)

    def with_times(self, *, start_time: int, finish_time: int) -> "Job":
        return replace(self, start_time=start_time, finish_time=finish_time)

    def demand_array(self) -> np.ndarray:
        return np.asarray(self.demand, dtype=np.float64)

    def demand_bins(self, config: DeepRMConfig) -> np.ndarray:
        bins = np.ceil(self.demand_array() / config.resource_capacity * config.resource_bins).astype(int)
        return np.clip(bins, 1, config.resource_bins)


@dataclass(frozen=True)
class WorkloadTrace:
    jobs: tuple[Job, ...]
    rate: float
    tail_alpha: float
    seed: int
    horizon: int

    def arrivals_by_time(self) -> dict[int, list[Job]]:
        out: dict[int, list[Job]] = {}
        for job in self.jobs:
            out.setdefault(job.arrival_time, []).append(job)
        return out


def sample_duration(rng: np.random.Generator, config: DeepRMConfig, tail_alpha: float) -> int:
    if rng.random() < config.short_job_probability:
        return int(rng.integers(config.short_duration_min, config.short_duration_max + 1))
    if isinf(tail_alpha):
        return int(rng.integers(config.long_duration_min, config.long_duration_max + 1))
    u = max(rng.random(), np.finfo(float).tiny)
    raw = config.tail_x_min * ((1.0 - u) ** (-1.0 / tail_alpha))
    return int(np.clip(round(raw), config.tail_x_min, config.tail_x_max))


def sample_demand(rng: np.random.Generator, config: DeepRMConfig) -> tuple[float, float]:
    dominant = int(rng.integers(0, config.num_resources))
    demand = np.zeros(config.num_resources, dtype=np.float64)
    for idx in range(config.num_resources):
        if idx == dominant:
            if config.demand_mode == "source_discrete":
                demand[idx] = rng.integers(
                    int(config.dominant_demand_min), int(config.dominant_demand_max) + 1
                )
            else:
                demand[idx] = rng.uniform(config.dominant_demand_min, config.dominant_demand_max)
        else:
            if config.demand_mode == "source_discrete":
                demand[idx] = rng.integers(
                    int(config.nondominant_demand_min), int(config.nondominant_demand_max) + 1
                )
            else:
                demand[idx] = rng.uniform(config.nondominant_demand_min, config.nondominant_demand_max)
    return (float(demand[0]), float(demand[1]))


def generate_trace(
    *,
    num_jobs: int,
    rate: float,
    seed: int,
    config: DeepRMConfig | None = None,
    tail_alpha: float = float("inf"),
) -> WorkloadTrace:
    """Generate a finite Bernoulli-arrival trace with exactly ``num_jobs`` jobs."""

    config = config or DeepRMConfig()
    rng = np.random.default_rng(seed)
    jobs: list[Job] = []
    t = 0
    while len(jobs) < num_jobs:
        if rng.random() < rate:
            jobs.append(
                Job(
                    id=len(jobs),
                    arrival_time=t,
                    duration=sample_duration(rng, config, tail_alpha),
                    demand=sample_demand(rng, config),
                )
            )
        t += 1
    horizon = (jobs[-1].arrival_time + 1) if jobs else 0
    return WorkloadTrace(jobs=tuple(jobs), rate=rate, tail_alpha=tail_alpha, seed=seed, horizon=horizon)


def generate_time_trace(
    *,
    horizon: int,
    rate: float,
    seed: int,
    config: DeepRMConfig | None = None,
    tail_alpha: float = float("inf"),
) -> WorkloadTrace:
    """Generate arrivals over a fixed time horizon, as in DeepRM training."""

    config = config or DeepRMConfig()
    rng = np.random.default_rng(seed)
    jobs: list[Job] = []
    for t in range(horizon):
        if rng.random() < rate:
            jobs.append(
                Job(
                    id=len(jobs),
                    arrival_time=t,
                    duration=sample_duration(rng, config, tail_alpha),
                    demand=sample_demand(rng, config),
                )
            )
    return WorkloadTrace(jobs=tuple(jobs), rate=rate, tail_alpha=tail_alpha, seed=seed, horizon=horizon)


def trace_from_jobs(
    jobs: Iterable[Job],
    *,
    rate: float,
    tail_alpha: float,
    seed: int,
    horizon: int | None = None,
) -> WorkloadTrace:
    jobs_tuple = tuple(jobs)
    if horizon is None:
        horizon = max((job.arrival_time for job in jobs_tuple), default=-1) + 1
    return WorkloadTrace(jobs=jobs_tuple, rate=rate, tail_alpha=tail_alpha, seed=seed, horizon=horizon)
