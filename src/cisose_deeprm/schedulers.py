"""Classical scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cisose_deeprm.simulator import DeepRMEnv


class Scheduler(Protocol):
    name: str

    def act(self, env: DeepRMEnv) -> int:
        ...


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _packing_score(a: np.ndarray, b: np.ndarray, *, source_dot: bool) -> float:
    if source_dot:
        return float(np.dot(b, a))
    return _cosine(a, b)


@dataclass(frozen=True)
class SJFScheduler:
    name: str = "SJF"

    def act(self, env: DeepRMEnv) -> int:
        best_idx = env.config.visible_slots
        best_score = -np.inf
        for idx, job in enumerate(env.visible_slots):
            if job is None or not env.machine.can_allocate_now(job):
                continue
            score = 1.0 / job.duration
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


@dataclass(frozen=True)
class PackerScheduler:
    name: str = "Packer"
    source_dot: bool = False

    def act(self, env: DeepRMEnv) -> int:
        best_idx = env.config.visible_slots
        best_score = -np.inf
        free_now = env.machine.availability[0, :]
        for idx, job in enumerate(env.visible_slots):
            if job is None or not env.machine.can_allocate_now(job):
                continue
            score = _packing_score(job.demand_array(), free_now, source_dot=self.source_dot)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


@dataclass(frozen=True)
class TetrisScheduler:
    alpha: float = 0.5
    name: str = "Tetris*"
    source_dot: bool = False

    def act(self, env: DeepRMEnv) -> int:
        best_idx = env.config.visible_slots
        best_score = -np.inf
        free_now = env.machine.availability[0, :]
        for idx, job in enumerate(env.visible_slots):
            if job is None or not env.machine.can_allocate_now(job):
                continue
            duration_score = 1.0 / job.duration
            packing_score = _packing_score(job.demand_array(), free_now, source_dot=self.source_dot)
            score = self.alpha * duration_score + (1.0 - self.alpha) * packing_score
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


@dataclass(frozen=True)
class SourceTetrisScheduler:
    """Official DeepRM source labels dot-product packer as "Tetris"."""

    name: str = "SourceTetris"

    def act(self, env: DeepRMEnv) -> int:
        return PackerScheduler(source_dot=True).act(env)


@dataclass
class RandomScheduler:
    name: str = "Random"
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, env: DeepRMEnv) -> int:
        return int(self._rng.integers(0, env.config.visible_slots + 1))
