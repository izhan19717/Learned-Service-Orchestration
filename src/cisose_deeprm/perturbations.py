"""Evaluation-time perturbation helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from cisose_deeprm.model import DeepRMPolicy
from cisose_deeprm.schedulers import Scheduler
from cisose_deeprm.simulator import DeepRMEnv
from cisose_deeprm.workload import Job


@dataclass(frozen=True)
class StateSnapshot:
    observation: np.ndarray
    slot_job_ids: tuple[int | None, ...]
    visible_jobs: tuple[Job | None, ...]
    availability: np.ndarray


def capture_snapshot(env: DeepRMEnv) -> StateSnapshot:
    return StateSnapshot(
        observation=env.observe().copy(),
        slot_job_ids=env.slot_job_ids(),
        visible_jobs=tuple(env.visible_slots),
        availability=env.machine.availability.copy(),
    )


class LagBuffer:
    def __init__(self, env: DeepRMEnv, lag: int):
        self.lag = lag
        initial = capture_snapshot(env)
        self.buffer: deque[StateSnapshot] = deque([initial] * (lag + 1), maxlen=lag + 1)

    def current(self) -> StateSnapshot:
        return self.buffer[0]

    def update(self, env: DeepRMEnv) -> None:
        self.buffer.append(capture_snapshot(env))


def heuristic_action_on_snapshot(scheduler: Scheduler, snapshot: StateSnapshot, env: DeepRMEnv) -> int:
    """Choose a heuristic action from stale structured state without mutating env."""

    # Local import avoids widening the public scheduler protocol.
    from cisose_deeprm.schedulers import PackerScheduler, SJFScheduler, TetrisScheduler

    visible = snapshot.visible_jobs
    availability = snapshot.availability
    if isinstance(scheduler, SJFScheduler):
        best_idx = env.config.visible_slots
        best_score = -np.inf
        for idx, job in enumerate(visible):
            if job is None or not _fits_now(job, availability):
                continue
            score = 1.0 / job.duration
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx
    if isinstance(scheduler, (PackerScheduler, TetrisScheduler)):
        best_idx = env.config.visible_slots
        best_score = -np.inf
        free_now = availability[0, :]
        for idx, job in enumerate(visible):
            if job is None or not _fits_now(job, availability):
                continue
            packing = _packing_score(
                job.demand_array(),
                free_now,
                source_dot=scheduler.source_dot,
            )
            if isinstance(scheduler, PackerScheduler):
                score = packing
            else:
                score = scheduler.alpha * (1.0 / job.duration) + (1.0 - scheduler.alpha) * packing
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx
    return scheduler.act(env)


def policy_action_on_observation(
    policy: DeepRMPolicy,
    observation: np.ndarray,
    *,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
) -> int:
    state = torch.from_numpy(observation).unsqueeze(0).float()
    with torch.no_grad():
        logits = policy(state)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())


def fgsm_observation(policy: DeepRMPolicy, observation: np.ndarray, epsilon: float) -> np.ndarray:
    state = torch.from_numpy(observation).unsqueeze(0).float()
    state.requires_grad_(True)
    logits = policy(state)
    action = torch.argmax(logits, dim=-1)
    log_prob = torch.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze()
    loss = -log_prob
    loss.backward()
    perturbed = torch.clamp(state + epsilon * torch.sign(state.grad), 0.0, 1.0)
    return perturbed.detach().squeeze(0).numpy()


def first_fit_action_on_current_state(env: DeepRMEnv) -> int:
    """Return the first visible slot whose current job can allocate now."""

    for idx, job in enumerate(env.visible_slots):
        if job is not None and env.machine.can_allocate_now(job):
            return idx
    return env.config.visible_slots


def step_with_stale_identity_first_fit_fallback(
    env: DeepRMEnv,
    action: int,
    expected_slot_job_ids: tuple[int | None, ...],
) -> tuple[np.ndarray, float, bool, object, str]:
    """Execute a stale slot action with first-fit fallback on true current state.

    This is a methodology-sensitivity rule for lag experiments. It differs from
    the locked protocol's no-op fallback only when the stale action selected a
    slot that contained a job in the stale snapshot, but the corresponding
    current slot no longer contains that same job. In that case, the action is
    interpreted as an intent to schedule, and the environment schedules the
    first current visible job that fits. Explicit void decisions, actions
    targeting slots that were empty in the stale snapshot, and actions targeting
    the same current job remain governed by the ordinary simulator semantics.
    """

    if 0 <= action < env.config.visible_slots:
        expected = expected_slot_job_ids[action] if action < len(expected_slot_job_ids) else None
        if expected is None:
            observation, reward, done, info = env.step(action)
            return observation, reward, done, info, "stale_empty_slot_no_fallback"
        current = env.visible_slots[action]
        if current is not None and current.id == expected:
            observation, reward, done, info = env.step(action)
            return observation, reward, done, info, f"same_identity_{info.status}"
        fallback = first_fit_action_on_current_state(env)
        observation, reward, done, info = env.step(fallback)
        if info.status == "Allocate":
            return observation, reward, done, info, "first_fit_fallback_allocate"
        return observation, reward, done, info, "first_fit_fallback_no_fit"

    observation, reward, done, info = env.step(action)
    return observation, reward, done, info, "explicit_void"


def _fits_now(job: Job, availability: np.ndarray) -> bool:
    if job.duration > availability.shape[0]:
        return False
    window = availability[: job.duration, :]
    return bool(np.all(window - job.demand_array() >= -1e-12))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _packing_score(a: np.ndarray, b: np.ndarray, *, source_dot: bool) -> float:
    if source_dot:
        return float(np.dot(b, a))
    return _cosine(a, b)
