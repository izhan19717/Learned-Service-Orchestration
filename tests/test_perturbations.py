import numpy as np

from cisose_deeprm.evaluation import run_lagged_scheduler_episode
from cisose_deeprm.model import DeepRMPolicy
from cisose_deeprm.perturbations import (
    capture_snapshot,
    fgsm_observation,
    step_with_stale_identity_first_fit_fallback,
)
from cisose_deeprm.protocol import DeepRMConfig
from cisose_deeprm.schedulers import TetrisScheduler
from cisose_deeprm.simulator import DeepRMEnv
from cisose_deeprm.workload import Job, trace_from_jobs


def _trace():
    return trace_from_jobs(
        [
            Job(id=0, arrival_time=0, duration=2, demand=(0.3, 0.1)),
            Job(id=1, arrival_time=1, duration=1, demand=(0.1, 0.3)),
            Job(id=2, arrival_time=2, duration=3, demand=(0.4, 0.1)),
        ],
        rate=0.7,
        tail_alpha=float("inf"),
        seed=9,
    )


def test_stale_identity_mismatch_becomes_noop():
    env = DeepRMEnv(_trace())
    snapshot = capture_snapshot(env)
    env.step(0)
    _, _, _, info = env.step_with_stale_identity(0, snapshot.slot_job_ids)
    assert info.status == "MoveOn"


def test_lagged_tetris_completes_trace():
    metrics = run_lagged_scheduler_episode(TetrisScheduler(), _trace(), lag=1)
    assert metrics.mean_slowdown >= 1.0


def test_fgsm_observation_stays_in_unit_range():
    env = DeepRMEnv(_trace())
    policy = DeepRMPolicy(DeepRMConfig())
    adv = fgsm_observation(policy, env.observe(), epsilon=0.05)
    assert adv.shape == env.observe().shape
    assert np.min(adv) >= 0.0
    assert np.max(adv) <= 1.0


def test_first_fit_fallback_schedules_current_fit_when_stale_job_changed():
    trace = trace_from_jobs(
        [
            Job(id=0, arrival_time=0, duration=1, demand=(0.2, 0.1)),
            Job(id=1, arrival_time=1, duration=1, demand=(0.2, 0.1)),
        ],
        rate=0.7,
        tail_alpha=float("inf"),
        seed=10,
        horizon=2,
    )
    env = DeepRMEnv(trace)
    snapshot = capture_snapshot(env)
    env.step(0)
    env.step(env.config.visible_slots)

    _, _, _, info, mode = step_with_stale_identity_first_fit_fallback(
        env,
        0,
        snapshot.slot_job_ids,
    )

    assert mode == "first_fit_fallback_allocate"
    assert info.status == "Allocate"


def test_first_fit_fallback_does_not_help_stale_empty_slot_action():
    env = DeepRMEnv(trace_from_jobs([], rate=0.7, tail_alpha=float("inf"), seed=11, horizon=0))
    snapshot = capture_snapshot(env)

    _, _, _, info, mode = step_with_stale_identity_first_fit_fallback(
        env,
        0,
        snapshot.slot_job_ids,
    )

    assert mode == "stale_empty_slot_no_fallback"
    assert info.status == "MoveOn"


def test_first_fit_fallback_does_not_replace_same_identity_nonfit_action():
    env = DeepRMEnv(_trace())
    snapshot = capture_snapshot(env)
    job = env.visible_slots[0]
    assert job is not None
    env.machine.availability[:, :] = 0.0

    _, _, _, info, mode = step_with_stale_identity_first_fit_fallback(
        env,
        0,
        snapshot.slot_job_ids,
    )

    assert mode == "same_identity_MoveOn"
    assert info.status == "MoveOn"
