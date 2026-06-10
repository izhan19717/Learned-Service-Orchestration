from cisose_deeprm.protocol import DeepRMConfig
from cisose_deeprm.schedulers import SJFScheduler, TetrisScheduler
from cisose_deeprm.simulator import DeepRMEnv
from cisose_deeprm.workload import Job, trace_from_jobs


def _tiny_trace():
    jobs = [
        Job(id=0, arrival_time=0, duration=2, demand=(0.3, 0.1)),
        Job(id=1, arrival_time=0, duration=1, demand=(0.1, 0.3)),
        Job(id=2, arrival_time=1, duration=3, demand=(0.4, 0.1)),
    ]
    return trace_from_jobs(jobs, rate=0.7, tail_alpha=float("inf"), seed=7)


def test_observation_shape_matches_parameter_count_contract():
    config = DeepRMConfig()
    env = DeepRMEnv(_tiny_trace(), config=config)
    assert env.observe().shape == config.state_shape
    assert config.state_dim * 20 + 20 + 20 * config.action_dim + config.action_dim == 89_451


def test_allocate_does_not_advance_time():
    env = DeepRMEnv(_tiny_trace())
    _, _, _, info = env.step(0)
    assert info.status == "Allocate"
    assert info.time == 0


def test_void_advances_time():
    env = DeepRMEnv(_tiny_trace())
    _, _, _, info = env.step(env.config.visible_slots)
    assert info.status == "MoveOn"
    assert info.time == 1


def test_sjf_completes_all_jobs():
    env = DeepRMEnv(_tiny_trace())
    scheduler = SJFScheduler()
    steps = 0
    while not env.done:
        env.step(scheduler.act(env))
        steps += 1
        assert steps < 100
    assert len(env.completed) == 3
    assert env.mean_slowdown() >= 1.0


def test_tetris_returns_valid_action():
    env = DeepRMEnv(_tiny_trace())
    action = TetrisScheduler().act(env)
    assert 0 <= action <= env.config.visible_slots

