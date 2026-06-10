import numpy as np

from cisose_deeprm.model import DeepRMPolicy
from cisose_deeprm.protocol import DeepRMConfig, author_source_config
from cisose_deeprm.schedulers import PackerScheduler, SourceTetrisScheduler
from cisose_deeprm.simulator import DeepRMEnv, Machine
from cisose_deeprm.workload import Job, WorkloadTrace, sample_demand


def test_author_source_config_records_paper_source_parameter_mismatch():
    paper_config = DeepRMConfig()
    source_config = author_source_config(0.7)

    assert paper_config.state_shape == (20, 223, 1)
    assert source_config.state_shape == (20, 224, 1)
    assert DeepRMPolicy(paper_config).num_parameters == 89_451
    assert DeepRMPolicy(source_config).num_parameters == 89_851


def test_author_source_demands_are_integer_resource_slots():
    config = author_source_config(0.7)
    rng = np.random.default_rng(42)
    demands = [sample_demand(rng, config) for _ in range(200)]

    assert all(float(value).is_integer() for demand in demands for value in demand)
    assert all(1 <= min(demand) <= 2 for demand in demands)
    assert all(5 <= max(demand) <= 10 for demand in demands)


def test_author_source_reward_is_only_paid_when_time_advances():
    config = author_source_config(0.7)
    trace = WorkloadTrace(
        jobs=(Job(id=0, arrival_time=0, duration=2, demand=(5.0, 1.0)),),
        rate=1.0,
        tail_alpha=float("inf"),
        seed=1,
        horizon=1,
    )
    env = DeepRMEnv(trace, config=config, drain=True)

    _, allocate_reward, _, allocate_info = env.step(0)
    _, move_reward, _, move_info = env.step(config.visible_slots)

    assert allocate_info.status == "Allocate"
    assert allocate_info.time == 0
    assert allocate_reward == 0.0
    assert move_info.status == "MoveOn"
    assert move_info.time == 1
    assert move_reward == -0.5


def test_author_source_max_start_offset_is_exclusive_like_public_repo():
    paper_config = DeepRMConfig(time_horizon=3, resource_capacity=10.0, max_start_inclusive=True)
    source_config = DeepRMConfig(time_horizon=3, resource_capacity=10.0, max_start_inclusive=False)
    job = Job(id=0, arrival_time=0, duration=3, demand=(1.0, 1.0))

    assert Machine(paper_config).earliest_offset(job) == 0
    assert Machine(source_config).earliest_offset(job) is None


def test_author_source_tetris_label_matches_public_repo_packer_action():
    config = author_source_config(0.7)
    trace = WorkloadTrace(
        jobs=(
            Job(id=0, arrival_time=0, duration=3, demand=(5.0, 1.0)),
            Job(id=1, arrival_time=0, duration=1, demand=(1.0, 5.0)),
        ),
        rate=1.0,
        tail_alpha=float("inf"),
        seed=2,
        horizon=1,
    )
    env = DeepRMEnv(trace, config=config, drain=True)

    assert SourceTetrisScheduler().act(env) == PackerScheduler(source_dot=True).act(env)
