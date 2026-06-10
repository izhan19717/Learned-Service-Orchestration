import math

from cisose_deeprm.protocol import DeepRMConfig
from cisose_deeprm.workload import generate_trace, sample_duration


def test_generate_trace_has_exact_job_count_and_monotone_arrivals():
    trace = generate_trace(num_jobs=25, rate=0.7, seed=123)
    assert len(trace.jobs) == 25
    arrivals = [job.arrival_time for job in trace.jobs]
    assert arrivals == sorted(arrivals)


def test_infinite_tail_recovers_deeprm_long_range():
    config = DeepRMConfig(short_job_probability=0.0, long_job_probability=1.0)
    rng_seeded = __import__("numpy").random.default_rng(1)
    values = [sample_duration(rng_seeded, config, math.inf) for _ in range(200)]
    assert min(values) >= 10
    assert max(values) <= 15


def test_pareto_tail_is_clipped():
    config = DeepRMConfig(short_job_probability=0.0, long_job_probability=1.0)
    rng_seeded = __import__("numpy").random.default_rng(2)
    values = [sample_duration(rng_seeded, config, 1.2) for _ in range(500)]
    assert min(values) >= 10
    assert max(values) <= 100

