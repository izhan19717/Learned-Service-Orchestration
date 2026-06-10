from pathlib import Path

from cisose_rossi.actions import DEFAULT_ACTIONS
from cisose_rossi.checkpointing import load_model_based_checkpoint, save_model_based_checkpoint
from cisose_rossi.config import DEFAULT_CONFIG
from cisose_rossi.controllers import DynaQ2Controller, ModelBasedController, ThresholdHPAController
from cisose_rossi.evaluation import reproduction_gate_report, table_i_metrics
from cisose_rossi.perturbations import capped_pareto_cv2, minimum_bucket_flip_utilization
from cisose_rossi.simulator import RladSimulator, ServiceState, StepRecord, total_cost, valid_actions
from cisose_rossi.state import RladState, discretize_util
from cisose_rossi.workload import java_slow_profile_sequence, load_profile, profile_summary


def test_source_state_space_and_actions_match_extraction():
    assert DEFAULT_CONFIG.state_count == 1100
    assert [(a.replica_delta, a.cpu_delta) for a in DEFAULT_ACTIONS] == [
        (-1, 0),
        (0, 0),
        (1, 0),
        (0, -10),
        (0, 10),
    ]
    assert RladState(1, 0, 10).hash() == 0
    assert RladState(10, 10, 100).hash() == 1099


def test_java_slow_profile_sequence_uses_even_lines_twice():
    seq = java_slow_profile_sequence([1, 4, 8, 19, 33], steps=6)
    assert seq == (4.0, 4.0, 19.0, 19.0, 200.0, 200.0)


def test_official_profile_summary_is_stable():
    profile = load_profile(Path("external/rlad-core-simulator/data/profile.dat"))
    summary = profile_summary(profile)
    assert summary["count"] == 525533
    assert summary["min"] == 1.0
    assert summary["max"] == 1029.0
    assert abs(summary["mean"] - 329.508880) < 1e-6


def test_threshold_hpa_formula_matches_java_basic_action():
    controller = ThresholdHPAController()
    service = ServiceState(replicas=2, cpu=100, utilization=0.8)
    assert controller.pick_action(service).label == "scale_out"
    service.utilization = 0.1
    assert controller.pick_action(service).label == "scale_in"
    service = ServiceState(replicas=1, cpu=100, utilization=0.1)
    assert controller.pick_action(service).label == "no_op"


def test_valid_actions_respect_bounds():
    at_min = valid_actions(ServiceState(replicas=1, cpu=10))
    assert "scale_in" not in {a.label for a in at_min}
    assert "vertical_down" not in {a.label for a in at_min}
    at_max = valid_actions(ServiceState(replicas=10, cpu=100))
    assert "scale_out" not in {a.label for a in at_max}
    assert "vertical_up" not in {a.label for a in at_max}


def test_simulator_smoke_run_produces_costs():
    rates = java_slow_profile_sequence([1, 4, 8, 19, 33, 29], steps=8)
    records = RladSimulator().run(DynaQ2Controller(seed=1), rates, horizon=8)
    assert len(records) == 8
    assert total_cost(records) >= 0
    assert all(record.cpu_after > 0 for record in records)


def test_model_based_controller_is_paper_primary_and_runs():
    rates = java_slow_profile_sequence([1, 4, 8, 19, 33, 29], steps=4)
    controller = ModelBasedController()
    records = RladSimulator().run(controller, rates, horizon=4)
    assert len(records) == 4
    assert total_cost(records) >= 0
    assert controller.q.shape == (DEFAULT_CONFIG.state_count, DEFAULT_CONFIG.action_count)
    assert controller.transition_prob.shape == (DEFAULT_CONFIG.util_states, DEFAULT_CONFIG.util_states)


def test_model_based_controller_can_be_frozen_for_evaluation():
    controller = ModelBasedController()
    controller.freeze()
    before = controller.q.copy()
    rates = java_slow_profile_sequence([1, 4, 8, 19], steps=3)
    RladSimulator().run(controller, rates, horizon=3)
    assert (controller.q == before).all()


def test_bucket_flip_returns_zero_when_no_policy_flip():
    controller = DynaQ2Controller(seed=1)
    delta = minimum_bucket_flip_utilization(
        controller=controller,
        replicas=1,
        cpu=100,
        utilization=0.34,
        epsilon=0.05,
    )
    assert delta == 0.0


def test_discretize_util_clips_at_one():
    assert discretize_util(1.7) == 10


def test_table_i_metrics_use_interval_state_and_non_noop_actions():
    records = (
        StepRecord(
            time=0,
            input_rate=100.0,
            replicas_before=1,
            cpu_before=100,
            utilization=0.5,
            response_time=0.010,
            cost=0.1,
            sla_violated=False,
            action_index=1,
            action_label="no_op",
            replicas_after=1,
            cpu_after=100,
        ),
        StepRecord(
            time=1,
            input_rate=400.0,
            replicas_before=2,
            cpu_before=80,
            utilization=0.9,
            response_time=0.100,
            cost=0.9,
            sla_violated=True,
            action_index=2,
            action_label="scale_out",
            replicas_after=3,
            cpu_after=80,
        ),
    )
    observed = table_i_metrics(records)
    assert observed.rmax_violations_pct == 50.0
    assert observed.avg_cpu_utilization_pct == 70.0
    assert observed.avg_cpu_share_pct == 90.0
    assert observed.avg_containers == 1.5
    assert observed.median_response_ms == 55.0
    assert observed.adaptations_pct == 50.0


def test_reproduction_gate_report_requires_all_metrics_within_tolerance():
    records = (
        StepRecord(
            time=0,
            input_rate=100.0,
            replicas_before=1,
            cpu_before=100,
            utilization=0.5,
            response_time=0.010,
            cost=0.1,
            sla_violated=False,
            action_index=1,
            action_label="no_op",
            replicas_after=1,
            cpu_after=100,
        ),
    )
    report = reproduction_gate_report(
        records,
        tolerance=0.15,
        targets={
            "rmax_violations_pct": 1.0,
            "avg_cpu_utilization_pct": 50.0,
            "avg_cpu_share_pct": 100.0,
            "avg_containers": 1.0,
            "median_response_ms": 10.0,
            "adaptations_pct": 1.0,
        },
    )
    assert report["passed"] is False
    assert len(report["rows"]) == 6


def test_model_based_checkpoint_round_trip(tmp_path):
    controller = ModelBasedController()
    controller.state = RladState(replicas=3, util_bucket=4, cpu=70)
    controller.previous_action = DEFAULT_ACTIONS[2]
    controller.lambda_tps = 123.0
    controller.q[0, 1] = 7.5
    path = tmp_path / "rossi_model_based.npz"
    save_model_based_checkpoint(path, controller, metadata={"example": True})

    loaded, metadata = load_model_based_checkpoint(path, freeze=True)

    assert metadata["example"] is True
    assert loaded.learning_enabled is False
    assert loaded.state == controller.state
    assert loaded.previous_action == controller.previous_action
    assert loaded.lambda_tps == controller.lambda_tps
    assert loaded.q[0, 1] == 7.5


def test_simulator_observation_lag_is_controller_local():
    class Recorder:
        def __init__(self):
            self.previous_action = DEFAULT_ACTIONS[1]
            self.seen = []

        def update(self, service, cost, input_rate):
            return None

        def pick_action(self, service, observed_utilization=None):
            self.seen.append(observed_utilization)
            return DEFAULT_ACTIONS[1]

    controller = Recorder()
    rates = (100.0, 200.0, 300.0)
    RladSimulator().run(controller, rates, horizon=3, observation_lag_steps=1)

    assert controller.seen[0] == controller.seen[1]
    assert controller.seen[2] != controller.seen[1]


def test_observation_lag_can_drive_online_update_state():
    class Recorder:
        def __init__(self):
            self.previous_action = DEFAULT_ACTIONS[1]
            self.updated = []
            self.seen = []

        def update(self, service, cost, input_rate):
            self.updated.append(service.utilization)

        def pick_action(self, service, observed_utilization=None):
            self.seen.append(observed_utilization)
            return DEFAULT_ACTIONS[1]

    controller = Recorder()
    rates = (100.0, 200.0, 300.0)
    RladSimulator().run(
        controller,
        rates,
        horizon=3,
        observation_lag_steps=1,
        observation_applies_to_update=True,
    )

    assert controller.updated == controller.seen
    assert controller.updated[0] == controller.updated[1]
    assert controller.updated[2] != controller.updated[1]


def test_observed_utilizations_and_observation_lag_are_exclusive():
    controller = ThresholdHPAController()
    try:
        RladSimulator().run(
            controller,
            (100.0,),
            horizon=1,
            observed_utilizations=(0.1,),
            observation_lag_steps=1,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected mutually exclusive observation sources to fail")


def test_observation_transform_cannot_drive_update_state():
    controller = ThresholdHPAController()
    try:
        RladSimulator().run(
            controller,
            (100.0,),
            horizon=1,
            observation_transform=lambda _controller, _service, util: util,
            observation_applies_to_update=True,
        )
    except ValueError as exc:
        assert "observation_transform" in str(exc)
    else:
        raise AssertionError("expected observation_transform/update combination to fail")


def test_observation_transform_records_delta():
    controller = ThresholdHPAController()
    records = RladSimulator().run(
        controller,
        (100.0,),
        horizon=1,
        observation_transform=lambda _controller, _service, util: util + 0.02,
    )
    assert records[0].observed_utilization == records[0].utilization + 0.02
    assert abs(records[0].observation_delta - 0.02) < 1e-12


def test_capped_pareto_cv2_is_finite_and_monotone():
    assert capped_pareto_cv2(float("inf")) == 0.0
    assert capped_pareto_cv2(3.0) < capped_pareto_cv2(1.5)
    assert capped_pareto_cv2(1.5) < capped_pareto_cv2(1.2)


def test_service_time_cv2_increases_response_time_without_changing_utilization():
    det = RladSimulator(service_time_cv2=0.0)
    tail = RladSimulator(service_time_cv2=2.0)
    det.service.utilization = det.utilization(100.0)
    tail.service.utilization = tail.utilization(100.0)
    assert det.service.utilization == tail.service.utilization
    assert tail.response_time(100.0, tail.service.utilization) > det.response_time(
        100.0,
        det.service.utilization,
    )


def test_bucket_flip_returns_zero_for_overloaded_utilization_outside_budget():
    controller = ModelBasedController()
    delta = minimum_bucket_flip_utilization(
        controller=controller,
        replicas=1,
        cpu=100,
        utilization=2.0,
        epsilon=0.05,
    )
    assert delta == 0.0
