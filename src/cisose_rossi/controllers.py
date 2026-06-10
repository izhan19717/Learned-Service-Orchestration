"""Tabular Rossi/RLAD controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cisose_rossi.actions import Action, DEFAULT_ACTIONS
from cisose_rossi.config import DEFAULT_CONFIG, RossiConfig
from cisose_rossi.simulator import ServiceState, valid_actions
from cisose_rossi.state import RladState, discretize_util


class ThresholdHPAController:
    """RLAD simulator-bundled threshold comparator.

    This class is retained for backward compatibility with the completed Rossi
    artifacts. New paper-facing artifacts should report it as
    `bundled_threshold`, not as Kubernetes HPA.
    """

    def __init__(self, config: RossiConfig = DEFAULT_CONFIG, threshold: float = 0.7):
        self.config = config
        self.threshold = threshold
        self.previous_action = DEFAULT_ACTIONS[1]

    def update(self, service: ServiceState, cost: float, input_rate: float) -> None:
        return None

    def pick_action(self, service: ServiceState, observed_utilization: float | None = None) -> Action:
        util = service.utilization if observed_utilization is None else observed_utilization
        if util > self.threshold and service.replicas < self.config.max_replication:
            self.previous_action = DEFAULT_ACTIONS[2]
            return self.previous_action
        scale_in_threshold = (
            0.75 * self.threshold * (service.replicas - 1) / service.replicas
            if service.replicas > 1
            else -1.0
        )
        if util < scale_in_threshold:
            self.previous_action = DEFAULT_ACTIONS[0]
            return self.previous_action
        self.previous_action = DEFAULT_ACTIONS[1]
        return self.previous_action


class HPAv2Controller:
    """Kubernetes autoscaling/v2-equivalent horizontal controller.

    The RLAD simulator's native actions are one-replica increments, but HPA
    applies a desired replica recommendation. This controller therefore emits a
    horizontal action whose replica delta moves directly to the HPA-applied
    recommendation at sync boundaries and emits no-op between syncs.
    """

    def __init__(
        self,
        config: RossiConfig = DEFAULT_CONFIG,
        *,
        sync_period_seconds: int = 15,
        target_utilization: float = 0.50,
        tolerance: float = 0.10,
        min_replicas: int = 1,
        max_replicas: int | None = None,
        scale_down_stabilization_seconds: int = 300,
        scale_up_stabilization_seconds: int = 0,
    ):
        self.config = config
        self.sync_period_seconds = sync_period_seconds
        self.target_utilization = target_utilization
        self.tolerance = tolerance
        self.min_replicas = min_replicas
        self.max_replicas = config.max_replication if max_replicas is None else max_replicas
        self.scale_down_stabilization_seconds = scale_down_stabilization_seconds
        self.scale_up_stabilization_seconds = scale_up_stabilization_seconds
        self.previous_action = DEFAULT_ACTIONS[1]
        self.t = 0
        self.recommendation_history: list[tuple[int, int]] = []

    def update(self, service: ServiceState, cost: float, input_rate: float) -> None:
        return None

    def pick_action(self, service: ServiceState, observed_utilization: float | None = None) -> Action:
        if self.t % self.sync_period_seconds != 0:
            self.t += 1
            self.previous_action = DEFAULT_ACTIONS[1]
            return self.previous_action

        util = service.utilization if observed_utilization is None else observed_utilization
        current = service.replicas
        recommendation = self._desired_replicas(current, float(util))
        self.recommendation_history.append((self.t, recommendation))
        applied = self._stabilized_recommendation(current, recommendation)
        applied = self._apply_rate_policies(current, applied)
        applied = min(self.max_replicas, max(self.min_replicas, applied))
        delta = applied - current
        if delta > 0:
            action = Action(2, f"hpa_scale_out_to_{applied}", delta, 0)
        elif delta < 0:
            action = Action(0, f"hpa_scale_in_to_{applied}", delta, 0)
        else:
            action = DEFAULT_ACTIONS[1]
        self.previous_action = action
        self.t += 1
        return action

    def _desired_replicas(self, current: int, utilization: float) -> int:
        if self.target_utilization <= 0.0:
            raise ValueError("target_utilization must be positive")
        ratio = utilization / self.target_utilization
        if abs(ratio - 1.0) < self.tolerance:
            return current
        desired = int(np.ceil(current * ratio))
        return min(self.max_replicas, max(self.min_replicas, desired))

    def _stabilized_recommendation(self, current: int, recommendation: int) -> int:
        if recommendation < current:
            window = self.scale_down_stabilization_seconds
            recent = self._recent_recommendations(window)
            return max(recent) if recent else recommendation
        if recommendation > current:
            window = self.scale_up_stabilization_seconds
            recent = self._recent_recommendations(window)
            return min(recent) if recent else recommendation
        return recommendation

    def _recent_recommendations(self, window_seconds: int) -> list[int]:
        lower = self.t - window_seconds
        self.recommendation_history = [
            (tick, value)
            for tick, value in self.recommendation_history
            if tick >= self.t - max(self.scale_down_stabilization_seconds, self.scale_up_stabilization_seconds)
        ]
        return [value for tick, value in self.recommendation_history if tick >= lower]

    def _apply_rate_policies(self, current: int, recommendation: int) -> int:
        if recommendation > current:
            max_percent_delta = max(1, int(np.ceil(current * 1.00)))
            max_pods_delta = 4
            max_delta = max(max_percent_delta, max_pods_delta)
            return min(recommendation, current + max_delta)
        if recommendation < current:
            max_percent_delta = max(1, int(np.ceil(current * 1.00)))
            return max(recommendation, current - max_percent_delta)
        return recommendation


@dataclass
class DynaQElement:
    state: RladState
    action: Action
    next_state: RladState
    cost: float


class DynaQ2Controller:
    """Default `AGENT_DYNAQ2` controller from the official Java simulator."""

    def __init__(
        self,
        config: RossiConfig = DEFAULT_CONFIG,
        *,
        seed: int = 0,
        preserve_java_quirk: bool | None = None,
        learning_enabled: bool = True,
    ):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((config.state_count, config.action_count), dtype=np.float64)
        self.state = RladState(replicas=1, util_bucket=0, cpu=config.initial_cpu)
        self.previous_action = DEFAULT_ACTIONS[1]
        self.t = 1
        self.model: list[DynaQElement] = []
        self.learning_enabled = learning_enabled
        self.preserve_java_quirk = (
            config.preserve_dynaq2_quirk if preserve_java_quirk is None else preserve_java_quirk
        )

    def update(self, service: ServiceState, cost: float, input_rate: float) -> None:
        if not self.learning_enabled:
            self.state = RladState(
                replicas=service.replicas,
                util_bucket=discretize_util(service.utilization, self.config),
                cpu=service.cpu,
            )
            return
        self.t += 1
        ns = self._next_state(service, self.previous_action)
        next_action = self.greedy_action(ns)
        s_idx = self.state.hash(self.config)
        ns_idx = ns.hash(self.config)
        old_value = self.q[s_idx, self.previous_action.index]
        estimate = cost + self.config.gamma * self.q[ns_idx, next_action.index]
        self.q[s_idx, self.previous_action.index] = (
            (1.0 - self.config.alpha) * old_value + self.config.alpha * estimate
        )
        old_state = self.state
        self.state = ns
        model_state = ns if self.preserve_java_quirk else old_state
        elem = DynaQElement(model_state, self.previous_action, ns, float(cost))
        self._upsert_model(elem)
        self._planning_updates()

    def pick_action(self, service: ServiceState, observed_utilization: float | None = None) -> Action:
        if observed_utilization is not None:
            self.state = RladState(
                replicas=service.replicas,
                util_bucket=discretize_util(observed_utilization, self.config),
                cpu=service.cpu,
            )
        epsilon = 1.0 / float(self.t)
        if self.rng.random() > epsilon:
            action = self.greedy_action(self.state)
        else:
            choices = valid_actions(ServiceState(self.state.replicas, self.state.cpu), self.config)
            action = choices[int(self.rng.integers(0, len(choices)))]
        self.previous_action = action
        return action

    def greedy_action(self, state: RladState) -> Action:
        service = ServiceState(replicas=state.replicas, cpu=state.cpu)
        candidates = valid_actions(service, self.config)
        q_row = self.q[state.hash(self.config)]
        best = candidates[0]
        best_cost = q_row[best.index]
        for action in candidates[1:]:
            value = q_row[action.index]
            if value < best_cost:
                best = action
                best_cost = value
        return best

    def _next_state(self, service: ServiceState, action: Action) -> RladState:
        return RladState(
            replicas=self.state.replicas + action.replica_delta,
            util_bucket=discretize_util(service.utilization, self.config),
            cpu=self.state.cpu + action.cpu_delta,
        )

    def _upsert_model(self, elem: DynaQElement) -> None:
        for idx, existing in enumerate(self.model):
            if existing.state == elem.state and existing.action == elem.action:
                self.model.pop(idx)
                break
        self.model.append(elem)
        if len(self.model) > self.config.dynaq2_model_max:
            self.model.pop(0)

    def _planning_updates(self) -> None:
        end = int(np.ceil((self.t * self.config.dynaq2_planning_percent) / 100.0))
        end = min(end, len(self.model))
        if end == 0:
            return
        for _ in range(end):
            elem = self.model[int(self.rng.integers(0, len(self.model)))]
            next_action = self.greedy_action(elem.next_state)
            s_idx = elem.state.hash(self.config)
            ns_idx = elem.next_state.hash(self.config)
            old_value = self.q[s_idx, elem.action.index]
            estimate = elem.cost + self.config.gamma * self.q[ns_idx, next_action.index]
            self.q[s_idx, elem.action.index] = (
                (1.0 - self.config.alpha) * old_value + self.config.alpha * estimate
            )

    def freeze(self) -> None:
        self.learning_enabled = False


class QLearningController(DynaQ2Controller):
    """Non-default plain Q-learning variant from the Java source."""

    def update(self, service: ServiceState, cost: float, input_rate: float) -> None:
        if not self.learning_enabled:
            self.state = RladState(
                replicas=service.replicas,
                util_bucket=discretize_util(service.utilization, self.config),
                cpu=service.cpu,
            )
            return
        self.t += 1
        ns = self._next_state(service, self.previous_action)
        next_action = self.greedy_action(ns)
        s_idx = self.state.hash(self.config)
        ns_idx = ns.hash(self.config)
        old_value = self.q[s_idx, self.previous_action.index]
        estimate = cost + self.config.gamma * self.q[ns_idx, next_action.index]
        self.q[s_idx, self.previous_action.index] = (
            (1.0 - self.config.alpha) * old_value + self.config.alpha * estimate
        )
        self.state = ns

    def pick_action(self, service: ServiceState, observed_utilization: float | None = None) -> Action:
        if observed_utilization is not None:
            self.state = RladState(
                replicas=service.replicas,
                util_bucket=discretize_util(observed_utilization, self.config),
                cpu=service.cpu,
            )
        epsilon = 0.1
        if self.rng.random() > epsilon:
            action = self.greedy_action(self.state)
        else:
            choices = valid_actions(ServiceState(self.state.replicas, self.state.cpu), self.config)
            action = choices[int(self.rng.integers(0, len(choices)))]
        self.previous_action = action
        return action


class ModelBasedController:
    """Paper-primary Rossi model-based tabular RL controller.

    This follows `ModelBasedAgent.java`: identity-initialized utilization
    transition estimates, online unknown-SLA-cost estimates, and in-place
    Bellman updates over the finite tabular state/action space.
    """

    def __init__(
        self,
        config: RossiConfig = DEFAULT_CONFIG,
        *,
        learning_enabled: bool = True,
    ):
        self.config = config
        self.q = np.zeros((config.state_count, config.action_count), dtype=np.float64)
        self.state = RladState(replicas=1, util_bucket=0, cpu=config.initial_cpu)
        self.previous_action = DEFAULT_ACTIONS[1]
        self.learning_enabled = learning_enabled
        self.lambda_tps = 0.0
        self.transition_counts = np.eye(config.util_states, dtype=np.float64)
        self.transition_prob = np.eye(config.util_states, dtype=np.float64)
        self.unknown_cost = np.zeros(config.state_count, dtype=np.float64)
        self._value_iteration(max_iter=1)

    def update(self, service: ServiceState, cost: float, input_rate: float) -> None:
        self.lambda_tps = float(input_rate)
        ns = RladState(
            replicas=service.replicas,
            util_bucket=discretize_util(service.utilization, self.config),
            cpu=service.cpu,
        )
        if self.learning_enabled:
            self._update_probability_estimate(self.state, ns)
            self._update_cost_estimate(self.state, ns, float(cost))
            self._value_iteration(max_iter=1)
        self.state = ns

    def pick_action(self, service: ServiceState, observed_utilization: float | None = None) -> Action:
        if observed_utilization is not None:
            self.state = RladState(
                replicas=service.replicas,
                util_bucket=discretize_util(observed_utilization, self.config),
                cpu=service.cpu,
            )
        action = self.greedy_action(self.state)
        self.previous_action = action
        return action

    def greedy_action(self, state: RladState) -> Action:
        service = ServiceState(replicas=state.replicas, cpu=state.cpu)
        candidates = valid_actions(service, self.config)
        q_row = self.q[state.hash(self.config)]
        best = candidates[0]
        best_cost = q_row[best.index]
        for action in candidates[1:]:
            value = q_row[action.index]
            if value < best_cost:
                best = action
                best_cost = value
        return best

    def freeze(self) -> None:
        self.learning_enabled = False

    def known_cost(self, state: RladState, action: Action) -> float:
        replicas = state.replicas + action.replica_delta
        cpu = state.cpu + action.cpu_delta
        resource = replicas * (cpu / 100.0) / self.config.max_replication
        reconfiguration = 1.0 if action.is_vertical_reconfiguration else 0.0
        return self.config.w_resources * resource + self.config.w_reconfiguration * reconfiguration

    def _update_probability_estimate(self, state: RladState, next_state: RladState) -> None:
        self.transition_counts[state.util_bucket, next_state.util_bucket] += 1.0
        row = self.transition_counts[state.util_bucket]
        self.transition_prob[state.util_bucket] = row / row.sum()

    def _update_cost_estimate(self, state: RladState, next_state: RladState, cost: float) -> None:
        known = self.known_cost(state, self.previous_action)
        unknown = cost - known
        idx = next_state.hash(self.config)
        old = self.unknown_cost[idx]
        new = (1.0 - self.config.alpha) * old + self.config.alpha * unknown
        self.unknown_cost[idx] = new
        if new > old:
            self._raise_monotone_dominated(next_state, new)
        elif new < old:
            self._lower_monotone_dominating(next_state, new)

    def _raise_monotone_dominated(self, state: RladState, value: float) -> None:
        for cpu in range(self.config.cpu_quantum, state.cpu + 1, self.config.cpu_quantum):
            for util in range(state.util_bucket, self.config.util_states):
                for replicas in range(1, state.replicas + 1):
                    idx = RladState(replicas, util, cpu).hash(self.config)
                    if self.unknown_cost[idx] < value:
                        self.unknown_cost[idx] = value

    def _lower_monotone_dominating(self, state: RladState, value: float) -> None:
        for cpu in range(state.cpu, self.config.cpu_max + 1, self.config.cpu_quantum):
            for util in range(0, state.util_bucket + 1):
                for replicas in range(state.replicas, self.config.max_replication + 1):
                    idx = RladState(replicas, util, cpu).hash(self.config)
                    if self.unknown_cost[idx] > value:
                        self.unknown_cost[idx] = value

    def _value_iteration(self, *, max_iter: int) -> None:
        # Java's do/while executes once more than the `max_iter` argument.
        for _ in range(max_iter + 1):
            self._value_iteration_pass()

    def _value_iteration_pass(self) -> None:
        for cpu in range(self.config.cpu_quantum, self.config.cpu_max + 1, self.config.cpu_quantum):
            for replicas in range(1, self.config.max_replication + 1):
                service = ServiceState(replicas=replicas, cpu=cpu)
                actions = valid_actions(service, self.config)
                for util in range(self.config.util_states):
                    state = RladState(replicas, util, cpu)
                    row = state.hash(self.config)
                    for action in actions:
                        self.q[row, action.index] = self._evaluate_q(state, action)

    def _evaluate_q(self, state: RladState, action: Action) -> float:
        qvalue = self.known_cost(state, action)
        next_replicas = state.replicas + action.replica_delta
        next_cpu = state.cpu + action.cpu_delta
        for next_util in range(self.config.util_states):
            probability = self.transition_prob[state.util_bucket, next_util]
            if probability <= 0.0:
                continue
            next_state = RladState(next_replicas, next_util, next_cpu)
            next_idx = next_state.hash(self.config)
            qvalue += probability * (
                self.unknown_cost[next_idx] + self.config.gamma * self._value(next_state)
            )
        return float(qvalue)

    def _value(self, state: RladState) -> float:
        candidates = valid_actions(ServiceState(state.replicas, state.cpu), self.config)
        q_row = self.q[state.hash(self.config)]
        return float(min(q_row[action.index] for action in candidates))
