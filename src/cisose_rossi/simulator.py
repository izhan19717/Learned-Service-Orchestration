"""Source-faithful Rossi/RLAD simulator core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from cisose_rossi.actions import Action, DEFAULT_ACTIONS
from cisose_rossi.config import DEFAULT_CONFIG, RossiConfig


@dataclass
class ServiceState:
    replicas: int = 1
    cpu: int = DEFAULT_CONFIG.initial_cpu
    utilization: float = 0.0


@dataclass(frozen=True)
class StepRecord:
    time: int
    input_rate: float
    replicas_before: int
    cpu_before: int
    utilization: float
    response_time: float
    cost: float
    sla_violated: bool
    action_index: int
    action_label: str
    replicas_after: int
    cpu_after: int
    observed_utilization: float | None = None
    observation_delta: float = 0.0


class RossiController(Protocol):
    previous_action: Action

    def update(self, service: ServiceState, cost: float, input_rate: float) -> None:
        ...

    def pick_action(self, service: ServiceState, observed_utilization: float | None = None) -> Action:
        ...


class RladSimulator:
    """Single-service simulator matching the Java operation order."""

    def __init__(
        self,
        config: RossiConfig = DEFAULT_CONFIG,
        *,
        service_time_cv2: float | None = None,
    ):
        self.config = config
        self.service_time_cv2 = service_time_cv2
        self.service = ServiceState()

    def reset(self) -> None:
        self.service = ServiceState(cpu=self.config.initial_cpu)

    def service_time_mean(self, cpu: int | None = None) -> float:
        cpu_value = self.service.cpu if cpu is None else cpu
        return 1.0 / (self.config.mu * (cpu_value / 100.0))

    def utilization(self, input_rate: float, replicas: int | None = None, cpu: int | None = None) -> float:
        rep = self.service.replicas if replicas is None else replicas
        return (float(input_rate) / float(rep)) * self.service_time_mean(cpu)

    def response_time(
        self,
        input_rate: float,
        utilization: float,
        replicas: int | None = None,
        cpu: int | None = None,
    ) -> float:
        if utilization >= 1.0:
            return 999999.0
        rep = self.service.replicas if replicas is None else replicas
        mean = self.service_time_mean(cpu)
        variance = (
            self.service_time_cv2 * mean * mean
            if self.service_time_cv2 is not None
            else self.config.service_time_variance
        )
        es2 = variance + mean * mean
        lambda_per_replica = float(input_rate) / float(rep)
        return mean + lambda_per_replica / 2.0 * es2 / (1.0 - utilization)

    def cost(self, action: Action, input_rate: float) -> tuple[float, bool, float]:
        util = self.service.utilization
        response = self.response_time(input_rate, util)
        resource_cost = (
            self.service.replicas * (self.service.cpu / 100.0) / self.config.max_replication
        )
        reconfiguration_cost = 1.0 if action.is_vertical_reconfiguration else 0.0
        sla_violated = response > self.config.sla_threshold
        sla_cost = 1.0 if sla_violated else 0.0
        total = (
            self.config.w_resources * resource_cost
            + self.config.w_reconfiguration * reconfiguration_cost
            + self.config.w_sla * sla_cost
        )
        return total, sla_violated, response

    def apply(self, action: Action) -> None:
        new_replicas = self.service.replicas + action.replica_delta
        new_cpu = self.service.cpu + action.cpu_delta
        if not (1 <= new_replicas <= self.config.max_replication):
            raise ValueError(f"invalid replica count after action: {new_replicas}")
        if not (0 < new_cpu <= self.config.cpu_max):
            raise ValueError(f"invalid cpu after action: {new_cpu}")
        self.service.replicas = new_replicas
        self.service.cpu = new_cpu

    def run(
        self,
        controller: RossiController,
        input_rates: tuple[float, ...],
        *,
        horizon: int | None = None,
        observed_utilizations: tuple[float, ...] | None = None,
        observation_lag_steps: int = 0,
        observation_transform: Callable[[RossiController, ServiceState, float], float] | None = None,
        observation_applies_to_update: bool = False,
    ) -> tuple[StepRecord, ...]:
        observation_sources = sum(
            source is not None and source != 0
            for source in (observed_utilizations, observation_lag_steps, observation_transform)
        )
        if observation_sources > 1:
            raise ValueError("observation sources are mutually exclusive")
        if observation_transform is not None and observation_applies_to_update:
            raise ValueError("observation_transform cannot be applied to controller updates")
        self.reset()
        limit = self.config.time_limit + 1 if horizon is None else horizon
        records: list[StepRecord] = []
        utilization_history: list[float] = []
        for t in range(limit):
            input_rate = input_rates[t] if t < len(input_rates) else 200.0
            self.service.utilization = self.utilization(input_rate)
            utilization_history.append(float(self.service.utilization))
            previous = controller.previous_action
            cost, violated, response = self.cost(previous, input_rate)
            if observed_utilizations is not None and t < len(observed_utilizations):
                observed_util = observed_utilizations[t]
            elif observation_lag_steps > 0:
                observed_util = utilization_history[max(0, t - observation_lag_steps)]
            elif observation_transform is None:
                observed_util = self.service.utilization
            else:
                observed_util = None
            update_service = self.service
            if observation_applies_to_update and observed_util is not None:
                update_service = ServiceState(
                    replicas=self.service.replicas,
                    cpu=self.service.cpu,
                    utilization=float(observed_util),
                )
            controller.update(update_service, cost, input_rate)
            if observation_transform is not None:
                observed_util = observation_transform(
                    controller,
                    self.service,
                    float(self.service.utilization),
                )
            observation_delta = float(observed_util) - float(self.service.utilization)
            action = controller.pick_action(self.service, observed_util)
            before_replicas = self.service.replicas
            before_cpu = self.service.cpu
            self.apply(action)
            records.append(
                StepRecord(
                    time=t,
                    input_rate=float(input_rate),
                    replicas_before=before_replicas,
                    cpu_before=before_cpu,
                    utilization=float(self.service.utilization),
                    response_time=float(response),
                    cost=float(cost),
                    sla_violated=violated,
                    action_index=action.index,
                    action_label=action.label,
                    replicas_after=self.service.replicas,
                    cpu_after=self.service.cpu,
                    observed_utilization=float(observed_util),
                    observation_delta=observation_delta,
                )
            )
        return tuple(records)


def total_cost(records: tuple[StepRecord, ...]) -> float:
    return float(np.sum([record.cost for record in records], dtype=np.float64))


def valid_actions(service: ServiceState, config: RossiConfig = DEFAULT_CONFIG) -> tuple[Action, ...]:
    actions = []
    for action in DEFAULT_ACTIONS:
        replicas = service.replicas + action.replica_delta
        cpu = service.cpu + action.cpu_delta
        if 1 <= replicas <= config.max_replication and 0 < cpu <= config.cpu_max:
            actions.append(action)
    return tuple(actions)
