"""Source-derived Rossi/RLAD configuration."""

from __future__ import annotations

from dataclasses import dataclass


RLAD_REPO_URL = "https://github.com/effereds/rlad-core-simulator"
RLAD_COMMIT = "d6a4ff136907eb1bd9e8b4151a9162231ce0ee6a"
PROFILE_SHA256 = "8bdead4788c9e275fd5282096ca8cc6fedcf9b42aca1afbf3c3b5b4ff21b4454"


@dataclass(frozen=True)
class RossiConfig:
    """Defaults extracted from `rl/simulator/Configuration.java`."""

    max_replication: int = 10
    initial_cpu: int = 100
    mu: float = 200.0
    service_time_variance: float = 0.0
    sla_threshold: float = 0.050
    time_limit: int = 4000
    util_max: float = 1.0
    util_quantum: float = 0.1
    cpu_max: int = 100
    cpu_quantum: int = 10
    w_resources: float = 0.09
    w_reconfiguration: float = 0.01
    w_sla: float = 0.90
    gamma: float = 0.99
    alpha: float = 0.1
    dynaq2_model_max: int = 100
    dynaq2_planning_percent: float = 10.0
    preserve_dynaq2_quirk: bool = True

    @property
    def util_states(self) -> int:
        return 1 + int(self.util_max / self.util_quantum)

    @property
    def cpu_states(self) -> int:
        return int(self.cpu_max / self.cpu_quantum)

    @property
    def state_count(self) -> int:
        return self.max_replication * self.util_states * self.cpu_states

    @property
    def action_count(self) -> int:
        return 5


DEFAULT_CONFIG = RossiConfig()
