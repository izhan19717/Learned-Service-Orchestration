"""State bucketing for Rossi/RLAD."""

from __future__ import annotations

from dataclasses import dataclass

from cisose_rossi.config import DEFAULT_CONFIG, RossiConfig


@dataclass(frozen=True)
class RladState:
    replicas: int
    util_bucket: int
    cpu: int

    def hash(self, config: RossiConfig = DEFAULT_CONFIG) -> int:
        cpu_bucket = int(self.cpu / config.cpu_quantum) - 1
        return (
            cpu_bucket
            + self.util_bucket * config.cpu_states
            + config.util_states * (self.replicas - 1) * config.cpu_states
        )


def discretize_util(utilization: float, config: RossiConfig = DEFAULT_CONFIG) -> int:
    clipped = min(float(utilization), config.util_max)
    return int(clipped / config.util_quantum)


def real_util(bucket: int, config: RossiConfig = DEFAULT_CONFIG) -> float:
    return bucket * config.util_quantum
