"""Action definitions matching the official RLAD simulator defaults."""

from __future__ import annotations

from dataclasses import dataclass

from cisose_rossi.config import DEFAULT_CONFIG, RossiConfig


@dataclass(frozen=True)
class Action:
    index: int
    label: str
    replica_delta: int
    cpu_delta: int

    @property
    def is_vertical_reconfiguration(self) -> bool:
        return self.cpu_delta != 0


def horizontal_or_vertical_actions(config: RossiConfig = DEFAULT_CONFIG) -> tuple[Action, ...]:
    q = config.cpu_quantum
    return (
        Action(0, "scale_in", -1, 0),
        Action(1, "no_op", 0, 0),
        Action(2, "scale_out", 1, 0),
        Action(3, "vertical_down", 0, -q),
        Action(4, "vertical_up", 0, q),
    )


DEFAULT_ACTIONS = horizontal_or_vertical_actions()
