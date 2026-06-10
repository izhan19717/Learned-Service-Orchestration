"""Protocol constants locked before experiment execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PROTOCOL_VERSION = "v2.2"
MLFLOW_EXPERIMENT = "cisose_deeprm_v2_1"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

PRIMARY_LOAD = 0.7
LOAD_SWEEP = (0.1, 0.3, 0.5, 0.7, 0.9)
LAG_SWEEP = (0, 1, 2, 5, 10, 20)
TAIL_SWEEP = (float("inf"), 3.0, 2.0, 1.5, 1.2)
ADVERSARIAL_EPS_SWEEP = (0.0, 0.01, 0.02, 0.05, 0.10)

ANCHOR_LAG = 10
ANCHOR_TAIL_ALPHA = 1.5
ANCHOR_EPSILON = 0.05

EVAL_NUM_SEEDS = 30
EVAL_TRACE_JOBS = 200
BOOTSTRAP_RESAMPLES = 5000
SIGN_FLIP_RESAMPLES = 100_000
TRAIN_JOBSETS = 100
TRAIN_ROLLOUTS_PER_JOBSET = 20
TRAIN_ITERATIONS = 1000
TRAIN_LEARNING_RATE = 0.001

DOC_FILES = (
    "docs/protocols/calibration_v2_2.md",
    "docs/protocols/preregistration_v2_2.md",
    "docs/protocols/protocol_amendment_v2_2.md",
    "docs/protocols/protocol_amendment_decima_simulator_gate.md",
    "docs/implementation_notes.md",
    "docs/PROTOCOL_INDEX.md",
)


@dataclass(frozen=True)
class DeepRMConfig:
    num_resources: int = 2
    resource_capacity: float = 1.0
    resource_bins: int = 10
    time_horizon: int = 20
    visible_slots: int = 10
    backlog_capacity: int = 60
    episode_length: int = 50
    short_job_probability: float = 0.8
    long_job_probability: float = 0.2
    short_duration_min: int = 1
    short_duration_max: int = 3
    long_duration_min: int = 10
    long_duration_max: int = 15
    tail_x_min: int = 10
    tail_x_max: int = 100
    dominant_demand_min: float = 0.25
    dominant_demand_max: float = 0.5
    nondominant_demand_min: float = 0.05
    nondominant_demand_max: float = 0.1
    demand_mode: Literal["paper_continuous", "source_discrete"] = "paper_continuous"
    include_extra_info: bool = False
    max_track_since_new: int = 10
    reward_on_allocate: bool = True
    external_admission: bool = True
    max_start_inclusive: bool = True
    planning_horizon: int | None = None
    primary_load: float = PRIMARY_LOAD
    discount: float = 1.0

    @property
    def action_dim(self) -> int:
        return self.visible_slots + 1

    @property
    def state_shape(self) -> tuple[int, int, int]:
        # One image with resource occupancy, visible job canvases, and backlog strip.
        width = (
            (self.resource_bins + self.resource_bins * self.visible_slots)
            * self.num_resources
            + self.backlog_capacity // self.time_horizon
            + (1 if self.include_extra_info else 0)
        )
        return (self.time_horizon, width, 1)

    @property
    def state_dim(self) -> int:
        h, w, c = self.state_shape
        return h * w * c


def author_source_config(load: float = PRIMARY_LOAD) -> DeepRMConfig:
    """Configuration matching the public DeepRM source operationally.

    The DeepRM paper reports continuous demands and 89,451 parameters. The
    public source uses integer resource slots and an extra-info image column.
    This helper follows the source because it is the executable artifact.
    """

    return DeepRMConfig(
        resource_capacity=10.0,
        dominant_demand_min=5.0,
        dominant_demand_max=10.0,
        nondominant_demand_min=1.0,
        nondominant_demand_max=2.0,
        demand_mode="source_discrete",
        include_extra_info=True,
        reward_on_allocate=False,
        external_admission=False,
        max_start_inclusive=False,
        primary_load=load,
    )


def repo_path(*parts: str) -> Path:
    return Path.cwd().joinpath(*parts)
