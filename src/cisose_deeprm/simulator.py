"""DeepRM-style scheduling simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cisose_deeprm.protocol import DeepRMConfig
from cisose_deeprm.workload import Job, WorkloadTrace


VOID_ACTION = -1


@dataclass
class RunningJob:
    job: Job
    start_time: int
    finish_time: int


@dataclass
class StepInfo:
    status: str
    time: int
    completed_jobs: tuple[Job, ...] = ()


@dataclass
class Machine:
    config: DeepRMConfig
    availability: np.ndarray = field(init=False)
    canvas: np.ndarray = field(init=False)
    running: list[RunningJob] = field(default_factory=list)

    def __post_init__(self) -> None:
        planning_horizon = self.config.planning_horizon or self.config.time_horizon
        self.availability = (
            np.ones((planning_horizon, self.config.num_resources), dtype=np.float64)
            * self.config.resource_capacity
        )
        self.canvas = np.zeros(
            (self.config.num_resources, self.config.time_horizon, self.config.resource_bins),
            dtype=np.float32,
        )

    def can_allocate_now(self, job: Job) -> bool:
        if job.duration > self.availability.shape[0]:
            return False
        demand = job.demand_array()
        window = self.availability[: job.duration, :]
        return bool(np.all(window - demand >= -1e-12))

    def earliest_offset(self, job: Job) -> int | None:
        planning_horizon = self.availability.shape[0]
        if job.duration > planning_horizon:
            return None
        demand = job.demand_array()
        max_start = planning_horizon - job.duration
        stop = max_start + 1 if self.config.max_start_inclusive else max_start
        for offset in range(max(0, stop)):
            window = self.availability[offset : offset + job.duration, :]
            if np.all(window - demand >= -1e-12):
                return offset
        return None

    def allocate(self, job: Job, curr_time: int) -> RunningJob | None:
        offset = self.earliest_offset(job)
        if offset is None:
            return None
        demand = job.demand_array()
        self.availability[offset : offset + job.duration, :] -= demand
        running = RunningJob(job=job, start_time=curr_time + offset, finish_time=curr_time + offset + job.duration)
        self.running.append(running)
        self._paint_job(job, offset)
        return running

    def advance_time(self, curr_time: int) -> list[Job]:
        self.availability[:-1, :] = self.availability[1:, :]
        self.availability[-1, :] = self.config.resource_capacity
        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0.0
        completed: list[Job] = []
        still_running: list[RunningJob] = []
        for running in self.running:
            if running.finish_time <= curr_time:
                completed.append(
                    running.job.with_times(
                        start_time=running.start_time,
                        finish_time=running.finish_time,
                    )
                )
            else:
                still_running.append(running)
        self.running = still_running
        return completed

    def _paint_job(self, job: Job, offset: int) -> None:
        color = ((job.id % 39) + 1) / 40.0
        bins = job.demand_bins(self.config)
        for res in range(self.config.num_resources):
            for row in range(offset, min(offset + job.duration, self.config.time_horizon)):
                free = np.where(self.canvas[res, row, :] == 0.0)[0]
                self.canvas[res, row, free[: bins[res]]] = color


class DeepRMEnv:
    """Event simulator with DeepRM's allocate-or-advance action semantics."""

    def __init__(self, trace: WorkloadTrace, config: DeepRMConfig | None = None, *, drain: bool = True):
        self.config = config or DeepRMConfig()
        self.trace = trace
        self.drain = drain
        self.arrivals = trace.arrivals_by_time()
        self.curr_time = 0
        self.machine = Machine(self.config)
        self.visible_slots: list[Job | None] = [None] * self.config.visible_slots
        self.backlog: deque[Job] = deque()
        self.external: deque[Job] = deque()
        self.completed: list[Job] = []
        self.dropped: list[Job] = []
        self.time_since_last_new_job = 0
        self._next_arrival_scan_time = 0
        self._admit_arrivals_through(0)
        self._fill_visible_from_waiting()

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def done(self) -> bool:
        if not self.drain and self.curr_time >= self.trace.horizon:
            return True
        if not self.config.external_admission:
            return (
                self._next_arrival_scan_time > self.trace.horizon - 1
                and not self.machine.running
                and all(job is None for job in self.visible_slots)
                and not self.backlog
            )
        return (
            len(self.completed) == len(self.trace.jobs)
            and not self.machine.running
            and all(job is None for job in self.visible_slots)
            and not self.backlog
            and not self.external
        )

    def slot_job_ids(self) -> tuple[int | None, ...]:
        return tuple(job.id if job is not None else None for job in self.visible_slots)

    def observe(self) -> np.ndarray:
        h, w, c = self.config.state_shape
        image = np.zeros((h, w), dtype=np.float32)
        ptr = 0
        for res in range(self.config.num_resources):
            image[:, ptr : ptr + self.config.resource_bins] = self.machine.canvas[res]
            ptr += self.config.resource_bins
            for job in self.visible_slots:
                if job is not None:
                    bins = job.demand_bins(self.config)
                    image[: job.duration, ptr : ptr + bins[res]] = 1.0
                ptr += self.config.resource_bins

        backlog_width = self.config.backlog_capacity // self.config.time_horizon
        observed_backlog = min(self.config.backlog_capacity, len(self.backlog) + len(self.external))
        full_rows, remainder = divmod(observed_backlog, backlog_width)
        if full_rows:
            image[:full_rows, ptr : ptr + backlog_width] = 1.0
        if full_rows < h and remainder:
            image[full_rows, ptr : ptr + remainder] = 1.0
        ptr += backlog_width
        if self.config.include_extra_info:
            image[:, ptr : ptr + 1] = self.time_since_last_new_job / float(self.config.max_track_since_new)
            ptr += 1
        assert ptr == w
        return image.reshape(h, w, c)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        status = "MoveOn"
        completed: list[Job] = []
        if 0 <= action < self.config.visible_slots:
            job = self.visible_slots[action]
            if job is not None:
                running = self.machine.allocate(job, self.curr_time)
                if running is not None:
                    self.visible_slots[action] = None
                    self._fill_slot(action)
                    status = "Allocate"

        if status == "MoveOn":
            self.curr_time += 1
            completed = self.machine.advance_time(self.curr_time)
            self.completed.extend(completed)
            self.time_since_last_new_job = min(
                self.config.max_track_since_new, self.time_since_last_new_job + 1
            )
            self._admit_arrivals_through(self.curr_time)
            self._fill_visible_from_waiting()

        reward = self.reward() if status == "MoveOn" or self.config.reward_on_allocate else 0.0
        info = StepInfo(status=status, time=self.curr_time, completed_jobs=tuple(completed))
        return self.observe(), reward, self.done, info

    def step_with_stale_identity(
        self, action: int, expected_slot_job_ids: tuple[int | None, ...]
    ) -> tuple[np.ndarray, float, bool, StepInfo]:
        if 0 <= action < self.config.visible_slots:
            current = self.visible_slots[action]
            expected = expected_slot_job_ids[action] if action < len(expected_slot_job_ids) else None
            if current is None or current.id != expected:
                return self.step(self.config.visible_slots)
        return self.step(action)

    def reward(self) -> float:
        total = 0.0
        for running in self.machine.running:
            total -= 1.0 / running.job.duration
        for job in self.visible_slots:
            if job is not None:
                total -= 1.0 / job.duration
        for job in self.backlog:
            total -= 1.0 / job.duration
        for job in self.external:
            total -= 1.0 / job.duration
        return total

    def mean_slowdown(self) -> float:
        if len(self.completed) != len(self.trace.jobs):
            raise ValueError("mean slowdown requested before all jobs completed")
        return float(np.mean([job.slowdown for job in self.completed]))

    def p95_completion_time(self) -> float:
        if len(self.completed) != len(self.trace.jobs):
            raise ValueError("p95 requested before all jobs completed")
        completion_times = [job.finish_time - job.arrival_time for job in self.completed]
        return float(np.percentile(completion_times, 95))

    def makespan(self) -> int:
        if not self.completed:
            return 0
        return max(job.finish_time or 0 for job in self.completed)

    def _admit_arrivals_through(self, time: int) -> None:
        while self._next_arrival_scan_time <= time:
            for job in self.arrivals.get(self._next_arrival_scan_time, []):
                self.external.append(job)
                self.time_since_last_new_job = 0
            self._next_arrival_scan_time += 1

    def _fill_visible_from_waiting(self) -> None:
        for idx, job in enumerate(self.visible_slots):
            if job is None:
                self._fill_slot(idx)

    def _fill_slot(self, idx: int) -> None:
        if self.backlog:
            self.visible_slots[idx] = self.backlog.popleft()
        elif self.external:
            self.visible_slots[idx] = self.external.popleft()
        else:
            self.visible_slots[idx] = None
        while self.external:
            if len(self.backlog) < self.config.backlog_capacity:
                self.backlog.append(self.external.popleft())
            elif self.config.external_admission:
                break
            else:
                self.dropped.append(self.external.popleft())
