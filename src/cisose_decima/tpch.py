"""TPC-H DAG template loading for Decima."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TpchDagTemplate:
    size: str
    query_id: int
    adjacency: np.ndarray
    task_counts: tuple[int, ...] | None = None
    node_work: tuple[float, ...] | None = None
    node_duration: tuple[float, ...] | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def num_edges(self) -> int:
        return int(np.count_nonzero(self.adjacency))

    @property
    def depth(self) -> int:
        if self.num_nodes == 0:
            return 0
        parents = {idx: set(np.flatnonzero(self.adjacency[:, idx])) for idx in range(self.num_nodes)}
        visiting: set[int] = set()
        memo: dict[int, int] = {}

        def visit(node: int) -> int:
            if node in memo:
                return memo[node]
            if node in visiting:
                raise ValueError("TPC-H template adjacency is cyclic")
            visiting.add(node)
            if parents[node]:
                depth = 1 + max(visit(parent) for parent in parents[node])
            else:
                depth = 0
            visiting.remove(node)
            memo[node] = depth
            return depth

        return int(max(visit(node) for node in range(self.num_nodes)) + 1)

    @property
    def branching_factor(self) -> float:
        if self.num_nodes == 0:
            return 0.0
        return float(self.num_edges / self.num_nodes)

    @property
    def size_score(self) -> float:
        return float(self.num_nodes + self.num_edges + self.depth)

    @property
    def total_work(self) -> float:
        if self.node_work is None:
            return float(self.num_nodes)
        return float(sum(self.node_work))


def load_tpch_templates(root: Path) -> tuple[TpchDagTemplate, ...]:
    templates: list[TpchDagTemplate] = []
    for size_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for adj_path in sorted(size_dir.glob("adj_mat_*.npy"), key=_query_sort_key):
            query_id = int(adj_path.stem.split("_")[-1])
            adjacency = np.load(adj_path)
            task_counts, node_work, node_duration = _load_task_duration_summary(
                size_dir / f"task_duration_{query_id}.npy",
                int(adjacency.shape[0]),
            )
            templates.append(
                TpchDagTemplate(
                    size_dir.name,
                    query_id,
                    adjacency,
                    task_counts=task_counts,
                    node_work=node_work,
                    node_duration=node_duration,
                )
            )
    return tuple(templates)


def template_summary(templates: tuple[TpchDagTemplate, ...]) -> dict[str, float]:
    nodes = np.asarray([t.num_nodes for t in templates], dtype=np.float64)
    edges = np.asarray([t.num_edges for t in templates], dtype=np.float64)
    depth = np.asarray([t.depth for t in templates], dtype=np.float64)
    return {
        "count": float(len(templates)),
        "mean_nodes": float(nodes.mean()) if len(nodes) else 0.0,
        "max_nodes": float(nodes.max()) if len(nodes) else 0.0,
        "mean_edges": float(edges.mean()) if len(edges) else 0.0,
        "max_edges": float(edges.max()) if len(edges) else 0.0,
        "mean_depth": float(depth.mean()) if len(depth) else 0.0,
        "max_depth": float(depth.max()) if len(depth) else 0.0,
        "mean_work": float(np.asarray([t.total_work for t in templates], dtype=np.float64).mean()) if templates else 0.0,
        "max_work": float(np.asarray([t.total_work for t in templates], dtype=np.float64).max()) if templates else 0.0,
    }


def _query_sort_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _load_task_duration_summary(
    path: Path,
    num_nodes: int,
) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None, tuple[float, ...] | None]:
    if not path.exists():
        return None, None, None
    raw = np.load(path, allow_pickle=True)
    task_durations = raw.item() if raw.shape == () else raw
    if len(task_durations) != num_nodes:
        return None, None, None
    counts: list[int] = []
    works: list[float] = []
    durations: list[float] = []
    for node_idx in range(num_nodes):
        node_profile = task_durations[node_idx]
        if not isinstance(node_profile, dict) or not node_profile:
            counts.append(1)
            works.append(1.0)
            durations.append(1.0)
            continue
        first_by_executor = dict(node_profile.get("first_wave", {}))
        rest_by_executor = dict(node_profile.get("rest_wave", {}))
        fresh_by_executor = dict(node_profile.get("fresh_durations", {}))
        if not first_by_executor:
            counts.append(1)
            works.append(1.0)
            durations.append(1.0)
            continue
        executor_key = next(iter(first_by_executor))
        first_wave = _as_float_list(first_by_executor.get(executor_key, []))
        rest_wave = _as_float_list(rest_by_executor.get(executor_key, []))
        processed_first = _pre_process_first_wave(first_by_executor, fresh_by_executor)
        task_count = max(len(first_wave) + len(rest_wave), 1)
        samples = _flatten_duration_dict(processed_first)
        samples.extend(_flatten_duration_dict(rest_by_executor))
        samples.extend(_flatten_duration_dict(fresh_by_executor))
        duration = float(np.mean(samples)) if samples else 1.0
        counts.append(task_count)
        durations.append(duration)
        works.append(float(task_count * duration))
    return tuple(counts), tuple(works), tuple(durations)


def _as_float_list(value: object) -> list[float]:
    if value is None:
        return []
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(item) for item in array if np.isfinite(item)]


def _flatten_duration_dict(value: dict[object, object]) -> list[float]:
    items: list[float] = []
    for durations in value.values():
        items.extend(_as_float_list(durations))
    return items


def _pre_process_first_wave(
    first_by_executor: dict[object, object],
    fresh_by_executor: dict[object, object],
) -> dict[object, list[float]]:
    clean: dict[object, list[float]] = {}
    for executor in first_by_executor:
        fresh_counts: dict[float, int] = {}
        for duration in _as_float_list(fresh_by_executor.get(executor, [])):
            fresh_counts[duration] = fresh_counts.get(duration, 0) + 1
        clean[executor] = []
        for duration in _as_float_list(first_by_executor[executor]):
            if fresh_counts.get(duration, 0) > 0:
                fresh_counts[duration] -= 1
            else:
                clean[executor].append(duration)
    last_first_wave: list[float] = []
    for executor in sorted(clean.keys()):
        if not clean[executor]:
            clean[executor] = list(last_first_wave)
        last_first_wave = clean[executor]
    return clean
