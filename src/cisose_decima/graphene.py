"""Graphene comparator scaffold for Decima.

Graphene's paper describes an offline preferred order that focuses first on
long-running, hard-to-pack tasks while respecting DAG dependencies, then lets an
online component enforce that order. The official Decima simulator does not
ship Graphene code, so this module is deliberately marked as a validation-gated
implementation scaffold rather than accepted paper evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cisose_decima.tpch import TpchDagTemplate


@dataclass(frozen=True)
class GrapheneNodeScore:
    node_index: int
    descendant_work: float
    critical_path_work: float
    out_degree: int

    @property
    def priority(self) -> tuple[float, float, int, int]:
        return (self.descendant_work, self.critical_path_work, self.out_degree, -self.node_index)


@dataclass(frozen=True)
class GrapheneSchedule:
    template: TpchDagTemplate
    node_order: tuple[int, ...]
    scores: tuple[GrapheneNodeScore, ...]


class GrapheneStyleComparator:
    """Dependency-aware static node-priority comparator.

    The comparator uses Graphene's documented direction, not an undocumented
    claim of exact equivalence: long/downstream-heavy nodes are prioritized
    subject to DAG-ready constraints. It remains behind the Graphene validation
    gate before any Decima paper result may use it.
    """

    def score_nodes(self, template: TpchDagTemplate) -> tuple[GrapheneNodeScore, ...]:
        work = _node_work(template)
        descendant = _descendant_work(template.adjacency, work)
        critical_path = _critical_path_work(template.adjacency, work)
        out_degrees = np.count_nonzero(template.adjacency, axis=1)
        return tuple(
            GrapheneNodeScore(
                node_index=idx,
                descendant_work=float(descendant[idx]),
                critical_path_work=float(critical_path[idx]),
                out_degree=int(out_degrees[idx]),
            )
            for idx in range(template.num_nodes)
        )

    def preferred_schedule(self, template: TpchDagTemplate) -> GrapheneSchedule:
        scores = self.score_nodes(template)
        by_node = {score.node_index: score for score in scores}
        remaining = set(range(template.num_nodes))
        completed: set[int] = set()
        order: list[int] = []
        while remaining:
            ready = [node for node in remaining if _parents(template.adjacency, node).issubset(completed)]
            if not ready:
                raise ValueError("template adjacency is cyclic or malformed")
            chosen = max(ready, key=lambda node: by_node[node].priority)
            order.append(chosen)
            remaining.remove(chosen)
            completed.add(chosen)
        return GrapheneSchedule(template=template, node_order=tuple(order), scores=scores)

    def choose_node(self, template: TpchDagTemplate, ready_nodes: tuple[int, ...]) -> int:
        if not ready_nodes:
            raise ValueError("ready_nodes must be non-empty")
        by_node = {score.node_index: score for score in self.score_nodes(template)}
        return max(ready_nodes, key=lambda node: by_node[node].priority)


def _node_work(template: TpchDagTemplate) -> np.ndarray:
    if template.node_work is None:
        return np.ones(template.num_nodes, dtype=np.float64)
    return np.asarray(template.node_work, dtype=np.float64)


def _parents(adjacency: np.ndarray, node: int) -> set[int]:
    return {int(parent) for parent in np.flatnonzero(adjacency[:, node])}


def _children(adjacency: np.ndarray, node: int) -> tuple[int, ...]:
    return tuple(int(child) for child in np.flatnonzero(adjacency[node, :]))


def _descendant_work(adjacency: np.ndarray, work: np.ndarray) -> np.ndarray:
    memo: dict[int, set[int]] = {}

    def visit(node: int) -> set[int]:
        if node in memo:
            return memo[node]
        descendants = {node}
        for child in _children(adjacency, node):
            descendants.update(visit(child))
        memo[node] = descendants
        return descendants

    return np.asarray(
        [float(np.sum(work[list(visit(node))])) for node in range(adjacency.shape[0])],
        dtype=np.float64,
    )


def _critical_path_work(adjacency: np.ndarray, work: np.ndarray) -> np.ndarray:
    memo: dict[int, float] = {}

    def visit(node: int) -> float:
        if node in memo:
            return memo[node]
        children = _children(adjacency, node)
        if not children:
            value = float(work[node])
        else:
            value = float(work[node]) + max(visit(child) for child in children)
        memo[node] = value
        return value

    return np.asarray([visit(node) for node in range(adjacency.shape[0])], dtype=np.float64)


GrapheneComparator = GrapheneStyleComparator
