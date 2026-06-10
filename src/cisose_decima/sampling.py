"""DAG sampling and Alibaba-style TPC-H reweighting."""

from __future__ import annotations

import numpy as np

from cisose_decima.tpch import TpchDagTemplate


def sampling_probabilities(
    templates: tuple[TpchDagTemplate, ...],
    *,
    w: float,
) -> np.ndarray:
    if not templates:
        raise ValueError("templates must be non-empty")
    scores = np.asarray([template.size_score for template in templates], dtype=np.float64)
    normalized = scores / max(float(scores.mean()), 1e-9)
    weights = np.power(normalized, float(w))
    weights = weights / weights.sum()
    return weights


def sample_templates(
    templates: tuple[TpchDagTemplate, ...],
    *,
    count: int,
    w: float,
    seed: int,
) -> tuple[TpchDagTemplate, ...]:
    rng = np.random.default_rng(seed)
    probabilities = sampling_probabilities(templates, w=w)
    indices = rng.choice(np.arange(len(templates)), size=count, replace=True, p=probabilities)
    return tuple(templates[int(idx)] for idx in indices)
