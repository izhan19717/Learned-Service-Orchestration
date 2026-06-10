"""Shared statistical routines for paired experiment cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    seed: int,
    resamples: int = 5000,
) -> tuple[float, float]:
    """Percentile bootstrap CI over paired per-seed differences."""

    diffs = np.asarray(differences, dtype=np.float64)
    if diffs.ndim != 1 or len(diffs) == 0:
        raise ValueError("differences must be a non-empty 1D sequence")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_pvalues(
    differences: Sequence[float],
    *,
    seed: int,
    resamples: int = 100_000,
) -> tuple[float, float]:
    """Paired sign-flip p-values for mean difference <0 and >0."""

    diffs = np.asarray(differences, dtype=np.float64)
    if diffs.ndim != 1 or len(diffs) == 0:
        raise ValueError("differences must be a non-empty 1D sequence")
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, len(diffs)))
    null_means = (signs * diffs).mean(axis=1)
    p_less = (np.count_nonzero(null_means <= observed) + 1.0) / (resamples + 1.0)
    p_greater = (np.count_nonzero(null_means >= observed) + 1.0) / (resamples + 1.0)
    return float(p_less), float(p_greater)


def holm_bonferroni(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values."""

    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, pvalue) in enumerate(ordered, start=1):
        adj = min(1.0, (m - rank + 1) * float(pvalue))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted
