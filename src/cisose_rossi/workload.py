"""Official RLAD workload-profile handling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cisose_rossi.config import DEFAULT_CONFIG


def load_profile(path: Path) -> tuple[float, ...]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                values.append(float(text))
    return tuple(values)


def java_slow_profile_sequence(
    values: Sequence[float],
    *,
    steps: int = DEFAULT_CONFIG.time_limit + 1,
    fallback: float = 200.0,
) -> tuple[float, ...]:
    """Reproduce `Simulation.readNextLambda()` slow-profile semantics.

    The Java code skips the first file line, uses the second, repeats it for two
    decision ticks, then uses the fourth line, and so on. When the profile ends,
    it falls back to 200.
    """

    out: list[float] = []
    file_index = 1
    while len(out) < steps:
        value = float(values[file_index]) if file_index < len(values) else fallback
        out.extend([value, value])
        file_index += 2
    return tuple(out[:steps])


def profile_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "mean": float(arr.mean()) if arr.size else float("nan"),
    }
