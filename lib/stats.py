"""Statistics helpers shared across benchmarks."""

from __future__ import annotations

import statistics

import numpy as np


_EMPTY = {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0,
          "min": 0.0, "max": 0.0}


def compute_stats(values: list[float]) -> dict:
    """Return mean/std/p50/p90/p95/min/max for a list of floats.

    Empty input returns zeros — callers can detect via len(values).
    Standard deviation uses population variance (pstdev) for stability with
    the small N typical in a benchmark.
    """
    if not values:
        return dict(_EMPTY)
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
