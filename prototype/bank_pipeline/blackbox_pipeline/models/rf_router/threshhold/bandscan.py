"""
Band enumeration: which bands a score can actually produce, and the threshold
that produces each one.

Named ``bandscan`` rather than ``bands`` to stay clear of ``rf_router.bands``,
which measures band confidence. Nothing here touches constraints or labels
beyond validation — it is pure geometry over a sorted score vector.

Only boundaries between DISTINCT probability values are realisable. If
``p_sorted[k-1] == p_sorted[k]`` no threshold can split that tie, so a solver
that enumerated every ``k`` would report bands the model cannot produce.

Both solvers scan every realisable band rather than assuming monotonicity.
Empirical precision is not monotone in the threshold at finite sample size, so a
bisection would silently stop at the first local violation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lower_band_sizes",
    "upper_band_sizes",
    "upper_edge",
    "lower_edge",
    "prepare_scores",
]


def lower_band_sizes(p_sorted: np.ndarray) -> np.ndarray:
    """Sizes ``k`` for which ``{p < t}`` is exactly the k lowest-scoring rows."""
    n = len(p_sorted)
    if n == 0:
        return np.empty(0, dtype=int)
    distinct = np.flatnonzero(p_sorted[:-1] < p_sorted[1:]) + 1
    return np.concatenate([distinct, [n]]).astype(int)


def upper_band_sizes(p_sorted: np.ndarray) -> np.ndarray:
    """Sizes ``m`` for which ``{p > t}`` is exactly the m highest-scoring rows."""
    n = len(p_sorted)
    if n == 0:
        return np.empty(0, dtype=int)
    distinct = n - (np.flatnonzero(p_sorted[:-1] < p_sorted[1:]) + 1)
    return np.sort(np.concatenate([distinct, [n]])).astype(int)


def upper_edge(p_sorted: np.ndarray, k: int) -> float:
    """Threshold ``t`` with ``{p < t}`` == the k lowest rows."""
    n = len(p_sorted)
    if k >= n:
        return float(np.nextafter(p_sorted[-1], np.inf))
    return float(p_sorted[k])


def lower_edge(p_sorted: np.ndarray, m: int) -> float:
    """Threshold ``t`` with ``{p > t}`` == the m highest rows."""
    n = len(p_sorted)
    if m >= n:
        return float(np.nextafter(p_sorted[0], -np.inf))
    return float(p_sorted[n - m - 1])


def prepare_scores(proba, y, label: str):
    """Validate and coerce one population's scores and labels."""
    p = np.asarray(proba, dtype=float)
    y_arr = np.asarray(y)

    if len(p) != len(y_arr):
        raise ValueError(
            f"{label}: proba/y length mismatch {len(p)} vs {len(y_arr)}"
        )
    if len(p) == 0:
        raise ValueError(f"{label}: empty population")
    if np.isnan(p).any():
        raise ValueError(
            f"{label}: proba contains NaN. Pass only the SCORED rows "
            "(OOFResult.scored_proba()), not the full padded array."
        )
    if not np.isin(y_arr, (0, 1)).all():
        raise ValueError(f"{label}: y must be binary 0/1")

    return p, y_arr.astype(int)