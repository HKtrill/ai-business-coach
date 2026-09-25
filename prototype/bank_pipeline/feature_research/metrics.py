"""
feature_research/metrics.py
============================
Statistical building blocks for feature-separation analysis.

Provides:
- :class:`SeparationMetrics`  — typed container for per-feature diagnostic scores
- :func:`cramers_v`           — categorical association (χ² derived)
- :func:`cohens_d`            — effect size for numeric features
- :func:`wilson_ci`           — binomial confidence interval (more accurate than
                                normal approximation for small / imbalanced samples)

All functions are stateless and side-effect-free; they accept plain numpy /
pandas inputs and return scalar Python floats or tuples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SeparationMetrics:
    """Typed container for all separation diagnostics computed for one feature.

    Numeric and categorical metrics are stored in separate optional fields so
    that downstream code can distinguish which branch was exercised without
    inspecting ``feature_type`` manually.

    Parameters
    ----------
    feature:
        Column name in the originating DataFrame.
    feature_type:
        ``'numeric'`` or ``'categorical'``.

    Notes
    -----
    ``composite_score`` is intentionally left to the caller to populate after
    all individual metrics are computed; this keeps the dataclass a pure
    container without embedded business logic.
    """

    feature: str
    feature_type: str  # 'numeric' | 'categorical'

    # -- Numeric-only ---------------------------------------------------------
    cohens_d: Optional[float] = field(default=None)
    ks_stat: Optional[float] = field(default=None)
    ks_pvalue: Optional[float] = field(default=None)
    point_biserial: Optional[float] = field(default=None)

    # -- Common ---------------------------------------------------------------
    mutual_info: Optional[float] = field(default=None)
    auc_probe: Optional[float] = field(default=None)

    # -- Categorical-only -----------------------------------------------------
    cramers_v: Optional[float] = field(default=None)
    target_rate_range: Optional[Tuple[float, float]] = field(default=None)
    n_categories: Optional[int] = field(default=None)

    # -- Derived --------------------------------------------------------------
    composite_score: Optional[float] = field(default=None)

    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Flatten to a plain dict for :class:`pandas.DataFrame` construction.

        Returns
        -------
        dict
            One key per field; preserves ``None`` for missing metrics so that
            DataFrame columns align across numeric and categorical rows.
        """
        return {
            "feature": self.feature,
            "type": self.feature_type,
            "cohens_d": self.cohens_d,
            "ks_stat": self.ks_stat,
            "ks_pvalue": self.ks_pvalue,
            "point_biserial": self.point_biserial,
            "mutual_info": self.mutual_info,
            "auc_probe": self.auc_probe,
            "cramers_v": self.cramers_v,
            "target_rate_range": self.target_rate_range,
            "n_categories": self.n_categories,
            "composite_score": self.composite_score,
        }


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V statistic for categorical–binary target association.

    .. math::

        V = \\sqrt{\\frac{\\chi^2}{n \\cdot \\min(k-1,\\, r-1)}}

    Parameters
    ----------
    x:
        Categorical feature (any hashable dtype).
    y:
        Binary target encoded as 0/1.

    Returns
    -------
    float
        Cramér's V in ``[0, 1]``.  Returns ``0.0`` when the contingency table
        is degenerate (single row or column).

    Examples
    --------
    >>> cramers_v(pd.Series(["a", "b", "a"]), pd.Series([0, 1, 0]))
    """
    contingency = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency)[0]
    n = len(x)
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)

    if min_dim == 0:
        return 0.0

    return float(np.sqrt(chi2 / (n * min_dim)))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two independent groups.

    Uses the pooled standard deviation (Hedges / Cohen convention):

    .. math::

        d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{\\text{pooled}}}

    Parameters
    ----------
    group1, group2:
        1-D arrays of observations for each class.

    Returns
    -------
    float
        Signed Cohen's d.  Returns ``0.0`` when pooled SD is zero (constant
        features).

    Examples
    --------
    >>> cohens_d(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    -3.0
    """
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0.0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def wilson_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation when sample sizes are small or
    proportions are near 0/1 — both common in imbalanced classification tasks.

    Parameters
    ----------
    successes:
        Number of positive outcomes.
    trials:
        Total number of trials.
    confidence:
        Desired confidence level (default ``0.95``).

    Returns
    -------
    Tuple[float, float]
        ``(lower_bound, upper_bound)`` clipped to ``[0, 1]``.
        Returns ``(0.0, 0.0)`` when ``trials == 0``.

    Examples
    --------
    >>> wilson_ci(50, 100)
    (0.404, 0.596)  # approximate
    """
    if trials == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / trials

    denominator = 1.0 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denominator
    spread = (
        z
        * np.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2))
        / denominator
    )

    lower = float(max(0.0, centre - spread))
    upper = float(min(1.0, centre + spread))
    return (lower, upper)