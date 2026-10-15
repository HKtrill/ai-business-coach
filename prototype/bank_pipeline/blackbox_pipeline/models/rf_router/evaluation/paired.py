"""
blackbox_pipeline.models.rf_router.evaluation.paired
=====================================================
Paired comparisons between routers on the SAME rows.

``paired_bootstrap`` resamples evaluation rows once per replicate and scores
every arm on that same resample, so each arm-to-arm difference carries only the
uncertainty in the difference — not the (much larger) uncertainty in each arm's
level. Metric definitions are identical to ``router_metrics.evaluate_router``
(same denominators); ``tests/test_rf_router_specialization.py`` checks that the
point estimates agree exactly.

``constraint_slippage`` evaluates a router's realised metrics against EVERY
outcome constraint in ``OperatingConstraints`` and decides whether the routed
set is feasible. A tuning objective that pushes the operating point to the edge
of the constraints on training OOF can buy coverage or recall with violations
out of sample; this is where that shows up.

Constraints evaluated (``CONSTRAINT_FIELDS``), all on a proportion scale so
``max_violation`` compares like with like:

* ``npv``               ``min_precision_not_subscribe``     Pass 1 NPV >= floor
* ``leakage``           ``max_subscriber_leakage_rate``     leakage rate <= cap
* ``leakage_absolute``  ``max_subscriber_leakage_absolute`` leakage rate <= cap / reference positives
* ``pass1_coverage``    ``min_pass1_coverage``              Pass 1 share >= floor
* ``pass1_support``     ``min_band_support_pass1``          Pass 1 share >= support / reference rows
* ``precision``         ``min_precision_subscribe``         Pass 2 precision >= floor
* ``pass2_recall``      ``min_recall_subscribe``            Pass 2 recall >= floor (configured denominator)
* ``pass2_support``     ``min_band_support_pass2``          Pass 2 share >= support / reference rows

The two count constraints (absolute leakage, band support) were applied by the
solver to the population the thresholds were solved on — the training split.
On another population they are evaluated at the same proportion of that
reference population (``reference_rows`` / ``reference_positives``), so a
test split a quarter of the size is held to the same requirement rather than
passing trivially or failing by construction. On the reference population
itself this is exactly the solver's count check. Comparisons are done in
integer counts, so a value exactly on a boundary is ``ok``, as in the solver.

Not evaluated, by design: ``recall_denominator`` (configures ``pass2_recall``)
and ``require_band_ordering`` (a solve-time check on the thresholds themselves,
not a property of routed outcomes).

Each constraint gets one status:

``ok`` / ``violated``
    The metric is defined and on the right / wrong side of the constraint.
``disabled``
    The constraint is not configured (``min_pass1_coverage`` or
    ``min_recall_subscribe`` is ``None``). It cannot be violated.
``empty_band``
    Inapplicable: the pass routed no rows, so its precision (NPV for Pass 1)
    has no denominator. The pass's band-support constraint still applies and
    an empty band violates it, so an empty band never makes an arm feasible.
``undefined``
    An invalid evaluation: a metric or count is missing or non-finite although
    it should exist, or a rate needs positives the evaluation split does not
    have (``evaluate_router`` reports those rates as 0.0 — that 0.0 is not a
    pass). Any ``undefined`` constraint makes ``evaluation_valid`` False,
    ``feasible`` False and ``max_violation`` NaN.

``feasible`` is True only when the evaluation is valid and no applicable
constraint is violated. ``tuning.ablation.specialization_verdict`` requires it
of any arm it recommends.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..router.decisions import PASS1, PASS2, UNCERTAIN

__all__ = [
    "BOOT_METRICS",
    "CONSTRAINT_FIELDS",
    "NON_OUTCOME_FIELDS",
    "SLIPPAGE_STATUSES",
    "paired_bootstrap",
    "constraint_slippage",
    "slippage_table",
]

SLIPPAGE_STATUSES = ("ok", "violated", "disabled", "empty_band", "undefined")

#: Outcome constraint -> the ``OperatingConstraints`` field it evaluates.
CONSTRAINT_FIELDS = {
    "npv": "min_precision_not_subscribe",
    "leakage": "max_subscriber_leakage_rate",
    "leakage_absolute": "max_subscriber_leakage_absolute",
    "pass1_coverage": "min_pass1_coverage",
    "pass1_support": "min_band_support_pass1",
    "precision": "min_precision_subscribe",
    "pass2_recall": "min_recall_subscribe",
    "pass2_support": "min_band_support_pass2",
}

#: ``OperatingConstraints`` fields that are not outcome constraints, and why.
NON_OUTCOME_FIELDS = {
    "recall_denominator": "configures how pass2_recall is computed",
    "require_band_ordering": "solve-time structural check on t1/t2, not a routed outcome",
}

_NAN = float("nan")

BOOT_METRICS = (
    "coverage",
    "abstain_rate",
    "pass1_share",
    "pass1_npv",
    "pass1_leakage_rate",
    "pass2_share",
    "pass2_precision",
    "pass2_recall",
    "abstain_base_rate",
)


def _div(num, den, zero_value):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    out = np.full(np.broadcast(num, den).shape, zero_value, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _metrics_from_sums(s: dict, n: int, total_pos) -> dict:
    """Same formulas and zero-denominator conventions as ``evaluate_router``."""
    return {
        "coverage": (s["p1"] + s["p2"]) / n,
        "abstain_rate": s["ab"] / n,
        "pass1_share": s["p1"] / n,
        "pass1_npv": _div(s["p1"] - s["p1y"], s["p1"], np.nan),
        "pass1_leakage_rate": _div(s["p1y"], total_pos, 0.0),
        "pass2_share": s["p2"] / n,
        "pass2_precision": _div(s["p2y"], s["p2"], np.nan),
        "pass2_recall": _div(s["p2y"], total_pos, 0.0),
        "abstain_base_rate": _div(s["aby"], s["ab"], np.nan),
    }


def _indicators(decisions: np.ndarray, y: np.ndarray) -> dict:
    d = np.asarray(decisions, dtype=object)
    p1, p2, ab = (d == PASS1), (d == PASS2), (d == UNCERTAIN)
    if not (p1 | p2 | ab).all():
        raise ValueError("decisions must be in {'pass1', 'pass2', 'uncertain'}")
    return {"p1": p1.astype(float), "p2": p2.astype(float), "ab": ab.astype(float),
            "p1y": (p1 & (y == 1)).astype(float), "p2y": (p2 & (y == 1)).astype(float),
            "aby": (ab & (y == 1)).astype(float)}


def paired_bootstrap(
    decisions: Mapping[str, np.ndarray],
    y_true,
    *,
    pairs: Sequence[Tuple[str, str, str, str]],
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    chunk: int = 200,
) -> pd.DataFrame:
    """
    Paired bootstrap of metric differences between arms.

    Parameters
    ----------
    decisions
        ``{arm name: decisions array}`` — every array over the SAME rows, in the
        same order as ``y_true``.
    pairs
        ``(label, new_arm, reference_arm, metric)`` tuples. ``metric`` must be in
        ``BOOT_METRICS``.
    n_boot, seed, ci, chunk
        Replicates, RNG seed, two-sided interval level, replicates per batch.

    Returns
    -------
    pandas.DataFrame
        One row per pair: both arms' point estimates, ``delta`` (new − ref),
        percentile CI over the replicates where the delta is defined,
        ``p_delta_gt_0``, and ``n_boot`` (defined replicates) beside
        ``n_boot_requested`` — a metric undefined in many replicates (an empty
        band) gives a CI over a biased subset, and the verdict checks for that.
    """
    y = np.asarray(y_true).astype(int)
    n = len(y)
    for _, new, ref, metric in pairs:
        if metric not in BOOT_METRICS:
            raise ValueError(f"metric {metric!r} not in BOOT_METRICS")
        for a in (new, ref):
            if a not in decisions:
                raise KeyError(f"no decisions for arm {a!r}")
            if len(decisions[a]) != n:
                raise ValueError(f"arm {a!r}: {len(decisions[a])} decisions for {n} rows")

    arms = sorted({a for _, new, ref, _ in pairs for a in (new, ref)})
    ind = {a: _indicators(decisions[a], y) for a in arms}
    point = {a: _metrics_from_sums({k: v.sum() for k, v in ind[a].items()}, n, y.sum())
             for a in arms}

    boot = {a: {m: np.empty(n_boot) for m in BOOT_METRICS} for a in arms}
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        total_pos = y[idx].sum(axis=1)
        for a in arms:
            sums = {k: v[idx].sum(axis=1) for k, v in ind[a].items()}
            for m, vals in _metrics_from_sums(sums, n, total_pos).items():
                boot[a][m][start:start + b] = vals

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    for label, new, ref, metric in pairs:
        delta = boot[new][metric] - boot[ref][metric]
        finite = delta[np.isfinite(delta)]
        rows.append({
            "comparison": label, "new": new, "reference": ref, "metric": metric,
            "new_value": float(point[new][metric]),
            "reference_value": float(point[ref][metric]),
            "delta": float(point[new][metric] - point[ref][metric]),
            "ci_low": float(np.quantile(finite, lo_q)) if finite.size else np.nan,
            "ci_high": float(np.quantile(finite, hi_q)) if finite.size else np.nan,
            "p_delta_gt_0": float((finite > 0).mean()) if finite.size else np.nan,
            "n_boot": int(finite.size),
            "n_boot_requested": int(n_boot),
        })
    return pd.DataFrame(rows)


def _as_float(metrics: Mapping, key: str) -> float:
    v = metrics.get(key, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return _NAN


def _as_count(metrics: Mapping, key: str) -> Optional[int]:
    """A non-negative integer count, or None when missing / non-finite / fractional."""
    v = _as_float(metrics, key)
    if not np.isfinite(v) or v < 0 or v != int(v):
        return None
    return int(v)


def _floor_status(value: float, floor: float) -> Tuple[str, float]:
    """``value >= floor`` is satisfied; equality is ``ok`` (solver convention)."""
    if not np.isfinite(value):
        return "undefined", _NAN
    shortfall = float(floor - value)
    return ("violated" if value < floor else "ok"), shortfall


def _band_precision_status(value: float, band_n: Optional[int], floor: float) -> Tuple[str, float]:
    """(status, shortfall) for a precision-type constraint on one band."""
    if band_n is None:
        return "undefined", _NAN
    if band_n == 0:
        return "empty_band", _NAN
    return _floor_status(value, floor)


def _support_status(band_n: Optional[int], n_eval: Optional[int], support: int,
                    reference_rows: int) -> Tuple[str, float]:
    """Band share >= support / reference_rows, compared exactly in counts."""
    if band_n is None or not n_eval:
        return "undefined", _NAN
    shortfall = support / reference_rows - band_n / n_eval
    ok = band_n * reference_rows >= support * n_eval
    return ("ok" if ok else "violated"), float(shortfall)


def constraint_slippage(metrics: Mapping, constraints, *, reference_rows: int,
                        reference_positives: int) -> dict:
    """
    Evaluate realised metrics against every outcome constraint.

    Parameters
    ----------
    metrics
        One ``evaluate_router`` output. Uses ``n_samples``, ``n_positives``,
        ``pass1_n``, ``pass1_npv``, ``pass1_subscribers_lost``,
        ``pass1_leakage_rate``, ``pass2_n``, ``pass2_precision``,
        ``pass2_recall`` and ``pass2_positives_found``.
    constraints
        The ``OperatingConstraints`` the router was solved under.
    reference_rows, reference_positives
        Rows and positives of the population the constraints were applied to
        when the thresholds were solved (the training split). Required: the
        count constraints cannot be translated to another population without
        them, and silently skipping them would weaken the feasibility check.

    Returns
    -------
    dict
        ``feasible``, ``evaluation_valid``, ``max_violation``, ``n_violated`` and
        the ``violated`` / ``undefined`` / ``disabled`` constraint names, then a
        ``<name>_margin`` (positive = violation, negative = slack, proportion
        scale) and ``<name>_status`` for each constraint in ``CONSTRAINT_FIELDS``.
    """
    if int(reference_rows) < 1 or int(reference_positives) < 1:
        raise ValueError("reference_rows and reference_positives must be >= 1")
    c = constraints
    n_eval = _as_count(metrics, "n_samples")
    n_pos = _as_count(metrics, "n_positives")
    p1_n = _as_count(metrics, "pass1_n")
    p2_n = _as_count(metrics, "pass2_n")
    lost = _as_count(metrics, "pass1_subscribers_lost")
    found = _as_count(metrics, "pass2_positives_found")
    has_pos = bool(n_pos)  # None or 0 -> rates over positives are undefined

    out: Dict[str, Tuple[str, float]] = {}

    # ---- Pass 1 -------------------------------------------------------------
    out["npv"] = _band_precision_status(
        _as_float(metrics, "pass1_npv"), p1_n, c.min_precision_not_subscribe)

    leak = _as_float(metrics, "pass1_leakage_rate")
    if not has_pos or not np.isfinite(leak):
        out["leakage"] = ("undefined", _NAN)
    else:
        excess = float(leak - c.max_subscriber_leakage_rate)
        out["leakage"] = ("violated" if leak > c.max_subscriber_leakage_rate else "ok", excess)

    if not has_pos or lost is None:
        out["leakage_absolute"] = ("undefined", _NAN)
    else:
        cap = int(c.max_subscriber_leakage_absolute)
        excess = lost / n_pos - cap / reference_positives
        ok = lost * reference_positives <= cap * n_pos
        out["leakage_absolute"] = ("ok" if ok else "violated", float(excess))

    if c.min_pass1_coverage is None:
        out["pass1_coverage"] = ("disabled", _NAN)
    elif p1_n is None or not n_eval:
        out["pass1_coverage"] = ("undefined", _NAN)
    else:
        out["pass1_coverage"] = _floor_status(p1_n / n_eval, float(c.min_pass1_coverage))

    out["pass1_support"] = _support_status(p1_n, n_eval, int(c.min_band_support_pass1),
                                           int(reference_rows))

    # ---- Pass 2 -------------------------------------------------------------
    out["precision"] = _band_precision_status(
        _as_float(metrics, "pass2_precision"), p2_n, c.min_precision_subscribe)

    if c.min_recall_subscribe is None:
        out["pass2_recall"] = ("disabled", _NAN)
    elif c.recall_denominator == "global":
        rec = _as_float(metrics, "pass2_recall")
        out["pass2_recall"] = (_floor_status(rec, float(c.min_recall_subscribe))
                               if has_pos else ("undefined", _NAN))
    else:  # "remainder": positives Pass 1 left for Pass 2
        rem_pos = None if (n_pos is None or lost is None) else n_pos - lost
        if not rem_pos or found is None:
            out["pass2_recall"] = ("undefined", _NAN)
        else:
            out["pass2_recall"] = _floor_status(found / rem_pos, float(c.min_recall_subscribe))

    out["pass2_support"] = _support_status(p2_n, n_eval, int(c.min_band_support_pass2),
                                           int(reference_rows))

    # ---- summary ------------------------------------------------------------
    undefined = [k for k, (st, _) in out.items() if st == "undefined"]
    violated = [k for k, (st, _) in out.items() if st == "violated"]
    disabled = [k for k, (st, _) in out.items() if st == "disabled"]
    valid = not undefined
    margins = [m for st, m in out.values() if st in ("ok", "violated")]
    result = {
        "feasible": bool(valid and not violated),
        "evaluation_valid": bool(valid),
        "max_violation": float(max([0.0, *margins])) if valid else _NAN,
        "n_violated": len(violated),
        "violated": ", ".join(violated),
        "undefined": ", ".join(undefined),
        "disabled": ", ".join(disabled),
    }
    for name, (status, margin) in out.items():
        result[f"{name}_margin"] = margin
        result[f"{name}_status"] = status
    return result


def slippage_table(metrics_frame: pd.DataFrame, constraints, *, reference_rows: int,
                   reference_positives: int) -> pd.DataFrame:
    """``constraint_slippage`` for every column of a ``compare_routers`` frame."""
    return pd.DataFrame({
        arm: constraint_slippage(metrics_frame[arm].to_dict(), constraints,
                                 reference_rows=reference_rows,
                                 reference_positives=reference_positives)
        for arm in metrics_frame.columns
    })
