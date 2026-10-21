"""
glass_pipeline.meta_ebm.router_evidence
=======================================
How strongly the GLASS Router's vote counts in the Meta-EBM arbiter
(PR 36 F25).

The router votes by route (Pass 2 -> SUBSCRIBE, Pass 1 -> NOT_SUBSCRIBE). Its
evidence is the matched rule's observed positive rate ``r`` measured against
the base rate ``b`` of the rows the router was trained on, in the vote's
direction:

    Pass 2   confidence = max(0, r - b)   rule rate above the base rate
    Pass 1   confidence = max(0, b - r)   rule rate below the base rate
    no route no vote

That is the rule LR and EBM already follow (distance of the voter's estimate
from its own decision point, in probability units): the router's decision
point is the base rate, where a rule stops being evidence for either route.
A rule on the wrong side of the base rate gets zero confidence, never
confidence for the opposite outcome. ``r`` is the rule's in-sample rate on
the router's training rows (selected rules: optimistic), not a calibrated
P(y = 1) for the routed row.

Before F25 the confidence was |p - 0.5| on the router's stored probability.
Pass 2 rules with precision below ~0.44 (every rule selected so far) then
cast WEAKER subscribe votes the more precise they were.

Recovering ``r``
----------------
The router stores ``p`` = ``GlassRouterPipeline.predict_proba``:

    Pass 1   p = b * (1 - precision_0) = b * r          precision_0 = P(y = 0 | rule)
    Pass 2   p = b + (1 - b) * precision_1               precision_1 = P(y = 1 | rule) = r

so ``r = p / b`` (Pass 1) and ``r = (p - b) / (1 - b)`` (Pass 2) exactly,
given the ``b`` the producing model used: for an out-of-fold training row the
base rate of its fold model's training rows (``fold_base_rates``), for a test
row the full training base rate. ``glass_blend_mismatch`` checks this against
the per-row rule precision the router artifact stores.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["fold_base_rates", "rule_positive_rate", "directional_confidence",
           "glass_blend_mismatch"]

_TOL = 1e-9


def fold_base_rates(y_train, fold_id) -> np.ndarray:
    """Per training row: the positive rate of the rows its fold model trained on."""
    y = np.asarray(y_train, dtype=float)
    f = np.asarray(fold_id)
    if len(f) != len(y):
        raise ValueError(f"{len(f)} fold ids for {len(y)} labels")
    out = np.empty(len(y), dtype=float)
    for k in np.unique(f):
        out[f == k] = y[f != k].mean()
    return out


def _base(base_rate, n: int) -> np.ndarray:
    b = np.broadcast_to(np.asarray(base_rate, dtype=float), (n,)).astype(float)
    if not ((b > 0) & (b < 1)).all():
        raise ValueError("router base rates must lie strictly between 0 and 1")
    return b


def rule_positive_rate(glass_prob, pass1_mask, pass2_mask, base_rate) -> np.ndarray:
    """The matched rule's positive rate ``r`` per row; NaN where not routed."""
    p = np.asarray(glass_prob, dtype=float)
    p1, p2 = np.asarray(pass1_mask, dtype=bool), np.asarray(pass2_mask, dtype=bool)
    b = _base(base_rate, len(p))
    r = np.full(len(p), np.nan)
    r[p1] = p[p1] / b[p1]
    r[p2] = (p[p2] - b[p2]) / (1.0 - b[p2])
    routed = p1 | p2
    bad = routed & ~((r >= -_TOL) & (r <= 1 + _TOL))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} routed row(s) have router probabilities that are not GLASS "
            "rule blends of their base rate (recovered rule rate outside [0, 1])")
    r[routed] = np.clip(r[routed], 0.0, 1.0)
    return r


def directional_confidence(rate, pass1_mask, pass2_mask, base_rate) -> np.ndarray:
    """Pass 2: max(0, r - b); Pass 1: max(0, b - r); 0 where not routed."""
    r = np.asarray(rate, dtype=float)
    p1, p2 = np.asarray(pass1_mask, dtype=bool), np.asarray(pass2_mask, dtype=bool)
    b = _base(base_rate, len(r))
    conf = np.zeros(len(r))
    conf[p2] = np.maximum(0.0, r[p2] - b[p2])
    conf[p1] = np.maximum(0.0, b[p1] - r[p1])
    return conf


def glass_blend_mismatch(glass_prob, decisions_pass1, decisions_pass2, base_rate,
                         rule_precision) -> Optional[str]:
    """
    ``None`` if every routed row's stored probability is the GLASS rule blend of
    its stored rule precision and base rate; otherwise a description. Unrouted
    rows match no rule, so their stored precision is not read.
    """
    p1, p2 = np.asarray(decisions_pass1, bool), np.asarray(decisions_pass2, bool)
    prec = np.asarray(rule_precision, dtype=float)
    if prec.shape != np.asarray(glass_prob).shape:
        return f"rule precision has shape {prec.shape}, probabilities {np.shape(glass_prob)}"
    routed = p1 | p2
    bad = routed & ~np.isfinite(prec)
    if bad.any():
        return f"{int(bad.sum())} routed row(s) have non-finite stored rule precision"
    try:
        r = rule_positive_rate(glass_prob, p1, p2, base_rate)
    except ValueError as exc:
        return str(exc)
    want = np.where(p1, 1.0 - prec, prec)
    err = np.abs(r[routed] - want[routed])
    if len(err) and not (err <= 1e-6).all():                 # NaN-safe: NaN fails <=
        return f"max |recovered rule rate - stored precision| = {np.nanmax(err):.3g}"
    return None
