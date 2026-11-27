"""
stage1_calibration.evidence
===========================
Evidence E-C and the registered rule C1-C5 of the frozen PR 39 execution
contract (``documentation/methodology/pr39_execution_contract.md`` 1.0.0,
section 6.1), as pure functions of training-side vectors.

Nothing here fits a model, reads a file or sees the test split. The inputs are,
per arm, the out-of-fold vectors one nested calibration run produced
(``shared.stage1.oof_calibration.calibrate_oof`` with ``method="auto"``): the
base scores and both families' calibrated probabilities, all from the same
base models and the same outer folds. A family's vector is exactly the
out-of-fold output a fixed-family policy would give from the same tuned base
model: each family's calibrators are fitted on their own.

Registered settings (contract 6.1; not parameters of the decision)
------------------------------------------------------------------
ECE and the Murphy terms   ``shared.metrics.calculate_ece`` bins (10 equal-width)
Paired bootstrap           2,000 replicates, seed 42, rows resampled with
                           replacement inside every (outer fold, label) stratum
                           keeping stratum sizes; percentile 95 % interval
                           (numpy ``linear``). The resampling is the PR 38
                           ``stratified_paired_bootstrap`` convention.
Stage 4 input diagnostic   ``find_recall_threshold`` of the GLASS Stage 4 arbiter
                           at recall 0.70, on each family's vector alone
C3                         (a) pooled ΔBrier < 0 and interval upper bound < 0;
                           (b) ΔBrier < 0 in >= 8 of 10 outer folds;
                           (c) isotonic achieved recall at the target <= 0.71;
                           all three in BOTH arms, else ``sigmoid`` (C2, C4)

ΔBrier is always isotonic minus sigmoid, so negative favours isotonic.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from shared.metrics import calculate_ece

__all__ = ["FAMILIES", "ARMS", "ECE_BINS", "BOOTSTRAP", "STAGE4_TARGET_RECALL",
           "C3_MIN_FOLDS", "C3_MAX_RECALL", "EvidenceError", "murphy_terms",
           "calibration_metrics", "stratified_bootstrap_delta", "stage4_inputs",
           "arm_evidence", "apply_rule"]

FAMILIES = ("sigmoid", "isotonic")
ARMS = ("glass_lr", "blackbox_mlp")
ECE_BINS = 10
BOOTSTRAP = {"replicates": 2000, "seed": 42, "level": 0.95,
             "strata": "outer fold x label, stratum sizes kept",
             "interval": "percentile (numpy linear)"}
STAGE4_TARGET_RECALL = 0.70
C3_MIN_FOLDS = 8
C3_MAX_RECALL = 0.71


class EvidenceError(ValueError):
    """The evidence is incomplete or invalid: no decision is taken (C4)."""


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _bin_masks(p: np.ndarray, n_bins: int):
    """The bins of ``shared.metrics.calculate_ece``: the first closed on the left."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        yield (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)


def murphy_terms(y, p, n_bins: int = ECE_BINS) -> Dict[str, float]:
    """Reliability, resolution and uncertainty of the binned Murphy decomposition."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    base, n = float(y.mean()), len(y)
    rel = res = 0.0
    for m in _bin_masks(p, n_bins):
        k = int(m.sum())
        if k:
            o, f = float(y[m].mean()), float(p[m].mean())
            rel += k / n * (f - o) ** 2
            res += k / n * (o - base) ** 2
    return {"reliability": float(rel), "resolution": float(res),
            "uncertainty": base * (1.0 - base)}


def calibration_metrics(y, p, base_score, n_bins: int = ECE_BINS) -> Dict[str, Any]:
    """One family on one set of rows: accuracy, reliability, resolution, ties."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    values, counts = np.unique(p, return_counts=True)
    return {
        "rows": int(len(y)), "positives": int(y.sum()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(calculate_ece(y, p, n_bins=n_bins)),
        **murphy_terms(y, p, n_bins),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "distinct_values": int(len(values)),
        "distinct_base_scores": int(len(np.unique(np.asarray(base_score, dtype=float)))),
        "largest_tie_share": float(counts.max() / len(p)),
    }


def stratified_row_weights(y, fold_id, replicates, seed):
    """Yield one fold/class-stratified multiplicity vector, reused between iterations."""
    strata = sorted({(int(f), int(v)) for f, v in zip(fold_id, y)})
    members = [np.flatnonzero((fold_id == f) & (y == v)) for f, v in strata]
    rng = np.random.default_rng(int(seed))
    w = np.zeros(len(y), dtype=int)
    for _ in range(int(replicates)):
        for rows in members:
            w[rows] = np.bincount(rng.integers(0, len(rows), len(rows)), minlength=len(rows))
        yield w


def stratified_bootstrap_delta(d, y, fold_id, *, replicates: int = BOOTSTRAP["replicates"],
                               seed: int = BOOTSTRAP["seed"],
                               level: float = BOOTSTRAP["level"]) -> Dict[str, Any]:
    """
    Percentile interval of mean(``d``) under the registered stratified resampling.

    ``d`` is the per-row Brier difference. Every replicate draws each (outer
    fold, label) stratum with replacement at its own size, as PR 38's
    ``stratified_paired_bootstrap`` does, so the row count never changes.
    """
    d = np.asarray(d, dtype=float)
    y, fold_id = np.asarray(y).astype(int), np.asarray(fold_id).astype(int)
    n = len(d)
    means = np.empty(int(replicates))
    for r, w in enumerate(stratified_row_weights(y, fold_id, replicates, seed)):
        means[r] = float(w @ d) / n
    lower, upper = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"point": float(d.mean()), "lower": float(lower), "upper": float(upper),
            "replicates": int(replicates), "seed": int(seed), "level": float(level),
            "strata": len(set(zip(fold_id, y)))}


def stage4_inputs(y, p_by_family: Mapping[str, np.ndarray],
                  target_recall: float = STAGE4_TARGET_RECALL) -> Dict[str, Any]:
    """
    What each family would hand the GLASS Stage 4 arbiter, Stage 1 alone.

    The arbiter cuts each upstream at ``find_recall_threshold`` (the highest
    observed value whose recall is >= the target) and uses |p - t| as that
    voter's confidence; tied probabilities make the achievable recall jump.
    """
    from glass_pipeline.glass_arbiter.meta_stage import find_recall_threshold

    y = np.asarray(y).astype(int)
    out: Dict[str, Any] = {"target_recall": float(target_recall)}
    votes = {}
    for f in FAMILIES:
        p = np.asarray(p_by_family[f], dtype=float)
        t = float(find_recall_threshold(y, p, target_recall))
        vote = p >= t
        votes[f] = vote
        out[f] = {"threshold": t, "achieved_recall": float(vote[y == 1].mean()),
                  "flags": int(vote.sum()),
                  "distinct_confidences": int(len(np.unique(np.abs(p - t))))}
    differ = votes["sigmoid"] != votes["isotonic"]
    out["vote_differences"] = {"rows": int(differ.sum()),
                               "positives": int((differ & (y == 1)).sum()),
                               "negatives": int((differ & (y == 0)).sum())}
    out["spearman"] = float(spearmanr(p_by_family["sigmoid"], p_by_family["isotonic"])[0])
    return out


# ----------------------------------------------------------------------
# One arm
# ----------------------------------------------------------------------
def _check_vectors(y, fold_id, base, p_by_family) -> None:
    n = len(y)
    if set(np.unique(y)) != {0, 1}:
        raise EvidenceError("labels must contain both classes and nothing else")
    arrays = {"fold_id": fold_id, "base_score": base, **p_by_family}
    for name, a in arrays.items():
        if len(a) != n:
            raise EvidenceError(f"{name}: {len(a)} rows for {n} labels")
        if not np.isfinite(np.asarray(a, dtype=float)).all():
            raise EvidenceError(f"{name}: non-finite values")
    for f in FAMILIES:
        p = np.asarray(p_by_family[f], dtype=float)
        if p.min() < 0 or p.max() > 1:
            raise EvidenceError(f"{f}: probabilities outside [0, 1]")


def arm_evidence(y, fold_id, base_score, p_by_family: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """E-C for one arm: pooled and per-fold metrics, ΔBrier, bootstrap, Stage 4 inputs."""
    if set(p_by_family) != set(FAMILIES):
        raise EvidenceError(f"need exactly the families {FAMILIES}, got {sorted(p_by_family)}")
    y = np.asarray(y).astype(int)
    fold_id = np.asarray(fold_id).astype(int)
    base = np.asarray(base_score, dtype=float)
    p = {f: np.asarray(p_by_family[f], dtype=float) for f in FAMILIES}
    _check_vectors(y, fold_id, base, p)

    d = (p["isotonic"] - y) ** 2 - (p["sigmoid"] - y) ** 2
    folds = []
    for k in sorted(np.unique(fold_id).tolist()):
        m = fold_id == k
        folds.append({"fold": int(k),
                      **{f: calibration_metrics(y[m], p[f][m], base[m]) for f in FAMILIES},
                      "delta_brier": float(d[m].mean())})
    deltas = np.array([r["delta_brier"] for r in folds])
    return {
        "pooled": {f: calibration_metrics(y, p[f], base) for f in FAMILIES},
        "folds": folds,
        "delta_brier": {
            "definition": "Brier(isotonic) - Brier(sigmoid), per row then averaged",
            "pooled": float(d.mean()),
            "bootstrap": stratified_bootstrap_delta(d, y, fold_id),
            "folds_isotonic_lower": int((deltas < 0).sum()),
            "folds": int(len(deltas)),
            "fold_sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
        },
        "stage4": stage4_inputs(y, p),
    }


# ----------------------------------------------------------------------
# The rule (C1-C5)
# ----------------------------------------------------------------------
def apply_rule(arms: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """
    The registered conservative rule. ``isotonic`` only if every C3 condition
    holds in both arms; otherwise ``sigmoid``. The neutral alternative (D1) is
    stated for information and never changes the selection (C5).
    """
    if set(arms) != set(ARMS):
        raise EvidenceError(f"the rule needs evidence for both arms {ARMS}, got {sorted(arms)}")
    checks: Dict[str, Any] = {}
    for arm in ARMS:
        dl, s4 = arms[arm]["delta_brier"], arms[arm]["stage4"]["isotonic"]
        checks[arm] = {
            "C3a": {"pooled_delta_brier": dl["pooled"], "interval_upper": dl["bootstrap"]["upper"],
                    "passed": bool(dl["pooled"] < 0 and dl["bootstrap"]["upper"] < 0)},
            "C3b": {"folds_isotonic_lower": dl["folds_isotonic_lower"], "folds": dl["folds"],
                    "required": C3_MIN_FOLDS,
                    "passed": bool(dl["folds_isotonic_lower"] >= C3_MIN_FOLDS)},
            "C3c": {"achieved_recall": s4["achieved_recall"], "maximum": C3_MAX_RECALL,
                    "passed": bool(s4["achieved_recall"] <= C3_MAX_RECALL)},
        }
    isotonic = all(c["passed"] for arm in ARMS for c in checks[arm].values())
    relative = {arm: arms[arm]["delta_brier"]["pooled"] / arms[arm]["pooled"]["sigmoid"]["brier"]
                for arm in ARMS}
    mean_relative = float(np.mean(list(relative.values())))
    return {
        "rule": "C1-C5, PR 39 execution contract 1.0.0, section 6.1 (conservative)",
        "C1": "one family for both arms and every outer fold",
        "C3": checks,
        "selected_family": "isotonic" if isotonic else "sigmoid",
        "basis": ("C3 holds in both arms" if isotonic else
                  "C3 does not hold in both arms: the C2 default applies (C4)"),
        "statement": ("the family the registered conservative rule selects; "
                      "not a claim that it is the better-calibrated family"),
        "neutral_alternative": {
            "definition": "lower mean relative ΔBrier across the arms, sigmoid on ties (D1)",
            "relative_delta_brier": relative,
            "mean_relative_delta_brier": mean_relative,
            "would_select": "isotonic" if mean_relative < 0 else "sigmoid",
            "status": "descriptive only (C5); does not change the selection",
        },
    }
