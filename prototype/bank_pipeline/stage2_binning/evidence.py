"""
stage2_binning.evidence
=======================
Pure functions of the PR 39 Step 3A evidence (execution contract 1.0.0,
section 6.2): per-bin statistics, the registered triggers T1-T4 (decision D6),
the retain-or-revise outcome, and the router-utility summary E-B2.

Nothing here fits, reads data or reads a label it is not handed. Every
statistic is computed from integer counts, so the record is byte-stable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from shared.stage2.binning import BINNING_STRATEGY, RF_FEATURES_BINARY

__all__ = ["EvidenceError", "FAMILIES", "DIRECTIONS", "CRITERIA", "HISTORICAL_RATIONALE",
           "family_bins", "family_labels", "wilson", "bin_table", "joint_table", "moved_rows",
           "triggers", "decide", "rule_literals", "rule_entries", "eb2_summary"]


class EvidenceError(RuntimeError):
    """The evidence cannot be computed or does not verify; the reason is stated."""


#: Scope (section 6.2): NSD and ECI only; the other families are retained
#: without revalidation.
FAMILIES = ("NSD", "ECI")

#: Intended semantics (section 6.2, from the rationale in shared/stage2/binning.py):
#: both families rise monotonically; ``*_cold`` lies below baseline;
#: ``nsd_elevated`` / ``eci_warm`` above it and ``*_hot`` well above; the
#: transitional ``nsd_warm`` / ``eci_mid`` claim no direction.
DIRECTIONS = {
    "nsd_cold": "below", "nsd_warm": None, "nsd_elevated": "above", "nsd_hot": "above",
    "eci_cold": "below", "eci_mid": None, "eci_warm": "above", "eci_hot": "above",
}

#: Decision D6 (approved at the Commit 1 review).
CRITERIA = {
    "T1": {"name": "sparse bin", "min_expected_positives": 10,
           "rule": "in any outer fold, held-out support x that fold's base rate < 10"},
    "T2": {"name": "reversal", "interval": "Wilson 95%", "z": 1.959963984540054,
           "rule": "pooled lift interval wholly on the wrong side of 1.0 for the intended "
                   "direction (bins with no intended direction cannot fire)"},
    "T3": {"name": "ordering", "min_folds": 4, "n_folds": 5,
           "rule": "an adjacent pair inverted in pooled OOF lift and in at least 4 of 5 folds"},
    "T4": {"name": "hot collapse", "min_lift": 1.5,
           "rule": "a *_hot bin's pooled OOF lift < 1.5"},
    "note": "binning criteria, not routing obligations: the routed-set floors (Pass 1 125, "
            "Pass 2 75 rows) are not applied to bins and are unchanged",
}

#: The per-bin lifts the thresholds were locked from (comments in
#: shared/stage2/binning.py: in-sample decile lift, research notebook Cell 13C).
HISTORICAL_RATIONALE = {
    "NSD": "Dead <=0.052 (0.27-0.49x), transition to 0.19 (~0.9x), warm 0.19-0.23 "
           "(1.04-1.43x), hot >0.23 (4.08x lift)",
    "ECI": "Dead <=0.029 (0.44-0.52x), mid 0.029-0.076 (mixed), warm 0.076-0.095 "
           "(1.49-1.67x), hot >0.095 (3.53x)",
}


def family_bins(family: str) -> List[str]:
    return [b[0] for b in BINNING_STRATEGY[family]["bins"]]


def family_labels(bins: pd.DataFrame, family: str) -> np.ndarray:
    """Each row's bin name in ``family``; exactly one bin per row, else refused."""
    names = family_bins(family)
    M = bins[names].to_numpy(dtype=np.int64)
    if not np.all(M.sum(axis=1) == 1):
        raise EvidenceError(f"{family}: rows without exactly one bin; the thresholds leave a "
                            "gap or an overlap")
    return np.asarray(names, dtype=object)[M.argmax(axis=1)]


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def wilson(positives: int, n: int, z: float = CRITERIA["T2"]["z"]) -> List[Optional[float]]:
    """Wilson score interval for a proportion; ``[None, None]`` for an empty bin."""
    if n <= 0:
        return [None, None]
    p = positives / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _cell(support: int, positives: int, base: float) -> Dict[str, Any]:
    rate = positives / support if support else None
    lo, hi = wilson(positives, support)
    return {"support": support, "positives": positives, "rate": rate,
            "lift": None if rate is None else rate / base,
            "lift_wilson95": [None if lo is None else lo / base,
                              None if hi is None else hi / base],
            "expected_positives": support * base}


def bin_table(labels: np.ndarray, y: np.ndarray, fold_id: np.ndarray, names: Sequence[str],
              n_folds: int, *, per_fold: bool = True) -> Dict[str, Any]:
    """
    Per bin, pooled and (``per_fold``) per outer fold: support, positives,
    rate, lift and its Wilson 95% interval, and the expected positives
    (support x base rate). The pooled lift is against the pooled base rate, a
    fold's against that fold's.
    """
    y = np.asarray(y).astype(int)
    fold_id = np.asarray(fold_id)
    n, pos = len(y), int(y.sum())
    base = pos / n
    fold_base = []
    for k in range(n_folds):
        fk = fold_id == k
        fold_base.append({"rows": int(fk.sum()), "positives": int(y[fk].sum()),
                          "base_rate": int(y[fk].sum()) / int(fk.sum())})
    out = {"base_rate": {"pooled": {"rows": n, "positives": pos, "base_rate": base}},
           "bins": {}}
    if per_fold:
        out["base_rate"]["folds"] = fold_base
    for name in names:
        m = labels == name
        entry = {"pooled": _cell(int(m.sum()), int(y[m].sum()), base)}
        if per_fold:
            entry["folds"] = [_cell(int((m & (fold_id == k)).sum()),
                                    int(y[m & (fold_id == k)].sum()), fold_base[k]["base_rate"])
                              for k in range(n_folds)]
        out["bins"][name] = entry
    return out


def joint_table(nsd: np.ndarray, eci: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """NSD x ECI rows and positives (overlap of the two families; descriptive)."""
    y = np.asarray(y).astype(int)
    rows, positives = [], []
    for a in family_bins("NSD"):
        rows.append([int(((nsd == a) & (eci == b)).sum()) for b in family_bins("ECI")])
        positives.append([int(y[(nsd == a) & (eci == b)].sum()) for b in family_bins("ECI")])
    return {"rows_are": family_bins("NSD"), "columns_are": family_bins("ECI"),
            "rows": rows, "positives": positives}


def moved_rows(oof: np.ndarray, in_sample: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    """Rows whose bin differs between the in-sample and the out-of-fold engineer."""
    moved = oof != in_sample
    return {"rows": int(moved.sum()),
            "by_in_sample_bin": {b: int((moved & (in_sample == b)).sum()) for b in names}}


# ----------------------------------------------------------------------
# triggers and decision
# ----------------------------------------------------------------------
def _wrong_side(direction: Optional[str], interval) -> bool:
    lo, hi = interval
    if direction is None or lo is None:
        return False
    return hi < 1.0 if direction == "above" else lo > 1.0


def triggers(family: str, oof: Mapping[str, Any]) -> Dict[str, Any]:
    """T1-T4 for one family, each with the numbers it was decided on."""
    names = family_bins(family)
    bins = oof["bins"]
    c = CRITERIA
    t1 = {b: [f["expected_positives"] for f in bins[b]["folds"]] for b in names}
    t1_fired = sorted(b for b, e in t1.items() if min(e) < c["T1"]["min_expected_positives"])
    t2 = {b: {"direction": DIRECTIONS[b], "pooled_lift_wilson95": bins[b]["pooled"]["lift_wilson95"]}
          for b in names}
    t2_fired = sorted(b for b, v in t2.items()
                      if _wrong_side(v["direction"], v["pooled_lift_wilson95"]))

    def inverted(a, b):                  # an empty bin has no lift (and fires T1)
        return a is not None and b is not None and a > b

    t3 = []
    for lower, upper in zip(names, names[1:]):
        pooled = inverted(bins[lower]["pooled"]["lift"], bins[upper]["pooled"]["lift"])
        folds = [k for k, (a, b) in enumerate(zip(bins[lower]["folds"], bins[upper]["folds"]))
                 if inverted(a["lift"], b["lift"])]
        t3.append({"pair": [lower, upper], "inverted_pooled": bool(pooled),
                   "inverted_folds": folds,
                   "fired": bool(pooled) and len(folds) >= c["T3"]["min_folds"]})
    hot = [b for b in names if b.endswith("_hot")]
    t4 = {b: bins[b]["pooled"]["lift"] for b in hot}
    t4_fired = sorted(b for b, lift in t4.items() if lift is None or lift < c["T4"]["min_lift"])
    return {
        "T1": {"fired": bool(t1_fired), "bins": t1_fired,
               "min_expected_positives_by_bin": {b: min(e) for b, e in t1.items()}},
        "T2": {"fired": bool(t2_fired), "bins": t2_fired, "by_bin": t2},
        "T3": {"fired": any(p["fired"] for p in t3), "pairs": t3},
        "T4": {"fired": bool(t4_fired), "bins": t4_fired, "pooled_lift": t4},
    }


def decide(trig: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Section 6.2: no trigger, the family's thresholds are retained unchanged; a
    trigger requires the scripted candidate (decile lift, out of fold) to be
    derived and recorded before any router fit. This record never adopts one.
    """
    out = {}
    for family, t in trig.items():
        fired = [k for k in ("T1", "T2", "T3", "T4") if t[k]["fired"]]
        out[family] = {
            "flagged": bool(fired), "fired": fired,
            "outcome": "candidate_required" if fired else "retain",
            "thresholds": [list(b) for b in BINNING_STRATEGY[family]["bins"]],
            "candidate": None,
            "preflight": ("required for any candidate (fold 0, both arms, frozen settings)"
                          if fired else "not required: no trigger fired"),
        }
    return out


# ----------------------------------------------------------------------
# E-B2: router utility in the accepted PR 38 run (fit-time records only)
# ----------------------------------------------------------------------
def rule_literals(rules: Iterable[Mapping[str, Any]]) -> List[List[List[Any]]]:
    """Each rule's conditions as sorted ``[feature, value]`` pairs."""
    out = []
    for r in rules:
        seg = [[str(f), int(v)] for f, v in r["segment"]]
        out.append(sorted(seg))
    return out


def rule_entries(rules: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Selected rules for the record: id, sorted conditions and fit-time precision/coverage."""
    out = []
    for r in rules:
        entry = {"rule_id": int(r["rule_id"]),
                 "conditions": sorted([str(f), int(v)] for f, v in r["segment"]),
                 "precision": None if r.get("precision") is None else round(float(r["precision"]), 12)}
        if r.get("coverage") is not None:
            entry["coverage"] = r["coverage"]
        out.append(entry)
    return out


def _literal_counts(rule_sets: Sequence[Sequence[Sequence[Any]]], name: str) -> Dict[str, int]:
    pos = sum(1 for rules in rule_sets for seg in rules for f, v in seg if f == name and v == 1)
    neg = sum(1 for rules in rule_sets for seg in rules for f, v in seg if f == name and v == 0)
    return {"as_1": pos, "as_0": neg}


def eb2_summary(*, pass1_by_fold: Sequence[Sequence[Any]], pass2_ids_by_fold: Sequence[Sequence[int]],
                pass1_full_train: Sequence[Any], pass2_full_train: Sequence[Any],
                rf_importance: Mapping[str, Sequence[float]], rf_feature_names: Sequence[str]
                ) -> Dict[str, Any]:
    """
    Which selected GLASS rules contain each NSD/ECI bin (as ``bin=1`` or
    ``bin=0``), which bins they are conjoined with, and each bin's impurity
    importance and rank in the saved RF Router forests. Descriptive: it
    vetoes nothing outside the preflight and is never a decision input.
    """
    names_rf = [str(f) for f in rf_feature_names]
    if sorted(names_rf) != sorted(RF_FEATURES_BINARY) or len(set(names_rf)) != len(names_rf):
        raise EvidenceError("the RF Router's feature names are not RF_FEATURES_BINARY")
    by_name = {}
    ranks = {}
    for pass_name, imp in rf_importance.items():
        imp = np.asarray(imp, dtype=float)
        if imp.shape != (len(names_rf),):
            raise EvidenceError(f"{pass_name}: {imp.shape} importances for {len(names_rf)} features")
        by_name[pass_name] = dict(zip(names_rf, imp.tolist()))
        order = np.argsort(-imp, kind="stable")
        ranks[pass_name] = {names_rf[i]: r + 1 for r, i in enumerate(order)}
    names = [b for fam in FAMILIES for b in family_bins(fam)]
    per_bin = {}
    for b in names:
        folds = [k for k, rules in enumerate(pass1_by_fold)
                 if any(f == b for seg in rules for f, _ in seg)]
        partners = sorted({f for rules in [*pass1_by_fold, pass1_full_train, pass2_full_train]
                           for seg in rules if any(g == b for g, _ in seg)
                           for f, _ in seg if f != b})
        per_bin[b] = {
            "glass_pass1_folds_with_bin": folds,
            "glass_pass1_folds": _literal_counts(pass1_by_fold, b),
            "glass_pass1_full_train": _literal_counts([pass1_full_train], b),
            "glass_pass2_full_train": _literal_counts([pass2_full_train], b),
            "conjoined_with": partners,
            "rf_mdi": {p: round(by_name[p][b], 12) for p in rf_importance},
            "rf_mdi_rank_of_29": {p: ranks[p][b] for p in rf_importance},
        }
    fam_mdi = {fam: {p: round(sum(by_name[p][b] for b in family_bins(fam)), 12)
                     for p in rf_importance} for fam in FAMILIES}
    return {
        "bins": per_bin,
        "family_rf_mdi": fam_mdi,
        "pass2_rule_ids_by_fold": [list(map(int, ids)) for ids in pass2_ids_by_fold],
    }
