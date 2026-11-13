"""
stage2_crossfit.evidence
========================
Fit-time evidence for one outer fold of each arm, taken from the fitted router
before any held-out row is scored, and the selection records the frozen PR 38
protocol classifies from it.

What is recorded (F17 G4, G5; F31 R1, R2, R5; protocol D-C11, M6):

GLASS (``glass_fold_evidence``)
    ``selection_reports()`` (Pass 1, Pass 2, every inner Pass 1), the union
    counts of every selection recounted by matching the selected rules on the
    rows each was selected on, the deployed Pass 1 rules (segment, class,
    precision; F17 G6), the remainder (rows, positives, prevalence, row
    identities), the M6 inner-OOF Pass 1 union counts, the nested-ranker
    provenance, and isolation checks.
RF Router (``rf_fold_evidence``)
    ``t1``, ``t2`` and their bands recounted from the inner-OOF scores, every
    ``t1_j`` with its recounted band (``nested_t1``), the R1 band record, the R2
    boundary strips, the R5 eligibility provenance, the remainder, and
    isolation checks.

The evidence is plain JSON (``to_json_safe``). The selection records are a
pure function of it and the frozen contract (``selection_records``), so a
fold can be re-classified after a reload exactly as it was at fit time.

R2 population (owner decision pending; see the Step 2A notes): strip A is
``t1_j <= p1_oof < t1`` and strip B ``t1 <= p1_oof < t1_j``, over the fold
router's own training rows (each row's ``t1_j`` is its inner fold's). The
flagged band is that router's inner-OOF Pass 2 band, ``p2_oof > t2`` on its
training remainder (F31 R2). Held-out rows have no ``p1_oof`` and no ``t1_j``,
so an outer-OOF band cannot contain strip rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

import shared.stage2.evaluation_protocol as ep
from blackbox_pipeline.models.rf_router.config import ELIGIBILITY_NESTED_T1
from glass_pipeline.glass_router.contract import (
    _REPRODUCED_KEYS, glass_fit_outcome, glass_record_reproduction,
)
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import SelectorInfeasibleError
from glass_pipeline.glass_router.pipeline import GlassRouterPipeline
from shared.stage2.contract import ContractViolation, SharedOperatingContractV2
from shared.stage2.obligations import (
    OUTCOME_FEASIBLE_OPTIMAL, OUTCOME_FEASIBLE_UNPROVEN, OUTCOME_INFEASIBLE, OUTCOME_INSUFFICIENT,
    OUTCOME_NO_SOLUTION, PassCounts, check_pass1_fit, check_pass2_fit, classify_fit_outcome,
)
from shared.stage_io import label_fingerprint

__all__ = ["ARM_GLASS", "ARM_RF", "EVIDENCE_FORMAT", "S1_SELECTIONS", "M3_KEY", "to_json_safe",
           "unproven_selections", "m3_record",
           "glass_fold_evidence", "pass1_rule_record", "rf_fold_evidence", "stop_evidence",
           "selection_records", "classify_fold", "fold_status", "s1_fold_status"]

#: Arm names as the frozen protocol spells them.
ARM_GLASS, ARM_RF = "glass", "blackbox"
EVIDENCE_FORMAT = "pr38.stage2.fold_evidence/1"

_STOP_STATUS = {OUTCOME_INFEASIBLE: ep.FIT_INFEASIBLE, OUTCOME_NO_SOLUTION: ep.FIT_NO_SOLUTION,
                OUTCOME_INSUFFICIENT: ep.FIT_INSUFFICIENT}


# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------
def to_json_safe(obj: Any) -> Any:
    """Recursively convert to JSON types, keeping arrays and indexes as lists."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    if isinstance(obj, Mapping):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Index, pd.Series, np.ndarray, list, tuple, set, frozenset)):
        seq = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else list(obj)
        return [to_json_safe(v) for v in seq]
    return str(obj)


def _counts(rows: int, positives: int, routed: np.ndarray, y: np.ndarray) -> dict:
    routed = np.asarray(routed, dtype=bool)
    return {"population_rows": int(rows), "population_positives": int(positives),
            "routed_rows": int(routed.sum()), "routed_positives": int(y[routed].sum())}


def _share(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def _index_list(index) -> list:
    return to_json_safe(pd.Index(index))


# ----------------------------------------------------------------------
# GLASS
# ----------------------------------------------------------------------
#: Inner-fold feature pathways (G13): historical, and the correction
#: (``stage2_crossfit.inner_features``).
PATHWAY_OUTER, PATHWAY_FOLD = "outer_engineer", "fold_label_excluded"


def _pathway(router) -> dict:
    """The router's inner-fold feature pathway (G13); historical fits record none."""
    return to_json_safe(getattr(router, "inner_feature_pathway_", None)
                        or {"pathway": PATHWAY_OUTER, "record": None})


def _plan_fp(plan) -> str:
    return hashlib.sha256(np.asarray(plan, dtype=np.int64).tobytes()).hexdigest()[:16]


def _bins_match(pathway, inner_features, X_fit, y, plan, who) -> bool:
    """The provider's inner bins for ``plan`` are the ones the fit recorded (fingerprints)."""
    if inner_features is None:
        raise ValueError(f"{who}: a corrected-pathway fit needs its inner_features provider")
    from .inner_features import frame_fingerprint
    frames = inner_features(X_fit, y, plan)
    return bool(pathway["plan_fingerprint"] == _plan_fp(plan)
                and [f["bins_fingerprint"] for f in pathway["inner_folds"]]
                == [frame_fingerprint(frames[j]) for j in sorted(frames)])


def pass1_rule_record(rules) -> List[Dict[str, Any]]:
    """
    The deployed Pass 1 rules in deployment order, as G6 compares them across
    the S1 pair (F17 section 6): each rule's segment (its conditions, sorted),
    predicted class and precision, with its ``rule_id`` for reference.
    """
    return to_json_safe([{"rule_id": r.rule_id,
                          "segment": [[str(f), v] for f, v in
                                      sorted(r.segment, key=lambda c: (str(c[0]), str(c[1])))],
                          "predicted_class": r.predicted_class, "precision": r.precision}
                         for r in rules])


def glass_fold_evidence(glass, X_fit: pd.DataFrame, y_fit, *, fold, heldout_index=None,
                        inner_features=None) -> dict:
    """
    Evidence of one fitted GLASS Router (``oof_remainder`` or ``full``) on its
    fitting rows ``X_fit`` (engineered) and labels ``y_fit``. ``fold`` is the
    outer fold (or ``"full_train"``); ``heldout_index`` the rows it must not
    have seen (none for the full-train fit). A fit on the corrected feature
    pathway (G13) needs its provider, ``inner_features``: each inner fold's
    Pass 1 union is recounted on the bins that fold was selected on.
    """
    y = np.asarray(y_fit).astype(int)
    n, pos = len(y), int(y.sum())
    rep = glass.fit_report_ or {}
    population = rep.get("pass2_population", getattr(glass, "pass2_population", "full"))
    match = GlassRouterPipeline._rules_mask
    counts = {"pass1": _counts(n, pos, match(glass.pass1_rules, X_fit), y)}
    ev: Dict[str, Any] = {
        "format": EVIDENCE_FORMAT, "arm": ARM_GLASS, "fold": fold,
        "pass2_population": population, "n_train": n, "train_positives": pos,
        "train_label_fingerprint": label_fingerprint(X_fit.index, y),
        "selection_reports": to_json_safe(glass.selection_reports()),
        "rule_ids": {"pass1": [r.rule_id for r in glass.pass1_rules],
                     "pass2": [r.rule_id for r in glass.pass2_rules]},
        "pass1_rules": pass1_rule_record(glass.pass1_rules),
    }
    isolation: Dict[str, Any] = {}
    held = pd.Index([]) if heldout_index is None else pd.Index(heldout_index)
    isolation["held_out_rows_in_fit"] = int(X_fit.index.isin(held).sum())
    ev["inner_feature_pathway"] = pathway = _pathway(glass)
    if population == "oof_remainder":
        rem = np.asarray(rep["remainder_mask"], dtype=bool)
        inner = np.asarray(glass.inner_fold_id_)
        inner_rules = glass.inner_pass1_rules_
        inner_bins = {j: X_fit for j in range(len(inner_rules))}
        if pathway["pathway"] != PATHWAY_OUTER:
            isolation["inner_features_match_the_fit"] = _bins_match(
                pathway, inner_features, X_fit, y, inner, "glass_fold_evidence")
            inner_bins = inner_features(X_fit, y, inner)
        for j, rules_j in enumerate(inner_rules):
            tr_j = inner != j
            counts[f"inner_pass1[{j}]"] = _counts(int(tr_j.sum()), int(y[tr_j].sum()),
                                                  match(rules_j, inner_bins[j][tr_j]), y[tr_j])
        counts["pass2"] = _counts(int(rem.sum()), int(y[rem].sum()),
                                  match(glass.pass2_rules, X_fit[rem]), y[rem])
        n_rem, rem_pos = int(rem.sum()), int(y[rem].sum())
        ev["remainder"] = {"rows": n_rem, "positives": rem_pos,
                           "prevalence": _share(rem_pos, n_rem),
                           "index": _index_list(X_fit.index[rem])}
        # M6: the inner-OOF Pass 1 union (the rows the remainder excludes); diagnostic only.
        routed = ~rem
        ev["m6_inner_oof_pass1"] = {**_counts(n, pos, routed, y),
                                    "npv": _share(int(routed.sum()) - int(y[routed].sum()),
                                                  int(routed.sum())),
                                    "leakage_share": _share(int(y[routed].sum()), pos)}
        ev["inner_fold_rows"] = [int((inner == j).sum()) for j in range(len(inner_rules))]
        prov = list(rep.get("inner_ranker_provenance") or [])
        ev["inner_ranker_provenance"] = to_json_safe(prov)
        isolation["remainder_held_out_rows"] = int(X_fit.index[rem].isin(held).sum())
        ranker_rows_ok = []
        for p in prov:
            tr_j = inner != int(p["fold"])
            ranker_rows_ok.append(
                int(p["n_train"]) == int(tr_j.sum())
                and p["train_label_fingerprint"] == label_fingerprint(X_fit.index[tr_j], y[tr_j]))
        isolation["inner_rankers_fitted_on_their_inner_train_rows"] = ranker_rows_ok
    else:
        counts["pass2"] = _counts(n, pos, match(glass.pass2_rules, X_fit), y)
    ev["counts"] = counts
    ev["isolation"] = isolation
    return ev


# ----------------------------------------------------------------------
# RF Router
# ----------------------------------------------------------------------
def rf_fold_evidence(router, X_fit: pd.DataFrame, y_fit, *, fold, heldout_index=None,
                     inner_features=None) -> dict:
    """
    Evidence of one fitted two-pass ``RFRouter`` on its fitting rows. A fit on
    the corrected feature pathway (G13) needs its provider, ``inner_features``,
    to check the inner bins against the fit's record.
    """
    cfg = router.config
    y = np.asarray(y_fit).astype(int)
    n, pos = len(y), int(y.sum())
    if not np.array_equal(y, np.asarray(router._y_train_).astype(int)):
        raise ValueError("rf_fold_evidence: y_fit is not the labels the router was fitted on")
    t1, t2 = float(router.thresholds_.t1), float(router.thresholds_.t2)
    p1 = np.asarray(router.oof_pass1_proba_, dtype=float)
    p2 = np.asarray(router.oof_pass2_proba_, dtype=float)
    rem = np.asarray(router.remainder_mask_, dtype=bool)
    inner = np.asarray(router.oof_pass1_fold_id_)
    routed1 = p1 < t1
    band2 = rem & (np.where(np.isnan(p2), -np.inf, p2) > t2)
    n_rem, rem_pos = int(rem.sum()), int(y[rem].sum())
    counts = {"pass1": _counts(n, pos, routed1, y), "pass2": _counts(n_rem, rem_pos, band2, y)}
    solver = {"pass1": router.thresholds_.pass1, "pass2": router.thresholds_.pass2}
    for key, res in solver.items():                  # independent recount vs the solver
        c = counts[key]
        if (c["routed_rows"], c["routed_positives"]) != (int(res.band_size),
                                                         int(res.positives_in_band)):
            raise AssertionError(f"RF {key}: recounted band {c} disagrees with the solver "
                                 f"(band {res.band_size}, positives {res.positives_in_band})")
    remainder_report = dict(router.fit_report_.remainder)
    eligibility = getattr(cfg, "eligibility", None)
    ev: Dict[str, Any] = {
        "format": EVIDENCE_FORMAT, "arm": ARM_RF, "fold": fold,
        "n_train": n, "train_positives": pos,
        "train_label_fingerprint": label_fingerprint(X_fit.index, y),
        "thresholds": {"t1": t1, "t2": t2},
        "counts": counts,
        "remainder": {"rows": n_rem, "positives": rem_pos, "prevalence": _share(rem_pos, n_rem),
                      "index": _index_list(X_fit.index[rem])},
        # R5: the rule, as the config and the fit report record it.
        "eligibility": {"rule": eligibility, "config_eligibility_folds": cfg.eligibility_folds,
                        "source": remainder_report.get("source"),
                        "report_rule": remainder_report.get("eligibility")},
    }
    # R1: both bands and their supports (inner-OOF; F31 Disposition A, not probabilities).
    bands = router.bands_.to_dict()
    floors = {"pass1": cfg.constraints.min_band_support_pass1,
              "pass2": cfg.constraints.min_band_support_pass2}
    ev["r1_bands"] = {
        "pass1_confidence": float(bands["pass1_confidence"]),
        "pass2_confidence": float(bands["pass2_confidence"]),
        "pass1_support": int(bands["pass1_support"]), "pass2_support": int(bands["pass2_support"]),
        "pass1_support_floor": int(floors["pass1"]), "pass2_support_floor": int(floors["pass2"]),
        "supports_at_or_above_floors": bool(int(bands["pass1_support"]) >= floors["pass1"]
                                            and int(bands["pass2_support"]) >= floors["pass2"]),
    }
    isolation: Dict[str, Any] = {}
    held = pd.Index([]) if heldout_index is None else pd.Index(heldout_index)
    isolation["held_out_rows_in_fit"] = int(X_fit.index.isin(held).sum())
    isolation["remainder_held_out_rows"] = int(X_fit.index[rem].isin(held).sum())
    ev["inner_feature_pathway"] = pathway = _pathway(router)
    if pathway["pathway"] != PATHWAY_OUTER:
        isolation["inner_features_match_the_fit"] = _bins_match(
            pathway, inner_features, X_fit, y, router.oof_pass1_fold_id_, "rf_fold_evidence")
    if eligibility == ELIGIBILITY_NESTED_T1:
        ids = [int(j) for j in remainder_report["eligibility_fold_ids"]]
        fold_t1 = [float(t) for t in remainder_report["eligibility_fold_t1"]]
        fold_counts = list(remainder_report["eligibility_fold_counts"])
        t1_row = np.full(n, np.nan)
        for j, t in zip(ids, fold_t1):
            t1_row[inner == j] = t
        if np.isnan(t1_row).any() or not np.array_equal(rem, p1 >= t1_row):
            raise AssertionError("RF nested_t1: the remainder is not p1_oof >= t1_j per inner fold")
        ev["eligibility"].update(fold_ids=ids, fold_t1=fold_t1, fold_counts=fold_counts)
        # G4: each t1_j was solved on exactly the rows outside inner fold j.
        isolation["eligibility_solves_exclude_their_fold"] = [
            int(c["population_rows"]) == int((inner != j).sum())
            and int(c["population_positives"]) == int(y[inner != j].sum())
            for j, c in zip(ids, fold_counts)]
        counts.update({f"eligibility_t1[{j}]": {
            "population_rows": c["population_rows"], "population_positives": c["population_positives"],
            "routed_rows": c["band_rows"], "routed_positives": c["band_positives"]}
            for j, c in zip(ids, fold_counts)})
        ev["r2_strips"] = _strips(p1, t1, t1_row, inner, ids, band2, y)
    return ev | {"isolation": isolation}


# ----------------------------------------------------------------------
# PR 38 versioned records (Step 2B), loaded against pinned fingerprints
# ----------------------------------------------------------------------
RECORDS_DIR = Path(__file__).with_name("records")
#: The G13 correction's specification (``stage2_crossfit.inner_features``).
FEATURE_PATHWAY_FILE, FEATURE_PATHWAY_FINGERPRINT = ("inner_feature_isolation_v1.json",
                                                     "0d4953ef34d1d3dc")
#: The proposed F31 R2 wording clarification; active only once approved in the authorization.
F31_R2_CLARIFICATION_FILE, F31_R2_CLARIFICATION_FINGERPRINT = ("f31_r2_clarification_v1.json",
                                                               "d55e285570ce47d4")


def record_fingerprint(rec: Mapping[str, Any]) -> str:
    """SHA-256 (16 hex) of the canonical JSON, as the frozen PR 37 records use."""
    blob = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_versioned_record(name: str, fingerprint: str) -> Dict[str, Any]:
    """A record from ``records/``, refused unless its fingerprint is the pinned one."""
    rec = json.loads((RECORDS_DIR / name).read_text(encoding="utf-8"))
    got = record_fingerprint(rec)
    if got != fingerprint:
        raise ValueError(f"{name}: fingerprint {got} != pinned {fingerprint}; a versioned "
                         "record is changed only by a new version")
    return rec


def record_identity(rec: Mapping[str, Any]) -> List[str]:
    return [rec["record_id"], rec["record_version"], record_fingerprint(rec)]


def f31_r2_clarification() -> Dict[str, Any]:
    return load_versioned_record(F31_R2_CLARIFICATION_FILE, F31_R2_CLARIFICATION_FINGERPRINT)


#: F31 R2 as implemented. PROVISIONAL until the owner approves the clarification
#: record (``f31_r2_clarification``): the frozen ``f31.measure`` says the outer-OOF
#: band, which has no ``p1_oof`` or ``t1_j`` for held-out rows.
R2_BASIS = ("provisional: the fold router's inner-OOF Pass 2 band; the frozen protocol's "
            "outer-OOF wording is unresolved until the owner approves "
            "pr38-stage2-evaluation-f31-r2-clarification 1.0.0")


def _strips(p1, t1, t1_row, inner, ids, band, y) -> dict:
    """F31 R2 for one fold router: strips A and B, and strip A inside the flagged band."""
    a = (p1 >= t1_row) & (p1 < t1)
    b = (p1 >= t1) & (p1 < t1_row)

    def block(mask):
        rows, positives = int(mask.sum()), int(y[mask].sum())
        return {"rows": rows, "positives": positives, "prevalence": _share(positives, rows)}

    band_rows = int(band.sum())
    a_in_band = a & band
    out = {"definition": "strip A: t1_j <= p1_oof < t1; strip B: t1 <= p1_oof < t1_j; band: "
                         "the fold router's inner-OOF Pass 2 band p2_oof > t2 on its training "
                         "remainder (F31 R2)",
           "interpretation": R2_BASIS,
           "strip_a": block(a), "strip_b": block(b),
           "strip_b_in_band": int((b & band).sum()),
           "strip_a_in_band": {"rows": int(a_in_band.sum()),
                               "positives": int(y[a_in_band].sum())},
           "band_rows": band_rows,
           "strip_a_share_of_band": _share(int(a_in_band.sum()), band_rows),
           "per_inner_fold": [{"inner_fold": j, "strip_a": block(a & (inner == j)),
                               "strip_b": block(b & (inner == j))} for j in ids]}
    if out["strip_b_in_band"]:
        raise AssertionError("strip B rows are not in the remainder, so none can be in the band")
    return out


# ----------------------------------------------------------------------
# Stops
# ----------------------------------------------------------------------
def stop_evidence(arm: str, fold, exc: BaseException, partial_reports=None) -> dict:
    """What a fit that raised leaves on the record (G8): its outcome and why."""
    msg = str(exc)
    if isinstance(exc, SelectorInfeasibleError):
        outcome = OUTCOME_INFEASIBLE if exc.proven else OUTCOME_NO_SOLUTION
    elif isinstance(exc, ValueError) and ("No feasible" in msg):
        outcome = OUTCOME_INFEASIBLE              # RF: the exhaustive threshold scan found none
    elif isinstance(exc, ValueError) and any(t in msg for t in (
            "remainder_min_size", "single-class", "single class", "Remainder is single",
            "band support too small", "below remainder_min_size", "leaves only")):
        outcome = OUTCOME_INSUFFICIENT
    else:
        outcome = None
    return {"format": EVIDENCE_FORMAT, "arm": arm, "fold": fold,
            "stop": {"outcome": outcome, "fit_status": _STOP_STATUS.get(outcome, ep.FIT_TIME_INVALID),
                     "exception": type(exc).__name__, "reason": msg},
            "selection_reports": to_json_safe(partial_reports)}


# ----------------------------------------------------------------------
# Selection records and fold status
# ----------------------------------------------------------------------
def _check(scope: str, counts: Mapping[str, int], contract: SharedOperatingContractV2):
    pc = PassCounts(**{k: int(counts[k]) for k in ("population_rows", "population_positives",
                                                   "routed_rows", "routed_positives")})
    return (check_pass1_fit if scope == "pass1" else check_pass2_fit)(pc, contract)


def _scope(key: str) -> str:
    return "pass2" if key == "pass2" else "pass1"


#: M3 (Commit 4 contract record; protocol section 10): an unproven GLASS ILP
#: incumbent is documented by repeating the identical fit once. The rerun's
#: selection fields sit beside the untouched original evidence under ``M3_KEY``;
#: the frozen ``glass_record_reproduction`` compares them when the fold is classified.
M3_KEY = "m3_reproduction"
M3_FORMAT = "pr38.stage2.m3_reproduction/1"
M3_FIELDS = tuple(_REPRODUCED_KEYS) + ("lexicographic_guarantee",)


def _glass_slots(reports) -> Dict[str, Any]:
    reports = reports or {}
    slots = {"pass1": reports.get("pass1"), "pass2": reports.get("pass2")}
    slots.update({f"inner_pass1[{j}]": r for j, r in enumerate(reports.get("inner_pass1") or [])})
    return slots


def unproven_selections(ev: Mapping[str, Any]) -> List[str]:
    """The GLASS selections that ended as an ILP incumbent without a lexicographic proof."""
    if ev.get("arm") != ARM_GLASS or "stop" in ev:
        return []
    return [k for k, r in _glass_slots(ev.get("selection_reports")).items()
            if isinstance(r, Mapping) and r.get("source") == "ilp"
            and r.get("lexicographic_guarantee") != "proven"]


def m3_record(keys: List[str], rerun_reports=None, *, error: Optional[str] = None) -> dict:
    """
    The M3 record of ``keys``: each one's ``M3_FIELDS`` from the repeated fit's
    selection reports, or ``None`` if the rerun raised (``error``) or made no
    such selection. It never changes the original reports.
    """
    slots = _glass_slots(rerun_reports) if rerun_reports is not None else {}
    return to_json_safe({
        "format": M3_FORMAT, "error": error,
        "method": "the identical fit repeated once (same rows, configuration and seed)",
        "selections": {k: {"rerun": ({f: slots[k].get(f) for f in M3_FIELDS}
                                     if isinstance(slots.get(k), Mapping) else None)}
                       for k in keys}})


def selection_records(ev: Mapping[str, Any],
                      contract: SharedOperatingContractV2) -> List[ep.SelectionRecord]:
    """
    The fold's selection records, rebuilt from its evidence. A feasible claim
    the independent recount contradicts cannot become a ``FitOutcome``; it is
    recorded as an unsupported record, so the fold is ``fit_time_invalid``.
    """
    if "stop" in ev:
        return []
    out = []
    if ev["arm"] == ARM_GLASS:
        slots = _glass_slots(ev["selection_reports"])
        m3 = (ev.get(M3_KEY) or {}).get("selections") or {}
        for key, rep in slots.items():
            if rep is None:
                continue                            # missing: the completeness check names it
            scope = _scope(key)
            feasible = rep.get("source") in ("ilp", "greedy_fallback")
            try:
                report = _check(scope, ev["counts"][key], contract) if feasible else None
                outcome = glass_fit_outcome(scope, rep, report)
                rerun = (m3.get(key) or {}).get("rerun")
                if outcome.outcome == OUTCOME_FEASIBLE_UNPROVEN and isinstance(rerun, Mapping):
                    outcome = glass_record_reproduction(outcome, to_json_safe(rep), rerun)
                out.append(ep.SelectionRecord(key, outcome))
            except (ContractViolation, KeyError) as exc:
                out.append(ep.SelectionRecord(key, {"invalid": f"{type(exc).__name__}: {exc}"}))
        return out
    for key, c in ev["counts"].items():               # RF: every threshold is an exhaustive scan
        scope = _scope(key)
        try:
            out.append(ep.SelectionRecord(key, classify_fit_outcome(
                OUTCOME_FEASIBLE_OPTIMAL, scope=scope,
                reason="exhaustive threshold scan: the largest feasible band",
                report=_check(scope, c, contract), solver={"method": "exhaustive_scan"})))
        except ContractViolation as exc:
            out.append(ep.SelectionRecord(key, {"invalid": f"ContractViolation: {exc}"}))
    return out


def classify_fold(ev: Mapping[str, Any], contract: SharedOperatingContractV2,
                  protocol, reviews=None):
    """``ep.classify_fold_fit`` on the evidence; ``None`` for a fit that stopped."""
    if "stop" in ev:
        return None
    reports = ev.get("selection_reports") if ev["arm"] == ARM_GLASS else None
    return ep.classify_fold_fit(selection_records(ev, contract), arm=ev["arm"], protocol=protocol,
                                reviews=reviews, selection_reports=reports)


def fold_status(ev: Mapping[str, Any], contract, protocol, reviews=None) -> dict:
    """The fold's fit status and why, JSON-safe, for the record."""
    if "stop" in ev:
        return {"status": ev["stop"]["fit_status"], "accepted": False, "status_only": True,
                "problems": [ev["stop"]["reason"]]}
    fit = classify_fold(ev, contract, protocol, reviews)
    return {"status": fit.status, "accepted": fit.status in ep.ACCEPTED_FIT_STATUSES,
            "status_only": False, "by_key": dict(fit.by_key),
            "problems": list(fit.completeness.problems())}


#: The selections an S1 fit (GLASS ``full``; F17 section 6) makes: no inner Pass 1.
S1_SELECTIONS = ("pass1", "pass2")


def s1_fold_status(ev: Mapping[str, Any], contract: SharedOperatingContractV2) -> dict:
    """
    The fit status of one S1 fold or full-train fit, JSON-safe. The frozen
    ``required_selection_keys`` describe the F17 primary, whose inner Pass 1
    selections S1 does not make by definition (``pass2_population="full"``),
    so the complete set here is exactly ``S1_SELECTIONS`` with no inner slot.
    Each selection is classified by the frozen per-selection rules: the
    independent recount (``selection_records``), its report's classification
    (``ep._report_problem``) and its status (``ep._outcome_status``). No review
    is read, so a greedy fallback is never accepted.
    """
    if "stop" in ev:
        return {"status": ev["stop"]["fit_status"], "accepted": False, "status_only": True,
                "problems": [ev["stop"]["reason"]]}
    problems: List[str] = []
    if ev.get("arm") != ARM_GLASS or ev.get("pass2_population") != "full":
        problems.append(f"not an S1 GLASS fit (arm {ev.get('arm')!r}, pass2_population "
                        f"{ev.get('pass2_population')!r})")
    try:
        slots = ep._report_slots(ev.get("selection_reports"))
    except ep.ProtocolError as exc:
        slots = {}
        problems.append(str(exc))
    entries = {k: r for k, r in slots.items() if r is not None}
    problems += [f"missing {k}" for k in S1_SELECTIONS if k not in entries]
    problems += [f"unexpected report {k}" for k in slots if k not in S1_SELECTIONS]
    records = selection_records(ev, contract) if not problems else []
    keys = [r.key for r in records]
    problems += [f"duplicated {k}" for k in sorted(set(keys)) if keys.count(k) > 1]
    by_key: Dict[str, str] = {}
    for r in records:
        problem = (f"{r.key}: {r.outcome}" if not isinstance(r.outcome, ep.FitOutcome)
                   else ep._report_problem(r.key, r.outcome, entries[r.key]))
        if problem:
            problems.append(problem)
        else:
            by_key[r.key] = ep._outcome_status(r.outcome, None)
    status = (ep.FIT_TIME_INVALID if problems or set(by_key) != set(S1_SELECTIONS)
              else min(by_key.values(), key=ep.FIT_STATUSES.index))
    return {"status": status, "accepted": status in ep.ACCEPTED_FIT_STATUSES,
            "status_only": False, "by_key": by_key, "problems": problems}
