"""
shared.stage2.diagnostics.overlap
=================================
Pass overlap and redundancy diagnosis (PR 37, Commit 2, Step 2D).

Two questions, kept apart:

  Q1 functional behaviour - do GLASS and RF route reliably differently relative
     to the upstream flags and to each other? Answered descriptively from the
     validated three-state trace (contingency tables, GLASS x RF cross-tab,
     cross-router Pass 2 overlap).
  Q2 incremental information - among rows with comparable upstream SCORES, do
     router states separate different positive rates? Answered only
     descriptively here (score strata, observed vs expected positives, score-
     boundary positions, arithmetic chance baselines); a controlled test needs
     Stage 4 ablations (PR 38).

    router_reference_contingency    router state x reference flag x label counts
    router_reference_agreement      agreement / disagreement rates, explicit denominators
    router_crosstab / crosstab_summary   GLASS state x RF state cells and agreement
    pass2_cross_router_overlap      Pass 2 positives both routers / one router recover,
                                    overall and unique relative to LR / EBM / union
    signal_availability             which pre-routing signals were persisted (and
                                    which overlap analyses are therefore unavailable)
    score_strata                    label-free within-fold score strata
    score_conditional_states        router state x score stratum label composition
    within_stratum_contrasts        positive-rate differences between states in a stratum
    state_standardization           observed vs expected positives per router state
    key_groups / group_observed_expected / boundary_diagnostics
                                    the disagreement populations, their expectations
                                    and their positions relative to the thresholds
    unique_positive_chance_baselines, model_disagreement_baseline
                                    arithmetic baselines (not statistical tests)
    overlap_notes                   the disclosures

Final states vs signals
-----------------------
Final Pass 1 and Pass 2 never overlap: Pass 1 claims first and Pass 2 only
claims what is left (Step 2A). Zero final overlap is a routing invariant, not
evidence that the underlying signals are independent. Candidate-rule matches,
per-fold rule sets and RF Pass 2 outer scores were never persisted, so a signal-
level Pass 1 / Pass 2 overlap is classified unavailable (``signal_availability``)
and never inferred from final states.

Score strata (fixed before any result was inspected)
----------------------------------------------------
Within each router outer fold, label-free quantile bins of: the LR calibrated
OOF probability (``SCORE_BINS``), the EBM raw OOF score (``SCORE_BINS``; raw
scales are never pooled across folds), and a coarse joint LR x EBM grid
(``JOINT_BINS`` x ``JOINT_BINS``). Interior quantile edges that coincide are
merged, so tied scores always share a bin and a fold may have fewer bins
(reported). Pooled ("all") rows sum bin k of every fold. Cells with fewer than
``MIN_SUPPORT`` rows are flagged low-support and excluded from contrasts.

Expectations (descriptive, not null distributions)
--------------------------------------------------
    expected_lr_calibrated   sum of LR calibrated OOF probabilities over the group
    expected_ebm_calibrated  sum of EBM calibrated OOF probabilities (per-fold
                             calibration fitted on each fold's inner OOF rows)
    expected_stratified_<stratifier>
                             sum over the group's rows of their (fold, stratum)
                             cell's observed positive rate (internal standardisation
                             on the same labels: if router state carried no
                             information beyond the stratum, observed = expected for
                             every state)
    o_minus_e_<expectation>  observed positives minus that expectation
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import DiagnosticResult, Ratio
from .operating_points import UpstreamFlags
from .schema import (
    ABSTAIN,
    NOT_RECORDED,
    PASS_1_NOT_SUBSCRIBE,
    PASS_2_SUBSCRIBE,
    ROUTER_STATES,
    RouterDiagnosticError,
    UnsupportedDiagnostic,
    UpstreamOOFView,
    _frozen,
)
from .suppression import _frame, _require_flags, _row
from .trace import (
    ARM_LABELS,
    CODE_LEVEL_ONLY,
    GLASS_ARM,
    NOT_VERIFIABLE,
    RF_ARM,
    TRACE_ARMS,
    VERIFIED_FROM_ARTIFACTS,
    RouterTrace,
    _groups,
    _require_trace,
    _rf_pass1_evidence,
)

__all__ = [
    "SCORE_BINS", "JOINT_BINS", "MIN_SUPPORT", "BOUNDARY_BAND", "STRATIFIERS", "OVERLAP_LABELS",
    "ScoreStrata", "router_reference_contingency", "router_reference_agreement",
    "router_crosstab", "crosstab_summary", "pass2_cross_router_overlap", "signal_availability",
    "score_strata", "score_conditional_states", "within_stratum_contrasts",
    "state_standardization", "key_groups", "group_observed_expected", "boundary_diagnostics",
    "unique_positive_chance_baselines", "model_disagreement_baseline", "overlap_notes",
]

#: Fixed before any Step 2D result was inspected (label-free; never tuned).
SCORE_BINS = 10
JOINT_BINS = 3
MIN_SUPPORT = 30
BOUNDARY_BAND = 0.05
STRATIFIERS = ("lr", "ebm_raw", "lr_x_ebm")

P1, P2, AB = PASS_1_NOT_SUBSCRIBE, PASS_2_SUBSCRIBE, ABSTAIN
_SHORT = MappingProxyType({P1: "pass1", P2: "pass2", AB: "abstain"})

OVERLAP_LABELS: Mapping[str, str] = MappingProxyType({
    "coverage": "routed rows (Pass 1 or Pass 2) / rows",
    "pass1_upstream_negative_agreement": "Pass 1 rows the reference does not flag / Pass 1 rows",
    "pass1_upstream_flag_rate": "Pass 1 rows the reference flags / Pass 1 rows (conflict)",
    "pass2_upstream_redundancy": "Pass 2 rows the reference flags / Pass 2 rows (agreement)",
    "pass2_upstream_disagreement": "Pass 2 rows the reference does not flag / Pass 2 rows",
    "covered_agreement": "routed rows where Pass 2 == reference flag / routed rows (routed only)",
    "abstain_upstream_flag_rate": "abstained rows the reference flags / abstained rows",
    "abstain_upstream_precision": "positives among abstained flagged rows / abstained flagged rows",
    "upstream_flags_in_pass1_share": "reference flags in Pass 1 / reference flags",
    "upstream_flags_in_pass2_share": "reference flags in Pass 2 / reference flags",
    "upstream_flags_in_abstain_share": "reference flags in abstain / reference flags",
    "cross_state_agreement": "rows with the same GLASS and RF state / rows",
    "cross_both_routed_rate": "rows both routers route / rows",
    "cross_routed_agreement": "same routed decision / rows both routers route (routed only)",
    "cross_opposite_rate": "rows one router sends to Pass 1 and the other to Pass 2 / rows",
    "cross_one_abstains_rate": "rows exactly one router abstains on / rows",
    "cross_both_abstain_rate": "rows both routers abstain on / rows",
    "above_threshold_share": "group rows at or above the reference threshold / group rows",
    "threshold_band_share": "group rows within the threshold band (within-fold percentile) / group rows",
    "clear_above_share": "group rows above the threshold and outside the band / group rows",
    "clear_below_share": "group rows below the threshold and outside the band / group rows",
})

_FOLD_RF_NOTE = (
    "Shared component (code-level): crossfit_stage2 defaults to glass_rf_model='fold_rf' and passes "
    "each outer fold's RF to the GLASS fit, whose RuleGenerator uses that forest's "
    "feature_importances_ to prioritise segments; that same fold forest's oof_proba is the RF "
    "Router's Pass 1 score (Commit 1 equivalence). glass_rf_model is not read by the diagnostics "
    "boundary, so the artifacts' own record of it is not checked here.")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _keep(trace: RouterTrace, mask) -> np.ndarray:
    return np.ones(trace.n, bool) if mask is None else mask


def _states(trace: RouterTrace, arm: str):
    v = trace.views[arm]
    return np.asarray(v.pass1), np.asarray(v.pass2), np.asarray(v.abstain)


def _require_refs(trace: RouterTrace, refs: Mapping[str, UpstreamFlags]) -> Dict[str, np.ndarray]:
    if not isinstance(refs, Mapping) or not refs:
        raise RouterDiagnosticError("refs must be a non-empty mapping of UpstreamFlags "
                                    "(primary_reference_flags)")
    return {k: np.asarray(_require_flags(f, trace)) for k, f in refs.items()}


def _long(recs: List[dict], lead: List[str]) -> pd.DataFrame:
    return _frame(recs)[lead + ["metric", "label", "numerator", "denominator", "value", "status"]]


# ======================================================================
# Router x reference
# ======================================================================
def router_reference_contingency(trace: RouterTrace,
                                 refs: Mapping[str, UpstreamFlags]) -> pd.DataFrame:
    """Per arm, reference and router fold: state x reference flag -> n, positives, negatives."""
    _require_trace(trace)
    flags = _require_refs(trace, refs)
    y = np.asarray(trace.y) == 1
    recs = []
    for key, f in flags.items():
        for arm in TRACE_ARMS:
            st = np.asarray(trace.views[arm].state)
            for g, mask in _groups(trace):
                k = _keep(trace, mask)
                for s in ROUTER_STATES:
                    for flagged in (False, True):
                        m = k & (st == s) & (f == flagged)
                        recs.append({"arm": arm, "reference": key, "fold": g, "state": s,
                                     "reference_flag": flagged, "n": int(m.sum()),
                                     "positives": int((m & y).sum()),
                                     "negatives": int((m & ~y).sum())})
    return pd.DataFrame(recs)


def router_reference_agreement(trace: RouterTrace,
                               refs: Mapping[str, UpstreamFlags]) -> pd.DataFrame:
    """
    Per arm, reference and router fold: agreement and disagreement rates with
    explicit denominators (``OVERLAP_LABELS``). Abstention is its own state;
    ``covered_agreement`` is over routed rows only (see ``coverage``).
    """
    _require_trace(trace)
    flags = _require_refs(trace, refs)
    pos_all = np.asarray(trace.y) == 1
    recs = []
    for key, f_all in flags.items():
        for arm in TRACE_ARMS:
            p1a, p2a, aba = _states(trace, arm)
            for g, mask in _groups(trace):
                k = _keep(trace, mask)
                p1, p2, ab, f, pos = p1a[k], p2a[k], aba[k], f_all[k], pos_all[k]
                cov = p1 | p2
                rows = [
                    ("coverage", cov.sum(), len(f)),
                    ("pass1_upstream_negative_agreement", (p1 & ~f).sum(), p1.sum()),
                    ("pass1_upstream_flag_rate", (p1 & f).sum(), p1.sum()),
                    ("pass2_upstream_redundancy", (p2 & f).sum(), p2.sum()),
                    ("pass2_upstream_disagreement", (p2 & ~f).sum(), p2.sum()),
                    ("covered_agreement", (cov & (p2 == f)).sum(), cov.sum()),
                    ("abstain_upstream_flag_rate", (ab & f).sum(), ab.sum()),
                    ("abstain_upstream_precision", (ab & f & pos).sum(), (ab & f).sum()),
                    ("upstream_flags_in_pass1_share", (p1 & f).sum(), f.sum()),
                    ("upstream_flags_in_pass2_share", (p2 & f).sum(), f.sum()),
                    ("upstream_flags_in_abstain_share", (ab & f).sum(), f.sum()),
                ]
                recs += [{"arm": arm, "reference": key, "fold": g,
                          **_row(m, n, d, OVERLAP_LABELS)} for m, n, d in rows]
    return _long(recs, ["arm", "reference", "fold"])


# ======================================================================
# GLASS x RF
# ======================================================================
def router_crosstab(trace: RouterTrace, refs: Mapping[str, UpstreamFlags]) -> pd.DataFrame:
    """
    GLASS state x RF state per router fold ("all" = pooled): n, positives,
    negatives, positive rate, share of the group's rows, and each reference's
    flags / flagged positives in the cell.
    """
    _require_trace(trace)
    flags = _require_refs(trace, refs)
    y = np.asarray(trace.y) == 1
    gs, rs = np.asarray(trace.views[GLASS_ARM].state), np.asarray(trace.views[RF_ARM].state)
    recs = []
    for g, mask in _groups(trace):
        k = _keep(trace, mask)
        for a in ROUTER_STATES:
            for b in ROUTER_STATES:
                m = k & (gs == a) & (rs == b)
                rate = Ratio("set_positive_rate", int((m & y).sum()), int(m.sum()))
                share = Ratio("cross_cell_share", int(m.sum()), int(k.sum()))
                rec = {"fold": g, "glass_state": a, "rf_state": b, "n": int(m.sum()),
                       "positives": int((m & y).sum()), "negatives": int((m & ~y).sum()),
                       "positive_rate": rate.as_float(), "positive_rate_status": rate.status,
                       "share": share.as_float()}
                for key, f in flags.items():
                    rec[f"{key}_flags"] = int((m & f).sum())
                    rec[f"{key}_flagged_positives"] = int((m & f & y).sum())
                recs.append(rec)
    return pd.DataFrame(recs)


def crosstab_summary(trace: RouterTrace) -> pd.DataFrame:
    """GLASS vs RF agreement per router fold; routed-only rates say so in their label."""
    _require_trace(trace)
    gs, rs = np.asarray(trace.views[GLASS_ARM].state), np.asarray(trace.views[RF_ARM].state)
    recs = []
    for g, mask in _groups(trace):
        k = _keep(trace, mask)
        a, b = gs[k], rs[k]
        both_routed = (a != AB) & (b != AB)
        opposite = ((a == P1) & (b == P2)) | ((a == P2) & (b == P1))
        rows = [("cross_state_agreement", (a == b).sum(), len(a)),
                ("cross_both_routed_rate", both_routed.sum(), len(a)),
                ("cross_routed_agreement", (both_routed & (a == b)).sum(), both_routed.sum()),
                ("cross_opposite_rate", opposite.sum(), len(a)),
                ("cross_one_abstains_rate", ((a == AB) ^ (b == AB)).sum(), len(a)),
                ("cross_both_abstain_rate", ((a == AB) & (b == AB)).sum(), len(a))]
        recs += [{"fold": g, **_row(m, n, d, OVERLAP_LABELS)} for m, n, d in rows]
    return _long(recs, ["fold"])


def pass2_cross_router_overlap(trace: RouterTrace,
                               refs: Mapping[str, UpstreamFlags]) -> pd.DataFrame:
    """
    Pass 2 rows both routers / only one router flag, per router fold, for all
    Pass 2 rows ("all_pass2") and for Pass 2 rows unique relative to each
    reference. TP and FP separately; where the other router put each one-router
    positive; Jaccard of the two TP sets; and an arithmetic expectation of the
    shared count if the two sets were unrelated draws from the positives either
    could flag (not a test).
    """
    _require_trace(trace)
    flags = {"all_pass2": np.zeros(trace.n, bool), **_require_refs(trace, refs)}
    y = np.asarray(trace.y) == 1
    g1, g2, ga = _states(trace, GLASS_ARM)
    r1, r2, ra = _states(trace, RF_ARM)
    recs = []
    for key, f in flags.items():
        sg, sr = g2 & ~f, r2 & ~f
        pool = y & ~f & ~g1 & ~r1
        for fold, mask in _groups(trace):
            k = _keep(trace, mask)
            rec = {"reference": key, "fold": fold}
            for lab, lm in (("tp", y), ("fp", ~y)):
                a, b = sg & lm & k, sr & lm & k
                rec.update({f"glass_unique_{lab}": int(a.sum()), f"rf_unique_{lab}": int(b.sum()),
                            f"shared_{lab}": int((a & b).sum()),
                            f"glass_only_{lab}": int((a & ~b).sum()),
                            f"rf_only_{lab}": int((b & ~a).sum())})
            a, b = sg & y & k, sr & y & k
            rec.update({"glass_only_tp_in_rf_pass1": int((a & ~b & r1).sum()),
                        "glass_only_tp_in_rf_abstain": int((a & ~b & ra).sum()),
                        "rf_only_tp_in_glass_pass1": int((b & ~a & g1).sum()),
                        "rf_only_tp_in_glass_abstain": int((b & ~a & ga).sum())})
            jac = Ratio("jaccard_with_router", int((a & b).sum()), int((a | b).sum()))
            rec.update({"tp_jaccard": jac.as_float(), "tp_jaccard_status": jac.status})
            pk = pool & k
            n_pool = int(pk.sum())
            rec.update({"pool_positives": n_pool,
                        "expected_shared_tp_if_unrelated":
                            (float((a & pk).sum()) * float((b & pk).sum()) / n_pool)
                            if n_pool else float("nan")})
            recs.append(rec)
    return pd.DataFrame(recs)


# ======================================================================
# Signal availability
# ======================================================================
def signal_availability(trace: RouterTrace, *, rf_candidate=None) -> pd.DataFrame:
    """
    Which pre-routing signals the historical artifacts support, and therefore
    which Pass 1 / Pass 2 overlap analyses are available. Statuses use the
    Step 2A audit vocabulary. Nothing is reconstructed.
    """
    _require_trace(trace)
    rows = []

    def add(arm, signal, status, basis, count, detail):
        rows.append({"arm": arm, "signal": signal, "status": status, "basis": basis,
                     "rows": count, "detail": detail})

    for arm in TRACE_ARMS:
        v = trace.views[arm]
        prov = {p.name: p for p in v.provenance}
        p1, p2, _ = _states(trace, arm)
        add(arm, "final routing state", VERIFIED_FROM_ARTIFACTS, "artifact", trace.n,
            "Step 2A trace checks T1-T7")
        add(arm, "final Pass 1 & Pass 2 rows", VERIFIED_FROM_ARTIFACTS, "artifact",
            int((p1 & p2).sum()),
            "routing invariant (Pass 1 precedence): zero is NOT evidence of independent signals")
        add(arm, "router confidence / router_transform score", VERIFIED_FROM_ARTIFACTS, "artifact",
            trace.n, "a deterministic function of the final state (and the matched rule's / band's "
            "precision), checked by the extractor; not an independent pre-routing signal")
        if arm == GLASS_ARM:
            for name in ("fold_rule_sets", "fold_summaries"):
                p = prov.get(name)
                add(arm, f"outer-fold {name.replace('_', ' ')}",
                    NOT_VERIFIABLE if p is None or p.status == NOT_RECORDED else
                    VERIFIED_FROM_ARTIFACTS, "none", None,
                    p.detail if p is not None else "not recorded")
            for pas in ("Pass 1", "Pass 2"):
                add(arm, f"{pas} candidate-rule matches per row", NOT_VERIFIABLE, "none", None,
                    "never persisted; cannot be inferred from final states")
            n_rules = v.metadata.get("n_rules")
            add(arm, "full-train selected rule set", NOT_VERIFIABLE, "invalid population", None,
                f"persisted for the full-train fit only (rule counts {n_rules}); applying it to "
                "training rows would be in-sample, not outer OOF - not used")
            add(arm, "fold RF passed to the GLASS fit", CODE_LEVEL_ONLY, "code", None,
                _FOLD_RF_NOTE)
        else:
            ev = _rf_pass1_evidence(trace, rf_candidate)
            add(arm, "Pass 1 outer score (all rows)", ev["status"], "artifact" if ev["report"]
                else "none", trace.n if ev["status"] == VERIFIED_FROM_ARTIFACTS else None,
                ev["why"])
            p = prov.get("pass2_outer_scores")
            add(arm, "Pass 2 outer score", NOT_VERIFIABLE, "none", None,
                p.detail if p is not None else "not recorded")
            add(arm, "per-fold t1 / t2", VERIFIED_FROM_ARTIFACTS if v.fold_thresholds else
                NOT_VERIFIABLE, "artifact", None, "oof_fold_summary (extractor-checked)")
    add("both", "analysis: final-state Pass 1 / Pass 2 overlap", VERIFIED_FROM_ARTIFACTS,
        "artifact", 0, "available; zero by construction (routing invariant)")
    add("both", "analysis: GLASS Pass 1 vs Pass 2 candidate / signal overlap", NOT_VERIFIABLE,
        "none", None, "unavailable: no per-row candidate matches or fold rule sets persisted")
    add("both", "analysis: RF Pass 1 vs Pass 2 signal overlap", NOT_VERIFIABLE, "none", None,
        "unavailable: RF Pass 2 outer scores were not persisted (the Pass 1 score alone cannot "
        "show what Pass 2 would have scored)")
    return pd.DataFrame(rows, columns=["arm", "signal", "status", "basis", "rows", "detail"])


# ======================================================================
# Score strata
# ======================================================================
@dataclass(frozen=True)
class ScoreStrata:
    """
    Label-free within-router-fold score strata and threshold positions. All
    arrays are read-only and aligned with the trace rows. ``bins[stratifier]``:
    stratum id per row (fold-specific; pooled tables sum stratum k of every
    fold). ``n_bins[stratifier][fold]``: occupied strata. ``edges``: interior
    quantile edges actually used per (stratifier, fold).
    """

    lr_probability: np.ndarray
    ebm_raw: np.ndarray
    ebm_probability: Optional[np.ndarray]
    lr_threshold: float
    ebm_threshold_row: np.ndarray
    bins: Mapping[str, np.ndarray]
    n_bins: Mapping[str, Mapping[int, int]]
    edges: Mapping[str, Mapping[int, Tuple[float, ...]]]
    lr_percentile: np.ndarray
    ebm_percentile: np.ndarray
    lr_threshold_percentile: np.ndarray
    ebm_threshold_percentile: np.ndarray
    provenance: Mapping[str, str]
    requested_bins: Mapping[str, int] = MappingProxyType(
        {"lr": SCORE_BINS, "ebm_raw": SCORE_BINS, "lr_x_ebm": JOINT_BINS * JOINT_BINS})


def _quantile_bins(s: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Label-free quantile strata; coinciding edges merge, tied scores share a stratum."""
    if len(s) == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    edges = np.unique(np.quantile(s, np.arange(1, n_bins) / n_bins))
    return np.searchsorted(edges, s, side="right").astype(int), edges


def _percentile(s: np.ndarray) -> np.ndarray:
    return pd.Series(s).rank(method="average", pct=True).to_numpy()


def score_strata(trace: RouterTrace, lr_view: UpstreamOOFView,
                 ebm_view: UpstreamOOFView) -> ScoreStrata:
    """The fixed label-free strata (``SCORE_BINS``, ``JOINT_BINS``) for ``trace``."""
    _require_trace(trace)
    for view, model in ((lr_view, "lr"), (ebm_view, "ebm")):
        if not isinstance(view, UpstreamOOFView) or view.model != model:
            raise RouterDiagnosticError(f"expected the {model.upper()} UpstreamOOFView")
        if not (view.index.equals(trace.index) and np.array_equal(view.y, trace.y)):
            raise RouterDiagnosticError(f"{model}: rows / labels differ from the trace")
    lr_s = lr_view.scores.get("calibrated_probability")
    ebm_r = ebm_view.scores.get("raw_model_score")
    ebm_c = ebm_view.scores.get("calibrated_probability")
    nested = ebm_view.stored_decisions.get("ebm.nested_f2")
    thr = lr_view.thresholds.get("lr.in_stage_f2_cv")
    for name, obj in (("LR calibrated_probability", lr_s), ("EBM raw_model_score", ebm_r),
                      ("EBM ebm.nested_f2 per-row thresholds", nested),
                      ("LR lr.in_stage_f2_cv threshold", thr)):
        if obj is None:
            raise UnsupportedDiagnostic(f"score strata need the {name}")
    n = trace.n
    p = np.asarray(lr_s.values, dtype=float)
    raw = np.asarray(ebm_r.values, dtype=float)
    pc = None if ebm_c is None else np.asarray(ebm_c.values, dtype=float)
    t_row = np.asarray(nested.per_row_threshold, dtype=float)
    for name, a in (("LR probability", p), ("EBM raw score", raw), ("EBM threshold", t_row),
                    *((("EBM probability", pc),) if pc is not None else ())):
        if a.shape != (n,) or not np.isfinite(a).all():
            raise UnsupportedDiagnostic(f"{name}: not a finite score on every trace row")
    fold = np.asarray(trace.fold_id)
    bins = {s: np.zeros(n, dtype=int) for s in STRATIFIERS}
    n_bins: Dict[str, Dict[int, int]] = {s: {} for s in STRATIFIERS}
    edges: Dict[str, Dict[int, Tuple[float, ...]]] = {s: {} for s in STRATIFIERS}
    lr_pct, ebm_pct = np.zeros(n), np.zeros(n)
    lr_tpct, ebm_tpct = np.zeros(n), np.zeros(n)
    for k in range(trace.n_folds):
        m = fold == k
        thr_k = np.unique(t_row[m])
        if len(thr_k) != 1:
            raise UnsupportedDiagnostic(f"EBM nested threshold is not constant in fold {k}")
        b_lr, e_lr = _quantile_bins(p[m], SCORE_BINS)
        b_eb, e_eb = _quantile_bins(raw[m], SCORE_BINS)
        j_lr, e_jl = _quantile_bins(p[m], JOINT_BINS)
        j_eb, e_je = _quantile_bins(raw[m], JOINT_BINS)
        bins["lr"][m], bins["ebm_raw"][m] = b_lr, b_eb
        bins["lr_x_ebm"][m] = j_lr * JOINT_BINS + j_eb
        for s, b, e in (("lr", b_lr, e_lr), ("ebm_raw", b_eb, e_eb),
                        ("lr_x_ebm", j_lr * JOINT_BINS + j_eb, tuple(e_jl) + tuple(e_je))):
            n_bins[s][k] = int(len(np.unique(b)))
            edges[s][k] = tuple(float(x) for x in e)
        lr_pct[m], ebm_pct[m] = _percentile(p[m]), _percentile(raw[m])
        lr_tpct[m] = float((p[m] < thr.value).mean())
        ebm_tpct[m] = float((raw[m] < thr_k[0]).mean())
    prov = {"lr": f"{lr_s.field}: {lr_s.provenance}",
            "ebm_raw": f"{ebm_r.field}: {ebm_r.provenance}",
            "ebm_probability": "not available" if ebm_c is None else
            f"{ebm_c.field}: {ebm_c.provenance}",
            "lr_threshold": f"{thr.field} = {thr.value!r} ({thr.selection})",
            "ebm_threshold": f"{nested.field}, by fold {tuple(nested.by_fold)} (raw space)"}
    return ScoreStrata(
        lr_probability=_frozen(p, float), ebm_raw=_frozen(raw, float),
        ebm_probability=None if pc is None else _frozen(pc, float),
        lr_threshold=float(thr.value), ebm_threshold_row=_frozen(t_row, float),
        bins=MappingProxyType({s: _frozen(b, int) for s, b in bins.items()}),
        n_bins=MappingProxyType({s: MappingProxyType(v) for s, v in n_bins.items()}),
        edges=MappingProxyType({s: MappingProxyType(v) for s, v in edges.items()}),
        lr_percentile=_frozen(lr_pct, float), ebm_percentile=_frozen(ebm_pct, float),
        lr_threshold_percentile=_frozen(lr_tpct, float),
        ebm_threshold_percentile=_frozen(ebm_tpct, float), provenance=MappingProxyType(prov))


def _require_strata(strata, trace: RouterTrace) -> ScoreStrata:
    if not isinstance(strata, ScoreStrata) or len(strata.lr_probability) != trace.n:
        raise RouterDiagnosticError("expected ScoreStrata from score_strata(trace, ...)")
    return strata


def _cell_ids(trace: RouterTrace, strata: ScoreStrata, stratifier: str) -> np.ndarray:
    if stratifier not in STRATIFIERS:
        raise RouterDiagnosticError(f"stratifier must be one of {STRATIFIERS}")
    return np.asarray(trace.fold_id) * 1000 + np.asarray(strata.bins[stratifier])


def _row_cell_rate(y: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Per row: the observed positive rate of its (fold, stratum) cell over all rows."""
    _, inv = np.unique(cells, return_inverse=True)
    return (np.bincount(inv, weights=y.astype(float)) / np.bincount(inv))[inv]


# ======================================================================
# Score-conditional tables
# ======================================================================
def score_conditional_states(trace: RouterTrace, strata: ScoreStrata,
                             refs: Optional[Mapping[str, UpstreamFlags]] = None) -> pd.DataFrame:
    """
    Per arm, stratifier, router fold and stratum: each router state's rows,
    positives, negatives, positive rate, calibrated expected positives
    (``expected_lr_calibrated`` / ``expected_ebm_calibrated``) and
    (optionally) reference flags; ``low_support`` when n < ``MIN_SUPPORT``.
    """
    _require_trace(trace)
    _require_strata(strata, trace)
    flags = _require_refs(trace, refs) if refs else {}
    y = np.asarray(trace.y) == 1
    p = np.asarray(strata.lr_probability)
    pc = None if strata.ebm_probability is None else np.asarray(strata.ebm_probability)
    recs = []
    for sname in STRATIFIERS:
        bins = np.asarray(strata.bins[sname])
        for arm in TRACE_ARMS:
            st = np.asarray(trace.views[arm].state)
            for g, mask in _groups(trace):
                k = _keep(trace, mask)
                for b in np.unique(bins[k]):
                    cell = k & (bins == b)
                    for s in ROUTER_STATES:
                        m = cell & (st == s)
                        rate = Ratio("set_positive_rate", int((m & y).sum()), int(m.sum()))
                        rec = {"arm": arm, "stratifier": sname, "fold": g, "stratum": int(b),
                               "state": s, "stratum_rows": int(cell.sum()),
                               "stratum_positives": int((cell & y).sum()), "n": int(m.sum()),
                               "positives": int((m & y).sum()), "negatives": int((m & ~y).sum()),
                               "positive_rate": rate.as_float(), "positive_rate_status": rate.status,
                               "expected_lr_calibrated": float(p[m].sum()),
                               "expected_ebm_calibrated": (float(pc[m].sum()) if pc is not None
                                                           else np.nan),
                               "low_support": bool(m.sum() < MIN_SUPPORT)}
                        for key, f in flags.items():
                            rec[f"{key}_flags"] = int((m & f).sum())
                        recs.append(rec)
    return pd.DataFrame(recs)


def within_stratum_contrasts(table: pd.DataFrame) -> pd.DataFrame:
    """
    From ``score_conditional_states``: per arm, stratifier, fold and stratum,
    the positive-rate differences (percentage points) Pass 1 - Abstain,
    Pass 2 - Abstain and Pass 2 - Pass 1, only where both states have at least
    ``MIN_SUPPORT`` rows (otherwise NaN, status ``low_support``).
    """
    out = []
    for key, grp in table.groupby(["arm", "stratifier", "fold", "stratum"], sort=False):
        r = grp.set_index("state")
        for a, b in ((P1, AB), (P2, AB), (P2, P1)):
            ok = (not r.loc[a, "low_support"]) and (not r.loc[b, "low_support"])
            diff = 100.0 * (r.loc[a, "positive_rate"] - r.loc[b, "positive_rate"]) if ok else np.nan
            out.append({"arm": key[0], "stratifier": key[1], "fold": key[2], "stratum": key[3],
                        "contrast": f"{_SHORT[a]} - {_SHORT[b]}", "n_a": int(r.loc[a, "n"]),
                        "n_b": int(r.loc[b, "n"]), "difference_pp": diff,
                        "status": "defined" if ok else "low_support"})
    return pd.DataFrame(out)


def _oe_record(y, group, p, pc, cell_rates) -> dict:
    obs = int((group & y).sum())
    rec = {"n": int(group.sum()), "observed": obs,
           "positive_rate": (obs / group.sum()) if group.sum() else np.nan,
           "expected_lr_calibrated": float(p[group].sum()),
           "expected_ebm_calibrated": float(pc[group].sum()) if pc is not None else np.nan}
    for sname, rates in cell_rates.items():
        rec[f"expected_stratified_{sname}"] = float(rates[group].sum())
    for col in [c for c in rec if c.startswith("expected_")]:
        rec[col.replace("expected_", "o_minus_e_", 1)] = obs - rec[col]
    return rec


def state_standardization(trace: RouterTrace, strata: ScoreStrata) -> pd.DataFrame:
    """
    Per arm, router fold and state: observed positives against the calibrated
    expectations (LR, EBM) and the internal stratified expectations for each
    stratifier. Within every (fold, stratum) cell the three states' stratified
    expectations sum to the cell's observed positives.
    """
    _require_trace(trace)
    _require_strata(strata, trace)
    y = np.asarray(trace.y) == 1
    p = np.asarray(strata.lr_probability)
    pc = None if strata.ebm_probability is None else np.asarray(strata.ebm_probability)
    rates = {s: _row_cell_rate(y, _cell_ids(trace, strata, s)) for s in STRATIFIERS}
    recs = []
    for arm in TRACE_ARMS:
        st = np.asarray(trace.views[arm].state)
        for g, mask in _groups(trace):
            k = _keep(trace, mask)
            for s in ROUTER_STATES:
                recs.append({"arm": arm, "fold": g, "state": s,
                             **_oe_record(y, k & (st == s), p, pc, rates)})
    return pd.DataFrame(recs)


# ======================================================================
# Disagreement populations
# ======================================================================
def key_groups(trace: RouterTrace, refs: Mapping[str, UpstreamFlags]
               ) -> Mapping[str, Tuple[np.ndarray, str]]:
    """The named populations Step 2D examines: mask (read-only) and description."""
    _require_trace(trace)
    f = _require_refs(trace, refs)
    for k in ("lr", "ebm", "union"):
        if k not in f:
            raise RouterDiagnosticError(f"key_groups needs the {k!r} reference")
    g1, g2, ga = _states(trace, GLASS_ARM)
    r1, r2, ra = _states(trace, RF_ARM)
    lr, ebm, union = f["lr"], f["ebm"], f["union"]
    spec = (
        ("glass_pass1", g1, "GLASS Pass 1"), ("rf_pass1", r1, "RF Pass 1"),
        ("glass_pass2", g2, "GLASS Pass 2"), ("rf_pass2", r2, "RF Pass 2"),
        ("glass_abstain", ga, "GLASS abstain"), ("rf_abstain", ra, "RF abstain"),
        ("glass_pass1_and_lr_flag", g1 & lr, "GLASS Pass 1 rows LR flags (intercepted)"),
        ("glass_pass1_and_ebm_flag", g1 & ebm, "GLASS Pass 1 rows EBM flags (intercepted)"),
        ("glass_pass1_and_union_flag", g1 & union, "GLASS Pass 1 rows LR or EBM flags"),
        ("rf_pass1_and_union_flag", r1 & union, "RF Pass 1 rows LR or EBM flags"),
        ("glass_pass2_unique_vs_lr", g2 & ~lr, "GLASS Pass 2 rows LR does not flag"),
        ("rf_pass2_unique_vs_lr", r2 & ~lr, "RF Pass 2 rows LR does not flag"),
        ("glass_pass2_unique_vs_union", g2 & ~union, "GLASS Pass 2 rows neither flags"),
        ("rf_pass2_unique_vs_union", r2 & ~union, "RF Pass 2 rows neither flags"),
        ("glass_pass1_and_rf_pass2", g1 & r2, "GLASS Pass 1, RF Pass 2 (opposite)"),
        ("glass_pass2_and_rf_pass1", g2 & r1, "GLASS Pass 2, RF Pass 1 (opposite)"),
        ("glass_pass1_and_rf_abstain", g1 & ra, "GLASS Pass 1, RF abstains"),
        ("glass_abstain_and_rf_pass1", ga & r1, "GLASS abstains, RF Pass 1"),
        ("glass_pass2_and_rf_abstain", g2 & ra, "GLASS Pass 2, RF abstains"),
        ("glass_abstain_and_rf_pass2", ga & r2, "GLASS abstains, RF Pass 2"),
        ("both_abstain", ga & ra, "both routers abstain"),
    )
    return MappingProxyType({name: (_frozen(m, bool), desc) for name, m, desc in spec})


def group_observed_expected(trace: RouterTrace, strata: ScoreStrata,
                            groups: Mapping[str, Tuple[np.ndarray, str]]) -> pd.DataFrame:
    """Per group and router fold: observed positives vs every expectation (module docstring)."""
    _require_trace(trace)
    _require_strata(strata, trace)
    y = np.asarray(trace.y) == 1
    p = np.asarray(strata.lr_probability)
    pc = None if strata.ebm_probability is None else np.asarray(strata.ebm_probability)
    rates = {s: _row_cell_rate(y, _cell_ids(trace, strata, s)) for s in STRATIFIERS}
    recs = []
    for name, (mask_g, desc) in groups.items():
        mg = np.asarray(mask_g, dtype=bool)
        if mg.shape != (trace.n,):
            raise RouterDiagnosticError(f"group {name}: mask does not cover the trace rows")
        for g, mask in _groups(trace):
            recs.append({"group": name, "description": desc, "fold": g,
                         **_oe_record(y, _keep(trace, mask) & mg, p, pc, rates)})
    return pd.DataFrame(recs)


def boundary_diagnostics(trace: RouterTrace, strata: ScoreStrata,
                         groups: Mapping[str, Tuple[np.ndarray, str]]) -> pd.DataFrame:
    """
    Where each group sits relative to the reference thresholds: per group, score
    (LR calibrated probability vs the stored lr.in_stage_f2_cv threshold; EBM raw
    score vs its fold's nested threshold) and router fold - shares at / above the
    threshold, within the band (|within-fold percentile - threshold percentile|
    <= ``BOUNDARY_BAND``), clearly above, clearly below. Pooled rows also carry
    quantiles of the LR probability and of both percentile margins (scale-free;
    no global raw EBM threshold is used).
    """
    _require_trace(trace)
    _require_strata(strata, trace)
    p = np.asarray(strata.lr_probability)
    raw = np.asarray(strata.ebm_raw)
    above = {"lr": p >= strata.lr_threshold, "ebm": raw >= np.asarray(strata.ebm_threshold_row)}
    margin = {"lr": np.asarray(strata.lr_percentile) - np.asarray(strata.lr_threshold_percentile),
              "ebm": np.asarray(strata.ebm_percentile) - np.asarray(strata.ebm_threshold_percentile)}
    band = {s: np.abs(margin[s]) <= BOUNDARY_BAND for s in margin}
    recs = []
    for name, (mask_g, _) in groups.items():
        mg = np.asarray(mask_g, dtype=bool)
        for g, mask in _groups(trace):
            m = _keep(trace, mask) & mg
            for score in ("lr", "ebm"):
                rows = [("above_threshold_share", (m & above[score]).sum(), m.sum()),
                        ("threshold_band_share", (m & band[score]).sum(), m.sum()),
                        ("clear_above_share", (m & above[score] & ~band[score]).sum(), m.sum()),
                        ("clear_below_share", (m & ~above[score] & ~band[score]).sum(), m.sum())]
                recs += [{"group": name, "score": score, "fold": g,
                          **_row(k, n, d, OVERLAP_LABELS)} for k, n, d in rows]
                if g == "all" and m.any():
                    qs = {"lr_probability": p[m]} if score == "lr" else {}
                    qs[f"{score}_percentile_margin"] = margin[score][m]
                    for qname, vals in qs.items():
                        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
                            recs.append({"group": name, "score": score, "fold": g,
                                         "metric": f"{qname}_q{int(q * 100):02d}",
                                         "label": f"{qname}, {int(q * 100)}th percentile",
                                         "numerator": None, "denominator": None,
                                         "value": float(np.quantile(vals, q)),
                                         "status": "quantile"})
    out = pd.DataFrame(recs)
    out["numerator"] = pd.array(out["numerator"].tolist(), dtype="Int64")
    out["denominator"] = pd.array(out["denominator"].tolist(), dtype="Int64")
    return out[["group", "score", "fold", "metric", "label", "numerator", "denominator", "value",
                "status"]]


# ======================================================================
# Arithmetic baselines (not statistical tests)
# ======================================================================
def unique_positive_chance_baselines(trace: RouterTrace, refs: Mapping[str, UpstreamFlags],
                                     matched: DiagnosticResult) -> pd.DataFrame:
    """
    Expected unique positives if a router's unique flags were drawn at random
    from the rows it could have flagged that the reference does not (frozen:
    not reference-flagged and not the arm's own Pass 1; matched: the common
    eligible rows outside the K-matched reference), and the naive random-veto
    expectation for Pass 1 interceptions. ``matched``: the Step 2C
    ``pass2_matched_contribution`` result (reused, not recomputed).
    """
    _require_trace(trace)
    flags = _require_refs(trace, refs)
    y = np.asarray(trace.y) == 1
    recs = []
    for arm in TRACE_ARMS:
        p1a, p2a, _ = _states(trace, arm)
        for key in ("lr", "ebm", "union"):
            fa = flags[key]
            for g, mask in _groups(trace):
                k = _keep(trace, mask)
                uf, pool = p2a & ~fa & k, ~fa & ~p1a & k
                rate = (pool & y).sum() / pool.sum() if pool.sum() else np.nan
                recs.append({"setting": "frozen_pass2_unique", "arm": arm, "reference": key,
                             "fold": g, "flags": int(uf.sum()), "pool_rows": int(pool.sum()),
                             "pool_positives": int((pool & y).sum()), "pool_rate": rate,
                             "expected_positives": float(uf.sum() * rate),
                             "observed_positives": int((uf & y).sum())})
                ic, fl = p1a & fa & k, fa & k
                prec = (fl & y).sum() / fl.sum() if fl.sum() else np.nan
                recs.append({"setting": "pass1_random_veto", "arm": arm, "reference": key,
                             "fold": g, "flags": int(ic.sum()), "pool_rows": int(fl.sum()),
                             "pool_positives": int((fl & y).sum()), "pool_rate": prec,
                             "expected_positives": float(ic.sum() * prec),
                             "observed_positives": int((ic & y).sum())})
    tab = matched.table
    for (arm, key, g), grp in tab[tab["reference"].isin(["lr_matched", "ebm_matched"])].groupby(
            ["arm", "reference", "fold"], sort=False):
        v = dict(zip(grp["metric"], grp["numerator"]))
        pool_rows = int(v["rows"]) - int(v["reference_flags"])
        pool_pos = int(v["positives"]) - int(v["reference_tp"])
        rate = pool_pos / pool_rows if pool_rows else np.nan
        uf = int(v["unique_tp"]) + int(v["unique_fp"])
        recs.append({"setting": "matched_pass2_unique", "arm": arm, "reference": key, "fold": g,
                     "flags": uf, "pool_rows": pool_rows, "pool_positives": pool_pos,
                     "pool_rate": rate, "expected_positives": float(uf * rate),
                     "observed_positives": int(v["unique_tp"])})
    out = pd.DataFrame(recs)
    out["excess_positives"] = out["observed_positives"] - out["expected_positives"]
    return out


def model_disagreement_baseline(trace: RouterTrace, refs: Mapping[str, UpstreamFlags],
                                matched: DiagnosticResult) -> pd.DataFrame:
    """
    How much two decent upstream models already disagree: frozen LR-only /
    EBM-only flags and positives; and, per arm, EBM vs LR re-cut at the SAME
    K as that arm's matched comparison (common eligible population), next to
    the router's own unique positives (Step 2C values, reused).
    """
    _require_trace(trace)
    flags = _require_refs(trace, refs)
    y = np.asarray(trace.y) == 1
    lr, ebm = flags["lr"], flags["ebm"]
    recs = []
    for g, mask in _groups(trace):
        k = _keep(trace, mask)
        recs.append({"setting": "frozen", "arm": "", "fold": g, "k": None,
                     "lr_flags": int((lr & k).sum()), "ebm_flags": int((ebm & k).sum()),
                     "lr_only_tp": int((lr & ~ebm & y & k).sum()),
                     "ebm_only_tp": int((ebm & ~lr & y & k).sum()),
                     "lr_only_fp": int((lr & ~ebm & ~y & k).sum()),
                     "ebm_only_fp": int((ebm & ~lr & ~y & k).sum())})
    sels = matched.meta["selections"]
    for arm in TRACE_ARMS:
        ml, me = np.asarray(sels[arm]["lr"].flags), np.asarray(sels[arm]["ebm"].flags)
        pop = np.asarray(sels[arm]["lr"].population)
        for g, mask in _groups(trace):
            k = _keep(trace, mask) & pop
            pool = k & ~ml
            rate = (pool & y).sum() / pool.sum() if pool.sum() else np.nan
            uf = me & ~ml & k
            sub = matched.table[(matched.table["arm"] == arm)
                                & matched.table["fold"].map(lambda f: f == g)]
            val = {(r, m): n for r, m, n in zip(sub["reference"], sub["metric"], sub["numerator"])}
            recs.append({"setting": "matched_same_k", "arm": arm, "fold": g,
                         "k": int((ml & k).sum()),
                         "lr_flags": int((ml & k).sum()), "ebm_flags": int((me & k).sum()),
                         "lr_only_tp": int((ml & ~me & y & k).sum()),
                         "ebm_only_tp": int((me & ~ml & y & k).sum()),
                         "lr_only_fp": int((ml & ~me & ~y & k).sum()),
                         "ebm_only_fp": int((me & ~ml & ~y & k).sum()),
                         "ebm_minus_lr_tp": int((me & y & k).sum()) - int((ml & y & k).sum()),
                         "ebm_only_tp_expected_from_pool": float(uf.sum() * rate),
                         "router_unique_tp_vs_lr": int(val[("lr_matched", "unique_tp")]),
                         "router_unique_tp_vs_ebm": int(val[("ebm_matched", "unique_tp")])})
    out = pd.DataFrame(recs)
    out["k"] = pd.array(out["k"].tolist(), dtype="Int64")
    return out


# ======================================================================
# Disclosures
# ======================================================================
def overlap_notes(trace: RouterTrace, strata: Optional[ScoreStrata] = None) -> Tuple[str, ...]:
    """The Step 2D disclosures (data-driven where the artifacts allow)."""
    _require_trace(trace)
    rf = trace.views[RF_ARM]
    c = rf.metadata.get("constraints")
    gp = trace.views[GLASS_ARM].pass2_population.get("normalised")
    rp = rf.pass2_population.get("normalised")
    notes = [
        "Q1 (functional behaviour) is answered descriptively from final routing states. Q2 "
        "(incremental information) is only described here: score-conditional rates and "
        "expectations are not a controlled test, and nothing here measures Stage 4 or the "
        "integrated cascade.",
        "Final Pass 1 and Pass 2 never overlap by construction (Pass 1 precedence). Zero final "
        "overlap is a routing invariant, not evidence of independent signals; candidate / signal "
        "overlap is unavailable because no per-row candidate matches, fold rule sets or RF Pass 2 "
        "outer scores were persisted.",
        f"Score strata were fixed before any result was inspected: within each router outer fold, "
        f"{SCORE_BINS} label-free quantile bins of the LR calibrated probability and of the EBM raw "
        f"score, and a {JOINT_BINS} x {JOINT_BINS} LR x EBM grid; merged duplicate edges keep tied "
        f"scores together; cells under {MIN_SUPPORT} rows are low-support; the threshold band is "
        f"+/- {BOUNDARY_BAND:.2f} in within-fold percentile.",
        "Expected positives are descriptive. expected_lr_calibrated / expected_ebm_calibrated "
        "assume the calibrated OOF probabilities are right; expected_stratified_* use the same "
        "labels inside each "
        "(fold, stratum) cell (internal standardisation). Neither is a null distribution, and no "
        "standard error is given because rows, folds and model decisions are not independent.",
        "Conditioning on one score at a time does not establish information beyond both; the "
        "joint grid is coarse, so within-cell variation remains possible.",
        "Unique = relative to the stated reference or to the other router, as labelled. "
        "Agreement is not inherently better and disagreement is not inherently complementary.",
        "Common reference: the GLASS arm's Stage 1 LR (threshold selected on all training labels; "
        "rates optimistic) and Stage 3 EBM (fold-nested raw thresholds) for both routers.",
        f"Pass 2 fit-time populations differ (GLASS {gp!r}, RF {rp!r}; Commit 3, F17).",
        f"RF constraint_source={rf.metadata.get('constraint_source')!r}; recorded constraints "
        f"{dict(c) if isinstance(c, Mapping) else c}. Derived from GLASS's realised OOF metrics, "
        "these targets make the arms dependent; they do not make the arms exactly rate-matched.",
        _FOLD_RF_NOTE,
        "The GLASS configuration is historical and not optimised; these results are a baseline, "
        "not its performance potential.",
    ]
    if strata is not None:
        collapsed = {s: [k for k, v in strata.n_bins[s].items() if v < strata.requested_bins[s]]
                     for s in STRATIFIERS}
        detail = "; ".join(f"{s}: folds {v}" for s, v in collapsed.items() if v) or "none"
        notes.append(f"Strata with merged or empty bins (fewer than requested): {detail}. "
                     f"LR score: {strata.provenance['lr']}. EBM: {strata.provenance['ebm_raw']}; "
                     f"calibrated {strata.provenance['ebm_probability']}.")
    return tuple(notes)
