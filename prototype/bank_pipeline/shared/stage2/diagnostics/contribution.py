"""
shared.stage2.diagnostics.contribution
======================================
Pass 2 unique-positive contribution (PR 37, Commit 2, Step 2C).

Does each router's Pass 2 (flag SUBSCRIBE) find actual positives that the
Stage 1 LR and Stage 3 EBM reference flags miss, and at what false-positive
cost? Observational, on the outer OOF training rows of a validated
``trace.RouterTrace``; nothing is fitted, tuned, recalibrated or written.

    pass2_population             per arm and router fold: Pass 2 rows, TP, FP,
                                 precision, recall, coverage, own Pass 1 remainder
    common_eligible_population   rows neither router routed to Pass 1, per fold
    pass2_positive_breakdown     Pass 2 TP / FP split by which frozen references
                                 also flag them (both, LR only, EBM only, neither)
    pass2_contribution           Pass 2 against one frozen reference flag set
    pass2_frozen_contribution    against lr.in_stage_f2_cv, ebm.nested_f2, their
                                 union and intersection, all training rows
    matched_reference_selection  label-free rate-matched LR / EBM flags on the
                                 common eligible population (one per arm)
    pass2_matched_contribution   Pass 2 against those matched flags

Definitions (P2 = Pass 2 rows, F = reference flags, within the stated population)
--------------------------------------------------------------------------------
    pass2_precision               P2 positives / P2 rows                   (registered)
    pass2_recall                  P2 positives / all positives of the population
    overlap_tp  (pass2_overlap_tp)        P2 & F & positive
    unique_tp   (unique_tp_vs_reference)  P2 & not F & positive  = the hypothetical
                                          OR's incremental TP over the reference
    unique_fp   (unique_fp_vs_reference)  P2 & not F & negative  = its incremental FP
    incremental_positive_recall   unique_tp / all positives of the population
    incremental_yield             unique_tp / P2 rows
    marginal_precision            unique_tp / (unique_tp + unique_fp)
    union_precision               (P2 or F) positives / (P2 or F) rows (hypothetical OR)

"Unique" is always relative to the stated reference flags, never across the two
router architectures (Step 2D). A hypothetical OR is not Stage 4's behaviour.

Matched flag rate
-----------------
Population: ``common_eligible = not GLASS Pass 1 and not RF Pass 1`` (a shared
decision-time population; not either router's fit-time Pass 2 population).
Within each router outer fold the reference's K highest-scoring common-eligible
rows are flagged, K = that arm's Pass 2 rows in the fold's common population
(arm-specific: GLASS and RF have different K). Scores: LR calibrated OOF
probability (comparable across LR's fold models by calibration); EBM raw OOF
score, ranked only inside its own fold (EBM shares the routers' outer folds; its
fold-specific nested thresholds show raw scores are not comparable across folds).
No label is read to choose a cut or break a tie. Ties at the K-th score are
broken by ``TIE_BREAK_RULE`` (fixed per row; not row order, which in the UCI bank
file follows contact date); ``operating_points.matched_rate_cut``'s strict and
inclusive cuts are reported as bounds on the tie effect. GLASS / RF Pass 2 are
never ranked or truncated (no continuous Pass 2 scores were persisted).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import DiagnosticResult, Ratio, _result, pass2_metrics
from .operating_points import UpstreamFlags, matched_rate_cut
from .schema import (
    SELECTION_RATE_MATCHED,
    RouterDiagnosticError,
    UnsupportedDiagnostic,
    UpstreamOOFView,
    _frozen,
)
from .suppression import (
    REFERENCE_KEYS,
    _frame,
    _require_flags,
    _row,
    primary_reference_flags,
)
from .trace import ARM_LABELS, GLASS_ARM, RF_ARM, TRACE_ARMS, RouterTrace, _groups, _require_trace

__all__ = [
    "POP_ALL", "POP_COMMON", "MATCH_SCORES", "TIE_BREAK_RULE", "MATCHED_REFERENCE_KEYS",
    "CONTRIBUTION_LABELS", "MatchedSelection", "pass2_population", "common_eligible_mask",
    "common_eligible_population", "pass2_positive_breakdown", "pass2_contribution",
    "pass2_frozen_contribution", "matched_reference_selection", "pass2_matched_contribution",
]

POP_ALL = "all_rows"
POP_COMMON = "common_eligible"
#: reference model -> the score space it is rate-matched in.
MATCH_SCORES: Mapping[str, str] = MappingProxyType({"lr": "calibrated_probability",
                                                   "ebm": "raw_model_score"})
TIE_BREAK_RULE = ("ascending SHA-256 of the row's index label (label-free, fixed per row; "
                  "not row order)")
MATCHED_REFERENCE_KEYS = ("lr_matched", "ebm_matched", "lr_or_ebm_matched")

CONTRIBUTION_LABELS: Mapping[str, str] = MappingProxyType({
    "rows": "rows in the population group",
    "positives": "actual positives in the population group",
    "full_population_positives": "actual positives of all training rows in the fold / pool",
    "pass2_n": "Pass 2 rows", "pass2_tp": "Pass 2 actual positives",
    "pass2_fp": "Pass 2 actual negatives",
    "reference_flags": "reference flags", "reference_tp": "reference true positives",
    "reference_fp": "reference false positives",
    "overlap_flags": "rows flagged by both Pass 2 and the reference",
    "overlap_tp": "positives flagged by both (pass2_overlap_tp)",
    "overlap_fp": "negatives flagged by both",
    "unique_tp": "Pass 2 positives the reference misses (unique TP = OR's incremental TP)",
    "unique_fp": "Pass 2 negatives the reference does not flag (unique FP = OR's incremental FP)",
    "reference_only_tp": "reference positives Pass 2 misses",
    "reference_only_fp": "reference negatives Pass 2 does not flag",
    "union_flags": "hypothetical OR: rows flagged by either", "union_tp": "hypothetical OR: TP",
    "union_fp": "hypothetical OR: FP",
    "matched_strict_n": "tie bound: strict cut rows (fewer than K)",
    "matched_inclusive_n": "tie bound: inclusive cut rows (more than K)",
    "tie_block": "rows sharing the K-th score (1 = no tie)",
    "tie_broken_rows": "rows chosen from the tied block by the tie-break rule",
    "unique_tp_strict": "tie bound: unique TP vs the strict cut",
    "unique_tp_inclusive": "tie bound: unique TP vs the inclusive cut",
    "pass2_rate": "Pass 2 flag rate (P2 rows / rows)",
    "upstream_flag_rate": "reference flag rate (flags / rows)",
    "pass2_precision": "Pass 2 precision (P2 TP / P2 rows)",
    "upstream_precision": "reference precision (TP / flags)",
    "pass2_recall": "Pass 2 recall (P2 TP / positives)",
    "upstream_recall": "reference recall (TP / positives)",
    "pass2_upstream_redundancy": "share of Pass 2 rows the reference also flags",
    "pass2_shared_positive_share": "share of Pass 2 positives the reference also flags",
    "pass2_unique_positive_share": "share of Pass 2 positives the reference misses",
    "pass2_new_false_positive_rate": "unique FP / Pass 2 rows",
    "incremental_positive_recall": "unique TP / positives (marginal positive recall)",
    "incremental_positive_recall_all": "unique TP / all training positives of the group",
    "incremental_yield": "unique TP / Pass 2 rows",
    "marginal_precision": "unique TP / (unique TP + unique FP)",
    "union_precision": "hypothetical OR precision",
    "jaccard_with_router": "rows in both / rows in either",
    # Pass 2 population
    "remainder_rows": "own Pass 1 remainder rows (eligible for Pass 2)",
    "remainder_positives": "positives in the own Pass 1 remainder",
    "pass2_remainder_rate": "Pass 2 rows / own Pass 1 remainder rows",
    "pass2_recall_eligible": "Pass 2 TP / positives in the own Pass 1 remainder",
})
_POP_COLUMNS = ["arm", "fold", "metric", "label", "numerator", "denominator", "value", "status"]
_COLUMNS = ["arm", "reference", "population", "fold", "metric", "label", "numerator",
            "denominator", "value", "status"]
_DIFF_METRICS = ("unique_tp", "unique_fp", "incremental_positive_recall", "marginal_precision",
                 "pass2_unique_positive_share")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _contribution_counts(y, p2, f) -> List[Tuple[str, int, Optional[int]]]:
    pos, neg = y == 1, y == 0
    union = p2 | f
    uniq = p2 & ~f
    return [
        ("rows", len(y), None), ("positives", pos.sum(), None),
        ("pass2_n", p2.sum(), None), ("pass2_tp", (p2 & pos).sum(), None),
        ("pass2_fp", (p2 & neg).sum(), None),
        ("reference_flags", f.sum(), None), ("reference_tp", (f & pos).sum(), None),
        ("reference_fp", (f & neg).sum(), None),
        ("overlap_flags", (p2 & f).sum(), None), ("overlap_tp", (p2 & f & pos).sum(), None),
        ("overlap_fp", (p2 & f & neg).sum(), None),
        ("unique_tp", (uniq & pos).sum(), None), ("unique_fp", (uniq & neg).sum(), None),
        ("reference_only_tp", (f & ~p2 & pos).sum(), None),
        ("reference_only_fp", (f & ~p2 & neg).sum(), None),
        ("union_flags", union.sum(), None), ("union_tp", (union & pos).sum(), None),
        ("union_fp", (union & neg).sum(), None),
        ("pass2_rate", p2.sum(), len(y)), ("upstream_flag_rate", f.sum(), len(y)),
        ("pass2_precision", (p2 & pos).sum(), p2.sum()),
        ("upstream_precision", (f & pos).sum(), f.sum()),
        ("pass2_recall", (p2 & pos).sum(), pos.sum()),
        ("upstream_recall", (f & pos).sum(), pos.sum()),
        ("pass2_upstream_redundancy", (p2 & f).sum(), p2.sum()),
        ("pass2_shared_positive_share", (p2 & f & pos).sum(), (p2 & pos).sum()),
        ("pass2_unique_positive_share", (uniq & pos).sum(), (p2 & pos).sum()),
        ("pass2_new_false_positive_rate", (uniq & neg).sum(), p2.sum()),
        ("incremental_positive_recall", (uniq & pos).sum(), pos.sum()),
        ("incremental_yield", (uniq & pos).sum(), p2.sum()),
        ("marginal_precision", (uniq & pos).sum(), uniq.sum()),
        ("union_precision", (union & pos).sum(), union.sum()),
        ("jaccard_with_router", (p2 & f).sum(), union.sum()),
    ]


def _records(arm, ref_key, population, fold, rows) -> List[dict]:
    return [{"arm": arm, "reference": ref_key, "population": population, "fold": fold,
             **_row(metric, num, den, CONTRIBUTION_LABELS)} for metric, num, den in rows]


def _dependence_note(trace: RouterTrace) -> str:
    rf = trace.views[RF_ARM]
    cs, c = rf.metadata.get("constraint_source"), rf.metadata.get("constraints")
    floor = c.get("min_precision_subscribe") if isinstance(c, Mapping) else None
    if cs == "match_glass_oof":
        return ("RF Router constraint_source='match_glass_oof': its Pass 2 precision floor "
                f"(min_precision_subscribe={floor!r}) and other constraints were derived from the "
                "GLASS Router's pooled OOF metrics, so Pass 2 precision and yield are not "
                "independent between the arms (Commit 4).")
    if cs is None:
        return "RF Router constraint_source is not recorded; cross-arm dependence unknown."
    return f"RF Router constraint_source={cs!r} (recorded constraints: {c})."


def _fit_population_note(trace: RouterTrace) -> str:
    g, r = trace.views[GLASS_ARM].pass2_population, trace.views[RF_ARM].pass2_population
    return (f"Pass 2 fit-time populations differ (GLASS {g.get('normalised')!r}, RF "
            f"{r.get('normalised')!r}; Commit 3, F17), so Pass 2 differences are not "
            "attributable to the symbolic selector or the forest alone.")


# ----------------------------------------------------------------------
# A. Pass 2 population
# ----------------------------------------------------------------------
def pass2_population(trace: RouterTrace) -> pd.DataFrame:
    """
    Per arm and router fold ("all" = pooled): Pass 2 rows / TP / FP, coverage
    (pass2_rate), precision, recall over all positives, and the arm's own
    Pass 1 remainder (rows, positives, pass2_remainder_rate,
    pass2_recall_eligible), via ``metrics.pass2_metrics``.
    """
    _require_trace(trace)
    y_all = np.asarray(trace.y)
    recs = []
    for arm in TRACE_ARMS:
        v = trace.views[arm]
        p1_all = np.asarray(v.pass1)
        for g, mask in _groups(trace):
            keep = np.ones(trace.n, bool) if mask is None else mask
            y, p1 = y_all[keep], p1_all[keep]
            m = pass2_metrics(v, mask)
            rem, rem_pos = int((~p1).sum()), int((y[~p1] == 1).sum())
            rows = [("rows", len(y), None), ("positives", int(y.sum()), None),
                    ("pass2_n", m["pass2_n"], None), ("pass2_tp", m["pass2_positives"], None),
                    ("pass2_fp", m["pass2_n"] - m["pass2_positives"], None),
                    ("remainder_rows", rem, None), ("remainder_positives", rem_pos, None)]
            for k in ("pass2_rate", "pass2_precision", "pass2_recall"):
                rows.append((k, m[k].numerator, m[k].denominator))
            rows.append(("pass2_remainder_rate", m["pass2_n"], rem))
            r = m["pass2_recall_eligible"]
            rows.append(("pass2_recall_eligible", r.numerator, r.denominator))
            recs += [{"arm": arm, "fold": g, **_row(k, n, d_, CONTRIBUTION_LABELS)}
                     for k, n, d_ in rows]
    return _frame(recs)[_POP_COLUMNS]


# ----------------------------------------------------------------------
# Common eligible population
# ----------------------------------------------------------------------
def common_eligible_mask(trace: RouterTrace) -> np.ndarray:
    """Rows neither router routed to Pass 1 (read-only)."""
    _require_trace(trace)
    return _frozen(~np.asarray(trace.views[GLASS_ARM].pass1)
                   & ~np.asarray(trace.views[RF_ARM].pass1), bool)


def common_eligible_population(trace: RouterTrace) -> pd.DataFrame:
    """Per router fold ("all" = pooled): the common population and what it excludes."""
    common = common_eligible_mask(trace)
    g, r = trace.views[GLASS_ARM], trace.views[RF_ARM]
    y = np.asarray(trace.y) == 1
    rows = []
    for fold, mask in _groups(trace):
        k = np.ones(trace.n, bool) if mask is None else mask
        rows.append({
            "fold": fold, "rows": int(k.sum()), "positives": int((k & y).sum()),
            "glass_remainder_rows": int((k & ~g.pass1).sum()),
            "rf_remainder_rows": int((k & ~r.pass1).sum()),
            "common_rows": int((k & common).sum()), "common_positives": int((k & common & y).sum()),
            "glass_pass2_in_common": int((k & g.pass2 & common).sum()),
            "rf_pass2_in_common": int((k & r.pass2 & common).sum()),
            "glass_pass2_outside_common": int((k & g.pass2 & r.pass1).sum()),
            "rf_pass2_outside_common": int((k & r.pass2 & g.pass1).sum()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# D. Pass 2 positives by which frozen references flag them
# ----------------------------------------------------------------------
_BREAKDOWN = ("both", "lr_only", "ebm_only", "neither")


def pass2_positive_breakdown(trace: RouterTrace,
                             refs: Mapping[str, UpstreamFlags]) -> pd.DataFrame:
    """
    Per arm and router fold: Pass 2 TP and FP split by the frozen references -
    flagged by both LR and EBM, LR only, EBM only, neither. The four parts
    partition Pass 2's TP (and FP): nothing is double-counted.
    """
    _require_trace(trace)
    lr, ebm = _require_flags(refs["lr"], trace), _require_flags(refs["ebm"], trace)
    parts = {"both": lr & ebm, "lr_only": lr & ~ebm, "ebm_only": ~lr & ebm,
             "neither": ~lr & ~ebm}
    y = np.asarray(trace.y) == 1
    out = []
    for arm in TRACE_ARMS:
        p2_all = np.asarray(trace.views[arm].pass2)
        for fold, mask in _groups(trace):
            k = np.ones(trace.n, bool) if mask is None else mask
            p2 = p2_all & k
            rec = {"arm": arm, "fold": fold, "pass2_tp": int((p2 & y).sum()),
                   "pass2_fp": int((p2 & ~y).sum())}
            for name, m in parts.items():
                rec[f"tp_{name}"] = int((p2 & y & m).sum())
                rec[f"fp_{name}"] = int((p2 & ~y & m).sum())
            out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Frozen operating points
# ----------------------------------------------------------------------
def pass2_contribution(trace: RouterTrace, flags: UpstreamFlags, *,
                       reference: Optional[str] = None) -> DiagnosticResult:
    """
    Both arms' Pass 2 against one frozen reference flag set on all training
    rows, pooled and per router fold. ``values``: pooled ``"{arm}.{metric}"``.
    """
    _require_trace(trace)
    f_all = _require_flags(flags, trace)
    key = reference or flags.name
    y_all = np.asarray(trace.y).astype(int)
    recs, values = [], {}
    for arm in TRACE_ARMS:
        p2_all = np.asarray(trace.views[arm].pass2)
        for g, mask in _groups(trace):
            keep = slice(None) if mask is None else mask
            rows = _contribution_counts(y_all[keep], p2_all[keep], f_all[keep])
            recs += _records(arm, key, POP_ALL, g, rows)
            if g == "all":
                for metric, num, den in rows:
                    values[f"{arm}.{metric}"] = (int(num) if den is None
                                                 else Ratio(metric, int(num), int(den)))
    return _result(f"Pass 2 vs {flags.name} (frozen, all training rows)", values,
                   (f"Reference {flags.name}: {flags.disclosure}",), _frame(recs)[_COLUMNS],
                   meta={"reference": flags.name, "reference_key": key,
                         "components": tuple(flags.components), "population": POP_ALL})


def pass2_frozen_contribution(trace: RouterTrace, lr_view: UpstreamOOFView,
                              ebm_view: UpstreamOOFView) -> DiagnosticResult:
    """
    Pass 2 against lr.in_stage_f2_cv, ebm.nested_f2, their union and
    intersection at the frozen operating points (Step 2B's
    ``primary_reference_flags``), all training rows. ``values``: pooled
    ``"{reference}.{arm}.{metric}"`` and RF - GLASS ``"difference.{reference}.{metric}"``
    (counts) / ``"difference_pp.{reference}.{metric}"`` (rates).
    """
    _require_trace(trace)
    refs = primary_reference_flags(lr_view, ebm_view)
    results = {k: pass2_contribution(trace, refs[k], reference=k) for k in REFERENCE_KEYS}
    table = pd.concat([results[k].table for k in REFERENCE_KEYS], ignore_index=True)
    values: Dict[str, object] = {}
    for k in REFERENCE_KEYS:
        for name, v in results[k].values.items():
            values[f"{k}.{name}"] = v
        for m in _DIFF_METRICS:
            a, b = results[k][f"{GLASS_ARM}.{m}"], results[k][f"{RF_ARM}.{m}"]
            if isinstance(a, Ratio):
                values[f"difference_pp.{k}.{m}"] = (
                    100.0 * (b.value - a.value) if a.value is not None and b.value is not None
                    else float("nan"))
            else:
                values[f"difference.{k}.{m}"] = int(b) - int(a)
    flag_counts = ", ".join(f"{k} {refs[k].n_flagged:,}" for k in REFERENCE_KEYS)
    p2_counts = ", ".join(f"{ARM_LABELS[a]} {int(trace.views[a].pass2.sum()):,}"
                          for a in TRACE_ARMS)
    notes = tuple(results[k].notes[1] for k in ("lr", "ebm")) + (
        "Unique = relative to the stated reference flags only, never across the two router "
        "architectures (Step 2D).",
        "Hypothetical OR: union_* counts describe 'Pass 2 or reference flag'. Its incremental "
        "TP / FP over the reference equal unique_tp / unique_fp. It is not Stage 4's behaviour "
        "and not a measured improvement of the integrated cascade.",
        f"Frozen operating points keep each side's own flag rate (Pass 2: {p2_counts}; "
        f"reference flags: {flag_counts}); rate differences drive unique counts - see the "
        "separate matched flag-rate analysis.",
        "Common reference: the GLASS arm's Stage 1 LR and Stage 3 EBM for both routers; the RF "
        "Router's own black-box cascade uses other upstream models.",
        _fit_population_note(trace),
        _dependence_note(trace),
        "Descriptive retrospective diagnostics on training OOF rows: no significance test, no "
        "ranking of the routers.",
    )
    return _result(f"{ARM_LABELS[GLASS_ARM]} / {ARM_LABELS[RF_ARM]} Pass 2 vs frozen LR / EBM "
                   "flags (all training rows)", values, notes, table,
                   meta={"population": POP_ALL, "difference": f"{RF_ARM} minus {GLASS_ARM}"})


# ----------------------------------------------------------------------
# Matched flag rate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MatchedSelection:
    """
    One arm's rate-matched reference flags on the common eligible population.
    ``flags``: exactly K rows per router fold (ties broken by ``tie_break``);
    ``strict`` / ``inclusive``: ``matched_rate_cut``'s bounds (fewer / more
    than K when the K-th score is tied). All arrays are read-only.
    """

    arm: str
    reference: str
    model: str
    score_name: str
    score_space: str
    score_field: str
    score_provenance: str
    population: np.ndarray
    router_set: np.ndarray
    flags: np.ndarray
    strict: np.ndarray
    inclusive: np.ndarray
    by_fold: Mapping[int, Mapping[str, object]]
    tie_break: str = TIE_BREAK_RULE
    selection: str = SELECTION_RATE_MATCHED

    @property
    def n_flagged(self) -> int:
        return int(self.flags.sum())


def _tie_key(index: pd.Index) -> np.ndarray:
    return np.array([int.from_bytes(hashlib.sha256(str(x).encode("utf-8")).digest()[:8], "big")
                     for x in index], dtype=np.uint64)


def _top_k(score: np.ndarray, pop: np.ndarray, k: int, key: np.ndarray) -> np.ndarray:
    out = np.zeros(len(score), dtype=bool)
    idx = np.flatnonzero(pop)
    if k > len(idx):
        raise RouterDiagnosticError(f"target {k} exceeds the {len(idx)} eligible rows")
    if k:
        order = np.lexsort((key[idx], -score[idx]))       # score descending, then key ascending
        out[idx[order[:k]]] = True
    return out


def matched_reference_selection(trace: RouterTrace, arm: str,
                                view: UpstreamOOFView) -> MatchedSelection:
    """
    The reference's K highest-scoring common-eligible rows in each router fold,
    K = ``arm``'s Pass 2 rows in that fold's common eligible population. No
    label is read. Score space per model: ``MATCH_SCORES``.
    """
    _require_trace(trace)
    if arm not in TRACE_ARMS:
        raise RouterDiagnosticError(f"arm must be one of {TRACE_ARMS}, got {arm!r}")
    if not isinstance(view, UpstreamOOFView) or view.model not in MATCH_SCORES:
        raise RouterDiagnosticError("expected the LR or EBM UpstreamOOFView")
    space = MATCH_SCORES[view.model]
    stream = view.scores.get(space)
    if stream is None:
        raise UnsupportedDiagnostic(f"{view.model}: no {space} score stream to rank")
    if not (view.index.equals(trace.index) and np.array_equal(view.y, trace.y)):
        raise RouterDiagnosticError(f"{view.model}: rows / labels differ from the trace")
    score = np.asarray(stream.values, dtype=float)
    if score.shape != (trace.n,):
        raise RouterDiagnosticError(f"{view.model}: score has shape {score.shape}")
    pop = np.asarray(common_eligible_mask(trace))
    if np.isnan(score[pop]).any():
        raise UnsupportedDiagnostic(f"{view.model}: {space} score has NaN in the common "
                                    "eligible population")
    fold = np.asarray(trace.fold_id)
    router_set = np.asarray(trace.views[arm].pass2) & pop
    target = {k: int((router_set & (fold == k)).sum()) for k in range(trace.n_folds)}
    cut = matched_rate_cut(stream, target_count=target, population_mask=pop, ranking="highest",
                           fold_id=fold)
    key = _tie_key(trace.index)
    exact = np.zeros(trace.n, dtype=bool)
    for k in range(trace.n_folds):
        exact |= _top_k(score, pop & (fold == k), target[k], key)
    strict, inclusive = np.asarray(cut.strict), np.asarray(cut.inclusive)
    if (strict & ~exact).any() or (exact & ~inclusive).any() or (exact & ~pop).any():
        raise RouterDiagnosticError("internal: tie-broken set is not between the strict and "
                                    "inclusive cuts inside the common population")
    for k in range(trace.n_folds):
        if int((exact & (fold == k)).sum()) != target[k]:
            raise RouterDiagnosticError(f"internal: fold {k} matched count != {target[k]}")
    by_fold = MappingProxyType({
        k: MappingProxyType({**dict(cut.by_fold[k]),
                             "tie_broken_rows": target[k] - int(cut.by_fold[k]["strict_count"])})
        for k in range(trace.n_folds)})
    return MatchedSelection(
        arm=arm, reference=f"{view.model}_matched", model=view.model, score_name=stream.name,
        score_space=space, score_field=stream.field, score_provenance=stream.provenance,
        population=_frozen(pop, bool), router_set=_frozen(router_set, bool),
        flags=_frozen(exact, bool), strict=_frozen(strict, bool),
        inclusive=_frozen(inclusive, bool), by_fold=by_fold)


def _bound_rows(sel: MatchedSelection, y, p2, keep, fold) -> List[Tuple[str, int, None]]:
    pos = y == 1
    folds = list(sel.by_fold) if fold == "all" else [fold]
    tie = sum(int(sel.by_fold[k]["tie_block"]) for k in folds)
    broken = sum(int(sel.by_fold[k]["tie_broken_rows"]) for k in folds)
    st, inc = np.asarray(sel.strict)[keep], np.asarray(sel.inclusive)[keep]
    return [("matched_strict_n", st.sum(), None), ("matched_inclusive_n", inc.sum(), None),
            ("tie_block", tie, None), ("tie_broken_rows", broken, None),
            ("unique_tp_strict", (p2 & pos & ~st).sum(), None),
            ("unique_tp_inclusive", (p2 & pos & ~inc).sum(), None)]


def pass2_matched_contribution(trace: RouterTrace, lr_view: UpstreamOOFView,
                               ebm_view: UpstreamOOFView) -> DiagnosticResult:
    """
    Each arm's Pass 2 against rate-matched LR and EBM flags (and the union of the
    two matched sets) on the common eligible population, pooled and per router
    fold. ``values``: pooled ``"{reference}.{arm}.{metric}"`` and
    ``"{reference}.{arm}.pass2_minus_reference_tp"`` (equal flag counts within an
    arm, so this is Pass 2's TP lead over the matched reference).
    ``meta["selections"]``: the ``MatchedSelection`` per arm and model.
    """
    _require_trace(trace)
    y_all = np.asarray(trace.y).astype(int)
    pop_all = np.asarray(common_eligible_mask(trace))
    full_pos = y_all == 1
    recs, values, selections = [], {}, {}
    for arm in TRACE_ARMS:
        p2_all = np.asarray(trace.views[arm].pass2) & pop_all
        sel = {m: matched_reference_selection(trace, arm, v)
               for m, v in (("lr", lr_view), ("ebm", ebm_view))}
        selections[arm] = MappingProxyType(sel)
        flag_sets = {"lr_matched": np.asarray(sel["lr"].flags),
                     "ebm_matched": np.asarray(sel["ebm"].flags),
                     "lr_or_ebm_matched": np.asarray(sel["lr"].flags) | np.asarray(sel["ebm"].flags)}
        for key, f_all in flag_sets.items():
            for g, mask in _groups(trace):
                group = np.ones(trace.n, bool) if mask is None else mask
                keep = group & pop_all
                y, p2, f = y_all[keep], p2_all[keep], f_all[keep]
                rows = _contribution_counts(y, p2, f)
                uniq_tp = int((p2 & ~f & (y == 1)).sum())
                rows += [("full_population_positives", int((group & full_pos).sum()), None),
                         ("incremental_positive_recall_all", uniq_tp,
                          int((group & full_pos).sum()))]
                if key != "lr_or_ebm_matched":
                    rows += _bound_rows(sel[key.split("_")[0]], y, p2, keep, g)
                recs += _records(arm, key, POP_COMMON, g, rows)
                if g == "all":
                    for metric, num, den in rows:
                        values[f"{key}.{arm}.{metric}"] = (int(num) if den is None
                                                           else Ratio(metric, int(num), int(den)))
            values[f"{key}.{arm}.pass2_minus_reference_tp"] = (
                values[f"{key}.{arm}.pass2_tp"] - values[f"{key}.{arm}.reference_tp"])
    lr_prov = selections[GLASS_ARM]["lr"].score_provenance
    excluded = common_eligible_population(trace).set_index("fold").loc["all"]
    notes = (
        f"Matched flag rate on the common eligible population ({int(excluded['common_rows']):,} "
        "rows neither router routed to Pass 1; a shared decision-time population, not either "
        f"router's fit-time Pass 2 population). It excludes "
        f"{int(excluded['glass_pass2_outside_common']):,} GLASS Pass 2 rows that RF routed to "
        f"Pass 1 and {int(excluded['rf_pass2_outside_common']):,} RF Pass 2 rows that GLASS "
        "routed to Pass 1; frozen results on all rows are reported separately.",
        "Selection: within each router outer fold, the reference's K highest-scoring "
        "common-eligible rows, K = that arm's Pass 2 rows in the fold's common population. "
        f"No label is read. Ties at the K-th score: {TIE_BREAK_RULE}; the strict / inclusive "
        "cuts (operating_points.matched_rate_cut) bound the tie effect and neither is preferred.",
        "K is the router's observed count: a descriptive operating-rate control, not an "
        "independent pre-registration and not evidence of superiority. The matched LR / EBM "
        "flags are re-cut at each router's rate; they are not the frozen operating points or a "
        "configuration either model was tuned for.",
        "Arm-specific: GLASS and RF have different K, so each arm is compared with its own "
        "matched references; this is not an equal-flag-count comparison between the routers. "
        "lr_or_ebm_matched is the union of two K-matched sets and its size is not matched.",
        f"Scores: LR {MATCH_SCORES['lr']} ({lr_prov}); rows of one router fold come from several "
        "LR fold models and are compared in the calibrated space. EBM "
        f"{MATCH_SCORES['ebm']}, ranked only within its own fold (EBM shares the routers' outer "
        "folds; its fold-specific nested thresholds show raw scores are not comparable across "
        "folds). GLASS / RF Pass 2 are never ranked or truncated.",
        _fit_population_note(trace),
        _dependence_note(trace),
    )
    return _result(f"{ARM_LABELS[GLASS_ARM]} / {ARM_LABELS[RF_ARM]} Pass 2 vs rate-matched LR / "
                   "EBM flags (common eligible population)", values, notes,
                   _frame(recs)[_COLUMNS],
                   meta={"population": POP_COMMON, "selections": MappingProxyType(selections),
                         "tie_break": TIE_BREAK_RULE})
