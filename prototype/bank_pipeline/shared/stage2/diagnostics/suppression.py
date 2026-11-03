"""
shared.stage2.diagnostics.suppression
=====================================
Pass 1 false-positive suppression (PR 37, Commit 2, Step 2B).

How each router's Pass 1 (route NOT_SUBSCRIBE) sits against the positive flags
of the registered upstream operating points: the reference false positives it
intercepts (a potential benefit), the reference true positives it suppresses
(an error) and what stays outside Pass 1. Observational, on the outer OOF
training rows of a validated ``trace.RouterTrace``. Nothing here fits, tunes,
recalibrates, re-thresholds or writes anything.

    pass1_population          per arm and router fold: Pass 1 rows, negatives,
                              positives, NPV and positive leakage
    pass1_suppression         one reference flag set against both arms' Pass 1,
                              pooled and per router fold
    primary_reference_flags   lr.in_stage_f2_cv, ebm.nested_f2, their union
                              (either flags) and intersection (both flag)
    reference_flag_summary    flags, TP, FP and precision of each reference,
                              with its operating-point provenance
    pass1_suppression_report  all four references for both arms, pooled values,
                              RF - GLASS differences and the disclosures

Definitions (P1 = Pass 1 rows, F = reference positive flags)
-----------------------------------------------------------
    pass1_npv                     negatives in P1 / P1 rows
    pass1_leakage_share           positives in P1 / all positives
    reference_fp                  F & negative          (reference false positives)
    reference_tp                  F & positive          (reference true positives)
    intercepted_fp                P1 & F & negative     (benefit)
    suppressed_tp                 P1 & F & positive     (error, never netted)
    pass1_fp_suppression          intercepted_fp / reference_fp
                                  (reference FP interception rate)
    pass1_tp_cost                 suppressed_tp / reference_tp
                                  (reference TP suppression rate)
    pass1_intercepted_flag_npv    intercepted_fp / (P1 & F)
    retained_fp / retained_tp     F & not P1 & negative / positive
    retained_upstream_precision   retained_tp / (F & not P1)   (a HYPOTHETICAL
                                  Pass 1 veto of the reference flags, not the
                                  cascade's output)

Each ratio is a ``metrics.Ratio`` named as in ``metrics.METRIC_DEFINITIONS``,
so an empty denominator is undefined, never 0. Pooled values equal
``metrics.router_upstream_metrics`` / ``metrics.pass1_metrics`` (tested).

Alignment and folds
-------------------
Reference flags must cover the trace's rows in the same order with the same
labels. Fold-level rows always use the ROUTERS' outer folds; a reference's own
fold plan may differ (Stage 1 LR cross-fits on 10 folds) and is recorded, not
refused. Only registered operating points (``operating_points.OPERATING_POINTS``,
or ``combine_flags`` of them) are accepted; there is no bare-threshold entry.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import DiagnosticResult, Ratio, _result, pass1_metrics
from .operating_points import (
    OPERATING_POINTS,
    UpstreamFlags,
    combine_flags,
    resolve_operating_point,
)
from .schema import RouterDiagnosticError, UpstreamOOFView
from .trace import ARM_LABELS, GLASS_ARM, RF_ARM, TRACE_ARMS, RouterTrace, _groups, _require_trace

__all__ = [
    "PRIMARY_REFERENCE_POINTS", "REFERENCE_KEYS", "SUPPRESSION_COUNTS", "SUPPRESSION_LABELS",
    "pass1_population", "pass1_suppression", "primary_reference_flags",
    "reference_flag_summary", "pass1_suppression_report",
]

#: The registered primary operating points Step 2B measures against.
PRIMARY_REFERENCE_POINTS: Mapping[str, str] = MappingProxyType({
    "lr": "lr.in_stage_f2_cv", "ebm": "ebm.nested_f2"})
#: lr, ebm, union (either flags), intersection (both flag; diagnostic context).
REFERENCE_KEYS: Tuple[str, ...] = ("lr", "ebm", "union", "intersection")

#: Count columns (no denominator).
SUPPRESSION_COUNTS: Mapping[str, str] = MappingProxyType({
    "rows": "rows in the group",
    "reference_flags": "rows the reference flags SUBSCRIBE",
    "reference_tp": "reference true positives (flagged positives)",
    "reference_fp": "reference false positives (flagged negatives)",
    "intercepted_flags": "reference flags routed by Pass 1",
    "intercepted_fp": "reference false positives routed by Pass 1 (potential benefit)",
    "suppressed_tp": "reference true positives routed by Pass 1 (error)",
    "retained_flags": "reference flags outside Pass 1",
    "retained_fp": "reference false positives outside Pass 1",
    "retained_tp": "reference true positives outside Pass 1",
    "pass1_positives_unflagged": "Pass 1 positives the reference does not flag",
})
SUPPRESSION_LABELS: Mapping[str, str] = MappingProxyType({
    **SUPPRESSION_COUNTS,
    "upstream_flag_rate": "reference flag rate (flags / rows)",
    "upstream_precision": "reference precision (TP / flags)",
    "pass1_fp_suppression": "reference FP interception rate (intercepted FP / reference FP)",
    "pass1_tp_cost": "reference TP suppression rate (suppressed TP / reference TP)",
    "pass1_upstream_flag_rate": "share of Pass 1 rows the reference flags",
    "pass1_intercepted_flag_npv": "negatives among intercepted flags (intercepted FP / flags in P1)",
    "retained_upstream_precision": "precision of flags outside Pass 1 (hypothetical veto)",
    "pass1_leak_upstream_flagged": "share of Pass 1 positives the reference flags",
    "pass1_leak_upstream_missed": "share of Pass 1 positives the reference misses",
})
_POPULATION_LABELS = MappingProxyType({
    "rows": "rows in the group", "positives": "actual positives", "negatives": "actual negatives",
    "pass1_rows": "Pass 1 rows", "pass1_negatives": "Pass 1 actual negatives (correct)",
    "pass1_positives": "Pass 1 actual positives (leaked)",
    "pass1_rate": "Pass 1 coverage (Pass 1 rows / rows)",
    "pass1_npv": "Pass 1 NPV (negatives / Pass 1 rows)",
    "pass1_positive_rate": "within-band leakage (positives / Pass 1 rows)",
    "pass1_leakage_share": "positive leakage (Pass 1 positives / all positives)",
    "pass1_negative_recall": "negatives routed by Pass 1 / all negatives",
})
_COLUMNS = ["arm", "reference", "fold", "metric", "label", "numerator", "denominator", "value",
            "status"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _row(metric: str, num, den, labels) -> dict:
    if den is None:
        return {"metric": metric, "label": labels[metric], "numerator": int(num),
                "denominator": None, "value": float(num), "status": "count"}
    r = Ratio(metric, int(num), int(den))
    return {"metric": metric, "label": labels[metric], "numerator": r.numerator,
            "denominator": r.denominator, "value": r.as_float(), "status": r.status}


def _frame(recs: List[dict]) -> pd.DataFrame:
    out = pd.DataFrame(recs)
    out["numerator"] = pd.array(out["numerator"].tolist(), dtype="Int64")
    out["denominator"] = pd.array(out["denominator"].tolist(), dtype="Int64")
    return out


def _require_flags(flags, trace: RouterTrace) -> np.ndarray:
    if not isinstance(flags, UpstreamFlags):
        raise RouterDiagnosticError(
            "expected UpstreamFlags from resolve_operating_point / combine_flags (registered "
            "operating points only; there is no bare-threshold entry)")
    if not flags.components or any(c not in OPERATING_POINTS for c in flags.components):
        raise RouterDiagnosticError(
            f"{flags.name}: components {flags.components} are not all registered operating points")
    f = np.asarray(flags.flags)
    if f.shape != (trace.n,) or f.dtype != bool:
        raise RouterDiagnosticError(
            f"{flags.name}: flags must be a boolean array over the trace's {trace.n:,} rows "
            f"(got shape {f.shape}, dtype {f.dtype})")
    if not flags.index.equals(trace.index):
        raise RouterDiagnosticError(f"{flags.name}: flags cover different rows / order than the trace")
    if not np.array_equal(np.asarray(flags.y), np.asarray(trace.y)):
        raise RouterDiagnosticError(f"{flags.name}: flag labels differ from the trace labels")
    return f


def _fold_plan(flags: UpstreamFlags, trace: RouterTrace) -> str:
    if len(flags.components) > 1:
        return f"combination of {', '.join(flags.components)} (see the component fold plans)"
    fold = np.asarray(flags.fold_id)
    if fold.shape == np.asarray(trace.fold_id).shape and np.array_equal(fold, trace.fold_id):
        return "same as the routers' outer folds"
    return (f"{len(np.unique(fold))} folds, not the routers' {trace.n_folds}; router-fold values "
            "group this reference's out-of-fold flags by the routers' partition")


def _provenance(flags: UpstreamFlags, trace: RouterTrace) -> Mapping[str, object]:
    return MappingProxyType({
        "name": flags.name, "components": tuple(flags.components), "model": flags.model,
        "score_space": flags.score_space, "threshold": flags.threshold, "field": flags.field,
        "selection": flags.selection, "disclosure": flags.disclosure,
        "fold_plan": _fold_plan(flags, trace), "n_flagged": flags.n_flagged})


def _suppression_counts(y, p1, f) -> List[Tuple[str, int, Optional[int]]]:
    pos, neg = y == 1, y == 0
    return [
        ("rows", len(y), None),
        ("reference_flags", f.sum(), None),
        ("reference_tp", (f & pos).sum(), None),
        ("reference_fp", (f & neg).sum(), None),
        ("intercepted_flags", (p1 & f).sum(), None),
        ("intercepted_fp", (p1 & f & neg).sum(), None),
        ("suppressed_tp", (p1 & f & pos).sum(), None),
        ("retained_flags", (f & ~p1).sum(), None),
        ("retained_fp", (f & ~p1 & neg).sum(), None),
        ("retained_tp", (f & ~p1 & pos).sum(), None),
        ("pass1_positives_unflagged", (p1 & pos & ~f).sum(), None),
        ("upstream_flag_rate", f.sum(), len(y)),
        ("upstream_precision", (f & pos).sum(), f.sum()),
        ("pass1_fp_suppression", (p1 & f & neg).sum(), (f & neg).sum()),
        ("pass1_tp_cost", (p1 & f & pos).sum(), (f & pos).sum()),
        ("pass1_upstream_flag_rate", (p1 & f).sum(), p1.sum()),
        ("pass1_intercepted_flag_npv", (p1 & f & neg).sum(), (p1 & f).sum()),
        ("retained_upstream_precision", (f & ~p1 & pos).sum(), (f & ~p1).sum()),
        ("pass1_leak_upstream_flagged", (p1 & pos & f).sum(), (p1 & pos).sum()),
        ("pass1_leak_upstream_missed", (p1 & pos & ~f).sum(), (p1 & pos).sum()),
    ]


# ----------------------------------------------------------------------
# Pass 1 population
# ----------------------------------------------------------------------
def pass1_population(trace: RouterTrace) -> pd.DataFrame:
    """
    Per arm and router fold ("all" = pooled): rows, positives, negatives,
    Pass 1 rows / negatives / positives, and ``metrics.pass1_metrics``'s
    pass1_rate, pass1_npv, pass1_positive_rate, pass1_leakage_share and
    pass1_negative_recall (long format, explicit numerator / denominator).
    """
    _require_trace(trace)
    y_all = np.asarray(trace.y)
    recs = []
    for arm in TRACE_ARMS:
        v = trace.views[arm]
        for g, mask in _groups(trace):
            y = y_all if mask is None else y_all[mask]
            m = pass1_metrics(v, mask)
            p1n, p1p = m["pass1_n"], m["pass1_positives"]
            counts = (("rows", len(y)), ("positives", int(y.sum())),
                      ("negatives", int((y == 0).sum())), ("pass1_rows", p1n),
                      ("pass1_negatives", p1n - p1p), ("pass1_positives", p1p))
            rows = [_row(k, c, None, _POPULATION_LABELS) for k, c in counts]
            for k in ("pass1_rate", "pass1_npv", "pass1_positive_rate", "pass1_leakage_share",
                      "pass1_negative_recall"):
                r = m[k]
                rows.append(_row(k, r.numerator, r.denominator, _POPULATION_LABELS))
            recs += [{"arm": arm, "fold": g, **r} for r in rows]
    return _frame(recs)[[c for c in _COLUMNS if c != "reference"]]


# ----------------------------------------------------------------------
# Pass 1 vs one reference flag set
# ----------------------------------------------------------------------
def pass1_suppression(trace: RouterTrace, flags: UpstreamFlags, *,
                      reference: Optional[str] = None) -> DiagnosticResult:
    """
    Both arms' Pass 1 against one reference flag set (a registered operating
    point, or ``combine_flags`` of registered points), pooled and per router
    fold. ``table``: long format (``_COLUMNS``); ``values``: pooled counts and
    Ratios keyed ``"{arm}.{metric}"``; ``meta["reference"]``: the operating
    point's provenance.
    """
    _require_trace(trace)
    f_all = _require_flags(flags, trace)
    ref_key = reference or flags.name
    y_all = np.asarray(trace.y).astype(int)
    recs, values = [], {}
    for arm in TRACE_ARMS:
        p1_all = np.asarray(trace.views[arm].pass1)
        for g, mask in _groups(trace):
            keep = slice(None) if mask is None else mask
            for metric, num, den in _suppression_counts(y_all[keep], p1_all[keep], f_all[keep]):
                rec = _row(metric, num, den, SUPPRESSION_LABELS)
                recs.append({"arm": arm, "reference": ref_key, "fold": g, **rec})
                if g == "all":
                    values[f"{arm}.{metric}"] = (int(num) if den is None
                                                 else Ratio(metric, int(num), int(den)))
    prov = _provenance(flags, trace)
    notes = (f"Reference {flags.name}: {flags.disclosure}",
             f"Reference fold plan: {prov['fold_plan']}. Fold-level rows use the routers' outer "
             "folds.")
    return _result(f"Pass 1 vs {flags.name} (outer OOF)", values, notes, _frame(recs),
                   meta={"reference": prov, "reference_key": ref_key})


# ----------------------------------------------------------------------
# The registered primary references
# ----------------------------------------------------------------------
def primary_reference_flags(lr_view: UpstreamOOFView,
                            ebm_view: UpstreamOOFView) -> Mapping[str, UpstreamFlags]:
    """
    {"lr": lr.in_stage_f2_cv, "ebm": ebm.nested_f2, "union": either flags,
    "intersection": both flag}, resolved only through the operating-point
    registry (LR: calibrated probability >= its stored threshold; EBM: the
    stored fold-nested raw-score decisions).
    """
    lr = resolve_operating_point(lr_view, PRIMARY_REFERENCE_POINTS["lr"])
    ebm = resolve_operating_point(ebm_view, PRIMARY_REFERENCE_POINTS["ebm"])
    return MappingProxyType({"lr": lr, "ebm": ebm, "union": combine_flags(lr, ebm, how="or"),
                             "intersection": combine_flags(lr, ebm, how="and")})


def reference_flag_summary(trace: RouterTrace,
                           refs: Mapping[str, UpstreamFlags]) -> pd.DataFrame:
    """One row per reference: provenance, flags, TP, FP and precision (pooled)."""
    _require_trace(trace)
    y = np.asarray(trace.y).astype(int)
    rows = []
    for key, flags in refs.items():
        f = _require_flags(flags, trace)
        prov = _provenance(flags, trace)
        prec = Ratio("upstream_precision", int((f & (y == 1)).sum()), int(f.sum()))
        rate = Ratio("upstream_flag_rate", int(f.sum()), trace.n)
        rows.append({"reference": key, "operating_point": flags.name,
                     "score_space": prov["score_space"], "threshold": prov["threshold"],
                     "selection": prov["selection"], "fold_plan": prov["fold_plan"],
                     "flags": int(f.sum()), "tp": prec.numerator,
                     "fp": int((f & (y == 0)).sum()), "flag_rate": rate.as_float(),
                     "precision": prec.as_float(), "precision_status": prec.status})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
_DIFF_METRICS = ("pass1_fp_suppression", "pass1_tp_cost", "pass1_intercepted_flag_npv",
                 "retained_upstream_precision")


def _report_notes(trace: RouterTrace, refs: Mapping[str, UpstreamFlags]) -> Tuple[str, ...]:
    rf = trace.views[RF_ARM]
    cs, constraints = rf.metadata.get("constraint_source"), rf.metadata.get("constraints")
    if cs == "match_glass_oof":
        dependence = (
            "RF Router constraint_source='match_glass_oof' (recorded constraints: "
            f"{dict(constraints) if isinstance(constraints, Mapping) else constraints}). Its "
            "Pass 1 operating constraints were derived from the GLASS Router's pooled OOF "
            "metrics, so Pass 1 leakage, NPV and suppression are not independent between the "
            "arms; independent shared criteria are Commit 4's.")
    elif cs is None:
        dependence = "RF Router constraint_source is not recorded; cross-arm dependence unknown."
    else:
        dependence = f"RF Router constraint_source={cs!r} (recorded constraints: {constraints})."
    plans = "; ".join(f"{k} ({refs[k].name}): {_fold_plan(refs[k], trace)}"
                      for k in ("lr", "ebm") if k in refs)
    return (
        "An intercepted reference false positive (Pass 1 & flag & negative) is a potential "
        "benefit; a suppressed reference true positive (Pass 1 & flag & positive) is an error. "
        "Both are reported with their own denominators and are never netted.",
        "These are overlaps on the same training rows, not the integrated cascade's performance: "
        "Stage 4 combines router and upstream evidence, so a reference flag inside Pass 1 is not "
        "necessarily removed from the final output. 'Retained' counts and "
        "retained_upstream_precision describe a hypothetical Pass 1 veto of the reference flags.",
        "Common reference: both routers are measured against the same flags, the GLASS arm's "
        "Stage 1 LR and Stage 3 EBM at their registered primary operating points. The RF "
        "Router's own black-box cascade uses other upstream models; this is not that cascade.",
        f"Fold plans: {plans}. Fold-level rows use the routers' outer folds; per-fold values are "
        "descriptive, not independent replicates, and no significance test is made.",
        dependence,
        "Scope: Pass 2 and abstained rows are not analysed here (Steps 2C / 2D), and no matched "
        "flag-rate comparison is made (Step 2C). The intersection reference is diagnostic "
        "context for the union, not a redundancy analysis (Step 2D).",
    )


def pass1_suppression_report(trace: RouterTrace, lr_view: UpstreamOOFView,
                             ebm_view: UpstreamOOFView) -> DiagnosticResult:
    """
    Step 2B: both arms' Pass 1 against lr.in_stage_f2_cv, ebm.nested_f2, their
    union and their intersection. ``table``: all references (long format);
    ``values``: pooled counts / Ratios ``"{reference}.{arm}.{metric}"``, Pass 1
    NPV and leakage ``"{arm}.pass1_npv"`` / ``"{arm}.pass1_leakage_share"``,
    RF - GLASS differences ``"difference_pp.{reference}.{metric}"`` and the
    union reconciliation; ``meta["references"]``: each reference's provenance.
    """
    _require_trace(trace)
    refs = primary_reference_flags(lr_view, ebm_view)
    results = {k: pass1_suppression(trace, refs[k], reference=k) for k in REFERENCE_KEYS}
    table = pd.concat([results[k].table for k in REFERENCE_KEYS], ignore_index=True)
    values: Dict[str, object] = {}
    for arm in TRACE_ARMS:
        m = pass1_metrics(trace.views[arm])
        values[f"{arm}.pass1_npv"] = m["pass1_npv"]
        values[f"{arm}.pass1_leakage_share"] = m["pass1_leakage_share"]
    for k in REFERENCE_KEYS:
        for name, v in results[k].values.items():
            values[f"{k}.{name}"] = v
        for metric in _DIFF_METRICS:
            a, b = results[k][f"{GLASS_ARM}.{metric}"], results[k][f"{RF_ARM}.{metric}"]
            values[f"difference_pp.{k}.{metric}"] = (
                100.0 * (b.value - a.value) if a.value is not None and b.value is not None
                else float("nan"))
    n = {k: refs[k].n_flagged for k in REFERENCE_KEYS}
    values["union_reconciles"] = n["union"] == n["lr"] + n["ebm"] - n["intersection"]
    notes = tuple(results[k].notes[1] for k in ("lr", "ebm"))     # operating-point disclosures
    notes += _report_notes(trace, refs)
    return _result(f"{ARM_LABELS[GLASS_ARM]} / {ARM_LABELS[RF_ARM]} Pass 1 vs primary LR / EBM "
                   "flags (outer OOF)", values, notes, table,
                   meta={"references": MappingProxyType(
                             {k: results[k].meta["reference"] for k in REFERENCE_KEYS}),
                         "difference": f"{RF_ARM} minus {GLASS_ARM}"})
