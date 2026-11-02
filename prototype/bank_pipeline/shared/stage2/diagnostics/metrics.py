"""
shared.stage2.diagnostics.metrics
=================================
Train-side, decision-level router diagnostics on outer out-of-fold outputs.

Every rate is a ``Ratio`` carrying its numerator and denominator; an empty
denominator gives ``value None`` with status ``undefined_empty_denominator``
rather than 0 or NaN. ``METRIC_DEFINITIONS`` is the single place each metric is
defined (Markdown docs are deferred to the PR 37 cleanup step).

These are diagnostics on training rows: the upstream thresholds they use were
selected on training labels (see each operating point's ``disclosure``), and
every result carries ``HISTORICAL_TEST_EXPOSURE``. Nothing here reads or
derives anything from the test split.

Conventions match ``rf_router.evaluation.router_metrics.evaluate_router``
(``test_router_diagnostics_metrics`` checks parity): Pass 1 leakage share and
Pass 2 recall are over ALL positives; Pass 2 recall over eligible positives
(those not claimed by Pass 1) is reported separately.

Score-level diagnostics need a score stream: ``require_score_stream`` refuses
GLASS (rule membership, no persisted per-fold rules) and RF Pass 2 (outer
scores not persisted), and allows RF Pass 1 only with a passed equivalence
report. Decision-level diagnostics never depend on that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .operating_points import MatchedCut, UpstreamFlags, matched_rate_cut
from .schema import (
    ABSTAIN,
    HISTORICAL_TEST_EXPOSURE,
    PASS_1_NOT_SUBSCRIBE,
    PASS_2_SUBSCRIBE,
    ROUTER_STATES,
    SELECTION_RATE_MATCHED,
    RouterDiagnosticError,
    RouterOOFView,
    ScoreStream,
    UnsupportedDiagnostic,
)

__all__ = [
    "Ratio", "DiagnosticResult", "METRIC_DEFINITIONS", "UNDEFINED",
    "state_counts", "pass1_metrics", "pass2_metrics", "router_metrics",
    "router_upstream_metrics", "router_router_table", "matched_rate_comparison",
    "require_score_stream",
]

UNDEFINED = "undefined_empty_denominator"
DEFINED = "defined"


@dataclass(frozen=True)
class Ratio:
    name: str
    numerator: int
    denominator: int

    @property
    def value(self) -> Optional[float]:
        return None if self.denominator == 0 else self.numerator / self.denominator

    @property
    def status(self) -> str:
        return UNDEFINED if self.denominator == 0 else DEFINED

    def as_float(self) -> float:
        """``value``, NaN when undefined (for tables and comparisons only)."""
        return float("nan") if self.value is None else float(self.value)


Value = Union[Ratio, int, float, str, None]

#: name -> (numerator, denominator). Counts are listed with denominator "".
METRIC_DEFINITIONS: Mapping[str, Tuple[str, str]] = MappingProxyType({
    # composition
    "coverage": ("rows routed Pass 1 or Pass 2", "all rows"),
    "abstain_rate": ("abstained rows", "all rows"),
    "pass1_rate": ("Pass 1 rows", "all rows"),
    "pass2_rate": ("Pass 2 rows", "all rows"),
    "abstain_positive_rate": ("positives among abstained rows", "abstained rows"),
    # Pass 1 (route NOT_SUBSCRIBE)
    "pass1_npv": ("negatives among Pass 1 rows", "Pass 1 rows"),
    "pass1_positive_rate": ("positives among Pass 1 rows (within-band leakage)", "Pass 1 rows"),
    "pass1_leakage_share": ("positives routed by Pass 1", "all positives"),
    "pass1_negative_recall": ("negatives routed by Pass 1", "all negatives"),
    # Pass 2 (flag SUBSCRIBE)
    "pass2_precision": ("positives among Pass 2 rows", "Pass 2 rows"),
    "pass2_recall": ("positives flagged by Pass 2", "all positives"),
    "pass2_recall_eligible": ("positives flagged by Pass 2",
                              "positives not routed by Pass 1 (eligible after Pass 1 precedence)"),
    # router x upstream flags
    "upstream_flag_rate": ("rows the upstream flags", "all rows"),
    "pass1_upstream_flag_rate": ("Pass 1 rows the upstream flags", "Pass 1 rows"),
    "pass1_fp_suppression": ("upstream false positives routed by Pass 1",
                             "upstream false positives (flagged negatives)"),
    "pass1_tp_cost": ("upstream true positives routed by Pass 1",
                      "upstream true positives (flagged positives)"),
    "pass1_leak_upstream_flagged": ("Pass 1 positives the upstream flags", "Pass 1 positives"),
    "pass1_leak_upstream_missed": ("Pass 1 positives the upstream does not flag", "Pass 1 positives"),
    "pass2_upstream_redundancy": ("Pass 2 rows the upstream also flags", "Pass 2 rows"),
    "pass2_unique_positive_share": ("Pass 2 positives the upstream does not flag",
                                    "Pass 2 positives"),
    "pass2_shared_positive_share": ("Pass 2 positives the upstream also flags", "Pass 2 positives"),
    "pass2_new_false_positive_rate": ("Pass 2 negatives the upstream does not flag", "Pass 2 rows"),
    "covered_agreement": ("covered rows where (Pass 2) == (upstream flag)", "covered rows"),
    "abstain_upstream_flag_rate": ("abstained rows the upstream flags", "abstained rows"),
    "abstain_upstream_precision": ("positives among abstained rows the upstream flags",
                                   "abstained rows the upstream flags"),
    # matched-rate comparison
    "set_positive_rate": ("positives in the set", "rows in the set"),
    "overlap_with_router": ("rows in both the set and the router's state", "rows in the set"),
    "jaccard_with_router": ("rows in both", "rows in either"),
})


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    values: Mapping[str, Value]
    notes: Tuple[str, ...]
    table: Optional[pd.DataFrame] = None
    meta: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __getitem__(self, key: str) -> Value:
        return self.values[key]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for k, v in self.values.items():
            if isinstance(v, Ratio):
                rows.append({"metric": k, "numerator": v.numerator, "denominator": v.denominator,
                             "value": v.as_float(), "status": v.status})
            else:
                rows.append({"metric": k, "numerator": None, "denominator": None,
                             "value": v, "status": "count" if isinstance(v, (int, np.integer))
                             else "value"})
        return pd.DataFrame(rows, columns=["metric", "numerator", "denominator", "value", "status"])


def _r(name: str, num, den) -> Ratio:
    return Ratio(name, int(num), int(den))


def _result(name, values, extra_notes=(), table=None, meta=None) -> DiagnosticResult:
    notes = (HISTORICAL_TEST_EXPOSURE,) + tuple(n for n in extra_notes if n)
    return DiagnosticResult(name, MappingProxyType(dict(values)), notes, table,
                            MappingProxyType(dict(meta or {})))


def _require_view(view) -> RouterOOFView:
    if not isinstance(view, RouterOOFView):
        raise RouterDiagnosticError("expected a RouterOOFView from extract_router_view")
    return view


def _masks(view: RouterOOFView, rows: Optional[np.ndarray] = None):
    keep = np.ones(view.n, bool) if rows is None else rows
    y = np.asarray(view.y)[keep].astype(int)
    st = np.asarray(view.state)[keep]
    return y, st == PASS_1_NOT_SUBSCRIBE, st == PASS_2_SUBSCRIBE, st == ABSTAIN


# ======================================================================
# Router-only
# ======================================================================
def state_counts(view: RouterOOFView, *, by_fold: bool = False) -> pd.DataFrame:
    """Rows, positives, negatives, share and positive rate per state (and fold)."""
    _require_view(view)
    groups = ([("all", np.ones(view.n, bool))] if not by_fold else
              [(int(k), np.asarray(view.fold_id) == k) for k in np.unique(view.fold_id)])
    rows = []
    for g, m in groups:
        y, *_ = _masks(view, m)
        st = np.asarray(view.state)[m]
        for s in ROUTER_STATES:
            sm = st == s
            pr = _r("positive_rate", y[sm].sum(), sm.sum())
            rows.append({"fold": g, "state": s, "n": int(sm.sum()), "positives": int(y[sm].sum()),
                         "negatives": int((y[sm] == 0).sum()), "share": int(sm.sum()) / len(y),
                         "positive_rate": pr.as_float(), "positive_rate_status": pr.status})
    out = pd.DataFrame(rows)
    out["state"] = pd.Categorical(out["state"], categories=list(ROUTER_STATES))
    return out if by_fold else out.drop(columns="fold")


def pass1_metrics(view: RouterOOFView, rows: Optional[np.ndarray] = None) -> dict:
    y, p1, _, _ = _masks(_require_view(view), rows)
    pos, neg = int(y.sum()), int((y == 0).sum())
    return {
        "pass1_n": int(p1.sum()),
        "pass1_positives": int(y[p1].sum()),
        "pass1_rate": _r("pass1_rate", p1.sum(), len(y)),
        "pass1_npv": _r("pass1_npv", (y[p1] == 0).sum(), p1.sum()),
        "pass1_positive_rate": _r("pass1_positive_rate", y[p1].sum(), p1.sum()),
        "pass1_leakage_share": _r("pass1_leakage_share", y[p1].sum(), pos),
        "pass1_negative_recall": _r("pass1_negative_recall", (y[p1] == 0).sum(), neg),
    }


def pass2_metrics(view: RouterOOFView, rows: Optional[np.ndarray] = None) -> dict:
    y, p1, p2, _ = _masks(_require_view(view), rows)
    pos = int(y.sum())
    return {
        "pass2_n": int(p2.sum()),
        "pass2_positives": int(y[p2].sum()),
        "pass2_rate": _r("pass2_rate", p2.sum(), len(y)),
        "pass2_precision": _r("pass2_precision", y[p2].sum(), p2.sum()),
        "pass2_recall": _r("pass2_recall", y[p2].sum(), pos),
        "pass2_recall_eligible": _r("pass2_recall_eligible", y[p2].sum(), pos - int(y[p1].sum())),
    }


def router_metrics(view: RouterOOFView, *, by_fold: bool = False) -> DiagnosticResult:
    """Composition, Pass 1, Pass 2 and abstain metrics (pooled, or per fold in ``table``)."""
    _require_view(view)

    def _one(rows=None):
        y, p1, p2, ab = _masks(view, rows)
        return {
            "n": len(y), "positives": int(y.sum()),
            "coverage": _r("coverage", (~ab).sum(), len(y)),
            "abstain_rate": _r("abstain_rate", ab.sum(), len(y)),
            **pass1_metrics(view, rows), **pass2_metrics(view, rows),
            "abstain_n": int(ab.sum()), "abstain_positives": int(y[ab].sum()),
            "abstain_positive_rate": _r("abstain_positive_rate", y[ab].sum(), ab.sum()),
        }

    values = _one()
    table = None
    if by_fold:
        recs = []
        for k in np.unique(view.fold_id):
            for name, v in _one(np.asarray(view.fold_id) == k).items():
                recs.append({"fold": int(k), "metric": name,
                             "value": v.as_float() if isinstance(v, Ratio) else v,
                             "status": v.status if isinstance(v, Ratio) else "count"})
        table = pd.DataFrame(recs)
    return _result(f"{view.arm} router (outer OOF)", values, table=table,
                   meta={"confidence_kind": view.confidence_kind,
                         "pass2_population": dict(view.pass2_population)})


# ======================================================================
# Router x upstream
# ======================================================================
def _aligned(view: RouterOOFView, flags: UpstreamFlags) -> None:
    if not isinstance(flags, UpstreamFlags):
        raise RouterDiagnosticError("expected UpstreamFlags from resolve_operating_point")
    if not (flags.index.equals(view.index) and np.array_equal(flags.y, view.y)):
        raise RouterDiagnosticError("router view and upstream flags cover different rows / labels")
    # Fold plans may differ legitimately (Stage 1 LR uses 10 folds): recorded in
    # the result's meta (same_fold_plan), not refused.


def router_upstream_metrics(view: RouterOOFView, flags: UpstreamFlags) -> DiagnosticResult:
    """How the router's three states sit against one upstream operating point."""
    _aligned(_require_view(view), flags)
    y = np.asarray(view.y).astype(int)
    f = np.asarray(flags.flags, bool)
    p1, p2, ab = view.pass1, view.pass2, view.abstain
    pos, neg = y == 1, y == 0
    cov = p1 | p2
    values = {
        "upstream_flag_rate": _r("upstream_flag_rate", f.sum(), len(f)),
        "pass1_upstream_flag_rate": _r("pass1_upstream_flag_rate", (p1 & f).sum(), p1.sum()),
        "pass1_fp_suppression": _r("pass1_fp_suppression", (p1 & f & neg).sum(), (f & neg).sum()),
        "pass1_tp_cost": _r("pass1_tp_cost", (p1 & f & pos).sum(), (f & pos).sum()),
        "pass1_leak_upstream_flagged": _r("pass1_leak_upstream_flagged", (p1 & pos & f).sum(),
                                          (p1 & pos).sum()),
        "pass1_leak_upstream_missed": _r("pass1_leak_upstream_missed", (p1 & pos & ~f).sum(),
                                         (p1 & pos).sum()),
        "pass2_upstream_redundancy": _r("pass2_upstream_redundancy", (p2 & f).sum(), p2.sum()),
        "pass2_unique_positive_share": _r("pass2_unique_positive_share", (p2 & pos & ~f).sum(),
                                          (p2 & pos).sum()),
        "pass2_shared_positive_share": _r("pass2_shared_positive_share", (p2 & pos & f).sum(),
                                          (p2 & pos).sum()),
        "pass2_new_false_positive_rate": _r("pass2_new_false_positive_rate",
                                            (p2 & neg & ~f).sum(), p2.sum()),
        "covered_agreement": _r("covered_agreement", (cov & (p2 == f)).sum(), cov.sum()),
        "abstain_upstream_flag_rate": _r("abstain_upstream_flag_rate", (ab & f).sum(), ab.sum()),
        "abstain_upstream_precision": _r("abstain_upstream_precision", (ab & f & pos).sum(),
                                         (ab & f).sum()),
        "pass1_fp_suppressed_n": int((p1 & f & neg).sum()),
        "pass1_tp_cost_n": int((p1 & f & pos).sum()),
        "pass2_unique_positives_n": int((p2 & pos & ~f).sum()),
    }
    return _result(f"{view.arm} router x {flags.name}", values,
                   extra_notes=(f"Upstream operating point: {flags.disclosure}",),
                   meta={"operating_point": flags.name, "selection": flags.selection,
                         "same_fold_plan": bool(np.array_equal(flags.fold_id, view.fold_id))})


def router_router_table(a: RouterOOFView, b: RouterOOFView) -> DiagnosticResult:
    """3 x 3 cross-tabulation of two routers' states on the same training rows."""
    _require_view(a), _require_view(b)
    if not (a.index.equals(b.index) and np.array_equal(a.y, b.y)):
        raise RouterDiagnosticError("the two router views cover different rows / labels")
    y = np.asarray(a.y).astype(int)
    rows = []
    for sa in ROUTER_STATES:
        for sb in ROUTER_STATES:
            m = (np.asarray(a.state) == sa) & (np.asarray(b.state) == sb)
            pr = _r("positive_rate", y[m].sum(), m.sum())
            rows.append({f"{a.arm}_state": sa, f"{b.arm}_state": sb, "n": int(m.sum()),
                         "positives": int(y[m].sum()), "positive_rate": pr.as_float(),
                         "positive_rate_status": pr.status})
    table = pd.DataFrame(rows)
    same = np.asarray(a.state) == np.asarray(b.state)
    values = {"state_agreement": _r("state_agreement", same.sum(), len(same)),
              "same_fold_plan": bool(np.array_equal(a.fold_id, b.fold_id))}
    return _result(f"{a.arm} router x {b.arm} router", values, table=table)


# ======================================================================
# Matched-rate comparison
# ======================================================================
def matched_rate_comparison(view: RouterOOFView, state: str, score: ScoreStream, *,
                            population: str = "all", ranking: Optional[str] = None,
                            by_fold: bool = False) -> DiagnosticResult:
    """
    The router's ``state`` set vs the label-free cut of ``score`` that selects
    the same number of rows (both achievable cuts when the boundary is tied).

    ``population``: "all" rows, or "pass1_remainder" (rows not routed by Pass 1;
    the natural population for a Pass 2 comparison). ``ranking`` defaults to
    "lowest" for Pass 1 and "highest" for Pass 2 (scores oriented to
    P(subscribe)). ``by_fold``: match each fold's count separately.
    """
    _require_view(view)
    if state not in (PASS_1_NOT_SUBSCRIBE, PASS_2_SUBSCRIBE):
        raise RouterDiagnosticError(f"state must be Pass 1 or Pass 2, got {state!r}")
    if not isinstance(score, ScoreStream) or len(score.values) != view.n:
        raise RouterDiagnosticError("score must be a ScoreStream over the view's rows")
    ranking = ranking or ("lowest" if state == PASS_1_NOT_SUBSCRIBE else "highest")
    if population == "all":
        pop = np.ones(view.n, bool)
    elif population == "pass1_remainder":
        pop = ~view.pass1
    else:
        raise RouterDiagnosticError("population must be 'all' or 'pass1_remainder'")
    router_set = (np.asarray(view.state) == state) & pop
    if by_fold:
        fold = np.asarray(view.fold_id)
        target = {int(k): int((router_set & (fold == k)).sum()) for k in np.unique(fold)}
        cut: MatchedCut = matched_rate_cut(score, target_count=target, population_mask=pop,
                                           ranking=ranking, fold_id=fold)
    else:
        cut = matched_rate_cut(score, target_count=int(router_set.sum()), population_mask=pop,
                               ranking=ranking)
    y = np.asarray(view.y).astype(int)
    rows = []
    for label, m in (("router", router_set), ("cut_strict", np.asarray(cut.strict)),
                     ("cut_inclusive", np.asarray(cut.inclusive))):
        rate = _r("set_positive_rate", y[m].sum(), m.sum())
        ov = _r("overlap_with_router", (m & router_set).sum(), m.sum())
        jac = _r("jaccard_with_router", (m & router_set).sum(), (m | router_set).sum())
        rows.append({"set": label, "n": int(m.sum()), "positives": int(y[m].sum()),
                     "positive_rate": rate.as_float(), "positive_rate_status": rate.status,
                     "overlap_with_router": ov.as_float(), "jaccard_with_router": jac.as_float()})
    table = pd.DataFrame(rows)
    values = {"target_count": cut.target_count, "exact": cut.exact, "tie_block": cut.tie_block,
              "strict_count": cut.strict_count, "inclusive_count": cut.inclusive_count,
              "cut_value": cut.cut_value}
    notes = (f"Score space: {score.space} ({score.field}); {SELECTION_RATE_MATCHED}.",
             None if cut.exact else
             "Boundary tie: no label-free cut selects exactly the router's count; both "
             "achievable cuts are reported, neither is preferred.")
    return _result(f"{view.arm} {state} vs matched {score.name} cut", values, notes, table,
                   meta={"cut": cut, "population": population, "ranking": ranking})


def require_score_stream(view: RouterOOFView, which: str, *, equivalence=None,
                         candidate=None) -> ScoreStream:
    """
    A router's continuous score for score-level diagnostics, if the persisted
    artifacts support it. ``which``: "router_transform" | "pass1" | "pass2".
    """
    _require_view(view)
    if which == "router_transform":
        return view.router_score
    if which not in ("pass1", "pass2"):
        raise RouterDiagnosticError(f"which must be router_transform / pass1 / pass2, got {which!r}")
    if view.model_family == "glass_router":
        raise UnsupportedDiagnostic(
            "GLASS routes by rule membership; it has no continuous per-pass score, and the "
            "per-fold rule sets and candidate rules are not persisted.")
    if which == "pass2":
        raise UnsupportedDiagnostic("RF Router Pass 2 outer out-of-fold scores are not persisted.")
    if equivalence is None or candidate is None:
        raise UnsupportedDiagnostic(
            "RF Pass 1 outer score: supply the candidate and a PASSED "
            "check_rf_pass1_score_equivalence report.")
    from .equivalence import verified_rf_pass1_score
    return verified_rf_pass1_score(equivalence, candidate)
