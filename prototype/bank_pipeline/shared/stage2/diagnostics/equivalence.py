"""
shared.stage2.diagnostics.equivalence
=====================================
Is the GLASS-arm RF result's ``oof_proba`` the RF Router's outer Pass 1 score?

The RF Router artifact stores the outer cross-fit's routes but not the Pass 1
scores they were cut from. Each outer fold router routes a held-out row to
Pass 1 iff ``p1 < t1[fold]`` (``rf_router.router.base.apply_cuts``), where
``p1`` comes from a forest refitted on that fold's training rows. The GLASS
cross-fit (``crossfit_stage2``) refits a forest per fold on the same shared
folds, and saves its held-out scores as ``oof_proba``. If the two forests are
the same (same rows, features, parameters, seed and sample weights), the
candidate reproduces every stored Pass 1 route exactly.

The check has two parts, both required:

* structural: same rows, folds and fold fingerprint, router split fingerprint,
  RF parameters (the six keys both forest builders read) and seed, and one
  finite ``t1`` per outer fold;
* row-wise: ``(candidate < t1[fold]) == (state == PASS_1)`` on EVERY training
  row, and no Pass 2 row scores below its fold's ``t1`` (Pass 1 precedence).

Exactly zero disagreements are required (Step 1C decision 6). Disagreements
within ``BOUNDARY_ATOL`` of ``t1`` are classified as boundary cases (forests
fitted with ``n_jobs != 1`` can differ in the last ULP; see
``rf_router.forest.build_pipe``) and documented, but they still fail the
check: the tolerance classifies, it never accepts. A failed check marks the RF
Pass 1 score stream unsupported; decision-level diagnostics are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

import numpy as np
import pandas as pd

from .schema import (
    PASS_1_NOT_SUBSCRIBE,
    PASS_2_SUBSCRIBE,
    RFPass1ScoreCandidate,
    RouterDiagnosticError,
    RouterOOFView,
    ScoreStream,
    UnsupportedDiagnostic,
)

__all__ = ["BOUNDARY_ATOL", "RF_PARAM_KEYS", "EquivalenceReport",
           "check_rf_pass1_score_equivalence", "verified_rf_pass1_score"]

#: Classifies a disagreement as a boundary case. Never used to accept one.
BOUNDARY_ATOL = 1e-12
#: The parameters both forest builders read (rf_training._build_pipe /
#: _make_sample_weights and rf_router.forest.build_pipe / make_sample_weights).
RF_PARAM_KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "max_features",
                 "minority_weight", "class_weight")


@dataclass(frozen=True)
class EquivalenceReport:
    passed: bool
    structural: Mapping[str, Tuple[bool, str]]
    n_rows: int
    n_disagreements: int
    n_boundary_disagreements: int
    pass2_precedence_violations: int
    disagreements: pd.DataFrame
    fold_summary: pd.DataFrame
    candidate_digest: str
    conclusion: str

    @property
    def structural_passed(self) -> bool:
        return all(ok for ok, _ in self.structural.values())

    def structural_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": k, "passed": ok, "detail": d}
                             for k, (ok, d) in self.structural.items()])


def _canon(v):
    if isinstance(v, (bool, np.bool_)) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return v


def _params_equal(a: Mapping, b: Mapping) -> Tuple[bool, str]:
    diffs = [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in RF_PARAM_KEYS
             if _canon(a.get(k, "<absent>")) != _canon(b.get(k, "<absent>"))]
    return (not diffs, "; ".join(diffs) or "the six forest parameters match")


def check_rf_pass1_score_equivalence(rf_view: RouterOOFView,
                                     candidate: RFPass1ScoreCandidate) -> EquivalenceReport:
    if not isinstance(rf_view, RouterOOFView) or not isinstance(candidate, RFPass1ScoreCandidate):
        raise RouterDiagnosticError("expected an RF RouterOOFView and an RFPass1ScoreCandidate")
    meta, proto = rf_view.metadata, rf_view.metadata.get("oof_protocol", {})
    rf_params = meta.get("rf_params") or {}
    seed = meta.get("random_state")
    ft = rf_view.fold_thresholds or {}
    s: dict = {}
    s["view is the two-pass RF Router"] = (
        rf_view.model_family == "rf_router" and rf_view.router_mode == "two_pass",
        f"{rf_view.model_family} / {rf_view.router_mode}")
    s["same training rows, same order"] = (candidate.index.equals(rf_view.index), "")
    same_fold = np.array_equal(candidate.fold_id, rf_view.fold_id)
    s["same outer fold ids"] = (same_fold, "")
    s["same fold fingerprint"] = (candidate.fold_fingerprint == proto.get("fold_fingerprint"),
                                  f"{candidate.fold_fingerprint} vs {proto.get('fold_fingerprint')}")
    s["same router split fingerprint"] = (
        candidate.split_fingerprint is not None
        and candidate.split_fingerprint == meta.get("split_fingerprint"),
        f"{candidate.split_fingerprint} vs {meta.get('split_fingerprint')}")
    s["same forest parameters"] = _params_equal(dict(candidate.params), dict(rf_params))
    s["same forest seed"] = (seed is not None and int(candidate.random_state) == int(seed)
                             and int(proto.get("random_state", seed)) == int(seed),
                             f"candidate {candidate.random_state}, router config {seed}, "
                             f"router protocol {proto.get('random_state')}")
    folds = sorted(int(k) for k in np.unique(rf_view.fold_id))
    s["one finite t1 per outer fold"] = (
        sorted(ft) == folds and all(np.isfinite(ft[k]["t1"]) for k in ft), f"folds {sorted(ft)}")

    n = rf_view.n
    rowwise_possible = (s["same training rows, same order"][0] and same_fold
                        and s["one finite t1 per outer fold"][0])
    cols = ["row", "fold", "score", "t1", "score_minus_t1", "state", "boundary"]
    if rowwise_possible:
        fold = np.asarray(rf_view.fold_id)
        t1 = np.array([ft[int(k)]["t1"] for k in fold], dtype=float)
        score = np.asarray(candidate.values, dtype=float)
        state = np.asarray(rf_view.state)
        predicted = score < t1
        actual = state == PASS_1_NOT_SUBSCRIBE
        bad = predicted != actual
        diff = score - t1
        boundary = np.abs(diff) <= BOUNDARY_ATOL
        dis = pd.DataFrame({"row": rf_view.index[bad], "fold": fold[bad], "score": score[bad],
                            "t1": t1[bad], "score_minus_t1": diff[bad], "state": state[bad],
                            "boundary": boundary[bad]}, columns=cols)
        precedence = int(((state == PASS_2_SUBSCRIBE) & (score < t1)).sum())
        summary = pd.DataFrame([{
            "fold": k, "n": int((fold == k).sum()), "t1": ft[k]["t1"],
            "pass1_routed": int((actual & (fold == k)).sum()),
            "pass1_reproduced": int((predicted & (fold == k)).sum()),
            "disagreements": int((bad & (fold == k)).sum()),
            "closest_abs_score_minus_t1": float(np.min(np.abs(diff[fold == k]))),
        } for k in folds])
        n_bad, n_boundary = int(bad.sum()), int((bad & boundary).sum())
    else:
        dis, summary = pd.DataFrame(columns=cols), pd.DataFrame()
        precedence, n_bad, n_boundary = -1, -1, -1
    structural_ok = all(ok for ok, _ in s.values())
    passed = bool(structural_ok and rowwise_possible and n_bad == 0 and precedence == 0)
    if passed:
        conclusion = (f"EQUIVALENT: the candidate reproduces every stored RF Pass 1 route "
                      f"({n:,} rows, 0 disagreements, 0 Pass 2 rows below t1).")
    elif not rowwise_possible:
        conclusion = "UNSUPPORTED: rows, folds or fold thresholds do not line up; no row-wise check."
    else:
        failed = [k for k, (ok, _) in s.items() if not ok]
        conclusion = ("UNSUPPORTED: " + "; ".join(
            ([f"structural checks failed: {failed}"] if failed else [])
            + ([f"{n_bad} row disagreement(s) ({n_boundary} within {BOUNDARY_ATOL:g} of t1; "
                "boundary cases are documented, not accepted)"] if n_bad else [])
            + ([f"{precedence} Pass 2 row(s) score below t1"] if precedence else []))
            + ". The RF Pass 1 score stream stays unsupported; decision-level diagnostics are "
              "unaffected.")
    return EquivalenceReport(passed=passed, structural=MappingProxyType(s), n_rows=n,
                             n_disagreements=n_bad, n_boundary_disagreements=n_boundary,
                             pass2_precedence_violations=precedence, disagreements=dis,
                             fold_summary=summary, candidate_digest=candidate.digest,
                             conclusion=conclusion)


def verified_rf_pass1_score(report: EquivalenceReport,
                            candidate: RFPass1ScoreCandidate) -> ScoreStream:
    """The candidate as an RF Pass 1 score stream, only under a PASSED report for it."""
    if not isinstance(report, EquivalenceReport) or not report.passed:
        raise UnsupportedDiagnostic(
            "RF Pass 1 outer score is unsupported: the equivalence check did not pass"
            + (f" ({report.conclusion})" if isinstance(report, EquivalenceReport) else ""))
    if report.candidate_digest != candidate.digest:
        raise RouterDiagnosticError("the equivalence report was produced for another candidate")
    return ScoreStream(
        "rf.pass1_outer_score", candidate.values, "weighted_forest_score",
        "GLASS-arm rf_result oof_proba",
        f"verified equal to the RF Router's outer Pass 1 routing ({report.n_rows:,} rows, "
        "0 disagreements); class-weighted forest score, not a probability")
