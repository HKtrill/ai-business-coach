"""
shared.stage2.diagnostics.trace
===============================
Three-state router trace (PR 37, Commit 2, Step 2A).

One row per training row: the GLASS Router's and the RF Router's historical
outer out-of-fold routing state, the row's label and its outer fold, built
from the two ``RouterOOFView``s (``boundary.extract_router_view``) and checked
against the ``TrainReference`` they were extracted against. Nothing here reads
an artifact, fits, tunes, writes files or changes a router, threshold or
operating constraint. Every rate is a ``metrics.Ratio`` with an explicit
denominator; an empty denominator is undefined, never 0.

    build_router_trace           checks T1-T7, then a read-only RouterTrace
    routing_policy_audit         what the persisted evidence can and cannot show
                                 about Pass 1 precedence and Pass 2 eligibility
    state_distribution           per arm, fold and state: counts, shares and
                                 class composition
    routing_rates                Pass 1 / Pass 2 / total coverage and abstention
    compare_state_distributions  observational GLASS vs RF differences

Trace checks
------------
    T1  inputs: a TrainReference and two RouterOOFViews of the expected arm and
        model family (GLASS "glass" / "glass_router", RF "blackbox" / "rf_router")
    T2  rows: index = the training index (same order), unique; every per-row
        field has one entry per training row; both arms cover the same rows
    T3  labels: equal to the training labels, binary; equal in both arms
    T4  states: each in ROUTER_STATES; exactly one of Pass 1 / Pass 2 / Abstain
        per row
    T5  each state follows from its stored token: decision tokens are exactly
        "pass1" / "pass2" / "uncertain", and state and source_pred equal
        SOURCE_DECISIONS[token]
    T6  folds: integer ids exactly 0..4, every fold non-empty; identical ids
        and declared fold fingerprints in both arms
    T7  reconciliation (through ``metrics.state_counts``): per-fold state
        counts and positives sum to the pooled values, which sum to the
        training rows and positives; each fold's states sum to its rows

T4 guards against malformed views. One stored token per row makes the states
exclusive by construction, so T4 is not evidence that a router behaves
correctly. ``RouterOOFView`` is a public frozen dataclass that can be built,
or ``dataclasses.replace``d, without the extractor, so the trace re-asserts
what it relies on; it does not repeat the extractor's transform, confidence or
fingerprint checks. Checks whose prerequisites failed are recorded as failed
("not evaluated"), so a malformed input never surfaces as an IndexError.

Decision time vs fit time
-------------------------
Both routers give Pass 1 precedence at DECISION time: GLASS
``GlassRouterPipeline.predict`` lets a Pass 2 rule claim only rows still
"uncertain" after every Pass 1 rule; RF ``apply_cuts`` flags Pass 2 only among
rows with ``p1 >= t1``. The historical ``pass2_population`` (GLASS "full", RF
the OOF Pass 1 remainder) is the population Pass 2 was FITTED and selected on,
not the rows it may claim. The audit checks the shared decision-time rule and
records the two fit-time declarations; it evaluates neither population policy
(Commit 3, F17).

Audit statuses
--------------
    verified_from_artifacts  checked on persisted outputs with zero violations
    evidence_failed          the artifact check ran but did not pass; the claim
                             is neither verified nor shown to be violated (the
                             evidence itself is in doubt)
    declared                 an artifact declaration, consistency-checked by the
                             extractor; the behaviour itself is not re-checked
    code_level_only          holds by construction in the router code and its
                             unit tests; cannot be re-checked on stored routes
    not_verifiable           the evidence was not persisted, or not supplied

An RF decision-time claim is ``verified_from_artifacts`` only when the RF Pass 1
score candidate passes ``check_rf_pass1_score_equivalence`` in full
(structural, row alignment, fingerprints, zero row disagreements, zero Pass 2
rows below ``t1``) and was built against this trace's training reference. A
failed check cannot separate "the routing broke precedence" from "the
candidate is not the routing score", so it is reported as ``evidence_failed``,
never as a violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from shared.validation import ValidationReport

from .boundary import ROUTER_N_FOLDS
from .equivalence import check_rf_pass1_score_equivalence
from .metrics import (
    DiagnosticResult,
    Ratio,
    _result,
    pass1_metrics,
    pass2_metrics,
    state_counts,
)
from .schema import (
    ABSTAIN,
    NOT_RECORDED,
    PASS_1_NOT_SUBSCRIBE,
    PASS_2_SUBSCRIBE,
    ROUTER_STATES,
    SOURCE_DECISIONS,
    VALIDATED,
    ProvenanceItem,
    RFPass1ScoreCandidate,
    RouterDiagnosticError,
    RouterOOFView,
    TrainReference,
    _frozen,
)

__all__ = [
    "GLASS_ARM", "RF_ARM", "TRACE_ARMS", "ARM_LABELS", "ARM_FAMILIES",
    "VERIFIED_FROM_ARTIFACTS", "EVIDENCE_FAILED", "DECLARED", "CODE_LEVEL_ONLY",
    "NOT_VERIFIABLE", "AUDIT_STATUSES", "AUDIT_STATUS_DEFINITIONS", "AUDIT_CLAIMS",
    "TraceCheckError", "RouterTrace", "build_router_trace", "routing_policy_audit",
    "state_distribution", "routing_rates", "compare_state_distributions",
]

# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------
GLASS_ARM = "glass"
RF_ARM = "blackbox"
TRACE_ARMS: Tuple[str, str] = (GLASS_ARM, RF_ARM)
ARM_LABELS: Mapping[str, str] = MappingProxyType({GLASS_ARM: "GLASS Router", RF_ARM: "RF Router"})
ARM_FAMILIES: Mapping[str, str] = MappingProxyType({GLASS_ARM: "glass_router",
                                                    RF_ARM: "rf_router"})

VERIFIED_FROM_ARTIFACTS = "verified_from_artifacts"
EVIDENCE_FAILED = "evidence_failed"
DECLARED = "declared"
CODE_LEVEL_ONLY = "code_level_only"
NOT_VERIFIABLE = "not_verifiable"
AUDIT_STATUSES = (VERIFIED_FROM_ARTIFACTS, EVIDENCE_FAILED, DECLARED, CODE_LEVEL_ONLY,
                  NOT_VERIFIABLE)
AUDIT_STATUS_DEFINITIONS: Mapping[str, str] = MappingProxyType({
    VERIFIED_FROM_ARTIFACTS: "checked on persisted outputs with zero violations",
    EVIDENCE_FAILED: "the artifact check ran but did not pass; neither verified nor shown "
                     "to be violated",
    DECLARED: "an artifact declaration, consistency-checked by the extractor; the behaviour "
              "itself is not re-checked",
    CODE_LEVEL_ONLY: "holds by construction in the router code and its unit tests; cannot be "
                     "re-checked on the stored routes",
    NOT_VERIFIABLE: "the evidence was not persisted, or not supplied",
})
_BASIS = MappingProxyType({VERIFIED_FROM_ARTIFACTS: "artifact", EVIDENCE_FAILED: "artifact",
                           DECLARED: "declaration", CODE_LEVEL_ONLY: "code",
                           NOT_VERIFIABLE: "none"})

#: claim id -> what the claim asserts (one audit row per claim and arm).
AUDIT_CLAIMS: Mapping[str, str] = MappingProxyType({
    "states.one_per_row":
        "every training row has exactly one stored routing state (exclusive, exhaustive)",
    "decision.pass1_precedence":
        "Pass 1 claims every row it qualifies for before Pass 2 is considered",
    "decision.pass2_eligibility":
        "Pass 2 claims only rows Pass 1 left (decision-time eligibility, shared by both routers)",
    "decision.pass2_condition":
        "every Pass 2 row met its fold's Pass 2 condition (GLASS rule match / RF p2 > t2)",
    "decision.abstain_not_pass1":
        "no abstained row qualified for Pass 1",
    "decision.abstain_not_pass2":
        "no abstained row met its fold's Pass 2 condition",
    "fit.pass2_population_declared":
        "the population Pass 2 was fitted and selected on, as declared (fit-time policy)",
    "fit.pass2_population_realised":
        "the rows Pass 2 was actually fitted on in each outer fold",
})

_STATE_LABELS = MappingProxyType({PASS_1_NOT_SUBSCRIBE: "Pass 1", PASS_2_SUBSCRIBE: "Pass 2",
                                  ABSTAIN: "Abstain"})
#: The registered rate that is each state's share of the rows.
_STATE_SHARE = MappingProxyType({PASS_1_NOT_SUBSCRIBE: "pass1_rate",
                                 PASS_2_SUBSCRIBE: "pass2_rate", ABSTAIN: "abstain_rate"})
#: routing_rates: registered metric -> label (the requested definitions).
_RATE_LABELS = MappingProxyType({
    "pass1_rate": "Pass 1 coverage (Pass 1 rows / all rows)",
    "pass2_rate": "Pass 2 coverage (Pass 2 rows / all rows)",
    "coverage": "Total coverage ((Pass 1 + Pass 2 rows) / all rows)",
    "abstain_rate": "Abstention rate (abstained rows / all rows)",
})
_ROW_FIELDS = ("y", "fold_id", "state", "source_decision", "source_pred")
_SECTION = "Stage 2 router trace"
_CROSS = f"{GLASS_ARM} x {RF_ARM}"
_NOT_EVALUATED = "not evaluated: a prerequisite check failed"

_GLASS_CODE_EVIDENCE = (
    "GlassRouterPipeline.predict starts every row 'uncertain', lets each Pass 1 rule claim "
    "only rows still 'uncertain', then each Pass 2 rule likewise; crossfit_stage2 stores each "
    "outer fold's predict() output unchanged. Pinned by tests/stage2/test_glass_router_routing.py "
    "(test_pass1_rows_are_never_rerouted_by_pass2, "
    "test_states_are_exclusive_exhaustive_and_consistent) and "
    "tests/stage2/test_glass_router_crossfit.py "
    "(test_oof_block_holds_each_fold_models_output_in_training_row_order). Not re-checkable on "
    "the stored routes: the per-fold rule sets and fold-engineered features were not persisted.")


class TraceCheckError(RouterDiagnosticError):
    """One or more router trace checks failed; ``report`` lists every check."""

    def __init__(self, message: str, report: pd.DataFrame):
        super().__init__(message)
        self.report = report


# ----------------------------------------------------------------------
# The trace
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RouterTrace:
    """
    The two routers' outer OOF states on the training rows. Built only by
    ``build_router_trace`` after every check passed. Arrays are private
    read-only copies; ``views`` holds the source views unchanged; ``checks``
    is the full check ledger (read-only rows); ``frame()`` / ``check_frame()``
    return fresh DataFrames on every call.
    """

    index: pd.Index
    y: np.ndarray
    fold_id: np.ndarray
    n_folds: int
    views: Mapping[str, RouterOOFView]
    checks: Tuple[Mapping[str, Any], ...]
    provenance: Tuple[ProvenanceItem, ...]
    router_split_fingerprint: str
    train_label_fingerprint: str
    stage_io_split_fingerprint: Optional[str] = None
    stage_io_attestation: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def n_positives(self) -> int:
        return int(np.asarray(self.y).sum())

    @property
    def arms(self) -> Tuple[str, str]:
        return TRACE_ARMS

    def frame(self) -> pd.DataFrame:
        """One row per training row (training index, X_train order)."""
        cols: Dict[str, Any] = {"y": np.array(self.y, copy=True),
                                "fold_id": np.array(self.fold_id, copy=True)}
        for arm in TRACE_ARMS:
            v = self.views[arm]
            cols[f"{arm}_state"] = pd.Categorical(np.array(v.state, dtype=object, copy=True),
                                                  categories=list(ROUTER_STATES))
            cols[f"{arm}_source_decision"] = np.array(v.source_decision, dtype=object, copy=True)
            cols[f"{arm}_source_pred"] = np.array(v.source_pred, copy=True)
        return pd.DataFrame(cols, index=self.index.copy())

    def check_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dict(r) for r in self.checks],
                            columns=["section", "stage", "check", "passed", "detail", "error"])

    def provenance_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"name": p.name, "status": p.status, "detail": p.detail}
                             for p in self.provenance], columns=["name", "status", "detail"])


def _require_trace(trace) -> RouterTrace:
    if not isinstance(trace, RouterTrace):
        raise RouterDiagnosticError("expected a RouterTrace from build_router_trace")
    return trace


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------
class _Checks:
    """``ValidationReport`` with prerequisites and crash-proof detail strings."""

    def __init__(self) -> None:
        self.rep = ValidationReport()

    def __call__(self, stage: str, name: str, fn: Callable[[], Any],
                 detail: Any = "", *, requires: bool = True) -> bool:
        if not requires:
            return self.rep.check(_SECTION, stage, name, lambda: False, _NOT_EVALUATED)
        if callable(detail):
            try:
                detail = str(detail())
            except Exception:  # noqa: BLE001 - a detail string must never mask the check
                detail = ""
        return self.rep.check(_SECTION, stage, name, fn, str(detail))


def _arr(view, attr) -> np.ndarray:
    return np.asarray(getattr(view, attr))


def _bad_states(view) -> List[str]:
    st = pd.Series(_arr(view, "state"), dtype=object)
    return sorted({repr(s) for s in st[~st.isin(ROUTER_STATES)]})


def _bad_tokens(view) -> List[str]:
    dec = _arr(view, "source_decision").ravel()
    return sorted({repr(t) for t in dec if not (isinstance(t, str) and t in SOURCE_DECISIONS)})


def _expected_from_tokens(view):
    dec = _arr(view, "source_decision")
    st = np.array([SOURCE_DECISIONS[t][0] for t in dec], dtype=object)
    pred = np.array([SOURCE_DECISIONS[t][1] for t in dec], dtype=float)
    return st, pred


def _arm_checks(chk: _Checks, view, arm: str, ref: TrainReference) -> Dict[str, bool]:
    n = ref.n_train
    family = ARM_FAMILIES[arm]
    is_view = chk(arm, "T1 input is a RouterOOFView (extract_router_view)",
                  lambda: isinstance(view, RouterOOFView), type(view).__name__)
    chk(arm, f"T1 arm is {arm!r} and model family is {family!r}",
        lambda: view.arm == arm and view.model_family == family,
        lambda: f"{view.arm!r} / {view.model_family!r}", requires=is_view)
    index_ok = chk(arm, "T2 row index = training index (same rows, same order)",
                   lambda: isinstance(view.index, pd.Index) and view.index.equals(ref.index_train),
                   lambda: f"{len(view.index):,} rows vs {n:,} training rows", requires=is_view)
    chk(arm, "T2 row index is unique", lambda: view.index.is_unique, requires=is_view)
    shapes_ok = chk(arm, "T2 every per-row field has one entry per training row",
                    lambda: all(_arr(view, a).shape == (n,) for a in _ROW_FIELDS),
                    lambda: ", ".join(f"{a} {_arr(view, a).shape}" for a in _ROW_FIELDS),
                    requires=is_view)
    rows_ok = is_view and index_ok and shapes_ok
    labels_ok = chk(arm, "T3 labels = training labels",
                    lambda: np.array_equal(_arr(view, "y"), ref.y_train), requires=rows_ok)
    binary_ok = chk(arm, "T3 labels are binary 0 / 1",
                    lambda: bool(np.isin(_arr(view, "y"), (0, 1)).all()), requires=rows_ok)
    states_ok = chk(arm, "T4 every state is one of ROUTER_STATES",
                    lambda: not _bad_states(view),
                    lambda: f"unexpected {_bad_states(view)[:5]}" if _bad_states(view)
                    else f"{n:,} rows", requires=rows_ok)
    one_ok = chk(arm, "T4 exactly one of Pass 1 / Pass 2 / Abstain per row",
                 lambda: bool(np.all((_arr(view, "state") == PASS_1_NOT_SUBSCRIBE).astype(int)
                                     + (_arr(view, "state") == PASS_2_SUBSCRIBE)
                                     + (_arr(view, "state") == ABSTAIN) == 1)),
                 requires=rows_ok)
    tokens_ok = chk(arm, "T5 decision tokens are exactly 'pass1' / 'pass2' / 'uncertain'",
                    lambda: not _bad_tokens(view),
                    lambda: f"unexpected {_bad_tokens(view)[:5]}" if _bad_tokens(view)
                    else f"{n:,} rows", requires=rows_ok)
    state_tok_ok = chk(
        arm, "T5 state = SOURCE_DECISIONS[token] on every row",
        lambda: np.array_equal(_arr(view, "state").astype(object), _expected_from_tokens(view)[0]),
        lambda: f"{int((_arr(view, 'state') != _expected_from_tokens(view)[0]).sum())} row(s) "
                "differ", requires=rows_ok and tokens_ok)
    pred_tok_ok = chk(
        arm, "T5 source_pred = SOURCE_DECISIONS[token] pred on every row",
        lambda: _arr(view, "source_pred").dtype.kind in "biuf" and np.array_equal(
            _arr(view, "source_pred").astype(float), _expected_from_tokens(view)[1]),
        lambda: f"dtype {_arr(view, 'source_pred').dtype}", requires=rows_ok and tokens_ok)
    folds_ok = chk(arm, f"T6 fold ids are integers 0..{ROUTER_N_FOLDS - 1}, every fold non-empty",
                   lambda: _arr(view, "fold_id").dtype.kind in "iu" and sorted(
                       int(k) for k in np.unique(_arr(view, "fold_id")))
                   == list(range(ROUTER_N_FOLDS)),
                   lambda: f"fold ids {sorted(np.unique(_arr(view, 'fold_id')).tolist())[:8]}",
                   requires=rows_ok)
    valid = all((rows_ok, labels_ok, binary_ok, states_ok, one_ok, tokens_ok, state_tok_ok,
                 pred_tok_ok, folds_ok))
    _reconcile(chk, view, arm, ref, requires=valid)
    return {"view": is_view, "rows": rows_ok, "valid": valid}


def _reconcile(chk: _Checks, view, arm: str, ref: TrainReference, *, requires: bool) -> None:
    """T7 through metrics.state_counts, the helper the distributions reuse."""
    n, pos = ref.n_train, int(ref.y_train.sum())
    pooled = by_fold = None
    if requires:
        pooled = state_counts(view).set_index("state")
        by_fold = state_counts(view, by_fold=True)
    chk(arm, "T7 pooled state counts sum to the training rows",
        lambda: int(pooled["n"].sum()) == n, lambda: f"{int(pooled['n'].sum()):,} vs {n:,}",
        requires=requires)
    chk(arm, "T7 pooled state positives sum to the training positives",
        lambda: int(pooled["positives"].sum()) == pos,
        lambda: f"{int(pooled['positives'].sum()):,} vs {pos:,}", requires=requires)

    def folds_sum_to_pooled() -> bool:
        summed = by_fold.groupby("state", observed=False)[["n", "positives", "negatives"]].sum()
        return all(int(summed.loc[s, c]) == int(pooled.loc[s, c])
                   for s in ROUTER_STATES for c in ("n", "positives", "negatives"))

    chk(arm, "T7 per-fold state counts, positives and negatives sum to the pooled values",
        folds_sum_to_pooled, requires=requires)

    def fold_rows() -> bool:
        fold = _arr(view, "fold_id")
        per = by_fold.groupby("fold")["n"].sum()
        return all(int(per.loc[k]) == int((fold == k).sum()) for k in range(ROUTER_N_FOLDS))

    chk(arm, "T7 each fold's Pass 1 + Pass 2 + Abstain = the fold's rows", fold_rows,
        requires=requires)


def _fold_fp(view) -> Optional[str]:
    proto = view.metadata.get("oof_protocol", {}) if isinstance(view.metadata, Mapping) else {}
    return proto.get("fold_fingerprint") if isinstance(proto, Mapping) else None


def build_router_trace(glass: RouterOOFView, rf: RouterOOFView,
                       ref: TrainReference) -> RouterTrace:
    """
    The two routers' outer OOF states on the training rows of ``ref``.

    ``glass``: ``extract_router_view(..., expected_arm="glass")``; ``rf``:
    ``extract_router_view(..., expected_arm="blackbox")``, both against ``ref``.
    Every check T1-T7 is run; if any fails, ``TraceCheckError`` is raised with
    the whole ledger (``.report``).
    """
    chk = _Checks()
    ref_ok = chk("reference", "T1 reference is a TrainReference (build_train_reference)",
                 lambda: isinstance(ref, TrainReference), type(ref).__name__)
    if ref_ok:
        g = _arm_checks(chk, glass, GLASS_ARM, ref)
        r = _arm_checks(chk, rf, RF_ARM, ref)
        both = g["rows"] and r["rows"]
        chk(_CROSS, "T2 both arms cover the same training rows in the same order",
            lambda: glass.index.equals(rf.index), requires=both)
        chk(_CROSS, "T3 both arms carry the same labels",
            lambda: np.array_equal(_arr(glass, "y"), _arr(rf, "y")), requires=both)
        chk(_CROSS, "T6 both arms use the same outer fold ids (shared five-fold plan)",
            lambda: np.array_equal(_arr(glass, "fold_id"), _arr(rf, "fold_id")),
            lambda: f"{int((_arr(glass, 'fold_id') != _arr(rf, 'fold_id')).sum())} row(s) differ",
            requires=both)
        chk(_CROSS, "T6 both arms declare the same fold fingerprint",
            lambda: _fold_fp(glass) is not None and _fold_fp(glass) == _fold_fp(rf),
            lambda: f"{_fold_fp(glass)} vs {_fold_fp(rf)}", requires=g["view"] and r["view"])
    report = chk.rep.frame()
    bad = report[~report["passed"]]
    if len(bad):
        lines = [f"  - [{x.stage}] {x.check}: {x.error or x.detail}" for x in bad.itertuples()]
        raise TraceCheckError(f"router trace: {len(bad)} check(s) failed:\n" + "\n".join(lines),
                              report)

    fold = _arr(glass, "fold_id")
    provenance = [
        ProvenanceItem("trace.rows", VALIDATED,
                       f"{ref.n_train:,} training rows; both arms indexed by the training index "
                       "in X_train order"),
        ProvenanceItem("trace.states", VALIDATED,
                       "one of PASS_1_NOT_SUBSCRIBE / PASS_2_SUBSCRIBE / ABSTAIN per row and arm, "
                       "from the stored tokens via SOURCE_DECISIONS"),
        ProvenanceItem("trace.folds", VALIDATED,
                       f"{ROUTER_N_FOLDS} outer folds shared by both arms "
                       f"(fold fingerprint {_fold_fp(glass)})"),
        ProvenanceItem("trace.reconciliation", VALIDATED,
                       "per-fold and pooled state counts and positives reconcile"),
    ]
    provenance += [ProvenanceItem(f"{arm}.{p.name}", p.status, p.detail)
                   for arm, v in ((GLASS_ARM, glass), (RF_ARM, rf)) for p in v.provenance]
    return RouterTrace(
        index=ref.index_train, y=_frozen(ref.y_train, int), fold_id=_frozen(fold, int),
        n_folds=ROUTER_N_FOLDS, views=MappingProxyType({GLASS_ARM: glass, RF_ARM: rf}),
        checks=tuple(MappingProxyType(rec) for rec in report.to_dict("records")),
        provenance=tuple(provenance), router_split_fingerprint=ref.router_split_fingerprint,
        train_label_fingerprint=ref.train_label_fingerprint,
        stage_io_split_fingerprint=ref.stage_io_split_fingerprint,
        stage_io_attestation=ref.stage_io_attestation)


# ----------------------------------------------------------------------
# Routing-policy audit
# ----------------------------------------------------------------------
def _rf_pass1_evidence(trace: RouterTrace, candidate) -> Dict[str, Any]:
    if candidate is None:
        return {"status": NOT_VERIFIABLE, "report": None,
                "why": "No RF Pass 1 score candidate supplied; the RF Router artifact does not "
                       "store its outer Pass 1 scores (see check_rf_pass1_score_equivalence)."}
    if not isinstance(candidate, RFPass1ScoreCandidate):
        raise RouterDiagnosticError(
            "rf_candidate must be an RFPass1ScoreCandidate from extract_rf_pass1_candidate")
    identity = {
        "candidate training-label fingerprint = trace":
            candidate.train_label_fingerprint == trace.train_label_fingerprint,
        "candidate router split fingerprint = trace":
            candidate.split_fingerprint == trace.router_split_fingerprint,
    }
    report = check_rf_pass1_score_equivalence(trace.views[RF_ARM], candidate)
    ok = (all(identity.values()) and report.passed and report.structural_passed
          and report.n_rows == trace.n and report.n_disagreements == 0
          and report.n_boundary_disagreements == 0 and report.pass2_precedence_violations == 0)
    failed = [k for k, v in identity.items() if not v]
    why = report.conclusion + (f" Identity checks failed: {failed}." if failed else "")
    return {"status": VERIFIED_FROM_ARTIFACTS if ok else EVIDENCE_FAILED, "report": report,
            "why": why}


def _population_row(view: RouterOOFView, arm: str) -> Tuple[str, str]:
    pop = dict(view.pass2_population)
    prov = {p.name: p.status for p in view.provenance}
    if prov.get("pass2_population") != VALIDATED:
        return NOT_VERIFIABLE, "the Pass 2 population declaration was not validated by the extractor"
    norm = pop.get("normalised")
    if arm == GLASS_ARM:
        how = ("Pass 2 rules are generated and selected on the fold's whole training population "
               "and applied only to rows Pass 1 left" if norm == "full" else
               f"Pass 2 is fitted on the {norm!r} population")
        text = (f"declared {pop.get('cross_fit')!r} for the cross-fit and "
                f"{pop.get('full_train')!r} for the full-train fit (normalised {norm!r}; "
                f"equality checked by extract_router_view). {how}.")
    else:
        how = ("each outer fold's Pass 2 forest is fitted on that fold's inner OOF Pass 1 "
               "remainder" if norm == "oof_remainder" else f"Pass 2 population {norm!r}")
        text = (f"declared {pop.get('cross_fit')!r} for the cross-fit (normalised {norm!r}; "
                f"exact declaration checked by extract_router_view). {how}.")
    return DECLARED, text


def routing_policy_audit(trace: RouterTrace, *,
                         rf_candidate: Optional[RFPass1ScoreCandidate] = None) -> pd.DataFrame:
    """
    One row per claim and arm: status (``AUDIT_STATUSES``), basis, violations
    (0 when verified, otherwise NA) and the evidence.

    ``rf_candidate``: the GLASS-arm RF result's ``oof_proba`` candidate
    (``extract_rf_pass1_candidate``); ``check_rf_pass1_score_equivalence`` is
    run here against the trace's RF view. Without it the RF decision-time
    claims are ``not_verifiable``.
    """
    _require_trace(trace)
    glass, rf = trace.views[GLASS_ARM], trace.views[RF_ARM]
    ev = _rf_pass1_evidence(trace, rf_candidate)
    rows: List[Dict[str, Any]] = []

    def add(arm, claim, status, evidence, violations=None):
        rows.append({"arm": arm, "router": ARM_LABELS[arm], "claim": claim,
                     "description": AUDIT_CLAIMS[claim], "status": status,
                     "basis": _BASIS[status],
                     "violations": 0 if status == VERIFIED_FROM_ARTIFACTS else violations,
                     "evidence": evidence})

    states_text = (f"trace checks T2-T7 passed on the stored decision tokens ({trace.n:,} rows). "
                   "Exclusivity holds by construction (one stored token per row): a "
                   "data-integrity check, not evidence of router correctness.")
    t2 = {k: v["t2"] for k, v in (rf.fold_thresholds or {}).items()}
    t2_text = ", ".join(f"fold {k}: {t:.6g}" for k, t in sorted(t2.items())) or "not recorded"
    rf_rule = {
        "decision.pass1_precedence":
            "Every row the verified Pass 1 score puts below its fold's t1 is stored Pass 1, so "
            "no Pass 1-qualifying row reached Pass 2 or abstention.",
        "decision.pass2_eligibility":
            "No stored Pass 2 row scores below its fold's t1 (pass2_precedence_violations = 0).",
        "decision.abstain_not_pass1":
            "Every abstained row scores at or above its fold's t1.",
    }
    for arm in TRACE_ARMS:
        add(arm, "states.one_per_row", VERIFIED_FROM_ARTIFACTS, states_text)
        for claim in ("decision.pass1_precedence", "decision.pass2_eligibility",
                      "decision.abstain_not_pass1"):
            if arm == GLASS_ARM:
                add(arm, claim, CODE_LEVEL_ONLY, _GLASS_CODE_EVIDENCE)
            elif ev["status"] == VERIFIED_FROM_ARTIFACTS:
                add(arm, claim, VERIFIED_FROM_ARTIFACTS, f"{ev['why']} {rf_rule[claim]}")
            else:
                add(arm, claim, ev["status"], ev["why"])
        if arm == GLASS_ARM:
            unrecorded = ("the fold's Pass 2 rule membership; the per-fold GLASS rule sets and "
                          "candidate rules were not persisted")
        else:
            unrecorded = (f"p2 > t2[fold]; RF Pass 2 outer scores were not persisted (per-fold t2 "
                          f"recorded: {t2_text})")
        add(arm, "decision.pass2_condition", NOT_VERIFIABLE, f"Needs {unrecorded}.")
        add(arm, "decision.abstain_not_pass2", NOT_VERIFIABLE, f"Needs {unrecorded}.")
        status, text = _population_row(glass if arm == GLASS_ARM else rf, arm)
        add(arm, "fit.pass2_population_declared", status, text)
        add(arm, "fit.pass2_population_realised", NOT_VERIFIABLE,
            "The per-fold GLASS fit populations are not persisted." if arm == GLASS_ARM else
            "Each outer fold's inner OOF Pass 1 remainder is not persisted (oof_fold_summary "
            "records a remainder fraction, which the diagnostics boundary does not read; "
            "Commit 3, F17).")
    out = pd.DataFrame(rows, columns=["arm", "router", "claim", "description", "status", "basis",
                                      "violations", "evidence"])
    out["violations"] = pd.array(out["violations"].tolist(), dtype="Int64")
    return out


# ----------------------------------------------------------------------
# Distributions and rates
# ----------------------------------------------------------------------
def _groups(trace: RouterTrace):
    fold = np.asarray(trace.fold_id)
    return [("all", None)] + [(k, fold == k) for k in range(trace.n_folds)]


def state_distribution(trace: RouterTrace) -> pd.DataFrame:
    """
    Per arm, fold ("all" = pooled) and state: ``rows`` (the fold's or pool's
    rows), ``n``, ``positives``, ``negatives`` (from ``metrics.state_counts``),
    ``share`` = n / rows (the registered pass1_rate / pass2_rate /
    abstain_rate), ``positive_rate`` = positives / n and ``negative_rate`` =
    negatives / n (``state_positive_rate`` / ``state_negative_rate``). An empty
    state's class proportions are NaN with status undefined_empty_denominator.
    """
    _require_trace(trace)
    recs = []
    for arm in TRACE_ARMS:
        v = trace.views[arm]
        pooled, by_fold = state_counts(v), state_counts(v, by_fold=True)
        for g, mask in _groups(trace):
            sc = pooled if mask is None else by_fold[by_fold["fold"] == g]
            rows_in = trace.n if mask is None else int(mask.sum())
            for s in ROUTER_STATES:
                r = sc[sc["state"] == s].iloc[0]
                n_s, pos, neg = int(r["n"]), int(r["positives"]), int(r["negatives"])
                share = Ratio(_STATE_SHARE[s], n_s, rows_in)
                pr = Ratio("state_positive_rate", pos, n_s)
                nr = Ratio("state_negative_rate", neg, n_s)
                recs.append({"arm": arm, "fold": g, "state": s, "rows": rows_in, "n": n_s,
                             "positives": pos, "negatives": neg,
                             "share": share.as_float(), "share_status": share.status,
                             "positive_rate": pr.as_float(), "positive_rate_status": pr.status,
                             "negative_rate": nr.as_float(), "negative_rate_status": nr.status})
    out = pd.DataFrame(recs)
    out["state"] = pd.Categorical(out["state"], categories=list(ROUTER_STATES))
    return out


def routing_rates(trace: RouterTrace) -> pd.DataFrame:
    """
    Per arm and fold ("all" = pooled): Pass 1 coverage (pass1_rate), Pass 2
    coverage (pass2_rate), total coverage and abstention rate, each with its
    numerator and denominator. Pass 1 / Pass 2 come from
    ``metrics.pass1_metrics`` / ``pass2_metrics``; coverage and abstain_rate
    use the registered definitions (pooled values equal ``router_metrics``).
    """
    _require_trace(trace)
    recs = []
    for arm in TRACE_ARMS:
        v = trace.views[arm]
        for g, mask in _groups(trace):
            ab = v.abstain if mask is None else v.abstain[mask]
            n_g = len(ab)
            ratios = (pass1_metrics(v, mask)["pass1_rate"], pass2_metrics(v, mask)["pass2_rate"],
                      Ratio("coverage", n_g - int(ab.sum()), n_g),
                      Ratio("abstain_rate", int(ab.sum()), n_g))
            for r in ratios:
                recs.append({"arm": arm, "fold": g, "metric": r.name, "label": _RATE_LABELS[r.name],
                             "numerator": r.numerator, "denominator": r.denominator,
                             "value": r.as_float(), "status": r.status})
    return pd.DataFrame(recs, columns=["arm", "fold", "metric", "label", "numerator",
                                       "denominator", "value", "status"])


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------
def _cmp_count(fold, quantity, metric, state, a: int, b: int) -> dict:
    return {"fold": fold, "quantity": quantity, "metric": metric, "state": state, "kind": "count",
            f"{GLASS_ARM}_numerator": a, f"{GLASS_ARM}_denominator": None,
            f"{GLASS_ARM}_value": float(a),
            f"{RF_ARM}_numerator": b, f"{RF_ARM}_denominator": None, f"{RF_ARM}_value": float(b),
            "difference": float(b - a), "difference_pp": float("nan"), "status": "count"}


def _cmp_rate(fold, quantity, metric, state, a: Ratio, b: Ratio) -> dict:
    defined = a.value is not None and b.value is not None
    diff = (b.value - a.value) if defined else float("nan")
    return {"fold": fold, "quantity": quantity, "metric": metric, "state": state, "kind": "rate",
            f"{GLASS_ARM}_numerator": a.numerator, f"{GLASS_ARM}_denominator": a.denominator,
            f"{GLASS_ARM}_value": a.as_float(),
            f"{RF_ARM}_numerator": b.numerator, f"{RF_ARM}_denominator": b.denominator,
            f"{RF_ARM}_value": b.as_float(),
            "difference": diff, "difference_pp": diff * 100.0 if defined else float("nan"),
            "status": "defined" if defined else "undefined_empty_denominator"}


def _comparison_notes(trace: RouterTrace) -> Tuple[str, ...]:
    glass, rf = trace.views[GLASS_ARM], trace.views[RF_ARM]
    gp, rp = dict(glass.pass2_population), dict(rf.pass2_population)
    cs = rf.metadata.get("constraint_source")
    if cs == "match_glass_oof":
        constraint = (
            "RF Router thresholds were tuned against constraints derived from the GLASS Router's "
            "historical pooled OOF metrics (constraint_source='match_glass_oof'). The arms are "
            "not independent, and because those pooled metrics include each outer fold's own "
            "rows, the RF Router's outer-fold thresholds depend weakly on that fold's labels. "
            "Independent shared criteria are Commit 4's.")
    elif cs is None:
        constraint = "RF Router constraint_source is not recorded; cross-arm dependence unknown."
    else:
        constraint = f"RF Router constraint_source={cs!r} (recorded in the artifact)."
    missing = [f"{ARM_LABELS[a]}: {p.name} ({p.detail})" for a in TRACE_ARMS
               for p in trace.views[a].provenance if p.status == NOT_RECORDED]
    return (
        f"Observational comparison of the historical outer OOF routes on the same "
        f"{trace.n:,} training rows and {trace.n_folds} outer folds. Coverage and abstention "
        "describe how much each router decides, not how well: abstained rows continue to "
        "Stages 3-4, and a router state is not the cascade's final outcome. No ranking is "
        "implied and no significance test is made; per-fold values come from models fitted on "
        "overlapping training sets and are not independent replicates.",
        f"Pass 2 fit-time populations differ: GLASS {gp.get('cross_fit')!r} (normalised "
        f"{gp.get('normalised')!r}); RF {rp.get('cross_fit')!r} (normalised "
        f"{rp.get('normalised')!r}). At decision time both routers let Pass 2 claim only rows "
        "Pass 1 left; the difference is the rows Pass 2 was fitted and selected on (Commit 3, "
        "F17), so Pass 2 differences are not attributable to the rule selector or forest alone.",
        constraint,
        "Not recorded in the historical artifacts, so no score-level or rule-level comparison "
        "is made: " + "; ".join(missing) + ".",
        "These observed distributions describe the historical routers. They must not be used "
        "to choose the Commit 4 operating-contract criteria, which are to be declared "
        "independently of both arms' observed performance.",
    )


def compare_state_distributions(trace: RouterTrace) -> DiagnosticResult:
    """
    Observational GLASS vs RF comparison, pooled and per fold. ``table``: one
    row per fold and quantity with both arms' numerator / denominator / value,
    ``difference`` = RF - GLASS (rows for counts, proportion for rates) and
    ``difference_pp`` (rates, percentage points). State shares are the coverage
    rows (Pass 1 share = Pass 1 coverage, Abstain share = abstention rate).
    Undefined on either side -> NaN difference with status
    undefined_empty_denominator. ``values``: pooled ratios and differences.
    """
    _require_trace(trace)
    rr, sd = routing_rates(trace), state_distribution(trace)

    def rate(arm, fold, metric) -> Ratio:
        x = rr[(rr["arm"] == arm) & (rr["fold"].map(lambda f: f == fold))
               & (rr["metric"] == metric)].iloc[0]
        return Ratio(metric, int(x["numerator"]), int(x["denominator"]))

    def state_row(arm, fold, s):
        return sd[(sd["arm"] == arm) & (sd["fold"].map(lambda f: f == fold))
                  & (sd["state"] == s)].iloc[0]

    recs, values = [], {}
    for fold, _ in _groups(trace):
        for metric, label in _RATE_LABELS.items():
            a, b = rate(GLASS_ARM, fold, metric), rate(RF_ARM, fold, metric)
            recs.append(_cmp_rate(fold, label, metric, "", a, b))
            if fold == "all":
                values[f"{GLASS_ARM}.{metric}"], values[f"{RF_ARM}.{metric}"] = a, b
                values[f"difference_pp.{metric}"] = recs[-1]["difference_pp"]
        for s in ROUTER_STATES:
            a, b, lab = state_row(GLASS_ARM, fold, s), state_row(RF_ARM, fold, s), _STATE_LABELS[s]
            for col in ("n", "positives", "negatives"):
                name = "rows" if col == "n" else col
                recs.append(_cmp_count(fold, f"{lab} {name}", f"{s}.{col}", s,
                                       int(a[col]), int(b[col])))
                if fold == "all":
                    values[f"difference.{s}.{col}"] = int(b[col]) - int(a[col])
            for col, metric in (("positives", "state_positive_rate"),
                                ("negatives", "state_negative_rate")):
                ra = Ratio(metric, int(a[col]), int(a["n"]))
                rb = Ratio(metric, int(b[col]), int(b["n"]))
                recs.append(_cmp_rate(fold, f"{lab} {col[:-1]} rate (within {lab} rows)",
                                      f"{s}.{metric}", s, ra, rb))
                if fold == "all":
                    values[f"difference_pp.{s}.{metric}"] = recs[-1]["difference_pp"]
    table = pd.DataFrame(recs)
    for c in (f"{GLASS_ARM}_numerator", f"{GLASS_ARM}_denominator", f"{RF_ARM}_numerator",
              f"{RF_ARM}_denominator"):
        table[c] = pd.array(table[c].tolist(), dtype="Int64")
    per_fold = table[(table["fold"].map(lambda f: f != "all")) & (table["kind"] == "rate")
                     & (table["state"] == "")]
    spread = MappingProxyType({
        m: (float(per_fold.loc[per_fold["metric"] == m, "difference_pp"].min()),
            float(per_fold.loc[per_fold["metric"] == m, "difference_pp"].max()))
        for m in _RATE_LABELS})
    return _result(f"{ARM_LABELS[GLASS_ARM]} vs {ARM_LABELS[RF_ARM]}: routing states (outer OOF)",
                   values, _comparison_notes(trace), table,
                   meta={"arms": TRACE_ARMS, "labels": ARM_LABELS,
                         "difference": f"{RF_ARM} minus {GLASS_ARM}",
                         "fold_difference_range_pp": spread})
