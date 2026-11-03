"""
PR 37 Commit 2 (Step 2A): the three-state router trace of
shared.stage2.diagnostics.trace on the synthetic Commit 1 world.

Synthetic tests prove the mechanics only (checks, statuses, denominators,
reconciliation, read-only behaviour). Historical validity is established
separately by tests/realdata/test_router_trace_artifacts.py on the saved
October 23 artifacts. Router behaviour itself is pinned by the existing GLASS
and RF routing tests, which the audit cites and the last test here checks
still exist.
"""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import trace as tr
from shared.stage2.diagnostics.boundary import RF_PASS2_REMAINDER_PROTOCOL
from tests.support import diagnostics_world as dw

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN
DECISION_CLAIMS = ("decision.pass1_precedence", "decision.pass2_eligibility",
                   "decision.abstain_not_pass1")


@pytest.fixture(scope="module")
def built():
    w = dw.world()
    ref, glass, rf, _, _ = dw.views(w)
    cand = d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), ref)
    return {"w": w, "ref": ref, "glass": glass, "rf": rf, "cand": cand,
            "trace": d.build_router_trace(glass, rf, ref)}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _reroute(view, rows, token):
    """Consistently re-route ``rows`` to ``token`` (tokens, states and preds agree)."""
    dec = np.array(view.source_decision, dtype=object, copy=True)
    dec[rows] = token
    st = np.array([d.SOURCE_DECISIONS[t][0] for t in dec], dtype=object)
    pred = np.array([d.SOURCE_DECISIONS[t][1] for t in dec])
    return replace(view, source_decision=dec, state=st, source_pred=pred)


def _with(view, **fields):
    """A view that bypasses the extractor (what the trace must not trust blindly)."""
    return replace(view, **fields)


def _first(view, state, fold=None):
    m = np.asarray(view.state) == state
    if fold is not None:
        m &= np.asarray(view.fold_id) == fold
    return int(np.flatnonzero(m)[0])


def _failures(glass, rf, ref):
    with pytest.raises(d.TraceCheckError) as e:
        d.build_router_trace(glass, rf, ref)
    report = e.value.report
    assert list(report.columns) == ["section", "stage", "check", "passed", "detail", "error"]
    # malformed inputs are reported as failed checks, never as a crash inside a check
    assert (report["error"] == "").all(), report.loc[report["error"] != "", ["check", "error"]]
    return report[~report["passed"]]


def _failed(bad, stage, prefix) -> bool:
    return bool(((bad["stage"] == stage) & bad["check"].str.startswith(prefix)).any())


def _audit_row(audit, arm, claim):
    rows = audit[(audit["arm"] == arm) & (audit["claim"] == claim)]
    assert len(rows) == 1
    return rows.iloc[0]


def _digest(view) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(pd.Series(view.index)).to_numpy().tobytes())
    for a in ("y", "fold_id", "source_pred", "confidence"):
        h.update(np.ascontiguousarray(getattr(view, a)).tobytes())
    h.update(np.ascontiguousarray(view.router_score.values).tobytes())
    for a in ("state", "source_decision"):
        h.update("|".join(map(str, getattr(view, a))).encode())
    h.update(repr((dict(view.metadata), dict(view.pass2_population), view.provenance,
                   view.fold_thresholds and {k: dict(v) for k, v in view.fold_thresholds.items()}))
             .encode())
    return h.hexdigest()


# ======================================================================
# 1-2. a valid trace: row identity, one state per row
# ======================================================================
def test_valid_trace_passes_every_check_t1_to_t7(built):
    checks = built["trace"].check_frame()
    assert checks["passed"].all() and (checks["error"] == "").all()
    assert set(checks["stage"]) == {"reference", "glass", "blackbox", "glass x blackbox"}
    for t in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        for arm in ("glass", "blackbox"):
            assert checks.loc[checks["stage"] == arm, "check"].str.startswith(t).any(), (t, arm)
    assert len(checks) == len(set(zip(checks["stage"], checks["check"])))


def test_trace_keeps_row_identity_labels_folds_tokens_and_preds(built):
    t, ref, glass, rf = built["trace"], built["ref"], built["glass"], built["rf"]
    f = t.frame()
    assert t.index.equals(ref.index_train) and f.index.equals(ref.index_train)
    assert t.n == ref.n_train and t.n_positives == int(ref.y_train.sum())
    np.testing.assert_array_equal(f["y"], ref.y_train)
    np.testing.assert_array_equal(f["fold_id"], glass.fold_id)
    np.testing.assert_array_equal(f["fold_id"], rf.fold_id)
    for arm, v in (("glass", glass), ("blackbox", rf)):
        np.testing.assert_array_equal(f[f"{arm}_state"].astype(object), v.state)
        np.testing.assert_array_equal(f[f"{arm}_source_decision"], v.source_decision)
        np.testing.assert_array_equal(f[f"{arm}_source_pred"], v.source_pred)
        assert list(f[f"{arm}_state"].cat.categories) == list(d.ROUTER_STATES)
    assert (t.router_split_fingerprint, t.train_label_fingerprint) == (
        ref.router_split_fingerprint, ref.train_label_fingerprint)
    assert t.stage_io_split_fingerprint == ref.stage_io_split_fingerprint


def test_each_row_has_exactly_one_state_in_each_arm(built):
    f = built["trace"].frame()
    for arm in ("glass", "blackbox"):
        one_hot = pd.get_dummies(f[f"{arm}_state"])
        assert list(one_hot.columns) == list(d.ROUTER_STATES)
        assert (one_hot.sum(axis=1) == 1).all()
        assert int(one_hot.to_numpy().sum()) == built["trace"].n


def test_trace_provenance_carries_both_views_items(built):
    prov = built["trace"].provenance_frame()
    status = dict(zip(prov["name"], prov["status"]))
    for item in ("trace.rows", "trace.states", "trace.folds", "trace.reconciliation"):
        assert status[item] == d.VALIDATED
    assert status["glass.router_transform"] == d.VALIDATED
    assert status["glass.fold_rule_sets"] == d.NOT_RECORDED
    assert status["blackbox.pass2_outer_scores"] == d.NOT_RECORDED


# ======================================================================
# 5. invalid tokens, predictions, folds, labels, indices and inputs
# ======================================================================
def _unknown_state(g, r, ref):
    st = np.array(g.state, dtype=object, copy=True)
    st[0] = "PASS_3"
    return _with(g, state=st), r, ref


def _state_vs_token(g, r, ref):
    i = _first(g, P1)
    st = np.array(g.state, dtype=object, copy=True)
    st[i] = P2
    return _with(g, state=st), r, ref


def _pred_vs_token(g, r, ref):
    i = _first(r, P1)
    pred = np.array(r.source_pred, copy=True)
    pred[i] = 1
    return g, _with(r, source_pred=pred), ref


def _unknown_token(g, r, ref):
    dec = np.array(r.source_decision, dtype=object, copy=True)
    dec[3] = "remainder"
    return g, _with(r, source_decision=dec), ref


def _reordered_index(g, r, ref):
    return _with(g, index=g.index[::-1]), r, ref


def _dropped_row(g, r, ref):
    cut = {a: np.asarray(getattr(g, a))[:-1] for a in
           ("y", "fold_id", "state", "source_decision", "source_pred", "confidence")}
    return _with(g, index=g.index[:-1], **cut), r, ref


def _flipped_label(g, r, ref):
    y = np.array(r.y, copy=True)
    y[0] = 1 - y[0]
    return g, _with(r, y=y), ref


def _non_binary_label(g, r, ref):
    y = np.array(g.y, copy=True)
    y[0] = 2
    return _with(g, y=y), r, ref


def _folds_differ_across_arms(g, r, ref):
    fold = np.array(r.fold_id, copy=True)
    i, j = _first(r, P1, fold=0), _first(r, P1, fold=1)
    fold[i], fold[j] = fold[j], fold[i]
    return g, _with(r, fold_id=fold), ref


def _fold_out_of_range(g, r, ref):
    fold = np.array(g.fold_id, copy=True)
    fold[0] = 5
    return _with(g, fold_id=fold), r, ref


def _float_folds(g, r, ref):
    return _with(g, fold_id=np.asarray(g.fold_id, dtype=float)), r, ref


def _fold_fingerprint_differs(g, r, ref):
    proto = dict(r.metadata["oof_protocol"], fold_fingerprint="0" * 16)
    return g, _with(r, metadata=MappingProxyType(dict(r.metadata, oof_protocol=proto))), ref


def _glass_twice(g, r, ref):
    return g, g, ref


def _rf_twice(g, r, ref):
    return r, r, ref


def _not_a_view(g, r, ref):
    return g, {"train_decisions": np.asarray(r.source_decision)}, ref


def _not_a_reference(g, r, ref):
    return g, r, {"index_train": ref.index_train}


@pytest.mark.parametrize("make, expected", [
    (_unknown_state, [("glass", "T4 every state"), ("glass", "T4 exactly one"),
                      ("glass", "T5 state =")]),
    (_state_vs_token, [("glass", "T5 state =")]),
    (_pred_vs_token, [("blackbox", "T5 source_pred =")]),
    (_unknown_token, [("blackbox", "T5 decision tokens")]),
    (_reordered_index, [("glass", "T2 row index = training index")]),
    (_dropped_row, [("glass", "T2 row index = training index"),
                    ("glass", "T2 every per-row field")]),
    (_flipped_label, [("blackbox", "T3 labels = training labels")]),
    (_non_binary_label, [("glass", "T3 labels = training labels"),
                         ("glass", "T3 labels are binary")]),
    (_folds_differ_across_arms, [("glass x blackbox", "T6 both arms use the same outer fold")]),
    (_fold_out_of_range, [("glass", "T6 fold ids"),
                          ("glass x blackbox", "T6 both arms use the same outer fold")]),
    (_float_folds, [("glass", "T6 fold ids")]),
    (_fold_fingerprint_differs, [("glass x blackbox", "T6 both arms declare")]),
    (_glass_twice, [("blackbox", "T1 arm is 'blackbox'")]),
    (_rf_twice, [("glass", "T1 arm is 'glass'")]),
    (_not_a_view, [("blackbox", "T1 input is a RouterOOFView")]),
    (_not_a_reference, [("reference", "T1 reference is a TrainReference")]),
], ids=lambda x: getattr(x, "__name__", ""))
def test_malformed_inputs_fail_the_named_checks_without_crashing(built, make, expected):
    bad = _failures(*make(built["glass"], built["rf"], built["ref"]))
    for stage, prefix in expected:
        assert _failed(bad, stage, prefix), (stage, prefix, bad[["stage", "check"]].to_string())


def test_checks_behind_a_failed_prerequisite_are_reported_not_evaluated(built):
    bad = _failures(*_dropped_row(built["glass"], built["rf"], built["ref"]))
    later = bad[(bad["stage"] == "glass") & bad["check"].str.match(r"T[3-7]")]
    assert len(later) and (later["detail"] == tr._NOT_EVALUATED).all()
    assert not (bad["stage"] == "blackbox").any()          # the other arm is still checked


def test_the_error_message_lists_every_failure(built):
    with pytest.raises(d.TraceCheckError) as e:
        d.build_router_trace(*_unknown_state(built["glass"], built["rf"], built["ref"]))
    n_failed = int((~e.value.report["passed"]).sum())
    assert f"{n_failed} check(s) failed" in str(e.value) and "PASS_3" in str(e.value)
    assert isinstance(e.value, d.RouterDiagnosticError)


# ======================================================================
# 3-4. Pass 1 precedence, Pass 2 eligibility and the evidence statuses
# ======================================================================
def test_rf_decision_claims_are_verified_only_by_a_passing_equivalence(built):
    rep = d.check_rf_pass1_score_equivalence(built["rf"], built["cand"])
    assert rep.passed                                     # the synthetic world is consistent
    audit = d.routing_policy_audit(built["trace"], rf_candidate=built["cand"])
    for claim in DECISION_CLAIMS:
        row = _audit_row(audit, "blackbox", claim)
        assert (row["status"], row["basis"], row["violations"]) == (
            tr.VERIFIED_FROM_ARTIFACTS, "artifact", 0)
        assert row["evidence"].startswith("EQUIVALENT")


def test_without_a_candidate_rf_decision_claims_are_not_verifiable(built):
    audit = d.routing_policy_audit(built["trace"])
    for claim in DECISION_CLAIMS:
        row = _audit_row(audit, "blackbox", claim)
        assert row["status"] == tr.NOT_VERIFIABLE and pd.isna(row["violations"])
        assert "No RF Pass 1 score candidate supplied" in row["evidence"]


def _far_disagreement_on_abstain(w):
    w["rf_result"]["oof_proba"][_first_dec(w, "uncertain")] = 0.0


def _pass2_row_below_t1(w):
    w["rf_result"]["oof_proba"][_first_dec(w, "pass2")] = 0.0


def _boundary_case_on_pass1(w):
    i = _first_dec(w, "pass1")
    w["rf_result"]["oof_proba"][i] = {r["fold"]: r["t1"] for r in
                                      w["rf"]["extra"]["oof_fold_summary"]}[int(w["fold"][i])]


def _forest_params_differ(w):
    w["rf_result"]["params"]["n_estimators"] = 100


def _first_dec(w, token):
    return int(np.flatnonzero(np.asarray(w["rf"]["extra"]["train_decisions"]) == token)[0])


@pytest.mark.parametrize("mutate, reason", [
    (_far_disagreement_on_abstain, "row disagreement"),
    (_pass2_row_below_t1, "Pass 2 row(s) score below t1"),
    (_boundary_case_on_pass1, "boundary cases are documented, not accepted"),
    (_forest_params_differ, "structural checks failed"),
], ids=lambda x: getattr(x, "__name__", ""))
def test_a_failed_equivalence_is_failed_evidence_never_verified(built, mutate, reason):
    w = dw.world()
    mutate(w)
    cand = d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), built["ref"])
    audit = d.routing_policy_audit(built["trace"], rf_candidate=cand)
    for claim in DECISION_CLAIMS:
        row = _audit_row(audit, "blackbox", claim)
        assert row["status"] == tr.EVIDENCE_FAILED and row["basis"] == "artifact"
        assert pd.isna(row["violations"])                 # never read as a routing violation
        assert row["evidence"].startswith("UNSUPPORTED") and reason in row["evidence"]


def test_a_candidate_for_another_reference_is_failed_evidence(built):
    cand = replace(built["cand"], train_label_fingerprint="0" * 12)
    audit = d.routing_policy_audit(built["trace"], rf_candidate=cand)
    row = _audit_row(audit, "blackbox", "decision.pass1_precedence")
    assert row["status"] == tr.EVIDENCE_FAILED and "Identity checks failed" in row["evidence"]


@pytest.mark.parametrize("with_candidate", [False, True])
def test_glass_decision_claims_are_code_level_only(built, with_candidate):
    audit = d.routing_policy_audit(built["trace"],
                                   rf_candidate=built["cand"] if with_candidate else None)
    for claim in DECISION_CLAIMS:
        row = _audit_row(audit, "glass", claim)
        assert (row["status"], row["basis"]) == (tr.CODE_LEVEL_ONLY, "code")
        assert pd.isna(row["violations"])
        assert "test_pass1_rows_are_never_rerouted_by_pass2" in row["evidence"]
        assert "not persisted" in row["evidence"]
    glass = audit[audit["arm"] == "glass"]
    assert set(glass.loc[glass["claim"].str.startswith("decision."), "status"]) <= {
        tr.CODE_LEVEL_ONLY, tr.NOT_VERIFIABLE}


def test_unpersisted_evidence_is_never_claimed(built):
    audit = d.routing_policy_audit(built["trace"], rf_candidate=built["cand"])
    for arm in ("glass", "blackbox"):
        for claim in ("decision.pass2_condition", "decision.abstain_not_pass2",
                      "fit.pass2_population_realised"):
            assert _audit_row(audit, arm, claim)["status"] == tr.NOT_VERIFIABLE, (arm, claim)
    rf_cond = _audit_row(audit, "blackbox", "decision.pass2_condition")["evidence"]
    assert "Pass 2 outer scores were not persisted" in rf_cond and "fold 4:" in rf_cond
    assert "rule sets" in _audit_row(audit, "glass", "decision.pass2_condition")["evidence"]


def test_fit_time_populations_are_declarations_read_from_the_views(built):
    audit = d.routing_policy_audit(built["trace"])
    g = _audit_row(audit, "glass", "fit.pass2_population_declared")
    r = _audit_row(audit, "blackbox", "fit.pass2_population_declared")
    assert g["status"] == r["status"] == tr.DECLARED
    assert "'full'" in g["evidence"] and "whole training population" in g["evidence"]
    assert RF_PASS2_REMAINDER_PROTOCOL in r["evidence"] and "'oof_remainder'" in r["evidence"]
    # not hardcoded: a view declaring another population is reported as declared
    pop = MappingProxyType({"cross_fit": "oof_remainder", "full_train": "oof_remainder",
                            "normalised": "oof_remainder"})
    glass2 = replace(built["glass"], pass2_population=pop)
    t2 = d.build_router_trace(glass2, built["rf"], built["ref"])
    g2 = _audit_row(d.routing_policy_audit(t2), "glass", "fit.pass2_population_declared")
    assert "'oof_remainder'" in g2["evidence"] and "whole training population" not in g2["evidence"]
    # an unvalidated declaration is not even "declared"
    prov = tuple(d.ProvenanceItem(p.name, d.NOT_RECORDED, p.detail) if p.name == "pass2_population"
                 else p for p in built["glass"].provenance)
    t3 = d.build_router_trace(replace(built["glass"], provenance=prov), built["rf"], built["ref"])
    assert _audit_row(d.routing_policy_audit(t3), "glass",
                      "fit.pass2_population_declared")["status"] == tr.NOT_VERIFIABLE


def test_audit_shape_and_vocabulary(built):
    audit = d.routing_policy_audit(built["trace"], rf_candidate=built["cand"])
    assert list(audit.columns) == ["arm", "router", "claim", "description", "status", "basis",
                                   "violations", "evidence"]
    assert len(audit) == 2 * len(tr.AUDIT_CLAIMS)
    assert set(audit["status"]) <= set(d.AUDIT_STATUSES)
    assert all(tr._BASIS[s] == b for s, b in zip(audit["status"], audit["basis"]))
    assert str(audit["violations"].dtype) == "Int64"
    assert set(tr.AUDIT_STATUS_DEFINITIONS) == set(d.AUDIT_STATUSES)
    assert (audit.loc[audit["status"] == tr.VERIFIED_FROM_ARTIFACTS, "violations"] == 0).all()
    assert audit.loc[audit["status"] != tr.VERIFIED_FROM_ARTIFACTS, "violations"].isna().all()


def test_audit_refuses_misuse(built):
    with pytest.raises(d.RouterDiagnosticError, match="RouterTrace"):
        d.routing_policy_audit(built["rf"])
    with pytest.raises(d.RouterDiagnosticError, match="RFPass1ScoreCandidate"):
        d.routing_policy_audit(built["trace"], rf_candidate=np.zeros(3))


# ======================================================================
# 6-8. state distributions, class composition, reconciliation, empty states
# ======================================================================
def test_state_distribution_by_hand(built):
    t = built["trace"]
    sd = d.state_distribution(t)
    assert len(sd) == 2 * (1 + t.n_folds) * 3
    y, fold = np.asarray(t.y), np.asarray(t.fold_id)
    for arm in ("glass", "blackbox"):
        st = np.asarray(t.views[arm].state)
        for g in ["all", *range(t.n_folds)]:
            keep = np.ones(t.n, bool) if g == "all" else fold == g
            for s in d.ROUTER_STATES:
                m = keep & (st == s)
                row = sd[(sd["arm"] == arm) & (sd["fold"].map(lambda f: f == g))
                         & (sd["state"] == s)].iloc[0]
                assert (row["rows"], row["n"], row["positives"], row["negatives"]) == (
                    keep.sum(), m.sum(), y[m].sum(), (y[m] == 0).sum())
                assert row["share"] == pytest.approx(m.sum() / keep.sum())
                assert row["positive_rate"] == pytest.approx(y[m].sum() / m.sum())
                assert row["negative_rate"] == pytest.approx((y[m] == 0).sum() / m.sum())
                assert row["positive_rate"] + row["negative_rate"] == pytest.approx(1.0)


def test_state_distribution_reuses_metrics_state_counts(built):
    sd = d.state_distribution(built["trace"])
    for arm in ("glass", "blackbox"):
        v = built["trace"].views[arm]
        pooled = d.state_counts(v)
        mine = sd[(sd["arm"] == arm) & (sd["fold"].map(lambda f: f == "all"))]
        for c in ("n", "positives", "negatives", "share", "positive_rate"):
            np.testing.assert_allclose(mine[c].to_numpy(float), pooled[c].to_numpy(float))
        by_fold = d.state_counts(v, by_fold=True)
        mine = sd[(sd["arm"] == arm) & (sd["fold"].map(lambda f: f != "all"))]
        np.testing.assert_array_equal(mine["n"].to_numpy(), by_fold["n"].to_numpy())


def test_per_fold_values_reconcile_with_the_pooled_values(built):
    t = built["trace"]
    sd = d.state_distribution(t)
    for arm in ("glass", "blackbox"):
        a = sd[sd["arm"] == arm]
        pooled = a[a["fold"].map(lambda f: f == "all")].set_index("state")
        folds = a[a["fold"].map(lambda f: f != "all")]
        summed = folds.groupby("state", observed=False)[["n", "positives", "negatives"]].sum()
        for c in ("n", "positives", "negatives"):
            np.testing.assert_array_equal(summed[c].to_numpy(), pooled[c].to_numpy())
        assert int(pooled["n"].sum()) == t.n and int(pooled["positives"].sum()) == t.n_positives
        fold_rows = folds.groupby("fold")["n"].sum()
        np.testing.assert_array_equal(fold_rows.to_numpy(), np.bincount(t.fold_id))
    rr = d.routing_rates(t)
    for (arm, metric), grp in rr.groupby(["arm", "metric"]):
        per_fold = grp[grp["fold"].map(lambda f: f != "all")]
        pooled = grp[grp["fold"].map(lambda f: f == "all")].iloc[0]
        assert int(per_fold["numerator"].sum()) == pooled["numerator"]
        assert int(per_fold["denominator"].sum()) == pooled["denominator"] == t.n


def test_an_empty_state_is_undefined_not_zero(built):
    g = built["glass"]
    rows = np.flatnonzero((np.asarray(g.state) == P2) & (np.asarray(g.fold_id) == 0))
    assert len(rows)
    t = d.build_router_trace(_reroute(g, rows, "uncertain"), built["rf"], built["ref"])
    sd = d.state_distribution(t)
    row = sd[(sd["arm"] == "glass") & (sd["fold"].map(lambda f: f == 0))
             & (sd["state"] == P2)].iloc[0]
    assert row["n"] == 0 and np.isnan(row["positive_rate"]) and np.isnan(row["negative_rate"])
    assert row["positive_rate_status"] == row["negative_rate_status"] == "undefined_empty_denominator"
    assert row["share"] == 0.0 and row["share_status"] == "defined"   # 0 of the fold's rows
    rr = d.routing_rates(t)
    p2 = rr[(rr["arm"] == "glass") & (rr["fold"].map(lambda f: f == 0))
            & (rr["metric"] == "pass2_rate")].iloc[0]
    assert (p2["numerator"], p2["value"], p2["status"]) == (0, 0.0, "defined")
    tab = d.compare_state_distributions(t).table
    cmp = tab[(tab["fold"].map(lambda f: f == 0))
              & (tab["metric"] == f"{P2}.state_positive_rate")].iloc[0]
    assert np.isnan(cmp["difference"]) and np.isnan(cmp["difference_pp"])
    assert cmp["status"] == "undefined_empty_denominator"
    pooled = tab[(tab["fold"].map(lambda f: f == "all"))
                 & (tab["metric"] == f"{P2}.state_positive_rate")].iloc[0]
    assert pooled["status"] == "defined"


# ======================================================================
# 9-10. coverage, abstention and parity with metrics.router_metrics
# ======================================================================
def test_routing_rates_by_hand(built):
    t = built["trace"]
    rr = d.routing_rates(t)
    assert list(rr.columns) == ["arm", "fold", "metric", "label", "numerator", "denominator",
                                "value", "status"]
    fold = np.asarray(t.fold_id)
    for arm in ("glass", "blackbox"):
        v = t.views[arm]
        for g in ["all", *range(t.n_folds)]:
            keep = np.ones(t.n, bool) if g == "all" else fold == g
            sub = rr[(rr["arm"] == arm) & (rr["fold"].map(lambda f: f == g))].set_index("metric")
            p1, p2, ab = (v.pass1 & keep).sum(), (v.pass2 & keep).sum(), (v.abstain & keep).sum()
            assert (sub.loc["pass1_rate", "numerator"], sub.loc["pass2_rate", "numerator"],
                    sub.loc["coverage", "numerator"], sub.loc["abstain_rate", "numerator"]) == (
                p1, p2, p1 + p2, ab)
            assert (sub["denominator"] == keep.sum()).all()
            assert p1 + p2 + ab == keep.sum()
            assert sub.loc["coverage", "value"] + sub.loc["abstain_rate", "value"] == pytest.approx(1)
            assert sub.loc["coverage", "value"] == pytest.approx(
                sub.loc["pass1_rate", "value"] + sub.loc["pass2_rate", "value"])


def test_routing_rates_match_router_metrics_pooled_and_per_fold(built):
    rr = d.routing_rates(built["trace"])
    for arm in ("glass", "blackbox"):
        v = built["trace"].views[arm]
        pooled = d.router_metrics(v)
        mine = rr[(rr["arm"] == arm) & (rr["fold"].map(lambda f: f == "all"))].set_index("metric")
        for m in ("pass1_rate", "pass2_rate", "coverage", "abstain_rate"):
            assert (mine.loc[m, "numerator"], mine.loc[m, "denominator"]) == (
                pooled[m].numerator, pooled[m].denominator), (arm, m)
        table = d.router_metrics(v, by_fold=True).table
        for m in ("pass1_rate", "pass2_rate", "coverage", "abstain_rate"):
            theirs = table[table["metric"] == m].sort_values("fold")["value"].to_numpy(float)
            ours = rr[(rr["arm"] == arm) & (rr["metric"] == m)
                      & (rr["fold"].map(lambda f: f != "all"))]["value"].to_numpy(float)
            np.testing.assert_allclose(ours, theirs)


def test_every_reported_rate_is_a_registered_metric(built):
    names = set(d.routing_rates(built["trace"])["metric"]) | {
        "state_positive_rate", "state_negative_rate", *tr._STATE_SHARE.values()}
    assert names <= set(d.METRIC_DEFINITIONS)
    assert d.METRIC_DEFINITIONS["state_positive_rate"][1] == "rows in the state"
    assert d.METRIC_DEFINITIONS["state_negative_rate"][1] == "rows in the state"
    for kept in ("coverage", "abstain_rate", "pass1_rate", "pass2_rate", "pass1_npv",
                 "pass2_precision", "abstain_positive_rate"):
        assert kept in d.METRIC_DEFINITIONS


# ======================================================================
# 11. GLASS vs RF comparison
# ======================================================================
def test_comparison_differences_by_hand(built):
    t = built["trace"]
    res = d.compare_state_distributions(t)
    tab = res.table
    assert len(tab) == (1 + t.n_folds) * (4 + 3 * 5)
    sd = d.state_distribution(t)
    for g in ["all", *range(t.n_folds)]:
        for s in d.ROUTER_STATES:
            a = sd[(sd["arm"] == "glass") & (sd["fold"].map(lambda f: f == g))
                   & (sd["state"] == s)].iloc[0]
            b = sd[(sd["arm"] == "blackbox") & (sd["fold"].map(lambda f: f == g))
                   & (sd["state"] == s)].iloc[0]
            row = tab[(tab["fold"].map(lambda f: f == g)) & (tab["metric"] == f"{s}.n")].iloc[0]
            assert row["difference"] == b["n"] - a["n"] and row["kind"] == "count"
            rate = tab[(tab["fold"].map(lambda f: f == g))
                       & (tab["metric"] == f"{s}.state_positive_rate")].iloc[0]
            assert rate["difference_pp"] == pytest.approx(
                100 * (b["positive_rate"] - a["positive_rate"]))
    cov = tab[(tab["fold"].map(lambda f: f == "all")) & (tab["metric"] == "coverage")].iloc[0]
    assert cov["difference_pp"] == pytest.approx(
        100 * (cov["blackbox_value"] - cov["glass_value"]))
    assert res["difference_pp.coverage"] == pytest.approx(cov["difference_pp"])
    assert res[f"difference.{P1}.n"] == int(t.views["blackbox"].pass1.sum()
                                             - t.views["glass"].pass1.sum())
    assert isinstance(res["glass.coverage"], d.Ratio)


def test_comparison_fold_spread_is_descriptive_min_max(built):
    res = d.compare_state_distributions(built["trace"])
    tab = res.table
    for m, (lo, hi) in res.meta["fold_difference_range_pp"].items():
        per_fold = tab[(tab["fold"].map(lambda f: f != "all")) & (tab["metric"] == m)]
        assert (lo, hi) == (per_fold["difference_pp"].min(), per_fold["difference_pp"].max())
        assert lo <= hi


def test_comparison_notes_carry_every_disclosure(built):
    res = d.compare_state_distributions(built["trace"])
    assert res.notes[0] == d.HISTORICAL_TEST_EXPOSURE
    text = " ".join(res.notes)
    for needle in ("Observational", "No ranking", "not independent replicates",
                   "'full'", RF_PASS2_REMAINDER_PROTOCOL, "decision time", "Commit 3",
                   "match_glass_oof", "not independent", "depend weakly",
                   "fold_rule_sets", "pass2_outer_scores", "Commit 4"):
        assert needle in text, needle


@pytest.mark.parametrize("source, needle", [
    ("other_source", "constraint_source='other_source'"),
    (None, "constraint_source is not recorded"),
])
def test_constraint_disclosure_follows_the_recorded_source(built, source, needle):
    meta = dict(built["rf"].metadata)
    if source is None:
        meta.pop("constraint_source")
    else:
        meta["constraint_source"] = source
    t = d.build_router_trace(built["glass"], replace(built["rf"], metadata=MappingProxyType(meta)),
                             built["ref"])
    text = " ".join(d.compare_state_distributions(t).notes)
    assert needle in text and "match_glass_oof" not in text


# ======================================================================
# 12. non-mutation and read-only results
# ======================================================================
def test_no_source_view_or_candidate_is_mutated(built):
    before = {k: _digest(built[k]) for k in ("glass", "rf")}
    cand_before = built["cand"].digest
    t = d.build_router_trace(built["glass"], built["rf"], built["ref"])
    d.routing_policy_audit(t, rf_candidate=built["cand"])
    d.state_distribution(t), d.routing_rates(t), d.compare_state_distributions(t)
    f = t.frame()
    f["y"] = 0
    f["fold_id"] = 9
    f["blackbox_source_pred"] = 7
    f["glass_source_decision"] = "pass2"
    assert {k: _digest(built[k]) for k in ("glass", "rf")} == before
    assert built["cand"].digest == cand_before
    assert t.views["glass"] is built["glass"] and t.views["blackbox"] is built["rf"]


def test_trace_is_read_only(built):
    t = built["trace"]
    for arr in (t.y, t.fold_id):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 1
    with pytest.raises(FrozenInstanceError):
        t.y = None
    with pytest.raises(TypeError):
        t.views["glass"] = None
    with pytest.raises(TypeError):
        t.checks[0]["passed"] = False
    f = t.frame()
    f.loc[:, "glass_source_decision"] = "pass2"
    assert (t.frame()["glass_source_decision"] == np.asarray(built["glass"].source_decision)).all()
    c = t.check_frame()
    c["passed"] = False
    assert t.check_frame()["passed"].all()
    meta = d.compare_state_distributions(t).meta
    with pytest.raises(TypeError):
        meta["fold_difference_range_pp"]["coverage"] = (0, 0)


@pytest.mark.parametrize("fn", [d.state_distribution, d.routing_rates,
                                d.compare_state_distributions])
def test_summaries_refuse_anything_but_a_trace(built, fn):
    with pytest.raises(d.RouterDiagnosticError, match="RouterTrace"):
        fn(built["glass"])


# ======================================================================
# the code-level evidence the audit cites still exists
# ======================================================================
@pytest.mark.parametrize("path, test_name", [
    ("test_glass_router_routing.py", "test_pass1_rows_are_never_rerouted_by_pass2"),
    ("test_glass_router_routing.py", "test_states_are_exclusive_exhaustive_and_consistent"),
    ("test_glass_router_crossfit.py",
     "test_oof_block_holds_each_fold_models_output_in_training_row_order"),
])
def test_cited_glass_routing_tests_exist(path, test_name):
    source = (Path(__file__).parent / path).read_text(encoding="utf-8")
    assert f"def {test_name}(" in source
    assert test_name in tr._GLASS_CODE_EVIDENCE
