"""
PR 38 Step 2A: the runner (``stage2_crossfit.runner``).

1. G1 / G2 / G9: the primary settings come only from the strict entry points,
   pass every check and record them; S1 differs from the primary in
   ``pass2_population`` only and is never accepted as the primary.
2. The train side: the shared fold plan and identities; a mismatch with the
   frozen population is reported.
3. Fit-only fold (G3 preflight): both arms fitted on the fold's training rows,
   evidence recorded, no held-out row scored, no outer-OOF metric computed.
   Feasibility, G3 (feasible and every fit status accepted) and the mandatory
   reviews are kept apart; the authorization cannot pass a failed G3 or skip a
   required review.
4. G8: a selection that is infeasible, unsolved or unsupported is recorded with
   its stop outcome and the fold is not accepted; the other arm still runs;
   nothing falls back.
5. Cross-fit, full-train fits and saving (G2, G5, G10): evidence taken before
   each fold is scored, no held-out label read and no outer-OOF metric
   computed, artifacts in the Stage 4 layouts with the evidence inside,
   accepted by the Stage 4 contract after a reload, a SHA-256 manifest, and no
   write into an existing directory. A fold that is not accepted stops the run
   before it is scored; nothing is saved without a complete FoldLedger and
   accepted full-train fits; saved artifacts are candidates (not accepted)
   until structural acceptance (Commit 3A).
6. The five-fold primary needs the Step 2B authorization record.

Synthetic data; the GLASS rule search and ILP are scripted (a real GLASS fit
takes ~30 s per fold); the RF Router is real with small forests.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json

import joblib
import numpy as np
import pytest

import glass_pipeline.glass_router.pipeline.crossfit as glass_crossfit

import stage2_crossfit.evidence as E
import stage2_crossfit.runner as R
from blackbox_pipeline.models.rf_router.contract import check_rf_constraints_follow_contract
from glass_pipeline.glass_router import profile as glass_profile
from glass_pipeline.glass_router.contract import check_glass_config_follows_contract
from blackbox_pipeline.models.rf_router import RFRouter
from shared.stage2.contract import ContractViolation, load_frozen_contract
from shared.stage4 import router_stream
from tests.support import stage2_crossfit_world as W
from tests.support import authorization_world as S
from tests.support.rf_router_world import serial_forests


# ======================================================================
# 1. strict settings
# ======================================================================
def test_the_primary_settings_come_from_the_strict_entry_points_and_pass_every_check():
    s = R.primary_settings()
    assert s.role == R.ROLE_PRIMARY
    glass_profile.check_glass_primary_config(s.glass_config)
    glass_profile.check_glass_primary_ranker(**s.ranker)
    first, second = s.rf_config_factory(), s.rf_config_factory()
    assert first is not second and first.to_dict() == second.to_dict()   # one factory, fresh configs
    for key in ("G1_glass_f17", "G1_rf_f17", "G1_glass_fields_set_by_contract"):
        assert s.checks[key]["passed"], key
    assert s.checks["G9_rf_constraint_provenance"]["constraint_source"] == "shared_contract"
    assert (s.checks["G9_rf_constraint_provenance"]["contract_fingerprint"]
            == s.checks["G9_glass_contract_provenance"]["contract_fingerprint"]
            == "0817825b9eecbd11")
    ids = s.checks["identities"]
    assert ids["protocol"][2] == "245a1a68fb68cf4e" and ids["contract"][2] == "0817825b9eecbd11"
    record = R.settings_record(s)
    assert json.loads(json.dumps(record))["rf_config"]["eligibility"] == "nested_t1"


def test_s1_differs_from_the_primary_in_pass2_population_only_and_never_passes_as_it():
    primary, s1 = R.primary_settings(), R.s1_settings()
    a, b = primary.glass_config.to_dict(), s1.glass_config.to_dict()
    assert sorted(k for k in a if a[k] != b[k]) == ["pass2_population"]
    assert b["pass2_population"] == "full" and b["inner_feature_ranker"] == "fold_nested_rf"
    assert s1.ranker == primary.ranker and s1.rf_config_factory is None
    with pytest.raises(ContractViolation, match="pass2_population"):
        glass_profile.check_glass_primary_config(s1.glass_config)
    with pytest.raises(ContractViolation, match="F17 primary"):
        check_glass_config_follows_contract(s1.glass_config, load_frozen_contract())
    with pytest.raises(ContractViolation, match="not the S1"):
        glass_profile.check_glass_s1_config(primary.glass_config)
    glass_profile.check_glass_s1_config(s1.glass_config)


# ======================================================================
# 2. the train side
# ======================================================================
def test_the_train_side_carries_the_shared_fold_plan_and_its_identities():
    ts = W.synthetic_train_side()
    assert sorted(np.unique(ts.fold_id)) == [0, 1, 2, 3, 4]
    for k, (tr, va) in enumerate(ts.folds):
        assert not np.intersect1d(tr, va).size and (ts.fold_id[va] == k).all()
    X_tr, y_tr, X_va = ts.fold_rows(0)
    assert X_tr.index.equals(y_tr.index) and not X_tr.index.isin(X_va.index).any()


def test_a_train_side_that_is_not_the_pinned_population_is_reported():
    ts = W.synthetic_train_side()
    protocol = R.primary_settings().protocol
    out = R.check_population(ts, protocol)
    assert out["passed"] is False
    assert out["checks"]["rows"]["passed"] is False
    assert out["checks"]["train_label_fingerprint"]["pinned"] == "b984d2a50527"


# ======================================================================
# 3. fit-only fold
# ======================================================================
@pytest.fixture
def fit0(monkeypatch):
    cls = W.wired(monkeypatch, W.wired_glass_class())
    ts = W.synthetic_train_side()
    forbidden = []
    from blackbox_pipeline.models.rf_router import RFRouter

    def no_routing(self, X):
        forbidden.append(len(X))
        raise AssertionError("fit-only: the RF Router routed rows")

    monkeypatch.setattr(RFRouter, "predict", no_routing)
    with serial_forests():
        res = R.fit_fold(ts, W.make_settings(), 0, keep_fits=True)
    return res, ts, forbidden, cls


def test_a_fit_only_fold_records_both_arms_and_scores_no_held_out_row(fit0):
    res, ts, forbidden, _ = fit0
    assert set(res["arms"]) == {E.ARM_GLASS, E.ARM_RF} and forbidden == []
    assert res["fits"][E.ARM_GLASS].glass.n_predict == 0
    held = set(ts.X_train.index[ts.folds[0][1]])
    for arm, ev in res["arms"].items():
        assert "stop" not in ev and ev["fit_status"]["status"] == "valid_optimal", arm
        assert ev["isolation"]["held_out_rows_in_fit"] == 0
        assert not held & set(ev["remainder"]["index"])
        assert ev["wall_seconds"] >= 0
        # no outer-OOF metric anywhere in the record
        text = json.dumps(ev)
        for token in ("auc", "pass1_rate", "pass2_rate", "abstain_rate", "recall"):
            assert token not in text, (arm, token)


PROTOCOL = R.ep.load_frozen_protocol()


def test_the_preflight_checks_follow_g3(fit0):
    res = fit0[0]
    ok = R.preflight_checks(res, remainder_min_size=50, protocol=PROTOCOL)
    assert ok["feasible"] and ok["passed"]
    assert ok["arms"][E.ARM_RF]["checks"]["r1_supports_at_or_above_floors"]
    assert ok["arms"][E.ARM_RF]["checks"]["f31_strip_review"]["basis"] == E.R2_BASIS
    assert ok["authorizable"] == (ok["reviews_required"] == [])
    strict = R.preflight_checks(res, protocol=PROTOCOL)  # the F17 floor, 500 rows
    small = [a for a, v in strict["arms"].items()
             if not v["checks"]["remainder_at_least_min_with_both_classes"]]
    assert small and not strict["feasible"] and not strict["passed"]


def _with_status(ev, status, accepted):
    return {**ev, "fit_status": {**ev["fit_status"], "status": status, "accepted": accepted}}


def test_a_feasible_fold_whose_status_is_not_accepted_does_not_pass_g3(fit0):
    res = fit0[0]
    arms = dict(res["arms"])
    arms[E.ARM_GLASS] = _with_status(arms[E.ARM_GLASS], "fallback_unreviewed", False)
    g3 = R.preflight_checks({"arms": arms}, remainder_min_size=50, protocol=PROTOCOL)
    glass = g3["arms"][E.ARM_GLASS]
    assert glass["feasible"] and glass["fit_status_accepted"] is False and not glass["passed"]
    assert g3["feasible"] and g3["passed"] is False and g3["authorizable"] is False


def test_a_triggered_f31_review_is_required_but_does_not_fail_g3(fit0):
    res = fit0[0]
    arms = dict(res["arms"])
    rf = arms[E.ARM_RF]
    band = rf["r2_strips"]["band_rows"]
    arms[E.ARM_RF] = {**rf, "r2_strips": {**rf["r2_strips"],
                                          "strip_a_in_band": {"rows": band, "positives": 0}}}
    g3 = R.preflight_checks({"arms": arms}, remainder_min_size=50, protocol=PROTOCOL)
    assert g3["passed"] is True and g3["authorizable"] is False
    assert g3["reviews_required"] == ["F31:blackbox:fold0:triggered"]
    arms[E.ARM_RF] = {**rf, "r2_strips": {**rf["r2_strips"], "band_rows": 0,
                                          "strip_a_in_band": {"rows": 0, "positives": 0}}}
    empty = R.preflight_checks({"arms": arms}, remainder_min_size=50, protocol=PROTOCOL)
    assert empty["reviews_required"] == ["F31:blackbox:fold0:undefined_empty_band"]


def test_a_contaminated_fit_is_not_feasible(fit0):
    arms = dict(fit0[0]["arms"])
    rf = arms[E.ARM_RF]
    arms[E.ARM_RF] = {**rf, "isolation": {**rf["isolation"], "held_out_rows_in_fit": 3}}
    g3 = R.preflight_checks({"arms": arms}, remainder_min_size=50, protocol=PROTOCOL)
    assert g3["arms"][E.ARM_RF]["checks"]["isolation_clean"] is False
    assert not g3["feasible"] and not g3["passed"]


# ======================================================================
# 4. G8: explicit rejection, recorded
# ======================================================================
@pytest.mark.parametrize("action, outcome", [("infeasible", "proven_infeasible"),
                                             ("no_solution", "no_feasible_solution")])
def test_a_glass_selection_that_stops_is_recorded_and_the_rf_arm_still_runs(
        monkeypatch, action, outcome):
    W.wired(monkeypatch, W.wired_glass_class(script={3: action}))     # inner fold 1
    with serial_forests():
        res = R.fit_fold(W.synthetic_train_side(), W.make_settings(), 0)
    g = res["arms"][E.ARM_GLASS]
    assert g["stop"]["fit_status"] == outcome and g["fit_status"]["accepted"] is False
    inner = g["selection_reports"]["inner_pass1"]
    assert len(inner) == 2 and inner[-1]["source"] == "infeasible"          # stays visible
    assert g["selection_reports"]["pass2"] is None                          # nothing after it
    assert res["arms"][E.ARM_RF]["fit_status"]["status"] == "valid_optimal"
    assert R.preflight_checks(res, protocol=PROTOCOL)["passed"] is False


def test_an_infeasible_rf_threshold_is_recorded_as_proven_infeasible(monkeypatch):
    W.wired(monkeypatch, W.wired_glass_class())
    strict = W._v2(min_pass1_npv=0.999, max_pass1_leakage_rate=0.001, min_pass2_precision=0.99,
                   min_pass1_support=20, min_pass2_support=10)
    settings = W.make_settings(rf_factory=lambda: W.rf_test_config(strict))
    with serial_forests():
        res = R.fit_fold(W.synthetic_train_side(), settings, 0, arms=(E.ARM_RF,))
    ev = res["arms"][E.ARM_RF]
    assert ev["stop"]["outcome"] == "proven_infeasible" and "No feasible" in ev["stop"]["reason"]


def test_an_insufficient_remainder_is_recorded_as_insufficient(monkeypatch):
    W.wired(monkeypatch, W.wired_glass_class())
    glass = dataclasses.replace(glass_profile.glass_primary_config(), remainder_min_size=10_000)
    with serial_forests():
        res = R.fit_fold(W.synthetic_train_side(), W.make_settings(glass_config=glass), 0,
                         arms=(E.ARM_GLASS,))
    stop = res["arms"][E.ARM_GLASS]["stop"]
    assert stop["outcome"] == "insufficient_population_or_support"
    assert stop["fit_status"] == "insufficient_population_or_support"


def test_nothing_falls_back_after_a_stop(monkeypatch):
    cls = W.wired(monkeypatch, W.wired_glass_class(script={2: "infeasible"}))
    built = []
    real_init = cls.__init__

    def spy(self, config, *a, **k):
        built.append(config.pass2_population)
        real_init(self, config, *a, **k)

    monkeypatch.setattr(cls, "__init__", spy)
    with serial_forests():
        R.fit_fold(W.synthetic_train_side(), W.make_settings(), 0, arms=(E.ARM_GLASS,))
    assert built == ["oof_remainder"]                     # no second, "full" fit


# ======================================================================
# 5. cross-fit, full-train fits, saving
# ======================================================================
@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    W.wired(mp, W.wired_glass_class())

    def no_outer_metric(*a, **k):
        raise AssertionError("an outer-OOF AUC was computed")

    mp.setattr(glass_crossfit, "roc_auc_score", no_outer_metric)
    ts, settings = W.synthetic_train_side(), W.make_settings()
    out = io.StringIO()
    try:
        with serial_forests(), contextlib.redirect_stdout(out):
            res = R.run_crossfit(ts, settings)
            R.fit_full_train(ts, settings, res)
            root = tmp_path_factory.mktemp("pr38") / "run"
            info = R.save_run(res, ts, settings, root, "t1")
    finally:
        mp.undo()
    return res, ts, settings, info, out.getvalue()


def test_each_folds_evidence_is_recorded_before_it_is_scored_and_nothing_is_printed(saved):
    res, ts, _, _, printed = saved
    for arm in (E.ARM_GLASS, E.ARM_RF):
        evs = res.evidence[arm]
        assert [ev["fold"] for ev in evs] == [0, 1, 2, 3, 4]
        assert all(ev["fit_status"]["status"] == "valid_optimal" for ev in evs)
    for token in ("OOF GLASS", "leak", "pass1 ", "RF AUC"):
        assert token not in printed, token


def test_no_held_out_label_is_read_and_no_outer_metric_kept(saved):
    res = saved[0]
    outer = {"rf_auc", "pass1_rate", "pass2_rate", "abstain_rate", "pass1_leakage",
             "pass2_recall"}
    for summary in res.glass_oof.fold_summaries:
        assert not outer & set(summary)
        assert {"n_val", "glass_pass1_rules", "remainder_rows"} <= set(summary)
    assert not {"pass1", "pass2", "abstain", "leak"} & set(res.rf_crossfit.folds.columns)


def test_the_saved_artifacts_are_candidates_with_a_complete_ledger(saved):
    _, _, _, info, _ = saved
    man = json.loads((info["root"] / "MANIFEST.json").read_text(encoding="utf-8"))
    ev = json.loads((info["root"] / "evidence.json").read_text(encoding="utf-8"))
    for rec in (man["acceptance"], ev["acceptance"],
                joblib.load(info["written"]["glass_router"])["pr38_evidence"]["acceptance"],
                joblib.load(info["written"]["rf_router"])["extra"]["pr38_evidence"]["acceptance"]):
        assert rec["accepted"] is False and rec["structural_acceptance"] is None
        assert rec["fold_ledger_complete"] is True and "3A" in rec["pending"]
    assert all(row["accepted"] == 5 for row in ev["fold_ledger"])


def test_a_glass_fold_that_is_not_accepted_stops_the_run_before_it_is_scored(monkeypatch):
    cls = W.wired(monkeypatch, W.wired_glass_class(script={1: "unproven"}))
    scored = []
    monkeypatch.setattr(cls, "predict", lambda self, X: scored.append(len(X)))
    res = R.RunResult("test")
    with serial_forests(), pytest.raises(R.FoldNotAccepted, match="glass fold 0"):
        R.run_crossfit(W.synthetic_train_side(), W.make_settings(), result=res)
    assert scored == []                                         # fold 0 never scored
    assert [ev["fold"] for ev in res.evidence[E.ARM_GLASS]] == [0]
    assert res.evidence[E.ARM_GLASS][0]["fit_status"]["status"] == "valid_unproven_undocumented"
    assert E.ARM_RF not in res.evidence                         # nothing after the stop


def test_an_rf_fold_that_is_not_accepted_stops_the_run_before_it_is_scored(monkeypatch):
    scored = []
    monkeypatch.setattr(RFRouter, "predict_with_proba", lambda self, X: scored.append(len(X)))
    # the router fits on the loose constraints, but is classified against the frozen contract
    settings = W.make_settings(contract=load_frozen_contract(),
                               rf_factory=lambda: W.rf_test_config(W.LOOSE))
    res = R.RunResult("test")
    with serial_forests(), pytest.raises(R.FoldNotAccepted, match="blackbox fold 0"):
        R.run_crossfit(W.synthetic_train_side(), settings, arms=(E.ARM_RF,), result=res)
    assert scored == []
    assert res.evidence[E.ARM_RF][0]["fit_status"]["status"] == "fit_time_invalid"


def _variant(res, **changes):
    return dataclasses.replace(res, **changes)


def test_nothing_is_saved_without_a_complete_ledger_and_accepted_full_train_fits(saved, tmp_path):
    res, ts, settings, _, _ = saved
    glass = res.evidence[E.ARM_GLASS]
    stopped = E.stop_evidence(E.ARM_GLASS, 3, ValueError("Remainder is single-class (0/10)"))
    stopped["fit_status"] = E.fold_status(stopped, settings.contract, settings.protocol)
    cases = {
        "a stopped fold": _variant(res, evidence={**res.evidence,
                                                  E.ARM_GLASS: glass[:3] + [stopped] + glass[4:]}),
        "a missing fold": _variant(res, evidence={**res.evidence, E.ARM_GLASS: glass[:4]}),
        "a relabelled status": _variant(res, evidence={**res.evidence, E.ARM_GLASS: [
            {**ev, "selection_reports": {**ev["selection_reports"], "pass2": None},
             "fit_status": {**ev["fit_status"], "status": "valid_optimal", "accepted": True}}
            if ev["fold"] == 1 else ev for ev in glass]}),
        "a full-train fit not accepted": _variant(res, full_train_evidence={
            **res.full_train_evidence, E.ARM_RF: {**res.full_train_evidence[E.ARM_RF],
                                                  "fit_status": {"status": "fit_time_invalid",
                                                                 "accepted": False}}}),
    }
    for name, bad in cases.items():
        root = tmp_path / name.replace(" ", "_")
        with pytest.raises(R.RunnerError, match="not saved"):
            R.save_run(bad, ts, settings, root, "t2")
        assert not root.exists(), name                          # nothing written


def test_the_saved_artifacts_carry_the_evidence_and_reload(saved):
    _, ts, settings, info, _ = saved
    g = joblib.load(info["written"]["glass_router"])
    r = joblib.load(info["written"]["rf_router"])
    assert g["pr38_evidence"]["role"] == "test" and len(g["pr38_evidence"]["folds"]) == 5
    assert g["pr38_evidence"]["full_train"]["fold"] == "full_train"
    assert g["oof_protocol"]["glass_pass2_population"] == "oof_remainder"
    assert g["fit_report"]["pass2_population"] == "oof_remainder"
    ev = r["extra"]["pr38_evidence"]
    assert len(ev["folds"]) == 5 and ev["folds"][0]["r2_strips"]["band_rows"] > 0
    summary = r["extra"]["oof_fold_summary"]
    assert all({"pass1_confidence", "pass2_support"} <= set(row) for row in summary)   # R1
    assert r["extra"]["oof_protocol"]["pass2_eligibility"] == r["config"]["eligibility"] \
        == "nested_t1"                                                                  # R5
    check_rf_constraints_follow_contract(r["constraints"], settings.contract,
                                         extra=r["extra"])                              # G9
    assert "test_metrics" not in g and r["test_metrics"] is None


def test_the_saved_evidence_reclassifies_exactly_after_a_reload(saved):
    _, _, settings, info, _ = saved
    ev = json.loads((info["root"] / "evidence.json").read_text(encoding="utf-8"))
    for arm in (E.ARM_GLASS, E.ARM_RF):
        for fold_ev in ev["folds"][arm]:
            fit = E.classify_fold(fold_ev, settings.contract, settings.protocol)
            assert fit.status == fold_ev["fit_status"]["status"] == "valid_optimal"


def test_the_stage4_contract_accepts_both_artifacts_after_a_reload(saved):
    _, ts, _, info, _ = saved
    y = ts.y_train.to_numpy()
    for name, path in info["written"].items():
        stream, checks = router_stream(str(path), name, name, y)
        failed = [c[2] for c in checks if not c[3]()]
        assert failed == [], (name, failed)
    rf_stream, _ = router_stream(str(info["written"]["rf_router"]), "rf", "RF", y)
    assert rf_stream.notes["pass2_eligibility"] == {"cross_fit": "nested_t1",
                                                    "full_train": "nested_t1"}


def test_the_manifest_hashes_every_file_and_no_run_overwrites_one(saved):
    res, ts, settings, info, _ = saved
    root = info["root"]
    man = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, meta in man["files"].items():
        assert R.sha256_file(root / rel) == meta["sha256"], rel
    assert (root / "MANIFEST.json.sha256").read_text().strip() == R.sha256_file(root / "MANIFEST.json")
    with pytest.raises(R.RunnerError, match="exists"):
        R.save_run(res, ts, settings, root, "t1")
    with pytest.raises(FileExistsError):
        R.write_record(root / "evidence.json", {})


# ======================================================================
# 6. authorization
# ======================================================================
def _authorization(tmp_path, *, g3=None, g13=None, **fields):
    """A complete authorization (Step 2B evidence: ``authorization_world``); ``g3`` / ``g13`` override
    the G3 block and the original G13 record, ``fields`` the authorization's own fields."""
    bodies = S.records()
    real = bodies["g3"]["g3"]
    bodies["g3"]["g3"] = g3 or {**real, "reviews_required": sorted(
        {*real["reviews_required"], S.F31_REVIEW_ID})}
    if g13 is not None:
        bodies["g13"] = g13
    return S.authorization_files(tmp_path, bodies=bodies, **fields)


def test_the_five_fold_primary_needs_every_step_2b_gate_passed(tmp_path):
    with pytest.raises(R.RunnerError, match="no authorization"):
        R.require_authorization(tmp_path / "authorization.json")
    gates = {g: "pass" for g in R.AUTHORIZATION_GATES}
    refused = {
        "G13": dict(gates={**gates, "G13": "pending"}),
        "approver": dict(approved_by=""),
        "g3_record": dict(g3_record=None),
        "changed": dict(g3_record_sha256="0" * 64),
        "mandatory reviews": dict(reviews={}),
        "G13 materiality": dict(g13_decision={"decision": "", "reviewer": "owner",
                                              "rationale": "x", "date": "2026-11-12"}),
    }
    for match, fields in refused.items():
        sub = tmp_path / match.replace(" ", "_")
        sub.mkdir()
        with pytest.raises(R.RunnerError, match=match):
            R.require_authorization(_authorization(sub, **fields))
    ok = tmp_path / "ok"
    ok.mkdir()
    assert R.require_authorization(_authorization(ok))["gates"] == gates


def test_no_authorization_can_pass_a_g3_the_record_failed(tmp_path):
    with pytest.raises(R.RunnerError, match="did not pass"):
        R.require_authorization(_authorization(tmp_path, g3={"passed": False,
                                                             "reviews_required": []}))
    other = tmp_path / "fmt"
    other.mkdir()
    with pytest.raises(R.RunnerError, match="not 'pr38.stage2.g13_fold_fits/2'"):
        R.require_authorization(_authorization(other, g13={"format": "pr38.stage2.g13_fold_fits/1"}))
