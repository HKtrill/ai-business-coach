"""
PR 38 Step 2A: fit-time evidence and selection records (``stage2_crossfit.evidence``).

1. GLASS (D-C11, G5, M6): every selection's union is recounted by matching the
   selected rules on the rows it was selected on; the records are
   ``glass_fit_outcome`` of the reports; a feasible claim the recount
   contradicts makes the fold ``fit_time_invalid``; an unproven incumbent is
   not accepted until documented; a greedy fallback needs its review.
2. RF Router (G5, R1, R2, R5): ``t1``, ``t2`` and every ``t1_j`` band recounted
   and classified; the R1 band record; the R2 strips match an independent
   recomputation and partition per inner fold; the eligibility provenance.
3. The evidence is JSON; records rebuilt after a round trip classify the same.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import stage2_crossfit.evidence as E
import stage2_crossfit.runner as R
import shared.stage2.evaluation_protocol as ep
from shared.stage2.contract import load_frozen_contract
from tests.support import pr38_world as W
from tests.support.rf_router_world import serial_forests


def _fold(monkeypatch, script=None, arms=(E.ARM_GLASS, E.ARM_RF), contract=W.LOOSE):
    W.wired(monkeypatch, W.wired_glass_class(script=script))
    with serial_forests():
        return R.fit_fold(W.synthetic_train_side(), W.make_settings(contract), 0, arms=arms,
                          keep_fits=True)


@pytest.fixture
def fold(monkeypatch):
    return _fold(monkeypatch)


# ======================================================================
# 1. GLASS
# ======================================================================
def test_glass_records_are_the_reports_classification_with_recounted_unions(fold):
    ev = fold["arms"][E.ARM_GLASS]
    fit = fold["fits"][E.ARM_GLASS]
    reports = ev["selection_reports"]
    assert len(reports["inner_pass1"]) == 5
    for j, rep in enumerate(reports["inner_pass1"]):
        c = ev["counts"][f"inner_pass1[{j}]"]
        assert (c["routed_rows"], c["routed_rows"] - c["routed_positives"]) == (
            rep["union_rows"], rep["union_good"])                      # recount == selector
        assert c["population_rows"] == int((fit.glass.inner_fold_id_ != j).sum())
    assert ev["counts"]["pass2"]["population_rows"] == ev["remainder"]["rows"]
    records = E.selection_records(ev, W.LOOSE)
    assert sorted(r.key for r in records) == sorted(ep.required_selection_keys(
        "glass", ep.load_frozen_protocol()))
    m6 = ev["m6_inner_oof_pass1"]
    assert m6["routed_rows"] + ev["remainder"]["rows"] == ev["n_train"]


def test_a_feasible_claim_the_recount_contradicts_makes_the_fold_fit_time_invalid(fold):
    ev = fold["arms"][E.ARM_GLASS]
    frozen = load_frozen_contract()                 # NPV 0.95: the scripted union fails it
    records = E.selection_records(ev, frozen)
    assert any(isinstance(r.outcome, dict) and "invalid" in r.outcome for r in records)
    status = E.fold_status(ev, frozen, ep.load_frozen_protocol())
    assert status["status"] == "fit_time_invalid" and not status["accepted"]


def test_an_unproven_incumbent_is_recorded_but_not_accepted_until_documented(monkeypatch):
    ev = _fold(monkeypatch, script={1: "unproven"}, arms=(E.ARM_GLASS,))["arms"][E.ARM_GLASS]
    assert ev["fit_status"]["by_key"]["pass1"] == "valid_unproven_undocumented"
    assert ev["fit_status"]["accepted"] is False
    g3 = R.preflight_checks({"arms": {E.ARM_GLASS: ev}}, 50, protocol=ep.load_frozen_protocol())
    arm = g3["arms"][E.ARM_GLASS]
    assert "M3" in arm["note"] and arm["feasible"] and not arm["passed"]   # feasible, not G3
    assert g3["passed"] is False and g3["authorizable"] is False


def test_a_greedy_fallback_needs_its_fit_time_review(monkeypatch):
    ev = _fold(monkeypatch, script={7: "greedy"}, arms=(E.ARM_GLASS,))["arms"][E.ARM_GLASS]
    protocol = ep.load_frozen_protocol()
    assert ev["fit_status"]["by_key"]["pass2"] == "fallback_unreviewed"
    reviewed = E.fold_status(ev, W.LOOSE, protocol,
                             reviews={"pass2": {"reviewer": "owner", "rationale": "gap reviewed"}})
    assert reviewed["by_key"]["pass2"] == "fallback_accepted_after_review"
    assert reviewed["accepted"] is True


def test_missing_glass_reports_make_the_fold_invalid(fold):
    ev = dict(fold["arms"][E.ARM_GLASS])
    ev["selection_reports"] = {**ev["selection_reports"],
                               "inner_pass1": ev["selection_reports"]["inner_pass1"][:4]}
    status = E.fold_status(ev, W.LOOSE, ep.load_frozen_protocol())
    assert status["status"] == "fit_time_invalid" and status["problems"]


# ======================================================================
# 2. RF Router
# ======================================================================
def test_rf_thresholds_and_every_t1_j_are_recounted_and_classified(fold):
    ev = fold["arms"][E.ARM_RF]
    router, X_eng, y = fold["fits"][E.ARM_RF]
    y = y.to_numpy()
    p1 = router.oof_pass1_proba_
    assert ev["counts"]["pass1"]["routed_rows"] == int((p1 < ev["thresholds"]["t1"]).sum())
    elig = ev["eligibility"]
    assert elig["rule"] == elig["report_rule"] == "nested_t1" and len(elig["fold_t1"]) == 5
    for j, c in zip(elig["fold_ids"], elig["fold_counts"]):
        assert c["population_rows"] == int((router.oof_pass1_fold_id_ != j).sum())
        assert ev["counts"][f"eligibility_t1[{j}]"]["routed_rows"] == c["band_rows"]
    assert ev["isolation"]["eligibility_solves_exclude_their_fold"] == [True] * 5     # G4
    records = E.selection_records(ev, W.LOOSE)
    assert sorted(r.key for r in records) == sorted(ep.required_selection_keys(
        "blackbox", ep.load_frozen_protocol()))
    assert all(r.outcome.outcome == "feasible_optimal" for r in records)


def test_the_r1_band_record_is_the_routers_bands_with_their_floors(fold):
    ev = fold["arms"][E.ARM_RF]
    router = fold["fits"][E.ARM_RF][0]
    bands = router.bands_.to_dict()
    for k in ("pass1_confidence", "pass2_confidence", "pass1_support", "pass2_support"):
        assert ev["r1_bands"][k] == bands[k]
    assert ev["r1_bands"]["pass1_support_floor"] == W.LOOSE.min_pass1_support


def test_the_r2_strips_match_an_independent_recomputation(fold):
    ev = fold["arms"][E.ARM_RF]
    router, _, y = fold["fits"][E.ARM_RF]
    y = y.to_numpy()
    p1, p2 = router.oof_pass1_proba_, router.oof_pass2_proba_
    t1, t2 = router.thresholds_.t1, router.thresholds_.t2
    rep = router.fit_report_.remainder
    a_rows = b_rows = a_band = 0
    for j, t1j in zip(rep["eligibility_fold_ids"], rep["eligibility_fold_t1"]):
        rows = router.oof_pass1_fold_id_ == j
        a = rows & (p1 >= t1j) & (p1 < t1)
        b = rows & (p1 >= t1) & (p1 < t1j)
        a_rows += int(a.sum()); b_rows += int(b.sum())
        a_band += int((a & (np.nan_to_num(p2, nan=-1.0) > t2)).sum())
    s = ev["r2_strips"]
    assert (s["strip_a"]["rows"], s["strip_b"]["rows"], s["strip_a_in_band"]["rows"]) == (
        a_rows, b_rows, a_band)
    assert sum(f["strip_a"]["rows"] for f in s["per_inner_fold"]) == a_rows
    assert s["band_rows"] == router.bands_.pass2_support
    review = ep.f31_strip_review({0: (s["strip_a_in_band"]["rows"], s["band_rows"])},
                                 ep.load_frozen_protocol())
    assert review.loc[0, "share"] == pytest.approx(s["strip_a_share_of_band"])
    assert s["interpretation"] == E.R2_BASIS and s["interpretation"].startswith("provisional")


def test_rf_evidence_refuses_labels_the_router_was_not_fitted_on(fold):
    router, X_eng, y = fold["fits"][E.ARM_RF]
    with pytest.raises(ValueError, match="not the labels"):
        E.rf_fold_evidence(router, X_eng, 1 - y, fold=0)


def test_the_nested_eligibility_counts_are_recounted_from_the_scores(fold):
    router = fold["fits"][E.ARM_RF][0]
    rep = router.fit_report_.remainder
    assert {"eligibility_fold_ids", "eligibility_fold_counts"} <= set(rep)
    assert [c["population_rows"] - c["band_rows"] for c in rep["eligibility_fold_counts"]]


# ======================================================================
# 3. JSON
# ======================================================================
def test_to_json_safe_keeps_arrays_and_exact_fractions():
    obj = {"a": np.int64(3), "b": np.array([1.5, np.nan]), "c": pd.Index([7, 8]),
           "d": Fraction(3, 20), "e": (np.bool_(True),), "f": {1: {2}}}
    out = E.to_json_safe(obj)
    assert json.loads(json.dumps(out)) == {"a": 3, "b": [1.5, None], "c": [7, 8], "d": "3/20",
                                           "e": [True], "f": {"1": [2]}}


def test_records_rebuilt_after_a_json_round_trip_classify_identically(fold):
    protocol = ep.load_frozen_protocol()
    for arm, ev in fold["arms"].items():
        again = json.loads(json.dumps(E.to_json_safe(ev)))
        assert E.fold_status(again, W.LOOSE, protocol) == ev["fit_status"], arm
