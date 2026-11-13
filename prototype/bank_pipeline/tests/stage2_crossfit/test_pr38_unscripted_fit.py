"""
PR 38 Step 2A, opt-in (``pytest -m slow``, a few minutes): the runner's
fit-only fold with a REAL GLASS Router (rule search, evaluation, CBC) under
``oof_remainder`` + ``fold_nested_rf`` and a real RF Router under
``nested_t1``, on synthetic data with a loose contract and a 30 s solve limit.

The evidence must recount every selected union exactly as the selector
reported it, carry real CBC phase records, classify completely under the
frozen protocol (D-C11), and the G13 audit must run on both arms' fits,
both channels (the retrained one re-runs the real inner fits).
"""

from __future__ import annotations

import dataclasses

import pytest

import glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector as ilp_mod
import stage2_crossfit.evidence as E
import stage2_crossfit.g13 as G
import stage2_crossfit.runner as R
from glass_pipeline.glass_router import profile as glass_profile
from glass_pipeline.glass_router.contract import glass_config_from_contract
from tests.support import stage2_crossfit_world as W
from tests.support.rf_router_world import serial_forests

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_fold():
    gs = W.split_frame(W.make_bank_frame(n=3000, seed=7, positive_rate=0.12), test_size=0.2)
    ts = R.make_train_side(gs["X_train"], gs["y_train"].astype(int), gs["X_test"])
    base = dataclasses.replace(
        glass_profile.glass_base_config(glass_profile.load_glass_primary_profile()),
        min_support_pass1=40, min_support_pass2=15)
    glass = dataclasses.replace(glass_config_from_contract(base, W.LOOSE), remainder_min_size=100)
    settings = W.make_settings(glass_config=glass)
    with pytest.MonkeyPatch.context() as mp, serial_forests():
        mp.setattr(ilp_mod.ILPRuleSelector, "TIME_LIMIT_PER_SOLVE", 30)
        res = R.fit_fold(ts, settings, 0, keep_fits=True)
    return ts, settings, res


def test_a_real_glass_fold_is_recounted_exactly_and_classified_completely(real_fold):
    ts, settings, res = real_fold
    ev = res["arms"][E.ARM_GLASS]
    if "stop" in ev:
        pytest.skip(f"the synthetic fold stopped ({ev['stop']}); recorded, as G8 requires")
    reports = ev["selection_reports"]
    slots = {"pass1": reports["pass1"], "pass2": reports["pass2"],
             **{f"inner_pass1[{j}]": r for j, r in enumerate(reports["inner_pass1"])}}
    for key, rep in slots.items():
        c = ev["counts"][key]
        good = c["routed_rows"] - c["routed_positives"] if key != "pass2" else c["routed_positives"]
        assert (c["routed_rows"], good) == (rep["union_rows"], rep["union_good"]), key
        assert rep["phase1"]["status"] is not None, key                       # real CBC record
    assert ev["fit_status"]["problems"] == []
    assert all(ev["isolation"]["inner_rankers_fitted_on_their_inner_train_rows"])


def test_the_g13_audit_runs_on_both_real_fits(real_fold, monkeypatch):
    ts, _, res = real_fold
    if any("stop" in ev for ev in res["arms"].values()):
        pytest.skip("a synthetic fold stopped")
    g = res["fits"][E.ARM_GLASS]
    router, rf_bins, _ = res["fits"][E.ARM_RF]
    X_tr, y_tr, _ = ts.fold_rows(0)
    monkeypatch.setattr(ilp_mod.ILPRuleSelector, "TIME_LIMIT_PER_SOLVE", 30)
    with serial_forests():
        rec = G.audit_fold(X_tr, y_tr, fold=0, glass=g.glass, glass_bins=g.X_tr_eng,
                           rf_router=router, rf_bins=rf_bins)
    assert rec["inner_folds_shared_by_both_arms"]
    assert rec["eligibility"]["rf"]["all"]["max_score_reproduction_difference"] <= 1e-9
    # the retrained channel re-runs the real inner fits: CBC and the RF t1_j solves
    rf = rec["eligibility_retrained"]["rf"]
    assert rf["t1_reproduction_max_difference"] <= 1e-9
    glass = rec["eligibility_retrained"]["glass"]
    for f in glass["folds"]:
        if f["base_reselection"]["lexicographic_guarantee"] == "proven":
            assert f["reproduced"], f                  # a proven optimum re-runs to itself
