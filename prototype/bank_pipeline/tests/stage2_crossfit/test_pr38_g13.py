"""
PR 38 Step 2A: the G13 feature-engineering audit (``stage2_crossfit.g13``).

1. The label-excluded engineer recomputes only label-derived statistics: with
   every row kept it reproduces the fitted engineer exactly; the KDE reference
   points are identified exactly, the subsampled case included.
2. The property the audit relies on: an inner fold's label-excluded features do
   not move when that fold's labels move, while the fitted engineer's do (so
   the audit can see an own-label dependency when there is one).
3. Eligibility under the label-excluded bins holds every fitted model fixed:
   GLASS reproduces its stored remainder, the RF Router its stored OOF scores
   and remainder; unchanged bins change nothing.
3b. The retrained (indirect) channel: fold j's inner models are refitted on
   features that carry no label of fold j on ANY row (training rows included);
   the re-runs reproduce the stored fit on the base bins (RF: scores and
   ``t1_j``; GLASS: rules); a change confined to the training rows is seen by
   this channel and not by the direct one; a re-selection that stops is
   recorded and the channel marked incomplete.
4. ``audit_fold`` on both arms: the record names the encoders, the affected
   rows and the eligibility changes, excludes each arm's labels on that arm's
   own inner plan, refuses bins the arms were not fitted on, and uses no test
   row.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest

import stage2_crossfit.evidence as E
import stage2_crossfit.g13 as G
import stage2_crossfit.runner as R
from shared.stage2.feature_engineering import RF_SOURCE_FEATURES, RFFeatureEngineer
from tests.support import pr38_world as W
from tests.support.rf_router_world import serial_forests
from tests.support.synthetic import make_bank_frame


def _inner(y, k=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    out = np.full(len(y), -1)
    for j, (_, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), np.asarray(y))):
        out[va] = j
    return out


@pytest.fixture(scope="module")
def fold_data():
    ts = W.synthetic_train_side()
    X_tr, y_tr, _ = ts.fold_rows(0)
    return X_tr, y_tr, RFFeatureEngineer().fit(X_tr, y_tr)


# ======================================================================
# 1. the label-excluded engineer
# ======================================================================
def test_keeping_every_row_reproduces_the_fitted_engineer(fold_data):
    X, y, eng = fold_data
    same = G.label_excluded_engineer(eng, X, y, np.ones(len(y), bool))
    pd.testing.assert_frame_equal(same.transform(X), eng.transform(X))


def test_the_kde_reference_rows_are_identified_exactly_when_subsampled():
    frame = make_bank_frame(n=10_500, seed=1, positive_rate=0.12)
    X, y = frame.drop(columns=["y"]), frame["y"].astype(int)
    eng = RFFeatureEngineer().fit(X, y)
    pos = G.kde_reference_positions(eng, X)
    assert len(pos) == 5_000 and len(np.unique(pos)) == 5_000
    with pytest.raises(AssertionError, match="reference points"):
        G.kde_reference_positions(eng, X.iloc[::-1])


def test_excluding_rows_drops_their_kde_references_and_label_counts_only(fold_data):
    X, y, eng = fold_data
    keep = np.ones(len(y), bool)
    keep[:100] = False
    alt = G.label_excluded_engineer(eng, X, y, keep)
    assert len(alt._integral_eng._X_ref) == len(eng._integral_eng._X_ref) - 100   # all rows are refs
    assert alt._integral_eng._scaler is not eng._integral_eng._scaler
    np.testing.assert_array_equal(alt._integral_eng._scaler.mean_, eng._integral_eng._scaler.mean_)
    cats, _ = eng._derivative_eng._curvature_store["euribor3m"]
    assert alt._derivative_eng._curvature_store["euribor3m"][0].equals(cats)       # edges fixed


# ======================================================================
# 2. what the audit relies on
# ======================================================================
def test_an_inner_folds_label_excluded_features_ignore_that_folds_labels(fold_data):
    X, y, _ = fold_data
    inner = _inner(y)
    rows = inner == 2
    y2 = y.copy()
    y2[rows] = 1 - y2[rows]                                  # every fold-2 label moves
    base = G.fold_excluded_features(RFFeatureEngineer().fit(X, y), X, y, inner)
    moved = G.fold_excluded_features(RFFeatureEngineer().fit(X, y2), X, y2, inner)
    pd.testing.assert_frame_equal(base[rows], moved[rows])   # fold 2 never reads fold 2's labels
    fitted = RFFeatureEngineer().fit(X, y).transform(X)
    fitted2 = RFFeatureEngineer().fit(X, y2).transform(X)
    changed = [f for f in RF_SOURCE_FEATURES
               if not np.allclose(fitted.loc[rows, f], fitted2.loc[rows, f])]
    assert set(changed) == set(G.LABEL_FITTED_FEATURES)      # the outer fit does read them


def test_the_retrained_frames_carry_no_label_of_the_excluded_fold_on_any_row(fold_data):
    X, y, _ = fold_data
    inner = _inner(y)
    rows = inner == 2
    y2 = y.copy()
    y2[rows] = 1 - y2[rows]                                  # every fold-2 label moves
    base = G.fold_excluded_frames(RFFeatureEngineer().fit(X, y), X, y, inner)[2]
    moved = G.fold_excluded_frames(RFFeatureEngineer().fit(X, y2), X, y2, inner)[2]
    pd.testing.assert_frame_equal(base, moved)               # training rows included
    fitted = RFFeatureEngineer().fit(X, y).transform(X[~rows])
    fitted2 = RFFeatureEngineer().fit(X, y2).transform(X[~rows])
    assert not fitted.equals(fitted2)        # the outer fit's TRAINING rows do read fold 2's labels
    direct = G.fold_excluded_features(RFFeatureEngineer().fit(X, y), X, y, inner)
    pd.testing.assert_frame_equal(base[rows], direct[rows])  # fold 2's own rows: same as direct


def test_the_inner_refit_variant_covers_every_row(fold_data):
    X, y, _ = fold_data
    out = G.inner_refit_features(X, y, _inner(y))
    assert out.index.equals(X.index) and list(out.columns) == RF_SOURCE_FEATURES


# ======================================================================
# 3. eligibility with every fitted model fixed
# ======================================================================
@pytest.fixture(scope="module")
def fits():
    mp = pytest.MonkeyPatch()
    W.wired(mp, W.wired_glass_class())
    try:
        with serial_forests():
            res = R.fit_fold(W.synthetic_train_side(), W.make_settings(), 0, keep_fits=True)
    finally:
        mp.undo()
    return res["fits"]


def test_unchanged_bins_change_no_eligibility_and_the_stored_state_reproduces(fits):
    g = fits[E.ARM_GLASS]
    router, rf_bins, y = fits[E.ARM_RF]
    gl = G.glass_eligibility_shift(g.glass, g.X_tr_eng, g.X_tr_eng, g.y_tr)
    rf = G.rf_eligibility_shift(router, rf_bins, rf_bins, y)
    assert gl["rows_changed"] == rf["rows_changed"] == 0
    assert gl["eligible_rows"] == int(g.glass.fit_report_["remainder_mask"].sum())
    assert rf["eligible_rows"] == int(router.remainder_mask_.sum())
    assert rf["max_score_reproduction_difference"] <= 1e-12          # n_jobs=-1: last-ULP only


def test_several_variants_are_scored_with_one_fit_each(fits):
    router, rf_bins, y = fits[E.ARM_RF]
    alt = rf_bins.copy()
    alt.iloc[:, :] = 0
    out = G.rf_eligibility_shift(router, rf_bins, {"same": rf_bins, "zeros": alt}, y)
    assert out["same"]["rows_changed"] == 0 and out["same"]["max_score_change"] <= 1e-12
    assert out["zeros"]["max_score_change"] > 0


def _frames(inner, frame):
    return {int(j): frame for j in np.unique(inner)}


def test_the_rf_retrained_channel_reproduces_t1j_and_unchanged_bins_change_nothing(fits):
    router, rf_bins, y = fits[E.ARM_RF]
    out = G.rf_retrained_shift(router, rf_bins, _frames(router.oof_pass1_fold_id_, rf_bins), y)
    assert out["complete"] and out["cut"] == "t1_j re-solved" and out["rows_changed"] == 0
    assert out["t1_reproduction_max_difference"] <= 1e-9
    for f in out["folds"]:
        assert f["t1_j_retrained"] == pytest.approx(f["t1_j"], abs=1e-9)
        assert f["max_score_change"] <= 1e-12


def test_a_change_confined_to_the_training_rows_is_seen_only_by_the_retrained_channel(fits):
    router, rf_bins, y = fits[E.ARM_RF]
    inner = np.asarray(router.oof_pass1_fold_id_)
    frames = {}
    for j in np.unique(inner):
        f = rf_bins.copy()
        f.loc[inner != j] = 1 - f.loc[inner != j]           # fold j's TRAINING rows only
        frames[int(j)] = f
    own = pd.concat([frames[j][inner == j] for j in frames]).loc[rf_bins.index]
    direct = G.rf_eligibility_shift(router, rf_bins, own, y)
    assert direct["max_score_change"] <= 1e-12 and direct["rows_changed"] == 0   # blind to it
    retrained = G.rf_retrained_shift(router, rf_bins, frames, y)
    assert max(f["max_score_change"] for f in retrained["folds"]) > 0.01


def test_the_glass_retrained_channel_reruns_each_inner_fit_on_its_frame(fits, monkeypatch):
    g = fits[E.ARM_GLASS]
    inner = np.asarray(g.glass.inner_fold_id_)
    frames = {int(j): g.X_tr_eng.copy() for j in np.unique(inner)}
    for j, f in frames.items():
        f.loc[inner != j, "jed_hot"] = 1 - f.loc[inner != j, "jed_hot"]
    calls = []
    real = type(g.glass)._inner_pass1_fit

    def spy(self, k, X_tr, y_tr, *a):
        calls.append((k, X_tr))
        return real(self, k, X_tr, y_tr, *a)

    monkeypatch.setattr(type(g.glass), "_inner_pass1_fit", spy)
    before = (len(g.glass.inner_pass1_selectors_), list(g.glass.inner_pass1_rules_))
    out = G.glass_retrained_shift(g.glass, g.X_tr_eng, frames, g.y_tr)
    assert out["complete"] and out["reproduced_all"] and out["rows_changed"] == 0
    assert len(calls) == 2 * len(frames)                   # base reproduction + retrained, per fold
    for (k, X_tr) in calls[1::2]:
        pd.testing.assert_frame_equal(X_tr, frames[k][inner != k])   # trained on the frame
    assert (len(g.glass.inner_pass1_selectors_), list(g.glass.inner_pass1_rules_)) == before


def test_a_retrained_selection_that_stops_is_recorded_not_replaced(fits, monkeypatch):
    g = fits[E.ARM_GLASS]
    frames = _frames(g.glass.inner_fold_id_, g.X_tr_eng)
    monkeypatch.setattr(g.glass, "script", {g.glass.n_selections + 2: "infeasible"})
    out = G.glass_retrained_shift(g.glass, g.X_tr_eng, frames, g.y_tr)
    first = out["folds"][0]
    assert out["complete"] is False and first["compared"] is False
    assert "SelectorInfeasibleError" in first["retrained_selection"]["error"]
    assert all(f["compared"] for f in out["folds"][1:])


def test_moved_bins_are_reported_row_by_row(fits):
    g = fits[E.ARM_GLASS]
    alt = g.X_tr_eng.copy()
    flip = alt.index[:40]
    alt.loc[flip, "behav_cold"] = 1 - alt.loc[flip, "behav_cold"]   # the Pass 1 rule's bin
    out = G.glass_eligibility_shift(g.glass, g.X_tr_eng, alt, g.y_tr)
    assert out["rows_changed"] == 40 and set(out["rows_changed_index"]) == set(flip)
    assert out["became_eligible"] + out["became_ineligible"] == 40


# ======================================================================
# 4. the fold audit
# ======================================================================
def test_the_fold_audit_records_encoders_rows_and_eligibility_for_both_arms(fits):
    g = fits[E.ARM_GLASS]
    router, rf_bins, _ = fits[E.ARM_RF]
    X_tr, y_tr, _ = W.synthetic_train_side().fold_rows(0)
    rec = G.audit_fold(X_tr, y_tr, fold=0, glass=g.glass, glass_bins=g.X_tr_eng,
                       rf_router=router, rf_bins=rf_bins)
    assert json.loads(json.dumps(E.to_json_safe(rec)))["format"] == G.G13_FORMAT
    assert rec["inner_folds_shared_by_both_arms"] is True and rec["uses_test_rows"] is False
    le = rec["label_excluded"]
    assert set(le["bin_families"]) >= {"nsd", "eci", "dow"}
    for feat, d in le["source_features"].items():
        if not d["label_fitted"]:
            assert d["rows_changed"] == 0, feat              # only label-fitted features move
    assert le["rows_any_bin_changed"] == len(le["rows_any_bin_changed_index"])
    assert set(rec["eligibility"]) == set(rec["eligibility_retrained"]) == {"glass", "rf"}
    for arm in ("glass", "rf"):
        assert set(rec["eligibility"][arm]) == {"all"} | {f"{f}_only"
                                                         for f in G.LABEL_FITTED_FEATURES}
        assert rec["eligibility_retrained"][arm]["complete"] is True
    assert rec["eligibility_retrained"]["glass"]["reproduced_all"] is True
    assert rec["eligibility_retrained"]["rf"]["t1_reproduction_max_difference"] <= 1e-9
    assert "per_arm_inner_plan" not in rec
    assert set(rec["materiality_inputs"]) >= {"glass_eligibility_rows_changed",
                                              "rf_eligibility_rows_changed",
                                              "glass_rows_any_bin_changed",
                                              "rf_rows_any_bin_changed",
                                              "glass_retrained_eligibility_rows_changed",
                                              "rf_retrained_eligibility_rows_changed",
                                              "rf_eligibility_rows_changed_by_encoder"}
    assert rec["fit_label_fingerprint"] == E.label_fingerprint(X_tr.index, y_tr)


def test_each_arm_is_audited_on_its_own_inner_plan(fits, monkeypatch):
    g = fits[E.ARM_GLASS]
    router, rf_bins, _ = fits[E.ARM_RF]
    X_tr, y_tr, _ = W.synthetic_train_side().fold_rows(0)
    rf_plan = np.asarray(router.oof_pass1_fold_id_)
    glass_plan = np.random.default_rng(0).permutation(rf_plan)  # a different partition
    fake_glass = types.SimpleNamespace(inner_fold_id_=glass_plan,
                                       fit_report_={"pass2_population": "oof_remainder"})
    seen = {}

    def glass_shift(glass, base_bins, alt, y):                 # the plan, not the GLASS fit
        seen["glass"] = alt
        return {k: {"rows_changed": 0, "share_of_remainder": 0.0} for k in alt}

    monkeypatch.setattr(G, "glass_eligibility_shift", glass_shift)

    def glass_retrained(glass, base_bins, frame_bins, y):
        seen["glass_frames"] = frame_bins
        return {"rows_changed": 0, "share_of_remainder": 0.0, "complete": True}

    monkeypatch.setattr(G, "glass_retrained_shift", glass_retrained)
    rec = G.audit_fold(X_tr, y_tr, fold=0, glass=fake_glass, glass_bins=g.X_tr_eng,
                       rf_router=router, rf_bins=rf_bins)
    assert rec["inner_folds_shared_by_both_arms"] is False
    assert set(rec["per_arm_inner_plan"]) == {"glass", "rf"}
    y = np.asarray(y_tr).astype(int)
    eng = RFFeatureEngineer().fit(X_tr, y_tr)
    expect = G.bin_source(G.fold_excluded_features(eng, X_tr, y, glass_plan))
    assert seen["glass"]["all"].equals(expect)                   # GLASS: its own plan
    j = int(glass_plan[0])
    assert seen["glass_frames"][j].equals(
        G.bin_source(G.fold_excluded_frames(eng, X_tr, y, glass_plan)[j]))
    assert not expect.equals(G.bin_source(G.fold_excluded_features(eng, X_tr, y, rf_plan)))
    assert rec["eligibility"]["rf"]["all"]["max_score_reproduction_difference"] <= 1e-12


def test_the_audit_refuses_bins_the_arms_were_not_fitted_on(fits):
    g = fits[E.ARM_GLASS]
    X_tr, y_tr, _ = W.synthetic_train_side().fold_rows(0)
    wrong = g.X_tr_eng.copy()
    wrong.iloc[0, 0] = 1 - wrong.iloc[0, 0]
    with pytest.raises(AssertionError, match="not the bins"):
        G.audit_fold(X_tr, y_tr, fold=0, glass=g.glass, glass_bins=wrong)
    with pytest.raises(ValueError, match="at least one"):
        G.audit_fold(X_tr, y_tr, fold=0)
