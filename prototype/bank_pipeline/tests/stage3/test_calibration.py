"""
Stage 3 calibration (shared.stage3.calibration.Stage3Calibrator).

Contract: the test-side gate is measured on all out-of-fold training
predictions and the test-time map is fitted on them; on the training side
each outer fold k has its own gate and map, both fitted on the inner
out-of-fold scores of fold k's training rows, so fold k's labels never reach
its calibrated output, even through the gate (audit F22); bad inputs,
including inner scores that cover a fold's own rows, are refused.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from shared.stage3.calibration import Stage3Calibrator, calculate_ece
from shared.stage3.data import FoldPlan
from tests.support import mutations as mut


def _overconfident(y, rng):
    return np.clip(0.05 + 0.75 * np.asarray(y) + rng.normal(0, 0.15, len(y)), 0.01, 0.99)


@pytest.fixture
def oof(bank_split):
    """Over-confident OOF scores (ECE well above the 0.05 gate), a fixed fold
    plan, and inner scores: row k covers fold k's training rows only."""
    y = bank_split["y_train"].astype(int)
    rng = np.random.default_rng(0)
    p = _overconfident(y, rng)
    plan = FoldPlan.build(y, n_splits=3, random_state=0)
    inner = np.full((plan.n_splits, len(y)), np.nan)
    for k, (tr, _) in enumerate(plan):
        inner[k, tr] = _overconfident(y.to_numpy()[tr], rng)
    return p, y, plan, inner


def _isotonic(s, y):
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)


def test_gate_not_breached_means_identity(oof):
    p, y, plan, inner = oof
    cal = Stage3Calibrator(ece_threshold=1.0).fit(p, y, plan, inner)
    assert cal.applied_ is False and "<= gate" in cal.report_.reason
    assert cal.fold_applied_ == [False] * plan.n_splits
    np.testing.assert_array_equal(cal.oof_calibrated_, p)
    np.testing.assert_array_equal(cal.inner_calibrated_, inner)
    q = np.linspace(0.05, 0.95, 7)
    np.testing.assert_array_equal(cal.transform(q), q)


def test_each_gate_reads_its_own_scores(oof):
    """Test side: ECE of all OOF rows. Fold k: ECE of its inner scores. A fold
    whose inner scores are already calibrated keeps its raw OOF scores."""
    p, y, plan, inner = oof
    tr0 = plan.splits()[0][0]
    inner = inner.copy()
    inner[0, tr0] = y.to_numpy()[tr0].mean()                 # calibrated in the large: ECE ~ 0
    cal = Stage3Calibrator().fit(p, y, plan, inner)
    assert cal.applied_ is True and "> gate" in cal.report_.reason
    assert cal.report_.ece_before == pytest.approx(calculate_ece(y, p, 10))
    assert cal.fold_applied_ == [False, True, True]
    y_arr = y.to_numpy()
    for k, (tr, va) in enumerate(plan):
        assert cal.report_.fold_ece_inner[k] == pytest.approx(calculate_ece(y_arr[tr], inner[k, tr], 10))
    np.testing.assert_array_equal(cal.oof_calibrated_[plan.holdout(0)], p[plan.holdout(0)])
    assert not np.array_equal(cal.oof_calibrated_[plan.holdout(1)], p[plan.holdout(1)])


def test_each_fold_is_mapped_by_a_calibrator_fitted_on_its_inner_scores(oof):
    p, y, plan, inner = oof
    cal = Stage3Calibrator(force=True).fit(p, y, plan, inner)
    y_arr = y.to_numpy()
    for k, (tr, va) in enumerate(plan):
        ref = _isotonic(inner[k, tr], y_arr[tr])
        np.testing.assert_allclose(cal.oof_calibrated_[va], np.clip(ref.predict(p[va]), 0, 1))
        np.testing.assert_allclose(cal.inner_calibrated_[k, tr], np.clip(ref.predict(inner[k, tr]), 0, 1))
        assert np.isnan(cal.inner_calibrated_[k, va]).all()
    assert cal.report_.ece_after < cal.report_.ece_before


def test_a_folds_own_labels_never_reach_its_calibrated_output(oof):
    """Given inner scores (which cannot depend on fold 0's labels), fold 0's
    gate and map ignore fold 0's labels, even when those labels tip the
    all-rows gate the old procedure used for every fold (F22)."""
    p, y, plan, inner = oof
    fold0 = plan.fold_id == 0
    own_y = mut.permute_labels(y, seed=4, rows=y.index[fold0])
    e_base, e_own = calculate_ece(y, p, 10), calculate_ece(own_y, p, 10)
    assert e_base != e_own
    gate = (e_base + e_own) / 2                               # between the two: the global gate flips
    base = Stage3Calibrator(ece_threshold=gate).fit(p, y, plan, inner)
    own = Stage3Calibrator(ece_threshold=gate).fit(p, own_y, plan, inner)
    assert base.applied_ != own.applied_
    assert own.fold_applied_[0] == base.fold_applied_[0]
    np.testing.assert_array_equal(own.oof_calibrated_[fold0], base.oof_calibrated_[fold0])
    np.testing.assert_array_equal(own.inner_calibrated_[0], base.inner_calibrated_[0])

    others = Stage3Calibrator(force=True).fit(
        p, mut.permute_labels(y, seed=4, rows=y.index[~fold0]), plan, inner)
    forced = Stage3Calibrator(force=True).fit(p, y, plan, inner)
    assert not np.array_equal(others.oof_calibrated_[fold0], forced.oof_calibrated_[fold0])  # can fail


def test_test_time_calibrator_is_fitted_on_all_oof_rows(oof):
    p, y, plan, inner = oof
    cal = Stage3Calibrator(force=True).fit(p, y, plan, inner)
    q = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(cal.transform(q), np.clip(_isotonic(p, y).predict(q), 0, 1))


def test_platt_scaling_is_monotone_and_bounded(oof):
    p, y, plan, inner = oof
    cal = Stage3Calibrator(method="sigmoid", force=True).fit(p, y, plan, inner)
    for out in (cal.transform(np.linspace(0, 1, 11)), np.sort(cal.oof_calibrated_)):
        assert ((out >= 0) & (out <= 1)).all() and (np.diff(out) >= 0).all()


@pytest.mark.parametrize("case, message", [
    ("nan", "NaN"),
    ("length", "length mismatch"),
    ("plan", "must not be reused across populations"),
    ("inner_shape", "must have shape"),
    ("inner_scores_own_rows", "must leave its held-out rows unscored"),
    ("inner_misses_a_training_row", "must cover all its training rows"),
])
def test_invalid_calibration_inputs_are_refused(oof, case, message):
    p, y, plan, inner = oof
    args = [p, y, plan, inner]
    if case == "nan":
        args[0] = mut.corrupt_probabilities(p, "nan")
    elif case == "length":
        args[0] = p[:-1]
    elif case == "plan":
        args[:2] = p[:-1], y.iloc[:-1]
    elif case == "inner_shape":
        args[3] = inner[1:]
    elif case == "inner_scores_own_rows":                      # e.g. the other folds' outer OOF
        args[3] = np.tile(p, (plan.n_splits, 1))
    else:
        args[3] = inner.copy()
        args[3][1, plan.splits()[1][0][0]] = np.nan
    with pytest.raises(ValueError, match=message):
        Stage3Calibrator(force=True).fit(*args)
    with pytest.raises(ValueError, match="unknown calibration method"):
        Stage3Calibrator(method="beta")


# ----------------------------------------------------------------------
# inside the shared runner, for every arm
# ----------------------------------------------------------------------
def test_runner_gate_reads_out_of_fold_not_in_sample_predictions(arm_run):
    art, cfg = arm_run.artifact, arm_run.config
    y_tr = np.asarray(art.y_train)
    oof_ece = calculate_ece(y_tr, np.asarray(art.oof["proba"]), cfg.ece_bins)
    in_sample_ece = calculate_ece(y_tr, np.asarray(art.refit["train_proba"]), cfg.ece_bins)
    assert art.calibration["ece_before"] == pytest.approx(oof_ece)
    assert art.calibration["ece_before"] != pytest.approx(in_sample_ece)


def test_runner_calibrated_streams_reproduce_from_stored_inputs(arm_run):
    art, cfg, block = arm_run.artifact, arm_run.config, arm_run.block
    cal = Stage3Calibrator(cfg.calibration_method, cfg.ece_threshold, cfg.ece_bins,
                           cfg.force_calibration, cfg.random_state)
    cal.fit(np.asarray(art.oof["proba"]), block.y_train, block.tune_plan,
            art.oof["inner_proba_by_fold"])
    assert cal.applied_ == art.calibration["applied"]
    assert cal.fold_applied_ == art.calibration["fold_applied"]
    np.testing.assert_allclose(cal.oof_calibrated_, art.oof["proba_calibrated"])
    np.testing.assert_allclose(cal.transform(np.asarray(art.refit["test_proba"])),
                               art.refit["test_proba_calibrated"])
