"""
GLASS Stage 1: calibrated logistic regression (glass_pipeline.lr) on the
notebook's LR features (shared.stage1.feature_engineering).

Contract: every training row gets one out-of-fold probability from a model
that never saw it, in training-row order, with the fold it was held out in;
the F2 threshold, the decisions and the StageOutput all agree with those
probabilities; a prefit calibrator, a contaminated threshold source, a changed
feature schema and misaligned or invalid inputs are refused; held-out test
labels influence nothing that is learned; and the protocol matches the
black-box MLP arm.

Fold isolation of the learned preprocessing (F3 / F4) is in
test_stage1_fold_isolation.py; the calibration procedure (F16 / F19) in
test_stage1_calibration.py — both for both arms.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score

import blackbox_pipeline.models.mlp.stage as mlp_stage_mod
import glass_pipeline.lr.lr_stage_calibrated as lr_mod
import shared.stage1.oof_calibration as ocal_mod
from blackbox_pipeline.models.mlp import calibration as mlp_calibration
from blackbox_pipeline.models.mlp.config import Stage1MLPConfig
from blackbox_pipeline.models.mlp.stage import CalibratedStage1MLP
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from shared.stage1 import calibration_guard
from shared.stage1.calibration_guard import CalibrationLeakageError, assert_refittable
from shared.stage1.feature_engineering import LR_FEATURES
from shared.stage1.oof_calibration import CALIBRATION_FAMILIES, Stage1CalibratedModel
from shared.stage_io import THRESHOLD_SOURCES, ThresholdContaminationError, fold_assignment
from tests.stage1.conftest import (
    LABEL_FREE_FEATURES, fit_glass_stage1, fold_levels, stage1_split, sub_log,
)
from tests.support import mutations as mut
from tests.support.checks import assert_never_fit_on, assert_oof_provenance

CV, SEED = 10, 42                      # CalibratedLRStage defaults (notebook)
GRID = np.arange(0.05, 0.50, 0.01)     # default threshold_grid


# ----------------------------------------------------------------------
# out-of-fold integrity
# ----------------------------------------------------------------------
def test_every_training_row_is_scored_once_by_a_model_that_never_saw_it(lr_run):
    train = lr_run.X_train.index
    outer, inner = fold_levels(lr_run.oof_log, train)
    assert len(outer) == CV and sum(map(len, inner.values())) == CV * CV
    assert_oof_provenance(sub_log(lr_run.oof_log, outer), train, methods=("predict_proba",))


def test_stored_fold_ids_are_the_folds_actually_used(lr_run):
    index = lr_run.X_train.index
    outer, _ = fold_levels(lr_run.oof_log, index)
    held_out = {}
    for k, fid in enumerate(sorted(outer)):
        rows = [r for c in lr_run.oof_log.calls(fid, ["predict_proba"]) for r in c.keys]
        held_out.update({r: k for r in rows})
    used = pd.Series(held_out).reindex(index)
    pd.testing.assert_series_equal(lr_run.stage.train_fold_id, used, check_names=False,
                                   check_dtype=False)
    expected = fold_assignment(lr_run.y_train, cv_folds=CV, random_state=SEED, index=index)
    pd.testing.assert_series_equal(lr_run.stage.train_fold_id, expected)


def test_oof_block_is_aligned_to_the_training_rows(lr_run):
    stage, out = lr_run.stage, lr_run.output
    p = stage.proba_train_oof
    assert p.name == "stage1_proba_oof"
    assert p.index.equals(lr_run.X_train.index) and p.index.equals(lr_run.y_train.index)
    assert stage.train_fold_id.index.equals(p.index)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    pd.testing.assert_series_equal(out.train_proba_oof, p)
    assert out.test_proba.index.equals(lr_run.X_test.index)


def test_stage_output_carries_oof_not_in_sample_scores(lr_run):
    in_sample = lr_run.stage.predict_proba(lr_run.X_train)
    assert not np.allclose(lr_run.output.train_proba_oof.to_numpy(), in_sample)
    assert lr_run.output.oof_provenance.startswith(f"refittable (nested cross-fit: {CV} outer")


def test_no_stage1_fit_ever_sees_test_rows(lr_run):
    test_rows = lr_run.X_test.index
    assert_never_fit_on(lr_run.encoder_log, test_rows)
    assert_never_fit_on(lr_run.scaler_log, test_rows)
    assert_never_fit_on(lr_run.oof_log, test_rows)
    assert not {k for c in lr_run.oof_log for k in c.keys} & set(test_rows)  # nor scored them


# ----------------------------------------------------------------------
# thresholds and decisions
# ----------------------------------------------------------------------
def test_threshold_is_the_f2_grid_argmax_of_the_oof_probabilities(lr_run):
    stage = lr_run.stage
    y, p = lr_run.y_train, stage.proba_train_oof
    scores = [fbeta_score(y, (p >= t).astype(int), beta=2.0, zero_division=0) for t in GRID]
    best = int(np.argmax(scores))                                  # first max: lowest t
    assert stage.optimal_threshold == pytest.approx(GRID[best])
    assert stage.cv_f2 == pytest.approx(scores[best])
    np.testing.assert_allclose(stage.threshold_sweep["threshold"], GRID)
    assert lr_run.output.threshold == pytest.approx(stage.optimal_threshold)


def test_decisions_reproduce_from_probabilities_and_threshold(lr_run):
    stage, X = lr_run.stage, lr_run.X_test
    p, t = stage.predict_proba(X), stage.optimal_threshold
    np.testing.assert_array_equal(stage.predict(X), (p >= t).astype(int))
    np.testing.assert_array_equal(lr_run.output.test_decisions().to_numpy(), (p >= t).astype(int))
    at_boundary = stage.predict(X, threshold=float(p[0]))
    assert at_boundary[0] == 1                                      # p == t is flagged


def test_contaminated_threshold_cannot_produce_train_decisions(lr_run):
    out = lr_run.output
    assert out.threshold_source == CalibratedLRStage.THRESHOLD_SOURCE == "in_stage_f2_cv"
    assert THRESHOLD_SOURCES[out.threshold_source] is False
    with pytest.raises(ThresholdContaminationError):
        out.train_decisions_oof()
    forced = out.train_decisions_oof(allow_contaminated=True)
    np.testing.assert_array_equal(forced.to_numpy(),
                                  (out.train_proba_oof >= out.threshold).astype(int).to_numpy())


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------
def test_calibration_records_the_winner_and_the_request(lr_run):
    stage, out = lr_run.stage, lr_run.output
    assert stage.calibration_requested == out.calibration_requested == "auto"
    assert stage.calibration_method == out.calibration_method
    assert stage.calibration_method in CALIBRATION_FAMILIES
    assert isinstance(stage.calibrated_model, Stage1CalibratedModel)
    assert stage.calibrated_model.base is stage.model                # the full-train fit
    assert stage.calibrated_model.method == stage.calibration_method


def test_a_prefit_base_model_stops_the_fit_before_any_oof_is_produced(monkeypatch, serial_stage1):
    FrozenEstimator = pytest.importorskip("sklearn.frozen").FrozenEstimator
    split = stage1_split()
    X, y = split["X_train"][LABEL_FREE_FEATURES], split["y_train"]
    fitted = LogisticRegression().fit([[0.0], [1.0]], [0, 1])          # any prefit model
    monkeypatch.setattr(lr_mod, "LogisticRegression", lambda **kw: FrozenEstimator(fitted))
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(CalibrationLeakageError):
        CalibratedLRStage(tune_hyperparameters=False, cv_folds=3).fit(X, y)


def test_calibration_guard_detects_every_prefit_pattern():
    assert assert_refittable(CalibratedClassifierCV(LogisticRegression(), cv=5)) == "refittable (cv=5)"
    prefit = [CalibratedClassifierCV(LogisticRegression(), cv="prefit")]
    try:
        from sklearn.frozen import FrozenEstimator
        prefit += [CalibratedClassifierCV(FrozenEstimator(LogisticRegression())),
                   FrozenEstimator(LogisticRegression())]
    except ImportError:            # sklearn < 1.6
        pass
    for calibrated in prefit:
        with pytest.raises(CalibrationLeakageError, match="PREFIT"):
            assert_refittable(calibrated)
        with contextlib.redirect_stdout(io.StringIO()):
            assert assert_refittable(calibrated, strict=False).startswith("PREFIT")


# ----------------------------------------------------------------------
# feature schema
# ----------------------------------------------------------------------
def test_feature_schema_is_the_lr_feature_list_in_order(lr_run):
    assert list(lr_run.X_train.columns) == list(lr_run.X_test.columns) == LR_FEATURES
    assert lr_run.stage.feature_names == lr_run.output.feature_names == LR_FEATURES
    assert list(lr_run.stage.scaler.feature_names_in_) == LR_FEATURES


@pytest.mark.parametrize("name", ["missing_column", "extra_column", "duplicated_column",
                                  "reordered_columns", "renamed_column"])
def test_scoring_refuses_a_changed_schema(lr_run, name):
    with pytest.raises(ValueError):
        lr_run.stage.predict_proba(mut.COLUMN_MUTATIONS[name](lr_run.X_test))


def test_a_dtype_only_change_scores_identically(lr_run):
    X = lr_run.X_test
    np.testing.assert_allclose(lr_run.stage.predict_proba(mut.cast_column(X, "cellular_crisis")),
                               lr_run.stage.predict_proba(X))


# ----------------------------------------------------------------------
# invalid inputs
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["shuffled_rows", "reset_index", "foreign_label"])
def test_labels_not_aligned_to_features_are_refused(serial_stage1, name):
    split = stage1_split()
    X = split["X_train"][LABEL_FREE_FEATURES]
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match="index"):
        CalibratedLRStage(tune_hyperparameters=False).fit(X, mut.INDEX_MUTATIONS[name](split["y_train"]))


@pytest.mark.parametrize("case, message", [
    ("nan_feature", "NaN"),
    ("non_binary_labels", "binary 0/1"),     # F18: refused before any model is fitted
    ("nan_label", "binary 0/1"),
])
def test_invalid_training_data_fails_loudly(serial_stage1, monkeypatch, case, message):
    split = stage1_split()
    X, y = split["X_train"][LABEL_FREE_FEATURES].copy(), split["y_train"].astype(float)
    if case == "nan_feature":
        X.iloc[0, 0] = np.nan
    elif case == "non_binary_labels":
        y.iloc[:3] = 2
    else:
        y.iloc[0] = np.nan
    fitted = []
    monkeypatch.setattr(lr_mod, "LogisticRegression",
                        lambda **kw: fitted.append(kw) or LogisticRegression(**kw))
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match=message):
        CalibratedLRStage(tune_hyperparameters=False).fit(X, y)
    assert (len(fitted) == 1) == (case == "nan_feature")          # labels: nothing was built


def test_out_of_order_calls_are_refused(lr_run, serial_stage1):
    with pytest.raises(ValueError, match="not fitted"):
        CalibratedLRStage().predict(lr_run.X_test)
    with contextlib.redirect_stdout(io.StringIO()):
        stage = CalibratedLRStage(tune_hyperparameters=False, cv_folds=3).fit(
            lr_run.split["X_train"][LABEL_FREE_FEATURES], lr_run.y_train)
    with pytest.raises(ValueError, match="evaluate"):
        stage.to_stage_output()


# ----------------------------------------------------------------------
# held-out test labels
# ----------------------------------------------------------------------
def test_test_labels_do_not_influence_anything_learned(lr_run_5):
    split = stage1_split()
    moved = fit_glass_stage1(stage1_split(y_test=mut.permute_labels(split["y_test"], seed=11)),
                             cv_folds=5)
    base = lr_run_5
    assert not moved.output.y_test.equals(base.output.y_test)       # the labels did change
    pd.testing.assert_frame_equal(moved.X_train, base.X_train)
    pd.testing.assert_frame_equal(moved.X_test, base.X_test)
    pd.testing.assert_series_equal(moved.stage.proba_train_oof, base.stage.proba_train_oof)
    pd.testing.assert_series_equal(moved.stage.train_fold_id, base.stage.train_fold_id)
    assert moved.stage.optimal_threshold == base.stage.optimal_threshold
    assert moved.stage.calibration_method == base.stage.calibration_method
    pd.testing.assert_series_equal(moved.output.test_proba, base.output.test_proba)
    assert moved.output.split_fingerprint != base.output.split_fingerprint


# ----------------------------------------------------------------------
# tuning (2 trials; the notebook runs 100)
# ----------------------------------------------------------------------
def _penalty_of(model) -> str:
    """The penalty a LogisticRegression fits, under either scikit-learn spelling."""
    if model.penalty in ("l1", "l2", "elasticnet"):
        return model.penalty
    return {0.0: "l2", 1.0: "l1"}.get(model.l1_ratio, "elasticnet")


def test_tuned_parameters_are_the_ones_fitted():
    run = fit_glass_stage1(tune_hyperparameters=True, n_trials=2, tune_cv_folds=3, cv_folds=5)
    stage, params = run.stage, run.stage.best_params
    assert stage.model.C == params["C"] and _penalty_of(stage.model) == params["penalty"]
    assert stage.model.class_weight == "balanced"
    assert len(stage.tuning_trials) == 2
    assert stage.best_cv_roc_auc == pytest.approx(stage.tuning_trials["value"].max())
    assert run.output.best_cv_score == stage.best_cv_roc_auc
    assert_never_fit_on(run.scaler_log, run.X_test.index)
    # tuning folds (3) differ from OOF folds (5): both refit encoder and scaler per fold
    train = run.X_train.index
    for log in (run.encoder_log, run.scaler_log):
        outer, _ = fold_levels(log, train)
        assert len(outer) == 5 + 3
        for fits in (sorted(outer)[:5], sorted(outer)[5:]):
            assert_oof_provenance(sub_log(log, fits), train, methods=("transform",))


# ----------------------------------------------------------------------
# protocol compatibility with the black-box MLP arm
# ----------------------------------------------------------------------
def test_defaults_match_the_blackbox_mlp_protocol():
    lr, mlp = CalibratedLRStage().config_dict(), Stage1MLPConfig()
    for field in ("cv_folds", "random_state", "threshold_beta", "n_trials"):
        assert lr[field] == getattr(mlp, field), field
    assert tuple(lr["threshold_grid"]) == tuple(mlp.threshold_grid)
    assert lr["calibration_requested"] == mlp.calibration_method
    assert lr["tune_cv_folds"] == mlp.cv_folds
    assert CalibratedLRStage.THRESHOLD_SOURCE == CalibratedStage1MLP.THRESHOLD_SOURCE


def test_both_arms_share_one_calibration_routine_and_guard():
    assert mlp_calibration.assert_refittable is calibration_guard.assert_refittable
    assert ocal_mod.assert_refittable is calibration_guard.assert_refittable
    assert lr_mod.calibrate_oof is mlp_stage_mod.calibrate_oof is ocal_mod.calibrate_oof
    assert lr_mod.Stage1CalibratedModel is mlp_stage_mod.Stage1CalibratedModel


def test_both_arms_validate_labels_with_one_routine_f18():
    from blackbox_pipeline.models.mlp import features as mlp_features
    from shared.stage1 import labels
    assert lr_mod.check_labels is mlp_features.check_labels is labels.check_labels
