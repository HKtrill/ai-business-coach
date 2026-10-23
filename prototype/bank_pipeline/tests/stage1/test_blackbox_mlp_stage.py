"""
Black-box Stage 1: the calibrated MLP stage (``blackbox_pipeline.models.mlp``)
on the shared LR features, through its artifact into the Stage 4 adapter.

Contract: under the notebook protocol (10 outer x 10 inner folds) every
training row gets one out-of-fold probability from a model that never saw it,
in training-row order, with the fold it was held out in; Optuna tunes on the
same folds; the test split is scored by the full-train MLP and one calibrator;
the F2 threshold, the decisions and the StageOutput agree with the OOF
probabilities, and the threshold's provenance cannot be mislabelled (F33);
the feature schema is enforced (columns are realigned by name, so their order
is free); misaligned or non-binary labels are refused before anything is
fitted; held-out test labels influence nothing learned; serial and parallel
fits are identical; the artifact reloads to the same predictions with its
schema and provenance, and the shared Stage 4 adapter takes it for its own
split only; the black-box notebook calls the stage the way it is tested here.

Shared with GLASS LR and tested for both arms elsewhere: per-fold encoders and
scalers (F3/F4, test_stage1_fold_isolation.py) and the nested calibration
procedure, its scikit-learn reference and a fold's own-label isolation
(F16/F19, test_stage1_calibration.py). The estimator itself is in
test_blackbox_mlp_estimator.py.

MLP runs use tiny synthetic folds, smaller than the tuned batch sizes, so
scikit-learn clips the batch and warns (impossible at production scale); the
filter matches that one message only.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler

import blackbox_pipeline.models.mlp.stage as mlp_stage_mod
from blackbox_pipeline.models.mlp import (
    STAGE1_FEATURES, CalibratedStage1MLP, Stage1MLPClassifier, Stage1MLPConfig,
    assert_matches_glass, load_stage1_mlp, params_to_kwargs, save_stage1_mlp,
)
from shared.stage1.calibration_guard import CalibrationLeakageError
from shared.stage1.feature_engineering import LR_FEATURES, LRFeatureEngineer, engineer_features
from shared.stage1.oof_calibration import CALIBRATION_FAMILIES, Stage1CalibratedModel
from shared.stage4 import Stage4InputError, build_stage4_inputs, stage_output_stream
from shared.stage_io import (
    THRESHOLD_SOURCES, StageOutput, StageOutputError, ThresholdContaminationError,
    fold_assignment,
)
from shared.validation import ValidationError
from tests.stage1.conftest import (
    MLP_TEST_CONFIG, fit_mlp_stage1, fold_levels, stage1_split, sub_log, unrecorded_scale,
)
from tests.support import mutations as mut
from tests.support.checks import assert_never_fit_on, assert_oof_provenance
from tests.support.notebooks import BANK_ROOT
from tests.support.synthetic import make_bank_frame

pytestmark = pytest.mark.filterwarnings(
    "ignore:Got `batch_size` less than 1 or larger than sample size:UserWarning")

CV, SEED = 10, 42                      # Stage1MLPConfig defaults (notebook)
GRID = np.arange(0.05, 0.50, 0.01)     # default threshold_grid
NOTEBOOK = BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def mlp10():
    """The notebook protocol: 10 outer x 10 inner folds, auto calibration (1 trial)."""
    return fit_mlp_stage1(cv_folds=CV)


@pytest.fixture(scope="module")
def plain():
    """An unrecorded fit (recorder subclasses are test-local and do not pickle)."""
    split = stage1_split()
    X_tr, X_te, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    stage = CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(X_tr, split["y_train"],
                                                       X_train_raw=split["X_train"])
    return split, X_tr, X_te, stage, stage.to_stage_output(X_te, split["y_test"])


def _unseen_rows(split) -> pd.DataFrame:
    """Stage 1 features of rows outside the split, built with the full-train encoder."""
    engineer = _quiet(LRFeatureEngineer().fit, split["X_train"], split["y_train"])
    raw = make_bank_frame(n=60, seed=7).drop(columns="y")
    return _quiet(engineer.transform, raw)[LR_FEATURES]


# ----------------------------------------------------------------------
# out-of-fold integrity (notebook protocol)
# ----------------------------------------------------------------------
def test_every_training_row_is_scored_once_by_an_outer_model_that_never_saw_it(mlp10):
    train = mlp10.X_train.index
    outer, inner = fold_levels(mlp10.oof_log, train)
    assert len(outer) == CV and sum(map(len, inner.values())) == CV * CV
    assert_oof_provenance(sub_log(mlp10.oof_log, outer), train, methods=("predict_proba",))
    assert {c.method for c in mlp10.oof_log} == {"fit", "predict_proba"}   # no decision_function


def test_stored_fold_ids_are_the_folds_actually_used(mlp10):
    index = mlp10.X_train.index
    outer, _ = fold_levels(mlp10.oof_log, index)
    held_out = {}
    for k, fid in enumerate(sorted(outer)):
        held_out.update({r: k for c in mlp10.oof_log.calls(fid, ["predict_proba"]) for r in c.keys})
    pd.testing.assert_series_equal(mlp10.stage.train_fold_id, pd.Series(held_out).reindex(index),
                                   check_names=False, check_dtype=False)
    expected = fold_assignment(mlp10.y_train, cv_folds=CV, random_state=SEED, index=index)
    pd.testing.assert_series_equal(mlp10.stage.train_fold_id, expected)
    pd.testing.assert_series_equal(mlp10.output.train_fold_id, expected)


def test_the_oof_output_follows_each_folds_own_family_choice_at_ten_folds(mlp10):
    stage = mlp10.stage
    fold, families = stage.train_fold_id.to_numpy(), stage.calibration_fold_family_
    assert sorted(families) == list(range(CV)) and set(families.values()) <= set(CALIBRATION_FAMILIES)
    expected = np.full(len(fold), np.nan)
    for k, f in families.items():
        expected[fold == k] = stage.calibration_oof_[f].to_numpy()[fold == k]
    np.testing.assert_array_equal(stage.proba_train_oof.to_numpy(), expected)


def test_oof_block_is_aligned_to_the_training_rows(mlp10):
    stage, out = mlp10.stage, mlp10.output
    p = stage.proba_train_oof
    assert p.name == "stage1_proba_oof"
    assert p.index.equals(mlp10.X_train.index) and p.index.equals(mlp10.y_train.index)
    assert stage.train_fold_id.index.equals(p.index) and stage.y_train_.equals(mlp10.y_train)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    pd.testing.assert_series_equal(out.train_proba_oof, p)
    assert out.test_proba.index.equals(mlp10.X_test.index) and out.test_proba.name == "stage1_proba"


def test_stage_output_carries_oof_not_in_sample_scores(mlp10):
    in_sample = mlp10.stage.predict_proba(mlp10.X_train)
    assert not np.allclose(mlp10.output.train_proba_oof.to_numpy(), in_sample)
    assert mlp10.output.oof_provenance == (
        f"refittable (nested cross-fit: {CV} outer x {CV} inner folds; "
        "encoder, scaler, model and calibrator fitted per fold)")


def test_no_stage1_fit_ever_sees_test_rows(mlp10):
    test_rows = mlp10.X_test.index
    for log in (mlp10.encoder_log, mlp10.scaler_log, mlp10.oof_log):
        assert_never_fit_on(log, test_rows)
    assert not {k for c in mlp10.oof_log for k in c.keys} & set(test_rows)   # nor scored them


def test_optuna_tunes_on_the_oof_folds_and_training_rows_only(monkeypatch):
    seen = {}
    real_tune, real_calibrate = mlp_stage_mod.tune_stage1_mlp_auc, mlp_stage_mod.calibrate_oof

    def tune(*args, folds=None, **kwargs):
        seen["tune"] = folds
        return real_tune(*args, folds=folds, **kwargs)

    def calibrate(base, designs, *args, **kwargs):
        seen["oof"] = designs
        return real_calibrate(base, designs, *args, **kwargs)

    monkeypatch.setattr(mlp_stage_mod, "tune_stage1_mlp_auc", tune)
    monkeypatch.setattr(mlp_stage_mod, "calibrate_oof", calibrate)
    split = stage1_split()
    X_tr, _, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    stage = CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(X_tr, split["y_train"],
                                                       X_train_raw=split["X_train"])
    assert seen["tune"] is seen["oof"] and len(seen["tune"]) == stage.config.cv_folds
    train, test = set(split["X_train"].index), set(split["X_test"].index)
    for d in seen["tune"]:
        assert set(d.index_train) | set(d.index_val) == train and not set(d.index_val) & set(d.index_train)
        assert not (set(d.index_train) | set(d.index_val)) & test
        assert (stage.train_fold_id.loc[d.index_val] == d.fold).all()


def test_the_scorer_is_the_full_train_mlp_with_the_tuned_parameters(mlp_run):
    stage, X_tr, y = mlp_run.stage, mlp_run.X_train, mlp_run.y_train
    np.testing.assert_allclose(stage.scaler.mean_, X_tr.to_numpy(dtype=float).mean(axis=0))
    fixed = stage.config.estimator_fixed_kwargs
    kwargs = params_to_kwargs(stage.best_params, **fixed)
    assert {k: stage.model.get_params()[k] for k in kwargs} == kwargs
    X = unrecorded_scale(stage, X_tr)
    fresh = Stage1MLPClassifier(**kwargs).fit(X, y)
    for a, b in zip(fresh.net_.coefs_, stage.model.net_.coefs_):
        np.testing.assert_array_equal(a, b)
    scorer = stage.calibrated_model
    assert isinstance(scorer, Stage1CalibratedModel) and scorer.base is stage.model
    X_te = unrecorded_scale(stage, mlp_run.X_test)
    np.testing.assert_allclose(mlp_run.output.test_proba.to_numpy(), scorer.predict_proba(X_te)[:, 1])


# ----------------------------------------------------------------------
# thresholds and decisions
# ----------------------------------------------------------------------
def test_threshold_is_the_f2_grid_argmax_of_the_oof_probabilities(mlp10):
    stage = mlp10.stage
    y, p = mlp10.y_train, stage.proba_train_oof
    scores = [fbeta_score(y, (p >= t).astype(int), beta=2.0, zero_division=0) for t in GRID]
    best = int(np.argmax(scores))                                  # first max: lowest t
    assert stage.optimal_threshold == pytest.approx(GRID[best])
    assert stage.cv_f2 == pytest.approx(scores[best])
    np.testing.assert_allclose(stage.threshold_sweep["threshold"], GRID)
    assert (mlp10.output.threshold, mlp10.output.cv_f2) == (stage.optimal_threshold, stage.cv_f2)


def test_decisions_reproduce_from_probabilities_and_threshold(mlp10):
    stage, X = mlp10.stage, mlp10.X_test
    p, t = stage.predict_proba(X), stage.optimal_threshold
    pred = stage.predict(X)
    assert pred.name == "stage1_pred" and pred.dtype == np.int8 and pred.index.equals(X.index)
    np.testing.assert_array_equal(pred, (p >= t).astype(int))
    np.testing.assert_array_equal(mlp10.output.test_decisions().to_numpy(), (p >= t).astype(int))
    assert stage.predict(X, threshold=float(p.iloc[0])).iloc[0] == 1          # p == t is flagged


def test_contaminated_threshold_cannot_produce_train_decisions(mlp10):
    out = mlp10.output
    assert out.threshold_source == CalibratedStage1MLP.THRESHOLD_SOURCE == "in_stage_f2_cv"
    assert THRESHOLD_SOURCES[out.threshold_source] is False
    with pytest.raises(ThresholdContaminationError):
        out.train_decisions_oof()
    forced = out.train_decisions_oof(allow_contaminated=True)
    np.testing.assert_array_equal(forced.to_numpy(),
                                  (out.train_proba_oof >= out.threshold).astype(int).to_numpy())


@pytest.mark.parametrize("source, threshold, error, message", [
    ("nested_per_fold", None, ValueError, "cross-fitted"),       # in-stage threshold declared nested
    ("nested_per_fold", 0.3, ValueError, "cross-fitted"),
    ("in_stage_f2_cv", 0.3, ValueError, "in-stage F2"),          # override passed off as in-stage
    (None, 0.3, ValueError, "threshold_source"),
    ("hand_picked", None, StageOutputError, "threshold_source"),
])
def test_threshold_provenance_cannot_be_mislabelled_f33(mlp_run, source, threshold, error, message):
    """Before F33 the first three were recorded as given; 'nested_per_fold'
    then released train-side decisions without the contamination opt-in."""
    with pytest.raises(error, match=message):
        mlp_run.stage.to_stage_output(mlp_run.X_test, mlp_run.split["y_test"],
                                      threshold=threshold, threshold_source=source)


def test_an_honestly_labelled_override_is_recorded_f33(mlp_run):
    stage = mlp_run.stage
    out = stage.to_stage_output(mlp_run.X_test, mlp_run.split["y_test"], threshold=0.3,
                                threshold_source="notebook_wide_sweep")
    assert (out.threshold, out.threshold_source, out.threshold_is_cross_fitted) == (
        0.3, "notebook_wide_sweep", False)
    with pytest.raises(ThresholdContaminationError):
        out.train_decisions_oof()
    np.testing.assert_array_equal(out.test_decisions(), (out.test_proba >= 0.3).astype(int))
    assert stage.optimal_threshold != 0.3                                  # the stage is unchanged


def test_the_notebook_hand_off_call_is_accepted_f33(mlp_run):
    """Cell 14A passes the in-stage pair explicitly; it must equal the default."""
    stage = mlp_run.stage
    explicit = stage.to_stage_output(mlp_run.X_test, mlp_run.split["y_test"],
                                     threshold=stage.optimal_threshold,
                                     threshold_source=stage.THRESHOLD_SOURCE)
    assert (explicit.threshold, explicit.threshold_source) == (
        mlp_run.output.threshold, mlp_run.output.threshold_source)


def test_metrics_and_description_report_the_fitted_stage(mlp_run):
    stage, y = mlp_run.stage, mlp_run.y_train
    t = stage.optimal_threshold
    table = stage.metrics_table(y, stage.proba_train_oof, "train OOF")
    assert list(table.index) == ["train OOF @ 0.50", f"train OOF @ {t:.2f}"]
    d = stage.describe()
    hidden = stage.model.hidden_layer_sizes
    assert d["layers"] == " → ".join(map(str, [len(STAGE1_FEATURES), *hidden, 1]))
    assert d["trainable_params"] == stage.model.n_trainable_params_
    assert {k: d[k] for k in stage.best_params} == stage.best_params
    assert (d["optimal_threshold"], d["calibration_method"], d["oof_provenance"]) == (
        t, stage.calibration_method, stage.oof_provenance_)
    assert d["best_epoch"] == stage.model.best_epoch_


# ----------------------------------------------------------------------
# calibration and the leakage guard
# ----------------------------------------------------------------------
def test_calibration_records_the_winner_and_the_request(mlp10):
    stage, out = mlp10.stage, mlp10.output
    assert stage.config.calibration_method == out.calibration_requested == "auto"
    assert stage.calibration_method == out.calibration_method in CALIBRATION_FAMILIES
    assert stage.calibrated_model.method == stage.calibration_method
    assert stage.calibration_metrics["oof_family_by_fold"].count(",") == CV - 1


def _frozen_mlp(monkeypatch, split, X_tr):
    """Make the stage's final MLP a prefit model (fitted on every training row)."""
    FrozenEstimator = pytest.importorskip("sklearn.frozen").FrozenEstimator
    prefit = Stage1MLPClassifier(max_epochs=2, patience=1, batch_size=64).fit(
        StandardScaler().fit_transform(X_tr), split["y_train"])
    monkeypatch.setattr(mlp_stage_mod, "Stage1MLPClassifier", lambda **kw: FrozenEstimator(prefit))


def test_a_prefit_final_model_stops_the_fit_before_any_oof_is_produced(monkeypatch):
    split = stage1_split()
    X_tr, _, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    _frozen_mlp(monkeypatch, split, X_tr)
    stage = CalibratedStage1MLP(**MLP_TEST_CONFIG)
    with pytest.raises(CalibrationLeakageError):
        stage.fit(X_tr, split["y_train"], X_train_raw=split["X_train"])
    assert not stage.fitted and stage.proba_train_oof is None


def test_strict_oof_off_records_prefit_provenance_that_stage4_refuses(monkeypatch):
    split = stage1_split()
    X_tr, X_te, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    _frozen_mlp(monkeypatch, split, X_tr)
    stage = _quiet(CalibratedStage1MLP(**MLP_TEST_CONFIG, strict_oof=False).fit,
                   X_tr, split["y_train"], X_train_raw=split["X_train"])
    out = stage.to_stage_output(X_te, split["y_test"])
    assert out.oof_provenance.startswith("PREFIT")
    _, checks = stage_output_stream(out.to_dict(), "mlp", "MLP", len(split["y_train"]))
    failed = {name for _, _, name, fn, _ in checks if not fn()}
    assert any("refittable" in n for n in failed) and any("nested" in n for n in failed)


# ----------------------------------------------------------------------
# feature schema and scoring inputs
# ----------------------------------------------------------------------
def test_feature_schema_is_the_shared_lr_feature_list_in_order(mlp_run):
    assert list(STAGE1_FEATURES) == list(LR_FEATURES)
    assert list(mlp_run.X_train.columns) == list(mlp_run.X_test.columns) == list(STAGE1_FEATURES)
    assert mlp_run.stage.feature_names_ == mlp_run.output.feature_names == list(STAGE1_FEATURES)
    assert list(mlp_run.stage.scaler.feature_names_in_) == list(STAGE1_FEATURES)


def test_drift_from_the_glass_feature_set_is_detected():
    assert_matches_glass(LR_FEATURES)
    assert_matches_glass(list(reversed(LR_FEATURES)))                 # membership, not order
    for drifted in (LR_FEATURES[:-1], [*LR_FEATURES, "previous"],
                    ["cellular_crisis", "euribor3m_local_rate", "dow_encoded"]):
        with pytest.raises(AssertionError, match="drifted"):
            assert_matches_glass(drifted)


@pytest.mark.parametrize("name", ["missing_column", "extra_column", "duplicated_column",
                                  "renamed_column"])
def test_a_changed_schema_is_refused_for_fitting_and_scoring(mlp_run, name):
    X_tr, X_te = (mut.COLUMN_MUTATIONS[name](X) for X in (mlp_run.X_train, mlp_run.X_test))
    with pytest.raises(ValueError, match="Stage 1 expects exactly"):
        mlp_run.stage.predict_proba(X_te)
    with pytest.raises(ValueError, match="Stage 1 expects exactly"):
        CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(X_tr, mlp_run.y_train,
                                                   X_train_raw=mlp_run.split["X_train"])


def test_non_dataframe_inputs_are_refused(mlp_run):
    with pytest.raises(TypeError, match="DataFrame"):
        mlp_run.stage.predict_proba(mlp_run.X_test.to_numpy())
    with pytest.raises(TypeError, match="DataFrame"):
        CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(mlp_run.X_train.to_numpy(), mlp_run.y_train)


def test_columns_are_realigned_by_name_so_their_order_is_free(mlp_run):
    """Unlike GLASS LR, which refuses a reordered frame, the MLP selects its
    columns by name; either way no column is ever read positionally."""
    stage, X = mlp_run.stage, mlp_run.X_test
    np.testing.assert_array_equal(stage.predict_proba(mut.swap_columns(X, 0, 2)), stage.predict_proba(X))


def test_a_dtype_only_change_scores_identically(mlp_run):
    X = mlp_run.X_test
    np.testing.assert_allclose(mlp_run.stage.predict_proba(mut.cast_column(X, "cellular_crisis")),
                               mlp_run.stage.predict_proba(X))


@pytest.mark.parametrize("value", [np.nan, np.inf, "high"])
def test_invalid_feature_values_are_refused_not_scored(mlp_run, value):
    X = mlp_run.X_test.astype(object) if isinstance(value, str) else mlp_run.X_test.copy()
    X.iloc[0, 1] = value
    with pytest.raises(ValueError):
        mlp_run.stage.predict_proba(X)


def test_scores_are_row_wise_and_follow_the_rows_index(mlp_run):
    stage, X = mlp_run.stage, mlp_run.X_test
    full = stage.predict_proba(X)
    shuffled = stage.predict_proba(mut.shuffle_rows(X))
    pd.testing.assert_series_equal(shuffled.loc[full.index], full)
    some = X.iloc[[5, 1, 3]]
    pd.testing.assert_series_equal(stage.predict_proba(some), full.loc[some.index])


def test_previously_unseen_rows_score_like_any_other(mlp_run):
    stage, split = mlp_run.stage, mlp_run.split
    unseen = _unseen_rows(split)
    p = stage.predict_proba(unseen)
    assert p.index.equals(unseen.index) and np.isfinite(p).all() and p.between(0, 1).all()
    X = unrecorded_scale(stage, unseen)
    np.testing.assert_allclose(p.to_numpy(), stage.calibrated_model.predict_proba(X)[:, 1])


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def engineered():
    split = stage1_split()
    X_tr, _, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    return split, X_tr


@pytest.mark.parametrize("name", ["shuffled_rows", "reset_index", "foreign_label"])
def test_labels_not_aligned_to_features_are_refused(engineered, name):
    split, X_tr = engineered
    with pytest.raises(ValueError, match="index"):
        CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(
            X_tr, mut.INDEX_MUTATIONS[name](split["y_train"]), X_train_raw=split["X_train"])


@pytest.mark.parametrize("case", ["non_binary", "fractional", "nan"])
def test_invalid_labels_are_refused_before_anything_is_fitted_f18(engineered, monkeypatch, case):
    split, X_tr = engineered
    y = split["y_train"].astype(float)
    y.iloc[:3] = {"non_binary": 2.0, "fractional": 0.5, "nan": np.nan}[case]
    built = []
    monkeypatch.setattr(mlp_stage_mod, "build_fold_designs", lambda *a, **k: built.append(1))
    monkeypatch.setattr(mlp_stage_mod, "tune_stage1_mlp_auc", lambda *a, **k: built.append(1))
    with pytest.raises(ValueError, match="binary 0/1"):
        CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(X_tr, y, X_train_raw=split["X_train"])
    assert not built


def test_test_labels_must_align_with_the_test_rows(mlp_run):
    with pytest.raises(ValueError, match="index"):
        mlp_run.stage.to_stage_output(mlp_run.X_test, mut.shuffle_rows(mlp_run.split["y_test"]))


def test_out_of_order_calls_are_refused(mlp_run):
    stage = CalibratedStage1MLP(**MLP_TEST_CONFIG)
    X, y = mlp_run.X_test, mlp_run.split["y_test"]
    for call in (lambda: stage.predict_proba(X), lambda: stage.predict(X),
                 lambda: stage.to_stage_output(X, y), lambda: stage.describe(),
                 lambda: stage.metrics_table(y, np.full(len(y), 0.5), "test")):
        with pytest.raises(RuntimeError, match="fit"):
            call()


# ----------------------------------------------------------------------
# held-out test labels
# ----------------------------------------------------------------------
def test_test_labels_do_not_influence_anything_learned(mlp_run):
    split = stage1_split()
    moved = fit_mlp_stage1(stage1_split(y_test=mut.permute_labels(split["y_test"], seed=11)))
    base = mlp_run
    assert not moved.output.y_test.equals(base.output.y_test)       # the labels did change
    pd.testing.assert_frame_equal(moved.X_train, base.X_train)
    pd.testing.assert_frame_equal(moved.X_test, base.X_test)
    assert moved.stage.best_params == base.stage.best_params
    pd.testing.assert_series_equal(moved.stage.proba_train_oof, base.stage.proba_train_oof)
    pd.testing.assert_series_equal(moved.stage.train_fold_id, base.stage.train_fold_id)
    assert moved.stage.optimal_threshold == base.stage.optimal_threshold
    assert moved.stage.calibration_method == base.stage.calibration_method
    pd.testing.assert_series_equal(moved.output.test_proba, base.output.test_proba)
    assert moved.output.split_fingerprint != base.output.split_fingerprint


# ----------------------------------------------------------------------
# tuning, determinism and parallelism
# ----------------------------------------------------------------------
def test_tuned_parameters_are_the_ones_fitted(engineered):
    split, X_tr = engineered
    stage = CalibratedStage1MLP(**{**MLP_TEST_CONFIG, "n_trials": 3}).fit(
        X_tr, split["y_train"], X_train_raw=split["X_train"])
    trials = stage.tuning_trials
    assert len(trials) == 3 and (trials["state"] == "COMPLETE").all()     # pruning starts at trial 6
    assert stage.best_cv_roc_auc == pytest.approx(trials["value"].max())
    assert set(stage.best_params) == {"n_layers", "width", "activation", "alpha",
                                      "learning_rate_init", "batch_size", "pos_ratio"}
    kwargs = params_to_kwargs(stage.best_params, **stage.config.estimator_fixed_kwargs)
    assert {k: stage.model.get_params()[k] for k in kwargs} == kwargs
    out = stage.to_stage_output(_quiet(engineer_features, split["X_train"], split["X_test"],
                                       split["y_train"])[1], split["y_test"])
    assert (out.best_params, out.best_cv_score) == (stage.best_params, stage.best_cv_roc_auc)


def test_parallel_and_serial_fits_are_identical(plain):
    """The notebook runs the calibration folds with n_jobs=-1; every other
    Stage 1 test runs them serially."""
    split, X_tr, X_te, serial, _ = plain
    parallel = CalibratedStage1MLP(**{**MLP_TEST_CONFIG, "n_jobs": 2}).fit(
        X_tr, split["y_train"], X_train_raw=split["X_train"])
    pd.testing.assert_series_equal(parallel.proba_train_oof, serial.proba_train_oof)
    pd.testing.assert_series_equal(parallel.predict_proba(X_te), serial.predict_proba(X_te))
    assert parallel.calibration_fold_family_ == serial.calibration_fold_family_
    assert parallel.calibration_metrics == serial.calibration_metrics
    assert parallel.optimal_threshold == serial.optimal_threshold


# ----------------------------------------------------------------------
# artifact
# ----------------------------------------------------------------------
def test_only_a_fitted_stage_can_be_saved(tmp_path):
    with pytest.raises(ValueError, match="not fitted"):
        save_stage1_mlp(CalibratedStage1MLP(**MLP_TEST_CONFIG), tmp_path)


@pytest.fixture(scope="module")
def saved(plain, tmp_path_factory):
    split, X_tr, X_te, stage, output = plain
    out_dir = tmp_path_factory.mktemp("mlp")
    path = _quiet(save_stage1_mlp, stage, out_dir, stage_output=output, test_metrics={"f2": 0.5})
    so_path = _quiet(output.save, out_dir)
    return out_dir, path, so_path


def test_reloaded_artifact_scores_held_out_and_unseen_rows_identically(plain, saved):
    split, _, X_te, stage, output = plain
    art = _quiet(load_stage1_mlp, saved[1])
    for X in (X_te, _unseen_rows(split)):
        p = art["calibrated_model"].predict_proba(art["scaler"].transform(X[art["feature_names"]]))[:, 1]
        np.testing.assert_array_equal(p, stage.predict_proba(X).to_numpy())
    np.testing.assert_array_equal(art["calibrated_model"].predict_proba(
        art["scaler"].transform(X_te))[:, 1], output.test_proba.to_numpy())
    assert art["model"].best_epoch_ == stage.model.best_epoch_
    assert art["model"].history_.equals(stage.model.history_)


def test_artifact_keeps_schema_config_and_provenance(plain, saved):
    _, _, _, stage, output = plain
    out_dir, path, _ = saved
    art = _quiet(load_stage1_mlp, path)
    assert art["artifact_version"] == "2.0"
    assert art["feature_names"] == list(STAGE1_FEATURES)
    assert list(art["scaler"].feature_names_in_) == list(STAGE1_FEATURES)
    meta = json.loads((out_dir / "stage1_mlp_metadata.json").read_text())
    assert Stage1MLPConfig.from_dict(meta["config"]) == stage.config      # via JSON lists
    assert not {"scaler", "model", "calibrated_model", "train_proba_oof", "train_fold_id",
                "y_train", "stage_output"} & set(meta)
    assert (art["optimal_threshold"], art["threshold_source"]) == (stage.optimal_threshold,
                                                                   stage.THRESHOLD_SOURCE)
    assert art["oof_provenance"] == stage.oof_provenance_ and "nested cross-fit" in art["oof_provenance"]
    assert art["calibration_method"] == stage.calibration_method
    pd.testing.assert_series_equal(art["train_proba_oof"], stage.proba_train_oof)
    pd.testing.assert_series_equal(art["train_fold_id"], stage.train_fold_id)
    pd.testing.assert_series_equal(art["y_train"], stage.y_train_)
    so = art["stage_output"]
    assert (so["stage"], so["arm"], so["model"]) == ("stage1", "blackbox", "calibrated_mlp")
    assert so["config"]["oof_preprocessing"]["feature_fit_scope"] == "per_fold"
    assert so["split_fingerprint"] == output.split_fingerprint
    sweep = pd.read_csv(out_dir / "stage1_mlp_threshold_sweep.csv")
    np.testing.assert_allclose(sweep["threshold"], GRID)
    assert len(pd.read_csv(out_dir / "stage1_mlp_tuning_trials.csv")) == stage.config.n_trials
    assert len(pd.read_csv(out_dir / "stage1_mlp_training_history.csv")) == len(stage.model.history_)


def test_the_contract_round_trips_and_a_wrong_split_is_refused(plain, saved):
    _, _, _, _, output = plain
    art = joblib.load(saved[1])
    for loaded in (StageOutput.from_dict(art), StageOutput.load(saved[2])):
        pd.testing.assert_series_equal(loaded.train_proba_oof, output.train_proba_oof)
        pd.testing.assert_series_equal(loaded.test_proba, output.test_proba)
        assert loaded.split_fingerprint == output.split_fingerprint
    d = art["stage_output"]
    relabelled = mut.with_field(d, "y_test", mut.permute_labels(d["y_test"], seed=3))
    with pytest.raises(StageOutputError, match="split_fingerprint"):
        StageOutput.from_dict(relabelled)
    moved = {k: (v.rename(index={v.index[0]: -1}) if k in ("test_proba", "y_test") else v)
             for k, v in d.items()}
    with pytest.raises(StageOutputError, match="split_fingerprint"):
        StageOutput.from_dict(moved)


def test_artifacts_without_a_current_contract_are_refused_downstream(plain, tmp_path):
    split, _, _, stage, output = plain
    bare = _quiet(save_stage1_mlp, stage, tmp_path, name="bare")           # no stage_output
    with pytest.raises(StageOutputError, match="WITHOUT a StageOutput"):
        StageOutput.from_dict(joblib.load(bare))
    with pytest.raises(Stage4InputError):
        stage_output_stream(bare, "mlp", "MLP", len(split["y_train"]))
    legacy = {k: v for k, v in joblib.load(bare).items() if k != "stage_output"}  # v1.0 layout
    with pytest.raises(Stage4InputError, match="no StageOutput contract"):
        stage_output_stream(legacy, "mlp", "MLP", len(split["y_train"]))
    stale = mut.with_field(output.to_dict(), "oof_provenance", "refittable (cv=10)")  # pre-F16/F19
    stream, checks = stage_output_stream(stale, "mlp", "MLP", len(split["y_train"]))
    with pytest.raises(ValidationError, match="nested inside each fold"):
        build_stage4_inputs(split, [stream], extra_checks=checks, verbose=False)


# ----------------------------------------------------------------------
# Stage 4 handoff
# ----------------------------------------------------------------------
def test_both_saved_forms_feed_the_same_stage4_stream(plain, saved):
    split, _, _, stage, output = plain
    n = len(split["y_train"])
    streams = [stage_output_stream(p, "mlp", "MLP", n) for p in (saved[1], saved[2])]
    for stream, checks in streams:
        np.testing.assert_array_equal(stream.train_proba, stage.proba_train_oof.to_numpy())
        np.testing.assert_array_equal(stream.test_proba, output.test_proba.to_numpy())
        np.testing.assert_array_equal(stream.train_fold_id, stage.train_fold_id.to_numpy())
        assert stream.train_source.endswith("train_proba_oof")
        assert stream.notes["arm"] == "blackbox" and stream.model == "calibrated_mlp"
        assert stream.standalone_threshold == stage.optimal_threshold
        assert stream.fold_seed == SEED
        assert all(fn() for _, _, _, fn, _ in checks), [name for _, _, name, fn, _ in checks if not fn()]


def test_stage4_takes_the_mlp_stream_for_its_own_split_only(plain, saved):
    split, _, _, stage, _ = plain
    stream, checks = stage_output_stream(saved[2], "mlp", "MLP", len(split["y_train"]))
    inputs = build_stage4_inputs(split, [stream], extra_checks=checks, verbose=False)
    assert inputs["mlp"].fingerprint_scheme == "stage_io"
    assert inputs.fold_provenance["streams"]["mlp"]["seed"] == SEED
    assert inputs.fold_provenance["streams"]["mlp"]["n_folds"] == stage.config.cv_folds
    other = {**split, "y_test": mut.permute_labels(split["y_test"], seed=4)}
    stream, checks = stage_output_stream(saved[2], "mlp", "MLP", len(split["y_train"]))
    with pytest.raises(ValidationError, match="split fingerprint"):
        build_stage4_inputs(other, [stream], extra_checks=checks, verbose=False)


# ----------------------------------------------------------------------
# the black-box notebook's Stage 1 calls (read, never executed)
# ----------------------------------------------------------------------
def _notebook_calls():
    cells = [("".join(c["source"])) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    trees = [ast.parse("\n".join(line for line in src.splitlines()
                                 if not line.lstrip().startswith(("%", "!")))) for src in cells]
    consts = {}
    for tree in trees:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                consts[node.targets[0].id] = node.value
    calls = [n for tree in trees for n in ast.walk(tree) if isinstance(n, ast.Call)]

    def named(name):
        return [c for c in calls if (getattr(c.func, "id", None) or getattr(c.func, "attr", None)) == name]

    def value(node):
        if isinstance(node, ast.Name) and node.id in consts:
            node = consts[node.id]
        try:
            return ast.literal_eval(node)
        except ValueError:
            return ast.unparse(node)
    return named, value, consts


def test_the_notebook_runs_stage1_under_the_tested_protocol():
    named, value, consts = _notebook_calls()
    (ctor,) = named("CalibratedStage1MLP")
    kw = {k.arg: value(k.value) for k in ctor.keywords}
    assert kw == {"calibration_method": "auto", "cv_folds": CV, "n_trials": 100, "random_state": SEED}
    fits = [c for c in named("fit") if any(k.arg == "X_train_raw" for k in c.keywords)]
    assert len(fits) == 1 and ast.unparse(fits[0].func.value) == "stage1"      # F3: raw rows passed
    assert ast.unparse(consts["STAGE1_THRESHOLD"]) == "stage1.optimal_threshold"
    assert ast.unparse(consts["STAGE1_THRESHOLD_SOURCE"]) == "stage1.THRESHOLD_SOURCE"
    (handoff,) = named("to_stage_output")
    kw = {k.arg: ast.unparse(k.value) for k in handoff.keywords}
    assert (kw["threshold"], kw["threshold_source"]) == ("STAGE1_THRESHOLD", "STAGE1_THRESHOLD_SOURCE")
    (save,) = named("save_stage1_mlp")
    assert {k.arg for k in save.keywords} >= {"stage_output"}
    meta_xgb = [c for c in named("train_meta_xgb_stage")
                if any(k.arg == "mlp_path" and ast.unparse(k.value) == "MLP_S1_PATH" for k in c.keywords)]
    assert len(meta_xgb) == 1 and ast.unparse(consts["MLP_S1_PATH"]).startswith("MLP_S1.save(")
