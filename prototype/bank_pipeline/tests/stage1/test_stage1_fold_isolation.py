"""
Stage 1 fold isolation for both arms (GLASS LR and black-box MLP): audit F3
(target encodings) and F4 (scaler), via ``shared.stage1.crossfit``.

Contract: every out-of-fold probability comes from an encoder, a scaler and a
calibrated model fitted on that fold's training rows only, so a fold's own
labels and statistics cannot reach its scores; target-encoded inputs are
refused without the raw rows to refit them from; both arms share the folds
and the code. The calibration procedure and the final scorer are covered in
test_stage1_calibration.py.

MLP runs use tiny synthetic folds (~100-200 rows), smaller than the tuned
batch sizes (128-512), so scikit-learn clips the batch and warns. That cannot
happen at production scale (~33k training rows); the filter below matches
that one message only.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import blackbox_pipeline.models.mlp.stage as mlp_stage_mod
import glass_pipeline.lr.lr_stage_calibrated as lr_mod
from blackbox_pipeline.models.mlp import CalibratedStage1MLP
from blackbox_pipeline.models.mlp.estimator import Stage1MLPClassifier
from blackbox_pipeline.models.mlp.tuning import params_to_kwargs, tune_stage1_mlp_auc
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from glass_pipeline.lr.penalty import penalty_kwargs
from glass_pipeline.lr.tuning import run_optuna_tuning
from shared.stage1 import crossfit as crossfit_mod
from shared.stage1.crossfit import build_fold_designs
from shared.stage1.feature_engineering import engineer_features
from shared.stage_io import fold_assignment
from tests.stage1.conftest import MLP_TEST_CONFIG, fold_levels, stage1_split, sub_log
from tests.support import mutations as mut
from tests.support.checks import assert_oof_provenance

pytestmark = pytest.mark.filterwarnings(
    "ignore:Got `batch_size` less than 1 or larger than sample size:UserWarning")

SEED = 42
ARMS = ["glass_lr", "blackbox_mlp"]


@pytest.fixture(params=ARMS)
def run(request):
    return request.getfixturevalue("lr_run" if request.param == "glass_lr" else "mlp_run")


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _n_folds(run) -> int:
    return run.stage.cv_folds if isinstance(run.stage, CalibratedLRStage) else run.stage.config.cv_folds


def _outer_fit_rows(log, train_rows):
    outer, _ = fold_levels(log, train_rows)
    fits = log.fits()
    return {fits[fid] for fid in outer}


# ----------------------------------------------------------------------
# provenance: what each outer fold's preprocessing and model were fitted on
# (inner calibration folds: test_stage1_calibration.py)
# ----------------------------------------------------------------------
def test_target_encoders_are_refit_on_each_folds_training_rows_f3(run):
    train = run.X_train.index
    outer, _ = fold_levels(run.encoder_log, train)
    assert len(outer) == _n_folds(run)
    assert_oof_provenance(sub_log(run.encoder_log, outer), train, methods=("transform",))


def test_scaler_is_refit_on_each_folds_training_rows_f4(run):
    train = run.X_train.index
    outer, _ = fold_levels(run.scaler_log, train)
    assert len(outer) == _n_folds(run)
    assert_oof_provenance(sub_log(run.scaler_log, outer), train, methods=("transform",))


def test_each_folds_encoder_scaler_and_model_see_the_same_rows(run):
    train = run.X_train.index
    assert (_outer_fit_rows(run.oof_log, train) == _outer_fit_rows(run.encoder_log, train)
            == _outer_fit_rows(run.scaler_log, train))


def test_full_train_preprocessing_is_kept_for_test_inference(run):
    """One full-train encoder (engineer_features) and one full-train scaler
    remain, and they are the only objects that ever transform test rows."""
    train, test = frozenset(run.X_train.index), frozenset(run.X_test.index)
    for log in (run.encoder_log, run.scaler_log):
        test_scorers = {c.fit_id for c in log if c.method == "transform" and frozenset(c.keys) == test}
        assert test_scorers and all(log.fits()[fid] == train for fid in test_scorers)


# Behaviour (a fold's own labels cannot move its scores, for both arms and
# both calibration families) is tested end to end in test_stage1_calibration.py.


# ----------------------------------------------------------------------
# tuning folds use the same per-fold preprocessing
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS)
def test_tuning_scores_each_fold_with_its_own_preprocessing(arm, engineered):
    """The trial value is the mean AUC over per-fold matrices, not over the
    globally encoded and scaled matrix (one trial: its parameters do not
    depend on any score)."""
    split, X_tr = engineered
    y = split["y_train"]
    designs = _quiet(build_fold_designs, X_tr, y, cv_folds=3, random_state=SEED,
                     X_raw=split["X_train"])
    X_global = StandardScaler().fit_transform(X_tr)
    if arm == "glass_lr":
        params, study, _ = _quiet(run_optuna_tuning, X_global, y, n_trials=1,
                                  random_state=SEED, cv_folds=3, folds=designs)
        value = study.best_value

        def make():
            return LogisticRegression(C=params["C"], solver="saga", max_iter=5000,
                                      random_state=SEED, class_weight="balanced",
                                      **penalty_kwargs(params["penalty"], params.get("l1_ratio")))
    else:
        fixed = CalibratedStage1MLP(**MLP_TEST_CONFIG).config.estimator_fixed_kwargs
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        params, value, _ = tune_stage1_mlp_auc(X_global, y, cv, 1, fixed, SEED,
                                               verbose=False, folds=designs)

        def make():
            return Stage1MLPClassifier(**params_to_kwargs(params, **fixed))

    def mean_auc(folds):
        return np.mean([roc_auc_score(yv, make().fit(Xt, yt).predict_proba(Xv)[:, 1])
                        for Xt, yt, Xv, yv in folds])

    per_fold = [(d.X_train, d.y_train, d.X_val, d.y_val) for d in designs]
    global_ = [(X_global[d.train], d.y_train, X_global[d.val], d.y_val) for d in designs]
    assert value == pytest.approx(mean_auc(per_fold))
    assert value != pytest.approx(mean_auc(global_))                 # the check can fail


# ----------------------------------------------------------------------
# refused inputs
# ----------------------------------------------------------------------
def _fit(arm, X, y, **kwargs):
    if arm == "glass_lr":
        return _quiet(CalibratedLRStage(tune_hyperparameters=False, cv_folds=3).fit, X, y, **kwargs)
    return CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(X, y, **kwargs)


@pytest.fixture(scope="module")
def engineered():
    split = stage1_split()
    X_tr, _, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    return split, X_tr


@pytest.mark.parametrize("arm", ARMS)
def test_target_encoded_inputs_without_raw_rows_are_refused(arm, engineered):
    split, X_tr = engineered
    with pytest.raises(ValueError, match="target encodings"):
        _fit(arm, X_tr, split["y_train"])


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("case, message", [
    ("shuffled_raw_rows", "same index"),
    ("other_raw_values", "not what the feature engineer produces"),
])
def test_raw_rows_must_rebuild_the_stage_features(arm, engineered, case, message):
    split, X_tr = engineered
    raw = split["X_train"]
    if case == "shuffled_raw_rows":
        raw = mut.shuffle_rows(raw)
    else:
        raw = raw.assign(euribor3m=raw["euribor3m"] + 0.25)
    with pytest.raises(ValueError, match=message):
        _fit(arm, X_tr, split["y_train"], X_train_raw=raw)


# ----------------------------------------------------------------------
# parity: one implementation, one record
# ----------------------------------------------------------------------
def test_both_arms_cross_validate_through_the_same_code():
    shared = crossfit_mod
    assert lr_mod.build_fold_designs is mlp_stage_mod.build_fold_designs is shared.build_fold_designs
    assert lr_mod.calibrate_oof is mlp_stage_mod.calibrate_oof


def test_both_arms_record_per_fold_preprocessing(lr_run, mlp_run):
    expected = crossfit_mod.preprocessing_record(lr_run.split["X_train"])
    assert expected["feature_fit_scope"] == expected["scaler_fit_scope"] == "per_fold"
    assert lr_run.output.config["oof_preprocessing"] == expected
    assert mlp_run.output.config["oof_preprocessing"] == expected


def test_fold_ids_are_unchanged_by_the_fix(run):
    expected = fold_assignment(run.y_train, cv_folds=_n_folds(run), random_state=SEED)
    pd.testing.assert_series_equal(run.output.train_fold_id, expected)
