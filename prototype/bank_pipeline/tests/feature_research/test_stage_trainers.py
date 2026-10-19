"""
Research trainers for the three stages (feature_research.model_training).

Run with minimal budgets (2 Optuna trials, 3 folds, tiny models). Each trainer
must: refit the feature pipeline inside every tuning and evaluation fold,
return out-of-fold probabilities aligned to the training rows, train its final
model on exactly the stage's feature block, and reproduce itself when rerun.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from feature_research.feature_engineering import EBM_FEATURES, LR_FEATURES
from feature_research.model_training.ebm import train_ebm
from feature_research.model_training.ebm.model import PARAM_KEYS, normalize_params
from feature_research.model_training.lr import train_lr
from feature_research.model_training.rf import (
    BinaryFeaturePipeline, create_binary_features, evaluate_rf, validate_binary_features,
)
from tests.support.checks import assert_oof_provenance
from tests.support.recorders import CallLog, RecordingEstimator
from tests.support.synthetic import TARGET, split_frame

N_FOLDS = 3
EBM_PARAMS = {"learning_rate": 0.05, "max_rounds": 50, "max_bins": 32,
              "max_interaction_bins": 16, "interactions": 0}
RF_PARAMS = {"n_estimators": 25, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.5,
             "minority_weight": 3.0, "class_weight": "balanced"}


def _recording(make_fp, log):
    return lambda: RecordingEstimator(make_fp(), log)


def _assert_per_plan_fold_fitting(log, train_idx, plan_sizes):
    """Fold fits come in plans (tune, eval); each plan must be honest OOF."""
    fold_fits = sorted(fid for fid, rows in log.fits().items() if rows < set(train_idx))
    assert len(fold_fits) == sum(plan_sizes)
    start = 0
    for size in plan_sizes:
        plan = set(fold_fits[start:start + size])
        assert_oof_provenance(CallLog(c for c in log if c.fit_id in plan), train_idx,
                              methods=("transform",))
        start += size
    full_fits = [rows for rows in log.fits().values() if rows == set(train_idx)]
    assert len(full_fits) == 1                                           # the final model only


def _assert_oof_series(oof, y):
    assert isinstance(oof, pd.Series) and oof.index.equals(y.index)
    assert np.isfinite(oof.to_numpy()).all() and oof.between(0, 1).all()


# ----------------------------------------------------------------------
# Stage 1 research: logistic regression
# ----------------------------------------------------------------------
def _train_lr(split, factory, cv):
    df = split["X_train"].assign(**{TARGET: split["y_train"]})
    return train_lr(df, LR_FEATURES, TARGET, n_trials=2, n_eval_folds=N_FOLDS,
                    random_state=0, cv=cv, feature_factory=factory)


def test_lr_trainer_contract_and_fold_fitting(bank_split, make_fp, cv):
    log = CallLog()
    res = _train_lr(bank_split, _recording(make_fp, log), cv)
    _assert_oof_series(res.oof_proba, bank_split["y_train"])
    assert list(res.pipe.feature_names_in_) == LR_FEATURES          # notebook Cell 18 check
    assert sorted(res.feature_importance["feature"]) == sorted(LR_FEATURES)
    assert {"auc_mean", "recall_at_10fpr"} <= set(res.metrics)
    _assert_per_plan_fold_fitting(log, bank_split["X_train"].index, [N_FOLDS, N_FOLDS])


def test_lr_trainer_is_reproducible_and_recording_is_transparent(bank_split, make_fp, cv):
    a = _train_lr(bank_split, make_fp, cv)
    b = _train_lr(bank_split, _recording(make_fp, CallLog()), cv)
    assert a.params == b.params
    pd.testing.assert_series_equal(a.oof_proba, b.oof_proba)


def test_lr_trainer_requires_the_target_column(bank_split, make_fp, cv):
    with pytest.raises(KeyError, match="target column"):
        train_lr(bank_split["X_train"], LR_FEATURES, TARGET, n_trials=1, cv=cv,
                 feature_factory=make_fp)


# ----------------------------------------------------------------------
# Stage 2 research: binary features + random forest
# ----------------------------------------------------------------------
def test_binary_pipeline_reproduces_the_binned_block(bank_split, make_fp):
    """Notebook Cell 18 contract: make_rf_feature_pipeline() output equals df_rf_binary."""
    X, y = bank_split["X_train"], bank_split["y_train"]
    df_bin, cols = create_binary_features(make_fp().fit_transform(X, y).assign(**{TARGET: y}), TARGET)
    assert validate_binary_features(df_bin, TARGET, verbose=False) is True
    assert set(np.unique(df_bin[cols].to_numpy())) <= {0, 1}
    per_fold = BinaryFeaturePipeline(make_fp).fit_transform(X, y)
    pd.testing.assert_frame_equal(per_fold[cols], df_bin[cols])


def test_binary_validation_rejects_broken_bin_groups(bank_split, make_fp):
    X, y = bank_split["X_train"], bank_split["y_train"]
    df_bin, cols = create_binary_features(make_fp().fit_transform(X, y).assign(**{TARGET: y}), TARGET)
    broken = df_bin.copy()
    broken.loc[broken.index[0], cols[0]] = 1 - broken.loc[broken.index[0], cols[0]]
    with pytest.raises(ValueError, match="bin sum"):
        validate_binary_features(broken, TARGET, verbose=False)


def _rf_cols(split, make_fp):
    X, y = split["X_train"], split["y_train"]
    return create_binary_features(make_fp().fit_transform(X, y).assign(**{TARGET: y}), TARGET)[1]


def test_rf_evaluation_refits_binary_features_in_each_fold(bank_split, make_fp, cv):
    log = CallLog()
    X, y = bank_split["X_train"], bank_split["y_train"]
    factory = lambda: RecordingEstimator(BinaryFeaturePipeline(make_fp), log)   # noqa: E731
    metrics = evaluate_rf(RF_PARAMS, X, y, N_FOLDS, 0, cv=cv, features=_rf_cols(bank_split, make_fp),
                          feature_factory=factory)
    assert {"auc_mean", "auc_std", "recall_at_10fpr"} <= set(metrics)
    fold_fits = [rows for rows in log.fits().values() if rows < set(X.index)]
    assert len(fold_fits) == N_FOLDS
    assert_oof_provenance(log, X.index, methods=("transform",))


def test_rf_out_of_fold_auc_is_at_chance_without_signal(null_signal_frame, make_fp, cv):
    """Null-signal control: honest OOF scores sit near 0.5, while the same
    forest scored in-sample looks excellent, so the check can tell them apart."""
    split = split_frame(null_signal_frame)
    X, y = split["X_train"], split["y_train"]
    cols = _rf_cols(split, make_fp)
    metrics = evaluate_rf(RF_PARAMS, X, y, N_FOLDS, 0, cv=cv, features=cols,
                          feature_factory=partial(BinaryFeaturePipeline, make_fp))
    assert 0.35 < metrics["auc_mean"] < 0.65

    X_bin = BinaryFeaturePipeline(make_fp).fit_transform(X, y)[cols]
    in_sample = RandomForestClassifier(n_estimators=25, min_samples_leaf=1, random_state=0)
    in_sample.fit(X_bin, y)
    assert roc_auc_score(y, in_sample.predict_proba(X_bin)[:, 1]) > 0.8


# ----------------------------------------------------------------------
# Stage 3 research: EBM
# ----------------------------------------------------------------------
def _train_ebm(split, factory, cv):
    return train_ebm(split["X_train"], split["y_train"], EBM_FEATURES, cv=cv,
                     feature_factory=factory, params=dict(EBM_PARAMS), n_folds_eval=N_FOLDS,
                     random_state=0, n_jobs=1)


def test_ebm_trainer_contract_fold_fitting_and_reproducibility(bank_split, make_fp, cv):
    log = CallLog()
    recorded = _train_ebm(bank_split, _recording(make_fp, log), cv)
    plain = _train_ebm(bank_split, make_fp, cv)
    _assert_oof_series(recorded.oof_proba, bank_split["y_train"])
    pd.testing.assert_series_equal(recorded.oof_proba, plain.oof_proba)  # deterministic
    assert recorded.study is None                                       # params given: no search
    assert list(recorded.features) == list(EBM_FEATURES)
    assert list(recorded.pipe.feature_names_in_) == list(EBM_FEATURES)
    _assert_per_plan_fold_fitting(log, bank_split["X_train"].index, [N_FOLDS])


def test_ebm_params_missing_keys_raise_and_extra_keys_are_dropped_f14():
    """F14 (documented behaviour): unlike the cascade's EBM estimator, research
    params drop unknown keys silently instead of rejecting them."""
    with pytest.raises(KeyError):
        normalize_params({k: v for k, v in EBM_PARAMS.items() if k != "max_bins"})
    out = normalize_params({**EBM_PARAMS, "max_depht": 3})
    assert list(out) == list(PARAM_KEYS) and "max_depht" not in out


def test_ebm_trainer_refuses_duplicate_feature_names(bank_split, make_fp, cv):
    with pytest.raises(ValueError, match="duplicate"):
        train_ebm(bank_split["X_train"], bank_split["y_train"], [*EBM_FEATURES, EBM_FEATURES[0]],
                  cv=cv, feature_factory=make_fp, params=dict(EBM_PARAMS), n_folds_eval=2,
                  n_jobs=1)
