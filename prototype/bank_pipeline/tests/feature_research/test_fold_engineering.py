"""
Fold plumbing shared by the research trainers (model_training.shared.folds).

build_folds must refit the feature pipeline inside every fold, so no
validation row's label reaches the features that row is scored on;
engineer_full is the one deliberate all-rows fit, for the final model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import PredefinedSplit

from feature_research.feature_engineering import EBM_FEATURES
from feature_research.model_training.shared.folds import build_folds, engineer_full
from tests.support import mutations as mut
from tests.support.checks import assert_oof_provenance
from tests.support.recorders import CallLog, RecordingEstimator

FEATURES = list(EBM_FEATURES)


def test_folds_follow_the_declared_splitter(bank_split, make_fp, cv):
    X, y = bank_split["X_train"], bank_split["y_train"]
    folds = build_folds(X, y, FEATURES, cv, make_fp)
    expected = list(cv.split(X, y))
    assert len(folds) == len(expected)
    for f, (tr, va) in zip(folds, expected):
        np.testing.assert_array_equal(f.va_idx, va)
        assert f.X_va.index.equals(X.index[va]) and f.y_va.index.equals(X.index[va])
        assert f.X_tr.index.equals(X.index[tr]) and f.y_tr.index.equals(X.index[tr])
        assert list(f.X_tr.columns) == list(f.X_va.columns) == FEATURES
    held_out = np.concatenate([f.va_idx for f in folds])
    assert sorted(held_out) == list(range(len(X)))                      # every row once


def test_feature_pipeline_is_refit_inside_each_fold(bank_split, make_fp, cv):
    log = CallLog()
    X, y = bank_split["X_train"], bank_split["y_train"]
    build_folds(X, y, FEATURES, cv, lambda: RecordingEstimator(make_fp(), log))
    assert_oof_provenance(log, X.index, methods=("transform",))


def test_validation_labels_never_reach_their_own_features(bank_split, make_fp, cv):
    """With the folds held fixed, permuting fold 0's labels leaves fold 0's
    features untouched, but does move the folds that trained on those rows."""
    X, y = bank_split["X_train"], bank_split["y_train"]
    test_fold = np.empty(len(X), dtype=int)
    for k, (_, va) in enumerate(cv.split(X, y)):
        test_fold[va] = k
    fixed = PredefinedSplit(test_fold)
    fold0_rows = X.index[test_fold == 0]

    base = build_folds(X, y, FEATURES, fixed, make_fp)
    moved = build_folds(X, mut.permute_labels(y, seed=5, rows=fold0_rows), FEATURES, fixed, make_fp)

    pd.testing.assert_frame_equal(moved[0].X_va, base[0].X_va)
    pd.testing.assert_frame_equal(moved[0].X_tr, base[0].X_tr)
    assert not moved[1].X_va["dow_month_encoded"].equals(base[1].X_va["dow_month_encoded"])


def test_engineer_full_is_the_single_all_rows_fit(bank_split, make_fp):
    """Notebook contract (Cells 14A, 16A): the final model's input is
    FeaturePipeline().fit_transform(X_train, y_train)[features]."""
    log = CallLog()
    X, y = bank_split["X_train"], bank_split["y_train"]
    out = engineer_full(X, y, FEATURES, lambda: RecordingEstimator(make_fp(), log))
    assert list(log.fits().values()) == [frozenset(X.index)]
    pd.testing.assert_frame_equal(out, make_fp().fit_transform(X, y)[FEATURES])


def test_build_folds_refuses_misaligned_inputs(bank_split, make_fp, cv):
    X, y = bank_split["X_train"], bank_split["y_train"]
    with pytest.raises(ValueError, match="not aligned"):
        build_folds(X, mut.shuffle_rows(y), FEATURES, cv, make_fp)


def test_build_folds_refuses_non_finite_model_inputs(bank_split, make_fp, cv):
    X, y = bank_split["X_train"].copy(), bank_split["y_train"]
    X.loc[X.index[0], "cons.conf.idx"] = np.nan                         # raw EBM feature
    with pytest.raises(ValueError, match="cons.conf.idx|non-finite|finite"):
        build_folds(X, y, FEATURES, cv, make_fp)


def test_without_a_factory_folds_slice_the_given_frame(bank_split, make_fp, cv):
    """No factory means no fitting: the caller's pre-engineered frame is sliced."""
    X, y = bank_split["X_train"], bank_split["y_train"]
    engineered = make_fp().fit_transform(X, y)
    folds = build_folds(engineered, y, FEATURES, cv)
    for f, (_, va) in zip(folds, cv.split(X, y)):
        pd.testing.assert_frame_equal(f.X_va, engineered.iloc[va][FEATURES])
