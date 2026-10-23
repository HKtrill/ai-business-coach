"""
GlobalSplitManager: the one train/test split both cascades share.

Contract: stratified, disjoint and exhaustive; deterministic per seed; rows
keep their original index labels; features and target stay aligned; the
attached CV object reproduces the shared fold plan (``fold_assignment``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.global_splits.split_manager import GlobalSplitManager
from shared.stage_io import fold_assignment
from tests.support.synthetic import TARGET


def _split(frame, test_size=0.25, seed=0, folds=5):
    return GlobalSplitManager(test_size=test_size, random_state=seed,
                              n_cv_folds=folds).create_split(frame, target_col=TARGET)


def test_split_is_disjoint_exhaustive_and_sized(bank_frame):
    s = _split(bank_frame)
    train, test = set(s["X_train"].index), set(s["X_test"].index)
    assert not train & test
    assert train | test == set(bank_frame.index)
    assert len(test) == 100
    np.testing.assert_array_equal(s["train_idx"], s["X_train"].index.to_numpy())
    np.testing.assert_array_equal(s["test_idx"], s["X_test"].index.to_numpy())


def test_split_is_stratified(bank_frame):
    s = _split(bank_frame)
    assert s["y_train"].mean() == pytest.approx(bank_frame[TARGET].mean(), abs=0.01)
    assert s["y_test"].mean() == pytest.approx(bank_frame[TARGET].mean(), abs=0.01)


def test_split_is_deterministic_per_seed(bank_frame):
    a, b, c = _split(bank_frame, seed=0), _split(bank_frame, seed=0), _split(bank_frame, seed=1)
    assert a["X_test"].index.equals(b["X_test"].index)
    pd.testing.assert_frame_equal(a["X_train"], b["X_train"])
    assert not a["X_test"].index.equals(c["X_test"].index)


def test_rows_keep_their_original_labels_and_values(bank_frame):
    s = _split(bank_frame)
    for part in ("train", "test"):
        X, y = s[f"X_{part}"], s[f"y_{part}"]
        assert X.index.isin(bank_frame.index).all()
        pd.testing.assert_frame_equal(X, bank_frame.drop(columns=TARGET).loc[X.index])
        assert X.index.equals(y.index)                                   # feature/target alignment
        pd.testing.assert_series_equal(y, bank_frame.loc[y.index, TARGET])


def test_target_removed_and_feature_order_kept(bank_frame):
    s = _split(bank_frame)
    expected = [c for c in bank_frame.columns if c != TARGET]
    assert TARGET not in s["X_train"].columns
    assert list(s["X_train"].columns) == list(s["X_test"].columns) == expected
    assert s["feature_names"] == expected and s["n_features"] == len(expected)


def test_attached_cv_reproduces_the_shared_fold_plan(bank_frame):
    s = _split(bank_frame, seed=7, folds=5)
    from_cv = np.full(len(s["y_train"]), -1)
    for k, (_, va) in enumerate(s["cv"].split(s["X_train"], s["y_train"])):
        from_cv[va] = k
    shared = fold_assignment(s["y_train"], cv_folds=5, random_state=7)
    np.testing.assert_array_equal(from_cv, shared.to_numpy())


def test_save_and_load_round_trip(bank_frame, tmp_path):
    manager = GlobalSplitManager(test_size=0.25, random_state=0, n_cv_folds=5)
    s = manager.create_split(bank_frame, target_col=TARGET)
    path = tmp_path / "nested" / "split.pkl"
    manager.save_split(str(path))
    loaded = GlobalSplitManager.load_split(str(path))
    for key in ("X_train", "X_test"):
        pd.testing.assert_frame_equal(loaded[key], s[key])
    for key in ("y_train", "y_test"):
        pd.testing.assert_series_equal(loaded[key], s[key])


def test_summary_and_save_require_a_split():
    manager = GlobalSplitManager()
    with pytest.raises(RuntimeError, match="create_split"):
        manager.get_split_summary()
    with pytest.raises(RuntimeError, match="create_split"):
        manager.save_split("unused.pkl")
