"""
Shared feature contracts (shared.stage3.data), as implemented today:

* ``Stage3FeatureContract`` enforces the column SET and duplicates, and
  realigns column ORDER by name; it does not check dtypes.
* ``FeatureCleaner`` learns its imputation medians from the rows it is fit on.
* ``Stage3Block`` fits the feature pipeline inside each fold (``per_fold``),
  including the inner folds of each tuning fold (audit F22), never on
  held-out or test rows, and keeps the row index intact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.stage3.data import FeatureCleaner, Stage3Block, Stage3FeatureContract
from tests.support import mutations as mut
from tests.support.checks import assert_never_fit_on, assert_oof_provenance
from tests.support.recorders import CallLog, RecordingEstimator


class Passthrough:
    """Index-preserving no-op feature pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ResetsIndex(Passthrough):
    """A broken pipeline that drops the row labels."""

    def transform(self, X):
        return X.reset_index(drop=True)


# ----------------------------------------------------------------------
# Stage3FeatureContract
# ----------------------------------------------------------------------
@pytest.fixture
def frames(bank_split):
    return bank_split["X_train"], bank_split["X_test"]


@pytest.mark.parametrize("name, outcome", [
    ("missing_column", "does not match the EBM feature contract"),
    ("extra_column", "does not match the EBM feature contract"),
    ("renamed_column", "does not match the EBM feature contract"),
    ("duplicated_column", "duplicate columns"),
    ("reordered_columns", "realigned"),
    ("changed_dtype", "accepted"),
])
def test_contract_on_both_splits(frames, name, outcome):
    X_tr, X_te = frames
    expected = list(X_tr.columns)
    mutate = mut.COLUMN_MUTATIONS[name]
    contract = Stage3FeatureContract(expected)
    if outcome in ("realigned", "accepted"):
        a, b, order = contract.align(mutate(X_tr), mutate(X_te))
        assert order == expected and list(a.columns) == list(b.columns) == expected
    else:
        with pytest.raises(ValueError, match=outcome):
            contract.validate(mutate(X_tr), mutate(X_te))


@pytest.mark.parametrize("name, outcome", [
    ("missing_column", "train/test columns differ"),
    ("extra_column", "train/test columns differ"),
    ("duplicated_column", "X_test has duplicate columns"),
    ("reordered_columns", "realigned"),
])
def test_contract_when_only_test_changes(frames, name, outcome):
    X_tr, X_te = frames
    contract = Stage3FeatureContract(list(X_tr.columns))
    if outcome == "realigned":
        _, b, _ = contract.align(X_tr, mut.COLUMN_MUTATIONS[name](X_te))
        assert list(b.columns) == list(X_tr.columns)
    else:
        with pytest.raises(ValueError, match=outcome):
            contract.validate(X_tr, mut.COLUMN_MUTATIONS[name](X_te))


def test_contract_requires_dataframes(frames):
    X_tr, X_te = frames
    with pytest.raises(TypeError, match="must be a DataFrame"):
        Stage3FeatureContract(list(X_tr.columns)).validate(X_tr.to_numpy(), X_te)


def test_contract_without_expected_features_only_checks_agreement(frames):
    X_tr, X_te = frames
    extra_tr, extra_te = mut.add_column(X_tr), mut.add_column(X_te)
    assert Stage3FeatureContract(None).validate(extra_tr, extra_te) == list(extra_tr.columns)


# ----------------------------------------------------------------------
# FeatureCleaner
# ----------------------------------------------------------------------
def test_cleaner_learns_medians_from_fit_rows_only(frames):
    X_tr, X_te = frames
    col = "euribor3m"
    X_tr = X_tr.copy()
    X_tr.loc[X_tr.index[:5], col] = np.nan
    X_te = X_te.copy()
    X_te[col] = 1e6                                          # extreme test-only values
    X_te.loc[X_te.index[0], col] = np.nan

    cleaner = FeatureCleaner().fit(X_tr)
    assert cleaner.medians_[col] == pytest.approx(X_tr[col].median())
    assert cleaner.transform(X_te).loc[X_te.index[0], col] == pytest.approx(X_tr[col].median())
    assert cleaner.report_["columns_imputed"] == [col]


def test_cleaner_replaces_infinities_then_imputes(frames):
    X_tr, _ = frames
    X = X_tr.copy()
    X.loc[X.index[0], "euribor3m"] = np.inf
    out = FeatureCleaner().fit_transform(X)
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    assert FeatureCleaner().fit(X).report_["n_inf_replaced"] == 1


def test_cleaner_failure_modes(frames):
    X_tr, X_te = frames
    with pytest.raises(ValueError, match="Call fit"):
        FeatureCleaner().transform(X_te)
    with pytest.raises(ValueError, match="missing fitted columns"):
        FeatureCleaner().fit(X_tr).transform(mut.drop_column(X_te))
    all_nan = X_tr.copy()
    all_nan["euribor3m"] = np.nan
    with pytest.raises(AssertionError, match="NaN values remain"):
        FeatureCleaner().fit_transform(all_nan)


# ----------------------------------------------------------------------
# Stage3Block
# ----------------------------------------------------------------------
N_TUNE, N_EVAL = 3, 4


def build_block(split, factory=Passthrough, y_train=None, y_test=None, scope="per_fold"):
    X_tr = split["X_train"]
    return Stage3Block.build(
        X_tr, split["y_train"] if y_train is None else y_train,
        split["X_test"], split["y_test"] if y_test is None else y_test,
        factory, feature_fit_scope=scope, n_tune_folds=N_TUNE, n_eval_folds=N_EVAL,
        random_state=0, expected_features=list(X_tr.columns), verbose=False,
    )


def test_block_is_deterministic_and_keeps_the_index(bank_split):
    a, b = build_block(bank_split), build_block(bank_split)
    assert a.block_fingerprint == b.block_fingerprint
    assert a.X_train.index.equals(bank_split["X_train"].index)
    assert a.X_test.index.equals(bank_split["X_test"].index)
    assert a.tune_plan.n_rows == a.report_plan.n_rows == len(bank_split["y_train"])


def test_block_rejects_a_pipeline_that_changes_the_index(bank_split):
    with pytest.raises(ValueError, match="changed the row index/order"):
        build_block(bank_split, factory=ResetsIndex)


@pytest.mark.parametrize("name", ["shuffled_rows", "reset_index", "foreign_label"])
def test_block_rejects_label_series_not_indexed_like_x(bank_split, name):
    y_bad = mut.INDEX_MUTATIONS[name](bank_split["y_train"])
    with pytest.raises(ValueError, match="index differs from its feature frame"):
        build_block(bank_split, y_train=y_bad)


def test_block_aligns_bare_label_arrays_by_position_f6(bank_split):
    """F6: arrays are accepted and trusted to be in X's row order."""
    y = bank_split["y_train"]
    as_series = build_block(bank_split)
    as_array = build_block(bank_split, y_train=y.to_numpy())
    assert as_array.block_fingerprint == as_series.block_fingerprint

    misordered = build_block(bank_split, y_train=y.to_numpy()[::-1])   # silently accepted
    assert not misordered.y_train.equals(as_series.y_train)
    with pytest.raises(ValueError, match="length"):
        build_block(bank_split, y_train=y.to_numpy()[:-1])


def test_block_fits_features_inside_each_fold_only(bank_split):
    log = CallLog()
    build_block(bank_split, factory=lambda: RecordingEstimator(Passthrough(), log))
    train_idx = bank_split["X_train"].index
    assert_never_fit_on(log, bank_split["X_test"].index)

    fold_fits = sorted(fid for fid, rows in log.fits().items() if rows < set(train_idx))
    assert len(fold_fits) == N_TUNE + N_EVAL + N_TUNE * N_TUNE
    plans = {"tuning": fold_fits[:N_TUNE], "reporting": fold_fits[N_TUNE:N_TUNE + N_EVAL]}
    for fits in plans.values():
        sub_log = CallLog(c for c in log if c.fit_id in fits)
        assert_oof_provenance(sub_log, train_idx, methods=("transform",))

    # inner folds: within each tuning fold's training rows only (F22)
    inner = fold_fits[N_TUNE + N_EVAL:]
    block = build_block(bank_split)
    for k, (tr, va) in enumerate(block.tune_plan):
        sub_log = CallLog(c for c in log if c.fit_id in inner[k * N_TUNE:(k + 1) * N_TUNE])
        assert_oof_provenance(sub_log, train_idx[tr], methods=("transform",))
        assert not any(set(c.keys) & set(train_idx[va]) for c in sub_log)


def test_global_scope_fits_the_pipeline_once_on_all_training_rows(bank_split):
    """The documented LEAKY comparison mode: no per-fold refits."""
    log = CallLog()
    build_block(bank_split, factory=lambda: RecordingEstimator(Passthrough(), log),
                scope="global")
    fits = log.fits()
    assert list(fits.values()) == [frozenset(bank_split["X_train"].index)]
