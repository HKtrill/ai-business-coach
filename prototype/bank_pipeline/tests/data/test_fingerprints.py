"""
Split and fold identity: the fingerprints and fold plans every stage carries.

Two split-fingerprint schemes are in use and both are pinned here:

* ``shared.stage_io.split_fingerprint`` — 12 hex, index AND labels
  (Stage 1 ``StageOutput``, GLASS Stage 3 block);
* ``rf_router.split_fingerprint``       — 16 hex, index only
  (Stage 2 routers, black-box Stage 3 block).

Golden values make any change to either algorithm a deliberate, visible event:
an unnoticed change would make every saved artifact look stale (or worse,
make a stale one look current).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from blackbox_pipeline.models.rf_router import split_fingerprint as router_fingerprint
from shared.stage3.data import FoldPlan
from shared.stage_io import fold_assignment, split_fingerprint
from tests.support import mutations as mut

TRAIN = pd.Index([10007, 10000, 10021])
TEST = pd.Index([10014, 10028])
Y_TRAIN, Y_TEST = [0, 1, 0], [1, 0]


def _frames(train=TRAIN, test=TEST):
    return (pd.DataFrame({"a": np.arange(len(train), dtype=float)}, index=train),
            pd.DataFrame({"a": np.arange(len(test), dtype=float)}, index=test))


# ----------------------------------------------------------------------
# shared.stage_io.split_fingerprint
# ----------------------------------------------------------------------
def test_stage_io_golden_values():
    assert split_fingerprint(TRAIN, TEST, Y_TRAIN, Y_TEST) == "2b6c2c833f38"
    assert split_fingerprint(TRAIN, TEST) == "4cce76c97c8f"


@pytest.mark.parametrize("train, test, y_train, y_test", [
    (TRAIN[::-1], TEST, Y_TRAIN[::-1], Y_TEST),                          # same rows, new order
    (TRAIN[:2], TEST.append(TRAIN[2:]), Y_TRAIN[:2], Y_TEST + Y_TRAIN[2:]),  # row moved to test
    (TRAIN, TEST, [1, 1, 0], Y_TEST),                                    # one label changed
], ids=["reordered", "row_moved", "relabelled"])
def test_stage_io_detects_changed_splits(train, test, y_train, y_test):
    assert split_fingerprint(train, test, y_train, y_test) != "2b6c2c833f38"


def test_stage_io_dtype_sensitivity_f7():
    """F7: values are hashed via str(), so how they print decides the hash."""
    base = split_fingerprint(TRAIN, TEST, pd.Series(Y_TRAIN), pd.Series(Y_TEST))
    same = {
        "int8 labels": split_fingerprint(TRAIN, TEST, np.int8(Y_TRAIN), np.int8(Y_TEST)),
        "list labels": split_fingerprint(TRAIN, TEST, Y_TRAIN, Y_TEST),
        "int32 index": split_fingerprint(TRAIN.astype("int32"), TEST.astype("int32"),
                                         Y_TRAIN, Y_TEST),
    }
    differ = {
        "bool labels": split_fingerprint(TRAIN, TEST, np.bool_(Y_TRAIN), np.bool_(Y_TEST)),
        "float labels": split_fingerprint(TRAIN, TEST, np.float64(Y_TRAIN), np.float64(Y_TEST)),
        "float index": split_fingerprint(TRAIN.astype(float), TEST.astype(float),
                                         Y_TRAIN, Y_TEST),
    }
    assert all(v == base for v in same.values()), same
    assert all(v != base for v in differ.values()), differ


# ----------------------------------------------------------------------
# rf_router.split_fingerprint (index only)
# ----------------------------------------------------------------------
def test_router_golden_value_and_index_only_scope():
    X_tr, X_te = _frames()
    assert router_fingerprint(X_tr, X_te) == "8f1d3d4ad06f5914"
    # feature values (and labels, which it never sees) do not enter the hash
    assert router_fingerprint(X_tr * 10, X_te) == "8f1d3d4ad06f5914"


def test_router_detects_reordered_and_moved_rows():
    X_tr, X_te = _frames()
    assert router_fingerprint(mut.shuffle_rows(X_tr), X_te) != "8f1d3d4ad06f5914"
    X_tr2, X_te2 = _frames(TRAIN[:2], TEST.append(TRAIN[2:]))
    assert router_fingerprint(X_tr2, X_te2) != "8f1d3d4ad06f5914"


def test_the_two_schemes_are_distinct():
    """Same split, different identifiers: compare like with like (see F12)."""
    X_tr, X_te = _frames()
    a, b = split_fingerprint(TRAIN, TEST, Y_TRAIN, Y_TEST), router_fingerprint(X_tr, X_te)
    assert (len(a), len(b)) == (12, 16) and a != b[:12]


# ----------------------------------------------------------------------
# fold_assignment: the shared fold plan
# ----------------------------------------------------------------------
def test_fold_assignment_is_the_documented_stratified_partition(bank_split):
    y = bank_split["y_train"]
    fold = fold_assignment(y, cv_folds=5, random_state=3)
    assert fold.index.equals(y.index) and fold.name == "fold_id"
    expected = np.full(len(y), -1)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=3)
                                .split(np.zeros(len(y)), y)):
        expected[va] = k
    np.testing.assert_array_equal(fold.to_numpy(), expected)
    pos_per_fold = y.groupby(fold).sum()
    assert sorted(fold.unique()) == list(range(5))
    assert pos_per_fold.max() - pos_per_fold.min() <= 1                  # stratified


def test_fold_assignment_is_deterministic_and_order_dependent(bank_split):
    y = bank_split["y_train"]
    a = fold_assignment(y, cv_folds=5, random_state=0)
    pd.testing.assert_series_equal(a, fold_assignment(y, cv_folds=5, random_state=0))
    shuffled = fold_assignment(mut.shuffle_rows(y), cv_folds=5, random_state=0)
    assert not shuffled.reindex(y.index).equals(a)       # row order is part of the plan


def test_fold_assignment_index_for_arrays(bank_split):
    y = bank_split["y_train"]
    fold = fold_assignment(y.to_numpy(), cv_folds=5, random_state=0, index=y.index)
    assert fold.index.equals(y.index)
    assert isinstance(fold_assignment(y.to_numpy(), cv_folds=5, random_state=0).index,
                      pd.RangeIndex)


# ----------------------------------------------------------------------
# FoldPlan (shared.stage3.data)
# ----------------------------------------------------------------------
def test_fold_plan_matches_fold_assignment_and_partitions_rows(bank_split):
    y = bank_split["y_train"]
    plan = FoldPlan.build(y, n_splits=5, random_state=0)
    np.testing.assert_array_equal(plan.fold_id,
                                  fold_assignment(y, cv_folds=5, random_state=0).to_numpy())
    held_out = np.concatenate([va for _, va in plan])
    assert sorted(held_out) == list(range(len(y)))
    for tr, va in plan:
        assert not set(tr) & set(va) and len(tr) + len(va) == len(y)
    restored = FoldPlan.from_dict(plan.to_dict())
    np.testing.assert_array_equal(restored.fold_id, plan.fold_id)
    assert (restored.n_splits, restored.random_state, restored.stratified, restored.purpose) == \
        (plan.n_splits, plan.random_state, plan.stratified, plan.purpose)


@pytest.mark.parametrize("kwargs, message", [
    (dict(n_splits=1), "n_splits must be >= 2"),
    (dict(n_splits=10_000), "exceeds the population size"),
    (dict(n_splits=81), "at least 81 samples in every class"),
    (dict(n_splits=5, stratify=False), "stratify=False is"),
])
def test_fold_plan_rejects_invalid_requests(bank_frame, kwargs, message):
    y = bank_frame["y"]                                                   # 80 positives
    with pytest.raises(ValueError, match=message):
        FoldPlan.build(y, random_state=0, **kwargs)


def test_fold_plan_refuses_the_wrong_population(bank_split):
    plan = FoldPlan.build(bank_split["y_train"], n_splits=5, random_state=0)
    with pytest.raises(ValueError, match="must not be reused across populations"):
        plan.assert_matches(len(bank_split["y_test"]), "test split")


@pytest.mark.parametrize("corruption", ["unassigned", "empty_fold"])
def test_fold_plan_validation_catches_corrupted_assignments(bank_split, corruption):
    plan = FoldPlan.build(bank_split["y_train"], n_splits=5, random_state=0)
    fold_id = plan.fold_id.copy()
    if corruption == "unassigned":
        fold_id[0] = -1
    else:
        fold_id[fold_id == 4] = 3
    with pytest.raises(AssertionError):
        FoldPlan(fold_id=fold_id, n_splits=5, random_state=0, stratified=True).assert_valid()
