"""
GLASS Stage 1 -> Stage 2 hand-off.

Both stages read the same global split and each derives its own features from
the raw columns. Contract: identical training rows in identical order; train
columns are out-of-fold and test columns come from full-train refits; each
stage's fold plan is reproducible from the labels; each stage's inputs follow
its own schema; held-out test rows and labels reach neither stage's fitting.

Stage 2 runs with the recording router stub (see conftest).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from glass_pipeline.glass_router.pipeline.crossfit import crossfit_stage2
from shared.stage1.feature_engineering import LR_FEATURES
from shared.stage2 import RF_FEATURES_BINARY
from shared.stage_io import fold_assignment, split_fingerprint
from tests.stage1.conftest import fit_glass_stage1, stage1_split
from tests.stage2.conftest import run_crossfit
from tests.support.checks import assert_never_fit_on
from tests.support.synthetic import BANK_SCHEMA, TARGET


@pytest.fixture(scope="module")
def stages():
    split = stage1_split()
    return split, fit_glass_stage1(split, cv_folds=5), run_crossfit(split)


def test_both_stages_hold_the_same_training_rows_in_the_same_order(stages):
    split, s1, s2 = stages
    index = split["X_train"].index
    assert s1.output.train_proba_oof.index.equals(index)
    assert s2.oof.index.equals(index)
    assert s2.oof.to_frame().index.equals(s1.output.train_proba_oof.index)
    expected = split_fingerprint(index, split["X_test"].index, split["y_train"], split["y_test"])
    assert s1.output.split_fingerprint == expected


def test_each_stage_derives_its_own_features_from_the_raw_split(stages):
    split, s1, s2 = stages
    raw = [c for c in BANK_SCHEMA if c != TARGET]
    assert list(split["X_train"].columns) == raw
    assert all(f["columns"] == raw for f in s2.fe_inputs)            # Stage 2 reads raw columns
    assert list(s1.X_train.columns) == LR_FEATURES                   # Stage 1 schema
    assert all(list(r.fit_X.columns) == list(RF_FEATURES_BINARY) for r in s2.routers)
    assert not set(LR_FEATURES) & set(RF_FEATURES_BINARY)            # neither feeds the other


def test_train_columns_are_oof_and_test_columns_come_from_refits(stages):
    split, s1, s2 = stages
    stage = s1.stage
    assert not np.allclose(s1.output.train_proba_oof, stage.predict_proba(s1.X_train))
    np.testing.assert_allclose(s1.output.test_proba, stage.predict_proba(s1.X_test))
    assert s2.oof.protocol["train_outputs_oof"] is True
    assert "full-train refit" in s2.oof.protocol["test_outputs"]


def test_each_stage_fold_plan_is_reproducible_from_the_labels(stages):
    """Stage 1 cross-fits with 10 folds, Stage 2 (default) with 5; both are the
    shared StratifiedKFold plan for their fold count, so either can be rebuilt."""
    split, s1, s2 = stages
    y = split["y_train"]
    pd.testing.assert_series_equal(
        s1.output.train_fold_id,
        fold_assignment(y, cv_folds=s1.stage.cv_folds, random_state=s1.stage.random_state))
    np.testing.assert_array_equal(
        s2.oof.fold_id,
        fold_assignment(y, cv_folds=s2.oof.protocol["n_folds"],
                        random_state=s2.oof.protocol["random_state"]).to_numpy())


def test_test_rows_and_labels_reach_neither_stage(stages):
    split, s1, s2 = stages
    test_rows = split["X_test"].index
    for log in (s1.encoder_log, s1.scaler_log, s2.fe_log, s2.rf_log, s2.router_log):
        assert_never_fit_on(log, test_rows)
    scored = {k for log in (s2.fe_log, s2.rf_log, s2.router_log) for c in log for k in c.keys}
    assert not scored & set(test_rows)                               # Stage 2 OOF never sees test
    params = inspect.signature(crossfit_stage2).parameters
    assert not any("test" in name for name in params)
