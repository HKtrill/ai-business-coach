"""
FeaturePipeline (feature_research.feature_engineering): the feature DAG behind
every research stage and the cascades' engineered blocks.

Contract: learned statistics come only from the rows passed to fit_transform;
transform is row-wise (a held-out row cannot move another row's features);
names, order, dtypes and index are preserved; inputs that differ from the
fitted schema are refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_research.feature_engineering import (
    LIVE_FEATURES, finalize_features, get_stage_df,
)
from feature_research.validation import validate_engineered_features
from tests.support import mutations as mut
from tests.support.synthetic import TARGET

TARGET_DEPENDENT = ["dow_month_encoded", "euribor3m_local_rate", "neighborhood_subscription_density"]
TARGET_FREE = ["cellular_crisis", "economic_crisis_score", "euribor3m_sigmoid_slope", "campaign"]
ALL_LIVE = sorted(set(sum(LIVE_FEATURES.values(), [])))


@pytest.fixture
def fitted(bank_split, make_fp):
    fp = make_fp()
    X_tr = fp.fit_transform(bank_split["X_train"], bank_split["y_train"])
    return fp, X_tr, fp.transform(bank_split["X_test"])


# ----------------------------------------------------------------------
# output contract
# ----------------------------------------------------------------------
def test_output_keeps_index_inputs_and_adds_every_live_feature(bank_split, fitted):
    fp, X_tr, X_te = fitted
    raw_cols = list(bank_split["X_train"].columns)
    assert X_tr.index.equals(bank_split["X_train"].index)
    assert X_te.index.equals(bank_split["X_test"].index)
    assert list(X_tr.columns[:len(raw_cols)]) == raw_cols               # inputs first, in order
    assert set(ALL_LIVE) <= set(X_tr.columns)
    assert list(X_te.columns) == list(X_tr.columns) == fp.output_columns_
    assert X_te.dtypes.equals(X_tr.dtypes)
    for frame in (X_tr, X_te):
        assert np.isfinite(frame[ALL_LIVE].to_numpy(dtype=float)).all()


def test_pipeline_is_deterministic(bank_split, make_fp):
    a, b = make_fp(), make_fp()
    pd.testing.assert_frame_equal(a.fit_transform(bank_split["X_train"], bank_split["y_train"]),
                                  b.fit_transform(bank_split["X_train"], bank_split["y_train"]))
    pd.testing.assert_frame_equal(a.transform(bank_split["X_test"]), b.transform(bank_split["X_test"]))


# ----------------------------------------------------------------------
# leakage: what the fitted state may depend on
# ----------------------------------------------------------------------
def test_transform_is_row_wise(bank_split, fitted):
    """Held-out rows cannot influence each other: batch splits change nothing."""
    fp, _, X_te = fitted
    X = bank_split["X_test"]
    halves = pd.concat([fp.transform(X.iloc[:37]), fp.transform(X.iloc[37:])])
    pd.testing.assert_frame_equal(halves, X_te)
    changed = X.copy()
    changed.loc[changed.index[0], "euribor3m"] = 4.9
    out = fp.transform(changed)
    pd.testing.assert_frame_equal(out.iloc[1:], X_te.iloc[1:])


def test_learned_features_use_training_labels_and_nothing_else(bank_split, make_fp):
    """Permuting y_train moves the target-encoded features; target-free ones stay put."""
    X_tr, y_tr, X_te = bank_split["X_train"], bank_split["y_train"], bank_split["X_test"]
    base = make_fp()
    base.fit_transform(X_tr, y_tr)
    permuted = make_fp()
    permuted.fit_transform(X_tr, mut.permute_labels(y_tr, seed=3))
    a, b = base.transform(X_te), permuted.transform(X_te)
    for col in TARGET_DEPENDENT:
        assert not np.allclose(a[col], b[col]), f"{col} ignores the training labels"
    pd.testing.assert_frame_equal(a[TARGET_FREE], b[TARGET_FREE])


# ----------------------------------------------------------------------
# input contract
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["missing_column", "extra_column", "duplicated_column",
                                  "reordered_columns", "renamed_column"])
def test_transform_refuses_a_different_input_schema(bank_split, fitted, name):
    fp, _, _ = fitted
    with pytest.raises(ValueError, match="input columns differ from fit"):
        fp.transform(mut.COLUMN_MUTATIONS[name](bank_split["X_test"]))


def test_transform_refuses_changed_dtypes(bank_split, fitted):
    fp, _, _ = fitted
    with pytest.raises(TypeError, match="dtypes differ from fit"):
        fp.transform(mut.cast_column(bank_split["X_test"], "campaign", "float64"))


@pytest.mark.parametrize("name", ["shuffled_rows", "reset_index", "foreign_label"])
def test_fit_refuses_labels_not_aligned_with_rows(bank_split, make_fp, name):
    y_bad = mut.INDEX_MUTATIONS[name](bank_split["y_train"])
    with pytest.raises(ValueError, match="different indices"):
        make_fp().fit_transform(bank_split["X_train"], y_bad)


def test_fit_refuses_length_mismatch_and_transform_requires_fit(bank_split, make_fp):
    with pytest.raises(ValueError, match="len\\(X\\)"):
        make_fp().fit_transform(bank_split["X_train"], bank_split["y_train"].to_numpy()[:-1])
    with pytest.raises(ValueError, match="not fitted"):
        make_fp().transform(bank_split["X_test"])


# ----------------------------------------------------------------------
# stage registries and the pre-training validation (notebook Cells 12G, 13)
# ----------------------------------------------------------------------
def test_finalize_and_stage_frames(bank_split, fitted):
    _, X_tr, _ = fitted
    df = finalize_features(X_tr.assign(**{TARGET: bank_split["y_train"]}), LIVE_FEATURES, TARGET)
    assert set(df.columns) == set(ALL_LIVE) | {TARGET}
    assert list(df.columns) == [c for c in X_tr.columns if c in set(ALL_LIVE)] + [TARGET]
    for stage, features in LIVE_FEATURES.items():
        assert list(get_stage_df(df, stage, LIVE_FEATURES, TARGET).columns) == [*features, TARGET]
    with pytest.raises(ValueError, match="stage must be one of"):
        get_stage_df(df, "stage9", LIVE_FEATURES, TARGET)
    with pytest.raises(KeyError, match="missing from df"):
        get_stage_df(df.drop(columns=LIVE_FEATURES["lr"][0]), "lr", LIVE_FEATURES, TARGET)


def test_pre_training_validation_passes_and_catches_bad_blocks(bank_split, fitted):
    _, X_tr, _ = fitted
    df = finalize_features(X_tr.assign(**{TARGET: bank_split["y_train"]}), LIVE_FEATURES, TARGET)
    assert validate_engineered_features(df, LIVE_FEATURES, TARGET) is True
    col = LIVE_FEATURES["ebm"][0]
    for bad_value in (np.nan, np.inf):
        bad = df.copy()
        bad.loc[bad.index[0], col] = bad_value
        with pytest.raises(AssertionError):
            validate_engineered_features(bad, LIVE_FEATURES, TARGET)
