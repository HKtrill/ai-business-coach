"""
GLASS Stage 3 feature block: ``EBMFeaturePipeline``
(shared.stage3.feature_engineering) and the EBM interaction pairs
(glass_pipeline.ebm.interactions).

Contract: the pipeline emits exactly ``EBM_FEATURES``, in that order, with
the same dtypes on every frame; it is deterministic; only the rows and labels
given to ``fit_transform`` shape what it learns, and ``transform`` is
row-wise; the leaky raw columns never reach it; it reproduces the legacy
``engineer_ebm_features`` + ``select_ebm_features`` path; unseen values give
finite features and non-finite raw input is refused. The interaction pairs are
the validated ones, depend on column names only, and only reference columns
that are present.

Per-fold fitting inside ``Stage3Block`` is checked generically in
tests/preprocessing/test_feature_contracts.py and with this DAG in
test_glass_ebm_stage.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glass_pipeline.ebm.interactions import define_ebm_interactions
from shared.stage3.feature_engineering import (
    EBM_FEATURES, EBM_FEATURES_12, EBM_FEATURES_13, LEAKY_FEATURES, EBMFeaturePipeline,
    engineer_ebm_features, select_ebm_features,
)
from tests.support import mutations as mut

# Fitted on (X, y) by the Integral / Derivative / Temporal engineers; every
# other EBM feature is a fixed function of the row's own raw values.
TARGET_DEPENDENT = {"decay_x_density", "economic_curvature_intensity", "dow_month_encoded"}
VALIDATED_PAIRS = [
    ("campaign", "emp_var_rate_sigmoid_slope"),                 # tier 1
    ("euribor3m_sigmoid_slope", "economic_curvature_intensity"),  # tier 2
    ("prior_x_stress", "decay_x_density"),                      # tier 2
]


def _close(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Equal frames up to floating-point summation order (~1e-17 here)."""
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# output schema
# ----------------------------------------------------------------------
def test_feature_registries():
    assert len(EBM_FEATURES) == 13 and len(set(EBM_FEATURES)) == 13
    assert EBM_FEATURES_13 == EBM_FEATURES                       # cut 1 is already out
    assert EBM_FEATURES_12 == [f for f in EBM_FEATURES if f != "emp_var_rate_sigmoid_slope"]
    assert not set(LEAKY_FEATURES) & set(EBM_FEATURES)


def test_output_is_exactly_the_ebm_feature_block(ebm_features):
    split, _, X_tr, X_te = ebm_features
    for X, raw in ((X_tr, split["X_train"]), (X_te, split["X_test"])):
        assert list(X.columns) == list(EBM_FEATURES)
        assert X.index.equals(raw.index)
        assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
        assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert X_te.dtypes.equals(X_tr.dtypes)


def test_pipeline_is_deterministic(ebm_features):
    split, _, X_tr, X_te = ebm_features
    again = EBMFeaturePipeline()
    pd.testing.assert_frame_equal(again.fit_transform(split["X_train"], split["y_train"]), X_tr)
    pd.testing.assert_frame_equal(again.transform(split["X_test"]), X_te)


def test_fit_transform_equals_transform_on_the_fit_rows(ebm_features):
    """No train-only path (e.g. in-fold smoothing): the rows a model trains on
    are engineered exactly as they would be scored."""
    split, pipe, X_tr, _ = ebm_features
    _close(pipe.transform(split["X_train"]), X_tr)


# ----------------------------------------------------------------------
# fit / transform boundary and labels
# ----------------------------------------------------------------------
def test_only_the_fit_labels_shape_the_learned_features(ebm_features):
    """Moving the fit labels moves exactly the target-dependent features, on
    the fit rows and on rows scored later; nothing else reads a label."""
    split, _, X_tr, X_te = ebm_features
    moved = EBMFeaturePipeline()
    M_tr = moved.fit_transform(split["X_train"], mut.permute_labels(split["y_train"], seed=1))
    M_te = moved.transform(split["X_test"])
    for M, X in ((M_tr, X_tr), (M_te, X_te)):
        changed = {c for c in EBM_FEATURES if not np.allclose(M[c], X[c])}
        assert changed == TARGET_DEPENDENT                              # the check can fail
        pd.testing.assert_frame_equal(M.drop(columns=list(changed)), X.drop(columns=list(changed)))


def test_transform_is_row_wise(ebm_features):
    """A scored row's features do not depend on which other rows are scored
    with it (validation folds, the test split, one serving row)."""
    split, pipe, _, X_te = ebm_features
    raw = split["X_test"]
    subset = raw.iloc[::3]
    _close(pipe.transform(subset), X_te.loc[subset.index])
    _close(pipe.transform(raw.iloc[[5]]), X_te.iloc[[5]])
    shuffled = mut.shuffle_rows(raw)
    _close(pipe.transform(shuffled), X_te.loc[shuffled.index])


def test_raw_column_order_does_not_matter(ebm_features):
    split, pipe, _, X_te = ebm_features
    pd.testing.assert_frame_equal(pipe.transform(split["X_test"].iloc[:, ::-1]), X_te)


def test_leaky_raw_columns_are_dropped_before_engineering(ebm_features):
    """``duration`` / ``pdays`` / ``poutcome`` never reach the DAG, even when
    they carry the label outright."""
    split, _, X_tr, X_te = ebm_features
    y = split["y_train"]

    def leaky(X, label):
        return X.assign(duration=1000 * label.astype(int), pdays=999, poutcome=label.astype(int))

    pipe = EBMFeaturePipeline()
    pd.testing.assert_frame_equal(pipe.fit_transform(leaky(split["X_train"], y), y), X_tr)
    y_te = split["y_test"]
    pd.testing.assert_frame_equal(pipe.transform(leaky(split["X_test"], y_te)), X_te)


def test_inputs_are_not_modified(ebm_features):
    split = ebm_features[0]
    X, y = split["X_train"].copy(), split["y_train"].copy()
    pipe = EBMFeaturePipeline()
    pipe.fit_transform(X, y)
    pipe.transform(X)
    pd.testing.assert_frame_equal(X, split["X_train"])
    pd.testing.assert_series_equal(y, split["y_train"])


def test_matches_the_legacy_engineering_path(ebm_features):
    """``engineer_ebm_features`` + ``select_ebm_features`` (research and the
    pre-PR-33 notebooks) is the same DAG as the pipeline."""
    split, _, X_tr, X_te = ebm_features
    L_tr, L_te, names = engineer_ebm_features(split["X_train"], split["X_test"], split["y_train"])
    L_tr, L_te = select_ebm_features(L_tr, L_te)
    assert names == list(EBM_FEATURES)
    pd.testing.assert_frame_equal(L_tr, X_tr)
    pd.testing.assert_frame_equal(L_te, X_te)


# ----------------------------------------------------------------------
# unseen and invalid inputs
# ----------------------------------------------------------------------
def test_unseen_values_give_finite_features(ebm_features):
    """Months, weekday-month cells and economic levels absent from the fit
    rows still produce finite features (no NaN for the cleaner to hide)."""
    split, pipe, _, _ = ebm_features
    raw = split["X_test"].assign(**{
        "month": 1, "day_of_week": 4, "campaign": 60, "previous": 7,
        "euribor3m": 9.0, "emp.var.rate": 5.0, "cons.price.idx": 99.0, "nr.employed": 6000.0,
    })
    assert 1 not in set(split["X_train"]["month"])
    out = pipe.transform(raw)
    assert list(out.columns) == list(EBM_FEATURES)
    assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_non_finite_raw_economics_are_refused(ebm_features):
    split, pipe, _, _ = ebm_features
    raw = split["X_test"].copy()
    raw.iloc[0, raw.columns.get_loc("euribor3m")] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        pipe.transform(raw)


@pytest.mark.parametrize("case, error, message", [
    ("no_labels", ValueError, "needs y"),
    ("misaligned_labels", ValueError, "y index must match X index"),
    ("not_fitted", ValueError, "call fit_transform"),
    ("missing_raw_column", KeyError, "euribor3m"),
    ("feature_not_produced", KeyError, "DAG did not produce"),
])
def test_invalid_use_is_refused(ebm_features, case, error, message):
    split, pipe, _, _ = ebm_features
    X, y = split["X_train"], split["y_train"]
    with pytest.raises(error, match=message):
        if case == "no_labels":
            EBMFeaturePipeline().fit_transform(X, None)
        elif case == "misaligned_labels":
            EBMFeaturePipeline().fit_transform(X, mut.shuffle_rows(y))
        elif case == "not_fitted":
            EBMFeaturePipeline().transform(X)
        elif case == "missing_raw_column":
            pipe.transform(X.drop(columns=["euribor3m"]))
        else:
            EBMFeaturePipeline(features=[*EBM_FEATURES, "not_a_feature"]).fit_transform(X, y)


# ----------------------------------------------------------------------
# interaction pairs
# ----------------------------------------------------------------------
def test_production_block_gets_the_three_validated_pairs(ebm_features):
    pairs = define_ebm_interactions(ebm_features[2])
    assert pairs == VALIDATED_PAIRS
    assert all(a != b and {a, b} <= set(EBM_FEATURES) for a, b in pairs)
    assert len({frozenset(p) for p in pairs}) == len(pairs)


@pytest.mark.parametrize("member", sorted({m for p in VALIDATED_PAIRS for m in p}))
def test_a_pair_is_dropped_when_a_member_is_pruned(member):
    pairs = define_ebm_interactions(pd.DataFrame(columns=[f for f in EBM_FEATURES if f != member]))
    assert pairs == [p for p in VALIDATED_PAIRS if member not in p]


def test_pruning_cut_2_collapses_the_tier_1_pair():
    assert define_ebm_interactions(pd.DataFrame(columns=EBM_FEATURES_12)) == VALIDATED_PAIRS[1:]


def test_pairs_depend_on_column_names_only(ebm_features):
    """No values or labels are read, so the pairs cannot leak a fold's labels."""
    X_tr = ebm_features[2]
    assert define_ebm_interactions(pd.DataFrame(columns=EBM_FEATURES)) == VALIDATED_PAIRS
    assert define_ebm_interactions(X_tr.iloc[:, ::-1]) == VALIDATED_PAIRS
    assert define_ebm_interactions(X_tr.sample(frac=1.0, random_state=0) * -1) == VALIDATED_PAIRS
