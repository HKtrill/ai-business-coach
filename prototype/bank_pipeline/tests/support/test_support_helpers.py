"""
Self-tests for tests/support: builders, mutations, recorders and checks.

Each checker is shown passing on an honest pattern AND failing on a
deliberately leaky one, so a green contract test elsewhere means something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tests.support import mutations as mut
from tests.support.checks import ProvenanceError, assert_never_fit_on, assert_oof_provenance
from tests.support.recorders import CallLog, RecordingEstimator, array_row_labels, row_keys
from tests.support.synthetic import BANK_SCHEMA, TARGET, make_bank_frame, split_frame

FOLDS = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)


def _clf():
    """Small classifier that converges on the raw bank-like features."""
    return make_pipeline(StandardScaler(), LogisticRegression())


def _manual_oof(X, y, log, *, leak=False, skip_fold=None, score_twice=False):
    """Hand-written K-fold scoring with optional, deliberate mistakes."""
    for k, (tr, va) in enumerate(FOLDS.split(X, y)):
        if k == skip_fold:
            continue
        fit_rows = np.concatenate([tr, va[:1]]) if leak else tr     # leak: one held-out row
        model = RecordingEstimator(_clf(), log)
        model.fit(X.iloc[fit_rows], y.iloc[fit_rows])
        model.predict_proba(X.iloc[va])
        if score_twice:
            model.predict_proba(X.iloc[va[:1]])


# ----------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------
def test_frame_is_deterministic_per_seed():
    pd.testing.assert_frame_equal(make_bank_frame(seed=3), make_bank_frame(seed=3))
    assert not make_bank_frame(seed=3).equals(make_bank_frame(seed=4))


def test_frame_matches_the_preprocessed_schema(bank_frame):
    assert list(bank_frame.columns) == [*BANK_SCHEMA, TARGET]
    assert bank_frame.drop(columns=TARGET).dtypes.astype(str).to_dict() == BANK_SCHEMA
    assert bank_frame[TARGET].dtype == "int8"
    assert set(bank_frame[TARGET].unique()) == {0, 1}
    assert int(bank_frame[TARGET].sum()) == 80                     # exact prevalence


def test_index_is_irregular(bank_frame):
    idx = bank_frame.index
    assert idx.is_unique and idx.dtype == "int64"
    assert not idx.is_monotonic_increasing and not idx.is_monotonic_decreasing
    assert not set(idx) & set(range(len(idx)))                   # labels never equal positions


def test_signal_is_learnable_and_null_signal_is_not(bank_frame, null_signal_frame):
    def auc(frame):
        X, y = frame.drop(columns=TARGET), frame[TARGET]
        p = cross_val_predict(LogisticRegression(max_iter=1000), StandardScaler().fit_transform(X),
                              y, cv=FOLDS, method="predict_proba")[:, 1]
        return roc_auc_score(y, p)

    assert auc(bank_frame) > 0.75
    assert 0.35 < auc(null_signal_frame) < 0.65


def test_split_is_disjoint_exhaustive_and_stratified(bank_frame, bank_split):
    s = bank_split
    assert not set(s["X_train"].index) & set(s["X_test"].index)
    assert set(s["X_train"].index) | set(s["X_test"].index) == set(bank_frame.index)
    assert s["X_train"].index.equals(s["y_train"].index)
    assert s["X_test"].index.equals(s["y_test"].index)
    assert abs(s["y_train"].mean() - s["y_test"].mean()) < 0.02
    assert TARGET not in s["X_train"].columns
    pd.testing.assert_frame_equal(split_frame(bank_frame)["X_test"], s["X_test"])


# ----------------------------------------------------------------------
# mutations
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(mut.INDEX_MUTATIONS))
def test_index_mutations_change_the_index_and_not_the_input(bank_frame, name):
    before = bank_frame.copy()
    out = mut.INDEX_MUTATIONS[name](bank_frame)
    pd.testing.assert_frame_equal(bank_frame, before)            # input untouched
    assert len(out) == len(bank_frame)
    assert not out.index.equals(bank_frame.index)
    expected = {
        "shuffled_rows": lambda o: set(o.index) == set(bank_frame.index),
        "reset_index": lambda o: isinstance(o.index, pd.RangeIndex),
        "duplicated_label": lambda o: not o.index.is_unique,
        "foreign_label": lambda o: bool(set(o.index) - set(bank_frame.index)),
    }[name]
    assert expected(out)


def test_index_mutations_work_on_series(bank_frame):
    y = bank_frame[TARGET]
    for fn in mut.INDEX_MUTATIONS.values():
        assert isinstance(fn(y), pd.Series)


@pytest.mark.parametrize("name", sorted(mut.COLUMN_MUTATIONS))
def test_column_mutations_break_the_schema_and_not_the_input(bank_frame, name):
    X = bank_frame.drop(columns=TARGET)
    before = X.copy()
    out = mut.COLUMN_MUTATIONS[name](X)
    pd.testing.assert_frame_equal(X, before)
    cols, new = list(X.columns), list(out.columns)
    expected = {
        "missing_column": len(new) == len(cols) - 1,
        "extra_column": len(new) == len(cols) + 1,
        "duplicated_column": len(new) != len(set(new)),
        "reordered_columns": set(new) == set(cols) and new != cols,
        "renamed_column": set(new) != set(cols) and len(new) == len(cols),
        "changed_dtype": new == cols and not out.dtypes.equals(X.dtypes),
    }[name]
    assert expected


@pytest.mark.parametrize("kind", mut.INVALID_PROBABILITY_KINDS)
@pytest.mark.parametrize("as_series", [True, False])
def test_corrupt_probabilities(kind, as_series):
    p = np.linspace(0.1, 0.9, 5)
    p = pd.Series(p, index=[7, 3, 9, 1, 5]) if as_series else p
    out = mut.corrupt_probabilities(p, kind, position=2)
    vals, orig = np.asarray(out, dtype=float), np.asarray(p, dtype=float)
    bad = vals[2]
    assert {"nan": np.isnan(bad), "inf": np.isinf(bad),
            "above_one": bad > 1, "below_zero": bad < 0}[kind]
    np.testing.assert_array_equal(np.delete(vals, 2), np.delete(orig, 2))
    assert np.isfinite(orig).all() and ((orig >= 0) & (orig <= 1)).all()   # input untouched
    if as_series:
        assert out.index.equals(p.index)


def test_corrupt_probabilities_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        mut.corrupt_probabilities(np.array([0.5]), "huge")


def test_permute_labels_moves_only_the_chosen_rows(bank_split):
    y = bank_split["y_test"]
    rows = y.index[:40]
    out = mut.permute_labels(y, seed=1, rows=rows)
    assert out.index.equals(y.index)
    assert out.loc[rows].sum() == y.loc[rows].sum()              # class counts kept
    assert not out.loc[rows].equals(y.loc[rows])                 # something moved
    assert out.drop(rows).equals(y.drop(rows))                   # everything else fixed


def test_with_field_copies_along_the_path():
    original = {"a": 1, "extra": {"protocol": {"fold_fp": "abc"}, "other": 2}}
    out = mut.with_field(original, "extra.protocol.fold_fp", "xyz")
    assert out["extra"]["protocol"]["fold_fp"] == "xyz"
    assert original["extra"]["protocol"]["fold_fp"] == "abc"
    assert out["extra"]["other"] == 2
    with pytest.raises(KeyError):
        mut.with_field(original, "extra.missing", 0)


# ----------------------------------------------------------------------
# recorders
# ----------------------------------------------------------------------
def test_recorder_is_transparent_inside_cross_val_predict(bank_split):
    X, y = bank_split["X_train"], bank_split["y_train"]
    log = CallLog()
    wrapped = RecordingEstimator(_clf(), log)
    assert is_classifier(wrapped)                                # stratification preserved
    plain = cross_val_predict(_clf(), X, y, cv=4, method="predict_proba")
    recorded = cross_val_predict(wrapped, X, y, cv=4, method="predict_proba")
    np.testing.assert_array_equal(plain, recorded)
    assert len(log.fits()) == 4                                  # every clone wrote to one log


def test_row_keys_for_frames_and_arrays(bank_split):
    X = bank_split["X_train"]
    assert row_keys(X) == tuple(X.index)
    arr = X.to_numpy(dtype=float)
    lookup = array_row_labels(arr, X.index)
    assert [lookup[k] for k in row_keys(arr[5:9])] == list(X.index[5:9])
    with pytest.raises(ValueError, match="duplicate rows"):
        array_row_labels(np.vstack([arr[:3], arr[:1]]), range(4))
    with pytest.raises(TypeError):
        row_keys(np.zeros(3))


def test_recorder_requires_a_log(bank_split):
    with pytest.raises(ValueError, match="CallLog"):
        RecordingEstimator(LogisticRegression()).fit(bank_split["X_train"], bank_split["y_train"])


# ----------------------------------------------------------------------
# checks: honest patterns pass, leaky ones fail
# ----------------------------------------------------------------------
def test_honest_manual_oof_passes(bank_split, call_log):
    X, y = bank_split["X_train"], bank_split["y_train"]
    _manual_oof(X, y, call_log)
    assert_oof_provenance(call_log, X.index)


def test_cross_val_predict_on_arrays_passes(bank_split, call_log):
    """The Stage 1 pattern: a scaled array through cross_val_predict."""
    X, y = bank_split["X_train"], bank_split["y_train"]
    arr = StandardScaler().fit_transform(X)
    cross_val_predict(RecordingEstimator(_clf(), call_log),
                      arr, y, cv=FOLDS, method="predict_proba")
    assert_oof_provenance(call_log, row_keys(arr))


def test_refit_model_is_ignored_by_oof_check(bank_split, call_log):
    X, y = bank_split["X_train"], bank_split["y_train"]
    _manual_oof(X, y, call_log)
    refit = RecordingEstimator(_clf(), call_log).fit(X, y)
    refit.predict_proba(X)                                       # in-sample diagnostic
    assert_oof_provenance(call_log, X.index)


@pytest.mark.parametrize("mistake, message", [
    (dict(leak=True), "scored 1 row"),
    (dict(skip_fold=2), "never scored"),
    (dict(score_twice=True), "more than once"),
])
def test_leaky_oof_patterns_fail(bank_split, call_log, mistake, message):
    X, y = bank_split["X_train"], bank_split["y_train"]
    _manual_oof(X, y, call_log, **mistake)
    with pytest.raises(ProvenanceError, match=message):
        assert_oof_provenance(call_log, X.index)


def test_in_sample_scores_presented_as_oof_fail(bank_split, call_log):
    X, y = bank_split["X_train"], bank_split["y_train"]
    model = RecordingEstimator(_clf(), call_log).fit(X, y)
    model.predict_proba(X)
    with pytest.raises(ProvenanceError, match="at least 2"):
        assert_oof_provenance(call_log, X.index)


def test_never_fit_on_catches_test_rows_in_a_transformer_fit(bank_split, call_log):
    X_tr, X_te = bank_split["X_train"], bank_split["X_test"]
    RecordingEstimator(StandardScaler(), call_log).fit(X_tr)
    assert_never_fit_on(call_log, X_te.index)                    # honest: passes

    RecordingEstimator(StandardScaler(), call_log).fit(pd.concat([X_tr, X_te]))
    with pytest.raises(ProvenanceError, match="forbidden rows"):
        assert_never_fit_on(call_log, X_te.index)


def test_per_fold_feature_fitting_check(bank_split):
    X, y = bank_split["X_train"], bank_split["y_train"]

    honest = CallLog()
    for tr, va in FOLDS.split(X, y):
        scaler = RecordingEstimator(StandardScaler(), honest)
        scaler.fit_transform(X.iloc[tr])
        scaler.transform(X.iloc[va])
    assert_oof_provenance(honest, X.index, methods=("transform",))

    leaky = CallLog()                                            # fitted once on all rows
    scaler = RecordingEstimator(StandardScaler(), leaky).fit(X)
    for _, va in FOLDS.split(X, y):
        scaler.transform(X.iloc[va])
    with pytest.raises(ProvenanceError, match="at least 2"):
        assert_oof_provenance(leaky, X.index, methods=("transform",))
