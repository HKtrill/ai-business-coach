"""
StageOutput: the cross-arm hand-off contract (shared.stage_io).

Every way a hand-off can silently describe the wrong rows must fail loudly:
reordered / reset / foreign / duplicated indices, invalid probabilities or
labels, an unknown threshold source, overlapping splits, and a fingerprint
that no longer matches the rows and labels carried (stale or tampered files).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.stage_io import (
    STAGE_OUTPUT_VERSION, StageOutput, StageOutputError, ThresholdContaminationError,
    assert_comparable, fold_assignment, split_fingerprint,
)
from tests.support import mutations as mut

META = dict(
    stage="stage1", arm="glass", model="calibrated_lr", feature_names=["f1", "f2", "f3"],
    threshold=0.3, threshold_source="in_stage_f2_cv", calibration_method="isotonic",
    calibration_requested="auto", oof_provenance="refit inside each fold",
)


def _parts(split, seed=0):
    rng = np.random.default_rng(seed)
    y_tr, y_te = split["y_train"].astype(int), split["y_test"].astype(int)
    return dict(
        train_proba_oof=pd.Series(rng.uniform(0.01, 0.99, len(y_tr)), index=y_tr.index),
        test_proba=pd.Series(rng.uniform(0.01, 0.99, len(y_te)), index=y_te.index),
        y_train=y_tr,
        y_test=y_te,
        train_fold_id=fold_assignment(y_tr, cv_folds=5, random_state=0),
    )


def make_output(split, seed=0, **overrides):
    return StageOutput(**{**META, **_parts(split, seed), **overrides})


# ----------------------------------------------------------------------
# valid construction
# ----------------------------------------------------------------------
def test_valid_output_fills_provenance(bank_split):
    out = make_output(bank_split)
    assert out.split_fingerprint == split_fingerprint(
        out.y_train.index, out.y_test.index, out.y_train, out.y_test)
    assert out.contract_version == STAGE_OUTPUT_VERSION
    assert {"python", "pandas", "numpy", "sklearn"} <= set(out.environment)
    assert out.created_utc


# ----------------------------------------------------------------------
# index and alignment
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(mut.INDEX_MUTATIONS))
@pytest.mark.parametrize("field, message", [
    ("train_proba_oof", "proba.index != y.index|duplicate labels"),
    ("train_fold_id", "proba.index != fold_id.index"),
    ("test_proba", "proba.index != y.index|duplicate labels"),
])
def test_misaligned_index_is_rejected(bank_split, name, field, message):
    parts = _parts(bank_split)
    with pytest.raises(StageOutputError, match=message):
        StageOutput(**{**META, **parts, field: mut.INDEX_MUTATIONS[name](parts[field])})


@pytest.mark.parametrize("side", ["train", "test"])
def test_duplicated_index_is_rejected_f2(bank_split, side):
    """F2 regression: the same duplicate in every field used to be accepted."""
    parts = _parts(bank_split)
    fields = (["train_proba_oof", "y_train", "train_fold_id"] if side == "train"
              else ["test_proba", "y_test"])
    for f in fields:
        parts[f] = mut.duplicate_index_label(parts[f])
    with pytest.raises(StageOutputError, match="duplicate labels"):
        StageOutput(**META, **parts)


def test_bare_arrays_are_rejected(bank_split):
    parts = _parts(bank_split)
    with pytest.raises(StageOutputError, match="must be a pandas Series"):
        StageOutput(**{**META, **parts, "train_proba_oof": parts["train_proba_oof"].to_numpy()})


def test_overlapping_splits_are_rejected(bank_split):
    parts = _parts(bank_split)
    shared_label = parts["y_train"].index[0]
    for f in ("test_proba", "y_test"):
        parts[f] = mut.replace_index_label(parts[f], position=0, label=shared_label)
    with pytest.raises(StageOutputError, match="overlap"):
        StageOutput(**META, **parts)


# ----------------------------------------------------------------------
# values
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", mut.INVALID_PROBABILITY_KINDS)
@pytest.mark.parametrize("field", ["train_proba_oof", "test_proba"])
def test_invalid_probabilities_are_rejected(bank_split, kind, field):
    parts = _parts(bank_split)
    bad = mut.corrupt_probabilities(parts[field], kind, position=3)
    with pytest.raises(StageOutputError, match="NaN|outside"):
        StageOutput(**{**META, **parts, field: bad})


def test_non_binary_labels_are_rejected(bank_split):
    parts = _parts(bank_split)
    y = parts["y_train"].copy()
    y.iloc[0] = 2
    with pytest.raises(StageOutputError, match="not binary"):
        StageOutput(**{**META, **parts, "y_train": y})


def test_unknown_threshold_source_is_rejected(bank_split):
    with pytest.raises(StageOutputError, match="threshold_source must be one of"):
        make_output(bank_split, threshold_source="chosen_on_test")


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.2, 1.5])
def test_threshold_outside_open_unit_interval_is_rejected(bank_split, threshold):
    with pytest.raises(StageOutputError, match="threshold out of range"):
        make_output(bank_split, threshold=threshold)


# ----------------------------------------------------------------------
# provenance: fingerprints, save / load, stale files
# ----------------------------------------------------------------------
def test_supplied_fingerprint_must_match_the_rows(bank_split):
    with pytest.raises(StageOutputError, match="does not match"):
        make_output(bank_split, split_fingerprint="0" * 12)


def test_save_load_round_trip(bank_split, tmp_path):
    out = make_output(bank_split)
    loaded = StageOutput.load(out.save(tmp_path))
    for f in ("train_proba_oof", "test_proba", "y_train", "y_test", "train_fold_id"):
        pd.testing.assert_series_equal(getattr(loaded, f), getattr(out, f))
    assert loaded.summary() == out.summary()


def test_tampered_fingerprint_fails_on_load(bank_split):
    d = make_output(bank_split).to_dict()
    with pytest.raises(StageOutputError, match="does not match"):
        StageOutput.from_dict(mut.with_field(d, "split_fingerprint", "f" * 12))


def test_relabelled_rows_fail_on_load(bank_split):
    """A file whose labels changed after saving no longer matches its fingerprint."""
    d = make_output(bank_split).to_dict()
    y = d["y_train"].copy()
    y.iloc[0] = 1 - y.iloc[0]
    with pytest.raises(StageOutputError, match="does not match"):
        StageOutput.from_dict(mut.with_field(d, "y_train", y))


def test_from_dict_requires_the_contract_marker(bank_split):
    d = make_output(bank_split).to_dict()
    d.pop("__stage_output__")
    with pytest.raises(StageOutputError, match="no '__stage_output__' marker"):
        StageOutput.from_dict(d)
    with pytest.raises(StageOutputError, match="saved WITHOUT a StageOutput"):
        StageOutput.from_dict({"stage_output": None, "model": object()})


def test_from_dict_unwraps_model_artifacts(bank_split):
    out = make_output(bank_split)
    wrapped = {"model": "fitted model", "stage_output": out.to_dict()}
    assert StageOutput.from_dict(wrapped).split_fingerprint == out.split_fingerprint


def test_version_mismatch_warns_and_unknown_keys_are_ignored(bank_split, capsys):
    """Documented leniency: newer files with extra fields still load."""
    d = mut.with_field(make_output(bank_split).to_dict(), "__stage_output__", "0.9")
    d["field_from_the_future"] = 1
    out = StageOutput.from_dict(d)
    assert "contract version 0.9" in capsys.readouterr().out
    assert not hasattr(out, "field_from_the_future")


# ----------------------------------------------------------------------
# decisions
# ----------------------------------------------------------------------
def test_decisions_reproduce_from_probabilities_and_threshold(bank_split):
    out = make_output(bank_split, threshold_source="nested_per_fold")
    out.test_proba.iloc[0] = out.threshold                      # boundary: p == t is positive
    expected = (out.test_proba >= out.threshold).astype("int8")
    pd.testing.assert_series_equal(out.test_decisions(), expected, check_names=False)
    assert out.test_decisions().iloc[0] == 1
    pd.testing.assert_series_equal(out.train_decisions_oof(),
                                   (out.train_proba_oof >= out.threshold).astype("int8"),
                                   check_names=False)


def test_contaminated_threshold_blocks_train_decisions(bank_split):
    out = make_output(bank_split)                                # in_stage_f2_cv
    assert not out.threshold_is_cross_fitted
    with pytest.raises(ThresholdContaminationError):
        out.train_decisions_oof()
    assert len(out.train_decisions_oof(allow_contaminated=True)) == len(out.y_train)


# ----------------------------------------------------------------------
# assert_comparable: paired cross-arm comparison
# ----------------------------------------------------------------------
def test_comparable_arms_pass(bank_split):
    assert_comparable(make_output(bank_split, seed=0),
                      make_output(bank_split, seed=1, arm="blackbox", model="mlp"))


@pytest.mark.parametrize("change, message", [
    (dict(stage="stage3"), "different stages"),
    (dict(feature_names=["f1", "f2", "other"]), "feature sets differ"),
    (dict(train_fold_id="reassigned"), "fold partitions differ"),
    (dict(y_train="relabelled"), "different splits"),
], ids=["stage", "features", "folds", "labels"])
def test_incomparable_arms_are_rejected(bank_split, change, message):
    a = make_output(bank_split)
    parts = _parts(bank_split, seed=1)
    if change.get("train_fold_id") == "reassigned":
        change = dict(train_fold_id=fold_assignment(parts["y_train"], cv_folds=5, random_state=99))
    if change.get("y_train") == "relabelled":
        y = parts["y_train"].copy()
        y.iloc[0] = 1 - y.iloc[0]
        change = dict(y_train=y)
    b = StageOutput(**{**META, **parts, "arm": "blackbox", **change})
    with pytest.raises(StageOutputError, match=message):
        assert_comparable(a, b)


def test_differing_threshold_source_warns_only(bank_split, capsys):
    assert_comparable(make_output(bank_split),
                      make_output(bank_split, seed=1, arm="blackbox",
                                  threshold_source="nested_per_fold"))
    assert "threshold_source differs" in capsys.readouterr().out
