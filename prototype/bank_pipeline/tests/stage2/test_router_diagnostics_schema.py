"""
PR 37 Commit 1 (Step 1C): vocabulary and read-only containers of
shared.stage2.diagnostics.schema.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import schema
from tests.support import diagnostics_world as dw


@pytest.fixture(scope="module")
def built():
    w = dw.world()
    return (w, *dw.views(w))


def test_route_tokens_map_exactly_to_three_states():
    assert dict(d.SOURCE_DECISIONS) == {
        "pass1": (d.PASS_1_NOT_SUBSCRIBE, 0),
        "pass2": (d.PASS_2_SUBSCRIBE, 1),
        "uncertain": (d.ABSTAIN, -1),
    }
    assert d.ROUTER_STATES == (d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN)
    with pytest.raises(TypeError):
        d.SOURCE_DECISIONS["abstain"] = (d.ABSTAIN, -1)        # read-only mapping


def test_score_spaces_do_not_call_router_or_forest_scores_probabilities():
    assert set(d.SCORE_SPACES) == {"calibrated_probability", "raw_model_score",
                                   "weighted_forest_score", "router_transform"}
    assert "not calibrated" in d.SCORE_SPACES["router_transform"]
    assert "not a probability" in d.SCORE_SPACES["weighted_forest_score"]


@pytest.mark.parametrize("make", [
    lambda: d.ProvenanceItem("x", "verified"),
    lambda: d.ScoreStream("x", np.zeros(2), "probability", "f", "p"),
    lambda: d.ThresholdRecord("x", 0.5, "probability", "f", d.schema.SELECTION_FOLD_NESTED),
    lambda: d.StoredDecision("x", np.zeros(2), np.zeros(2), "raw", "f", "s"),
])
def test_unknown_status_or_space_is_refused(make):
    with pytest.raises(d.RouterDiagnosticError):
        make()


def test_errors_are_value_errors():
    assert issubclass(d.RouterDiagnosticError, ValueError)
    assert issubclass(d.UnsupportedDiagnostic, d.RouterDiagnosticError)
    assert issubclass(d.BoundaryCheckError, d.RouterDiagnosticError)


def test_views_are_frozen_and_their_arrays_read_only(built):
    _, ref, glass, rf, lr, ebm = built
    for arr in (glass.state, glass.y, glass.fold_id, glass.confidence, glass.router_score.values,
                rf.source_decision, lr.scores["calibrated_probability"].values,
                ebm.stored_decisions["ebm.nested_f2"].values, ref.y_train):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = arr[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        glass.arm = "blackbox"
    with pytest.raises(TypeError):
        rf.metadata["rf_params"] = {}
    with pytest.raises(TypeError):
        rf.fold_thresholds[0] = {"t1": 0.0, "t2": 0.0}


def test_view_frame_is_in_training_order_with_categorical_states(built):
    w, _, glass, *_ = built
    f = glass.to_frame()
    assert f.index.equals(w["idx_tr"])
    assert list(f["state"].cat.categories) == list(d.ROUTER_STATES)
    assert (f["pass1"] == (f["source_decision"] == "pass1")).all()
    assert (f["abstain"] == (f["source_pred"] == -1)).all()
    assert (f[["pass1", "pass2", "abstain"]].sum(axis=1) == 1).all()


def test_fold_base_rates_are_the_fitting_rows_positive_rate(built):
    w, _, glass, *_ = built
    np.testing.assert_array_equal(glass.fold_base_rates(), dw.fold_base_rates(w["y"], w["fold"]))


def test_train_reference_holds_no_test_labels(built):
    _, ref, *_ = built
    assert [f.name for f in dataclasses.fields(ref)] == [
        "index_train", "y_train", "router_split_fingerprint", "train_label_fingerprint",
        "stage_io_split_fingerprint", "stage_io_attestation"]
    np.testing.assert_array_equal(ref.fold_ids(5, 42), dw.world()["fold"])


def test_candidate_digest_tracks_scores_and_folds(built):
    w, ref, *_ = built
    cand = d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), ref)
    other = dw.world()
    other["rf_result"]["oof_proba"] = np.asarray(other["rf_result"]["oof_proba"]) * (1 - 1e-15)
    cand2 = d.extract_rf_pass1_candidate(dw.trapped(other["rf_result"]), ref)
    assert cand.digest != cand2.digest and len(cand.digest) == 64


def test_every_view_records_the_historical_test_exposure(built):
    _, _, glass, rf, lr, ebm = built
    for v in (glass, rf, lr, ebm):
        assert v.source["historical_test_exposure"] == d.HISTORICAL_TEST_EXPOSURE
    assert "do not make the test split untouched" in d.HISTORICAL_TEST_EXPOSURE


def test_provenance_frames_use_known_statuses(built):
    _, _, glass, rf, lr, ebm = built
    for v in (glass, rf, lr, ebm):
        frame = v.provenance_frame()
        assert set(frame["status"]) <= set(schema.PROVENANCE_STATUSES)
        assert list(frame.columns) == ["name", "status", "detail"]
    assert isinstance(glass.provenance_frame(), pd.DataFrame)
