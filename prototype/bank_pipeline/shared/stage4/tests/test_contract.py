"""shared.stage4 — adapter + contract tests (arm-agnostic)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from shared.stage4 import (
    ABSTENTION_GRID, ABSTENTION_MIN_COVERAGE, Stage4InputError, build_stage4_inputs,
    router_stream, stage3_stream, stage_output_stream, tune_min_confidence,
)
from shared.stage4.tests.fixtures import (
    ROUTER_FP, build_world, make_router, make_stage1, make_stage3,
)
from shared.validation import ValidationError


@pytest.fixture
def w():
    return build_world()


def streams(w, s1=None, s2=None, s3=None):
    y = w["GS"]["y_train"].to_numpy()
    a, c1 = stage_output_stream(s1 or make_stage1(w, arm="blackbox", model="mlp"),
                                "mlp", "MLP", len(y))
    b, c2 = router_stream(s2 or make_router(w, layout="extra"), "rf_router", "RF Router", y)
    c, c3 = stage3_stream(s3 or make_stage3(w, family="xgboost"), "xgb", "XGBoost")
    return [a, b, c], c1 + c2 + c3


def test_blackbox_formats_load(w):
    ss, cc = streams(w)
    inp = build_stage4_inputs(w["GS"], ss, cc, verbose=False, extra_fingerprints=ROUTER_FP)
    rf = inp["rf_router"]
    assert rf.train_source == "extra.train_proba" and rf.notes["artifact_layout"] == "extra"
    assert inp["xgb"].fingerprint_scheme == "router" and inp["mlp"].fingerprint_scheme == "stage_io"
    assert ["rf_router", "xgb", 5] in inp.fold_provenance["shared_partitions"]


def test_two_column_router_proba_takes_positive_class(w):
    ss, _ = streams(w, s2=make_router(w, layout="extra", two_col=True))
    assert ss[1].train_proba.ndim == 1


def test_router_scheme_required(w):
    ss, cc = streams(w)
    with pytest.raises(ValidationError, match="no scheme matches"):
        build_stage4_inputs(w["GS"], ss, cc, verbose=False)          # router scheme absent


def test_router_pred_decision_inconsistency_fails(w):
    r = make_router(w, layout="extra")
    r["extra"]["train_pred"] = np.zeros_like(r["extra"]["train_pred"])
    ss, cc = streams(w, s2=r)
    with pytest.raises(ValidationError, match="train_pred agrees"):
        build_stage4_inputs(w["GS"], ss, cc, verbose=False, extra_fingerprints=ROUTER_FP)


def test_router_orientation_checked(w):
    r = make_router(w, layout="extra")
    r["extra"]["train_proba"] = 1 - r["extra"]["train_proba"]           # P(not subscribe)
    ss, cc = streams(w, s2=r)
    with pytest.raises(ValidationError, match="P\\(subscribe\\)"):
        build_stage4_inputs(w["GS"], ss, cc, verbose=False, extra_fingerprints=ROUTER_FP)


def test_fold_fingerprint_checked_when_function_given(w):
    r = make_router(w, layout="extra")
    r["extra"]["oof_protocol"]["fold_fingerprint"] = "abc"
    y = w["GS"]["y_train"].to_numpy()
    _, cc = router_stream(r, "rf_router", "RF", y, fold_fingerprint_fn=lambda f: "xyz")
    assert not [c for c in cc if "fold fingerprint" in c[2]][0][3]()


def test_prefit_stage1_refused(w):
    s1 = make_stage1(w)
    s1["stage_output"]["oof_provenance"] = "PREFIT — in-sample"
    ss, cc = streams(w, s1=s1)
    with pytest.raises(ValidationError, match="refittable"):
        build_stage4_inputs(w["GS"], ss, cc, verbose=False, extra_fingerprints=ROUTER_FP)


def test_legacy_stage3_dict_refused(w):
    with pytest.raises(Stage4InputError, match="pre-PR-33"):
        stage3_stream({"train_predictions": [0.1]}, "xgb", "XGB")


def test_abstention_protocol_mirrors_meta_ebm():
    from glass_pipeline.meta_ebm import meta_stage, tuning
    sig = inspect.signature(tuning.tune_arbiter_threshold)
    assert sig.parameters["min_coverage"].default == ABSTENTION_MIN_COVERAGE
    assert meta_stage._MIN_COVERAGE == ABSTENTION_MIN_COVERAGE
    assert np.allclose(np.arange(0.03, 0.55, 0.02), ABSTENTION_GRID)   # tuning.py default


def test_tune_min_confidence_rule():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < .15).astype(int)
    p = np.clip(y * .3 + rng.random(2000) * .7, 0, 1)
    d = (p >= .4).astype(int)
    best = tune_min_confidence(y, d, np.abs(p - .4))
    assert best["train_coverage"] >= ABSTENTION_MIN_COVERAGE
    assert tune_min_confidence(y, d, np.zeros(2000)) is None           # nothing ≥ 0.03
