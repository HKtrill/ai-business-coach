"""
Black-box Stage 1-3 -> Stage 4 on artifacts the current producers write: the
MLP ``StageOutput`` (``CalibratedStage1MLP``), the RF Router two-pass artifact
with its Stage 4 block (tests.support.rf_router_world, notebook Cell 24) and
the XGBoost ``Stage3Artifact`` (``train_xgb_stage`` on the per-fold GLASS
feature DAG), all on one synthetic bank split and saved to disk, then
``train_meta_xgb_stage`` with the real fingerprint schemes (no stand-ins).

Contract: Meta-XGB accepts the three real artifacts and every input, feature
and meta-level check passes; each stream is matched by its own fingerprint
scheme (Stage 1 ``stage_io``, Stages 2-3 router); the feature block holds each
artifact's OOF column (train) and full-train column (test), never an
in-sample one; the router's, XGBoost's and Meta-XGB's 5-fold partitions are
one partition and the MLP's 3-fold partition is recorded as distinct; the
saved Meta-XGB artifact, with the saved upstream files, reproduces its test
probabilities and decisions.

Small settings throughout (MLP ``MLP_TEST_CONFIG``, three reporting folds,
fixed small hyperparameters); the contracts are those of
test_meta_xgb_stage.py, here on producer-written artifacts.
"""

from __future__ import annotations

import dataclasses

import joblib
import numpy as np
import pandas as pd
import pytest

from blackbox_pipeline.models.meta_xgb import (
    STAGE4_FEATURES, build_feature_frames, load_blackbox_stage4_inputs, load_meta_xgb,
    train_meta_xgb_stage,
)
from blackbox_pipeline.models.mlp import CalibratedStage1MLP
from blackbox_pipeline.models.rf_router import split_fingerprint as router_split_fingerprint
from blackbox_pipeline.models.xgb import XGBStage3Config, train_xgb_stage
from shared.stage1.feature_engineering import engineer_features as stage1_features
from shared.stage3.data import Stage3Block
from shared.stage3.feature_engineering import EBM_FEATURES, EBMFeaturePipeline
from shared.stage_io import fold_assignment, split_fingerprint
from tests.stage1.conftest import MLP_TEST_CONFIG
from tests.stage3.conftest import ARMS
from tests.support.meta_xgb_world import META_PARAMS, N_META_EVAL, meta_config
from tests.support.rf_router_world import N_STAGE_FOLDS, RANDOM_STATE, bank_world, quiet, save_world

pytestmark = pytest.mark.filterwarnings(        # the MLP's tiny synthetic folds (see Stage 1 suite)
    "ignore:Got `batch_size` less than 1 or larger than sample size:UserWarning")


@dataclasses.dataclass
class Handoff:
    split: dict
    paths: dict
    mlp: object
    rf: dict
    xgb: object
    meta: object
    meta_path: str


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    out = tmp_path_factory.mktemp("blackbox")
    world = bank_world()
    gs = world.global_split
    rf_path = str(save_world(world, out / "rf_router"))
    X_tr, X_te, _ = quiet(stage1_features, gs["X_train"], gs["X_test"], gs["y_train"])
    mlp = CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(X_tr, gs["y_train"], X_train_raw=gs["X_train"])
    output = mlp.to_stage_output(X_te, gs["y_test"])
    mlp_path = str(quiet(output.save, str(out / "mlp")))
    cfg = XGBStage3Config(n_eval_folds=N_META_EVAL, n_jobs=1, verbose=False, show_progress_bar=False,
                          artifact_dir=str(out / "xgb"), expected_features=list(EBM_FEATURES))
    fp = router_split_fingerprint(gs["X_train"], gs["X_test"])          # Cell 30's _FP_NOW
    block = Stage3Block.from_global_split(gs, EBMFeaturePipeline, cfg, split_fingerprint=fp)
    xgb, xgb_path = train_xgb_stage(block, config=cfg, params=dict(ARMS["xgboost"].params),
                                    params_source="test")
    meta, meta_path = train_meta_xgb_stage(
        gs, mlp_path=mlp_path, rf_path=rf_path, xgb_path=xgb_path,
        config=meta_config(out / "meta"), params=dict(META_PARAMS), params_source="test")
    return Handoff(gs, {"mlp": mlp_path, "rf": rf_path, "xgb": xgb_path}, output,
                   joblib.load(rf_path), xgb, meta, meta_path)


def test_meta_xgb_takes_the_real_artifacts_under_their_own_fingerprint_schemes(handoff):
    art, gs = handoff.meta, handoff.split
    assert {k: all(r["passed"] for r in v) for k, v in art.validation.items()} == {
        "inputs": True, "features": True, "meta": True}
    streams = art.stage4_inputs["streams"]
    assert {n: s["fingerprint_scheme"] for n, s in streams.items()} == {
        "mlp": "stage_io", "rf_router": "router", "xgb": "router"}
    assert art.split_fingerprints == {
        "stage_io": split_fingerprint(gs["X_train"].index, gs["X_test"].index,
                                      gs["y_train"].astype(int), gs["y_test"].astype(int)),
        "router": router_split_fingerprint(gs["X_train"], gs["X_test"])}
    assert {n: s["artifact_path"] for n, s in streams.items()} == {
        "mlp": handoff.paths["mlp"], "rf_router": handoff.paths["rf"], "xgb": handoff.paths["xgb"]}
    assert (streams["mlp"]["arm"], streams["xgb"]["model_family"], streams["xgb"]["arm"]) == (
        "blackbox", "xgboost", "blackbox")


def test_the_feature_block_holds_each_artifacts_oof_and_full_train_columns(handoff):
    inputs = load_blackbox_stage4_inputs(handoff.split, handoff.paths["mlp"], handoff.paths["rf"],
                                         handoff.paths["xgb"], verbose=False)
    X_tr, X_te, _ = build_feature_frames(inputs)
    rf, xgb, mlp = handoff.rf["extra"], handoff.xgb, handoff.mlp
    np.testing.assert_array_equal(X_tr["stage1_proba"], mlp.train_proba_oof.to_numpy())
    np.testing.assert_array_equal(X_te["stage1_proba"], mlp.test_proba.to_numpy())
    np.testing.assert_array_equal(X_tr["stage3_proba"], xgb.oof["proba_calibrated"])
    np.testing.assert_array_equal(X_te["stage3_proba"], xgb.refit["test_proba_calibrated"])
    assert not np.allclose(X_tr["stage3_proba"], xgb.refit["train_proba_calibrated"])
    for X, side in ((X_tr, "train"), (X_te, "test")):
        assert list(X.columns) == list(STAGE4_FEATURES)
        abstain = X["stage2_abstain"].to_numpy() == 1
        assert abstain.any() and (~abstain).any()
        router = np.asarray(rf[f"{side}_proba"], dtype=float)
        router = router[:, 1] if router.ndim == 2 else router              # predict_proba: P(y = 1)
        np.testing.assert_array_equal(X["stage2_proba"].to_numpy()[~abstain], router[~abstain])
        assert X["stage2_proba"].isna().to_numpy().tolist() == abstain.tolist()
    np.testing.assert_array_equal(X_te.index, handoff.split["X_test"].index)


def test_the_router_xgboost_and_meta_partitions_are_one(handoff):
    art, gs = handoff.meta, handoff.split
    shared = fold_assignment(gs["y_train"].astype(int), cv_folds=N_STAGE_FOLDS,
                             random_state=RANDOM_STATE).to_numpy()
    np.testing.assert_array_equal(np.asarray(handoff.rf["extra"]["train_fold_id"]), shared)
    np.testing.assert_array_equal(handoff.xgb.oof["fold_id"], shared)
    np.testing.assert_array_equal(art.meta_oof_fold_id, shared)
    up = art.fold_provenance["upstream"]
    assert up["shared_partitions"] == [["rf_router", "xgb", N_STAGE_FOLDS]]
    assert {(n, p["n_folds"], p["seed"]) for n, p in up["streams"].items()} == {
        ("mlp", MLP_TEST_CONFIG["cv_folds"], RANDOM_STATE), ("rf_router", N_STAGE_FOLDS, RANDOM_STATE),
        ("xgb", N_STAGE_FOLDS, RANDOM_STATE)}


def test_the_saved_meta_artifact_reproduces_its_decisions_from_the_saved_upstream(handoff):
    art = load_meta_xgb(handoff.meta_path,
                        expected_split_fingerprint=handoff.meta.split_fingerprints["router"])
    inputs = load_blackbox_stage4_inputs(handoff.split, handoff.paths["mlp"], handoff.paths["rf"],
                                         handoff.paths["xgb"], verbose=False)
    _, X_te, _ = build_feature_frames(inputs)
    p = art.learner.serving_model.predict_proba_raw(X_te)
    np.testing.assert_array_equal(p, art.test["proba"])
    t, cut = art.operating["threshold"], art.abstention["min_confidence"]
    np.testing.assert_array_equal(art.test["pred_with_abstain"],
                                  np.where(np.abs(p - t) >= cut, (p >= t).astype(int), -1))
    assert pd.Index(art.index_test).equals(handoff.split["X_test"].index)
