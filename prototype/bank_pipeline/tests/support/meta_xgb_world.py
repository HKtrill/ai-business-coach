"""
Black-box Stage 4 (Meta-XGB) builders for the Stage 4 suite.

blackbox_world()
    The synthetic Stage 1-3 artifacts of ``tests.support.stage4_world`` in
    the black-box formats: an MLP ``StageOutput`` (10 folds, ``stage_io``
    fingerprint), an RF Router artifact (Stage 4 block under ``"extra"``,
    5 folds, router fingerprint, routes "pass1" / "pass2" / "uncertain") and
    an XGBoost ``Stage3Artifact`` (5 folds, per-fold calibration metadata).
    1,150 training and 350 test rows. Fresh objects on every call, so a test
    may tamper with them.
stage4_frames(world, arts)
    The verified Stage 4 inputs and the Meta-XGB feature block built from them.
meta_config(artifact_dir, **overrides)
    ``MetaXGBConfig`` as the notebook builds it (5 meta folds, so the meta
    partition is the router's and XGBoost's), with 3 reporting folds and
    quiet output.
run_meta(world, arts, artifact_dir, ...)
    ``train_meta_xgb_stage`` on those artifacts with fixed small
    hyperparameters (no Optuna), the router fingerprint scheme supplied as in
    the other Stage 4 tests.
"""

from __future__ import annotations

from typing import Optional

from blackbox_pipeline.models.meta_xgb import (
    MetaXGBConfig, build_feature_frames, load_blackbox_stage4_inputs, train_meta_xgb_stage,
)
from tests.stage3.conftest import ARMS
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3

META_PARAMS = dict(ARMS["meta_xgb"].params)          # the seven tuned keys, small values
N_META_EVAL = 3


def blackbox_world(seed: int = 0):
    w = build_world(n=1500, n_test=350, seed=seed)
    arts = {"mlp": make_stage1(w, arm="blackbox", model="mlp"),
            "rf": make_router(w, layout="extra"),
            "xgb": make_stage3(w, family="xgboost")}
    return w, arts


def stage4_frames(world, arts, GS: Optional[dict] = None):
    """(inputs, X_train, X_test)."""
    inputs = load_blackbox_stage4_inputs(GS or world["GS"], arts["mlp"], arts["rf"], arts["xgb"],
                                         verbose=False, extra_fingerprints=ROUTER_FP)
    X_tr, X_te, _ = build_feature_frames(inputs)
    return inputs, X_tr, X_te


def meta_config(artifact_dir, **overrides) -> MetaXGBConfig:
    return MetaXGBConfig(**{"n_eval_folds": N_META_EVAL, "verbose": False,
                            "show_progress_bar": False, "artifact_dir": str(artifact_dir),
                            **overrides})


def run_meta(world, arts, artifact_dir, save: bool = False, config=None, params=None,
             GS: Optional[dict] = None, **kwargs):
    cfg = config if config is not None else meta_config(artifact_dir)
    return train_meta_xgb_stage(
        GS or world["GS"], arts["mlp"], arts["rf"], arts["xgb"], config=cfg,
        params=dict(META_PARAMS) if params is None else params, params_source="test",
        save=save, verbose=False, extra_fingerprints=ROUTER_FP, **kwargs)
