"""
The GLASS cascade (glass_cascade.ipynb, Cells 4-19) run end to end on small
synthetic bank data with the production producers, savers and Stage 4.

    split    data.global_splits.GlobalSplitManager(test_size=0.20, random_state=42)
    Stage 1  shared.stage1 engineer_features -> CalibratedLRStage (10-fold nested
             cross-fit, calibration "auto") -> save()            [Cells 5, 8]
    Stage 2  glass_router.rf engineer_features -> full-train GlassRouterPipeline ->
             build_glass_output / predict_proba (test); crossfit_stage2 (5 folds,
             seed 42) -> ModelSaver.save_model(oof=..., split_fingerprint=
             rf_router.split_fingerprint)                         [Cells 10, 13, 14, 16]
    Stage 3  Stage3Block.from_global_split(EBMFeaturePipeline, per-fold GLASS DAG)
             -> train_ebm_stage (5 tuning folds, nested calibration and thresholds)
                                                                  [Cells 17, 18]
    Stage 4  train_meta_stage(GLOBAL_SPLIT, lr_path, glass_path, ebm_path)
             with the real fingerprint schemes (no stand-ins)     [Cell 19]

Lightweight on purpose, and the same everywhere it is used:

* Hyperparameters are fixed (LR tuning off, EBM ``EBM_TEST_PARAMS`` with two
  outer bags, RF ``TINY_RF``). Every isolation property below therefore
  holds conditional on fixed hyperparameters: selecting them by Optuna on the
  same folds is the open finding F24.
* The GLASS Router's rule search (rule generator + ILP, ~30 s per fit) is
  replaced by ``FixedRuleRouter.fit``: two single-feature Pass 1 segments with
  the lowest and two Pass 2 segments with the highest positive rate on the
  fit rows, precision measured on those rows (as the evaluator does). Its
  routing, ``predict_proba`` blend, cross-fit and saver are production code.
  Rule search itself is tested in tests/stage2.
* Stage 3 runs ``EBMStage3Config`` (the GLASS EBM config), as the notebook does
  since F21; ``train_ebm_stage`` refuses any other config.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pytest

import glass_pipeline.ebm.ebm_stage as ebm_stage_mod
import glass_pipeline.glass_router.pipeline.crossfit as crossfit_mod
from blackbox_pipeline.models.rf_router import split_fingerprint as router_split_fingerprint
from data.global_splits.split_manager import GlobalSplitManager
from glass_pipeline.ebm.config import EBMStage3Config
from glass_pipeline.ebm.ebm_stage import train_ebm_stage
from glass_pipeline.glass_router.core.rule import ABSTAIN
from glass_pipeline.glass_router.model_outputs import build_glass_output
from glass_pipeline.glass_router.model_saver import ModelSaver
from glass_pipeline.glass_router.pipeline import GlassRouterPipeline
from glass_pipeline.glass_router.rf import engineer_features as router_features
from glass_pipeline.glass_router.rule_generator.rule_logger import RuleLogger
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from glass_pipeline.meta_ebm.meta_stage import train_meta_stage
from shared.stage1.feature_engineering import engineer_features as lr_features
from shared.stage2 import RF_FEATURES_BINARY
from shared.stage3.data import Stage3Block
from shared.stage3.feature_engineering import EBM_FEATURES, EBMFeaturePipeline
import shared.stage_io as stage_io
from shared.stage_io import split_fingerprint
from tests.stage2.conftest import TINY_RF, legacy_config, selected_rule
from tests.stage3.conftest import EBM_TEST_BAGS, EBM_TEST_PARAMS
from tests.support.synthetic import TARGET, make_bank_frame

N_ROWS, POSITIVE_RATE = 500, 0.19        # 76 training positives: fold base rates differ


class FixedRuleRouter(GlassRouterPipeline):
    """Production GLASS Router routing with a fixed, deterministic rule choice."""

    MIN_SUPPORT = 15

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        y = np.asarray(y_train, dtype=float)
        base = float(y.mean())
        rates = []
        for f in RF_FEATURES_BINARY:
            if f in X_train.columns:
                m = (X_train[f] == 1).to_numpy()
                if m.sum() >= self.MIN_SUPPORT:
                    rates.append((float(y[m].mean()), f))
        low = [(r, f) for r, f in sorted(rates) if r < base][:2]
        high = [(r, f) for r, f in sorted(rates, reverse=True) if r > base][:2]
        self.training_base_rate = base
        self.train_label_fingerprint_ = stage_io.label_fingerprint(y_train.index, y_train)  # as the real fit
        self.pass1_rules = [selected_rule(f"P1_{f}", [(f, 1)], "pass1", 1.0 - r) for r, f in low]
        self.pass2_rules = [selected_rule(f"P2_{f}", [(f, 1)], "pass2", r) for r, f in high]
        self.is_fitted = True
        return self


@contextlib.contextmanager
def _lightweight():
    """Fixed-rule router inside the cross-fit; two outer bags per EBM fit."""
    real_factory = ebm_stage_mod.make_ebm_factory

    def factory(config, block):
        f = real_factory(config, block)
        f.extra_params = dict(EBM_TEST_BAGS)
        return f

    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
        mp.setattr(crossfit_mod, "GlassRouterPipeline", FixedRuleRouter)
        mp.setattr(ebm_stage_mod, "make_ebm_factory", factory)
        yield


@dataclass
class GlassCascade:
    GS: dict
    paths: dict                    # lr / glass / ebm / meta -> saved file
    lr_stage: CalibratedLRStage
    lr_X: tuple                    # (X_train_eng, X_test_eng)
    router: FixedRuleRouter        # full-train router
    router_X: tuple                # (router features train, test)
    oof: object                    # Stage2OOF
    config: object                 # GlassRouterConfig
    ebm: object                    # Stage3Artifact as returned
    meta: dict                     # Meta-EBM artifact as returned

    def load(self, name):
        return joblib.load(self.paths[name])


def glass_split(y_test: Optional[pd.Series] = None) -> dict:
    frame = make_bank_frame(n=N_ROWS, positive_rate=POSITIVE_RATE)
    with contextlib.redirect_stdout(io.StringIO()):
        GS = GlobalSplitManager(test_size=0.20, random_state=42).create_split(frame, target_col=TARGET)
    if y_test is not None:
        GS = {**GS, "y_test": y_test}
    return GS


def build_glass_cascade(out_dir: Path, GS: Optional[dict] = None) -> GlassCascade:
    """Run Stages 1-4 as glass_cascade does; artifacts are written under ``out_dir``."""
    GS = glass_split() if GS is None else GS
    X_tr, X_te, y_tr, y_te = GS["X_train"], GS["X_test"], GS["y_train"], GS["y_test"]
    out_dir = Path(out_dir)
    with _lightweight():
        # Stage 1
        lr_tr, lr_te, _ = lr_features(X_tr, X_te, y_tr)
        stage = CalibratedLRStage(tune_hyperparameters=False, calibration_method="auto",
                                  optimize_threshold=True, cv_folds=10)
        stage.fit(lr_tr, y_tr, X_train_raw=X_tr)
        stage.evaluate(lr_te, y_te, plot=False)
        lr_path = stage.save(str(out_dir / "lr"))["path"]

        # Stage 2
        config = legacy_config()
        r_tr, r_te = router_features(X_tr, X_te, y_tr)
        router = FixedRuleRouter(config, rule_logger=RuleLogger(enabled=False)).fit(r_tr, y_tr)
        test_out = build_glass_output(router, r_te, abstain_value=ABSTAIN)
        oof = crossfit_mod.crossfit_stage2(X_tr, y_tr, config, TINY_RF, n_splits=5,
                                           random_state=42, glass_rf_model="fold_rf")
        glass_path = ModelSaver(glass_model=router).save_model(
            train_out=None, train_proba=None, oof=oof,
            train_insample_proba=router.predict_proba(r_tr),
            test_out=test_out, test_proba=router.predict_proba(r_te), global_split=GS,
            split_fingerprint=router_split_fingerprint(X_tr, X_te), y_test=y_te,
            output_path=str(out_dir / "glass_router.joblib"))

        # Stage 3
        ebm_cfg = EBMStage3Config(n_tune_folds=5, n_eval_folds=2, n_jobs=1, verbose=False,
                                  show_progress_bar=False, expected_features=list(EBM_FEATURES),
                                  artifact_dir=str(out_dir / "ebm"))
        block = Stage3Block.from_global_split(
            GS, EBMFeaturePipeline, ebm_cfg,
            split_fingerprint=split_fingerprint(X_tr.index, X_te.index, y_tr, y_te))
        ebm_art, ebm_path = train_ebm_stage(block=block, config=ebm_cfg, params=dict(EBM_TEST_PARAMS))

        # Stage 4
        meta, meta_path = train_meta_stage(GS, lr_path=lr_path, glass_path=glass_path,
                                           ebm_path=ebm_path, artifact_dir=str(out_dir / "meta"),
                                           target_recall=0.70, verbose=False)
    return GlassCascade(GS=GS, paths={"lr": lr_path, "glass": glass_path, "ebm": ebm_path,
                                      "meta": meta_path},
                        lr_stage=stage, lr_X=(lr_tr, lr_te), router=router, router_X=(r_tr, r_te),
                        oof=oof, config=config, ebm=ebm_art, meta=meta)


@pytest.fixture(scope="module")
def cascade(tmp_path_factory) -> GlassCascade:
    return build_glass_cascade(tmp_path_factory.mktemp("glass_cascade"))
