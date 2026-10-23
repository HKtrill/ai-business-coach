"""
Shared Stage 3 runs for the conformance suite.

Each arm (GLASS EBM, black-box XGBoost, Meta-XGB learner) is run once per test
module through the REAL ``shared.stage3.runner.Stage3Runner`` on synthetic
data, with tiny supplied hyperparameters (no Optuna) and a recording wrapper
around the arm's estimator. Tests then read the artifact and the call log.

Meta-XGB runs on the generic ``Stage3Config``: its own ``MetaXGBConfig`` pins
the Stage 4 feature contract, which the Stage 4 tests cover. The runner,
folds, calibration and thresholds exercised here are identical.

The conformance runs above use pre-engineered features (``Passthrough``) and
no interaction pairs. The GLASS EBM helpers at the end add what is specific
to that arm: the real ``EBMFeaturePipeline``, the validated interaction pairs
and ``train_ebm_stage``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable

import pytest

import shared.stage3.data as data_mod

from blackbox_pipeline.models.meta_xgb.estimator import MetaXGBEstimator
from blackbox_pipeline.models.xgb.shared_protocol import XGBEstimator, XGBStage3Config
from glass_pipeline.ebm.config import EBMStage3Config
from glass_pipeline.ebm.estimator import EBMFactory
from shared.stage3.config import Stage3Config
from shared.stage3.data import Stage3Block
from shared.stage3.feature_engineering import EBM_FEATURES, EBMFeaturePipeline
from shared.stage3.runner import Stage3Runner
from shared.stage_io import split_fingerprint
from tests.support.recorders import CallLog, RecordingStage3Estimator
from tests.support.synthetic import make_bank_frame, split_frame

N_TUNE, N_EVAL = 3, 3
# interpret's default is 14 outer bags; 2 keep each fit ~0.1 s. Bagging only
# averages fits made on the same rows, so no contract tested here depends on it.
EBM_TEST_BAGS = {"outer_bags": 2}


class Passthrough:
    """Index-preserving no-op feature pipeline (features are pre-engineered)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()

    def fit_transform(self, X, y=None):
        return X.copy()


@dataclass(frozen=True)
class Arm:
    name: str
    make_config: Callable[..., Stage3Config]
    make_estimator: Callable[[], Any]
    params: dict


ARMS = {
    "ebm": Arm(
        "ebm",
        EBMStage3Config,
        lambda: EBMFactory(interactions=[], n_jobs=1, extra_params=dict(EBM_TEST_BAGS)),
        {"learning_rate": 0.05, "max_rounds": 40, "max_bins": 32, "max_interaction_bins": 16},
    ),
    "xgboost": Arm(
        "xgboost",
        XGBStage3Config,
        lambda: XGBEstimator(n_jobs=1),
        {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 2, "min_child_weight": 1.0,
         "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0},
    ),
    "meta_xgb": Arm(
        "meta_xgb",
        lambda **kw: Stage3Config(model_family="meta_xgb", **kw),
        lambda: MetaXGBEstimator(n_jobs=1),
        {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 2, "min_child_weight": 1.0,
         "subsample": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.01},
    ),
}


@dataclass
class ArmRun:
    arm: Arm
    split: dict
    config: Stage3Config
    block: Stage3Block
    artifact: Any
    log: CallLog
    split_fp: str


def stage3_split(y_test=None) -> dict:
    split = split_frame(make_bank_frame(n=400))
    if y_test is not None:
        split = {**split, "y_test": y_test}
    return split


@contextlib.contextmanager
def folds_from(y):
    """
    Build every full-population Stage 3 fold plan from ``y``, whatever labels
    the block is given, so labels can move while the outer folds stay put.
    Inner plans (over one outer fold's training rows) are built as usual:
    they read only those rows' labels.
    """
    real = data_mod.fold_assignment

    def assignment(labels, cv_folds, random_state):
        source = y if len(labels) == len(y) else labels
        return real(source, cv_folds=cv_folds, random_state=random_state)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "fold_assignment", assignment)
        yield


def run_arm(arm: Arm, artifact_dir, split=None, tune: bool = False,
            fold_labels=None, **config_overrides) -> ArmRun:
    """
    One full shared-protocol fit of ``arm``: block -> runner -> artifact.

    ``tune=False`` supplies ``arm.params`` (no Optuna) with reuse provenance,
    as the notebooks do when reusing parameters; ``tune=True`` runs the
    arm's real Optuna search with ``config.n_trials`` trials. ``fold_labels``
    pins the outer folds to those labels (``folds_from``).
    """
    split = split if split is not None else stage3_split()
    cfg = arm.make_config(
        n_tune_folds=N_TUNE, n_eval_folds=N_EVAL, n_jobs=1, verbose=False,
        show_progress_bar=False, artifact_dir=str(artifact_dir),
        expected_features=list(split["X_train"].columns), **config_overrides,
    )
    fp = split_fingerprint(split["X_train"].index, split["X_test"].index,
                           split["y_train"].astype(int), split["y_test"].astype(int))
    with folds_from(fold_labels) if fold_labels is not None else contextlib.nullcontext():
        block = Stage3Block.from_global_split(split, Passthrough, cfg, split_fingerprint=fp)
    log = CallLog()
    runner = Stage3Runner(cfg, RecordingStage3Estimator(arm.make_estimator(), log))
    if tune:
        artifact = runner.fit(block)
    else:
        provenance = {"n_trials_requested": cfg.n_trials, "random_state": cfg.random_state,
                      "feature_fit_scope": cfg.feature_fit_scope}
        artifact = runner.fit(block, params=dict(arm.params), params_source="test",
                              params_provenance=provenance)
    return ArmRun(arm, split, cfg, block, artifact, log, fp)


@pytest.fixture(scope="session")
def arm_runs(tmp_path_factory):
    """One recorded run per arm, shared by every Stage 3 test (read-only)."""
    return {name: run_arm(arm, tmp_path_factory.mktemp(name)) for name, arm in ARMS.items()}


@pytest.fixture(params=sorted(ARMS))
def arm_run(request, arm_runs):
    return arm_runs[request.param]


# ----------------------------------------------------------------------
# GLASS EBM: the real feature DAG (EBMFeaturePipeline), interaction pairs
# and EBM, as glass_cascade builds them (test_glass_ebm_*.py)
# ----------------------------------------------------------------------
EBM_TEST_PARAMS = dict(ARMS["ebm"].params)


def glass_ebm_config(artifact_dir, **overrides) -> EBMStage3Config:
    """``EBMStage3Config`` with the small fold counts used across Stage 3 tests."""
    return EBMStage3Config(**{
        "n_tune_folds": N_TUNE, "n_eval_folds": N_EVAL, "n_jobs": 1, "verbose": False,
        "show_progress_bar": False, "artifact_dir": str(artifact_dir),
        "expected_features": list(EBM_FEATURES), **overrides,
    })


def cheap_ebm(interactions=(), **kwargs) -> EBMFactory:
    return EBMFactory(interactions=list(interactions), n_jobs=1, extra_params=dict(EBM_TEST_BAGS),
                      **kwargs)


@pytest.fixture(scope="session")
def ebm_features():
    """``(split, fitted EBMFeaturePipeline, X_train, X_test)`` on the Stage 3 split."""
    split = stage3_split()
    pipe = EBMFeaturePipeline()
    X_tr = pipe.fit_transform(split["X_train"], split["y_train"])
    return split, pipe, X_tr, pipe.transform(split["X_test"])
