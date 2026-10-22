"""
Black-box Stage 2 (RF Router) builders for the Stage 2 suite.

serial_forests()
    Context in which every forest is built with ``n_jobs=1``. Production uses
    ``n_jobs=-1``, whose probability sums vary in the last ULP between runs
    (documented in ``rf_router.forest.build_pipe``); serial forests make runs
    bit-identical and are faster at this size. ``n_jobs`` is fixed at
    construction, so routers built inside the context stay serial afterwards.
recording_forests(log, params=None)
    Context in which every ``ForestTrainer.fit`` and
    ``ForestTrainer.positive_proba`` call is appended to ``log``, keyed by
    index label (fit ids in call order); ``params``, when given, maps each fit
    id to its trainer's hyperparameters. Outputs are unchanged.
BANK_CONSTRAINTS
    Feasible on the engineered synthetic bank split, for the full training
    rows and for every outer-fold router, with all three routes populated.
bank_world()
    Cached and shared across modules (nothing may modify it). The notebook's
    Stage 2 path on a synthetic bank frame: 80/20 split ->
    ``shared.stage2.engineer_features`` (29 binary bins) ->
    ``unpack_global_split`` -> two-pass router (5 folds, seed 42) -> outer
    cross-fit on the shared fold plan (GLASS ``make_folds``) -> one-pass test
    outputs. The router fit and the cross-fit are recorded; each outer fold's
    calls in the cross-fit log start with a ``fold_start`` marker.
fold_segments(log)
    The cross-fit log split at those markers, one ``CallLog`` per outer fold.
stage4_block(...) / save_world(world, out_dir)
    The ``extra_metadata`` dict notebook Cell 24 passes to ``save_rf_router``,
    and that call.
"""

from __future__ import annotations

import contextlib
import functools
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

import blackbox_pipeline.models.rf_router.forest as forest_mod
from blackbox_pipeline.models.rf_router import (
    OperatingConstraints, RFRouter, RFRouterConfig, crossfit_rf_router, save_rf_router,
    unpack_global_split,
)
from blackbox_pipeline.models.rf_router.forest import ForestTrainer
from glass_pipeline.glass_router.pipeline import fold_fingerprint, make_folds
from shared.stage2.feature_engineering import engineer_features
from tests.support.recorders import Call, CallLog
from tests.support.stage2_synthetic import SMALL_PARAMS
from tests.support.synthetic import make_bank_frame, split_frame

N_STAGE_FOLDS, RANDOM_STATE = 5, 42     # notebook values
TEST_SIZE = 0.20                        # GlobalSplitManager(test_size=0.20)

BANK_CONSTRAINTS = OperatingConstraints(
    min_precision_not_subscribe=0.92, max_subscriber_leakage_rate=0.30,
    max_subscriber_leakage_absolute=10_000, min_precision_subscribe=0.50,
    min_band_support_pass1=40, min_band_support_pass2=20,
)

_REAL_BUILD_PIPE = forest_mod.build_pipe


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@contextlib.contextmanager
def serial_forests():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(forest_mod, "build_pipe",
                   lambda params, random_state, n_jobs=-1: _REAL_BUILD_PIPE(params, random_state, n_jobs=1))
        yield


@contextlib.contextmanager
def recording_forests(log: CallLog, params: Optional[dict] = None):
    real_fit, real_score = ForestTrainer.fit, ForestTrainer.positive_proba

    def fit(self, X, y, random_state=None):
        pipe = real_fit(self, X, y, random_state)
        pipe._rec_fit_id = log.new_fit_id()
        log.append(Call("fit", pipe._rec_fit_id, tuple(X.index)))
        if params is not None:
            params[pipe._rec_fit_id] = dict(self.params)
        return pipe

    def positive_proba(pipe, X):
        log.append(Call("predict_proba", getattr(pipe, "_rec_fit_id", -1), tuple(X.index)))
        return real_score(pipe, X)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ForestTrainer, "fit", fit)
        mp.setattr(ForestTrainer, "positive_proba", staticmethod(positive_proba))
        yield


def fit_ids(log: CallLog) -> list:
    """Fit ids in the order the fits happened."""
    return [c.fit_id for c in log if c.method == "fit"]


def fold_segments(log: CallLog) -> list:
    """One CallLog per outer fold, split at the ``fold_start`` markers."""
    segments = []
    for c in log:
        if c.method == "fold_start":
            segments.append(CallLog())
        elif segments:
            segments[-1].append(c)
    return segments


def router_config(constraints=BANK_CONSTRAINTS, **overrides) -> RFRouterConfig:
    """Notebook ``make_router_config`` with small forests and support floors."""
    kw = dict(mode="two_pass", constraints=constraints, rf_params=SMALL_PARAMS,
              n_oof_folds=N_STAGE_FOLDS, random_state=RANDOM_STATE, remainder_min_size=60,
              verbose=False)
    return RFRouterConfig(**{**kw, **overrides})


def engineer(X_tr, X_va, y_tr):
    return quiet(engineer_features, X_tr, X_va, y_tr)


def stage_folds(y):
    """Notebook Cell 18: the shared fold plan, its fold ids and fingerprint."""
    folds = make_folds(y, N_STAGE_FOLDS, RANDOM_STATE)
    fold_id = np.full(len(y), -1)
    for k, (_, va) in enumerate(folds):
        fold_id[va] = k
    return folds, fold_id, fold_fingerprint(fold_id)


def stage4_block(stage2, crossfit, fold_id, test_out) -> dict:
    """Notebook Cell 24: ``_common_meta`` plus the two-pass Stage 4 fields."""
    p_te, c_te, d_te, proba_te = test_out
    return {
        "index_train": stage2["X_train"].index,
        "index_test": stage2["X_test"].index,
        "constraint_source": "test_fixture",
        "rf_params_origin": "tests.support.stage2_synthetic.SMALL_PARAMS",
        "glass_artifact": "none (test fixture)",
        "train_outputs_oof": True,
        "train_pred": crossfit.oof["pred"],
        "train_confidence": crossfit.oof["confidence"],
        "train_decisions": crossfit.oof["decisions"],
        "train_proba": crossfit.oof["proba"],
        "train_fold_id": fold_id,
        "oof_protocol": crossfit.protocol,
        "oof_fold_summary": crossfit.folds.to_dict("records"),
        "test_pred": p_te,
        "test_confidence": c_te,
        "test_decisions": d_te,
        "test_proba": proba_te,
    }


@dataclass(frozen=True)
class BankWorld:
    global_split: dict
    stage2: dict              # unpack_global_split(..., engineered=...)
    folds: list
    fold_id: np.ndarray
    fold_fp: str
    router: RFRouter
    router_log: CallLog       # the router's fit (recorded), nothing after
    crossfit: object          # CrossfitResult
    crossfit_log: CallLog
    test_out: tuple           # router.predict_with_proba(X_test), one pass


@functools.lru_cache(maxsize=None)
def bank_world() -> BankWorld:
    gs = split_frame(make_bank_frame(n=1000), test_size=TEST_SIZE)
    X_tr, X_te = engineer(gs["X_train"], gs["X_test"], gs["y_train"])
    s2 = unpack_global_split(gs, engineered={"X_eng_train": X_tr, "X_eng_test": X_te})
    folds, fold_id, fold_fp = stage_folds(s2["y_train"])
    router_log, crossfit_log = CallLog(), CallLog()
    with serial_forests():
        with recording_forests(router_log):
            router = RFRouter(router_config()).fit(s2["X_train"], s2["y_train"], s2["split_fingerprint"])
        def fold_config():                     # called once per outer fold, before its fit
            crossfit_log.append(Call("fold_start", -1, ()))
            return router_config()

        with recording_forests(crossfit_log):
            cf = crossfit_rf_router(gs["X_train"], s2["y_train"], folds, engineer=engineer,
                                    make_config=fold_config, split_fp=s2["split_fingerprint"],
                                    fold_fp=fold_fp, random_state=RANDOM_STATE, verbose=False)
    test_out = router.predict_with_proba(s2["X_test"])
    return BankWorld(gs, s2, folds, fold_id, fold_fp, router, CallLog(router_log), cf,
                     CallLog(crossfit_log), test_out)


def save_world(world: BankWorld, out_dir) -> Path:
    return quiet(save_rf_router, world.router, out_dir, name="rf_router_two_pass",
                 extra_metadata=stage4_block(world.stage2, world.crossfit, world.fold_id,
                                             world.test_out))
