"""
Recorded Stage 1 runs (GLASS LR and black-box MLP) for the Stage 1 suite.

``fit_glass_stage1`` / ``fit_mlp_stage1`` run each notebook's Stage 1 path on
synthetic data: ``engineer_features`` (LR target encodings) -> the stage's
``fit`` with ``X_train_raw`` -> test scoring -> ``StageOutput``. Optuna is off
for LR unless a test asks for it; the MLP always tunes (1 trial here). Three
things are recorded:

* ``oof``     every fold model of the shared calibration procedure
              (``shared.stage1.oof_calibration``): outer and inner fits and
              their scoring, keyed by the folds' index labels.
* ``encoder`` every ``LRFeatureEngineer`` fit / transform (index labels):
              the full-train ``engineer_features`` fit, the stage's rebuild
              check, one refit per outer fold and one per inner fold.
* ``scaler``  every ``StandardScaler`` fit / transform (index labels): the
              stage's full-train scaler, one per outer and one per inner fold.

``fold_levels`` splits a log into outer-fold fits (they and their held-out
rows cover the whole training split) and inner-fold fits, grouped by the
outer fold whose training rows they live in.

A transform of exactly the rows an object was fitted on is logged as
``transform_own`` (part of fitting), so ``methods=("transform",)`` in the
provenance checks sees held-out and test scoring only.

Runs are serial (``n_jobs=1`` instead of ``-1``): parallelism only, identical
outputs, and the recorders need the folds in this process.

``fold_ids`` pins the outer partition, so a test can permute labels without
the stratified folds moving with them; inner folds stay stratified.
"""

from __future__ import annotations

import contextlib
import io
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import PredefinedSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import blackbox_pipeline.models.mlp.stage as mlp_stage_mod
import glass_pipeline.lr.base_stage as base_stage_mod
import glass_pipeline.lr.lr_stage_calibrated as lr_mod
import shared.stage1.crossfit as crossfit_mod
import shared.stage1.feature_engineering as fe_mod
import shared.stage1.oof_calibration as ocal_mod
from blackbox_pipeline.models.mlp import CalibratedStage1MLP
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from shared.stage1.feature_engineering import LRFeatureEngineer
from shared.stage_io import StageOutput
from tests.support.recorders import Call, CallLog, row_keys
from tests.support.synthetic import make_bank_frame, split_frame

#: Raw, label-free columns: a control feature set with no target encoding.
LABEL_FREE_FEATURES = ["euribor3m", "emp.var.rate", "nr.employed", "cons.conf.idx", "previous"]

#: Small MLP settings for synthetic data (the notebook runs 100 trials).
MLP_TEST_CONFIG = dict(cv_folds=3, n_trials=1, max_epochs=15, patience=5, verbose=False, n_jobs=1)


@dataclass
class Stage1Run:
    split: dict
    X_train: pd.DataFrame          # what the stage was fitted on (engineered)
    X_test: pd.DataFrame
    stage: object
    output: StageOutput
    oof_log: CallLog               # keys: row positions in X_train
    encoder_log: CallLog           # keys: index labels
    scaler_log: CallLog            # keys: index labels

    @property
    def y_train(self) -> pd.Series:
        return self.split["y_train"]


def stage1_split(y_test: Optional[pd.Series] = None, n: int = 400) -> dict:
    split = split_frame(make_bank_frame(n=n))
    if y_test is not None:
        split = {**split, "y_test": y_test}
    return split


# ----------------------------------------------------------------------
# recorders patched into the production modules
# ----------------------------------------------------------------------
def _log_transform(log: CallLog, obj, X) -> None:
    keys = row_keys(X)
    own = set(keys) == getattr(obj, "_fit_rows", None)
    log.append(Call("transform_own" if own else "transform", getattr(obj, "_fit_id", -1), keys))


def _start_fit(log: CallLog, obj, X) -> None:
    obj._fit_id = log.new_fit_id()
    obj._fit_rows = set(row_keys(X))
    log.append(Call("fit", obj._fit_id, row_keys(X)))


def _recording_encoder(log: CallLog):
    class RecordingLRFeatureEngineer(LRFeatureEngineer):
        def fit(self, X_train, y_train):
            if not X_train.index.equals(y_train.index):
                raise AssertionError("encoder fitted on labels that are not aligned to X")
            _start_fit(log, self, X_train)
            return super().fit(X_train, y_train)

        def transform(self, X):
            _log_transform(log, self, X)
            return super().transform(X)
    return RecordingLRFeatureEngineer


def _recording_scaler(log: CallLog):
    class RecordingStandardScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            _start_fit(log, self, X)
            return super().fit(X, y, sample_weight=sample_weight)

        def transform(self, X, copy=None):
            _log_transform(log, self, X)
            return super().transform(X, copy=copy)
    return RecordingStandardScaler


class _ScoredFoldModel:
    """A fold model that logs its scoring, and may only score its own held-out fold."""

    def __init__(self, model, design, fit_id, log):
        self._model, self._design, self._fit_id, self._log = model, design, fit_id, log

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        method = getattr(self._model, name)            # AttributeError -> hasattr() is False
        if name not in ("decision_function", "predict_proba"):
            return method

        def scored(X):
            if X is not self._design.X_val:
                raise AssertionError("a fold model scored rows other than its held-out fold")
            self._log.append(Call(name, self._fit_id, tuple(self._design.index_val)))
            return method(X)
        return scored


def _recording_fit_on(log: CallLog):
    real = ocal_mod._fit_on

    def fit_on(base, design):
        model = real(base, design)
        fid = log.new_fit_id()
        log.append(Call("fit", fid, tuple(design.index_train)))
        return _ScoredFoldModel(model, design, fid, log)
    return fit_on


def _serial_calibrate(real):
    def run(*args, **kwargs):
        return real(*args, **{**kwargs, "n_jobs": 1})
    return run


def _pinned_folds(fold_ids: np.ndarray):
    """Pin the OUTER StratifiedKFold (and its reconstruction) to ``fold_ids``;
    inner folds (fewer rows) stay ordinary stratified folds."""
    class splitter:
        def __init__(self, n_splits=None, shuffle=None, random_state=None):
            self.real = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                                        random_state=random_state)

        def split(self, X, y=None, groups=None):
            if len(X) == len(fold_ids):
                return PredefinedSplit(np.asarray(fold_ids)).split()
            return self.real.split(X, y)

    def assignment(y, *, cv_folds, random_state, index=None):
        index = y.index if index is None and isinstance(y, pd.Series) else index
        return pd.Series(np.asarray(fold_ids), index=index, name="fold_id")
    return splitter, assignment


@contextlib.contextmanager
def recorded_stage1(logs: dict, fold_ids: Optional[np.ndarray] = None):
    """Patch both arms' Stage 1 modules with the serial recorders."""
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
        serial = _serial_calibrate(ocal_mod.calibrate_oof)
        mp.setattr(lr_mod, "calibrate_oof", serial)
        mp.setattr(mlp_stage_mod, "calibrate_oof", serial)
        mp.setattr(ocal_mod, "_fit_on", _recording_fit_on(logs["oof"]))
        mp.setattr(fe_mod, "LRFeatureEngineer", _recording_encoder(logs["encoder"]))
        scaler = _recording_scaler(logs["scaler"])
        for mod in (base_stage_mod, crossfit_mod, mlp_stage_mod):
            mp.setattr(mod, "StandardScaler", scaler)
        if fold_ids is not None:
            splitter, assignment = _pinned_folds(fold_ids)
            mp.setattr(crossfit_mod, "StratifiedKFold", splitter)
            for mod in (lr_mod, mlp_stage_mod):
                mp.setattr(mod, "fold_assignment", assignment)
        yield


def _engineered(split: dict, features: Optional[Sequence[str]]):
    """(X_train, X_test, X_train_raw) as the notebooks build them."""
    if features is None:
        X_tr, X_te, _ = fe_mod.engineer_features(split["X_train"], split["X_test"], split["y_train"])
        return X_tr, X_te, split["X_train"]
    return split["X_train"][list(features)], split["X_test"][list(features)], None


def _frozen(split, X_tr, X_te, stage, output, logs) -> Stage1Run:
    frozen = {name: CallLog(log) for name, log in logs.items()}
    return Stage1Run(split, X_tr, X_te, stage, output,
                     frozen["oof"], frozen["encoder"], frozen["scaler"])


def fit_glass_stage1(
    split: Optional[dict] = None,
    *,
    features: Optional[Sequence[str]] = None,
    fold_ids: Optional[np.ndarray] = None,
    tune_hyperparameters: bool = False,
    **stage_kwargs,
) -> Stage1Run:
    """
    One recorded GLASS LR run. ``features=None`` engineers the LR features
    and passes ``X_train_raw`` as the notebook does; a list of raw columns is
    a label-free control. Logs are copied when the run ends.
    """
    split = split if split is not None else stage1_split()
    logs = {name: CallLog() for name in ("oof", "encoder", "scaler")}
    with recorded_stage1(logs, fold_ids):
        X_tr, X_te, X_raw = _engineered(split, features)
        stage = CalibratedLRStage(tune_hyperparameters=tune_hyperparameters, **stage_kwargs)
        stage.fit(X_tr, split["y_train"], X_train_raw=X_raw)
        stage.evaluate(X_te, split["y_test"], plot=False)
        output = stage.to_stage_output()
    return _frozen(split, X_tr, X_te, stage, output, logs)


def fit_mlp_stage1(
    split: Optional[dict] = None,
    *,
    fold_ids: Optional[np.ndarray] = None,
    **config,
) -> Stage1Run:
    """One recorded black-box MLP run (engineered features + X_train_raw)."""
    split = split if split is not None else stage1_split()
    logs = {name: CallLog() for name in ("oof", "encoder", "scaler")}
    with recorded_stage1(logs, fold_ids):
        X_tr, X_te, X_raw = _engineered(split, None)
        stage = CalibratedStage1MLP(**{**MLP_TEST_CONFIG, **config})
        stage.fit(X_tr, split["y_train"], X_train_raw=X_raw)
        output = stage.to_stage_output(X_te, split["y_test"])
    return _frozen(split, X_tr, X_te, stage, output, logs)


def fold_levels(log: CallLog, train_rows) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    (outer fit ids, {outer fit id: inner fit ids}) for fold-level fits in ``log``.

    An outer fit and the rows it scored/transformed cover the whole training
    split; an inner fit covers one outer fold's training rows, and is filed
    under the outer fit fitted on exactly those rows. Fits on the whole split
    (full-train objects) are left out.
    """
    train = frozenset(train_rows)
    fits = log.fits()
    reached = defaultdict(set)
    for c in log:
        if c.method not in ("fit", "fit_transform", "transform_own"):
            reached[c.fit_id].update(c.keys)
    outer = [fid for fid, rows in fits.items() if rows < train and rows | reached[fid] == train]
    by_rows = {fits[fid]: fid for fid in outer}
    inner: Dict[int, List[int]] = defaultdict(list)
    for fid, rows in fits.items():
        if rows < train and fid not in outer:
            inner[by_rows[frozenset(rows | reached[fid])]].append(fid)
    return outer, dict(inner)


def sub_log(log: CallLog, fit_ids) -> CallLog:
    ids = set(fit_ids)
    return CallLog(c for c in log if c.fit_id in ids)


def unrecorded_scale(stage, X: pd.DataFrame) -> np.ndarray:
    """``stage.scaler.transform(X)`` without adding a call to the scaler log."""
    return StandardScaler.transform(stage.scaler, X)


@pytest.fixture(scope="session")
def lr_run() -> Stage1Run:
    """The notebook configuration (auto calibration, 10 folds), without Optuna."""
    return fit_glass_stage1()


@pytest.fixture(scope="session")
def lr_run_5() -> Stage1Run:
    """GLASS LR with 5 outer / 5 inner folds: for tests where the fold count is immaterial."""
    return fit_glass_stage1(cv_folds=5)


@pytest.fixture(scope="session")
def mlp_run() -> Stage1Run:
    return fit_mlp_stage1()


@pytest.fixture
def serial_stage1(monkeypatch):
    """Unrecorded serial fits for tests that call a stage's ``fit`` directly."""
    serial = _serial_calibrate(ocal_mod.calibrate_oof)
    monkeypatch.setattr(lr_mod, "calibrate_oof", serial)
    monkeypatch.setattr(mlp_stage_mod, "calibrate_oof", serial)
