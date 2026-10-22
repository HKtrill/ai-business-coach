"""
Black-box Stage 3 end to end: ``train_xgb_stage``
(blackbox_pipeline.models.xgb.shared_protocol) on the per-fold GLASS feature
DAG (``EBMFeaturePipeline``) and the shared runner, built the way
blackbox_cascade Cells 29-31 build it (router-format split fingerprint, the
explicit ``XGBStage3Config``).

Contract: ``train_xgb_stage`` accepts only an ``XGBStage3Config`` with the
black-box XGBoost identity, refusing anything else before a model is fitted
(F37), and builds its estimator from that config; the XGBoost block IS the
GLASS EBM block (same fingerprint, features and folds), and the refit model is
fitted on exactly the EBM feature columns; every training row is scored once by
a model that never saw it, inner fits stay inside their outer fold's training
rows, nothing is fitted on test rows; a fold's own labels never reach any of its
training-side outputs (F22) and test labels reach nothing learned; the saved
artifact is found by its fingerprints, identifies the black-box XGBoost arm,
and after reloading scores raw rows as the refit did; Stage 4 reads it as
calibrated OOF train / calibrated refit test on the shared fold plan, for its
own split only, and refuses pre-PR-33 dicts, pre-F22 artifacts, global feature
fitting, in-sample substitution and foreign fold ids. The notebook calls all of
this the way it is tested here.

Protocol steps the arms share (OOF provenance, calibration, thresholds,
store, lookup and parameter reuse) are covered for this arm, on pre-engineered
features, in the other Stage 3 modules; this module re-checks only what the
GLASS feature DAG and ``train_xgb_stage`` could change. Meta-XGBoost (Stage 4)
is Step 2B.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

import blackbox_pipeline.models.xgb.shared_protocol as protocol_mod
from blackbox_pipeline.models.rf_router import split_fingerprint as router_split_fingerprint
from blackbox_pipeline.models.xgb.shared_protocol import (
    XGBSearchSpace, XGBStage3Config, train_xgb_stage,
)
from shared.stage3.artifact import STAGE4_FEATURE_SUFFIXES, Stage3ArtifactStore, find_stage3_artifact
from shared.stage3.config import Stage3Config
from shared.stage3.data import Stage3Block
from shared.stage3.feature_engineering import EBM_FEATURES, EBMFeaturePipeline
from shared.stage4 import Stage4InputError, build_stage4_inputs, stage3_stream
from shared.stage_io import fold_assignment
from shared.validation import ValidationError
from tests.stage3.conftest import ARMS, N_EVAL, N_TUNE, folds_from, glass_ebm_config, stage3_split
from tests.support import mutations as mut
from tests.support.checks import assert_never_fit_on, assert_oof_provenance
from tests.support.notebooks import BANK_ROOT, project_imports
from tests.support.recorders import CallLog, RecordingStage3Estimator
from tests.support.synthetic import make_bank_frame, split_frame

NOTEBOOK = BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"
PARAMS = dict(ARMS["xgboost"].params)
TRAIN_SIDE = ("proba", "proba_calibrated", "decision", "margin", "confidence", "threshold_per_row")


@dataclasses.dataclass
class XGBRun:
    split: dict                      # raw rows with the block's labels
    config: XGBStage3Config
    block: Stage3Block
    artifact: Any
    path: Optional[str]
    log: CallLog
    estimators: list                 # what train_xgb_stage handed the runner


def xgb_config(artifact_dir, **overrides) -> XGBStage3Config:
    """The notebook's ``XGBStage3Config`` with the small fold counts used across Stage 3 tests."""
    return XGBStage3Config(**{
        "n_tune_folds": N_TUNE, "n_eval_folds": N_EVAL, "n_jobs": 1, "verbose": False,
        "show_progress_bar": False, "artifact_dir": str(artifact_dir),
        "expected_features": list(EBM_FEATURES), **overrides})


def notebook_split(y_train=None, y_test=None) -> dict:
    split = stage3_split()
    return {**split, "y_train": split["y_train"] if y_train is None else y_train,
            "y_test": split["y_test"] if y_test is None else y_test}


def build_block(config, y_train=None, y_test=None, pipeline=EBMFeaturePipeline) -> Stage3Block:
    """Cell 30's block, optionally with moved labels; folds always follow the
    unmoved training labels. Stamped with the router-format fingerprint, as
    the notebook's ``_FP_NOW``."""
    split = notebook_split(y_train, y_test)
    fp = router_split_fingerprint(split["X_train"], split["X_test"])
    with folds_from(stage3_split()["y_train"]):
        return Stage3Block.from_global_split(split, pipeline, config, split_fingerprint=fp)


@contextlib.contextmanager
def recorded(log: CallLog, estimators: list):
    """``train_xgb_stage`` as written, with every fit and score of its
    estimator recorded (the models themselves are unchanged)."""
    real = protocol_mod.Stage3Runner

    def runner(cfg, estimator):
        estimators.append(estimator)
        return real(cfg, RecordingStage3Estimator(estimator, log))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(protocol_mod, "Stage3Runner", runner)
        yield


def run_stage(block: Stage3Block, config, save: bool = False) -> XGBRun:
    log, estimators = CallLog(), []
    with recorded(log, estimators):
        artifact, path = train_xgb_stage(block=block, config=config, params=dict(PARAMS),
                                         params_source="test", save=save)
    raw = stage3_split()
    split = {"X_train": raw["X_train"], "X_test": raw["X_test"],
             "y_train": block.y_train, "y_test": block.y_test}
    return XGBRun(split, config, block, artifact, path, log, estimators)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    cfg = xgb_config(tmp_path_factory.mktemp("xgb"))
    return run_stage(build_block(cfg), cfg, save=True)


@pytest.fixture(scope="module")
def fold0(base):
    """Training rows of tuning fold 0 (positions and index labels)."""
    pos = base.block.tune_plan.holdout(0)
    return pos, base.block.index_train[pos]


@pytest.fixture(scope="module")
def moved_runs(base, fold0):
    """Runs whose labels moved inside tuning fold 0 ("own"), outside it
    ("others"), or on the test split ("test")."""
    y, y_te = base.block.y_train, base.block.y_test
    blocks = {
        "own": build_block(base.config, y_train=mut.permute_labels(y, seed=3, rows=fold0[1])),
        "others": build_block(base.config, y_train=mut.permute_labels(
            y, seed=3, rows=y.index.difference(fold0[1]))),
        "test": build_block(base.config, y_test=mut.permute_labels(y_te, seed=3)),
    }
    return {k: run_stage(b, base.config).artifact for k, b in blocks.items()}


# ----------------------------------------------------------------------
# orchestration and identity (audit F37)
# ----------------------------------------------------------------------
def test_the_estimator_is_built_from_the_config(base, tmp_path, monkeypatch):
    seen = []

    def runner(cfg, estimator):
        seen.append((cfg, estimator))
        raise RuntimeError("stop")

    monkeypatch.setattr(protocol_mod, "Stage3Runner", runner)
    space = XGBSearchSpace(max_depth=(2, 3))
    cfg = xgb_config(tmp_path, search_space=space, random_state=7, n_jobs=2)
    with pytest.raises(RuntimeError, match="stop"):
        train_xgb_stage(block=base.block, config=cfg)
    ((got, est),) = seen
    assert got is cfg and type(est) is protocol_mod.XGBEstimator
    assert est.space is space and (est.random_state, est.n_jobs) == (7, 2)


def test_the_artifact_carries_the_blackbox_xgboost_identity_and_the_applied_params(base):
    (est,) = base.estimators
    cfg = base.config
    assert est.space is cfg.search_space
    art = base.artifact
    assert (art.model_family, art.config["arm"], art.column_prefix) == ("xgboost", "blackbox", "xgb_stage3_")
    assert art.best_params == PARAMS and art.tuning["tuned"] is False
    applied = art.refit["model"].get_params()
    assert {k: applied[k] for k in PARAMS} == PARAMS
    assert (applied["random_state"], applied["n_jobs"]) == (cfg.random_state, cfg.n_jobs)


def _misused(case, tmp_path):
    small = {"n_tune_folds": N_TUNE, "n_eval_folds": N_EVAL, "verbose": False,
             "show_progress_bar": False, "artifact_dir": str(tmp_path),
             "expected_features": list(EBM_FEATURES)}
    return {
        "ebm_config": lambda: glass_ebm_config(tmp_path),        # would be labelled 'ebm' / 'glass'
        "generic_config": lambda: Stage3Config(**small),          # would be labelled 'unset'
        "family_overridden": lambda: xgb_config(tmp_path, model_family="ebm"),
        "arm_overridden": lambda: xgb_config(tmp_path, arm="glass"),
    }[case]()


@pytest.mark.parametrize("case, error, message", [
    ("ebm_config", TypeError, "must be an XGBStage3Config, got EBMStage3Config"),
    ("generic_config", TypeError, "must be an XGBStage3Config, got Stage3Config"),
    ("family_overridden", ValueError, "model_family='ebm'"),
    ("arm_overridden", ValueError, "arm='glass'"),
])
def test_a_config_without_the_blackbox_xgboost_identity_is_refused_f37(base, tmp_path, case, error,
                                                                       message):
    """Before PR 36 each of these trained XGBoost and labelled the artifact
    with the config's identity (F37); now nothing is fitted or saved."""
    cfg = _misused(case, tmp_path)
    log, estimators = CallLog(), []
    with recorded(log, estimators), pytest.raises(error, match=message):
        train_xgb_stage(block=base.block, config=cfg, params=dict(PARAMS), params_source="test")
    assert not estimators and not log and not list(tmp_path.iterdir())


def test_the_xgboost_block_is_the_glass_ebm_block(base, tmp_path):
    """Same split and protocol → the same per-fold features, folds and
    weights as GLASS Stage 3, so the two arms compare like with like."""
    glass_cfg = glass_ebm_config(tmp_path)
    glass = build_block(glass_cfg)
    assert glass.block_fingerprint == base.block.block_fingerprint
    assert base.block.features == list(EBM_FEATURES)
    pd.testing.assert_frame_equal(glass.X_train, base.block.X_train)
    np.testing.assert_array_equal(glass.tune_plan.fold_id, base.block.tune_plan.fold_id)
    assert list(base.artifact.features) == list(EBM_FEATURES)
    assert list(base.artifact.refit["model"].feature_names_in_) == list(EBM_FEATURES)


def test_a_pipeline_off_the_ebm_feature_contract_is_refused(tmp_path):
    class DropsOne(EBMFeaturePipeline):
        def transform(self, X):
            return super().transform(X).drop(columns=[EBM_FEATURES[-1]])

        def fit_transform(self, X, y=None):
            return super().fit_transform(X, y).drop(columns=[EBM_FEATURES[-1]])

    with pytest.raises(ValueError, match=EBM_FEATURES[-1]):
        build_block(xgb_config(tmp_path), pipeline=DropsOne)


# ----------------------------------------------------------------------
# out-of-fold integrity on the GLASS feature DAG
# ----------------------------------------------------------------------
def _plans(run):
    """Fold fits in run order: k-fold report (N_EVAL), OOF (N_TUNE), then the
    inner fits (N_TUNE per outer fold, F22)."""
    train = set(run.block.index_train)
    fold_fits = sorted(fid for fid, rows in run.log.fits().items() if rows < train)
    assert len(fold_fits) == N_EVAL + N_TUNE + N_TUNE * N_TUNE
    inner = fold_fits[N_EVAL + N_TUNE:]
    return {"report": fold_fits[:N_EVAL], "oof": fold_fits[N_EVAL:N_EVAL + N_TUNE],
            "inner": [inner[k * N_TUNE:(k + 1) * N_TUNE] for k in range(N_TUNE)]}


def test_every_training_row_is_scored_once_by_a_model_that_never_saw_it(base):
    index, plans = base.block.index_train, _plans(base)
    for name in ("report", "oof"):
        assert_oof_provenance(CallLog(c for c in base.log if c.fit_id in set(plans[name])), index)
    for (tr, va), fits in zip(base.block.tune_plan, plans["inner"]):
        sub_log = CallLog(c for c in base.log if c.fit_id in set(fits))
        assert_oof_provenance(sub_log, index[tr])
        assert_never_fit_on(sub_log, index[va])
        assert not any(set(c.keys) & set(index[va]) for c in sub_log)    # nor scored


def test_one_model_is_fitted_on_all_training_rows_and_none_on_test_rows(base):
    assert_never_fit_on(base.log, base.block.index_test)
    fits = base.log.fits().values()
    assert sum(rows == set(base.block.index_train) for rows in fits) == 1
    assert base.artifact.population == "full_train"


def test_stage4_frames_are_keyed_by_the_raw_split_index(base):
    art, split = base.artifact, base.split
    train, meta = art.stage4_frame(), art.stage4_metadata()
    assert train.index.equals(split["X_train"].index) and meta.index.equals(train.index)
    assert list(train.columns) == [f"xgb_stage3_{s}" for s in STAGE4_FEATURE_SUFFIXES]
    shared = fold_assignment(split["y_train"], cv_folds=N_TUNE, random_state=base.config.random_state)
    np.testing.assert_array_equal(meta["xgb_stage3_fold_id"], shared.to_numpy())
    for source in ("refit", "fold_mean"):
        test = art.stage4_test_frame(source)
        assert test.index.equals(split["X_test"].index) and list(test.columns) == list(train.columns)
        assert np.isfinite(test.to_numpy(dtype=float)).all()
    assert np.isfinite(train.to_numpy(dtype=float)).all()


# ----------------------------------------------------------------------
# label isolation (audit F22)
# ----------------------------------------------------------------------
def test_a_folds_own_labels_never_reach_its_training_side_outputs_f22(base, fold0, moved_runs):
    """Fold 0's raw score, calibration gate and map, nested threshold and
    decision columns come from models, features and inner folds fitted inside
    folds 1.. only (F22). Moving fold 0's labels moves every other fold,
    never fold 0; moving the other folds' labels moves fold 0."""
    pos = fold0[0]
    rest = np.setdiff1d(np.arange(len(base.block.index_train)), pos)
    b, own, others = base.artifact, moved_runs["own"], moved_runs["others"]
    assert b.calibration["applied"] and all(b.calibration["fold_applied"])  # not trivially equal
    for key in TRAIN_SIDE:
        np.testing.assert_array_equal(own.oof[key][pos], b.oof[key][pos], err_msg=key)
    np.testing.assert_array_equal(own.oof["inner_proba_by_fold"][0], b.oof["inner_proba_by_fold"][0])
    assert own.threshold["oof_nested"]["by_fold"][0] == b.threshold["oof_nested"]["by_fold"][0]
    assert own.calibration["fold_applied"][0] == b.calibration["fold_applied"][0]
    assert own.calibration["fold_ece_inner"][0] == b.calibration["fold_ece_inner"][0]
    np.testing.assert_array_equal(own.fold_mean["test_proba_by_fold"][0],
                                  b.fold_mean["test_proba_by_fold"][0])

    assert not np.allclose(own.oof["proba"][rest], b.oof["proba"][rest])       # the checks
    for key in ("proba", "proba_calibrated"):                                 # can fail
        assert not np.allclose(others.oof[key][pos], b.oof[key][pos]), key
    assert others.calibration["fold_ece_inner"][0] != b.calibration["fold_ece_inner"][0]


def test_test_labels_reach_nothing_learned(base, moved_runs):
    b, moved = base.artifact, moved_runs["test"]
    for key in TRAIN_SIDE + ("fold_id",):
        np.testing.assert_array_equal(moved.oof[key], b.oof[key], err_msg=key)
    for key in ("test_proba", "test_proba_calibrated", "test_decision"):
        np.testing.assert_array_equal(moved.refit[key], b.refit[key], err_msg=key)
    np.testing.assert_array_equal(moved.fold_mean["test_proba"], b.fold_mean["test_proba"])
    assert moved.threshold["oof"] == b.threshold["oof"]
    assert moved.threshold["oof_calibrated_space"] == b.threshold["oof_calibrated_space"]
    def learned(calibration):                     # test_* entries are test-split evaluation
        return {k: v for k, v in calibration.items() if not k.startswith("test_")}
    assert learned(moved.calibration) == learned(b.calibration) and moved.best_params == b.best_params
    assert moved.block_fingerprint != b.block_fingerprint                     # the labels did move
    assert moved.metrics["test_at_operating_threshold"] != b.metrics["test_at_operating_threshold"]
    assert moved.calibration["test_ece_raw"] != b.calibration["test_ece_raw"]


# ----------------------------------------------------------------------
# artifact and provenance
# ----------------------------------------------------------------------
def test_saved_artifact_is_found_by_fingerprint_and_identifies_the_blackbox_arm(base):
    art, cfg = base.artifact, base.config
    path = Path(base.path)
    assert path.parent == Path(cfg.artifact_dir) and path.name.startswith("xgb_stage3_")
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert (side["split_fingerprint"], side["block_fingerprint"], side["features"]) == (
        art.split_fingerprint, art.block_fingerprint, list(EBM_FEATURES))
    assert side["split_fingerprint"] == router_split_fingerprint(base.split["X_train"],
                                                                 base.split["X_test"])
    assert side["model_family"] == "xgboost"
    proto = side["protocol"]
    assert (proto["arm"], proto["model_family"], proto["column_prefix"], proto["counterpart"]) == (
        "blackbox", "xgboost", "xgb_stage3_", "glass_pipeline.ebm")
    assert proto["search_space"] == json.loads(json.dumps(XGBSearchSpace().to_dict()))
    assert side["best_params"] == PARAMS

    found, found_path = find_stage3_artifact([path.parent], cfg.artifact_stem, art.split_fingerprint,
                                             art.block_fingerprint, list(EBM_FEATURES), verbose=False)
    assert found_path == str(path)
    for key in ("proba", "proba_calibrated", "fold_id", "decision", "threshold_per_row"):
        np.testing.assert_array_equal(found.oof[key], art.oof[key], err_msg=key)
    assert found.best_params == art.best_params and found.fold_plan == art.fold_plan
    assert XGBStage3Config.from_dict(found.config) == cfg
    assert found.calibration["fold_applied"] == art.calibration["fold_applied"]


def test_reloaded_artifact_scores_raw_rows_as_the_refit_did(base):
    art = Stage3ArtifactStore.load(base.path)
    raw, serving = base.split["X_test"], art.serving_model
    pd.testing.assert_frame_equal(serving.engineer(raw), base.block.X_test)
    np.testing.assert_array_equal(serving.predict_proba_raw(raw), art.refit["test_proba"])
    np.testing.assert_array_equal(serving.predict_proba_calibrated(raw), art.refit["test_proba_calibrated"])
    np.testing.assert_array_equal(art.refit["test_proba"], base.artifact.refit["test_proba"])
    applied = art.refit["model"].get_params()
    assert {k: applied[k] for k in PARAMS} == art.best_params

    # serving inputs that differ only in form score the same rows the same way
    expected = pd.Series(art.refit["test_proba"], index=raw.index)
    for variant in (raw.iloc[:, ::-1], raw.assign(duration=1, pdays=999, poutcome=0),
                    mut.shuffle_rows(raw), raw.iloc[[3]]):
        np.testing.assert_allclose(serving.predict_proba_raw(variant), expected.loc[variant.index],
                                   rtol=0, atol=1e-12)


def test_stage_output_carries_the_blackbox_xgboost_identity(base):
    art = base.artifact
    out = art.to_stage_output()
    assert (out.stage, out.arm, out.model) == ("stage3", "blackbox", "xgboost")
    assert out.feature_names == list(EBM_FEATURES)
    np.testing.assert_array_equal(out.train_proba_oof.to_numpy(), art.oof["proba_calibrated"])
    np.testing.assert_array_equal(out.test_proba.to_numpy(), art.refit["test_proba_calibrated"])
    assert out.train_proba_oof.index.equals(base.split["X_train"].index)
    assert out.test_proba.index.equals(base.split["X_test"].index)


def test_stage4_takes_the_xgboost_stream_for_its_own_split_only(base):
    art = base.artifact
    stream, checks = stage3_stream(base.path, "xgb", "XGBoost")
    assert (stream.notes["model_family"], stream.notes["arm"]) == ("xgboost", "blackbox")
    assert stream.notes["calibration_fold_applied"] == art.calibration["fold_applied"]
    np.testing.assert_array_equal(stream.train_proba, art.oof["proba_calibrated"])
    np.testing.assert_array_equal(stream.test_proba, art.refit["test_proba_calibrated"])
    np.testing.assert_array_equal(stream.train_fold_id, art.oof["fold_id"])
    inputs = build_stage4_inputs(notebook_split(), [stream], extra_checks=checks, verbose=False)
    assert inputs["xgb"].fingerprint_scheme == "router"     # Cell 30's _FP_NOW format

    other = split_frame(make_bank_frame(n=1000), test_size=0.20, seed=1)
    stream, checks = stage3_stream(base.path, "xgb", "XGBoost")
    with pytest.raises(ValidationError, match="split fingerprint"):
        build_stage4_inputs(other, [stream], extra_checks=checks, verbose=False)


def _corrupt(art, case):
    replace = dataclasses.replace
    if case == "pre_pr33_dict":
        return art.to_glass_schema()
    if case == "pre_f22":
        return replace(art, calibration={k: v for k, v in art.calibration.items() if k != "fold_applied"})
    if case == "no_gate":
        return replace(art, calibration={k: v for k, v in art.calibration.items() if k != "applied"})
    if case == "global_features":
        return replace(art, feature_fit_scope="global")
    if case == "in_sample_train":
        return replace(art, oof={**art.oof, "proba_calibrated": art.refit["train_proba_calibrated"]})
    if case == "foreign_fold_ids":
        return replace(art, oof={**art.oof, "fold_id": np.roll(art.oof["fold_id"], 1)})
    raise AssertionError(case)


@pytest.mark.parametrize("case, error, message", [
    ("pre_pr33_dict", Stage4InputError, "pre-PR-33"),
    ("pre_f22", Stage4InputError, "predates PR 36 F22"),
    ("no_gate", Stage4InputError, r"calibration\['applied'\] missing"),
    ("global_features", Stage4InputError, "feature_fit_scope='global'"),
    ("in_sample_train", ValidationError, "in-sample"),
    ("foreign_fold_ids", ValidationError, "oof fold_id = fold_plan fold_id"),
])
def test_corrupted_stage3_artifacts_are_refused_by_stage4(base, case, error, message):
    with pytest.raises(error, match=message):
        stream, checks = stage3_stream(_corrupt(base.artifact, case), "xgb", "XGBoost")
        build_stage4_inputs(notebook_split(), [stream], extra_checks=checks, verbose=False)


# ----------------------------------------------------------------------
# the black-box notebook's Stage 3 calls (read, never executed)
# ----------------------------------------------------------------------
def _notebook():
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    trees = [ast.parse("\n".join(line for line in src.splitlines()
                                 if not line.lstrip().startswith(("%", "!")))) for src in cells]
    consts = {}
    for tree in trees:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                consts[node.targets[0].id] = node.value
    calls = [n for tree in trees for n in ast.walk(tree) if isinstance(n, ast.Call)]

    def named(name):
        return [c for c in calls if (getattr(c.func, "id", None) or getattr(c.func, "attr", None)) == name]
    return named, consts


def _same(node, source: str) -> bool:
    """``node`` is the expression ``source`` (compared after normalising both)."""
    return ast.unparse(node) == ast.unparse(ast.parse(source, mode="eval").body)


def _keywords(call) -> dict:
    return {k.arg: k.value for k in call.keywords}


def test_the_notebook_runs_stage3_under_the_tested_protocol():
    named, consts = _notebook()
    assert ast.literal_eval(consts["XGB_TUNE_TRIALS"]) == XGBStage3Config().n_trials == 150
    (cfg_call,) = named("XGBStage3Config")
    kw = _keywords(cfg_call)
    assert set(kw) == {"expected_features", "n_trials", "artifact_dir", "verbose"}  # protocol defaults
    assert _same(kw["expected_features"], "list(EBM_FEATURES)") and _same(kw["n_trials"], "XGB_TUNE_TRIALS")

    (block_call,) = named("from_global_split")
    assert [ast.unparse(a) for a in block_call.args] == ["GLOBAL_SPLIT", "EBMFeaturePipeline",
                                                         "XGB_STAGE3_CONFIG"]
    assert _same(_keywords(block_call)["split_fingerprint"], "_FP_NOW")
    assert _same(consts["_FP_NOW"], 'split_fingerprint(GLOBAL_SPLIT["X_train"], GLOBAL_SPLIT["X_test"])')

    (train_call,) = named("train_xgb_stage")
    kw = _keywords(train_call)
    assert (ast.unparse(kw["block"]), ast.unparse(kw["config"]), ast.unparse(kw["params"])) == (
        "STAGE3_BLOCK", "XGB_STAGE3_CONFIG", "_params")
    assert "save" not in kw                                               # the artifact is saved
    (reuse,) = named("check_param_reuse")
    assert [ast.unparse(a) for a in reuse.args[1:]] == [
        "XGB_STAGE3_CONFIG", "_FP_NOW", "STAGE3_BLOCK.block_fingerprint", "list(EBM_FEATURES)"]


def test_the_notebook_imports_stage3_from_the_shared_protocol_only():
    """Cell 30's ``split_fingerprint`` is the router's (imported in Cell 18),
    and every XGBoost name comes from the package, which re-exports
    ``shared_protocol``; the legacy standalone modules (removed in Commit 4)
    are never imported."""
    imports = project_imports(NOTEBOOK)
    xgb = [i for i in imports if i.module.startswith("blackbox_pipeline.models.xgb")]
    assert {(i.module, i.name) for i in xgb} == {
        ("blackbox_pipeline.models.xgb", "XGBStage3Config"),
        ("blackbox_pipeline.models.xgb", "train_xgb_stage")}
    block_cell = next(i.cell for i in xgb if i.name == "XGBStage3Config")
    bound = [i for i in imports if i.name == "split_fingerprint" and i.cell <= block_cell]
    assert bound[-1].module == "blackbox_pipeline.models.rf_router"
