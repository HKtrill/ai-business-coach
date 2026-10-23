"""
GLASS Stage 3 end to end: ``train_ebm_stage`` (glass_pipeline.ebm.ebm_stage)
with the real feature DAG, interaction pairs and EBM on the shared runner.

Contract: ``train_ebm_stage`` accepts only an ``EBMStage3Config`` with the
GLASS EBM identity, refusing anything else before a block is built or a model
fitted (F21); the factory is built from the block and the config, and the pairs
are recorded with the artifact; every fold frame (outer and inner) carries
the EBM block schema and its target-dependent features see only that fold's
training labels; a fold's own labels never reach any of its training-side
outputs (raw and calibrated OOF, calibration gate, nested threshold,
decisions; F22); test labels reach nothing built from the block; the saved
artifact is found by its fingerprints, identifies the GLASS arm, and after
reloading rebuilds the same features and probabilities from raw rows; a
cached block can be checked against its fingerprint. glass_cascade Cells 17-18
build and train Stage 3 the way it is tested here.

Protocol steps the arms share (OOF provenance, calibration, thresholds,
artifact store and lookup) are covered for every arm, on pre-engineered
features, in the other Stage 3 modules; this module re-checks only what the
GLASS feature DAG, pairs and orchestration could change.
"""

from __future__ import annotations

import ast
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import pytest

import glass_pipeline.ebm.ebm_stage as ebm_stage_mod
from blackbox_pipeline.models.xgb.shared_protocol import XGBStage3Config
from glass_pipeline.ebm.artifacts import save_ebm_artifacts
from glass_pipeline.ebm.calibration import calibrate_if_needed
from glass_pipeline.ebm.config import EBMStage3Config
from glass_pipeline.ebm.ebm_stage import make_ebm_factory, train_ebm_stage
from glass_pipeline.ebm.estimator import EBMSearchSpace
from glass_pipeline.ebm.evaluation import evaluate_ebm
from glass_pipeline.ebm.interactions import define_ebm_interactions
from glass_pipeline.ebm.tuning import tune_ebm
from shared.stage3.artifact import (
    STAGE4_FEATURE_SUFFIXES, Stage3ArtifactStore, find_stage3_artifact,
)
from shared.stage3.config import Stage3Config
from shared.stage3.data import FoldPlan, Stage3Block
from shared.stage3.feature_engineering import EBM_FEATURES, EBMFeaturePipeline
from shared.stage_io import split_fingerprint
from tests.stage3.conftest import (
    EBM_TEST_BAGS, EBM_TEST_PARAMS, N_EVAL, N_TUNE, folds_from, glass_ebm_config, stage3_split,
)
from tests.support import mutations as mut
from tests.support.notebooks import BANK_ROOT, project_imports

NOTEBOOK = BANK_ROOT / "glass_pipeline" / "glass_cascade.ipynb"
TARGET_DEPENDENT = ["decay_x_density", "economic_curvature_intensity", "dow_month_encoded"]


@contextlib.contextmanager
def cheap_bags():
    """``train_ebm_stage`` as written, with 2 outer bags per EBM (speed only)."""
    real = ebm_stage_mod.make_ebm_factory

    def make(config, block):
        factory = real(config, block)
        assert factory.extra_params == {}
        factory.extra_params = dict(EBM_TEST_BAGS)
        return factory

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ebm_stage_mod, "make_ebm_factory", make)
        yield


@dataclass
class GlassRun:
    split: dict
    config: Stage3Config
    block: Stage3Block
    artifact: Any
    path: Optional[str]


def build_block(config, y_train=None, y_test=None) -> Stage3Block:
    """The glass_cascade block (``EBMFeaturePipeline`` per fold), optionally
    with moved labels; folds always follow the unmoved training labels."""
    split = stage3_split()
    labels = split["y_train"]
    split = {**split, "y_train": labels if y_train is None else y_train,
             "y_test": split["y_test"] if y_test is None else y_test}
    fp = split_fingerprint(split["X_train"].index, split["X_test"].index,
                           split["y_train"].astype(int), split["y_test"].astype(int))
    with folds_from(labels):
        return Stage3Block.from_global_split(split, EBMFeaturePipeline, config, split_fingerprint=fp)


def run_stage(block: Stage3Block, config, save=False) -> GlassRun:
    with cheap_bags():
        artifact, path = train_ebm_stage(block=block, config=config, params=dict(EBM_TEST_PARAMS),
                                         params_source="test", save=save)
    raw = stage3_split()
    split = {"X_train": raw["X_train"], "X_test": raw["X_test"],
             "y_train": block.y_train, "y_test": block.y_test}
    return GlassRun(split, config, block, artifact, path)


def _pairs(model) -> set:
    names = list(model.feature_names_in_)
    return {frozenset(names[i] for i in term) for term in model.term_features_ if len(term) == 2}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    cfg = glass_ebm_config(tmp_path_factory.mktemp("glass_ebm"))
    return run_stage(build_block(cfg), cfg, save=True)


@pytest.fixture(scope="module")
def fold0(base):
    """Training rows of tuning fold 0 (positions and index labels)."""
    pos = base.block.tune_plan.holdout(0)
    return pos, base.block.index_train[pos]


@pytest.fixture(scope="module")
def moved_blocks(base, fold0):
    """Blocks whose labels moved inside tuning fold 0 ("own") or outside it ("others")."""
    y = base.block.y_train
    rows = fold0[1]
    return {
        "own": build_block(base.config, y_train=mut.permute_labels(y, seed=3, rows=rows)),
        "others": build_block(base.config, y_train=mut.permute_labels(
            y, seed=3, rows=y.index.difference(rows))),
    }


@pytest.fixture(scope="module")
def moved_runs(base, moved_blocks):
    return {k: run_stage(b, base.config).artifact for k, b in moved_blocks.items()}


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
def test_factory_is_built_from_the_block_and_the_config(base):
    pairs = define_ebm_interactions(base.block.X_train)
    factory = make_ebm_factory(base.config, base.block)
    assert len(pairs) == 3 and factory.interactions == pairs
    assert factory.search_space is base.config.search_space
    assert (factory.random_state, factory.n_jobs, factory.extra_params) == (
        base.config.random_state, base.config.n_jobs, {})

    art = base.artifact
    assert art.extras["interactions"] == [list(p) for p in pairs]
    assert art.refit["estimator"]["interactions"] == [list(p) for p in pairs]
    assert _pairs(art.refit["model"]) == {frozenset(p) for p in pairs}
    assert art.best_params == EBM_TEST_PARAMS and art.tuning["tuned"] is False


def test_global_split_entry_builds_the_notebook_block(base, monkeypatch):
    """``train_ebm_stage(GLOBAL_SPLIT)`` builds the same block glass_cascade
    builds with ``EBMFeaturePipeline``, stamped with the split fingerprint."""
    seen = {}

    class Runner:
        def __init__(self, config, estimator):
            seen.update(config=config, estimator=estimator)

        def fit(self, block, **kwargs):
            seen.update(block=block, **kwargs)
            return "artifact"

    monkeypatch.setattr(ebm_stage_mod, "Stage3Runner", Runner)
    out = train_ebm_stage(base.split, config=base.config,
                          split_fingerprint=base.block.split_fingerprint, save=False)
    assert out == ("artifact", None)
    block = seen["block"]
    assert block.block_fingerprint == base.block.block_fingerprint
    assert block.split_fingerprint == base.block.split_fingerprint
    assert block.settings["pipeline"] == "shared.stage3.feature_engineering.EBMFeaturePipeline"
    assert seen["config"] is base.config
    assert seen["extras"] == {"interactions": [list(p) for p in seen["estimator"].interactions]}


def test_default_config_enforces_the_ebm_feature_contract(monkeypatch):
    seen = {}

    class Block:
        @staticmethod
        def from_global_split(split, pipeline_factory, config, split_fingerprint=None):
            seen.update(pipeline_factory=pipeline_factory, config=config)
            raise RuntimeError("stop")

    monkeypatch.setattr(ebm_stage_mod, "Stage3Block", Block)
    with pytest.raises(RuntimeError, match="stop"):
        train_ebm_stage(stage3_split(), save=False)
    assert seen["pipeline_factory"] is EBMFeaturePipeline
    assert isinstance(seen["config"], EBMStage3Config)
    assert seen["config"].expected_features == list(EBM_FEATURES)


@pytest.mark.parametrize("case, error, message", [
    ("no_input", ValueError, "pass block="),
    ("other_split_fingerprint", ValueError, "split_fingerprint differs"),
    ("block_built_differently", ValueError, "built with different settings"),
    ("legacy_tune_ebm", NotImplementedError, "removed in PR 33"),
    ("legacy_calibrate_if_needed", NotImplementedError, "removed in PR 33"),
    ("legacy_dict_payload", TypeError, "expects a Stage3Artifact"),
    ("threshold_from_test_labels", ValueError, "explicit threshold"),
])
def test_misuse_is_refused_before_anything_is_fitted(base, tmp_path, case, error, message):
    cheap = dict(params=dict(EBM_TEST_PARAMS), save=False)            # if not refused: fit fast, fail
    with pytest.raises(error, match=message), cheap_bags():
        if case == "no_input":
            train_ebm_stage(**cheap)
        elif case == "other_split_fingerprint":
            train_ebm_stage(block=base.block, config=base.config, split_fingerprint="0" * 12, **cheap)
        elif case == "block_built_differently":
            train_ebm_stage(block=base.block, config=glass_ebm_config(tmp_path, n_eval_folds=N_EVAL + 1),
                            **cheap)
        elif case == "legacy_tune_ebm":
            tune_ebm()
        elif case == "legacy_calibrate_if_needed":
            calibrate_if_needed()
        elif case == "legacy_dict_payload":
            save_ebm_artifacts({"model": None}, base_path=str(tmp_path))
        else:
            evaluate_ebm(base.artifact.refit["model"], base.block.X_test, base.block.y_test)
    assert not list(tmp_path.iterdir())                               # nothing written


# ----------------------------------------------------------------------
# the feature block
# ----------------------------------------------------------------------
def _plans(block):
    """(plan, its frames, has test frames, rows it partitions) for every fold set."""
    out = [(block.tune_plan, block.tune_folds, True, None),
           (block.report_plan, block.report_folds, False, None)]
    for k, (tr, _) in enumerate(block.tune_plan):
        inner_plan = FoldPlan.build(block.y_train.iloc[tr], N_TUNE, block.settings["random_state"])
        out.append((inner_plan, block.inner_folds[k], False, tr))
    return out


def test_every_fold_frame_has_the_ebm_block_schema(base):
    """Outer, reporting and inner (F22) folds: same columns, order and dtypes."""
    block = base.block
    assert block.features == list(EBM_FEATURES)
    assert block.X_train.index.equals(base.split["X_train"].index)
    assert block.pipeline.n_fit_rows_ == len(block.index_train)      # serving pipeline: train only
    assert len(block.inner_folds) == N_TUNE and block.settings["n_inner_folds"] == N_TUNE
    dtypes = block.X_train.dtypes
    for plan, folds, with_test, rows in _plans(block):
        assert [f.k for f in folds] == list(range(plan.n_splits))
        for f, (tr, va) in zip(folds, plan):
            if rows is not None:
                tr, va = rows[tr], rows[va]
            frames = [f.X_tr, f.X_val] + ([f.X_test] if with_test else [])
            assert (f.X_test is not None) == with_test
            for X in frames:
                assert list(X.columns) == list(EBM_FEATURES) and X.dtypes.equals(dtypes)
                assert np.isfinite(X.to_numpy(dtype=float)).all()
            assert f.X_tr.index.equals(block.index_train[tr])
            assert f.X_val.index.equals(block.index_train[va])
            if with_test:
                assert f.X_test.index.equals(block.index_test)


def test_fold_features_see_only_that_folds_training_labels(base, moved_blocks):
    """Fold 0's pipeline is fitted on the other folds' rows and labels: its
    training, validation and test frames ignore fold 0's labels and follow
    everyone else's."""
    f, own, others = (b.tune_folds[0] for b in (base.block, moved_blocks["own"],
                                                 moved_blocks["others"]))
    for name in ("X_tr", "X_val", "X_test"):
        pd.testing.assert_frame_equal(getattr(own, name), getattr(f, name))
    for a, b in zip(moved_blocks["own"].inner_folds[0], base.block.inner_folds[0]):   # F22
        np.testing.assert_array_equal(a.val_idx, b.val_idx)
        pd.testing.assert_frame_equal(a.X_tr, b.X_tr)
        pd.testing.assert_frame_equal(a.X_val, b.X_val)
    changed = [c for c in EBM_FEATURES if not np.allclose(others.X_val[c], f.X_val[c])]
    assert changed == TARGET_DEPENDENT                                # the check can fail


def test_test_labels_reach_nothing_built_from_the_block(base):
    moved = build_block(base.config, y_test=mut.permute_labels(base.block.y_test, seed=11))
    assert not moved.y_test.equals(base.block.y_test)
    pd.testing.assert_frame_equal(moved.X_train, base.block.X_train)
    pd.testing.assert_frame_equal(moved.X_test, base.block.X_test)
    np.testing.assert_array_equal(moved.weights.vector, base.block.weights.vector)
    def every_fold(blk):
        return blk.tune_folds + blk.report_folds + [f for inner in blk.inner_folds for f in inner]

    for a, b in zip(every_fold(moved), every_fold(base.block)):
        for name in ("X_tr", "X_val", "X_test"):
            if getattr(b, name) is not None:
                pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))


# ----------------------------------------------------------------------
# OOF and calibration on GLASS features
# ----------------------------------------------------------------------
def test_stage4_frames_are_keyed_by_the_raw_split_index(base):
    art, split = base.artifact, base.split
    train, meta = art.stage4_frame(), art.stage4_metadata()
    assert train.index.equals(split["X_train"].index) and meta.index.equals(train.index)
    assert list(train.columns) == [f"ebm_stage3_{s}" for s in STAGE4_FEATURE_SUFFIXES]
    plan = FoldPlan.build(split["y_train"], N_TUNE, base.config.random_state)
    np.testing.assert_array_equal(meta["ebm_stage3_fold_id"], plan.fold_id)
    assert art.stage4_test_frame().index.equals(split["X_test"].index)
    assert np.isfinite(train.to_numpy(dtype=float)).all()


TRAIN_SIDE = ("proba", "proba_calibrated", "decision", "margin", "confidence", "threshold_per_row")


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


# ----------------------------------------------------------------------
# artifact and provenance
# ----------------------------------------------------------------------
def test_saved_artifact_is_found_by_fingerprint_and_identifies_the_glass_arm(base):
    art, cfg = base.artifact, base.config
    path = Path(base.path)
    assert path.parent == Path(cfg.artifact_dir) and path.name.startswith("ebm_stage3_")
    side = json.loads(path.with_suffix(".json").read_text())
    assert (side["split_fingerprint"], side["block_fingerprint"], side["features"]) == (
        art.split_fingerprint, art.block_fingerprint, list(EBM_FEATURES))
    assert side["model_family"] == "ebm" and side["protocol"]["arm"] == "glass"
    assert side["protocol"]["search_space"] == EBMSearchSpace().to_dict()
    assert side["extras"]["interactions"] == art.extras["interactions"]

    found, found_path = find_stage3_artifact([path.parent], cfg.artifact_stem, art.split_fingerprint,
                                             art.block_fingerprint, list(EBM_FEATURES), verbose=False)
    assert found_path == str(path)
    for key in ("proba", "proba_calibrated", "fold_id", "decision", "threshold_per_row"):
        np.testing.assert_array_equal(found.oof[key], art.oof[key], err_msg=key)
    assert {k: v for k, v in found.threshold.items() if k != "sweep"} == \
        {k: v for k, v in art.threshold.items() if k != "sweep"}
    pd.testing.assert_frame_equal(found.threshold["sweep"], art.threshold["sweep"])
    assert found.best_params == art.best_params
    assert found.fold_plan == art.fold_plan
    assert EBMStage3Config.from_dict(found.config) == cfg


def test_reloaded_artifact_rebuilds_features_and_scores_from_raw_rows(base):
    art = Stage3ArtifactStore.load(base.path)
    raw, serving = base.split["X_test"], art.serving_model
    pd.testing.assert_frame_equal(serving.engineer(raw), base.block.X_test)
    np.testing.assert_array_equal(serving.predict_proba_raw(raw), art.refit["test_proba"])
    np.testing.assert_array_equal(serving.predict_proba_calibrated(raw), art.refit["test_proba_calibrated"])
    assert _pairs(art.refit["model"]) == {frozenset(p) for p in art.extras["interactions"]}

    # serving inputs that differ only in form score the same rows the same way
    expected = pd.Series(art.refit["test_proba"], index=raw.index)
    shuffled = mut.shuffle_rows(raw)
    for variant in (raw.iloc[:, ::-1], raw.assign(duration=1, pdays=999, poutcome=0),
                    shuffled, raw.iloc[[3]]):
        np.testing.assert_allclose(serving.predict_proba_raw(variant), expected.loc[variant.index],
                                   rtol=0, atol=1e-12)

    legacy = art.to_glass_schema()
    np.testing.assert_array_equal(legacy["calibrated_model"].predict_proba(base.block.X_test)[:, 1],
                                  art.refit["test_proba_calibrated"])
    assert legacy["interactions"] == art.extras["interactions"]
    assert legacy["optimal_threshold"] == art.threshold["oof"]["threshold"]


def test_cached_block_can_be_checked_against_its_fingerprint(base, tmp_path):
    """glass_cascade caches the block with joblib; the reloaded block
    recomputes the same fingerprint (and a changed frame would not)."""
    path = tmp_path / "block.joblib"
    joblib.dump(base.block, path)
    cached = joblib.load(path)
    assert cached.compute_fingerprint() == base.block.block_fingerprint
    cached.check_config(base.config)
    raw = base.split["X_test"]
    pd.testing.assert_frame_equal(
        cached.cleaner.transform(cached.pipeline.transform(raw)[cached.features]), base.block.X_test)
    cached.tune_folds[0].X_val.iloc[0, 0] += 1.0
    assert cached.compute_fingerprint() != base.block.block_fingerprint   # the check can fail


def _misconfigured(case, tmp_path):
    small = {"n_tune_folds": N_TUNE, "n_eval_folds": N_EVAL, "verbose": False,
             "show_progress_bar": False, "artifact_dir": str(tmp_path),
             "expected_features": list(EBM_FEATURES)}
    return {
        "generic_config": lambda: Stage3Config(**small),        # glass_cascade before F21: 'unset'
        "xgboost_config": lambda: XGBStage3Config(**small),     # would be labelled 'xgboost' / 'blackbox'
        "family_overridden": lambda: glass_ebm_config(tmp_path, model_family="xgboost"),
        "arm_overridden": lambda: glass_ebm_config(tmp_path, arm="blackbox"),
    }[case]()


@pytest.mark.parametrize("entry", ["block", "GLOBAL_SPLIT"])
@pytest.mark.parametrize("case, error, message", [
    ("generic_config", TypeError, "must be an EBMStage3Config, got Stage3Config"),
    ("xgboost_config", TypeError, "must be an EBMStage3Config, got XGBStage3Config"),
    ("family_overridden", ValueError, "model_family='xgboost'"),
    ("arm_overridden", ValueError, "arm='blackbox'"),
])
def test_a_config_without_the_glass_ebm_identity_is_refused_f21(base, tmp_path, monkeypatch, entry,
                                                                case, error, message):
    """Before PR 36 each of these trained an EBM and labelled the artifact with
    the config's identity; the generic config, which glass_cascade passed,
    labelled it 'unset' with no search space (F21). Now nothing is built,
    fitted or saved, from either entry point."""
    cfg = _misconfigured(case, tmp_path)
    seen = []

    class Block:
        @staticmethod
        def from_global_split(*args, **kwargs):
            seen.append("block built")

    monkeypatch.setattr(ebm_stage_mod, "Stage3Block", Block)
    monkeypatch.setattr(ebm_stage_mod, "Stage3Runner", lambda *a, **k: seen.append("runner built"))
    source = {"block": {"block": base.block}, "GLOBAL_SPLIT": {"GLOBAL_SPLIT": base.split}}[entry]
    with pytest.raises(error, match=message):
        train_ebm_stage(**source, config=cfg, params=dict(EBM_TEST_PARAMS), params_source="test")
    assert not seen and not list(tmp_path.iterdir())


# ----------------------------------------------------------------------
# the GLASS notebook's Stage 3 calls (read, never executed)
# ----------------------------------------------------------------------
def _notebook_cells() -> list:
    """(cell index, source, parsed code) for every code cell; indexes count all
    cells, as ``project_imports`` does. IPython magics and shell lines are dropped."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    out = []
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            out.append((i, src, ast.parse("\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith(("%", "!"))))))
    return out


def _calls(trees, name: str) -> list:
    return [n for t in trees for n in ast.walk(t) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == name]


def _same(node, source: str) -> bool:
    """``node`` is the expression ``source`` (compared after normalising both)."""
    return ast.unparse(node) == ast.unparse(ast.parse(source, mode="eval").body)


def test_the_notebook_builds_and_trains_stage3_with_the_glass_ebm_config_f21():
    """Cell 17 builds ``EBMStage3Config`` (protocol defaults; only the feature
    contract and verbosity are set) and gets the block with it from the
    validated cache (``load_or_build_block``, F23; tests/stage3/
    test_block_cache.py), and Cell 18 trains with it, so the artifact carries
    the GLASS EBM identity and search space and is saved where
    blackbox_cascade Cell 33 looks for it (the config's artifact_dir and
    stem). The generic ``Stage3Config`` appears nowhere."""
    cells = _notebook_cells()
    stage3 = [(i, t) for i, src, t in cells if "# CELL 17:" in src or "# CELL 18:" in src]
    assert len(stage3) == 2, "glass_cascade Stage 3 cells (17, 18) not found"
    trees = [t for _, t in stage3]
    assigned = {n.targets[0].id: n.value for t in trees for n in ast.walk(t)
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}

    (cfg_call,) = _calls(trees, "EBMStage3Config")
    assert assigned["EBM_STAGE3_CONFIG"] is cfg_call
    kw = {k.arg: k.value for k in cfg_call.keywords}
    assert set(kw) == {"expected_features", "n_trials", "artifact_dir", "verbose"}
    assert _same(kw["expected_features"], "list(EBM_FEATURES)")
    notebook_cfg = EBMStage3Config(expected_features=list(EBM_FEATURES), verbose=False,
                                   n_trials=ast.literal_eval(kw["n_trials"]),
                                   artifact_dir=ast.literal_eval(kw["artifact_dir"]))
    default = EBMStage3Config(expected_features=list(EBM_FEATURES), verbose=False)
    assert notebook_cfg == default                                    # protocol defaults
    assert (default.model_family, default.arm, default.artifact_stem) == ("ebm", "glass", "ebm_stage3")

    # the block comes from the validated cache (F23), built with the same config
    (block_call,) = _calls(trees, "load_or_build_block")
    assert [ast.unparse(a) for a in block_call.args] == ["GLOBAL_SPLIT", "EBMFeaturePipeline",
                                                         "EBM_STAGE3_CONFIG"]
    block_kw = {k.arg: k.value for k in block_call.keywords}
    assert _same(block_kw["split_fingerprint"], "_FP_NOW") and _same(block_kw["cache_path"], "_BLOCK_CACHE")
    (assign,) = [n for t in trees for n in ast.walk(t) if isinstance(n, ast.Assign) and n.value is block_call]
    assert ast.unparse(assign.targets[0]) == "(STAGE3_BLOCK, _block_status)"
    assert not _calls([t for _, _, t in cells], "from_global_split")       # no uncached path
    assert not [c for _, _, t in cells for c in _calls([t], "load") + _calls([t], "dump")
                if "BLOCK" in ast.unparse(c)]                               # no raw joblib cache
    (train_call,) = _calls(trees, "train_ebm_stage")
    assert not train_call.args and {k.arg: ast.unparse(k.value) for k in train_call.keywords} == {
        "block": "STAGE3_BLOCK", "config": "EBM_STAGE3_CONFIG"}           # saved (save=True)

    imports = project_imports(NOTEBOOK)
    assert ("glass_pipeline.ebm", "EBMStage3Config") in {(i.module, i.name) for i in imports}
    assert not [i for i in imports if i.name == "Stage3Config"]
    assert not _calls([t for _, _, t in cells], "Stage3Config")

    # the block is stamped with the label-aware stage_io fingerprint (Cell 33 tries it)
    assert _same(assigned["_FP_NOW"], 'split_fingerprint(GLOBAL_SPLIT["X_train"].index, '
                 'GLOBAL_SPLIT["X_test"].index, GLOBAL_SPLIT["y_train"], GLOBAL_SPLIT["y_test"])')
    last_stage3_cell = max(i for i, _ in stage3)
    bound = [i for i in imports if i.name == "split_fingerprint" and i.cell <= last_stage3_cell]
    assert bound[-1].module == "shared"
