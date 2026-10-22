"""
Black-box Stage 4 end to end: ``train_meta_xgb_stage``
(blackbox_pipeline.models.meta_xgb) on black-box-format Stage 1-3 artifacts
saved to disk (tests.support.meta_xgb_world), with every meta-model fit and
score recorded.

Contract:

* hand-off: the six-column feature block has the contract's names, order,
  dtypes and every split row; training cells are the upstream OOF columns
  (MLP ``train_proba_oof``, router ``extra.train_proba``, XGBoost per-fold
  calibrated OOF; F28 when the gates disagree) and test cells the full-train
  outputs, never an in-sample column; the router's probability is missing
  exactly where it abstains and its values there change nothing; a stale,
  misaligned, mislabelled or foreign input is refused before any meta-model is
  fitted, and only a ``MetaXGBConfig`` is accepted (F37);
* meta-level OOF: every meta-training row is scored once by a meta-model that
  never saw it, inner fits stay inside their outer fold's training rows,
  nothing is fitted on test rows, one model is fitted on all training rows; the
  meta folds are the router's and XGBoost's 5-fold partition;
* isolation: a meta fold's own labels never reach its training-side meta
  outputs (F22), and the test labels and test inputs reach nothing learned or
  frozen;
* decisions: the stored test decisions and the training-side evaluation
  reproduce from the meta probabilities, the operating threshold (selected on
  meta OOF), the shared exact abstention search (F30) and the router-free
  decision rule, in the raw and the calibrated operating space; degenerate
  scores resolve deterministically;
* artifact: identity, split, rows, feature contract, upstream sources and
  folds are recorded; the sidecar is a JSON summary; the artifact reloads for
  its own split and schema only and, with the saved upstream artifacts,
  reproduces every test probability and decision; a rerun is identical.

Holds conditional on the upstream OOF columns and on fixed hyperparameters.
Not claimed: the upstream models that produced a meta-training row's features
were fitted on rows that include every other meta fold, so a meta fold's labels
reach the other folds' features (standard stacking; F39, documented), and
Optuna selection on the same folds (F24). Upstream arm / model family is
recorded and enforced: another arm's artifact is read only in a declared
2x2 crossover (F29; tests/parity/test_stage4_crossover.py). The estimator is
in test_meta_xgb_estimator.py; the shared runner
protocol in tests/stage3; the abstention rule in test_abstention_search.py.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import pytest

import blackbox_pipeline.models.meta_xgb.runner as runner_mod
from blackbox_pipeline.models.meta_xgb import (
    SCHEMA, STAGE4_FEATURES, MetaXGBEstimator, MetaXGBRunner, load_meta_xgb,
)
from blackbox_pipeline.models.meta_xgb.evaluation import abstention_metrics, classification_metrics
from blackbox_pipeline.models.xgb import XGBStage3Config
from shared.stage3.config import Stage3Config
from shared.stage3.data import PassthroughPipeline, Stage3Block
from shared.stage4 import ABSTENTION_MIN_COVERAGE, ABSTENTION_SEARCH, Stage4InputError, tune_min_confidence
from shared.stage_io import StageOutputError, fold_assignment
from shared.validation import ValidationError
from tests.stage3.conftest import folds_from
from tests.support import mutations as mut
from tests.support.checks import assert_never_fit_on, assert_oof_provenance
from tests.support.meta_xgb_world import (
    META_PARAMS, N_META_EVAL, blackbox_world, meta_config, run_meta, stage4_frames,
)
from tests.support.notebooks import BANK_ROOT
from tests.support.recorders import CallLog, RecordingStage3Estimator
from tests.support.stage4_world import make_stage3

NOTEBOOK = BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"
N_META = 5                                           # MetaXGBConfig.n_tune_folds
TRAIN_SIDE = ("proba", "proba_calibrated", "decision", "threshold_per_row")


@dataclasses.dataclass
class MetaRun:
    world: dict
    arts: dict
    paths: dict
    artifact: Any
    path: Optional[str]
    log: CallLog
    inputs: Any
    X_train: pd.DataFrame
    X_test: pd.DataFrame


def dump(arts, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, art in arts.items():
        paths[name] = str(out_dir / f"{name}.joblib")
        joblib.dump(art, paths[name])
    return paths


@contextlib.contextmanager
def recorded(log: CallLog, made: list):
    """``train_meta_xgb_stage`` as written, with every meta-model fit and
    score recorded (the models themselves are unchanged)."""
    real = runner_mod.MetaXGBEstimator

    def make(*args, **kwargs):
        est = real(*args, **kwargs)
        made.append(est)
        return RecordingStage3Estimator(est, log)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner_mod, "MetaXGBEstimator", make)
        yield


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    w, arts = blackbox_world()
    paths = dump(arts, tmp_path_factory.mktemp("upstream"))
    log, made = CallLog(), []
    with recorded(log, made):
        art, path = run_meta(w, paths, tmp_path_factory.mktemp("meta"), save=True)
    inputs, X_tr, X_te = stage4_frames(w, paths)
    return MetaRun(w, arts, paths, art, path, log, inputs, X_tr, X_te)


def _block(run: MetaRun, y_train=None) -> Stage3Block:
    y = pd.Series(run.inputs.y_train, index=run.inputs.index_train)
    y_te = pd.Series(run.inputs.y_test, index=run.inputs.index_test)
    with folds_from(y):
        return Stage3Block.build(
            run.X_train, y if y_train is None else y_train, run.X_test, y_te, PassthroughPipeline,
            n_tune_folds=N_META, n_eval_folds=N_META_EVAL, impute_missing=False,
            expected_features=list(STAGE4_FEATURES), verbose=False)


def _runner_fit(run: MetaRun, tmp_path, y_train=None):
    runner = MetaXGBRunner(meta_config(tmp_path), MetaXGBEstimator(n_jobs=1))
    return runner, runner.fit(_block(run, y_train), params=dict(META_PARAMS))


# ----------------------------------------------------------------------
# the Stage 1-3 -> Stage 4 hand-off
# ----------------------------------------------------------------------
def test_the_feature_block_has_the_contract_schema_on_every_split_row(base):
    GS = base.world["GS"]
    for X, index in ((base.X_train, GS["X_train"].index), (base.X_test, GS["X_test"].index)):
        assert list(X.columns) == list(STAGE4_FEATURES) and X.index.equals(index)
        assert {c: str(t) for c, t in X.dtypes.items()} == {
            "stage1_proba": "float64", "stage2_proba": "float64", "stage2_pass1": "int8",
            "stage2_pass2": "int8", "stage2_abstain": "int8", "stage3_proba": "float64"}
        route = X[["stage2_pass1", "stage2_pass2", "stage2_abstain"]]
        assert (route.sum(axis=1) == 1).all() and route.isin([0, 1]).all().all()
        assert X["stage2_proba"].isna().equals(X["stage2_abstain"] == 1)       # missing iff abstain
        assert X.drop(columns="stage2_proba").notna().all().all()
        p = X[["stage1_proba", "stage2_proba", "stage3_proba"]].to_numpy()
        p = p[~np.isnan(p)]
        assert ((p >= 0) & (p <= 1)).all()


def test_train_cells_are_upstream_oof_and_test_cells_full_train_outputs(base):
    so, rf, s3 = base.arts["mlp"]["stage_output"], base.arts["rf"]["extra"], base.arts["xgb"]
    X_tr, X_te = base.X_train, base.X_test
    for X, side, s3_col in ((X_tr, "train", s3.oof["proba_calibrated"]),
                            (X_te, "test", s3.refit["test_proba_calibrated"])):
        s1 = so["train_proba_oof"] if side == "train" else so["test_proba"]
        np.testing.assert_array_equal(X["stage1_proba"], s1.to_numpy())
        np.testing.assert_array_equal(X["stage3_proba"], s3_col)
        dec = np.asarray(rf[f"{side}_decisions"])
        covered = dec != "uncertain"
        np.testing.assert_array_equal(X["stage2_proba"].to_numpy()[covered],
                                      np.asarray(rf[f"{side}_proba"])[covered])
        for col, route in (("stage2_pass1", "pass1"), ("stage2_pass2", "pass2"),
                           ("stage2_abstain", "uncertain")):
            np.testing.assert_array_equal(X[col], (dec == route).astype(np.int8))
    # the in-sample columns the artifacts also carry are never consumed
    for col, insample in (("stage1_proba", base.arts["mlp"]["train_predictions"]),
                          ("stage3_proba", s3.refit["train_proba_calibrated"]),
                          ("stage3_proba", s3.oof["proba"])):            # calibrated, not raw OOF
        assert not np.allclose(X_tr[col], insample), col


def test_disagreeing_stage3_calibration_gates_are_consumed_per_fold_f28():
    """All-rows gate not breached (test side identity), every fold's own gate
    breached: training rows take the per-fold calibrated column, test rows the
    raw refit column, as the StageOutput pairing does."""
    w, arts = blackbox_world()
    s3 = arts["xgb"]
    assert all(s3.calibration["fold_applied"])
    arts["xgb"] = dataclasses.replace(
        s3, calibration={**s3.calibration, "applied": False},
        refit={**s3.refit, "test_proba_calibrated": s3.refit["test_proba"]})
    _, X_tr, X_te = stage4_frames(w, arts)
    np.testing.assert_array_equal(X_tr["stage3_proba"], s3.oof["proba_calibrated"])
    assert not np.allclose(X_tr["stage3_proba"], s3.oof["proba"])
    np.testing.assert_array_equal(X_te["stage3_proba"], s3.refit["test_proba"])


def test_the_routers_values_where_it_abstains_change_nothing(base, tmp_path):
    w, arts = blackbox_world()
    rf, rng = arts["rf"]["extra"], np.random.default_rng(5)
    for side in ("train", "test"):
        p = np.array(rf[f"{side}_proba"], dtype=float)
        abstain = np.asarray(rf[f"{side}_decisions"]) == "uncertain"
        p[abstain] = rng.random(abstain.sum())
        rf[f"{side}_proba"] = p
    art, _ = run_meta(w, arts, tmp_path)
    b = base.artifact
    for key in TRAIN_SIDE:
        np.testing.assert_array_equal(art.learner.oof[key], b.learner.oof[key], err_msg=key)
    for key in ("proba", "pred_with_abstain"):
        np.testing.assert_array_equal(art.test[key], b.test[key], err_msg=key)
    assert (art.operating, art.abstention["min_confidence"]) == (b.operating, b.abstention["min_confidence"])


def _tamper(arts, GS, case):
    so, rf, s3 = arts["mlp"]["stage_output"], arts["rf"]["extra"], arts["xgb"]
    replace = dataclasses.replace
    if case == "stage1_pre_nested_calibration":
        so["oof_provenance"] = "refittable (cv=10)"                       # pre-F16/F19
    elif case == "stage1_reordered_rows":
        so["train_proba_oof"] = mut.shuffle_rows(so["train_proba_oof"])
    elif case == "router_train_side_not_oof":
        rf["train_outputs_oof"] = False
    elif case == "router_unknown_route":
        rf["train_decisions"] = np.where(np.arange(len(rf["train_decisions"])) == 0, "pass3",
                                         rf["train_decisions"]).astype(object)
    elif case == "router_nan_on_a_covered_row":
        covered = np.flatnonzero(np.asarray(rf["train_decisions"]) != "uncertain")[0]
        rf["train_proba"] = np.where(np.arange(len(rf["train_proba"])) == covered, np.nan, rf["train_proba"])
    elif case == "router_duplicated_row":
        index = list(rf["index_train"])
        index[1] = index[0]
        rf["index_train"] = index
    elif case == "stage3_pre_f22":
        arts["xgb"] = replace(s3, calibration={k: v for k, v in s3.calibration.items()
                                               if k != "fold_applied"})
    elif case == "stage3_gate_contradicts_its_column":
        arts["xgb"] = replace(s3, calibration={**s3.calibration,
                                               "fold_applied": [False] + s3.calibration["fold_applied"][1:]})
    elif case == "stage3_other_split":
        arts["xgb"] = replace(s3, split_fingerprint="deadbeefdeadbeef")
    elif case == "stage3_other_fold_partition":
        rolled = np.roll(np.asarray(s3.oof["fold_id"]), 1)
        arts["xgb"] = replace(s3, oof={**s3.oof, "fold_id": rolled},
                              fold_plan={**s3.fold_plan, "fold_id": rolled})
    elif case == "stage3_in_sample_train_column":
        arts["xgb"] = replace(s3, oof={**s3.oof, "proba_calibrated": s3.refit["train_proba_calibrated"]})
    elif case == "stage3_short_train_column":
        arts["xgb"] = replace(s3, oof={**s3.oof, "proba_calibrated": s3.oof["proba_calibrated"][:-1]})
    elif case == "stage3_pre_pr33_dict":
        arts["xgb"] = s3.to_glass_schema()
    elif case == "labels_not_binary":
        y = GS["y_train"].copy()
        y.iloc[0] = 2
        GS = {**GS, "y_train": y}
    else:
        raise AssertionError(case)
    return arts, GS


@pytest.mark.parametrize("case, error, message", [
    ("stage1_pre_nested_calibration", ValidationError, "nested inside each fold"),
    ("stage1_reordered_rows", StageOutputError, "proba.index != y.index"),
    ("router_train_side_not_oof", Stage4InputError, "train_outputs_oof"),
    ("router_unknown_route", Stage4InputError, "unrecognised router decision"),
    ("router_nan_on_a_covered_row", ValidationError, "non-finite"),
    ("router_duplicated_row", ValidationError, "does not match GLOBAL_SPLIT"),
    ("stage3_pre_f22", Stage4InputError, "predates PR 36 F22"),
    ("stage3_gate_contradicts_its_column", ValidationError, "folds whose own gate was not breached"),
    ("stage3_other_split", ValidationError, "split fingerprint"),
    ("stage3_other_fold_partition", ValidationError, "identical 5-fold partition"),
    ("stage3_in_sample_train_column", ValidationError, "in-sample"),
    ("stage3_short_train_column", ValidationError, "train proba has len"),
    ("stage3_pre_pr33_dict", Stage4InputError, "pre-PR-33"),
    ("labels_not_binary", Stage4InputError, "not binary"),
])
def test_a_stale_misaligned_or_foreign_input_is_refused_before_any_fit(tmp_path, case, error, message):
    w, arts = blackbox_world()
    arts, GS = _tamper(arts, w["GS"], case)
    log, made = CallLog(), []
    out = tmp_path / "meta"
    with recorded(log, made), pytest.raises(error, match=message):
        run_meta(w, dump(arts, tmp_path / "upstream"), out, save=True, GS=GS)
    assert not made and not log and not out.exists()


@pytest.mark.parametrize("config", [
    Stage3Config(verbose=False), XGBStage3Config(verbose=False),
], ids=["generic_stage3_config", "xgboost_stage3_config"])
def test_only_a_meta_xgb_config_is_accepted_f37(tmp_path, monkeypatch, config):
    """Before PR 36 both loaded every input and fitted before failing (no
    ``operating_space``; the Stage 3 search space)."""
    loaded = []
    monkeypatch.setattr(runner_mod, "load_blackbox_stage4_inputs", lambda *a, **k: loaded.append(1))
    w, arts = blackbox_world()
    with pytest.raises(TypeError, match="must be a MetaXGBConfig"):
        run_meta(w, arts, tmp_path, config=config)
    assert not loaded


# ----------------------------------------------------------------------
# meta-level out-of-fold integrity
# ----------------------------------------------------------------------
def _plans(run: MetaRun):
    """Meta fold fits in run order: k-fold report, OOF, then the inner fits."""
    train = set(run.inputs.index_train)
    fold_fits = sorted(fid for fid, rows in run.log.fits().items() if rows < train)
    assert len(fold_fits) == N_META_EVAL + N_META + N_META * N_META
    inner = fold_fits[N_META_EVAL + N_META:]
    return {"report": fold_fits[:N_META_EVAL], "oof": fold_fits[N_META_EVAL:N_META_EVAL + N_META],
            "inner": [inner[k * N_META:(k + 1) * N_META] for k in range(N_META)]}


def test_every_meta_training_row_is_scored_once_by_a_meta_model_that_never_saw_it(base):
    index, plans = pd.Index(base.inputs.index_train), _plans(base)
    for name in ("report", "oof"):
        assert_oof_provenance(CallLog(c for c in base.log if c.fit_id in set(plans[name])), index)
    plan = base.artifact.learner.fold_plan["fold_id"]
    for k, fits in enumerate(plans["inner"]):
        tr, va = np.flatnonzero(np.asarray(plan) != k), np.flatnonzero(np.asarray(plan) == k)
        sub_log = CallLog(c for c in base.log if c.fit_id in set(fits))
        assert_oof_provenance(sub_log, index[tr])
        assert_never_fit_on(sub_log, index[va])
        assert not any(set(c.keys) & set(index[va]) for c in sub_log)    # nor scored


def test_one_meta_model_is_fitted_on_all_training_rows_and_none_on_test_rows(base):
    assert_never_fit_on(base.log, base.inputs.index_test)
    full = [fid for fid, rows in base.log.fits().items() if rows == set(base.inputs.index_train)]
    assert len(full) == 1
    test_calls = [c for c in base.log.calls(full[0], ["predict_proba"])
                  if set(c.keys) == set(base.inputs.index_test)]
    assert len(test_calls) == 1                                          # test scored once by it
    np.testing.assert_array_equal(
        MetaXGBEstimator().positive_proba(base.artifact.model, base.X_test), base.artifact.test["proba_raw"])


def test_the_meta_folds_are_the_routers_and_xgboosts_partition(base):
    art, y = base.artifact, base.inputs.y_train
    shared = fold_assignment(pd.Series(y, index=base.inputs.index_train), cv_folds=N_META,
                             random_state=42).to_numpy()
    np.testing.assert_array_equal(art.meta_oof_fold_id, shared)
    np.testing.assert_array_equal(base.arts["rf"]["extra"]["train_fold_id"], shared)
    np.testing.assert_array_equal(base.arts["xgb"].oof["fold_id"], shared)
    up = art.fold_provenance["upstream"]
    assert up["shared_partitions"] == [["rf_router", "xgb", 5]]
    assert {tuple(d["streams"]) for d in up["distinct_partitions"]} == {("mlp", "rf_router"), ("mlp", "xgb")}
    passed = {r["check"] for r in art.validation["meta"] if r["passed"]}
    assert {"meta folds = rf_router's 5-fold partition", "meta folds = xgb's 5-fold partition"} <= passed


def test_every_meta_honesty_check_passed_before_the_test_split_was_scored(base):
    art = base.artifact
    assert {k: all(r["passed"] for r in v) for k, v in art.validation.items()} == {
        "inputs": True, "features": True, "meta": True}
    ev = art.events
    assert ev.index("operating threshold selected on meta OOF") < ev.index("configuration frozen") \
        < ev.index("test scored (refit model, once)") == len(ev) - 1


# ----------------------------------------------------------------------
# label isolation (audit F22 at the meta level; test labels)
# ----------------------------------------------------------------------
def test_a_meta_folds_own_labels_never_reach_its_training_side_outputs_f22(base, tmp_path):
    """Upstream columns fixed, labels moved inside meta fold 0 ("own") or
    outside it ("others"). Fold 0's meta OOF scores, calibrated column,
    nested threshold, decisions and |p - t| abstention confidence come from
    meta-models and inner folds fitted on folds 1.. only. The abstention cut
    itself is one value chosen on every meta OOF row (reported in-selection,
    as the Meta-EBM's)."""
    y = pd.Series(base.inputs.y_train, index=base.inputs.index_train)
    b = base.artifact.learner                         # the same block, labels unmoved
    pos = np.flatnonzero(np.asarray(b.oof["fold_id"]) == 0)
    rest = np.setdiff1d(np.arange(len(y)), pos)
    runs = {k: _runner_fit(base, tmp_path, mut.permute_labels(y, seed=3, rows=rows))
            for k, rows in (("own", y.index[pos]), ("others", y.index[rest]))}
    own, others = runs["own"][1], runs["others"][1]

    def conf(a):
        return np.abs(np.asarray(a.oof["proba"]) - np.asarray(a.oof["threshold_per_row"]))
    for key in TRAIN_SIDE:
        np.testing.assert_array_equal(np.asarray(own.oof[key])[pos], np.asarray(b.oof[key])[pos], err_msg=key)
    np.testing.assert_array_equal(conf(own)[pos], conf(b)[pos])
    assert not np.allclose(np.asarray(own.oof["proba"])[rest], np.asarray(b.oof["proba"])[rest])
    assert not np.allclose(np.asarray(others.oof["proba"])[pos], np.asarray(b.oof["proba"])[pos])
    assert runs["own"][0].abstention_["tuned_on_n"] == len(y)


def test_test_labels_and_test_inputs_reach_nothing_learned(base, tmp_path, monkeypatch):
    real = runner_mod.load_blackbox_stage4_inputs

    def scrambled(*args, **kwargs):
        inputs = real(*args, **kwargs)
        rng = np.random.default_rng(7)
        inputs.y_test = rng.permutation(inputs.y_test)
        for s in inputs.streams.values():
            s.test_proba = rng.random(len(s.test_proba))
            if s.test_decisions is not None:
                s.test_decisions = rng.permutation(s.test_decisions)
        return inputs

    monkeypatch.setattr(runner_mod, "load_blackbox_stage4_inputs", scrambled)
    art, _ = run_meta(base.world, base.paths, tmp_path)
    b = base.artifact

    def learned(cal):                                 # test_* entries are test-split evaluation
        return {k: v for k, v in cal.items() if not k.startswith("test_")}
    for key in TRAIN_SIDE + ("fold_id",):
        np.testing.assert_array_equal(art.learner.oof[key], b.learner.oof[key], err_msg=key)
    assert art.best_params == b.best_params and art.operating == b.operating
    assert {k: v for k, v in art.abstention.items() if k != "sweep"} == \
        {k: v for k, v in b.abstention.items() if k != "sweep"}
    assert learned(art.learner.calibration) == learned(b.learner.calibration)
    assert art.train_oof_eval == b.train_oof_eval
    assert not np.allclose(art.test["proba"], b.test["proba"])             # the test side did move


# ----------------------------------------------------------------------
# decisions, calibration, thresholds and abstention
# ----------------------------------------------------------------------
def _check_test_decisions(art, y_test, space):
    L, op, cut = art.learner, art.operating, art.abstention["min_confidence"]
    key = "test_proba" if space == "raw" else "test_proba_calibrated"
    p = np.asarray(L.refit[key], dtype=float)
    np.testing.assert_array_equal(art.test["proba"], p)
    t = L.threshold["oof" if space == "raw" else "oof_calibrated_space"]["threshold"]
    assert (op["space"], op["threshold"]) == (space, t)
    pred, retained = (p >= t).astype(int), np.abs(p - t) >= cut
    np.testing.assert_array_equal(art.test["pred"], pred)
    np.testing.assert_array_equal(art.test["retained"], retained)
    np.testing.assert_array_equal(art.test["pred_with_abstain"], np.where(retained, pred, -1))
    assert art.test["no_abstain"] == classification_metrics(y_test, pred, p)
    assert art.test["with_abstain"] == abstention_metrics(y_test, pred, p, retained)


def test_stored_test_decisions_reproduce_from_the_frozen_configuration(base):
    art = base.artifact
    _check_test_decisions(art, base.inputs.y_test, "raw")
    sel = art.learner.threshold["oof"]
    assert sel["selected_on"] == "out-of-fold training predictions" and sel["n_rows"] == base.inputs.n_train
    assert art.operating["threshold_key"] == "oof" != "test_oracle"


def test_the_training_side_evaluation_and_the_abstention_cut_reproduce_from_meta_oof(base):
    art = base.artifact
    y = base.inputs.y_train
    p, t_rows = np.asarray(art.learner.oof["proba"]), np.asarray(art.learner.oof["threshold_per_row"])
    pred, conf = (p >= t_rows).astype(int), np.abs(p - t_rows)
    np.testing.assert_array_equal(art.learner.oof["decision"], pred)
    best = tune_min_confidence(y, pred, conf, None, ABSTENTION_MIN_COVERAGE)        # F30
    ab = art.abstention
    assert (ab["min_confidence"], ab["train_f2"], ab["train_coverage"]) == (
        best["min_confidence"], best["train_f2"], best["train_coverage"])
    assert ab["search"] == ABSTENTION_SEARCH and ab["tuned_on_n"] == len(y)
    assert ab["train_coverage"] >= ABSTENTION_MIN_COVERAGE
    assert art.train_oof_eval["no_abstain"] == classification_metrics(y, pred, p)
    assert art.train_oof_eval["with_abstain"] == abstention_metrics(y, pred, p, conf >= ab["min_confidence"])


def test_a_test_row_exactly_at_the_abstention_cut_is_decided(base, tmp_path, monkeypatch):
    """The shared rule abstains where confidence < cut: a test row whose
    |p - t| equals the cut is retained and decided, deterministically."""
    p, t = base.artifact.test["proba"], base.artifact.operating["threshold"]
    j = int(np.argsort(np.abs(p - t))[len(p) // 2])
    cut = float(np.abs(p[j] - t))
    real = runner_mod.tune_min_confidence
    monkeypatch.setattr(runner_mod, "tune_min_confidence",
                        lambda *a, **k: {**real(*a, **k), "min_confidence": cut})
    art, _ = run_meta(base.world, base.paths, tmp_path)
    np.testing.assert_array_equal(art.test["proba"], p)
    assert art.abstention["min_confidence"] == cut
    assert art.test["retained"][j] and art.test["pred_with_abstain"][j] == art.test["pred"][j]
    np.testing.assert_array_equal(art.test["retained"], np.abs(p - t) >= cut)


def test_the_calibrated_operating_space_decides_in_calibrated_space(base, tmp_path):
    art, _ = run_meta(base.world, base.paths, tmp_path,
                      config=meta_config(tmp_path, operating_space="calibrated"))
    _check_test_decisions(art, base.inputs.y_test, "calibrated")
    assert art.operating["threshold_key"] == "oof_calibrated_space"
    np.testing.assert_array_equal(art.learner.oof["proba"], base.artifact.learner.oof["proba"])


def test_degenerate_meta_scores_resolve_deterministically(base, tmp_path):
    """Every meta input constant: the meta probability is (near) constant, the
    threshold tie goes to the lowest grid value, the abstention cut to the
    smallest confidence (full coverage), and a rerun is identical."""
    X_tr = base.X_train.assign(stage1_proba=0.3, stage3_proba=0.3, stage2_proba=np.nan,
                               stage2_pass1=np.int8(0), stage2_pass2=np.int8(0), stage2_abstain=np.int8(1))
    X_te = base.X_test.assign(stage1_proba=0.3, stage3_proba=0.3, stage2_proba=np.nan,
                              stage2_pass1=np.int8(0), stage2_pass2=np.int8(0), stage2_abstain=np.int8(1))
    flat = dataclasses.replace(base, X_train=X_tr, X_test=X_te)
    outs = [_runner_fit(flat, tmp_path) for _ in range(2)]
    (r, a), (r2, a2) = outs
    cfg = meta_config(tmp_path)
    assert r.operating_["threshold"] == cfg.threshold_low
    assert r.abstention_["train_coverage"] == 1.0
    assert len(np.unique(a.refit["test_proba"])) == 1
    np.testing.assert_array_equal(a.oof["proba"], a2.oof["proba"])
    assert (r.operating_, r.abstention_["min_confidence"]) == (
        r2.operating_, r2.abstention_["min_confidence"])


# ----------------------------------------------------------------------
# artifact and provenance
# ----------------------------------------------------------------------
def test_the_artifact_identifies_meta_xgb_its_split_rows_and_upstream(base):
    art, GS = base.artifact, base.world["GS"]
    assert (art.schema, art.stage, art.model_family, art.arm) == (SCHEMA, "stage4", "meta_xgb", "blackbox")
    L = art.learner
    assert (L.stage, L.model_family, L.config["arm"], L.column_prefix) == (
        "stage4", "meta_xgb", "blackbox", "meta_xgb_")
    assert art.feature_names == list(STAGE4_FEATURES) == list(art.feature_contract["features"])
    assert art.feature_contract["router_proba_on_abstain"] == "missing"
    assert art.index_train == list(GS["X_train"].index) and art.index_test == list(GS["X_test"].index)
    assert art.split_fingerprints == {"stage_io": base.world["fp_io"], "router": base.world["fp_router"]}
    streams = art.stage4_inputs["streams"]
    assert {n: s["artifact_path"] for n, s in streams.items()} == {
        "mlp": base.paths["mlp"], "rf_router": base.paths["rf"], "xgb": base.paths["xgb"]}
    assert {n: (s["train_source"], s["test_source"]) for n, s in streams.items()} == {
        "mlp": ("stage_output.train_proba_oof", "stage_output.test_proba"),
        "rf_router": ("extra.train_proba", "extra.test_proba"),
        "xgb": ("Stage3Artifact.oof['proba_calibrated']", "Stage3Artifact.refit['test_proba_calibrated']")}
    assert {n: s["fingerprint_scheme"] for n, s in streams.items()} == {
        "mlp": "stage_io", "rf_router": "router", "xgb": "router"}
    assert all(s["train_is_oof"] and not s["standalone_threshold_used_by_stage4"] for s in streams.values())
    assert art.fold_provenance["meta"]["n_splits"] == N_META


def test_the_upstream_artifacts_declared_arm_and_family_are_recorded_and_enforced(base, tmp_path):
    """PR 36 F29: recorded, and enforced — a GLASS EBM artifact in the XGBoost
    slot of a black-box run used to be consumed, identifiable only by what it
    declared. Another arm's upstream is read only when declared (the 2x2
    crossover), and never mixed."""
    streams = base.artifact.stage4_inputs["streams"]
    assert (streams["mlp"]["arm"], streams["xgb"]["arm"], streams["xgb"]["model_family"]) == (
        "blackbox", "blackbox", "xgboost")
    up = base.artifact.stage4_inputs["upstream"]
    assert (up["arm"], up["arbiter"], up["crossover"]) == ("blackbox", "meta_xgb", False)
    w, arts = blackbox_world()
    arts["xgb"] = make_stage3(w, family="ebm")
    with pytest.raises(ValidationError, match="xgb: black-box XGBoost artifact"):
        run_meta(w, arts, tmp_path)


def test_the_sidecar_is_a_json_summary_without_arrays(base):
    side = json.loads(Path(base.path).with_suffix(".json").read_text(encoding="utf-8"))
    art = base.artifact
    assert (side["schema"], side["stage"], side["model_family"], side["arm"]) == (
        SCHEMA, "stage4", "meta_xgb", "blackbox")
    assert (side["n_train"], side["n_test"]) == (base.inputs.n_train, base.inputs.n_test)
    assert side["feature_names"] == list(STAGE4_FEATURES) and side["best_params"] == META_PARAMS
    assert side["operating"] == json.loads(json.dumps(art.operating))
    assert "sweep" not in side["abstention"] and side["abstention"]["min_confidence"] == \
        art.abstention["min_confidence"]
    assert side["validation_passed"] == {"inputs": True, "features": True, "meta": True}
    assert set(side["upstream"]) == {"mlp", "rf_router", "xgb"}


def test_the_reloaded_artifact_reproduces_every_decision_from_the_saved_upstream(base):
    art = load_meta_xgb(base.path, expected_split_fingerprint=base.world["fp_io"])
    _, _, X_te = stage4_frames(base.world, base.paths)                 # rebuilt from the saved files
    serving = art.learner.serving_model
    np.testing.assert_array_equal(serving.predict_proba_raw(X_te), art.test["proba_raw"])
    np.testing.assert_array_equal(serving.predict_proba_calibrated(X_te), art.test["proba_calibrated"])
    _check_test_decisions(art, base.inputs.y_test, "raw")


def test_an_artifact_is_reloaded_for_its_own_split_and_schema_only(base, tmp_path):
    assert load_meta_xgb(base.path, expected_split_fingerprint=base.world["fp_router"]).schema == SCHEMA
    with pytest.raises(ValueError, match="another split"):
        load_meta_xgb(base.path, expected_split_fingerprint="0" * 12)
    other = tmp_path / "stage3.joblib"
    joblib.dump(base.arts["xgb"], other)
    with pytest.raises(ValueError, match=f"not a {SCHEMA} artifact"):
        load_meta_xgb(str(other))


def test_a_rerun_from_the_saved_upstream_artifacts_is_identical(base, tmp_path):
    art, _ = run_meta(base.world, base.paths, tmp_path)
    b = base.artifact
    for key in TRAIN_SIDE + ("fold_id",):
        np.testing.assert_array_equal(art.learner.oof[key], b.learner.oof[key], err_msg=key)
    for key in ("proba", "proba_calibrated", "pred_with_abstain"):
        np.testing.assert_array_equal(art.test[key], b.test[key], err_msg=key)
    assert (art.operating, art.abstention["min_confidence"], art.best_params) == (
        b.operating, b.abstention["min_confidence"], b.best_params)


# ----------------------------------------------------------------------
# the black-box notebook's Stage 4 calls (read, never executed)
# ----------------------------------------------------------------------
def test_the_notebook_runs_meta_xgb_under_the_tested_protocol():
    """Cells 35-36: the default MetaXGBConfig (only its artifact folder set)
    on the three black-box artifacts, tuned (no ``params``), with the real
    fingerprint schemes (no ``extra_fingerprints``)."""
    trees = [ast.parse("\n".join(line for line in "".join(c["source"]).splitlines()
                                 if not line.lstrip().startswith(("%", "!"))))
             for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    consts = {n.targets[0].id: n.value for t in trees for n in ast.walk(t)
              if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    cfg = consts["META_XGB_CONFIG"]
    assert isinstance(cfg, ast.Call) and ast.unparse(cfg.func) == "MetaXGBConfig"
    assert [k.arg for k in cfg.keywords] == ["artifact_dir"] and not cfg.args
    calls = [n for t in trees for n in ast.walk(t) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "train_meta_xgb_stage"]
    main = [c for c in calls if any(k.arg == "config" and ast.unparse(k.value) == "META_XGB_CONFIG"
                                    for k in c.keywords)]
    assert len(main) == 1
    kw = {k.arg: ast.unparse(k.value) for k in main[0].keywords}
    assert [ast.unparse(a) for a in main[0].args] == ["GLOBAL_SPLIT"]
    assert kw == {"mlp_path": "MLP_S1_PATH", "rf_path": "RF_ROUTER_PATH", "xgb_path": "XGB_STAGE3_PATH",
                  "config": "META_XGB_CONFIG", "meta_ebm_path": "META_EBM_PATH"}
