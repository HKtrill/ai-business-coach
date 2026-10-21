"""
Stage 3 artifact provenance (shared.stage3.artifact / protocol / runner).

Contract: artifacts round-trip through the store; lookup matches split, block
and feature contract and never falls back to a stale file; hyperparameters are
reused only from a compatible, actually-tuned artifact; the two mirrored arms
must have run the same protocol; a runner refuses a block built differently.
All files go to temporary directories.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from blackbox_pipeline.models.xgb.shared_protocol import XGBSearchSpace, XGBStage3Config
from shared.stage3.artifact import Stage3ArtifactStore, find_stage3_artifact
from shared.stage3.protocol import assert_matched_protocol, check_param_reuse
from shared.stage3.runner import Stage3Runner
from tests.stage3.conftest import ARMS, N_EVAL, N_TUNE, run_arm


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    """A real (2-trial) Optuna run, so the artifact carries a search record."""
    return run_arm(ARMS["xgboost"], tmp_path_factory.mktemp("tuned"), tune=True,
                   n_trials=2, pruner_startup_trials=1)


def _config(run, **changes):
    return XGBStage3Config(**{**{k: getattr(run.config, k) for k in (
        "n_tune_folds", "n_eval_folds", "n_jobs", "verbose", "show_progress_bar",
        "expected_features", "n_trials", "pruner_startup_trials")}, **changes})


# ----------------------------------------------------------------------
# store and lookup
# ----------------------------------------------------------------------
def test_store_round_trip_with_sidecar(arm_runs, tmp_path):
    art = arm_runs["xgboost"].artifact
    store = Stage3ArtifactStore(str(tmp_path), "xgb_stage3")
    path = store.save(art, verbose=False)
    side = json.loads(Path(path).with_suffix(".json").read_text())
    assert (side["split_fingerprint"], side["block_fingerprint"], side["features"]) == (
        art.split_fingerprint, art.block_fingerprint, art.features)
    for loaded in (Stage3ArtifactStore.load(path), store.load_latest()):
        np.testing.assert_array_equal(loaded.oof["proba"], art.oof["proba"])
        assert loaded.best_params == art.best_params
        assert loaded.threshold["oof"]["threshold"] == art.threshold["oof"]["threshold"]


def test_lookup_matches_split_block_and_features_and_prefers_the_newest(arm_runs, tmp_path):
    art = arm_runs["xgboost"].artifact
    store = Stage3ArtifactStore(str(tmp_path), "xgb_stage3", write_latest_pointer=False)
    store.save(art, verbose=False)
    newer = copy.copy(art)
    newer.created_at = "9999-12-31T23:59:59"
    newer_path = store.save(newer, verbose=False)
    found, path = find_stage3_artifact([tmp_path], "xgb_stage3", art.split_fingerprint,
                                       art.block_fingerprint, art.features, verbose=False)
    assert path == newer_path and found.split_fingerprint == art.split_fingerprint


@pytest.mark.parametrize("field", ["split", "block", "features"])
def test_lookup_never_falls_back_to_a_mismatched_artifact(arm_runs, tmp_path, field):
    art = arm_runs["xgboost"].artifact
    Stage3ArtifactStore(str(tmp_path), "xgb_stage3").save(art, verbose=False)
    wanted = {"split": art.split_fingerprint, "block": art.block_fingerprint,
              "features": list(art.features)}
    wanted[field] = {"split": "0" * 12, "block": "f" * 64, "features": art.features[::-1]}[field]
    with pytest.raises(FileNotFoundError, match=f"mismatch {field}"):
        find_stage3_artifact([tmp_path], "xgb_stage3", wanted["split"], wanted["block"],
                             wanted["features"], verbose=False)


def test_lookup_detects_a_sidecar_that_disagrees_with_its_payload(arm_runs, tmp_path):
    art = arm_runs["xgboost"].artifact
    path = Stage3ArtifactStore(str(tmp_path), "xgb_stage3", write_latest_pointer=False).save(
        art, verbose=False)
    side = path[:-len(".joblib")] + ".json"
    payload = json.loads(open(side).read())
    payload["split_fingerprint"] = "0" * 12
    open(side, "w").write(json.dumps(payload))
    with pytest.raises(ValueError, match="payload fingerprint differs from its sidecar"):
        find_stage3_artifact([tmp_path], "xgb_stage3", "0" * 12, verbose=False)


# ----------------------------------------------------------------------
# hyperparameter reuse
# ----------------------------------------------------------------------
def test_reuse_from_a_compatible_tuned_artifact(tuned):
    art = tuned.artifact
    out = check_param_reuse(art, _config(tuned), tuned.split_fp, art.block_fingerprint, art.features)
    assert out["params"] == art.best_params
    assert out["provenance"]["n_trials_requested"] == 2


@pytest.mark.parametrize("change, message", [
    ("split", "split"),
    ("block", "block fingerprint differs"),
    ("features", "feature contract differs"),
    ("n_trials", "n_trials 2 != 3"),
    ("search_space", "search space differs"),
    ("no_search", "no Optuna search on record"),
])
def test_reuse_is_refused_from_an_incompatible_artifact(tuned, change, message):
    art, cfg = tuned.artifact, _config(tuned)
    split_fp, block_fp, features = tuned.split_fp, art.block_fingerprint, list(art.features)
    if change == "split":
        split_fp = "0" * 12
    elif change == "block":
        block_fp = "f" * 64
    elif change == "features":
        features = features[::-1]
    elif change == "n_trials":
        cfg = _config(tuned, n_trials=3)
    elif change == "search_space":
        cfg = _config(tuned, search_space=XGBSearchSpace(max_depth=(2, 4)))
    else:
        art = copy.copy(art)
        art.tuning = {"tuned": False, "best_params": art.best_params, "provenance": None}
    with pytest.raises(ValueError, match=message):
        check_param_reuse(art, cfg, split_fp, block_fp, features)


# ----------------------------------------------------------------------
# mirrored-arm protocol and block/config agreement
# ----------------------------------------------------------------------
def test_mirrored_arms_ran_the_same_protocol(arm_runs):
    diffs = assert_matched_protocol(arm_runs["ebm"].artifact, arm_runs["xgboost"].artifact,
                                    verbose=False)
    assert {d[0] for d in diffs} <= {"search_space"}


@pytest.mark.parametrize("field", ["split_fingerprint", "fold_id", "features", "n_eval_folds"])
def test_protocol_mismatches_are_blocking(arm_runs, field):
    other = copy.deepcopy(arm_runs["xgboost"].artifact)
    if field == "split_fingerprint":
        other.split_fingerprint = "0" * 12
    elif field == "fold_id":
        other.fold_plan["fold_id"] = list(np.roll(other.fold_plan["fold_id"], 1))
    elif field == "features":
        other.features = other.features[::-1]
    else:
        other.config["n_eval_folds"] = N_EVAL + 1
    with pytest.raises(AssertionError, match=field):
        assert_matched_protocol(arm_runs["ebm"].artifact, other, verbose=False)


def test_runner_refuses_a_block_built_with_different_settings(arm_runs):
    run = arm_runs["xgboost"]
    cfg = _config(run, n_eval_folds=N_EVAL + 1)
    with pytest.raises(ValueError, match="built with different settings"):
        Stage3Runner(cfg, ARMS["xgboost"].make_estimator()).fit(run.block, params=dict(run.arm.params))
    assert run.block.settings["n_tune_folds"] == N_TUNE
