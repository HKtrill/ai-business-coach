"""
F23 — the GLASS notebook's cached Stage 3 block (shared.stage3.block_cache;
glass_cascade Cell 17), on the real feature DAG at small fold counts.

Contract: a cache is reused — nothing is built — only when its recorded
identity equals the current one (raw data, split fingerprint, the config's
build settings, the pipeline, the source of every module the block's
construction imports, library versions, cache schema) and the cached block
still passes its own contracts (inner folds, content fingerprint, settings,
split fingerprint, rows). Anything else — malformed metadata and a cached
block too malformed to check included — is rebuilt with
``Stage3Block.from_global_split``, the cache is replaced atomically with the
new identity, and the next call reuses it; an error from that build itself
propagates. A rebuilt block is the block a direct build makes.

The cache does not depend on which pipeline builds the block, so the reuse /
rebuild matrix runs a cheap counting pipeline; the real feature DAG
(``EBMFeaturePipeline``) is used where it matters — a rebuilt block equals a
direct build, and the implementation identity follows the DAG's imports. The
block's own guarantees (fold isolation, schema, fingerprint sensitivity) are
tested in test_glass_ebm_stage.py and the other Stage 3 modules.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

import shared.stage3.block_cache as block_cache
from glass_pipeline.ebm.config import EBMStage3Config
from shared.stage3.block_cache import (
    BLOCK_CACHE_SCHEMA, block_identity, implementation_identity, load_or_build_block,
)
from shared.stage3.data import Stage3Block
from shared.stage3.feature_engineering import EBM_FEATURES, EBMFeaturePipeline
from tests.stage3.conftest import N_EVAL, N_TUNE, stage3_split

FP = "0123456789ab"
FEATURES = ["age", "campaign", "euribor3m", "nr.employed"]
PIPELINES_PER_BUILD = 1 + N_TUNE + N_EVAL + N_TUNE * N_TUNE     # full train, folds, inner folds


class CountingPipeline:
    """A cheap feature pipeline that counts its instances: 0 new ones = nothing built."""

    made = 0

    def __init__(self):
        CountingPipeline.made += 1

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X[FEATURES].astype(float)


class OtherPipeline(CountingPipeline):
    """Same features, another pipeline (factory) identity."""


def config(**overrides) -> EBMStage3Config:
    return EBMStage3Config(**{"expected_features": list(FEATURES), "n_tune_folds": N_TUNE,
                              "n_eval_folds": N_EVAL, "verbose": False, "show_progress_bar": False,
                              **overrides})


def run(split, path, cfg=None, factory=CountingPipeline, fp=FP):
    """``(block, status, pipelines made)`` for one notebook-style call."""
    before = CountingPipeline.made
    block, status = load_or_build_block(split, factory, cfg or config(), split_fingerprint=fp,
                                        cache_path=path)
    return block, status, CountingPipeline.made - before


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A valid cache of the default block (built once; tests copy it)."""
    split = stage3_split()
    path = tmp_path_factory.mktemp("cache") / "stage3_block_default.joblib"
    block, status, made = run(split, path)
    assert status == "built (no cache)" and made == PIPELINES_PER_BUILD
    return split, path, block


@pytest.fixture
def cache(built, tmp_path):
    """A fresh copy of the valid cache, in its own folder."""
    split, path, block = built
    target = tmp_path / "models" / "stage3" / path.name
    target.parent.mkdir(parents=True)
    shutil.copy(path, target)
    return split, target, block


def assert_valid_rebuild(block, status, made, split, path, cfg, factory, fp, why):
    assert status.startswith("rebuilt (") and why in status, status
    assert made > 0
    assert block.require_inner_folds() and block.compute_fingerprint() == block.block_fingerprint
    assert block.features == list(cfg.expected_features or FEATURES)
    assert block.split_fingerprint == fp
    block.check_config(cfg)
    payload = joblib.load(path)                                      # saved with current provenance
    assert payload["identity"] == block_identity(split, factory, cfg, fp)
    assert payload["block"].block_fingerprint == block.block_fingerprint
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]   # no partial files
    again, status, made = run(split, path, cfg, factory, fp)
    assert (status, made, again.block_fingerprint) == ("reused", 0, block.block_fingerprint)


# ----------------------------------------------------------------------
# reuse
# ----------------------------------------------------------------------
def test_a_valid_cache_is_reused_without_building_anything(cache):
    split, path, block = cache
    stamp = path.stat().st_mtime_ns
    again, status, made = run(split, path)
    assert (status, made) == ("reused", 0)
    assert again.block_fingerprint == block.block_fingerprint == again.compute_fingerprint()
    assert again.features == FEATURES and len(again.inner_folds) == N_TUNE
    pd.testing.assert_frame_equal(again.X_train, block.X_train)
    assert path.stat().st_mtime_ns == stamp                          # not rewritten
    payload = joblib.load(path)
    assert payload["identity"]["schema"] == BLOCK_CACHE_SCHEMA


def test_a_rebuilt_block_is_the_block_a_direct_build_makes(tmp_path):
    """On the real feature DAG: schema, feature order, fingerprint and frames."""
    split = stage3_split()
    cfg = config(expected_features=list(EBM_FEATURES))
    joblib.dump("not a cache", tmp_path / "stale.joblib")
    block, status = load_or_build_block(split, EBMFeaturePipeline, cfg, split_fingerprint=FP,
                                        cache_path=tmp_path / "stale.joblib")
    direct = Stage3Block.from_global_split(split, EBMFeaturePipeline, cfg, split_fingerprint=FP)
    assert status.startswith("rebuilt (no cache metadata")
    assert block.block_fingerprint == direct.block_fingerprint
    assert block.features == direct.features == list(EBM_FEATURES)
    assert block.settings == direct.settings
    pd.testing.assert_frame_equal(block.X_train, direct.X_train)
    pd.testing.assert_frame_equal(block.inner_folds[0][0].X_tr, direct.inner_folds[0][0].X_tr)


# ----------------------------------------------------------------------
# a change in what the block is built from
# ----------------------------------------------------------------------
def _changed(case, split, monkeypatch):
    """(split, config, factory, fingerprint, expected reason) after ``case``."""
    cfg, factory, fp = config(), CountingPipeline, FP
    split = {k: v.copy() for k, v in split.items()}
    if case == "a raw training value":
        split["X_train"].iloc[0, split["X_train"].columns.get_loc("age")] += 1
        why = "changed: data"
    elif case == "a raw test value":
        split["X_test"].iloc[0, split["X_test"].columns.get_loc("age")] += 1
        why = "changed: data"
    elif case == "a training label":
        split["y_train"].iloc[0] = 1 - split["y_train"].iloc[0]
        why = "changed: data"
    elif case == "the split fingerprint":
        fp, why = "ba9876543210", "changed: split"
    elif case == "the tuning fold count":
        cfg, why = config(n_tune_folds=N_TUNE + 1), "changed: build"
    elif case == "the fold seed":
        cfg, why = config(random_state=7), "changed: build"
    elif case == "the feature fit scope":
        cfg, why = config(feature_fit_scope="global"), "changed: build"
    elif case == "the expected features":
        cfg, why = config(expected_features=None), "changed: build"
    elif case == "the pipeline":
        factory, why = OtherPipeline, "changed: pipeline"
    elif case == "a library version":
        real = block_cache._libraries
        monkeypatch.setattr(block_cache, "_libraries", lambda: {**real(), "pandas": "0.0"})
        why = "changed: libraries"
    else:                                                            # a feature-engineering source
        real = block_cache._module_digest
        monkeypatch.setattr(block_cache, "_module_digest", lambda name, source: (
            "edited" if name == "feature_research.config" else real(name, source)))
        why = "changed: implementation (feature_research.config)"
    return split, cfg, factory, fp, why


@pytest.mark.parametrize("case", [
    "a raw training value", "a raw test value", "a training label", "the split fingerprint",
    "the tuning fold count", "the fold seed", "the feature fit scope", "the expected features",
    "the pipeline", "a library version", "a feature-engineering source",
])
def test_a_change_in_what_the_block_is_built_from_rebuilds_it(cache, monkeypatch, case):
    split, path, _ = cache
    split, cfg, factory, fp, why = _changed(case, split, monkeypatch)
    block, status, made = run(split, path, cfg, factory, fp)
    assert_valid_rebuild(block, status, made, split, path, cfg, factory, fp, why)


# ----------------------------------------------------------------------
# missing, malformed or incompatible metadata, and failed block contracts
# ----------------------------------------------------------------------
def _corrupt(case, path, split):
    payload = joblib.load(path)
    block = payload["block"]
    if case == "a bare block (the pre-F23 cache format)":
        joblib.dump(block, path)
        return "no cache metadata (Stage3Block"
    if case == "metadata without an identity":
        joblib.dump({"block": block}, path)
        return "no cache metadata"
    if case == "an older cache schema":
        joblib.dump({**payload, "identity": {**payload["identity"], "schema": "stage3_block_cache/0"}},
                    path)
        return "cache schema 'stage3_block_cache/0'"
    if case == "an identity that is not a mapping":
        joblib.dump({**payload, "identity": "?"}, path)
        return "cache schema '?'"
    if case == "a corrupt file":
        path.write_bytes(b"\x80\x04 not a pickle")
        return "unreadable cache"
    if case == "an empty file":
        path.write_bytes(b"")
        return "unreadable cache"
    if case == "a payload that is not a block":
        joblib.dump({**payload, "block": {"X_train": None}}, path)
        return "payload holds dict"
    if case == "a pre-F22 block (no inner folds)":
        block.inner_folds = []
        message = "pre-F22 block"
    elif case == "a block edited after it was saved":
        block.tune_folds[0].X_val.iloc[0, 0] += 1.0
        message = "block content differs from its fingerprint"
    elif case == "a block whose settings were edited":
        block.settings["random_state"] = 7
        message = "block settings differ from the config"
    elif case == "a block stamped with another split":
        block.split_fingerprint = "ba9876543210"
        message = "split fingerprint differs"
    else:
        raise AssertionError(case)
    joblib.dump(payload, path)
    return message


@pytest.mark.parametrize("case", [
    "a bare block (the pre-F23 cache format)", "metadata without an identity", "an older cache schema",
    "an identity that is not a mapping", "a corrupt file", "an empty file",
    "a payload that is not a block", "a pre-F22 block (no inner folds)",
    "a block edited after it was saved", "a block whose settings were edited",
    "a block stamped with another split",
])
def test_a_cache_with_bad_metadata_or_a_failed_contract_is_rebuilt_not_reused(cache, case):
    split, path, block = cache
    why = _corrupt(case, path, split)
    rebuilt, status, made = run(split, path)
    assert_valid_rebuild(rebuilt, status, made, split, path, config(), CountingPipeline, FP, why)
    assert rebuilt.block_fingerprint == block.block_fingerprint     # the same inputs, rebuilt


# ----------------------------------------------------------------------
# malformed metadata and malformed block state: rebuilt, never an error
# ----------------------------------------------------------------------
DIGEST = "0" * 64
NOT_A_MAPPING = "malformed cache metadata (implementation is {}, not a mapping)"
NOT_DIGESTS = "malformed cache metadata (implementation does not map module names to digests)"
MALFORMED_IDENTITY = {                       # case -> (edit of the stored identity, expected reason)
    "implementation is a string": (lambda i: i.update(implementation=DIGEST),
                                   NOT_A_MAPPING.format("str")),
    "implementation is a list": (lambda i: i.update(implementation=sorted(i["implementation"])),
                                 NOT_A_MAPPING.format("list")),
    "implementation is a number": (lambda i: i.update(implementation=11), NOT_A_MAPPING.format("int")),
    "implementation is missing": (lambda i: i.pop("implementation"), NOT_A_MAPPING.format("NoneType")),
    "a module name that is not a string": (lambda i: i["implementation"].update({3: DIGEST}), NOT_DIGESTS),
    "a digest that is not a string": (
        lambda i: i["implementation"].update({"shared.stage3.data": [DIGEST]}), NOT_DIGESTS),
    "a field that cannot be compared": (lambda i: i.update(build=np.arange(3)), "changed: build"),
    "an unexpected field": (lambda i: i.update(extra=1), "changed: unexpected keys 'extra'"),
}
MALFORMED_BLOCK = {                          # case -> (edit of the cached block, expected reason)
    "inner folds missing (a pre-F22 pickle)": (lambda b: delattr(b, "inner_folds"),
                                               "missing or incomplete inner folds"),
    "inner folds for too few tuning folds": (lambda b: setattr(b, "inner_folds", b.inner_folds[:-1]),
                                             "missing or incomplete inner folds"),
    "inner folds that are a number": (lambda b: setattr(b, "inner_folds", N_TUNE),
                                      "malformed block (TypeError)"),
    "an inner fold without its rows": (lambda b: delattr(b.inner_folds[0][0], "val_idx"),
                                       "malformed block (AttributeError)"),
    "no tuning plan to count inner folds against": (lambda b: delattr(b, "tune_plan"),
                                                    "malformed block (AttributeError)"),
    "a split fingerprint that cannot be compared": (
        lambda b: setattr(b, "split_fingerprint", np.array([FP, FP])), "split fingerprint differs"),
}


@pytest.mark.parametrize("case", MALFORMED_IDENTITY)
def test_malformed_metadata_is_rebuilt_not_an_error(cache, case):
    split, path, block = cache
    edit, why = MALFORMED_IDENTITY[case]
    payload = joblib.load(path)
    edit(payload["identity"])
    joblib.dump(payload, path)
    rebuilt, status, made = run(split, path)
    assert_valid_rebuild(rebuilt, status, made, split, path, config(), CountingPipeline, FP, why)
    assert rebuilt.block_fingerprint == block.block_fingerprint


@pytest.mark.parametrize("case", MALFORMED_BLOCK)
def test_a_malformed_cached_block_is_rebuilt_not_an_error(cache, case):
    split, path, block = cache
    edit, why = MALFORMED_BLOCK[case]
    payload = joblib.load(path)
    edit(payload["block"])
    joblib.dump(payload, path)
    rebuilt, status, made = run(split, path)
    assert_valid_rebuild(rebuilt, status, made, split, path, config(), CountingPipeline, FP, why)
    assert rebuilt.block_fingerprint == block.block_fingerprint
    assert len(rebuilt.require_inner_folds()) == N_TUNE


@pytest.mark.parametrize("error", [ValueError, AttributeError, TypeError])
def test_an_error_while_building_the_replacement_propagates(cache, monkeypatch, error):
    """The errors a malformed cached block is rebuilt for are not caught when
    building its replacement raises them; the stale cache is left as it was."""
    split, path, _ = cache
    payload = joblib.load(path)
    del payload["block"].tune_plan                                   # forces a rebuild
    joblib.dump(payload, path)
    before = path.read_bytes()

    def broken(self, X, y=None):
        raise error("genuine build failure")

    monkeypatch.setattr(CountingPipeline, "fit_transform", broken)   # same identity, failing build
    with pytest.raises(error, match="genuine build failure"):
        run(split, path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_no_cache_builds_and_saves_one(tmp_path):
    split = stage3_split()
    path = tmp_path / "new" / "stage3_block.joblib"
    block, status, made = run(split, path)
    assert (status, made) == ("built (no cache)", PIPELINES_PER_BUILD)
    assert joblib.load(path)["identity"] == block_identity(split, CountingPipeline, config(), FP)


def test_a_failed_write_leaves_the_previous_cache_and_no_partial_file(cache, monkeypatch):
    split, path, _ = cache
    before = path.read_bytes()
    moved = {**split, "y_train": split["y_train"].copy()}
    moved["y_train"].iloc[0] = 1 - moved["y_train"].iloc[0]          # forces a rebuild

    def interrupted(obj, filename, *args, **kwargs):
        with open(filename, "wb") as fh:
            fh.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(block_cache.joblib, "dump", interrupted)
    with pytest.raises(OSError, match="disk full"):
        run(moved, path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


# ----------------------------------------------------------------------
# implementation identity
# ----------------------------------------------------------------------
def test_the_implementation_identity_covers_every_module_the_block_imports():
    identity = implementation_identity(Stage3Block, EBMFeaturePipeline)
    assert {"shared.stage3.data", "shared.stage_io", "shared.stage3.feature_engineering",
            "feature_research.config",                              # an imported constant (the seed)
            "feature_research.feature_engineering._checks",
            *(f"feature_research.feature_engineering.{m}" for m in
              ("crisis", "integrals", "derivatives", "temporal", "prior", "overlap"))} <= set(identity)
    assert all(name.split(".")[0] in block_cache.PROJECT_PACKAGES for name in identity)
    assert all(isinstance(d, str) and len(d) == 64 for d in identity.values())


def test_the_implementation_identity_changes_when_a_reachable_source_changes(tmp_path, monkeypatch):
    """A module the pipeline imports, edited on disk, changes the identity —
    and only that module's entry."""
    source = tmp_path / "probe.py"
    source.write_text("from feature_research.config import RANDOM_SEED\n\n"
                      "class Probe:\n    seed = RANDOM_SEED\n", encoding="utf-8")
    name = "feature_research._block_cache_probe"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    before = implementation_identity(module.Probe)
    assert {name, "feature_research.config"} <= set(before)
    source.write_text(source.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    after = implementation_identity(module.Probe)
    assert {k for k in before if before[k] != after[k]} == {name}
    assert set(after) == set(before)
