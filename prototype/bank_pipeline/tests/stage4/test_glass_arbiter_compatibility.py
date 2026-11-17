"""
PR 39, Commit 3, Step 3B (decision D5): the GLASS Arbiter's names, and what
stays readable from PR 36-38, when it was "Meta-EBM".

Current: ``glass_pipeline.glass_arbiter``, ``train_glass_arbiter``, schema
``glass_arbiter/1``, ``models/glass_arbiter/glass_arbiter_*.joblib``, upstream
arbiter ``glass_arbiter``, comparison names ``*_glass_arbiter*``.

Historical, kept: the ``glass_pipeline.meta_ebm`` import path (the same module
objects) and pickles naming it; ``meta_ebm/2`` artifacts and ``meta_ebm_*``
files, read as saved; the aliases ``train_meta_stage``, ``save_meta_ebm``,
``load_meta_ebm_for_comparison``, ``find_meta_ebm_artifact``,
``compare_with_meta_ebm`` and ``meta_ebm_path=``.

Unchanged: every number (a renamed artifact differs from its historical form
in names only), the Meta-XGB, RF Router and GLASS Router identities, every
arm and model_family, and the Stage 4 slot names.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

import blackbox_pipeline.models.meta_xgb as meta_xgb_pkg
import blackbox_pipeline.models.meta_xgb.comparison as C
import glass_pipeline.glass_arbiter as ga
from glass_pipeline.glass_arbiter import artifacts as A
from glass_pipeline.glass_arbiter import meta_stage as ms
from shared.stage4 import UPSTREAM, Stage4InputError
from tests.support.meta_xgb_world import run_meta
from tests.support.notebooks import BANK_ROOT
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3

SUBMODULES = ("arbiter", "artifacts", "calibration", "evaluation", "loader", "meta_stage",
              "router_evidence", "tracer", "tuning", "weighting")
RF_REMAINDER = "OOF Pass 1 remainder (RF router design)"


# ======================================================================
# identifiers
# ======================================================================
def test_current_and_historical_identifiers_are_distinct_and_both_readable():
    assert (A.ARBITER, A.ARTIFACT_SCHEMA, A.ARTIFACT_DIR, A.ARTIFACT_PREFIX) == (
        "glass_arbiter", "glass_arbiter/1", "./models/glass_arbiter", "glass_arbiter_")
    assert (A.HISTORICAL_ARBITER, A.HISTORICAL_ARTIFACT_SCHEMA, A.HISTORICAL_ARTIFACT_DIR,
            A.HISTORICAL_ARTIFACT_PREFIX) == ("meta_ebm", "meta_ebm/2", "./models/meta_ebm", "meta_ebm_")
    assert A.SCHEMA_ARBITER == {"glass_arbiter/1": "glass_arbiter", "meta_ebm/2": "meta_ebm"}
    assert A.READABLE_SCHEMAS == ("glass_arbiter/1", "meta_ebm/2")
    assert A.ARTIFACT_PATTERNS == ("glass_arbiter_*.joblib", "meta_ebm_*.joblib")
    assert ms.ARTIFACT_SCHEMA == A.ARTIFACT_SCHEMA


def test_the_names_that_do_not_change():
    from blackbox_pipeline.models.meta_xgb import SCHEMA, MetaXGBConfig
    from blackbox_pipeline.models.rf_router.config import RFRouterConfig
    cfg = MetaXGBConfig()
    assert (cfg.model_family, cfg.arm, cfg.stage) == ("meta_xgb", "blackbox", "stage4")
    assert SCHEMA.startswith("meta_xgb")
    assert cfg.counterpart.startswith("glass_pipeline.glass_arbiter ")      # a pointer, not an identity
    assert {arm: {s: v["want"] for s, v in st.items()} for arm, st in UPSTREAM.items()} == {
        "glass": {"stage1": {"arm": "glass"}, "stage2": {"model_family": "glass_router"},
                  "stage3": {"model_family": "ebm", "arm": "glass"}},
        "blackbox": {"stage1": {"arm": "blackbox"}, "stage2": {"model_family": "rf_router"},
                     "stage3": {"model_family": "xgboost", "arm": "blackbox"}}}
    assert RFRouterConfig.__name__ == "RFRouterConfig"


# ======================================================================
# the historical import path and pickles
# ======================================================================
@pytest.mark.parametrize("name", SUBMODULES)
def test_the_historical_import_path_is_the_same_module(name):
    old = importlib.import_module(f"glass_pipeline.meta_ebm.{name}")
    assert old is importlib.import_module(f"glass_pipeline.glass_arbiter.{name}")
    assert importlib.util.find_spec(f"glass_pipeline.meta_ebm.{name}") is not None


def test_the_historical_package_re_exports_the_current_api_and_aliases():
    import glass_pipeline.meta_ebm as old
    assert old.__all__ == ga.__all__
    assert old.train_meta_stage is ga.train_glass_arbiter is ms.train_meta_stage
    assert old.save_meta_ebm is A.save_glass_arbiter
    from glass_pipeline.meta_ebm.loader import _router_evidence_checks     # PR 38 acceptance V1
    assert _router_evidence_checks is ga.loader._router_evidence_checks


def test_a_pickle_naming_the_historical_path_loads_in_a_clean_interpreter(tmp_path):
    blob = pickle.dumps(ga.meta_arbiter, protocol=2).replace(
        b"glass_pipeline.glass_arbiter.arbiter", b"glass_pipeline.meta_ebm.arbiter")
    assert b"glass_pipeline.meta_ebm.arbiter" in blob
    (tmp_path / "old.pkl").write_bytes(blob)
    run = subprocess.run(
        [sys.executable, "-c", "import pickle, sys; f = pickle.load(open(sys.argv[1], 'rb')); "
                               "print(f.__module__, f.__name__)", str(tmp_path / "old.pkl")],
        capture_output=True, text=True, cwd=str(BANK_ROOT), timeout=300,
        env={**os.environ, "PYTHONPATH": str(BANK_ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split() == ["glass_pipeline.glass_arbiter.arbiter", "meta_arbiter"]


# ======================================================================
# artifacts: current schema written, historical schema read as saved
# ======================================================================
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = build_world(n=1500, n_test=350)
    root = tmp_path_factory.mktemp("arbiter_names")
    files = {}
    for arm, arts in (
            ("glass", [make_stage1(w), make_router(w, pass2_population="full", label_fingerprint=True),
                       make_stage3(w)]),
            ("blackbox", [make_stage1(w, arm="blackbox", model="mlp"),
                          make_router(w, layout="extra", band_blend=True, pass2_population=RF_REMAINDER,
                                      label_fingerprint=True),
                          make_stage3(w, family="xgboost")])):
        files[arm] = []
        for stage, art in zip(("stage1", "stage2", "stage3"), arts):
            files[arm].append(str(root / f"{arm}_{stage}.joblib"))
            joblib.dump(art, files[arm][-1])
    with pytest.MonkeyPatch.context() as mp:
        real = ms.load_stage_outputs
        mp.setattr(ms, "load_stage_outputs", lambda *a, **k: real(*a, extra_fingerprints=ROUTER_FP, **k))
        arbiter, path = ga.train_glass_arbiter(w["GS"], *files["glass"], verbose=False,
                                               artifact_dir=str(root / "models" / "glass_arbiter"))
    xgb, _ = run_meta(w, dict(zip(("mlp", "rf", "xgb"), files["blackbox"])), root / "xgb", save=False)
    return w, root, arbiter, Path(path), xgb


def _as_pr36_38(art: dict) -> dict:
    """The same artifact as PR 36-38 wrote it (names only)."""
    old = copy.deepcopy(art)
    old["schema"] = "meta_ebm/2"
    up = old["stage4_inputs"]["upstream"]
    up.update(arbiter="meta_ebm", combination=f"{up['arm']} upstream -> meta_ebm")
    return old


def test_a_new_artifact_records_the_current_names(world):
    _, _, art, path, _ = world
    assert path.parent.name == "glass_arbiter" and path.name.startswith("glass_arbiter_")
    assert art["schema"] == "glass_arbiter/1"
    up = art["stage4_inputs"]["upstream"]
    assert (up["arbiter"], up["arbiter_arm"], up["combination"]) == (
        "glass_arbiter", "glass", "glass upstream -> glass_arbiter")
    assert sorted(art["stage4_inputs"]["streams"]) == ["ebm", "glass", "lr"]       # slots unchanged


def test_a_historical_artifact_is_read_and_found_as_saved(world, tmp_path):
    w, _, art, _, _ = world
    old = _as_pr36_38(art)
    folder = tmp_path / "models" / "meta_ebm"
    folder.mkdir(parents=True)
    joblib.dump(old, folder / "meta_ebm_20261023_070055_573687.joblib")
    loaded = C.load_glass_arbiter_for_comparison(w["GS"], str(folder / "meta_ebm_20261023_070055_573687.joblib"),
                                                 ROUTER_FP)
    assert loaded["schema"] == "meta_ebm/2" and loaded["stage4_inputs"]["upstream"]["arbiter"] == "meta_ebm"
    found, found_path = C.find_glass_arbiter_artifact([folder], w["GS"], ROUTER_FP, verbose=False)
    assert Path(found_path).name == "meta_ebm_20261023_070055_573687.joblib"
    with pytest.raises(Stage4InputError, match="glass_arbiter/1"):
        C.load_glass_arbiter_for_comparison(w["GS"], {**old, "schema": "meta_ebm/1"}, ROUTER_FP)


def test_a_historical_artifact_compares_exactly_as_the_renamed_one(world):
    w, _, art, _, xgb = world
    new = C.compare_with_glass_arbiter(w["GS"], art, xgb, n_boot=50, verbose=False,
                                       extra_fingerprints=ROUTER_FP)
    old = C.compare_with_glass_arbiter(w["GS"], _as_pr36_38(art), xgb, n_boot=50, verbose=False,
                                       extra_fingerprints=ROUTER_FP)
    assert new["table"].equals(old["table"]) and new["bootstrap"].equals(old["bootstrap"])
    assert new["paired"] == old["paired"]
    assert new["glass_arbiter_config"] == {**old["glass_arbiter_config"], "schema": "glass_arbiter/1"}
    assert (new["protocol"]["upstream"]["glass_arbiter"], old["protocol"]["upstream"]["glass_arbiter"]) == (
        "glass upstream -> glass_arbiter", "glass upstream -> meta_ebm")
    assert list(new["table"].index.get_level_values(0).unique()) == ["GLASS Arbiter", "Meta-XGB"]
    assert set(new["paired"]["no_abstain"]) >= {"glass_arbiter_only_correct", "meta_xgb_only_correct"}
    assert set(new["paired"]["retention"]) == {"both", "glass_arbiter_only", "meta_xgb_only", "neither"}


# ======================================================================
# aliases (kept through PR 39)
# ======================================================================
def test_the_pr36_38_function_names_are_aliases(world):
    w, _, art, _, xgb = world
    assert ms.train_meta_stage is ms.train_glass_arbiter
    assert A.save_meta_ebm is A.save_glass_arbiter
    assert C.load_meta_ebm_for_comparison is C.load_glass_arbiter_for_comparison
    assert C.find_meta_ebm_artifact is C.find_glass_arbiter_artifact
    assert meta_xgb_pkg.compare_with_meta_ebm is C.compare_with_meta_ebm
    via_alias = C.compare_with_meta_ebm(w["GS"], art, xgb, n_boot=20, verbose=False,
                                        extra_fingerprints=ROUTER_FP)
    direct = C.compare_with_glass_arbiter(w["GS"], art, xgb, n_boot=20, verbose=False,
                                          extra_fingerprints=ROUTER_FP)
    assert via_alias["bootstrap"].equals(direct["bootstrap"]) and via_alias["paired"] == direct["paired"]


def test_meta_xgb_takes_meta_ebm_path_as_an_alias_but_not_both(world):
    w, root, _, path, _ = world
    files = sorted(str(p) for p in root.glob("blackbox_stage*.joblib"))
    arts = dict(zip(("mlp", "rf", "xgb"), files))
    by_alias, _ = run_meta(w, arts, root / "alias", save=False, meta_ebm_path=str(path))
    by_name, _ = run_meta(w, arts, root / "name", save=False, glass_arbiter_path=str(path))
    assert by_alias.comparison["paired"] == by_name.comparison["paired"]
    assert by_alias.comparison["glass_arbiter_config"]["schema"] == "glass_arbiter/1"
    with pytest.raises(TypeError, match="not both"):
        run_meta(w, arts, root / "both", save=False, meta_ebm_path=str(path),
                 glass_arbiter_path=str(path))
    np.testing.assert_array_equal(by_alias.test["proba"], by_name.test["proba"])
