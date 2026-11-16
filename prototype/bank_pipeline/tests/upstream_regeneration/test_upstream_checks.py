"""
PR 39 Step 4A: the Stage 1-3 checks of gates A4.3-A4.5
(upstream_regeneration.checks), on synthetic artifacts in the saved formats.

* Stage 1 (A4.3): both arms pass on the frozen family (sigmoid), feature
  contract 2.0.0 and artifact version 3.0, nested OOF provenance, the 10-fold
  seed-42 plan and the protocol's split; each departure fails by name.
* Stage 3 (A4.4): both arms pass on one block, one protocol, the shared 5-fold
  plan, per-fold features, per-fold calibration gates and OOF thresholds; each
  departure fails by name. The EBM carries the ``stage_io`` fingerprint and the
  XGBoost the router one, as GLASS Cell 17 and black-box Cell 30 stamp them
  (test_stage3_split_identity.py covers the two schemes in detail).
* Stage 4 inputs (A4.5): all four arbiter-upstream input builds pass, train and
  test side, and nothing is fitted: every Stage 4 fitting entry point is
  replaced by one that raises.
* The Stage 2 reuse decision (section 5.2): ``reuse`` only when every trigger
  is clear and every reuse check passed, ``report --check`` included.
"""

from __future__ import annotations

import copy

import joblib
import numpy as np
import pandas as pd
import pytest

from glass_pipeline.glass_router.pipeline import fold_fingerprint
from shared.stage1.feature_engineering import (HISTORICAL_STAGE1_FEATURE_CONTRACT, LR_FEATURES,
                                               STAGE1_FEATURE_CONTRACT, feature_contract_record)
from shared.stage3.feature_engineering import EBM_FEATURES
from shared.stage_io import StageOutput, fold_assignment
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage3
from upstream_regeneration import checks as C

RF_REMAINDER = "OOF Pass 1 remainder (RF router design)"


def _so(w, arm, *, family="sigmoid", names=LR_FEATURES, folds=10, provenance=None):
    GS, rng = w["GS"], np.random.default_rng(1 if arm == "glass" else 2)
    ytr = GS["y_train"]
    p_tr = 1 / (1 + np.exp(-(2 * w["z_tr"] - 2.4 + rng.normal(0, .6, len(ytr)))))
    p_te = 1 / (1 + np.exp(-(2 * w["z_te"] - 2.4 + rng.normal(0, .5, len(w["z_te"])))))
    return StageOutput(
        stage="stage1", arm=arm, model="calibrated_lr" if arm == "glass" else "mlp",
        feature_names=list(names),
        train_proba_oof=pd.Series(p_tr, index=GS["X_train"].index),
        test_proba=pd.Series(p_te, index=GS["X_test"].index),
        y_train=ytr.astype(int), y_test=GS["y_test"].astype(int),
        train_fold_id=fold_assignment(ytr, cv_folds=folds, random_state=42),
        threshold=0.21, threshold_source="in_stage_f2_cv",
        calibration_method=family, calibration_requested=family,
        oof_provenance=provenance or "refittable (nested cross-fit: 10 outer x 10 inner folds)",
        config={"random_state": 42})


def _payload(so, version="3.0", contract=STAGE1_FEATURE_CONTRACT):
    return {"artifact_version": version, "feature_names": list(so.feature_names),
            "feature_contract": feature_contract_record(contract), "stage_output": so.to_dict(),
            "train_proba_oof": so.train_proba_oof, "train_fold_id": so.train_fold_id}


@pytest.fixture(scope="module")
def world():
    w = build_world(n=1500, n_test=350)
    plan = make_stage3(w)                                  # the shared 5-fold plan of the world
    population = {"stage_io_split_fingerprint": w["fp_io"], "router_split_fingerprint": w["fp_router"],
                  "fold_fingerprint": fold_fingerprint(plan.oof["fold_id"])}
    return w, population


def _stage3(w, family):
    a = make_stage3(w, family=family)
    if family == "ebm":                     # GLASS Cell 17 stamps Stage 3 with the stage_io scheme
        a.split_fingerprint = w["fp_io"]
    a.features = list(EBM_FEATURES)
    a.tuning = {"tuned": True, "n_trials_requested": 3, "random_state": 42,
                "feature_fit_scope": "per_fold"}
    return a


def _stage1(w):
    return {"stage1.lr": _payload(_so(w, "glass")), "stage1.mlp": _payload(_so(w, "blackbox"))}


# ----------------------------------------------------------------------
# A4.3 Stage 1
# ----------------------------------------------------------------------
def test_both_stage1_arms_pass_on_the_frozen_family_names_and_folds(world):
    w, pop = world
    gate = C.Gate("A4.3")
    out = C.stage1_checks(gate, _stage1(w), w["GS"], pop)
    assert gate.passed, gate.failed()
    assert out["stage1.lr"]["calibration_method"] == "sigmoid" and out["stage1.mlp"]["n_folds"] == 10
    assert out["stage1.lr"]["fold_fingerprint"] == out["stage1.mlp"]["fold_fingerprint"]


@pytest.mark.parametrize("role, edit, failed", [
    ("stage1.lr", lambda w: _payload(_so(w, "glass", family="isotonic")), "calibration family sigmoid"),
    ("stage1.mlp", lambda w: _payload(_so(w, "blackbox"), version="2.0"), "artifact version 3.0"),
    ("stage1.lr", lambda w: _payload(_so(w, "glass", names=HISTORICAL_STAGE1_FEATURE_CONTRACT["features"]),
                                     contract=HISTORICAL_STAGE1_FEATURE_CONTRACT),
     "ordered feature names = LR_FEATURES"),
    ("stage1.mlp", lambda w: _payload(_so(w, "blackbox", folds=5)), "Stage 1 fold plan"),
    ("stage1.lr", lambda w: _payload(_so(w, "glass", provenance="refittable (outer folds only)")),
     "nested cross-fit"),
    ("stage1.lr", lambda w: {**_payload(_so(w, "glass")), "stage_output": None}, "holds a StageOutput"),
])
def test_each_stage1_departure_fails_by_name(world, role, edit, failed):
    w, pop = world
    arts = _stage1(w)
    arts[role] = edit(w)
    gate = C.Gate("A4.3")
    C.stage1_checks(gate, arts, w["GS"], pop)
    assert any(role in f and failed in f for f in gate.failed()), gate.failed()


def test_a_stage1_artifact_from_another_split_fails(world):
    w, pop = world
    gate = C.Gate("A4.3")
    C.stage1_checks(gate, _stage1(w), w["GS"], {**pop, "stage_io_split_fingerprint": "0" * 12})
    assert [f for f in gate.failed() if "split fingerprint" in f] == [
        "stage1.lr: split fingerprint = the frozen protocol's stage I/O split",
        "stage1.mlp: split fingerprint = the frozen protocol's stage I/O split"]


# ----------------------------------------------------------------------
# A4.4 Stage 3
# ----------------------------------------------------------------------
def test_both_stage3_arms_pass_on_one_block_and_one_protocol(world):
    w, pop = world
    gate = C.Gate("A4.4")
    out = C.stage3_checks(gate, {"stage3.ebm": _stage3(w, "ebm"), "stage3.xgb": _stage3(w, "xgboost")}, pop,
                          split=w["GS"], extra_fingerprints=ROUTER_FP)
    assert gate.passed, gate.failed()
    assert out["stage3.ebm"]["fold_fingerprint"] == pop["fold_fingerprint"]
    assert out["stage3.xgb"]["arm"] == "blackbox"
    assert {r: v["scheme"] for r, v in out["split_identity"].items()} == {
        "stage3.ebm": "stage_io", "stage3.xgb": "router"}
    assert out["protocol_allowed"] == ["search_space", "split_fingerprint"]


def _edit(attr, value):
    def apply(a):
        setattr(a, attr, value)
    return apply


@pytest.mark.parametrize("role, edit, failed", [
    ("stage3.xgb", _edit("block_fingerprint", "c" * 64), "one block"),
    ("stage3.ebm", _edit("features", list(EBM_FEATURES)[::-1]), "features = EBM_FEATURES"),
    ("stage3.xgb", _edit("feature_fit_scope", "global"), "per-fold feature fit"),
    ("stage3.ebm", lambda a: a.config.update(arm="blackbox"), "model_family / arm"),
    ("stage3.xgb", lambda a: a.threshold["oof"].update(selected_on="test"), "out-of-fold training"),
    ("stage3.ebm", lambda a: a.calibration.pop("fold_applied"), "calibration gate recorded per fold"),
    ("stage3.xgb", lambda a: a.oof.update(fold_id=np.roll(a.oof["fold_id"], 1)), "shared 5-fold plan"),
    ("stage3.ebm", _edit("split_fingerprint", "0" * 16), "split fingerprint"),
    ("stage3.xgb", lambda a: a.tuning.update(n_trials_requested=4), "one shared protocol"),
])
def test_each_stage3_departure_fails_by_name(world, role, edit, failed):
    w, pop = world
    arts = {"stage3.ebm": _stage3(w, "ebm"), "stage3.xgb": _stage3(w, "xgboost")}
    edit(arts[role])
    gate = C.Gate("A4.4")
    C.stage3_checks(gate, arts, pop, split=w["GS"], extra_fingerprints=ROUTER_FP)
    assert any(failed in f for f in gate.failed()), gate.failed()


# ----------------------------------------------------------------------
# A4.5 Stage 4 inputs: built, never fitted
# ----------------------------------------------------------------------
@pytest.fixture
def upstream_files(world, tmp_path):
    w, _ = world
    arts = {**_stage1(w),
            "stage2.glass_router": make_router(w, pass2_population="oof_remainder"),
            "stage2.rf_router": make_router(w, layout="extra", band_blend=True,
                                            pass2_population=RF_REMAINDER),
            "stage3.ebm": _stage3(w, "ebm"), "stage3.xgb": _stage3(w, "xgboost")}
    paths = {}
    for role, art in arts.items():
        paths[role] = tmp_path / f"{role}.joblib"
        joblib.dump(art, paths[role])
    return paths


def test_all_four_stage4_input_builds_pass_and_nothing_is_fitted(world, upstream_files, monkeypatch):
    w, _ = world

    def forbidden(*a, **k):
        raise AssertionError("a Stage 4 model was fitted or tuned")

    import blackbox_pipeline.models.meta_xgb.runner as meta_runner
    import glass_pipeline.glass_arbiter.meta_stage as meta_stage
    import shared.stage4.protocol as s4protocol
    import xgboost
    for module, name in ((meta_stage, "train_glass_arbiter"), (meta_runner, "train_meta_xgb_stage"),
                         (s4protocol, "tune_min_confidence")):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(xgboost.XGBClassifier, "fit", forbidden)
    gate = C.Gate("A4.5")
    out = C.stage4_input_checks(gate, w["GS"], upstream_files, extra_fingerprints=ROUTER_FP)
    assert gate.passed, gate.failed()
    assert sorted(out) == ["glass_arbiter <- blackbox upstream", "glass_arbiter <- glass upstream",
                           "meta_xgb <- blackbox upstream", "meta_xgb <- glass upstream"]
    assert {v["combination"] for v in out.values()} == {
        "glass upstream -> glass_arbiter", "blackbox upstream -> glass_arbiter",
        "glass upstream -> meta_xgb", "blackbox upstream -> meta_xgb"}


def test_a_stage4_input_build_that_fails_is_recorded_not_raised(world, upstream_files):
    w, _ = world
    paths = dict(upstream_files)
    paths["stage2.rf_router"] = paths["stage2.glass_router"]          # a GLASS bundle in the RF slot
    gate = C.Gate("A4.5")
    C.stage4_input_checks(gate, w["GS"], paths, extra_fingerprints=ROUTER_FP)
    assert any("blackbox upstream" in f for f in gate.failed()), gate.failed()
    assert not any("<- glass upstream" in f for f in gate.failed())


def test_every_upstream_file_reloads_in_a_fresh_interpreter(upstream_files):
    gate = C.Gate("A4.3")
    got = C.fresh_reload(gate, upstream_files)
    assert gate.passed, gate.failed()
    assert got["stage3.ebm"]["type"] == "shared.stage3.artifact.Stage3Artifact"
    assert "stage_output" in got["stage1.lr"]["keys"]


# ----------------------------------------------------------------------
# section 5.2: the decision
# ----------------------------------------------------------------------
PASSED_PREFLIGHT = {"evidence": {"report_check": {"passed": True}}}


def test_reuse_needs_every_check_and_report_check():
    ok = {"checks": [{"check": "reuse 1", "passed": True}], "failed": []}
    d = C.decide(PASSED_PREFLIGHT, ok)
    assert d["outcome"] == "reuse" and d["authorization"]["decision"] == "D3"
    assert C.decide({"evidence": {"report_check": {"passed": False}}}, ok)["outcome"] == "regenerate"
    bad = C.decide(PASSED_PREFLIGHT, {"checks": [{"check": "trigger 2", "passed": False}],
                                      "failed": ["trigger 2: binning identity"]})
    assert bad["outcome"] == "regenerate" and bad["failed"] == ["trigger 2: binning identity"]
    assert "STOP" in bad["next"] and "models/pr39/<run_id>/" in bad["next"]
    assert C.decide(PASSED_PREFLIGHT, {"checks": [], "failed": []})["outcome"] == "regenerate"


def test_the_names_check_compares_the_accepted_routers_with_the_current_bins():
    from shared.stage2.binning import RF_FEATURES_BINARY
    # the rules as ModelSaver._serialize_rules saves them: one list, (feature, value) conditions
    glass = {"features": list(RF_FEATURES_BINARY),
             "rules": [{"pass": "Pass 1", "segment": [(RF_FEATURES_BINARY[0], 1)]},
                       {"pass": "Pass 2", "segment": [(RF_FEATURES_BINARY[3], 1)]}]}
    gate = C.Gate("A4.2")
    C._names_checks(gate, glass, {"feature_names": list(RF_FEATURES_BINARY)})
    assert gate.passed, gate.failed()
    renamed = copy.deepcopy(glass)
    renamed["rules"][1]["segment"] = [("dow_month_encoded_hot", 1)]
    gate = C.Gate("A4.2")
    C._names_checks(gate, renamed, {"feature_names": list(RF_FEATURES_BINARY)[::-1]})
    assert gate.failed() == ["trigger 2: the RF Router's feature_names = RF_FEATURES_BINARY, in order",
                             "trigger 2: every feature the accepted GLASS rules use is a current bin"]
