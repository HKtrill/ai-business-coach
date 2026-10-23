"""
Black-box Stage 2 (RF Router): the saved artifact and the Stage 4 handoff, on
the notebook's Stage 2 path (tests.support.rf_router_world).

Contract: an artifact carries the fitted forests, the frozen operating point,
the 29-bin feature contract, the configuration and the split fingerprint, and
those alone reproduce the router's decisions, confidences and probabilities —
on the test split and on rows it never saw; it reloads only for its own split.
The Stage 4 block saved with it (notebook Cell 24) is the outer cross-fit on
the train side and the full-train router on the test side, and the shared
Stage 4 adapter reads it as exactly that: OOF rows aligned to GLOBAL_SPLIT, the
shared 5-fold plan, P(subscribe) orientation, Pass 1 below and Pass 2 above
the base rate. A block that is not OOF, is reordered, duplicated, truncated,
contradictory, on another fold plan or another split is refused. The Stage 1
MLP and the Stage 2 router describe the same rows of one split, each on its
own fold plan (10 and 5 folds in the notebook), and Stage 4 takes them
together. The black-box notebook calls all of this the way it is tested here.

Not covered here: arm / model-family labels — the RF artifact carries none
beyond ``mode`` and its ``extra`` layout; checking them across arms is F29
(Step 3).
"""

from __future__ import annotations

import ast
import dataclasses
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from blackbox_pipeline.models.mlp import CalibratedStage1MLP
from blackbox_pipeline.models.rf_router import (
    RFRouterConfig, SingleRFRouter, find_artifact_for_split, load_rf_router, save_rf_router,
)
from blackbox_pipeline.models.rf_router import split_fingerprint as router_split_fingerprint
from blackbox_pipeline.models.rf_router.bands import BandConfidence
from blackbox_pipeline.models.rf_router.forest import ForestTrainer
from blackbox_pipeline.models.rf_router.router.base import apply_cuts
from shared.stage1.feature_engineering import engineer_features as stage1_features
from shared.stage2 import RF_FEATURES_BINARY
from shared.stage4 import Stage4InputError, build_stage4_inputs, router_stream, stage_output_stream
from shared.stage4.adapters import PASS2_COMPLETE
from shared.stage4.contract import LABELS_FINGERPRINTED
from shared.validation import ValidationError
from tests.stage1.conftest import MLP_TEST_CONFIG
from tests.support import mutations as mut
from tests.support.notebooks import BANK_ROOT
from tests.support.rf_router_world import (
    BANK_CONSTRAINTS, N_STAGE_FOLDS, RANDOM_STATE, bank_world, engineer, quiet, router_config,
    save_world, serial_forests, stage4_block,
)
from tests.support.stage4_world import build_world as build_glass_world
from tests.support.stage4_world import make_router as make_glass_router
from tests.support.synthetic import make_bank_frame, split_frame

pytestmark = pytest.mark.filterwarnings(        # the MLP's tiny synthetic folds (see Stage 1 suite)
    "ignore:Got `batch_size` less than 1 or larger than sample size:UserWarning")

NOTEBOOK = BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"


@pytest.fixture(scope="module")
def world():
    return bank_world()


@pytest.fixture(scope="module")
def saved(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("rf_router")
    return out, save_world(world, out)


@pytest.fixture(scope="module")
def unseen(world):
    """Engineered bins for rows outside the split (encoder fitted on the training rows)."""
    raw = make_bank_frame(n=200, seed=9).drop(columns="y")
    gs = world.global_split
    return engineer(gs["X_train"], raw, gs["y_train"])[1]


def _restored_predict(art, X):
    """Route ``X`` from an artifact's forests, thresholds and bands alone."""
    X = X[art["feature_names"]]
    t1, t2 = art["thresholds"]["t1"], art["thresholds"]["t2"]
    p1 = ForestTrainer.positive_proba(art["pass1_model"], X)
    p2 = np.full(len(X), np.nan)
    rem = p1 >= t1
    if rem.any():
        p2[rem] = ForestTrainer.positive_proba(art["pass2_model"], X[rem])
    preds, decisions = apply_cuts(p1, p2, t1, t2)
    return preds, BandConfidence.from_dict(art["bands"]).assign(decisions), decisions


def _stream(art_or_path, world):
    return router_stream(art_or_path, "rf_router", "RF Router (two-pass)",
                         world.stage2["y_train"].to_numpy())


def _failed(checks):
    return [name for _, _, name, fn, _ in checks if not fn()]


# ----------------------------------------------------------------------
# artifact
# ----------------------------------------------------------------------
def test_only_a_fitted_router_can_be_saved(tmp_path):
    from blackbox_pipeline.models.rf_router import RFRouter
    with pytest.raises(ValueError, match="not fitted"):
        save_rf_router(RFRouter(router_config()), tmp_path)


def test_the_artifact_carries_the_operating_point_and_its_provenance(world, saved):
    out, path = saved
    r = world.router
    art = quiet(load_rf_router, path, expected_split_fingerprint=world.stage2["split_fingerprint"])
    assert art["artifact_version"] == "1.0" and art["mode"] == "two_pass"
    assert art["feature_names"] == r.feature_names_ == list(RF_FEATURES_BINARY)
    assert RFRouterConfig.from_dict(art["config"]).to_dict() == r.config.to_dict()
    assert art["constraints"] == BANK_CONSTRAINTS.to_dict()
    assert art["thresholds"] == r.thresholds_.to_dict() and art["bands"] == r.bands_.to_dict()
    assert art["training_base_rate"] == r.training_base_rate
    assert art["split_fingerprint"] == r.split_fingerprint_ == world.stage2["split_fingerprint"]
    np.testing.assert_array_equal(art["remainder_mask"], r.remainder_mask_)
    assert art["fit_report"]["remainder"]["source"] == "OOF pass1 probabilities >= t1"
    meta = json.loads((out / "rf_router_two_pass_metadata.json").read_text(encoding="utf-8"))
    assert not {"pass1_model", "pass2_model", "remainder_mask", "model"} & set(meta)
    for name, res in (("pass1", r.thresholds_.pass1), ("pass2", r.thresholds_.pass2)):
        sweep = pd.read_csv(out / f"rf_router_two_pass_{name}_sweep.csv")
        np.testing.assert_allclose(sweep["threshold"], res.sweep["threshold"])


def test_the_artifact_alone_reproduces_the_router(world, saved, unseen):
    art = quiet(load_rf_router, saved[1])
    r, X_te = world.router, world.stage2["X_test"]
    for X in (X_te, unseen):
        for got, want in zip(_restored_predict(art, X), r.predict(X)):
            np.testing.assert_array_equal(got, want)
    _, conf, decisions = _restored_predict(art, X_te)
    proba = r._proba_from_triple(conf, decisions, art["training_base_rate"])
    np.testing.assert_array_equal(proba, art["extra"]["test_proba"])        # the saved Stage 4 test side


def test_an_artifact_is_reloaded_for_its_own_split_only(world, saved):
    out, path = saved
    other = split_frame(make_bank_frame(n=1000), test_size=0.20, seed=1)
    from blackbox_pipeline.models.rf_router import split_fingerprint
    other_fp = split_fingerprint(other["X_train"], other["X_test"])
    assert other_fp != world.stage2["split_fingerprint"]
    with pytest.raises(ValueError, match="Split fingerprint mismatch"):
        quiet(load_rf_router, path, expected_split_fingerprint=other_fp)
    assert quiet(find_artifact_for_split, [out], "rf_router_*.joblib", other_fp) == (None, None)
    found, found_path = quiet(find_artifact_for_split, [out], "rf_router_*.joblib",
                              world.stage2["split_fingerprint"])
    assert found_path == path and found["mode"] == "two_pass"


def test_the_ablation_artifact_holds_its_single_forest(world, tmp_path, unseen):
    s2 = world.stage2
    relaxed = dataclasses.replace(BANK_CONSTRAINTS, require_band_ordering=False)
    with serial_forests():
        ablation = SingleRFRouter(router_config(constraints=relaxed, mode="single_rf")).fit(
            s2["X_train"], s2["y_train"], s2["split_fingerprint"])
    art = quiet(load_rf_router, quiet(save_rf_router, ablation, tmp_path, name="rf_router_single"))
    assert art["mode"] == "single_rf" and "model" in art and "pass1_model" not in art
    p = ForestTrainer.positive_proba(art["model"], unseen[art["feature_names"]])
    preds, decisions = apply_cuts(p, p, art["thresholds"]["t1"], art["thresholds"]["t2"])
    want = ablation.predict(unseen)
    np.testing.assert_array_equal(preds, want[0])
    np.testing.assert_array_equal(decisions, want[2])


# ----------------------------------------------------------------------
# the Stage 4 block
# ----------------------------------------------------------------------
def test_stage4_reads_the_crossfit_train_side_and_the_full_train_test_side(world, saved):
    stream, checks = _stream(saved[1], world)
    cf, (_, _, d_te, proba_te) = world.crossfit, world.test_out
    np.testing.assert_array_equal(stream.train_proba, cf.oof["proba"][:, 1])
    np.testing.assert_array_equal(stream.test_proba, proba_te[:, 1])
    np.testing.assert_array_equal(stream.train_decisions == "pass1", cf.oof["decisions"] == "pass1")
    np.testing.assert_array_equal(stream.test_decisions == "abstain", d_te == "uncertain")
    np.testing.assert_array_equal(stream.train_fold_id, world.fold_id)
    assert (stream.train_source, stream.notes["artifact_layout"]) == ("extra.train_proba", "extra")
    assert stream.fold_seed == RANDOM_STATE
    assert stream.notes["oof_protocol"]["pass2_population"] == "OOF Pass 1 remainder (RF router design)"
    assert _failed(checks) == []                    # OOF flag, fold fingerprint, pred/decision, semantics


def test_stage4_takes_the_rf_stream_for_its_own_split_only(world, saved):
    stream, checks = _stream(saved[1], world)
    inputs = build_stage4_inputs(world.global_split, [stream], extra_checks=checks, verbose=False)
    assert inputs["rf_router"].fingerprint_scheme == "router"
    prov = inputs.fold_provenance["streams"]["rf_router"]
    assert (prov["n_folds"], prov["seed"]) == (N_STAGE_FOLDS, RANDOM_STATE)
    other = split_frame(make_bank_frame(n=1000), test_size=0.20, seed=1)
    stream, checks = _stream(saved[1], world)
    with pytest.raises(ValidationError, match="split fingerprint"):
        build_stage4_inputs(other, [stream], extra_checks=checks, verbose=False)


def _corrupt(extra, case, rng=np.random.default_rng(0)):
    e = dict(extra)
    n = len(e["train_pred"])
    if case == "not_oof":
        e["train_outputs_oof"] = False
    elif case == "numeric_decisions":
        e["train_decisions"] = np.asarray(e["train_pred"])
    elif case == "unknown_route":
        e["train_decisions"] = np.array(e["train_decisions"], dtype=object)
        e["train_decisions"][0] = "maybe"
    elif case == "reordered_rows":
        e["index_train"] = mut.shuffle_rows(pd.Series(0, index=e["index_train"])).index
    elif case == "duplicated_row":
        e["index_train"] = mut.duplicate_index_label(pd.Series(0, index=e["index_train"])).index
    elif case == "missing_row":
        for k in ("train_pred", "train_confidence", "train_decisions", "train_proba", "train_fold_id"):
            e[k] = np.asarray(e[k])[:-1]
        e["index_train"] = e["index_train"][:-1]
    elif case == "contradictory_pred":
        e["train_pred"] = np.where(np.asarray(e["train_pred"]) == 0, 1, np.asarray(e["train_pred"]))
    elif case == "foreign_fold_plan":
        e["train_fold_id"] = rng.permutation(np.asarray(e["train_fold_id"]))
    assert n                                                             # the block is populated
    return e


@pytest.mark.parametrize("case, error, message", [
    ("not_oof", Stage4InputError, "train_outputs_oof"),
    ("numeric_decisions", Stage4InputError, "numeric"),
    ("unknown_route", Stage4InputError, "unrecognised"),
    ("reordered_rows", ValidationError, "train index"),
    ("duplicated_row", ValidationError, "train index"),
    ("missing_row", ValidationError, "len\\(y_train\\)"),
    ("contradictory_pred", ValidationError, "train_pred agrees"),
    ("foreign_fold_plan", ValidationError, "fold fingerprint"),
])
def test_corrupted_stage4_blocks_are_refused(world, saved, case, error, message):
    art = dict(joblib.load(saved[1]))
    art["extra"] = _corrupt(art["extra"], case)
    with pytest.raises(error, match=message):
        stream, checks = _stream(art, world)
        build_stage4_inputs(world.global_split, [stream], extra_checks=checks, verbose=False)


def test_the_saved_block_is_the_notebooks(world):
    """The fixture's Stage 4 block has exactly the keys Cell 24 saves."""
    named, _, consts = _notebook()
    (call,) = [c for c in named("save_rf_router")
               if any(k.arg == "name" and ast.literal_eval(k.value) == "rf_router_two_pass"
                      for k in c.keywords)]
    extra = next(k.value for k in call.keywords if k.arg == "extra_metadata")
    keys = {ast.literal_eval(k) for k in extra.keys if k is not None}
    for k, v in zip(extra.keys, extra.values):
        if k is None:                                                    # **_common_meta
            keys |= {ast.literal_eval(kk) for kk in consts[v.id].keys}
    assert keys == set(stage4_block(world.stage2, world.crossfit, world.fold_id, world.test_out))


# ----------------------------------------------------------------------
# Stage 1 (MLP) and Stage 2 (RF Router) on one split
# ----------------------------------------------------------------------
def test_the_mlp_and_the_router_describe_the_same_rows_for_stage4(world, saved):
    gs = world.global_split
    X_tr, X_te, _ = quiet(stage1_features, gs["X_train"], gs["X_test"], gs["y_train"])
    mlp = CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(X_tr, gs["y_train"], X_train_raw=gs["X_train"])
    s1, c1 = stage_output_stream(mlp.to_stage_output(X_te, gs["y_test"]).to_dict(), "mlp", "MLP",
                                 len(gs["y_train"]))
    s2, c2 = _stream(saved[1], world)
    inputs = build_stage4_inputs(gs, [s1, s2], extra_checks=c1 + c2, verbose=False)
    assert pd.Index(s1.train_index).equals(pd.Index(s2.train_index)) and \
        pd.Index(s1.test_index).equals(pd.Index(s2.test_index))
    assert {n: inputs[n].fingerprint_scheme for n in ("mlp", "rf_router")} == {
        "mlp": "stage_io", "rf_router": "router"}
    prov = inputs.fold_provenance
    assert {n: (p["n_folds"], p["seed"]) for n, p in prov["streams"].items()} == {
        "mlp": (MLP_TEST_CONFIG["cv_folds"], RANDOM_STATE), "rf_router": (N_STAGE_FOLDS, RANDOM_STATE)}
    assert prov["distinct_partitions"][0]["streams"] == ["mlp", "rf_router"]


# ----------------------------------------------------------------------
# the black-box notebook's Stage 2 calls (read, never executed)
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

    def value(node):
        if isinstance(node, ast.Name) and node.id in consts:
            node = consts[node.id]
        try:
            return ast.literal_eval(node)
        except ValueError:
            return ast.unparse(node)
    return named, value, consts


def _same(node, source: str) -> bool:
    """``node`` is the expression ``source`` (compared after normalising both)."""
    return ast.unparse(node) == ast.unparse(ast.parse(source, mode="eval").body)


def test_the_notebook_runs_stage2_under_the_tested_protocol():
    named, value, consts = _notebook()
    assert (value(consts["N_STAGE_FOLDS"]), value(consts["RANDOM_STATE"])) == (N_STAGE_FOLDS, RANDOM_STATE)
    assert _same(consts["STAGE_FOLDS"], 'make_folds(STAGE2["y_train"], N_STAGE_FOLDS, RANDOM_STATE)')
    configs = {value(next(k.value for k in c.keywords if k.arg == "mode")):
               {k.arg: ast.unparse(k.value) for k in c.keywords} for c in named("RFRouterConfig")}
    for mode in ("single_rf", "two_pass"):
        assert (configs[mode]["n_oof_folds"], configs[mode]["random_state"]) == (
            "N_STAGE_FOLDS", "RANDOM_STATE")
    (cf,) = named("crossfit_rf_router")
    assert ast.unparse(cf.args[2]) == "STAGE_FOLDS"
    make_config = next(k.value for k in cf.keywords if k.arg == "make_config")
    assert "validate_nested=False" in ast.unparse(make_config)
    loads = [c for c in named("load_rf_router")
             if any(k.arg == "expected_split_fingerprint" for k in c.keywords)]
    assert loads, "the saved router is reloaded with its split fingerprint checked"
    assert _same(consts["RF_ROUTER_PATH"], 'f"{STAGE2_OUT}/rf_router_two_pass.joblib"')
    meta_xgb = [c for c in named("train_meta_xgb_stage") if any(
        k.arg == "rf_path" and ast.unparse(k.value) == "RF_ROUTER_PATH" for k in c.keywords)]
    assert len(meta_xgb) == 1


def test_the_notebook_validates_the_glass_bundle_before_taking_constraints_from_it():
    """Cell 18 runs the Stage 4 router contract — label fingerprints for both
    sides (F12), declared Pass 2 populations (F17), split and fold provenance —
    on the GLASS bundle it found, in the same cell, before Cell 19 reads that
    bundle's out-of-fold metrics into the RF Router's operating constraints."""
    cells = [(i, ast.parse("".join(c["source"])))
             for i, c in enumerate(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
             if c["cell_type"] == "code"]

    def calls(name):
        return [(i, n) for i, t in cells for n in ast.walk(t) if isinstance(n, ast.Call)
                and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == name]

    (found_in, find), = calls("find_artifact_for_split")
    assert [ast.unparse(a) for a in find.args] == ["GLASS_ARTIFACT_DIRS", "'glass_router_*.joblib'",
                                                   "SPLIT_FP"]
    (stream_in, stream), = calls("router_stream")
    assert [ast.unparse(a) for a in stream.args] == ["GLASS_BUNDLE", "'glass'", "'GLASS Router'",
                                                     "STAGE2['y_train'].to_numpy()"]
    checked = [(i, c) for i, c in calls("build_stage4_inputs")
               if [ast.unparse(a) for a in c.args] == ["GLOBAL_SPLIT", "[_glass_stream]", "_glass_checks"]]
    assert len(checked) == 1 and checked[0][0] == stream_in == found_in
    assert stream.lineno < checked[0][1].lineno
    readers = [i for i, c in calls("evaluate_router") if "GLASS_BUNDLE" in ast.unparse(c)]
    assert readers and min(readers) > stream_in                       # constraints come after


def test_the_glass_evidence_cell_18_prints_is_what_the_contract_records():
    """Cell 18 prints two notes of the GLASS stream: ``pass2_population_evidence``
    (recorded by ``router_stream``) and ``label_evidence`` (recorded by
    ``build_stage4_inputs`` itself, on the stream the cell holds — so read only
    after that call). Run as the cell runs them, on a current GLASS bundle
    stamped with the real router split fingerprint, both are recorded."""
    (cell,) = [ast.parse(src) for src in ("".join(c["source"]) for c in
               json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code")
               if "# CELL 18:" in src]
    reads = [n for n in ast.walk(cell) if isinstance(n, ast.Subscript)
             and ast.unparse(n.value) == "_glass_stream.notes"]
    (contract,) = [n for n in ast.walk(cell) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", None) == "build_stage4_inputs"]
    assert {n.slice.value for n in reads} == {"label_evidence", "pass2_population_evidence"}
    assert all(n.lineno > contract.lineno for n in reads)
    w = build_glass_world()
    GS = w["GS"]
    bundle = make_glass_router(w, pass2_population="full")
    bundle["split_fingerprint"] = router_split_fingerprint(GS["X_train"], GS["X_test"])  # GLASS Cell 16
    stream, checks = router_stream(bundle, "glass", "GLASS Router", GS["y_train"].to_numpy())
    build_stage4_inputs(GS, [stream], checks, verbose=False)        # the real fingerprint schemes
    assert {n.slice.value: stream.notes[n.slice.value] for n in reads} == {
        "label_evidence": LABELS_FINGERPRINTED, "pass2_population_evidence": PASS2_COMPLETE}
