"""
F29 — the 2x2 Stage 4 crossover through an explicit, validated interface
(PR 36, Commit 3 Step 3C).

                     GLASS Meta-EBM                     black-box Meta-XGB
    GLASS upstream   native (default)                   upstream="glass"
    black-box        upstream="blackbox"                native (default)

Contract
* The upstream arm is declared; every Stage 1-3 artifact must belong to it
  (StageOutput.arm; router layout — GLASS bundle top-level, RF artifact
  "extra"; Stage3Artifact family / arm). A mixed, mislabelled, undeclared or
  'unset' (F21) upstream is refused; F32 is the GLASS arbiter's case of it.
* Same adapters, same split / fold / OOF / label-fingerprint checks (F11,
  F12) in every cell; arbiter slot names stay the arbiter's own.
* Meta-XGB features are model-agnostic (probabilities, router P(subscribe)
  where it routes, route one-hot): GLASS outputs need no reinterpretation.
* The GLASS arbiter's router evidence (F25) reads the routed unit's positive
  rate back from the router probability. The RF Router's probabilities are
  the same blend of its band precision (Pass 1 NPV / Pass 2 precision on
  training OOF), so the evidence is the band rate vs the RF Router's base
  rate — checked per row, exactly as for a GLASS bundle; a router whose
  probabilities are not that blend is refused, never reinterpreted.
* Each artifact records the combination (``stage4_inputs["upstream"]``):
  arm, arbiter, crossover flag and per stage the slot, family, arm,
  probability space, Pass 2 population, label evidence and a content
  fingerprint of the outputs consumed. The arbiter-effect comparison
  (``compare_with_meta_ebm(require_same_upstream=True)``) refuses unless both
  arbiters consumed identical upstream outputs.

Synthetic artifacts in the real formats (tests.support.stage4_world; the RF
Router as band blends, ``band_blend=True``), small fixed Meta-XGB
hyperparameters, no Optuna.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import shutil
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

import shared.stage4.contract as contract
from blackbox_pipeline.models.meta_xgb import build_feature_frames, load_blackbox_stage4_inputs
from blackbox_pipeline.models.meta_xgb.artifact import load_meta_xgb
from blackbox_pipeline.models.meta_xgb.comparison import (
    compare_with_meta_ebm, find_meta_ebm_artifact, load_meta_ebm_for_comparison,
)
from glass_pipeline.meta_ebm import meta_stage as ms
from glass_pipeline.meta_ebm.arbiter import meta_arbiter
from glass_pipeline.meta_ebm.loader import load_stage_outputs
from glass_pipeline.meta_ebm.meta_stage import train_meta_stage
from glass_pipeline.meta_ebm.router_evidence import (
    directional_confidence, fold_base_rates, rule_positive_rate,
)
import shared.stage4 as stage4
from shared.stage4 import ABSTAIN, PASS1, PASS2, Stage4InputError
from shared.validation import ValidationError
from tests.support.meta_xgb_world import run_meta
from tests.support.notebooks import BANK_ROOT
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3

RF_REMAINDER = "OOF Pass 1 remainder (RF router design)"
ARBITERS = ("meta_ebm", "meta_xgb")
UPSTREAMS = ("glass", "blackbox")
NATIVE = {"meta_ebm": "glass", "meta_xgb": "blackbox"}
SLOTS = {"meta_ebm": ("lr", "glass", "ebm"), "meta_xgb": ("mlp", "rf_router", "xgb")}
FAMILY = {"glass": {"stage1": (None, "glass"), "stage2": ("glass_router", "glass"),
                    "stage3": ("ebm", "glass")},
          "blackbox": {"stage1": (None, "blackbox"), "stage2": ("rf_router", "blackbox"),
                       "stage3": ("xgboost", "blackbox")}}


def glass_upstream(w, **kw):
    return [make_stage1(w), make_router(w, pass2_population="full", **kw), make_stage3(w)]


def blackbox_upstream(w, **kw):
    return [make_stage1(w, arm="blackbox", model="mlp"),
            make_router(w, layout="extra", band_blend=True, pass2_population=RF_REMAINDER, **kw),
            make_stage3(w, family="xgboost")]


def _with_router_scheme(mp):
    """train_meta_stage has no extra_fingerprints argument: route it via the loader."""
    real = ms.load_stage_outputs
    mp.setattr(ms, "load_stage_outputs", lambda *a, **k: real(*a, extra_fingerprints=ROUTER_FP, **k))


def run_arbiter(arbiter, w, arts, out, upstream=None, verbose=False):
    """(artifact, path). ``arts``: [stage1, stage2, stage3] objects or paths."""
    if arbiter == "meta_ebm":
        with pytest.MonkeyPatch.context() as mp:
            _with_router_scheme(mp)
            return train_meta_stage(w["GS"], lr_path=arts[0], glass_path=arts[1], ebm_path=arts[2],
                                    artifact_dir=str(out), verbose=verbose, upstream=upstream)
    kw = {} if upstream is None else {"upstream": upstream}
    return run_meta(w, dict(zip(("mlp", "rf", "xgb"), arts)), out, save=True, **kw)


def load_inputs(arbiter, w, arts, upstream=None):
    loader = load_stage_outputs if arbiter == "meta_ebm" else load_blackbox_stage4_inputs
    return loader(w["GS"], *arts, verbose=False, extra_fingerprints=ROUTER_FP, upstream=upstream)


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    """All four cells on one split, from saved upstream artifacts."""
    w = build_world(n=1500, n_test=350)
    root = tmp_path_factory.mktemp("two_by_two")
    files = {}
    for arm, build in (("glass", glass_upstream), ("blackbox", blackbox_upstream)):
        arts = build(w, label_fingerprint=True)
        files[arm] = []
        for stage, art in zip(("stage1", "stage2", "stage3"), arts):
            path = str(root / f"{arm}_{stage}.joblib")
            joblib.dump(art, path)
            files[arm].append(path)
    runs = {}
    for arbiter in ARBITERS:
        for arm in UPSTREAMS:
            up = None if arm == NATIVE[arbiter] else arm
            runs[(arm, arbiter)] = run_arbiter(arbiter, w, files[arm], root / f"{arbiter}_on_{arm}", up)
    return SimpleNamespace(w=w, GS=w["GS"], files=files, runs=runs, root=root)


def upstream_of(run):
    art = run[0]
    return (art if isinstance(art, dict) else vars(art))["stage4_inputs"]["upstream"]


# ----------------------------------------------------------------------
# The four cells: identity, sources, provenance
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", UPSTREAMS)
@pytest.mark.parametrize("arbiter", ARBITERS)
def test_every_cell_records_the_combination_it_ran(grid, arbiter, arm):
    up = upstream_of(grid.runs[(arm, arbiter)])
    assert (up["arm"], up["arbiter"], up["arbiter_arm"], up["crossover"]) == (
        arm, arbiter, NATIVE[arbiter], arm != NATIVE[arbiter])
    assert up["combination"] == f"{arm} upstream -> {arbiter}"
    for (stage, slot), path in zip(zip(("stage1", "stage2", "stage3"), SLOTS[arbiter]), grid.files[arm]):
        rec = up["stages"][stage]
        family, a = FAMILY[arm][stage]
        assert (rec["slot"], rec["arm"], rec["artifact_path"]) == (slot, a, path)
        if family:
            assert rec["model_family"] == family


@pytest.mark.parametrize("arm", UPSTREAMS)
def test_both_arbiters_on_one_upstream_consumed_identical_outputs(grid, arm):
    ebm, xgb = upstream_of(grid.runs[(arm, "meta_ebm")]), upstream_of(grid.runs[(arm, "meta_xgb")])
    for stage in ("stage1", "stage2", "stage3"):
        for key in ("content_fingerprint", "probability_space", "label_evidence", "model_family", "arm"):
            assert ebm["stages"][stage][key] == xgb["stages"][stage][key], (stage, key)
    other = upstream_of(grid.runs[("blackbox" if arm == "glass" else "glass", "meta_ebm")])
    assert all(ebm["stages"][s]["content_fingerprint"] != other["stages"][s]["content_fingerprint"]
               for s in ("stage1", "stage2", "stage3"))


def test_meta_xgb_on_the_glass_upstream_reads_the_glass_outputs_unchanged(grid):
    lr, router, ebm = (joblib.load(p) for p in grid.files["glass"])
    inputs = load_inputs("meta_xgb", grid.w, grid.files["glass"], upstream="glass")
    X_tr, X_te, _ = build_feature_frames(inputs)
    so = lr["stage_output"]
    np.testing.assert_array_equal(X_tr["stage1_proba"], so["train_proba_oof"].to_numpy())
    np.testing.assert_array_equal(X_te["stage1_proba"], so["test_proba"].to_numpy())
    np.testing.assert_array_equal(X_tr["stage3_proba"], ebm.oof["proba_calibrated"])
    np.testing.assert_array_equal(X_te["stage3_proba"], ebm.refit["test_proba_calibrated"])
    for X, side in ((X_tr, "train"), (X_te, "test")):
        dec = np.asarray(router[f"{side}_decisions"]).astype(str)
        routed = dec != "uncertain"
        np.testing.assert_array_equal(X["stage2_proba"].to_numpy()[routed],
                                      np.asarray(router[f"{side}_proba"], float)[routed])
        assert X["stage2_abstain"].to_numpy().tolist() == (~routed).astype(int).tolist()
        assert X["stage2_pass1"].to_numpy().tolist() == (dec == "pass1").astype(int).tolist()
        assert X["stage2_proba"].isna().to_numpy().tolist() == (~routed).tolist()
    assert [inputs[n].model for n in SLOTS["meta_xgb"]] == ["calibrated_lr", "GLASS Router",
                                                            "GLASS EBM (ebm)"]


def test_the_glass_arbiter_on_the_blackbox_upstream_reads_the_blackbox_outputs(grid):
    art, _ = grid.runs[("blackbox", "meta_ebm")]
    mlp, rf, xgb = (joblib.load(p) for p in grid.files["blackbox"])
    e = rf["extra"]
    np.testing.assert_array_equal(art["train_probs"]["lr"],
                                  mlp["stage_output"]["train_proba_oof"].to_numpy())
    np.testing.assert_array_equal(art["train_probs"]["ebm"], xgb.oof["proba_calibrated"])
    np.testing.assert_array_equal(art["train_probs"]["glass"], np.asarray(e["train_proba"], float))
    np.testing.assert_array_equal(art["test_probs"]["glass"], np.asarray(e["test_proba"], float))
    models = {s: r["model"] for s, r in art["stage4_inputs"]["upstream"]["stages"].items()}
    assert models == {"stage1": "mlp", "stage2": "RF Router (two-pass)", "stage3": "XGBoost (xgboost)"}
    dec_tr, dec_te = np.asarray(e["train_decisions"]), np.asarray(e["test_decisions"])
    assert art["glass_decisions_train"].tolist() == np.where(dec_tr == "uncertain", ABSTAIN, dec_tr).tolist()
    assert art["test_preds"]["glass"].tolist() == (dec_te == "pass2").astype(int).tolist()


def test_the_glass_arbiter_reads_the_rf_routers_band_rate_as_its_router_evidence(grid):
    """F25 on the RF Router: the recovered rate is the band's (1 - NPV, precision),
    against the producing router's base rate (fold router for OOF rows)."""
    art, _ = grid.runs[("blackbox", "meta_ebm")]
    rf = joblib.load(grid.files["blackbox"][1])
    e, y = rf["extra"], np.asarray(grid.GS["y_train"], int)
    dec, conf = np.asarray(e["train_decisions"]), np.asarray(e["train_confidence"], float)
    p1, p2 = dec == "pass1", dec == "pass2"
    base = fold_base_rates(y, e["train_fold_id"])
    r = rule_positive_rate(np.asarray(e["train_proba"], float), p1, p2, base)
    np.testing.assert_allclose(r[p1], 1 - conf[p1], atol=1e-12)          # band positive rate
    np.testing.assert_allclose(r[p2], conf[p2], atol=1e-12)
    ev = directional_confidence(r, p1, p2, base)
    np.testing.assert_allclose(ev[p1], np.maximum(0, base[p1] - (1 - conf[p1])), atol=1e-12)
    np.testing.assert_allclose(ev[p2], np.maximum(0, conf[p2] - base[p2]), atol=1e-12)
    assert art["thresholds"]["router_confidence"].startswith("RF Router band positive rate")
    assert grid.runs[("glass", "meta_ebm")][0]["thresholds"]["router_confidence"].startswith(
        "rule positive rate")


@pytest.mark.parametrize("arm", UPSTREAMS)
def test_meta_ebm_decisions_reconstruct_from_its_saved_artifact(grid, arm):
    art, path = grid.runs[(arm, "meta_ebm")]
    loaded = joblib.load(path)
    t, tp = loaded["thresholds"], loaded["test_probs"]
    dec = np.asarray(loaded["glass_decisions_test"])
    args = (tp["lr"], tp["ebm"], tp["glass"], dec == PASS1, dec == PASS2, t["lr"], t["ebm"],
            loaded["weights"])
    pred_na, prob_na, _ = meta_arbiter(*args, allow_abstain=False, glass_base_rate=t["glass"])
    pred_a, _, _ = meta_arbiter(*args, allow_abstain=True, glass_base_rate=t["glass"],
                                min_weighted_confidence=t["min_weighted_confidence"])
    np.testing.assert_array_equal(pred_na, loaded["test_preds"]["no_abstain"])
    np.testing.assert_array_equal(pred_a, loaded["test_preds"]["with_abstain"])
    np.testing.assert_array_equal(prob_na, loaded["test_probs"]["meta"])


@pytest.mark.parametrize("arm", UPSTREAMS)
def test_meta_xgb_decisions_reconstruct_from_its_reloaded_artifact(grid, arm):
    art, path = grid.runs[(arm, "meta_xgb")]
    loaded = load_meta_xgb(path, expected_split_fingerprint=art.split_fingerprint)
    up = None if arm == "blackbox" else arm
    X_te = build_feature_frames(load_inputs("meta_xgb", grid.w, grid.files[arm], up))[1]
    serving = loaded.learner.serving_model
    np.testing.assert_array_equal(serving.predict_proba_raw(X_te), loaded.test["proba_raw"])
    np.testing.assert_array_equal(serving.predict_proba_calibrated(X_te), loaded.test["proba_calibrated"])
    assert loaded.stage4_inputs["upstream"] == art.stage4_inputs["upstream"]


@pytest.mark.parametrize("arm", UPSTREAMS)
@pytest.mark.parametrize("arbiter", ARBITERS)
def test_split_fold_and_label_provenance_hold_in_every_cell(grid, arbiter, arm):
    art = grid.runs[(arm, arbiter)][0]
    s4 = (art if isinstance(art, dict) else vars(art))["stage4_inputs"]
    s1, s2, s3 = SLOTS[arbiter]
    streams = s4["folds"]["streams"]
    assert {n: (p["n_folds"], p["seed"], p["reproduced_by_shared_fold_assignment"])
            for n, p in streams.items()} == {s1: (10, 42, True), s2: (5, 42, True), s3: (5, 42, True)}
    assert s4["folds"]["shared_partitions"] == [[s2, s3, 5]]
    assert all(r["passed"] for r in s4["validation"])
    assert s4["split_fingerprint"] == grid.w["fp_io"]
    ev = {stage: rec["label_evidence"] for stage, rec in s4["upstream"]["stages"].items()}
    assert ev == {"stage1": contract.LABELS_CARRIED, "stage2": contract.LABELS_FINGERPRINTED,
                  "stage3": contract.LABELS_CARRIED}
    spaces = {stage: rec["probability_space"] for stage, rec in s4["upstream"]["stages"].items()}
    assert spaces["stage1"].startswith("calibrated") and spaces["stage3"].startswith("calibrated per fold")
    assert spaces["stage2"] == "router-native P(subscribe) (positive column)"
    want = {"glass": "full", "blackbox": "oof_remainder"}[arm]                      # F17, recorded
    assert s4["upstream"]["stages"]["stage2"]["pass2_population"] == {"cross_fit": want, "full_train": want}


def test_a_crossover_run_is_deterministic_and_its_artifact_reloads(grid, tmp_path):
    art, path = grid.runs[("blackbox", "meta_ebm")]
    again, _ = run_arbiter("meta_ebm", grid.w, grid.files["blackbox"], tmp_path, "blackbox")
    for key in ("no_abstain", "with_abstain"):
        np.testing.assert_array_equal(again["test_preds"][key], art["test_preds"][key])
    assert again["weights"] == art["weights"] and again["thresholds"] == art["thresholds"]
    assert again["stage4_inputs"]["upstream"] == art["stage4_inputs"]["upstream"]
    found = load_meta_ebm_for_comparison(grid.GS, path, ROUTER_FP)
    assert found["stage4_inputs"]["upstream"]["crossover"] is True


def test_native_runs_are_unchanged_by_the_upstream_interface(grid):
    for arbiter, arm in NATIVE.items():
        default = load_inputs(arbiter, grid.w, grid.files[arm])
        explicit = load_inputs(arbiter, grid.w, grid.files[arm], upstream=arm)
        for name in SLOTS[arbiter]:
            np.testing.assert_array_equal(default[name].train_proba, explicit[name].train_proba)
            assert stage4.stream_fingerprint(default[name]) == stage4.stream_fingerprint(explicit[name])
        assert default.upstream == explicit.upstream and default.upstream["crossover"] is False


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arbiter", ARBITERS)
def test_an_undeclared_crossover_is_refused_with_the_declaration_it_needs(grid, arbiter):
    other = "blackbox" if NATIVE[arbiter] == "glass" else "glass"
    with pytest.raises(ValidationError) as exc:
        load_inputs(arbiter, grid.w, grid.files[other])
    hints = [line.strip()[:8] for line in str(exc.value).splitlines() if f"upstream='{other}'" in line]
    assert hints == ["[stage1]", "[stage2]", "[stage3]"]          # one per artifact


@pytest.mark.parametrize("upstream, swap, message", [
    ("glass", 1, "glass: GLASS Router bundle (top-level Stage 4 layout)"),
    ("glass", 2, "GLASS EBM artifact"),
    ("blackbox", 0, "black-box Stage 1 output"),
    ("blackbox", 2, "black-box XGBoost artifact"),
])
@pytest.mark.parametrize("arbiter", ARBITERS)
def test_a_mixed_upstream_is_refused_in_every_cell(grid, arbiter, upstream, swap, message):
    other = "blackbox" if upstream == "glass" else "glass"
    arts = list(grid.files[upstream])
    arts[swap] = grid.files[other][swap]
    up = None if upstream == NATIVE[arbiter] else upstream
    name = SLOTS[arbiter][swap]
    with pytest.raises(ValidationError) as exc:
        load_inputs(arbiter, grid.w, arts, upstream=up)
    msg = message.replace("glass: ", f"{name}: ") if message.startswith("glass:") else message
    assert msg in str(exc.value)


@pytest.mark.parametrize("arbiter", ARBITERS)
def test_an_unset_stage3_label_is_refused_in_a_crossover_too_f21(grid, arbiter):
    arm = "glass" if arbiter == "meta_xgb" else "blackbox"
    arts = list(grid.files[arm])
    ebm = joblib.load(arts[2])
    ebm.model_family, ebm.config = "unset", {**ebm.config, "arm": "unset"}
    arts[2] = ebm
    with pytest.raises(ValidationError, match="PR 36 F21"):
        load_inputs(arbiter, grid.w, arts, upstream=arm)


@pytest.mark.parametrize("upstream", ["rf", "", "GLASS", 1])
@pytest.mark.parametrize("arbiter", ARBITERS)
def test_an_unknown_upstream_is_refused(grid, arbiter, upstream):
    with pytest.raises(Stage4InputError, match="upstream must be one of"):
        load_inputs(arbiter, grid.w, grid.files["glass"], upstream=upstream)


@pytest.mark.parametrize("case, message", [
    ("free-form RF probabilities", "train probabilities = band blends at fold base rates"),
    ("band blend at the full base rate", "train probabilities = band blends at fold base rates"),
    ("no training_base_rate", "RF Router artifact with training_base_rate"),
    ("test confidence reassigned", "test probabilities = band blends at the training base rate"),
])
def test_the_glass_arbiter_refuses_an_rf_router_it_cannot_read_as_a_band_blend(grid, case, message):
    w = grid.w
    arts = blackbox_upstream(w)
    rf = arts[1]
    e = rf["extra"]
    if case == "free-form RF probabilities":
        arts[1] = make_router(w, layout="extra", pass2_population=RF_REMAINDER)
        arts[1].update(training_base_rate=float(np.mean(w["GS"]["y_train"])), mode="two_pass")
    elif case == "band blend at the full base rate":
        d, c = np.asarray(e["train_decisions"]), np.asarray(e["train_confidence"])
        b = rf["training_base_rate"]
        e["train_proba"] = np.where(d == "pass1", b * (1 - c), np.where(d == "pass2", b + (1 - b) * c, b))
    elif case == "no training_base_rate":
        del rf["training_base_rate"]
    else:
        c = np.asarray(e["test_confidence"], float).copy()
        c[np.asarray(e["test_decisions"]) == "pass2"] -= 0.05
        e["test_confidence"] = c
    with pytest.raises(ValidationError, match=message):
        load_inputs("meta_ebm", w, arts, upstream="blackbox")


# ----------------------------------------------------------------------
# The arbiter comparison
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", UPSTREAMS)
def test_the_arbiter_effect_comparison_verifies_the_upstream_was_held_fixed(grid, arm):
    out = compare_with_meta_ebm(grid.GS, grid.runs[(arm, "meta_ebm")][1], grid.runs[(arm, "meta_xgb")][0],
                                n_boot=20, verbose=False, extra_fingerprints=ROUTER_FP,
                                require_same_upstream=True)
    up = out["protocol"]["upstream"]
    assert (up["held_fixed"], up["required"], up["why_not"]) == (True, True, None)
    assert (up["meta_ebm"], up["meta_xgb"]) == (f"{arm} upstream -> meta_ebm", f"{arm} upstream -> meta_xgb")


def test_a_cascade_comparison_records_that_the_upstreams_differ(grid):
    ebm_path, xgb = grid.runs[("glass", "meta_ebm")][1], grid.runs[("blackbox", "meta_xgb")][0]
    out = compare_with_meta_ebm(grid.GS, ebm_path, xgb, n_boot=20, verbose=False,
                                extra_fingerprints=ROUTER_FP)                 # Cell 36: allowed
    assert out["protocol"]["upstream"]["held_fixed"] is False
    assert "upstream arms differ" in out["protocol"]["upstream"]["why_not"]
    with pytest.raises(Stage4InputError, match="upstream held fixed: upstream arms differ"):
        compare_with_meta_ebm(grid.GS, ebm_path, xgb, n_boot=20, verbose=False,
                              extra_fingerprints=ROUTER_FP, require_same_upstream=True)


def test_an_arbiter_comparison_over_different_upstream_outputs_is_refused(grid, tmp_path):
    """Same arm, another router artifact: not the same upstream."""
    arts = list(grid.files["glass"])
    router = joblib.load(arts[1])
    p = np.asarray(router["test_proba"], float).copy()
    routed = np.asarray(router["test_decisions"]) != "uncertain"
    i = np.flatnonzero(np.asarray(router["test_decisions"]) == "pass2")[0]
    router["test_confidence"] = np.asarray(router["test_confidence"], float).copy()
    router["test_confidence"][i] = min(0.99, router["test_confidence"][i] + 0.01)
    b = router["training_base_rate"]
    p[i] = b + (1 - b) * router["test_confidence"][i]
    router["test_proba"] = np.where(routed, p, np.nan)
    arts[1] = router
    xgb, _ = run_arbiter("meta_xgb", grid.w, arts, tmp_path, "glass")
    with pytest.raises(Stage4InputError, match="stage2 outputs differ"):
        compare_with_meta_ebm(grid.GS, grid.runs[("glass", "meta_ebm")][1], xgb, n_boot=20, verbose=False,
                              extra_fingerprints=ROUTER_FP, require_same_upstream=True)


def test_an_artifact_written_before_the_upstream_record_cannot_prove_it(grid):
    legacy = copy.deepcopy(joblib.load(grid.runs[("glass", "meta_ebm")][1]))
    del legacy["stage4_inputs"]["upstream"]
    xgb = grid.runs[("glass", "meta_xgb")][0]
    out = compare_with_meta_ebm(grid.GS, legacy, xgb, n_boot=20, verbose=False,
                                extra_fingerprints=ROUTER_FP)
    assert out["protocol"]["upstream"]["held_fixed"] is False
    with pytest.raises(Stage4InputError, match="records no upstream"):
        compare_with_meta_ebm(grid.GS, legacy, xgb, n_boot=20, verbose=False, extra_fingerprints=ROUTER_FP,
                              require_same_upstream=True)


def test_the_meta_ebm_finder_returns_the_upstream_it_is_asked_for(grid, tmp_path):
    """A newer crossover Meta-EBM in the same folder is not "the GLASS Meta-EBM"."""
    native, crossover = (grid.runs[(arm, "meta_ebm")][1] for arm in UPSTREAMS)
    for i, src in enumerate((native, crossover)):                    # crossover saved last
        dst = tmp_path / f"meta_ebm_{i}.joblib"
        shutil.copy(src, dst)
        os.utime(dst, (1_700_000_000 + i, 1_700_000_000 + i))
    found, path = find_meta_ebm_artifact([tmp_path], grid.GS, ROUTER_FP, verbose=False)
    assert path.endswith("meta_ebm_0.joblib") and found["stage4_inputs"]["upstream"]["arm"] == "glass"
    found, path = find_meta_ebm_artifact([tmp_path], grid.GS, ROUTER_FP, verbose=False,
                                         upstream="blackbox")
    assert path.endswith("meta_ebm_1.joblib") and found["stage4_inputs"]["upstream"]["crossover"]


# ----------------------------------------------------------------------
# The notebook's crossover cells (read, never executed)
# ----------------------------------------------------------------------
def _calls(cell_title):
    cells = json.loads((BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb")
                       .read_text(encoding="utf-8"))["cells"]
    (src,) = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"
              and cell_title in "".join(c["source"])]
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]


def _kw(call):
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def test_the_notebook_declares_both_crossovers_and_compares_with_the_upstream_held_fixed():
    calls = _calls("CELL 37: STAGE 4 — 2×2 ARBITER CROSSOVER")
    (glass_arbiter,) = [c for c in calls if ast.unparse(c.func) == "train_meta_stage"]
    (meta_xgb,) = [c for c in calls if ast.unparse(c.func) == "train_meta_xgb_stage"]
    assert _kw(glass_arbiter)["upstream"] == "'blackbox'"
    assert {k: _kw(glass_arbiter)[k] for k in ("lr_path", "glass_path", "ebm_path")} == {
        "lr_path": "MLP_S1_PATH", "glass_path": "RF_ROUTER_PATH", "ebm_path": "XGB_STAGE3_PATH"}
    assert _kw(meta_xgb)["upstream"] == "'glass'"
    assert {k: _kw(meta_xgb)[k] for k in ("mlp_path", "rf_path", "xgb_path")} == {
        "mlp_path": "GLASS_S1_PATH", "rf_path": "GLASS_BUNDLE_PATH", "xgb_path": "_ebm_path"}
    comparisons = [c for c in _calls("CELL 38: STAGE 4 — CROSSOVER RESULTS")
                   if ast.unparse(c.func) == "compare_with_meta_ebm"]
    assert [ast.unparse(c.args[1]) + " vs " + ast.unparse(c.args[2]) for c in comparisons] == [
        "META_EBM_PATH vs XO_XGB_ON_GLASS", "XO_ARB_ON_BB_PATH vs META_XGB"]
    assert all(_kw(c)["require_same_upstream"] == "True" for c in comparisons)
