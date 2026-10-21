"""
GLASS Stage 4 Meta-EBM end to end: ``train_meta_stage`` on the shared Stage 4
inputs, the Stage 3 -> Stage 4 handoff, and the saved artifact.

Contract: every configuration value (Stage 4 thresholds, trust weights,
abstention cut) is reproduced from the training streams and training labels
alone, and the stored test decisions from the frozen configuration; the
router's evidence is read at each row's own router base rate (fold model on
out-of-fold training rows, full training rate on test rows; F25) and only a
GLASS rule-blend router whose stored probabilities, rule precisions and base
rate agree is accepted; the router's values where it does not route change
nothing; runs are deterministic; the artifact round-trips and is matched to
its split. A real
``train_ebm_stage`` artifact hands Stage 4 its per-fold calibrated OOF column
(train) and its all-OOF calibrated refit column (test), on the shared 5-fold
partition and the training rows' own index; anything misaligned,
mislabelled, re-ordered or made before the PR 36 leakage fixes is refused
(F26; Stage 3 artifacts without F22's per-fold metadata whether or not their
calibration gate fired), and a Stage 3 artifact whose per-fold and all-rows calibration gates
disagree is consumed per fold, not refused or switched to raw (F28).

Input-contract basics (formats, router semantics, fingerprints, in-sample
guards, test-label isolation) are in test_meta_ebm_stage4_inputs.py.
"""

from __future__ import annotations

import contextlib

import joblib
import numpy as np
import pandas as pd
import pytest

import glass_pipeline.ebm.ebm_stage as ebm_stage_mod
from blackbox_pipeline.models.meta_xgb.comparison import (
    find_meta_ebm_artifact, load_meta_ebm_for_comparison,
)
from glass_pipeline.ebm.config import EBMStage3Config
from glass_pipeline.ebm.ebm_stage import train_ebm_stage
from glass_pipeline.meta_ebm import meta_stage as ms
from glass_pipeline.meta_ebm.arbiter import meta_arbiter
from glass_pipeline.meta_ebm.calibration import compute_calibration
from glass_pipeline.meta_ebm.evaluation import compute_metrics, evaluate_with_abstention
from glass_pipeline.meta_ebm.loader import PASS1, PASS2, load_stage_outputs
from glass_pipeline.meta_ebm.meta_stage import find_recall_threshold, train_meta_stage
from glass_pipeline.meta_ebm.router_evidence import fold_base_rates
from glass_pipeline.meta_ebm.tuning import tune_arbiter_threshold
from glass_pipeline.meta_ebm.weighting import compute_hybrid_weights
from shared.stage3.calibration import Stage3Calibrator
from shared.stage3.data import FoldPlan, PassthroughPipeline, Stage3Block
from shared.stage4 import ABSTENTION_SEARCH, Stage4InputError
from shared.validation import ValidationError
from tests.stage3.conftest import EBM_TEST_BAGS, EBM_TEST_PARAMS
from tests.support import mutations as mut
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3

CONFIG_KEYS = ("thresholds", "weights", "abstention", "calibration", "disagreement")


@pytest.fixture(autouse=True)
def _router_scheme(monkeypatch):
    """train_meta_stage has no extra_fingerprints arg: route it via the loader."""
    real = ms.load_stage_outputs
    monkeypatch.setattr(ms, "load_stage_outputs",
                        lambda *a, **k: real(*a, extra_fingerprints=ROUTER_FP, **k))


@pytest.fixture
def world():
    return build_world()


@pytest.fixture
def arts(world):
    return {"lr": make_stage1(world), "router": make_router(world), "ebm": make_stage3(world)}


def dump(tmp_path, arts) -> dict:
    out = {}
    for k, v in arts.items():
        out[k] = str(tmp_path / f"{k}.joblib")
        joblib.dump(v, out[k])
    return out


def inputs(world, arts, tmp_path):
    p = dump(tmp_path, arts)
    return load_stage_outputs(world["GS"], p["lr"], p["router"], p["ebm"], verbose=False,
                              extra_fingerprints=ROUTER_FP)


def run(world, arts, tmp_path, **kw):
    p = dump(tmp_path, arts)
    return train_meta_stage(world["GS"], lr_path=p["lr"], glass_path=p["router"],
                            ebm_path=p["ebm"], artifact_dir=str(tmp_path / "meta"),
                            verbose=False, **kw)


# ----------------------------------------------------------------------
# configuration: train streams only, then frozen for test
# ----------------------------------------------------------------------
def test_every_configured_value_reproduces_from_the_train_streams(world, arts, tmp_path):
    art, _ = run(world, arts, tmp_path)
    inp = inputs(world, arts, tmp_path)
    y = inp.y_train
    lr, gl, ebm = inp["lr"], inp["glass"], inp["ebm"]
    p1, p2 = gl.train_decisions == PASS1, gl.train_decisions == PASS2
    covered = p1 | p2

    t = art["thresholds"]
    fold = gl.train_fold_id
    assert t["glass"] == y.mean()                          # the router's training base rate
    assert t["glass_train_base_rates"] == {k: y[fold != k].mean() for k in range(5)}
    assert "F25" in t["router_confidence"]
    assert t["lr"] == find_recall_threshold(y, lr.train_proba, 0.70)
    assert t["ebm"] == find_recall_threshold(y, ebm.train_proba, 0.70)
    cal = {"lr": compute_calibration(y, lr.train_proba), "ebm": compute_calibration(y, ebm.train_proba),
           "glass": compute_calibration(y[covered], gl.train_proba[covered])}
    assert {k: art["calibration"][k] for k in cal} == cal
    weights = compute_hybrid_weights(cal["lr"], cal["ebm"], cal["glass"], y, lr.train_proba,
                                     ebm.train_proba, p2.astype(int), t["lr"], t["ebm"],
                                     glass_covered_mask=covered)
    assert art["weights"] == {k: weights[k] for k in ("lr", "ebm", "glass")}
    best = tune_arbiter_threshold(meta_arbiter, (lr.train_proba, ebm.train_proba, gl.train_proba),
                                  (p1, p2), y, t["lr"], t["ebm"], art["weights"],
                                  glass_base_rate=fold_base_rates(y, fold))
    assert art["abstention"]["min_weighted_confidence"] == best["min_weighted_confidence"]
    ab = art["abstention"]                                  # exact search (F30): the cut is a
    assert ab["search"] == ABSTENTION_SEARCH                # training confidence, and applying
    pred_tr, _, _, conf_tr = meta_arbiter(                  # it gives the selected coverage
        lr.train_proba, ebm.train_proba, gl.train_proba, p1, p2, t["lr"], t["ebm"], art["weights"],
        allow_abstain=False, return_confidence=True, glass_base_rate=fold_base_rates(y, fold))
    assert ab["min_weighted_confidence"] in conf_tr
    assert (conf_tr >= ab["min_weighted_confidence"]).mean() == ab["train_coverage"] >= 0.5
    assert ab["n_candidates"] == len(np.unique(conf_tr))

    # the test side is scored once, at the frozen configuration
    q1, q2 = gl.test_decisions == PASS1, gl.test_decisions == PASS2
    args = (lr.test_proba, ebm.test_proba, gl.test_proba, q1, q2, t["lr"], t["ebm"], art["weights"])
    pred_na, prob_na, _ = meta_arbiter(*args, allow_abstain=False, glass_base_rate=y.mean())
    pred_a, prob_a, _ = meta_arbiter(*args, min_weighted_confidence=t["min_weighted_confidence"],
                                     glass_base_rate=y.mean())
    np.testing.assert_array_equal(art["test_preds"]["no_abstain"], pred_na)
    np.testing.assert_array_equal(art["test_preds"]["with_abstain"], pred_a)
    np.testing.assert_array_equal(art["test_probs"]["meta"], prob_na)
    assert art["no_abstain"] == compute_metrics(inp.y_test, pred_na, prob_na)
    assert art["with_abstain"] == evaluate_with_abstention(inp.y_test, pred_a, prob_a)


@pytest.mark.parametrize("target", [0.5, 0.9])
def test_target_recall_drives_the_stage4_thresholds(world, arts, tmp_path, target):
    art, _ = run(world, arts, tmp_path, target_recall=target)
    inp = inputs(world, arts, tmp_path)
    y = inp.y_train
    assert art["thresholds"]["target_recall"] == target
    for name in ("lr", "ebm"):
        t = art["thresholds"][name]
        assert (inp[name].train_proba[y == 1] >= t).mean() >= target
    assert art["thresholds"]["glass"] == y.mean()           # not a recall cut (F25)


def test_router_evidence_is_read_at_each_rows_own_base_rate_f25(world, arts, tmp_path, monkeypatch):
    """Out-of-fold rows were scored by fold routers trained at their own base
    rates; test rows by the full-train router."""
    seen = []
    real = ms.meta_arbiter

    def spy(*a, **k):
        seen.append((len(a[0]), np.broadcast_to(k["glass_base_rate"], (len(a[0]),)).copy()))
        return real(*a, **k)

    monkeypatch.setattr(ms, "meta_arbiter", spy)
    run(world, arts, tmp_path)
    y = world["GS"]["y_train"].to_numpy()
    fold = np.asarray(arts["router"]["train_fold_id"])
    n_tr, n_te = len(y), len(world["GS"]["y_test"])
    train_calls = [b for n, b in seen if n == n_tr]
    test_calls = [b for n, b in seen if n == n_te]
    assert len(train_calls) == 1 and len(test_calls) == 2 and len(seen) == 3   # tuning: one call (F30)
    for b in train_calls:
        np.testing.assert_array_equal(b, fold_base_rates(y, fold))
        assert len(np.unique(b)) > 1
    for b in test_calls:
        assert (b == y.mean()).all()


def test_repeated_runs_are_identical(world, arts, tmp_path):
    a1, _ = run(world, arts, tmp_path)
    a2, _ = run(world, arts, tmp_path)
    for key in CONFIG_KEYS + ("no_abstain", "with_abstain"):
        np.testing.assert_equal(a2[key], a1[key], err_msg=key)
    for key in ("no_abstain", "with_abstain"):
        np.testing.assert_array_equal(a2["test_preds"][key], a1["test_preds"][key])
    np.testing.assert_array_equal(a2["test_probs"]["meta"], a1["test_probs"]["meta"])


def test_router_values_where_it_does_not_route_are_inert(world, arts, tmp_path):
    """The loader sets NaN to 0.5 on abstain rows; any other value must do as little."""
    base, _ = run(world, arts, tmp_path)
    r = arts["router"]
    for side in ("train", "test"):
        abstain = np.asarray(r[f"{side}_decisions"]) == "uncertain"
        r[f"{side}_proba"] = np.where(abstain, 0.99, r[f"{side}_proba"])
    moved, _ = run(world, arts, tmp_path)
    for key in CONFIG_KEYS + ("no_abstain", "with_abstain"):
        np.testing.assert_equal(moved[key], base[key], err_msg=key)     # NaN-safe
    for key in ("no_abstain", "with_abstain"):
        np.testing.assert_array_equal(moved["test_preds"][key], base["test_preds"][key])
    np.testing.assert_array_equal(moved["test_probs"]["meta"], base["test_probs"]["meta"])


def _tamper_router(world, r, case):
    y = world["GS"]["y_train"].to_numpy()
    if case == "rf_router_layout":
        return make_router(world, layout="extra")
    if case == "no_training_base_rate":
        del r["training_base_rate"]
    elif case == "other_training_base_rate":
        r["training_base_rate"] += 0.01
    elif case == "train_rule_precision_reassigned":
        r["train_confidence"] = np.random.default_rng(1).permutation(r["train_confidence"])
    elif case == "test_rule_precision_reassigned":
        r["test_confidence"] = np.random.default_rng(1).permutation(r["test_confidence"])
    elif case in ("train_rule_precision_nan", "test_rule_precision_inf"):
        side, bad = ("train", np.nan) if case.startswith("train") else ("test", np.inf)
        prec = np.asarray(r[f"{side}_confidence"], dtype=float).copy()
        prec[np.flatnonzero(np.asarray(r[f"{side}_decisions"]) == "pass2")[0]] = bad
        r[f"{side}_confidence"] = prec
    elif case == "train_blended_at_the_full_base_rate":
        d, c, b = np.asarray(r["train_decisions"]), np.asarray(r["train_confidence"]), y.mean()
        r["train_proba"] = np.where(d == "pass1", b * (1 - c),
                                    np.where(d == "pass2", b + (1 - b) * c, np.nan))
    return r


@pytest.mark.parametrize("case, message", [
    ("rf_router_layout", "GLASS Router bundle with training_base_rate"),
    ("no_training_base_rate", "GLASS Router bundle with training_base_rate"),
    ("other_training_base_rate", "training_base_rate = GLOBAL_SPLIT"),
    ("train_rule_precision_reassigned", "train probabilities = rule blends"),
    ("test_rule_precision_reassigned", "test probabilities = rule blends"),
    ("train_blended_at_the_full_base_rate", "train probabilities = rule blends at fold base rates"),
    ("train_rule_precision_nan", "train probabilities = rule blends at fold base rates"),
    ("test_rule_precision_inf", "test probabilities = rule blends at the training base rate"),
])
def test_a_router_whose_rule_evidence_cannot_be_read_is_refused_f25(world, arts, tmp_path,
                                                                   case, message):
    """The arbiter reads each routed row's rule rate back from the router's
    probability; that is exact only for a GLASS rule blend at the base rate
    the producing model used. The untouched bundle loads, so each check can fail."""
    inputs(world, arts, tmp_path)
    arts["router"] = _tamper_router(world, arts["router"], case)
    with pytest.raises((Stage4InputError, ValidationError), match=message):
        inputs(world, arts, tmp_path)


def test_unrouted_rows_may_store_any_rule_precision(world, arts, tmp_path):
    """Only routed rows carry a rule; the loader reads nothing else."""
    r = arts["router"]
    for side in ("train", "test"):
        prec = np.asarray(r[f"{side}_confidence"], dtype=float).copy()
        prec[np.asarray(r[f"{side}_decisions"]) == "uncertain"] = np.nan
        r[f"{side}_confidence"] = prec
    inputs(world, arts, tmp_path)


# ----------------------------------------------------------------------
# artifact
# ----------------------------------------------------------------------
def test_saved_artifact_round_trips_and_is_matched_to_its_split(world, arts, tmp_path):
    art, path = run(world, arts, tmp_path)
    loaded = joblib.load(path)
    assert loaded["schema"] == "meta_ebm/2" and loaded["train_probs_are_oof"] is True
    for key in CONFIG_KEYS + ("no_abstain", "with_abstain", "split_fingerprints"):
        np.testing.assert_equal(loaded[key], art[key], err_msg=key)     # NaN-safe
    GS = world["GS"]
    assert loaded["index_train"] == list(GS["X_train"].index)
    assert loaded["index_test"] == list(GS["X_test"].index)
    for key in ("no_abstain", "with_abstain", "lr", "ebm", "glass"):
        assert len(loaded["test_preds"][key]) == len(GS["X_test"])
    inp = inputs(world, arts, tmp_path)
    for name, s in inp.streams.items():
        np.testing.assert_array_equal(loaded["train_probs"][name], s.train_proba)
        np.testing.assert_array_equal(loaded["train_fold_ids"][name], s.train_fold_id)
    assert loaded["stage4_inputs"]["streams"]["ebm"]["train_is_oof"] is True

    # the black-box arm finds it by split, never by recency alone
    assert load_meta_ebm_for_comparison(GS, path, ROUTER_FP)["schema"] == "meta_ebm/2"
    found, found_path = find_meta_ebm_artifact([tmp_path / "meta"], GS, ROUTER_FP, verbose=False)
    assert found_path == path
    other = build_world(seed=1)["GS"]
    with pytest.raises(Stage4InputError, match="another split"):
        load_meta_ebm_for_comparison(other, path, ROUTER_FP)
    with pytest.raises(Stage4InputError, match="meta_ebm/2"):
        load_meta_ebm_for_comparison(GS, {**loaded, "schema": "meta_ebm/1"}, ROUTER_FP)


# ----------------------------------------------------------------------
# Stage 3 -> Stage 4 handoff from a real train_ebm_stage artifact
# ----------------------------------------------------------------------
@contextlib.contextmanager
def _cheap_bags():
    real = ebm_stage_mod.make_ebm_factory

    def make(config, block):
        factory = real(config, block)
        factory.extra_params = dict(EBM_TEST_BAGS)
        return factory

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ebm_stage_mod, "make_ebm_factory", make)
        yield


@pytest.fixture(scope="module")
def real_stage3(tmp_path_factory):
    """``train_ebm_stage`` on the Stage 4 world's split: the shared 5-fold
    partition, the stage_io fingerprint glass_cascade stamps, saved and reloaded."""
    world = build_world()                                  # deterministic: same split as ``world``
    GS = world["GS"]
    cfg = EBMStage3Config(n_tune_folds=5, n_eval_folds=2, n_jobs=1, verbose=False,
                          show_progress_bar=False, expected_features=["f1", "f2"],
                          artifact_dir=str(tmp_path_factory.mktemp("stage3")))
    block = Stage3Block.from_global_split(GS, PassthroughPipeline, cfg,
                                          split_fingerprint=world["fp_io"])
    with _cheap_bags():
        art, path = train_ebm_stage(block=block, config=cfg, params=dict(EBM_TEST_PARAMS))
    return joblib.load(path), path


def test_real_stage3_artifact_hands_over_its_per_fold_calibrated_oof(world, real_stage3, tmp_path):
    art, path = real_stage3
    arts = {"lr": make_stage1(world), "router": make_router(world)}
    p = dump(tmp_path, arts)
    inp = load_stage_outputs(world["GS"], p["lr"], p["router"], path, verbose=False,
                             extra_fingerprints=ROUTER_FP)
    ebm = inp["ebm"]
    assert art.calibration["applied"] and len(art.calibration["fold_applied"]) == 5
    np.testing.assert_array_equal(ebm.train_proba, art.oof["proba_calibrated"])
    np.testing.assert_array_equal(ebm.test_proba, art.refit["test_proba_calibrated"])
    assert ebm.train_source == "Stage3Artifact.oof['proba_calibrated']"
    assert ebm.fingerprint_scheme == "stage_io"
    assert ebm.notes["calibration_fold_applied"] == art.calibration["fold_applied"]
    assert ebm.standalone_threshold == art.threshold["oof_calibrated_space"]["threshold"]

    folds = inp.fold_provenance
    assert ["glass", "ebm", 5] in folds["shared_partitions"]
    assert {"streams": ["lr", "ebm"], "n_folds": [10, 5]} in [
        {k: d[k] for k in ("streams", "n_folds")} for d in folds["distinct_partitions"]]
    assert folds["streams"]["ebm"]["reproduced_by_shared_fold_assignment"]
    assert all(r["passed"] for r in inp.validation)

    meta, _ = train_meta_stage(world["GS"], lr_path=p["lr"], glass_path=p["router"],
                               ebm_path=path, artifact_dir=str(tmp_path / "meta"), verbose=False)
    np.testing.assert_array_equal(meta["train_probs"]["ebm"], art.oof["proba_calibrated"])
    assert meta["stage4_inputs"]["streams"]["ebm"]["n_folds"] == 5


def _tamper(art, case):
    y = np.asarray(art.y_train)
    if case == "reordered_rows":
        art.index_train = list(mut.shuffle_rows(pd.Series(0, index=art.index_train)).index)
    elif case == "foreign_row":
        art.index_train = [-1] + list(art.index_train[1:])
    elif case == "short_train_stream":
        art.oof["proba_calibrated"] = art.oof["proba_calibrated"][:-1]
    elif case == "other_labels":
        art.y_train = [int(v) for v in 1 - y]
    elif case == "other_fold_partition":
        art.oof["fold_id"] = art.fold_plan["fold_id"] = np.roll(np.asarray(art.oof["fold_id"]), 1)
    elif case == "non_finite":
        art.oof["proba_calibrated"] = np.where(np.arange(len(y)) == 0, np.nan,
                                               art.oof["proba_calibrated"])
    elif case == "global_feature_scope":
        art.feature_fit_scope = "global"
    elif case == "calibrated_column_on_an_identity_fold":
        art.calibration = {**art.calibration, "fold_applied": [False] + art.calibration["fold_applied"][1:]}
    return art


@pytest.mark.parametrize("case, message", [
    ("reordered_rows", "different ORDER"),
    ("foreign_row", "does not match GLOBAL_SPLIT"),
    ("short_train_stream", "train proba has len"),
    ("other_labels", "y_train = GLOBAL_SPLIT"),
    ("other_fold_partition", "identical 5-fold partition"),
    ("non_finite", "non-finite"),
    ("global_feature_scope", "comparison-only"),
    ("calibrated_column_on_an_identity_fold", "folds whose own gate was not breached"),
])
def test_a_misaligned_or_stale_stage3_artifact_is_refused(world, arts, tmp_path, case, message):
    arts["ebm"] = _tamper(arts["ebm"], case)
    with pytest.raises((Stage4InputError, ValidationError), match=message):
        inputs(world, arts, tmp_path)


F22_FIELDS = ("fold_applied", "fold_ece_inner", "fold_gate_measured_on")


@pytest.mark.parametrize("applied", [True, False], ids=["calibrated", "uncalibrated"])
def test_a_stage3_artifact_from_before_f22_is_refused_calibrated_or_not_f26(world, tmp_path, applied):
    """F26: without F22's per-fold calibration metadata the artifact's
    training-side outputs (calibrated OOF, gate, fold-nested thresholds and
    decisions) were fitted on OOF scores from models trained on the held-out
    fold, whether or not its all-rows calibration gate fired. The same
    artifact with the metadata loads, so the check can fail."""
    arts = {"lr": make_stage1(world), "router": make_router(world),
            "ebm": make_stage3(world, applied=applied)}
    assert arts["ebm"].calibration["applied"] is applied
    inputs(world, arts, tmp_path)                                        # loads as written
    legacy = arts["ebm"]
    legacy.calibration = {k: v for k, v in legacy.calibration.items() if k not in F22_FIELDS}
    with pytest.raises(Stage4InputError, match="predates PR 36 F22"):
        inputs(world, arts, tmp_path)


def test_a_stage1_output_from_before_the_nested_calibration_is_refused_f26(world, arts, tmp_path):
    arts["lr"]["stage_output"]["oof_provenance"] = "refittable (CalibratedClassifierCV ensemble)"
    with pytest.raises(ValidationError, match="nested inside each fold"):
        inputs(world, arts, tmp_path)


def test_disagreeing_calibration_gates_are_consumed_per_fold_f28(world, arts, tmp_path):
    """All-rows gate not breached, fold 0's own gate breached (F22 allows it).
    Stage 4 takes the per-fold column for training rows and the identity
    (all-OOF policy) for test rows, as Stage1's StageOutput pairs them; before
    F28 it took raw OOF for every row (the all-rows gate deciding) and refused
    this artifact."""
    art = arts["ebm"]
    y = np.asarray(art.y_train)
    plan = FoldPlan.from_dict(art.fold_plan)

    def true_p(z):                                         # the world's P(y = 1 | z)
        return 1 / (1 + np.exp(-(2.2 * z - 2.6)))

    raw = true_p(world["z_tr"])                            # calibrated: low all-rows ECE
    art.oof["proba"], art.refit["test_proba"] = raw, true_p(world["z_te"])
    inner = np.where(plan.fold_id[None, :] == np.arange(5)[:, None], np.nan, raw)
    tr0 = plan.splits()[0][0]
    inner[0, tr0] = np.clip(raw[tr0] + 0.3, 0, 1)        # fold 0's inner scores: miscalibrated
    probe = Stage3Calibrator("isotonic", ece_threshold=0.0).fit(raw, y, plan, inner).report_
    gate = (probe.ece_before + probe.fold_ece_inner[0]) / 2
    cal = Stage3Calibrator("isotonic", ece_threshold=gate).fit(raw, y, plan, inner)
    assert cal.applied_ is False and cal.fold_applied_ == [True, False, False, False, False]
    art.oof["proba_calibrated"] = cal.oof_calibrated_
    art.refit["test_proba_calibrated"] = cal.transform(art.refit["test_proba"])
    art.calibration = {**cal.report_.to_dict()}

    ebm = inputs(world, arts, tmp_path)["ebm"]
    np.testing.assert_array_equal(ebm.train_proba, cal.oof_calibrated_)
    assert not np.array_equal(ebm.train_proba[plan.holdout(0)], raw[plan.holdout(0)])
    np.testing.assert_array_equal(ebm.test_proba, art.refit["test_proba"])
    assert ebm.standalone_threshold == art.threshold["oof"]["threshold"]
    assert "1/5 folds mapped" in ebm.probability_space


def test_labels_must_be_binary(world, arts, tmp_path):
    GS = dict(world["GS"])
    GS["y_train"] = GS["y_train"].where(GS["y_train"].index != GS["y_train"].index[0], 2)
    p = dump(tmp_path, arts)
    with pytest.raises(Stage4InputError, match="not binary"):
        load_stage_outputs(GS, p["lr"], p["router"], p["ebm"], verbose=False,
                           extra_fingerprints=ROUTER_FP)
