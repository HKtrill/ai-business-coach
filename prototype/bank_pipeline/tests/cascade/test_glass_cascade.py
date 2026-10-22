"""
GLASS cascade integration, Stages 1-4, on the production producers and savers
(see conftest). Stages 1-3 do not feed each other: each reads GLOBAL_SPLIT
and derives its own features, and Stage 4 joins their saved artifacts. The
cross-stage contracts are therefore the shared split, rows, folds and
fingerprints, and what Stage 4 takes from each artifact.

Contract checked here, on artifacts the current code produces:

* split and provenance: each artifact carries its own fingerprint scheme
  (Stage 1 and 3 the stage_io index+labels hash, the router the frames hash
  the notebook stamps) and Stage 4 matches every one with no stand-in
  scheme; every per-row output keeps the split's rows in split order; the
  Stage 1 10-fold and Stage 2/3 5-fold partitions coexist (equal-k streams
  share one partition, the 10-fold stream is recorded as distinct by design);
  Stage 1 and 3 must come from the GLASS arm (F32);
* OOF honesty: Stage 4 trains on each producer's out-of-fold column and
  scores the test rows with its full-train refit; in-sample substitution,
  reordered, duplicated, dropped or foreign rows, non-finite values,
  unknown router states and stale (pre-F22 / pre-nested) artifacts are
  refused at the hand-off;
* routing: the router's three states, its first-match Pass 1 / Pass 2
  semantics and its test-side outputs reach Stage 4 unchanged;
* decisions: the stored final decisions reconstruct from the saved stage
  outputs and the frozen Stage 4 settings, with the router's confidence
  read from its stored rule precision (F25) and the abstention cut the exact
  search on the training confidence (F30); each threshold lives in the
  probability space its stage hands over;
* reproducibility and isolation: Stage 4 reruns identically from the reloaded
  artifacts; re-running the whole cascade with the test labels permuted
  reproduces every training-side output, every test prediction and every
  decision, and changes only the test metrics and the label-bound
  fingerprints. This holds with fixed hyperparameters; Optuna selection on
  the same folds is the open finding F24.

Not repeated here: per-stage fold isolation and calibration (tests/stage1,
stage3), router fitting and rule selection (tests/stage2), and the Stage 4
input contract on hand-built artifacts (tests/stage4).
"""

from __future__ import annotations

import copy

import joblib
import numpy as np
import pandas as pd
import pytest

import shared.stage4.contract as contract
import shared.stage_io as stage_io

from blackbox_pipeline.models.meta_xgb.comparison import load_meta_ebm_for_comparison
from blackbox_pipeline.models.rf_router import split_fingerprint as router_split_fingerprint
from glass_pipeline.glass_router.pipeline import fold_fingerprint
from glass_pipeline.meta_ebm.arbiter import meta_arbiter
from glass_pipeline.meta_ebm.loader import load_stage_outputs
from glass_pipeline.meta_ebm.meta_stage import find_recall_threshold, train_meta_stage
from glass_pipeline.meta_ebm.router_evidence import fold_base_rates
from shared.stage3.data import FoldPlan
from shared.stage4 import ABSTENTION_SEARCH, Stage4InputError, tune_min_confidence
from shared.stage_io import StageOutputError, fold_assignment, split_fingerprint
from shared.validation import ValidationError
from tests.cascade.conftest import build_glass_cascade, glass_split

STATES = ("pass1", "pass2", "uncertain")
CONFIG_KEYS = ("thresholds", "weights", "abstention", "calibration", "disagreement")


def stage4_inputs(c, lr=None, glass=None, ebm=None):
    return load_stage_outputs(c.GS, c.paths["lr"] if lr is None else lr,
                              c.paths["glass"] if glass is None else glass,
                              c.paths["ebm"] if ebm is None else ebm, verbose=False)


# ----------------------------------------------------------------------
# split, rows, folds, provenance
# ----------------------------------------------------------------------
def test_each_stage_is_bound_to_the_split_by_its_own_fingerprint_scheme(cascade):
    X_tr, X_te = cascade.GS["X_train"], cascade.GS["X_test"]
    io_fp = split_fingerprint(X_tr.index, X_te.index, cascade.GS["y_train"], cascade.GS["y_test"])
    router_fp = router_split_fingerprint(X_tr, X_te)
    assert cascade.load("lr")["stage_output"]["split_fingerprint"] == io_fp
    assert cascade.load("glass")["split_fingerprint"] == router_fp
    assert cascade.ebm.split_fingerprint == io_fp
    inp = stage4_inputs(cascade)                                   # real schemes, no stand-ins
    assert {n: s.fingerprint_scheme for n, s in inp.streams.items()} == \
        {"lr": "stage_io", "glass": "router", "ebm": "stage_io"}
    assert {k: cascade.meta["split_fingerprints"][k] for k in ("stage_io", "router")} == \
        {"stage_io": io_fp, "router": router_fp}


def test_both_router_fits_record_the_splits_training_labels_and_stage4_checks_them_f12(cascade):
    """ModelSaver writes the full-train fit's train_label_fingerprint and the
    cross-fit's (in oof_protocol); Meta-EBM's Stage 4 verified both."""
    want = stage_io.label_fingerprint(cascade.GS["X_train"].index, cascade.GS["y_train"])
    bundle = joblib.load(cascade.paths["glass"])
    assert bundle["train_label_fingerprint"] == bundle["oof_protocol"]["train_label_fingerprint"] == want
    streams = cascade.meta["stage4_inputs"]["streams"]
    assert streams["glass"]["label_evidence"] == contract.LABELS_FINGERPRINTED
    assert streams["lr"]["label_evidence"] == streams["ebm"]["label_evidence"] == contract.LABELS_CARRIED


def test_every_stage_holds_the_split_rows_in_split_order(cascade):
    idx_tr, idx_te = cascade.GS["X_train"].index, cascade.GS["X_test"].index
    so, gl = cascade.load("lr")["stage_output"], cascade.load("glass")
    for s in ("train_proba_oof", "train_fold_id", "y_train"):
        assert so[s].index.equals(idx_tr), s
    for s in ("test_proba", "y_test"):
        assert so[s].index.equals(idx_te), s
    assert pd.Index(gl["train_idx"]).equals(idx_tr) and pd.Index(gl["index_test"]).equals(idx_te)
    assert cascade.oof.index.equals(idx_tr)
    assert pd.Index(cascade.ebm.index_train).equals(idx_tr)
    assert pd.Index(cascade.ebm.index_test).equals(idx_te)
    meta = cascade.meta
    assert meta["index_train"] == list(idx_tr) and meta["index_test"] == list(idx_te)
    n_tr, n_te = len(idx_tr), len(idx_te)
    for name in ("lr", "glass", "ebm"):
        assert len(meta["train_probs"][name]) == len(meta["train_fold_ids"][name]) == n_tr
        assert len(meta["test_probs"][name]) == len(meta["test_preds"][name]) == n_te
    for key in ("meta",):
        assert len(meta["test_probs"][key]) == n_te
    for key in ("no_abstain", "with_abstain"):
        assert len(meta["test_preds"][key]) == n_te
    assert len(meta["glass_decisions_train"]) == n_tr and len(meta["explanations"]) == n_te


def test_ten_and_five_fold_protocols_coexist_without_a_shared_partition_requirement(cascade):
    """Stage 1 cross-fits on 10 folds, Stages 2 and 3 on one shared 5-fold
    partition. Stage 4 requires equal-k streams to share their partition and
    records the 10-fold stream as distinct by design."""
    y = cascade.GS["y_train"]
    folds = cascade.meta["stage4_inputs"]["folds"]
    assert {n: s["n_folds"] for n, s in folds["streams"].items()} == {"lr": 10, "glass": 5, "ebm": 5}
    assert all(s["reproduced_by_shared_fold_assignment"] and s["seed"] == 42
               for s in folds["streams"].values())
    assert folds["shared_partitions"] == [["glass", "ebm", 5]]
    assert sorted(d["streams"][1] for d in folds["distinct_partitions"]) == ["ebm", "glass"]
    lr_fold = cascade.load("lr")["stage_output"]["train_fold_id"].to_numpy()
    router_fold = np.asarray(cascade.load("glass")["train_fold_id"])
    np.testing.assert_array_equal(lr_fold, fold_assignment(y, cv_folds=10, random_state=42))
    np.testing.assert_array_equal(router_fold, cascade.ebm.oof["fold_id"])
    np.testing.assert_array_equal(router_fold, FoldPlan.from_dict(cascade.ebm.fold_plan).fold_id)
    assert cascade.load("glass")["oof_protocol"]["fold_fingerprint"] == fold_fingerprint(router_fold)
    assert not np.array_equal(lr_fold % 5, router_fold)            # genuinely different partitions


@pytest.mark.parametrize("stage, relabel, message", [
    ("lr", {"arm": "blackbox", "model": "mlp"}, "GLASS-arm Stage 1 output"),
    ("ebm", {"model_family": "xgboost", "arm": "blackbox"}, "GLASS EBM artifact"),
    ("ebm", {"model_family": "unset", "arm": "unset"}, "PR 36 F21"),
])
def test_stage1_and_stage3_artifacts_from_another_arm_are_refused_f32(cascade, stage, relabel,
                                                                    message):
    """Before F32 each of these loaded as "GLASS LR" / "GLASS EBM". The router
    is held to the GLASS rule blend by F25."""
    if stage == "lr":
        art = copy.deepcopy(cascade.load("lr"))
        art["stage_output"].update(relabel)
        kwargs = {"lr": art}
    else:
        art = copy.deepcopy(cascade.load("ebm"))
        art.model_family = relabel["model_family"]
        art.config = {**art.config, "arm": relabel["arm"]}
        kwargs = {"ebm": art}
    with pytest.raises(ValidationError, match=message):
        stage4_inputs(cascade, **kwargs)
    prov = cascade.meta["stage4_inputs"]["streams"]
    assert prov["lr"]["arm"] == "glass"
    assert (prov["ebm"]["model_family"], prov["ebm"]["arm"]) == ("ebm", "glass")


# ----------------------------------------------------------------------
# OOF honesty at the hand-off
# ----------------------------------------------------------------------
def test_stage4_trains_on_each_producers_oof_column_and_scores_its_refit(cascade):
    inp = stage4_inputs(cascade)
    lr_tr, lr_te = cascade.lr_X
    r_tr, r_te = cascade.router_X
    so, gl, ebm = cascade.load("lr")["stage_output"], cascade.load("glass"), cascade.ebm
    # training side: the producers' out-of-fold columns ...
    np.testing.assert_array_equal(inp["lr"].train_proba, so["train_proba_oof"].to_numpy())
    np.testing.assert_array_equal(inp["glass"].train_proba, cascade.oof.glass_proba[:, 1])
    np.testing.assert_array_equal(inp["ebm"].train_proba, ebm.oof["proba_calibrated"])
    # ... never the full-train models' in-sample scores
    assert not np.allclose(inp["lr"].train_proba, cascade.lr_stage.predict_proba(lr_tr))
    assert not np.allclose(inp["glass"].train_proba, cascade.router.predict_proba(r_tr)[:, 1])
    assert not np.allclose(inp["ebm"].train_proba, ebm.refit["train_proba_calibrated"])
    # test side: the full-train refits
    np.testing.assert_allclose(inp["lr"].test_proba, cascade.lr_stage.predict_proba(lr_te),
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(inp["glass"].test_proba, cascade.router.predict_proba(r_te)[:, 1])
    np.testing.assert_array_equal(inp["ebm"].test_proba, ebm.refit["test_proba_calibrated"])
    np.testing.assert_array_equal(gl["test_proba"][:, 1], inp["glass"].test_proba)


def _tamper(c, case):
    """(which artifact, tampered copy) for one way a hand-off can go wrong."""
    if case.startswith("lr"):
        art = copy.deepcopy(c.load("lr"))
        so = art["stage_output"]
        if case == "lr_train_rows_reordered":
            so["train_proba_oof"] = so["train_proba_oof"].iloc[::-1]
        elif case == "lr_train_row_duplicated":
            s = so["train_proba_oof"]
            so["train_proba_oof"] = pd.Series(s.to_numpy(), index=[s.index[1], *s.index[1:]])
        elif case == "lr_test_value_nan":
            so["test_proba"] = so["test_proba"].where(np.arange(len(so["test_proba"])) != 3)
        elif case == "lr_before_nested_calibration":
            so["oof_provenance"] = "refittable (CalibratedClassifierCV ensemble)"
        elif case == "lr_pre_audit_artifact":
            del art["stage_output"]
        return "lr", art
    if case.startswith("glass"):
        gl = copy.deepcopy(c.load("glass"))
        if case == "glass_unknown_router_state":
            gl["train_decisions"] = np.where(np.arange(len(gl["train_decisions"])) == 0, "maybe",
                                             gl["train_decisions"]).astype(object)
        elif case == "glass_training_row_dropped":
            for k in ("train_proba", "train_decisions", "train_pred", "train_confidence",
                      "train_fold_id", "train_covered", "train_abstained"):
                gl[k] = gl[k][1:]
        elif case == "glass_decisions_misaligned":
            gl["train_decisions"] = np.roll(gl["train_decisions"], 7)
            gl["train_pred"] = np.roll(gl["train_pred"], 7)
        elif case == "glass_in_sample_substitution":
            gl["train_proba"] = gl["train_insample_proba_diagnostic"].copy()
        elif case == "glass_routed_value_inf":
            routed = np.flatnonzero(np.asarray(gl["train_decisions"]) == "pass2")[0]
            gl["train_proba"] = gl["train_proba"].copy()
            gl["train_proba"][routed] = [-np.inf, np.inf]
        return "glass", gl
    art = copy.deepcopy(c.ebm)
    if case == "ebm_foreign_row":
        art.index_train = [-1] + list(art.index_train[1:])
    elif case == "ebm_in_sample_substitution":
        art.oof["proba_calibrated"] = np.asarray(art.refit["train_proba_calibrated"]).copy()
    elif case == "ebm_before_f22":
        art.calibration = {k: v for k, v in art.calibration.items()
                           if k not in ("fold_applied", "fold_ece_inner", "fold_gate_measured_on")}
    return "ebm", art


@pytest.mark.parametrize("case, message", [
    ("lr_train_rows_reordered", r"proba.index != y.index"),      # StageOutput re-validates
    ("lr_train_row_duplicated", "duplicate labels"),
    ("lr_test_value_nan", "proba contains NaN"),
    ("lr_before_nested_calibration", "nested inside each fold"),
    ("lr_pre_audit_artifact", "holds no StageOutput"),
    ("glass_unknown_router_state", "unrecognised router decision"),
    ("glass_training_row_dropped", "glass: train proba has len"),
    ("glass_decisions_misaligned", "train probabilities = rule blends"),
    ("glass_in_sample_substitution", "in-sample diagnostic"),
    ("glass_routed_value_inf", "glass: train proba finite"),
    ("ebm_foreign_row", "does not match GLOBAL_SPLIT"),
    ("ebm_in_sample_substitution", "refit in-sample train_proba_calibrated"),
    ("ebm_before_f22", "predates PR 36 F22"),
])
def test_a_broken_hand_off_is_refused(cascade, case, message):
    which, tampered = _tamper(cascade, case)
    with pytest.raises((Stage4InputError, StageOutputError, ValidationError), match=message):
        stage4_inputs(cascade, **{which: tampered})


# ----------------------------------------------------------------------
# routing semantics
# ----------------------------------------------------------------------
def test_the_routers_three_states_reach_stage4_unchanged(cascade):
    gl, meta = cascade.load("glass"), cascade.meta
    inp = stage4_inputs(cascade)
    code = {"pass1": 0, "pass2": 1, "uncertain": -1}
    for side in ("train", "test"):
        dec = np.asarray(gl[f"{side}_decisions"])
        assert set(dec) == set(STATES), side                          # all three states occur
        np.testing.assert_array_equal(gl[f"{side}_pred"], [code[d] for d in dec])
        np.testing.assert_array_equal(gl[f"{side}_covered"], dec != "uncertain")
        stream = getattr(inp["glass"], f"{side}_decisions")
        np.testing.assert_array_equal(stream, np.where(dec == "uncertain", "abstain", dec))
    np.testing.assert_array_equal(gl["train_decisions"], cascade.oof.glass_decisions)
    np.testing.assert_array_equal(meta["glass_decisions_train"], inp["glass"].train_decisions)
    np.testing.assert_array_equal(meta["glass_decisions_test"], inp["glass"].test_decisions)
    np.testing.assert_array_equal(meta["test_preds"]["glass"], inp["glass"].test_decisions == "pass2")
    np.testing.assert_array_equal(gl["train_confidence"], cascade.oof.glass_confidence)


def test_pass2_routes_only_rows_no_pass1_rule_matches(cascade):
    """First match wins, Pass 1 first; the cross-fit records the declared
    Pass 2 population (the legacy profile's "full")."""
    gl, r_te = cascade.load("glass"), cascade.router_X[1]
    dec = np.asarray(gl["test_decisions"])

    def matches(rules):
        hit = np.zeros(len(r_te), bool)
        for rule in rules:
            m = np.ones(len(r_te), bool)
            for f, v in rule.segment:
                m &= (r_te[f] == v).to_numpy()
            hit |= m
        return hit

    in_p1, in_p2 = matches(cascade.router.pass1_rules), matches(cascade.router.pass2_rules)
    np.testing.assert_array_equal(dec == "pass1", in_p1)
    np.testing.assert_array_equal(dec == "pass2", in_p2 & ~in_p1)
    np.testing.assert_array_equal(dec == "uncertain", ~in_p1 & ~in_p2)
    assert (in_p1 & in_p2).any()                                   # overlaps exist and go to Pass 1
    assert gl["oof_protocol"]["glass_pass2_population"] == cascade.config.pass2_population


# ----------------------------------------------------------------------
# thresholds and final decisions
# ----------------------------------------------------------------------
def test_each_threshold_lives_in_the_probability_space_its_stage_hands_over(cascade):
    y = cascade.GS["y_train"].to_numpy()
    so, gl, ebm, t = cascade.load("lr")["stage_output"], cascade.load("glass"), cascade.ebm, \
        cascade.meta["thresholds"]
    lr_oof, ebm_oof = so["train_proba_oof"].to_numpy(), np.asarray(ebm.oof["proba_calibrated"])
    assert t["lr"] == find_recall_threshold(y, lr_oof, 0.70)           # calibrated LR OOF
    assert t["ebm"] == find_recall_threshold(y, ebm_oof, 0.70)         # calibrated EBM OOF
    assert t["ebm"] != find_recall_threshold(y, np.asarray(ebm.oof["proba"]), 0.70)   # not raw
    assert t["glass"] == gl["training_base_rate"] == y.mean()          # router reference (F25)
    fold = np.asarray(gl["train_fold_id"])
    assert t["glass_train_base_rates"] == {k: y[fold != k].mean() for k in range(5)}
    assert len(set(t["glass_train_base_rates"].values())) > 1
    # the standalone thresholds reported alongside are in the consumed spaces too
    ref = cascade.meta["artifact_thresholds"]
    assert ref["lr"] == so["threshold"] and ref["ebm"] == ebm.threshold["oof_calibrated_space"]["threshold"]
    from sklearn.metrics import fbeta_score
    assert fbeta_score(y, (lr_oof >= so["threshold"]).astype(int), beta=2) == \
        pytest.approx(cascade.load("lr")["cv_f2"])


def test_final_decisions_reconstruct_from_the_saved_stage_outputs(cascade):
    """From the saved artifacts alone, with the router's evidence read from its
    stored rule precision rather than its probability (F25): each voter's
    vote and confidence, weights renormalised over the row's voters, ties to
    NOT_SUBSCRIBE, abstain below the frozen cut."""
    meta, gl, ebm = cascade.meta, cascade.load("glass"), cascade.ebm
    t, w, cut = meta["thresholds"], meta["weights"], meta["abstention"]["min_weighted_confidence"]
    lr_p = cascade.load("lr")["stage_output"]["test_proba"].to_numpy()
    ebm_p = np.asarray(ebm.refit["test_proba_calibrated"])
    dec, prec = np.asarray(gl["test_decisions"]), np.asarray(gl["test_confidence"], float)
    b = gl["training_base_rate"]
    rate = np.where(dec == "pass1", 1 - prec, prec)
    router_conf = np.where(dec == "pass2", np.maximum(0, rate - b),
                           np.where(dec == "pass1", np.maximum(0, b - rate), 0.0))
    W = np.column_stack([np.full(len(dec), w["lr"]), np.full(len(dec), w["ebm"]),
                         np.where(dec == "uncertain", 0.0, w["glass"])])
    W = W / W.sum(1, keepdims=True)
    vote = np.column_stack([lr_p >= t["lr"], ebm_p >= t["ebm"], dec == "pass2"])
    conf = np.column_stack([np.abs(lr_p - t["lr"]), np.abs(ebm_p - t["ebm"]), router_conf])
    c1, c0 = (W * conf * vote).sum(1), (W * conf * ~vote).sum(1)
    pred = (c1 > c0).astype(int)
    stored_na, stored_a = meta["test_preds"]["no_abstain"], meta["test_preds"]["with_abstain"]
    side = np.maximum(c1, c0)
    near = np.isclose(c1, c0, rtol=0, atol=1e-12) | np.isclose(side, cut, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(stored_na[~near], pred[~near])
    np.testing.assert_array_equal(stored_a[~near], np.where(side < cut, -1, pred)[~near])
    assert near.mean() < 0.02
    # internal consistency of the stored final columns
    kept = stored_a != -1
    np.testing.assert_array_equal(stored_a[kept], stored_na[kept])
    expl = np.asarray(meta["explanations"]).astype(str)
    np.testing.assert_array_equal(np.char.startswith(expl, "ABSTAIN"), ~kept)
    np.testing.assert_array_equal(np.char.startswith(expl, "PREDICT 1"), kept & (stored_na == 1))
    assert meta["with_abstain"]["coverage"] == pytest.approx(kept.mean())
    assert ((dec == "pass2") & (stored_na == 0)).any() or ((dec == "pass1") & (stored_na == 1)).any()


def test_the_abstention_cut_is_the_exact_search_on_the_cascades_training_confidence(cascade):
    meta = cascade.meta
    inp = stage4_inputs(cascade)
    y = inp.y_train
    lr, gl, ebm = inp["lr"], inp["glass"], inp["ebm"]
    t = meta["thresholds"]
    pred, _, _, conf = meta_arbiter(
        lr.train_proba, ebm.train_proba, gl.train_proba, gl.train_decisions == "pass1",
        gl.train_decisions == "pass2", t["lr"], t["ebm"], meta["weights"], allow_abstain=False,
        return_confidence=True, glass_base_rate=fold_base_rates(y, gl.train_fold_id))
    best = tune_min_confidence(y, pred, conf)
    ab = meta["abstention"]
    assert ab["search"] == ABSTENTION_SEARCH
    assert ab["min_weighted_confidence"] == best["min_confidence"]
    assert ab["train_coverage"] == best["train_coverage"] >= 0.5
    assert ab["train_f2"] == best["train_f2"] and ab["n_candidates"] == best["n_candidates"]


# ----------------------------------------------------------------------
# reload, rerun, test-label isolation
# ----------------------------------------------------------------------
def test_stage4_reruns_identically_from_the_reloaded_artifacts(cascade, tmp_path):
    loaded = cascade.load("meta")
    again, path = train_meta_stage(cascade.GS, lr_path=cascade.paths["lr"],
                                   glass_path=cascade.paths["glass"], ebm_path=cascade.paths["ebm"],
                                   artifact_dir=str(tmp_path), verbose=False)
    for key in CONFIG_KEYS + ("no_abstain", "with_abstain", "split_fingerprints"):
        np.testing.assert_equal(again[key], cascade.meta[key], err_msg=key)
        np.testing.assert_equal(loaded[key], cascade.meta[key], err_msg=key)
    for key in ("lr", "ebm", "glass", "no_abstain", "with_abstain"):
        np.testing.assert_array_equal(again["test_preds"][key], cascade.meta["test_preds"][key])
        np.testing.assert_array_equal(loaded["test_preds"][key], cascade.meta["test_preds"][key])
    np.testing.assert_array_equal(again["test_probs"]["meta"], cascade.meta["test_probs"]["meta"])
    found = load_meta_ebm_for_comparison(cascade.GS, cascade.paths["meta"])   # real schemes
    np.testing.assert_array_equal(found["test_preds"]["with_abstain"],
                                  cascade.meta["test_preds"]["with_abstain"])


@pytest.fixture(scope="module")
def cascade_with_permuted_test_labels(cascade, tmp_path_factory):
    y_te = cascade.GS["y_test"]
    permuted = pd.Series(np.random.default_rng(3).permutation(y_te.to_numpy()), index=y_te.index,
                         name=y_te.name)
    assert not permuted.equals(y_te)
    return build_glass_cascade(tmp_path_factory.mktemp("glass_cascade_permuted"),
                               glass_split(y_test=permuted))


def test_test_labels_reach_no_training_side_output_or_decision(cascade,
                                                               cascade_with_permuted_test_labels):
    """The whole cascade, re-run with the test labels permuted: everything
    learned, every test prediction and every decision reproduces exactly
    (also the cascade's determinism); only test metrics and the label-bound
    fingerprints move. Conditional on fixed hyperparameters (F24)."""
    a, b = cascade, cascade_with_permuted_test_labels
    so_a, so_b = a.load("lr")["stage_output"], b.load("lr")["stage_output"]
    for k in ("train_proba_oof", "test_proba", "train_fold_id"):
        pd.testing.assert_series_equal(so_a[k], so_b[k])
    assert so_a["threshold"] == so_b["threshold"]
    gl_a, gl_b = a.load("glass"), b.load("glass")
    for k in ("train_proba", "train_decisions", "train_confidence", "train_fold_id",
              "test_proba", "test_decisions", "test_confidence", "training_base_rate", "rules"):
        np.testing.assert_equal(gl_a[k], gl_b[k], err_msg=k)
    for k in ("proba", "proba_calibrated", "fold_id", "threshold_per_row", "decision"):
        np.testing.assert_array_equal(a.ebm.oof[k], b.ebm.oof[k])
    for k in ("test_proba", "test_proba_calibrated"):
        np.testing.assert_array_equal(a.ebm.refit[k], b.ebm.refit[k])
    for k in ("oof", "oof_calibrated_space"):
        assert a.ebm.threshold[k]["threshold"] == b.ebm.threshold[k]["threshold"]
    for key in CONFIG_KEYS:
        np.testing.assert_equal(a.meta[key], b.meta[key], err_msg=key)
    for key in a.meta["test_preds"]:
        np.testing.assert_array_equal(a.meta["test_preds"][key], b.meta["test_preds"][key])
    np.testing.assert_array_equal(a.meta["test_probs"]["meta"], b.meta["test_probs"]["meta"])
    np.testing.assert_array_equal(a.meta["explanations"], b.meta["explanations"])
    # what may move: test labels' own consequences
    assert so_a["split_fingerprint"] != so_b["split_fingerprint"]
    assert gl_a["split_fingerprint"] == gl_b["split_fingerprint"]           # frames only
    assert a.meta["no_abstain"] != b.meta["no_abstain"]
