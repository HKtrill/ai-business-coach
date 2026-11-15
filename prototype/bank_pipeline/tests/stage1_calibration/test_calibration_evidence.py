"""
PR 39 Commit 2, Step 2A: the Stage 1 calibration evidence (E-C) and rule C1-C5.

1. The metrics use the shared ECE bins; the Murphy terms decompose Brier.
2. The bootstrap is deterministic, keeps every row count and centres on the mean.
3. The Stage 4 diagnostic uses the GLASS arbiter's own recall cut.
4. The rule: isotonic only when every C3 condition holds in both arms.
5. The evidence settings mirror production Stage 1 apart from the E-C
   calibration request.
6. End to end on synthetic data, with the real stage code: the per-family
   vectors are a fixed-family policy's OOF output, the record is written once,
   and ``check`` recomputes it and refuses tampering.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

import glass_pipeline.lr.lr_stage_calibrated as lr_mod
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from glass_pipeline.meta_ebm.meta_stage import find_recall_threshold
from shared.metrics import calculate_ece
from shared.stage1.feature_engineering import LRFeatureEngineer
from shared.stage_io import fold_assignment
from stage1_calibration import evidence as E
from stage1_calibration import runner as R
from stage2_crossfit.runner import make_train_side
from tests.support.notebooks import BANK_ROOT
from tests.support.synthetic_bank import make_bank_frame, split_frame


# ======================================================================
# 1. Metrics
# ======================================================================
def test_the_murphy_bins_are_the_shared_ece_bins():
    rng = np.random.default_rng(0)
    p = np.r_[0.0, 0.1, 0.2, 1.0, rng.uniform(0, 1, 500)]
    y = (rng.uniform(0, 1, len(p)) < p).astype(int)
    by_bins = sum(m.mean() * abs(y[m].mean() - p[m].mean())
                  for m in E._bin_masks(p, E.ECE_BINS) if m.any())
    assert by_bins == pytest.approx(calculate_ece(y, p), abs=1e-15)


def test_with_one_forecast_per_bin_brier_is_reliability_minus_resolution_plus_uncertainty():
    rng = np.random.default_rng(1)
    p = rng.choice([0.03, 0.15, 0.35, 0.62, 0.95], 2000)
    y = (rng.uniform(0, 1, len(p)) < p).astype(int)
    m = E.calibration_metrics(y, p, base_score=p)
    assert m["brier"] == pytest.approx(m["reliability"] - m["resolution"] + m["uncertainty"],
                                       abs=1e-12)
    assert m["distinct_values"] == 5 and 0 < m["largest_tie_share"] < 1


# ======================================================================
# 2. Bootstrap
# ======================================================================
def _toy(n=600, seed=2):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(0, 1, n) < 0.2).astype(int)
    fold_id = np.arange(n) % 10
    return y, fold_id, rng.normal(0, 1, n)


def test_the_bootstrap_is_deterministic_and_centred_on_the_mean():
    y, fold_id, d = _toy()
    a = E.stratified_bootstrap_delta(d, y, fold_id, replicates=300)
    b = E.stratified_bootstrap_delta(d, y, fold_id, replicates=300)
    assert a == b
    assert a["point"] == pytest.approx(d.mean())
    assert a["lower"] < a["point"] < a["upper"]
    assert a["strata"] == 20 and a["seed"] == 42


def test_a_constant_difference_has_a_degenerate_interval():
    y, fold_id, _ = _toy()
    r = E.stratified_bootstrap_delta(np.full(len(y), -0.25), y, fold_id, replicates=50)
    assert r["lower"] == r["upper"] == r["point"] == pytest.approx(-0.25)


def test_the_registered_bootstrap_settings_are_the_defaults():
    y, fold_id, d = _toy(n=200)
    r = E.stratified_bootstrap_delta(d, y, fold_id)
    assert (r["replicates"], r["seed"], r["level"]) == (2000, 42, 0.95)


# ======================================================================
# 3. Stage 4 input diagnostic
# ======================================================================
def test_the_stage4_cut_is_the_arbiters_and_ties_overshoot_the_target():
    rng = np.random.default_rng(3)
    y = (rng.uniform(0, 1, 3000) < 0.12).astype(int)
    smooth = np.clip(0.12 + 0.2 * (y - 0.12) + rng.normal(0, 0.08, len(y)), 0, 1)
    stepped = np.round(smooth, 1)                      # a few tied blocks, like isotonic
    out = E.stage4_inputs(y, {"sigmoid": smooth, "isotonic": stepped})
    for f, p in (("sigmoid", smooth), ("isotonic", stepped)):
        assert out[f]["threshold"] == find_recall_threshold(y, p, 0.70)
        assert out[f]["achieved_recall"] >= 0.70
    assert out["isotonic"]["achieved_recall"] > out["sigmoid"]["achieved_recall"]
    assert out["isotonic"]["distinct_confidences"] < out["sigmoid"]["distinct_confidences"]
    flips = out["vote_differences"]
    assert flips["rows"] == flips["positives"] + flips["negatives"] > 0


# ======================================================================
# 4. The rule
# ======================================================================
def _arm(pooled=-0.001, upper=-0.0002, lower_folds=9, recall=0.705, brier=0.08):
    return {"delta_brier": {"pooled": pooled, "bootstrap": {"upper": upper},
                            "folds_isotonic_lower": lower_folds, "folds": 10},
            "stage4": {"isotonic": {"achieved_recall": recall}},
            "pooled": {"sigmoid": {"brier": brier}}}


def test_isotonic_is_selected_only_when_every_c3_condition_holds_in_both_arms():
    d = E.apply_rule({"glass_lr": _arm(), "blackbox_mlp": _arm()})
    assert d["selected_family"] == "isotonic"
    assert all(c["passed"] for arm in d["C3"].values() for c in arm.values())


@pytest.mark.parametrize("arm", E.ARMS)
@pytest.mark.parametrize("failure, condition", [
    (dict(pooled=0.0), "C3a"),                      # a tie favours nobody
    (dict(pooled=0.0004, upper=0.001), "C3a"),
    (dict(upper=0.0001), "C3a"),                    # interval reaches zero
    (dict(lower_folds=7), "C3b"),
    (dict(recall=0.7103), "C3c"),
])
def test_any_single_c3_failure_in_either_arm_keeps_sigmoid(arm, failure, condition):
    arms = {a: _arm() for a in E.ARMS}
    arms[arm] = _arm(**failure)
    d = E.apply_rule(arms)
    assert d["selected_family"] == "sigmoid"
    assert d["C3"][arm][condition]["passed"] is False
    assert "C2 default" in d["basis"]


def test_the_neutral_alternative_is_reported_and_never_changes_the_selection():
    d = E.apply_rule({"glass_lr": _arm(pooled=-0.004, upper=-0.002),
                      "blackbox_mlp": _arm(pooled=0.0005, upper=0.001)})
    assert d["selected_family"] == "sigmoid"
    assert d["neutral_alternative"]["would_select"] == "isotonic"
    assert "not a claim" in d["statement"]


def test_the_rule_refuses_evidence_for_one_arm_only():
    with pytest.raises(E.EvidenceError, match="both arms"):
        E.apply_rule({"glass_lr": _arm()})


@pytest.mark.parametrize("bad, match", [
    ({"isotonic": None}, "families"),
    ({"sigmoid": np.full(40, np.nan)}, "non-finite"),
    ({"sigmoid": np.full(40, 1.2)}, "outside"),
])
def test_arm_evidence_refuses_incomplete_or_invalid_vectors(bad, match):
    y = np.r_[np.zeros(30, int), np.ones(10, int)]
    p = {"sigmoid": np.full(40, 0.25), "isotonic": np.full(40, 0.25)}
    p.update({k: v for k, v in bad.items() if v is not None})
    if bad.get("isotonic", 0) is None:
        p.pop("isotonic")
    with pytest.raises(E.EvidenceError, match=match):
        E.arm_evidence(y, np.arange(40) % 4, np.zeros(40), p)


# ======================================================================
# 5. The settings mirror production Stage 1
# ======================================================================
def test_the_lr_settings_are_production_stage1_apart_from_the_e_c_request(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def capture(**kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(lr_mod, "CalibratedLRStage", capture)
    with pytest.raises(Stop):
        lr_mod.train_lr_stage(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int),
                              pd.Series(dtype=int))
    import inspect
    defaults = {k: v.default for k, v in
                inspect.signature(CalibratedLRStage.__init__).parameters.items()
                if v.default is not inspect.Parameter.empty}
    production = {**defaults, **seen}
    registered = {**R.REGISTERED_SETTINGS["glass_lr"],
                  "random_state": R.REGISTERED_SETTINGS["random_state"]}
    assert registered["calibration_method"] == "auto"        # E-C: both families
    assert {k: production[k] for k in registered if k != "calibration_method"} == \
        {k: v for k, v in registered.items() if k != "calibration_method"}


def test_the_mlp_settings_are_the_blackbox_notebooks_apart_from_the_e_c_request():
    nb = json.loads((BANK_ROOT / "blackbox_pipeline/blackbox_cascade.ipynb")
                    .read_text(encoding="utf-8"))
    import ast
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))
    (ctor,) = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "CalibratedStage1MLP"]
    kw = {k.arg: ast.unparse(k.value) for k in ctor.keywords}
    assert (kw["cv_folds"], kw["n_trials"], kw["random_state"]) == (
        "10", "STAGE1_N_TRIALS", "RANDOM_STATE")
    assert "STAGE1_N_TRIALS = 100" in src and "RANDOM_STATE = 42" in src
    mlp = R.REGISTERED_SETTINGS["blackbox_mlp"]
    assert (mlp["cv_folds"], mlp["n_trials"], R.REGISTERED_SETTINGS["random_state"]) == \
        (10, 100, 42)
    assert mlp["calibration_method"] == "auto"
    assert R.REGISTERED_SETTINGS["engineer"] == {"n_bins": 20, "smoothing_factor": 100}


# ======================================================================
# 6. End to end, synthetic data, real stage code
# ======================================================================
SMALL = {
    "random_state": 42,
    "engineer": {"n_bins": 20, "smoothing_factor": 100},
    "glass_lr": {"tune_hyperparameters": False, "calibration_method": "auto",
                 "optimize_threshold": True, "cv_folds": 3, "n_trials": 1, "tune_cv_folds": 3},
    "blackbox_mlp": {"calibration_method": "auto", "cv_folds": 3, "n_trials": 1,
                     "max_epochs": 15, "patience": 5, "verbose": False, "n_jobs": 1},
}


@pytest.fixture(scope="module")
def train_side():
    split = split_frame(make_bank_frame(n=400))
    return make_train_side(split["X_train"], split["y_train"].astype(int), split["X_test"])


@pytest.fixture(scope="module")
def fitted(train_side):
    fits = R.fit_arms(train_side, SMALL)
    return fits, *R.vectors_frame(train_side, fits, SMALL)


def test_the_vectors_are_each_fits_own_out_of_fold_output(fitted, train_side):
    fits, frame, meta, plan = fitted
    y = train_side.y_train
    assert frame.index.equals(y.index) and (frame["y"].to_numpy() == y.to_numpy()).all()
    want = fold_assignment(y, cv_folds=3, random_state=42, index=y.index).to_numpy()
    assert np.array_equal(frame["fold_id"].to_numpy(), want)
    for arm in E.ARMS:
        stage = fits[arm]["stage"]
        for f in E.FAMILIES:
            assert np.array_equal(frame[f"{arm}_{f}"].to_numpy(),
                                  stage.calibration_oof_[f].to_numpy())
        assert meta[arm]["final_model_family_auto"] == stage.calibration_method
        assert set(meta[arm]["fold_inner"]) == {0, 1, 2}
    assert plan["folds"] == 3


@pytest.mark.parametrize("family", E.FAMILIES)
def test_a_family_vector_is_what_the_fixed_family_policy_produces(fitted, train_side, family):
    _, frame, _, _ = fitted
    X_raw, y = train_side.X_train, train_side.y_train
    X_eng = LRFeatureEngineer(**SMALL["engineer"]).fit_transform(X_raw, y)
    fixed = CalibratedLRStage(random_state=42, **{**SMALL["glass_lr"],
                                                  "calibration_method": family})
    fixed.fit(X_eng, y, X_train_raw=X_raw)
    assert np.array_equal(fixed.proba_train_oof.to_numpy(), frame[f"glass_lr_{family}"].to_numpy())
    assert set(fixed.calibration_fold_family_.values()) == {family}


def test_the_record_is_written_once_and_check_recomputes_it(tmp_path, train_side):
    roots = dict(record_root=tmp_path / "research_logs", output_root=tmp_path / "models")
    record = R.run(run_id="c2_test", settings=SMALL, train_side=train_side, **roots)
    assert record["registered"] is False and "NOT official" in record["status"]
    assert record["decision"]["selected_family"] in E.FAMILIES
    assert record["vectors"]["rows"] == len(train_side.y_train)
    assert R.check("c2_test", **roots)["run_id"] == "c2_test"

    with pytest.raises(R.RunnerError, match="written once"):
        R.run(run_id="c2_test", settings=SMALL, train_side=train_side, **roots)

    rec_path = roots["record_root"] / "c2_test" / R.RECORD_FILE
    original = rec_path.read_bytes()
    tampered = json.loads(original)
    tampered["decision"]["selected_family"] = (
        "isotonic" if tampered["decision"]["selected_family"] == "sigmoid" else "sigmoid")
    rec_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(R.RunnerError, match="decision"):
        R.check("c2_test", **roots)

    rec_path.write_bytes(original)
    vec_path = roots["output_root"] / "c2_test" / R.VECTORS_FILE
    vec_path.write_bytes(vec_path.read_bytes() + b"\0")
    with pytest.raises(R.RunnerError, match="SHA-256"):
        R.check("c2_test", **roots)


def test_an_unofficial_record_names_its_settings_and_never_claims_registration(tmp_path, train_side):
    record = R.run(run_id="c2_small", settings=copy.deepcopy(SMALL), train_side=train_side,
                   record_root=tmp_path / "r", output_root=tmp_path / "m")
    assert record["settings"]["glass_lr"]["cv_folds"] == 3
    assert record["population"]["frozen_protocol_check"]["passed"] is None
    assert record["code"]["files_sha256_lf"]["stage1_calibration/evidence.py"]


# ======================================================================
# 7. The decision record
# ======================================================================
def _policy_for(record, run_id):
    return {"id": "pr39-stage1-calibration", "version": "1.0.0", "evidence_run": run_id,
            "family": record["decision"]["selected_family"]}


def test_a_decision_pins_the_verified_evidence_and_is_written_once(tmp_path, train_side):
    roots = dict(record_root=tmp_path / "r", output_root=tmp_path / "m")
    record = R.run(run_id="c2_dec", settings=SMALL, train_side=train_side, **roots)
    policy = _policy_for(record, "c2_dec")
    with pytest.raises(R.RunnerError, match="only registered"):
        R.decide("c2_dec", policy=policy, **roots)
    wrong = {**policy, "family": "isotonic" if policy["family"] == "sigmoid" else "sigmoid"}
    with pytest.raises(R.RunnerError, match="the evidence selects"):
        R.decide("c2_dec", policy=wrong, require_registered=False, **roots)

    decision = R.decide("c2_dec", policy=policy, require_registered=False, **roots)
    evidence_path = roots["record_root"] / "c2_dec" / R.RECORD_FILE
    import hashlib
    assert decision["evidence"]["record_sha256"] == \
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert decision["policy"]["family"] == decision["outcome"]["selected_family"]
    assert any("not a finding" in line for line in decision["interpretation"])
    assert decision["neutral_alternative"]["would_select"] in E.FAMILIES
    assert "does not change the selection" in decision["neutral_alternative"]["status"]
    assert any("neutral alternative" in line for line in decision["interpretation"])
    R.check("c2_dec", **roots)                                  # verifies the decision too

    with pytest.raises(R.RunnerError, match="written once"):
        R.decide("c2_dec", policy=policy, require_registered=False, **roots)

    path = evidence_path.with_name(R.DECISION_FILE)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["interpretation"] = ["sigmoid is better"]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(R.RunnerError, match="does not match the verified evidence"):
        R.check("c2_dec", **roots)


def test_the_committed_decision_is_the_code_policy_and_pins_its_evidence():
    import hashlib

    from shared.stage1.oof_calibration import (STAGE1_CALIBRATION_FAMILY,
                                               STAGE1_CALIBRATION_POLICY)
    path = BANK_ROOT / STAGE1_CALIBRATION_POLICY["decision_record"]
    if not path.is_file():
        pytest.skip("written by 'python -m stage1_calibration decide' and committed with Commit 2")
    decision = json.loads(path.read_text(encoding="utf-8"))
    evidence_path = path.with_name(R.RECORD_FILE)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert decision["schema_version"] == R.DECISION_FORMAT
    assert decision["record_version"] == STAGE1_CALIBRATION_POLICY["version"]
    assert decision["policy"]["family"] == STAGE1_CALIBRATION_FAMILY
    assert decision["evidence"]["run_id"] == STAGE1_CALIBRATION_POLICY["evidence_run"]
    assert decision["evidence"]["record_sha256"] == \
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert evidence["registered"] is True
    assert evidence["decision"]["selected_family"] == STAGE1_CALIBRATION_FAMILY
