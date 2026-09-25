"""
GLASS Stage 4 (Meta-EBM) — input contract + behaviour.

Fixtures: ``shared.stage4.tests.fixtures`` (real StageOutput / Stage3Artifact
classes, real router artifact layout). Run from the repo root:

    python -m pytest glass_pipeline/meta_ebm/tests shared/stage4/tests -q
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from shared.stage4 import Stage4InputError
from shared.stage4.tests.fixtures import (
    ROUTER_FP, build_world, make_router, make_stage1, make_stage3,
)
from shared.validation import ValidationError

from glass_pipeline.meta_ebm import meta_stage as ms
from glass_pipeline.meta_ebm.loader import load_stage_outputs
from glass_pipeline.meta_ebm.meta_stage import find_recall_threshold, train_meta_stage


@pytest.fixture
def world(tmp_path):
    w = build_world()
    arts = {"lr": make_stage1(w), "router": make_router(w), "ebm": make_stage3(w)}
    return w, arts, tmp_path


def dump(tmp_path, arts):
    out = {}
    for k, v in arts.items():
        out[k] = str(tmp_path / f"{k}.joblib")
        joblib.dump(v, out[k])
    return out


def load(w, arts, tmp_path):
    p = dump(tmp_path, arts)
    return load_stage_outputs(w["GS"], p["lr"], p["router"], p["ebm"], verbose=False,
                              extra_fingerprints=ROUTER_FP)


def run(w, arts, tmp_path, **kw):
    p = dump(tmp_path, arts)
    return train_meta_stage(w["GS"], lr_path=p["lr"], glass_path=p["router"],
                            ebm_path=p["ebm"], artifact_dir=str(tmp_path / "meta"),
                            verbose=kw.pop("verbose", False), **kw)


@pytest.fixture(autouse=True)
def _router_scheme(monkeypatch):
    """train_meta_stage has no extra_fingerprints arg: route it via the loader."""
    real = ms.load_stage_outputs
    monkeypatch.setattr(ms, "load_stage_outputs",
                        lambda *a, **k: real(*a, extra_fingerprints=ROUTER_FP, **k))


# ======================================================================
def test_current_contracts_load(world):
    w, arts, tmp = world
    inp = load(w, arts, tmp)
    so = arts["lr"]["stage_output"]
    np.testing.assert_array_equal(inp["lr"].train_proba, so["train_proba_oof"].to_numpy())
    np.testing.assert_array_equal(inp["lr"].test_proba, so["test_proba"].to_numpy())
    assert inp["lr"].fingerprint_scheme == "stage_io"
    assert inp["glass"].fingerprint_scheme == inp["ebm"].fingerprint_scheme == "router"
    assert inp["glass"].train_source == "train_proba"            # flag layout
    assert all(r["passed"] for r in inp.validation)


def test_stage_output_file_form_also_loads(world):
    w, arts, tmp = world
    arts["lr"] = make_stage1(w, wrap=False)                    # StageOutput.save() file
    inp = load(w, arts, tmp)
    assert inp["lr"].train_source == "train_proba_oof"


def test_router_uncertain_is_abstain_and_nan_inert(world):
    w, arts, tmp = world
    inp = load(w, arts, tmp)
    d = inp["glass"].train_decisions
    raw = np.asarray(arts["router"]["train_decisions"])
    assert ((raw == "uncertain") == (d == "abstain")).all() and (d == "abstain").any()
    assert np.isfinite(inp["glass"].train_proba).all()          # NaN → 0.5, never read


def test_router_flag_false_refused(world):
    w, arts, tmp = world
    arts["router"]["train_outputs_oof"] = False
    with pytest.raises(Stage4InputError, match="train_outputs_oof"):
        load(w, arts, tmp)


def test_stage3_pairs_and_oracle(world):
    w, arts, tmp = world
    inp = load(w, arts, tmp)
    np.testing.assert_array_equal(inp["ebm"].train_proba, arts["ebm"].oof["proba_calibrated"])
    assert inp["ebm"].standalone_threshold == 0.19 != 0.777
    arts["ebm"] = make_stage3(w, applied=False)
    del arts["ebm"].threshold["test_oracle"]                    # never read
    inp = load(w, arts, tmp)
    np.testing.assert_array_equal(inp["ebm"].test_proba, arts["ebm"].refit["test_proba"])
    assert inp["ebm"].standalone_threshold == 0.55


def test_insample_guards(world):
    w, arts, tmp = world
    r = arts["router"]
    r["train_proba"] = r["train_insample_proba_diagnostic"].copy()
    with pytest.raises(ValidationError, match="in-sample diagnostic"):
        load(w, arts, tmp)


def test_fingerprint_mismatch_and_folds(world):
    w, arts, tmp = world
    inp = load(w, arts, tmp)
    assert ["glass", "ebm", 5] in inp.fold_provenance["shared_partitions"]
    arts["router"]["split_fingerprint"] = "deadbeefdeadbeef"
    with pytest.raises(ValidationError, match="fingerprint"):
        load(w, arts, tmp)


def test_router_semantics_inverted_fails(world):
    w, arts, tmp = world
    r = arts["router"]
    swap = {"pass1": "pass2", "pass2": "pass1", "uncertain": "uncertain"}
    r["train_decisions"] = np.array([swap[d] for d in r["train_decisions"]], dtype=object)
    r["train_pred"] = np.where(r["train_decisions"] == "pass1", 0,
                               np.where(r["train_decisions"] == "pass2", 1, -1))
    with pytest.raises(ValidationError, match="NOT_SUBSCRIBE"):
        load(w, arts, tmp)


def test_end_to_end_and_provenance(world):
    w, arts, tmp = world
    art, path = run(w, arts, tmp)
    assert joblib.load(path)["schema"] == "meta_ebm/2"
    prov = art["stage4_inputs"]["streams"]
    assert prov["ebm"]["train_source"] == "Stage3Artifact.oof['proba_calibrated']"
    assert art["artifact_thresholds"]["used_by_stage4"] is False
    assert set(art["split_fingerprints"]) >= {"stage_io", "router"}
    assert art["weights_details"]["glass_accuracy_population"] == "covered_rows"


def test_test_data_does_not_affect_configuration(world, monkeypatch):
    w, arts, tmp = world
    a1, _ = run(w, arts, tmp)
    inner = ms.load_stage_outputs

    def scrambled(*a, **k):
        inp = inner(*a, **k)
        rng = np.random.default_rng(7)
        inp.y_test = rng.permutation(inp.y_test)
        for s in inp.streams.values():
            s.test_proba = rng.random(len(s.test_proba))
            if s.test_decisions is not None:
                s.test_decisions = rng.permutation(s.test_decisions)
        return inp

    monkeypatch.setattr(ms, "load_stage_outputs", scrambled)
    a2, _ = run(w, arts, tmp)
    for k in ("thresholds", "weights", "abstention", "calibration", "disagreement"):
        assert a1[k] == a2[k], k
    assert a1["no_abstain"] != a2["no_abstain"]


def test_tuning_fallback_does_not_crash(world, monkeypatch):
    w, arts, tmp = world
    monkeypatch.setattr(ms, "tune_arbiter_threshold", lambda *a, **k: None)
    art, _ = run(w, arts, tmp, verbose=True)
    assert art["abstention"]["selected_by"].startswith("fallback")


def _recall_threshold_original(y_true, y_prob, target_recall=0.70):
    thresholds = np.sort(np.unique(y_prob))[::-1]
    pos_mask = y_true == 1
    n_pos = pos_mask.sum()
    if n_pos == 0:
        return 0.5
    for t in thresholds:
        if ((y_prob >= t) & pos_mask).sum() / n_pos >= target_recall:
            return float(t)
    return float(thresholds[-1])


@pytest.mark.parametrize("seed", range(4))
def test_find_recall_threshold_matches_original(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(800) < 0.15).astype(int)
    p = np.round(rng.random(800), 2 if seed % 2 else 6)
    for tr in (0.0, 0.3, 0.7, 0.95, 1.0):
        assert find_recall_threshold(y, p, tr) == _recall_threshold_original(y, p, tr)
