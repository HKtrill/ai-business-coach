"""
PR 43 Step 2C: nested Stage 4 references on fixed upstream OOF inputs.

Synthetic tables have the Step 2A columns. The production worlds of tests/stage4 check that the
Step 2C fitting paths are the accepted GLASS Arbiter and Meta-XGBoost procedures.
"""
import joblib
import numpy as np
import pandas as pd
import pytest

from feature_research import stage4_nested as N
from feature_research import stage4_policy as s
from glass_pipeline.glass_arbiter.router_evidence import fold_base_rates
from shared.stage3.thresholding import F2ThresholdSelector
from shared.stage4 import ABSTAIN
from shared.stage4.protocol import tune_min_confidence
from shared.stage_io import fold_assignment

SMALL = [dict(n_estimators=8, learning_rate=0.2, max_depth=2, min_child_weight=1.0, subsample=1.0,
              reg_lambda=1.0, reg_alpha=0.001),
         dict(n_estimators=15, learning_rate=0.1, max_depth=1, min_child_weight=5.0, subsample=0.8,
              reg_lambda=5.0, reg_alpha=0.01)]
ARMS = {"formula": N.Formula, "meta": lambda: N.MetaXGB(SMALL)}


def _table(n=700, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    f5 = fold_assignment(y, cv_folds=5, random_state=42).to_numpy()
    b, z, rate = fold_base_rates(y, f5), 1.4 * y + rng.normal(0, 1, n), rng.uniform(0, 1, n)
    route = np.where(z < 0.2, "pass1", np.where(z > 1.3, "pass2", "uncertain"))
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    return pd.DataFrame({
        "y": y, "stage1_fold": fold_assignment(y, cv_folds=10, random_state=42).to_numpy(),
        "stage2_fold": f5, "stage3_fold": f5, "stage1_proba": sig(1.2 * z - 2.6 + rng.normal(0, .5, n)),
        "route": route, "router_base_rate": b,
        "router_score": np.where(route == "pass1", b * rate * .6,
                                 np.where(route == "pass2", b + (1 - b) * rate * .5, b)),
        "stage3_proba": sig(1.4 * z - 2.8 + rng.normal(0, .5, n)),
    }, index=pd.Index(rng.permutation(10 * n)[:n], name="row"))


@pytest.fixture
def policy():
    return s.load_policy()


@pytest.fixture
def tab():
    return _table()


def _run(arm, table, policy):
    outer, inner = N.nested_folds({"glass": table})
    return N.run_nested(arm, table, outer, inner, policy)


def test_registered_constants_are_the_policy_text(policy):
    assert "common five outer folds using accepted shared fold IDs" in policy["nesting"]
    assert "three stratified inner folds (shuffle, seed 42)" in policy["nesting"]
    assert "0.10, 0.20, 1/3, 0.50" in policy["search"]["common"]
    assert N.CLASS_THRESHOLDS == (0.10, 0.20, 1 / 3, 0.50)
    assert "RandomSampler(seed=42)" in policy["search"]["meta"]
    assert "12 unique proposals or 24 attempts" in policy["search"]["meta"]
    assert "recall 0.70, alpha 0.50" in policy["search"]["formula"] and N.FORMULA["alpha"] == 0.5


def test_proposals_follow_the_registered_random_budget():
    est = N.MetaXGBEstimator()
    first, attempts = N.draw_proposals(est)
    assert (first, attempts) == N.draw_proposals(est) and len(first) == attempts == 12
    assert all(est.normalize_params(p) == p for p in first)
    assert len({p["max_depth"] for p in first}) > 1                 # seven dimensions are sampled

    class Repeats:                                                   # three distinct proposals only
        calls = 0

        def suggest(self, trial):
            Repeats.calls += 1
            return {"n": Repeats.calls % 3}

        def normalize_params(self, p):
            return p
    assert N.draw_proposals(Repeats()) == ([{"n": 1}, {"n": 2}, {"n": 0}], 24)


def test_inner_folds_stay_inside_each_outer_training_partition(tab):
    outer, inner = N.nested_folds({"glass": tab, "blackbox": tab.copy()})
    np.testing.assert_array_equal(outer, tab.stage3_fold)
    for k, ids in inner.items():
        y_T = tab.y.to_numpy()[outer != k]
        assert len(ids) == len(y_T) and set(ids) == {0, 1, 2}
        np.testing.assert_array_equal(ids, fold_assignment(y_T, cv_folds=3, random_state=42))
        assert np.ptp(np.bincount(ids, weights=y_T)) <= 1             # stratified
    moved = tab.assign(stage3_fold=np.roll(tab.stage3_fold, 1))
    with pytest.raises(ValueError, match="differ"):
        N.nested_folds({"glass": tab, "blackbox": moved})


@pytest.mark.parametrize("name", ARMS)
def test_outer_labels_and_outer_inputs_reach_nothing_learned(tab, policy, name):
    k, at = 2, tab.stage3_fold.to_numpy() == 2
    base_frames, base, _, base_scores = _run(ARMS[name](), tab, policy)
    flipped = tab.assign(y=np.where(at, 1 - tab.y, tab.y))
    moved = tab.assign(stage1_proba=np.where(at, tab.stage1_proba ** 2, tab.stage1_proba),
                       stage3_proba=np.where(at, tab.stage3_proba ** 2, tab.stage3_proba))
    for changed in (flipped, moved):
        frames, ev, _, scores = _run(ARMS[name](), changed, policy)
        assert ev["configuration"][k] == base["configuration"][k]
        for key in (key for key in scores if key.startswith("inner.")):   # stored inner evidence too
            np.testing.assert_array_equal(scores[key][:, k], base_scores[key][:, k])
        assert ev["selection"][k] == base["selection"][k]
        for rule in N.RULES:
            for col in ("threshold", "cut", *(f"inner_{c}" for c in N.INNER_KEYS)):
                assert ev["rules"][rule]["folds"][col][k] == base["rules"][rule]["folds"][col][k]
            cols = ["ranking", "decision", "confidence", "kept"]
            same = frames[rule].loc[at, cols].equals(base_frames[rule].loc[at, cols])
            assert same is (changed is flipped)                       # inputs move scores, not cuts


@pytest.mark.parametrize("name", ARMS)
def test_every_row_is_scored_once_under_the_registered_accounting(tab, policy, name):
    frames, ev, seconds, scores = _run(ARMS[name](), tab, policy)
    n_cand, outer = len(ARMS[name]().candidates), tab.stage3_fold.to_numpy()
    assert ev["fits"] == {"attempted": 5 * (3 * n_cand + 1), "budget": 5 * (3 * n_cand + 1)}
    assert len(seconds) == 5
    assert ("inner.ranking" in scores) is (name == "meta")            # reusable search predictions
    if name == "meta":
        assert scores["inner.ranking"].shape == (n_cand, 5, len(tab))
        np.testing.assert_array_equal(np.isnan(scores["inner.ranking"][0]),
                                      outer[None, :] == np.arange(5)[:, None])
    for rule, f in frames.items():
        res, cost = ev["rules"][rule], policy["cost"]
        assert f.index.equals(tab.index) and np.array_equal(f.outer_fold, tab.stage3_fold)
        assert f[["ranking", "confidence"]].notna().all().all() and set(f.decision) <= {0, 1}
        np.testing.assert_array_equal(f.kept, f.confidence >= f.cut)
        folds, pooled = pd.DataFrame(res["folds"]), res["pooled"]
        assert pooled["outer"]["loss"] == pytest.approx(f.row_loss.mean(), abs=1e-12)
        assert pooled["outer"]["loss"] == pytest.approx(np.average(folds.outer_loss, weights=folds.rows))
        np.testing.assert_allclose(folds.loss_optimism, folds.outer_loss - folds.inner_loss)
        assert pooled["loss_optimism_row_weighted"] == pytest.approx(
            pooled["outer"]["loss"] - pooled["inner_loss_row_weighted"])
        for x in cost["sensitivity_defer"]:
            f_x = np.where(f.kept, f.row_loss, x)
            assert pooled["outer"][f"loss_defer_{x:g}"] == pytest.approx(f_x.mean())
        assert ("log_loss" in pooled["all_rows"]) is (name == "meta")  # no probability claim for GLASS
        assert res["outer_predictions"]["sha256"] == N._digest(f)
        np.testing.assert_array_equal(scores["outer.ranking"], f.ranking)   # one store for every rule


def test_every_rule_is_locked_before_the_one_outer_scoring(tab, policy, monkeypatch):
    events, real = [], N.lock
    monkeypatch.setattr(N, "lock", lambda *a: events.append("lock") or real(*a))

    class Recorded(N.Formula):
        def score(self, fitted, rows):
            events.append("outer" if rows.stage3_fold.nunique() == 1 else "inner")
            return super().score(fitted, rows)
    _run(Recorded(), tab, policy)
    assert events == (["inner"] * 3 + ["lock"] * 2 + ["outer"]) * 5


def test_zero_retention_keeps_the_whole_system_accounting(tab, policy):
    y, cost = tab.y.to_numpy(), policy["cost"]
    d = (tab.stage3_proba.to_numpy() >= 0.2).astype(int)
    none, some = N._outcome(y, d, np.zeros(len(y), bool), cost), N._outcome(y, d, d == 1, cost)
    assert set(none) == set(some)
    counts = ("kept", "tp", "fp", "tn", "fn", "positive_kept", "negative_kept")
    assert {c: none[c] for c in counts} == dict.fromkeys(counts, 0) and none["coverage"] == 0
    assert (none["deferred_positive"], none["deferred_negative"]) == (y.sum(), len(y) - y.sum())
    assert (none["correct_deferred"], none["wrong_deferred"]) == ((d == y).sum(), (d != y).sum())
    assert none["effective_recall"] == none["positive_retention"] == none["negative_retention"] == 0
    assert none["risk"] is none["retained_prevalence"] is none["f2"] is None    # retained-only: undefined
    assert none["loss"] == cost["defer"]
    assert all(none[f"loss_defer_{x:g}"] == x for x in cost["sensitivity_defer"])
    negatives = N._outcome(np.zeros(4, int), np.zeros(4, int), np.zeros(4, bool), cost)
    assert negatives["positive_retention"] is negatives["effective_recall"] is None


def test_locked_rules_are_the_registered_and_historical_searches(tab, policy):
    y = tab.y.to_numpy()
    score = pd.DataFrame({"ranking": tab.stage3_proba.to_numpy()})
    arm = N.MetaXGB(SMALL)
    hist = N.lock("historical", arm, score, y, policy)
    t = F2ThresholdSelector().select(y, score.ranking).threshold
    d, c = arm.operate(score, t)
    assert (hist["threshold"], hist["cut"]) == (t, tune_min_confidence(y, d, c)["min_confidence"])
    reg, cost = N.lock("registered", arm, score, y, policy), policy["cost"]
    options = []
    for order, t in enumerate(N.CLASS_THRESHOLDS):
        d, c = arm.operate(score, t)
        cut = s.operating_cut(s.risk_curve(y, d, c, cost), policy["coverage"])
        if cut is not None:
            o = s.risk_curve(y, d, (c >= cut).astype(float), cost).iloc[0]
            options.append(((o.loss, -o.kept, order), t, cut))
    assert (reg["threshold"], reg["cut"]) == min(options)[1:]
    strict = dict(policy, coverage=dict(policy["coverage"], min_retained_each_class=10 ** 6))
    assert N.lock("registered", arm, score, y, strict) is None
    frames, ev, _, _ = _run(N.Formula(), tab, strict)              # recorded, never relaxed
    assert ev["rules"]["registered"]["pooled"] is None and set(frames) == {"historical"}
    assert ev["rules"]["registered"]["infeasible_folds"] == [0, 1, 2, 3, 4]


def test_one_record_covers_every_procedure_on_identical_rows(tab, policy, tmp_path, monkeypatch):
    from glass_pipeline.glass_router.pipeline import fold_fingerprint
    monkeypatch.setattr(N, "verified_inputs", lambda evidence, root: (policy, {"verified": "stub"}))
    monkeypatch.setattr(N, "reproduce_references", lambda t, e: (dict(zip(N.UPSTREAMS, SMALL)), {}))
    monkeypatch.setattr(N, "draw_proposals", lambda est: (SMALL, 2))
    tables = {"glass": tab, "blackbox": tab.assign(stage1_proba=tab.stage1_proba ** 1.5)}
    evidence = {"population": {"shared_fold_fingerprint": fold_fingerprint(tab.stage3_fold)}}
    frames, ev, local = N.nested_references(tables, evidence, root=tmp_path)
    assert {u: set(p) for u, p in ev["procedures"].items()} == {
        u: {"glass_formula", "meta_xgb", "meta_xgb_accepted_params"} for u in N.UPSTREAMS}
    assert all(f.index.equals(tab.index) for p in frames.values() for a in p.values() for f in a.values())
    rows = np.bincount(tab.stage3_fold)
    assert ev["design"]["matched_counts"]["0.85"] == [int(np.ceil(0.85 * m - 1e-9)) for m in rows]
    unchanged = ev["design"]["producer_folds_unchanged"]["blackbox"]
    assert unchanged["stage1_fold"] == fold_fingerprint(tab.stage1_fold)
    assert N.summarize(ev).shape[0] == 12
    paths = N.record_nested(ev, local, root=tmp_path)
    run = paths[0].parent.name
    saved, arrays = N.load_predictions(run, tmp_path, index=tab.index)
    assert saved["content_sha256"] == s.records.content_digest(ev) and paths[1].name == N.PREDICTIONS
    assert sorted(arrays) == sorted(ev["predictions"]["arrays"])
    assert not [key for key in arrays if ".glass_formula.inner." in key]
    for u in N.UPSTREAMS:
        inner = arrays[f"{u}.meta_xgb.inner.ranking"]               # XGBoost scores: exact as float32
        assert inner.dtype == np.float32 and inner.shape == (2, 5, len(tab))
        np.testing.assert_array_equal(np.isnan(inner[0]), arrays["inner_fold"] < 0)
        for name, rules in frames[u].items():
            for f in rules.values():
                np.testing.assert_array_equal(arrays[f"{u}.{name}.outer.ranking"], f.ranking)
    with pytest.raises(ValueError, match="row order"):
        N.load_predictions(run, tmp_path, index=tab.index[::-1])
    arrays["outer_fold"][0] ^= 1
    with open(paths[1], "wb") as fh:
        np.savez_compressed(fh, **arrays)
    with pytest.raises(ValueError, match="outer_fold"):
        N.load_predictions(run, tmp_path)
    with pytest.raises(ValueError, match="shared fold"):
        N.nested_references(tables, {"population": {"shared_fold_fingerprint": "other"}}, root=tmp_path)


def test_formula_is_the_accepted_glass_arbiter_procedure(tmp_path, monkeypatch, policy):
    from glass_pipeline.glass_arbiter import meta_stage as ms
    from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3
    w, paths = build_world(), {}
    for key, art in {"lr": make_stage1(w), "router": make_router(w), "ebm": make_stage3(w)}.items():
        paths[key] = str(tmp_path / f"{key}.joblib")
        joblib.dump(art, paths[key])
    real = ms.load_stage_outputs
    monkeypatch.setattr(ms, "load_stage_outputs", lambda *a, **k: real(*a, extra_fingerprints=ROUTER_FP, **k))
    art, _ = ms.train_glass_arbiter(w["GS"], lr_path=paths["lr"], glass_path=paths["router"],
                                    ebm_path=paths["ebm"], artifact_dir=str(tmp_path / "a"), verbose=False)
    inp = ms.load_stage_outputs(w["GS"], paths["lr"], paths["router"], paths["ebm"], verbose=False)
    gl, y = inp["glass"], inp.y_train
    table = pd.DataFrame({"y": y, "stage1_proba": inp["lr"].train_proba,
                          "stage3_proba": inp["ebm"].train_proba,
                          "router_score": gl.train_proba, "route": gl.train_decisions,
                          "router_base_rate": fold_base_rates(y, gl.train_fold_id)})
    fit = N.Formula().fit(N.FORMULA, table)
    assert (fit["lr"], fit["ebm"], fit["weights"]) == (art["thresholds"]["lr"], art["thresholds"]["ebm"],
                                                       art["weights"])
    locked = N.lock("historical", N.Formula(), N.Formula().score(fit, table), y, policy)
    assert locked["cut"] == art["abstention"]["min_weighted_confidence"]


def test_meta_features_and_oof_are_the_accepted_meta_xgb_procedure(tmp_path):
    from tests.support.meta_xgb_world import blackbox_world, run_meta, stage4_frames
    w, arts = blackbox_world()
    art, _ = run_meta(w, arts, tmp_path)
    inputs, X_train, _ = stage4_frames(w, arts)
    s1, s2, s3 = inputs["mlp"], inputs["rf_router"], inputs["xgb"]
    table = pd.DataFrame({"y": inputs.y_train, "stage1_proba": s1.train_proba,
                          "route": np.where(s2.train_decisions == ABSTAIN, "uncertain", s2.train_decisions),
                          "router_score": s2.train_proba, "stage3_proba": s3.train_proba},
                         index=inputs.index_train)
    pd.testing.assert_frame_equal(N.meta_features(table), X_train)
    learner = art.learner
    params = N.MetaXGBEstimator().normalize_params(learner.best_params)
    np.testing.assert_array_equal(N.meta_oof(params, table, np.asarray(learner.oof["fold_id"])),
                                  learner.oof["proba"])


def test_step2c_runs_only_on_the_registered_step2a_inputs(tmp_path, monkeypatch, policy):
    rr = s.records
    evidence = {"population": policy["population"],
                "upstreams": dict.fromkeys(("stage1.lr", "stage1.mlp", "stage2.glass_router",
                                            "stage2.rf_router", "stage3.ebm", "stage3.xgb")),
                "historical_reference": dict.fromkeys(("glass_arbiter", "glass_arbiter_on_blackbox",
                                                       "meta_xgb", "meta_xgb_on_glass"))}
    run = rr.start_run(commit=2, seeds={}, root=tmp_path, run_id=policy["input_run"], notebook=N.NOTEBOOK,
                       code_files=N.PRODUCER_CODE)
    rr.write_run_record(run, "frozen_inputs", evidence)
    monkeypatch.setattr(N, "REGISTRATION", s.register(tmp_path).parent.name)
    _, identity = N.verified_inputs(evidence, tmp_path)
    assert identity["frozen_inputs"]["content_sha256"] == rr.content_digest(evidence)
    with pytest.raises(ValueError, match="replayed Step 2A content"):
        N.verified_inputs({**evidence, "population": {}}, tmp_path)
    monkeypatch.setattr(N, "file_sha256_lf", lambda path: "edited")
    with pytest.raises(ValueError, match="producer code"):
        N.verified_inputs(evidence, tmp_path)
