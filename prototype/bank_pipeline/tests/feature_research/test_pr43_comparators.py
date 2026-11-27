import numpy as np
import pandas as pd
import pytest

from feature_research import stage4_comparators as c
from tests.feature_research.test_pr43_stage4_nested import _table


def test_normalization_and_conditional_budget():
    table = _table()
    model = c.FORMULA.fit(c.n.FORMULA, table)
    raw = c.FORMULA.score(model, table)
    scores, terms = c.normalized(table, model, raw)
    np.testing.assert_array_equal(scores.ranking, raw.ranking)
    assert (terms[table.route == "uncertain", 2] == 0).all()
    assert (terms[table.route == "pass1", 2] <= 0).all()
    assert (terms[table.route == "pass2", 2] >= 0).all()
    np.testing.assert_allclose(scores.confidence, np.maximum(np.maximum(terms, 0).sum(1), -np.minimum(terms, 0).sum(1)))
    loss = {c.BASE: .125, "blend25": .12, "recall60": .12, "blend75": 1., "recall80": 1.}
    assert c.corners_enabled(loss)
    loss["recall60"] += .00001
    assert not c.corners_enabled(loss)
    assert len(c.MANDATORY) == 6 and len(c.CORNERS) == 4


def test_inner_calibration_isolation_and_cached_winner_identity():
    t = _table()
    outer, inner = c.n.nested_folds({"glass": t})
    train, ev = t[outer != 1], t[outer == 1]
    ids = inner[1]

    class Scorer:
        def __init__(self):
            self.calls = []

        def fit(self, params, rows):
            self.calls.append(set(rows.index))
            return set(rows.index)

        def score(self, model, rows):
            assert model.isdisjoint(rows.index)
            return pd.DataFrame({"ranking": rows.stage1_proba}, index=rows.index)
    arm = Scorer()
    raw = train.stage1_proba.to_numpy()
    mapper, info = c.calibrated(train, ids, raw, {}, arm)
    inside, out = mapper.oof_calibrated_, mapper.transform(ev.stage1_proba.to_numpy())
    assert len(arm.calls) == 9 and info["method"] == "isotonic"
    for j in range(3):
        assert all(rows <= set(train.index[ids != j]) for rows in arm.calls[3 * j:3 * j + 3])
    assert np.isfinite(inside).all() and np.all(np.diff(out[np.argsort(ev.stage1_proba)]) >= 0)
    cube = np.full((12, 5, len(t)), np.nan)
    cube[:, 1, outer != 1] = raw
    ref = {"meta_xgb": {"selection": [{"chosen": 0, "failures": {}}] * 5, "candidates": [{"i": i} for i in range(12)]},
           "meta_xgb_accepted_params": {"candidates": [{"i": 1}]}}
    arrays = {"glass.meta_xgb.inner.ranking": cube}

    def sources():
        return c.winners(train, 1, outer, "glass", ref, arrays)[-1]
    assert sources() == {0: "meta_xgb"}
    cube[1, 1, outer != 1] = np.where(train.y, .9, .1)
    assert sources() == {0: "meta_xgb", 1: "meta_xgb_accepted_params"}
    cube[2, 1, outer != 1] = np.where(train.y, .99, .01)
    assert sources() == {0: "meta_xgb", 2: None}
    ref["meta_xgb"]["candidates"][1] = {"i": 0}
    with pytest.raises(ValueError, match="distinct"):
        sources()


def test_combined_pipeline_reconstructs_accepted_rows(monkeypatch, tmp_path):
    # Exercise both upstream-independent evaluation paths with synthetic Step 2C/4A evidence.
    class CheapMeta(c.n.MetaXGB):
        def fit(self, params, rows):
            return params["i"]

        def score(self, model, rows):
            p = np.clip(rows.stage1_proba.to_numpy() + model / 100, .001, .999)
            return pd.DataFrame({"ranking": p}, index=rows.index)

    t = _table(780)
    outer, inner = c.n.nested_folds({"glass": t})
    policy = c.n.load_policy()
    policy["uncertainty"]["replicates"] = 20  # keep this integration fixture inexpensive
    meta = CheapMeta([{"i": i} for i in range(12)])
    arrays, refs = {}, {}
    for name, arm in (("glass_formula", c.FORMULA), ("meta_xgb", meta), ("meta_xgb_accepted_params", CheapMeta([{"i": 1}]))):
        _, rec, _, store = c.n.run_nested(arm, t, outer, inner, policy)
        refs[name] = rec
        arrays.update({f"glass.{name}.{k}": c.n._compact(v) for k, v in store.items()})
    ref = {"procedures": {"glass": refs}}
    with c.threadpool_limits(limits=1):
        accepted = c.linear.evaluate(t, "glass", outer, inner, {("glass", k): "not met" for k in inner}, policy, ref, arrays)
        plans = {k: c.winners(t[outer != k], k, outer, "glass", refs, arrays) for k in inner}
        monkeypatch.setattr(c, "META", meta)
        out = c.evaluate(t, "glass", outer, inner, ref, arrays, accepted, policy, plans, {})
    assert out["linear_disposition"] == accepted["disposition"]
    assert len(out["matched"]) == 20 and len(out["design"]) == 5
    for name in ("formula_reference", "formula_selected", "meta_f2_raw", "meta_logloss_calibrated", "linear"):
        metrics = out["arms"][name]["pooled"]["outer"]
        assert sum(metrics[k] for k in ("tp", "tn", "fp", "fn", "deferred_negative", "deferred_positive")) == len(t)
    assert all(v["interval"]["level"] == .95 for v in out["secondary"].values())
    assert "ece" not in out["secondary"]["logit_margin"]["safeguards"]
    assert "ece" in out["secondary"]["meta_logloss_calibrated"]["safeguards"]
    rr = c.n.records
    run = rr.start_run(commit=4, seeds={}, root=tmp_path, notebook=c.n.NOTEBOOK)
    rr.write_run_record(run, "stage4_comparators", {"results": out})
    assert rr.read_run_record(run.run_id, "stage4_comparators", root=tmp_path)["results"]["linear_disposition"] == accepted["disposition"]
