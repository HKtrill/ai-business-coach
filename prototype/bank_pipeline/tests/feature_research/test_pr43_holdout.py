import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from feature_research import stage4_holdout as h
from tests.feature_research.test_pr43_stage4_nested import _table, SMALL


def test_split_and_majority():
    y = pd.Series(np.tile([0, 1, 0, 0], 80))
    fit, held = h.split_rows(y)
    assert len(held) == 64 and y[held].sum() == 16 and not np.intersect1d(fit, held).size
    np.testing.assert_array_equal(h.split_rows(y)[1], held)
    states = ["pass1", "pass1", "uncertain", "pass2", "pass2"]
    frames = [
        h.prod.routes([17], [state], [0 if state == "uncertain" else 0.5], 0.1 + i * 0.02)
        for i, state in enumerate(states)
    ]
    row = h.prod.average_routes(frames).iloc[0]
    assert row.route == "uncertain" and row.router_score == pytest.approx(0.14)
    row = h.prod.average_routes(frames[:3]).iloc[0]
    assert row.route == "pass1" and row.router_score == pytest.approx(0.055)


def test_resume_rejects_corruption_and_preserves_failures(tmp_path):
    def success():
        return {"p": np.array([0.25])}, {"base_fits": 1}

    h.checkpoint(tmp_path, "abc", "test", 0, success)
    z, _ = h.checkpoint(tmp_path, "abc", "test", 0, lambda: pytest.fail("repeated producer"))
    assert z["p"][0] == 0.25
    with pytest.raises(ValueError, match="Stale"):
        h.checkpoint(tmp_path, "other", "test", 0, success)
    (tmp_path / "test_0/scores.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="damaged"):
        h.checkpoint(tmp_path, "abc", "test", 0, success)
    with pytest.raises(ZeroDivisionError):
        h.checkpoint(tmp_path, "abc", "test", 1, lambda: 1 / 0)
    assert list(tmp_path.glob("test_1_attempt_*/failure.txt"))


def test_thresholds_include_the_registered_boundaries():
    delta = dict(coverage=0.049, positive_retention=0.0, negative_retention=0.0, loss=0.0049)
    arms = {"meta_xgb": {"paired": {"upstream": {"delta": delta}}}}
    assert not h.trigger(arms, False, True)["expand"]
    assert h.trigger(arms, True, True)["necessary_missing_rule_evidence"]
    for k, lim in [("coverage", 0.05), ("negative_retention", -0.05), ("loss", -0.005)]:
        old = delta[k]
        delta[k] = lim
        assert h.trigger(arms, False, False)["numeric"][0][1] == k
        delta[k] = old


def test_independent_rule_gate_requires_matching_supported_rules():
    y = pd.Series(np.tile([0, 0, 1, 1], 180))
    va = y.index
    rule = dict(id=7, route="pass2", order=0, segment=[("b", 1)], precision=0.9)
    mask = np.ones((2, len(y) + 2), bool)
    full = (-1, [rule], np.ones((2, 2), bool), pd.Index([], dtype=int))
    evidence, conf = h.prod.independent_evidence([(0, [rule], mask, va), full], y, pd.Index([800, 801]))
    assert evidence["triggered_rules"] and np.all(conf == 0.5)
    assert evidence["producers"][0]["issued"] == dict(count=720, positive=360)
    absent = dict(rule, segment=[("b", 0)])
    r, conf = h.prod.independent_evidence(
        [(0, [rule], mask, va), (-1, [absent], full[2], full[3])], y, pd.Index([800, 801])
    )
    assert r["unsupported"] and conf is None


def test_activation_does_not_lend_support_to_another_rule():
    y = pd.Series(np.tile([0, 1], 600))
    fold = h.prod.FoldPlan.build(y, 3, 42).fold_id
    trigger = dict(id=1, route="pass2", order=0, segment=[("a", 1)], precision=1.0)
    other = dict(trigger, id=2, order=1, segment=[("b", 1)], precision=0.0)
    first = np.ones(len(y) + 1, bool)
    full = (-1, [other], np.ones((2, 1), bool), y.index[:0])

    for counts, supported in [((1, 0), False), ((200, 99), False), ((100, 100), True)]:
        second = np.zeros_like(first)
        for j, count in enumerate(counts):
            second[np.flatnonzero(fold == j)[:count]] = True
        # All rows match the second rule, but only rows missed by the first receive its decision.
        masks = np.stack([~second, ~second, first, second])
        r, conf = h.prod.independent_evidence(
            [(0, [trigger, other], masks, y.index), full], y, pd.Index([2000])
        )
        assert len(r["triggered_rules"]) == 1
        assert r["inner_gate"][1]["transfer_supported"] is supported
        assert not r["inner_gate"][1]["eligible"]
        if supported:
            assert not r["unsupported"] and conf[0] == y[second[:-1]].mean()
        else:
            assert r["unsupported"] and conf is None


def test_stage1_actual_inner_calibration_excludes_holdout(monkeypatch):
    # Instrument the encoder at every actual fit. The real scaler, base-model fits and sigmoid run.
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(160, 3)), columns=h.prod.LR_FEATURES)
    y = pd.Series(np.tile([0, 1], 80))
    held = x.iloc[:8].set_axis(range(1000, 1008))
    seen = []

    class Encoder:
        def fit(self, rows, labels):
            assert rows.index.equals(labels.index) and not rows.index.intersection(held.index).size
            seen.append(set(rows.index))
            return self

        def transform(self, rows):
            return rows

    monkeypatch.setattr(h.prod.crossfit._fe, "LRFeatureEngineer", Encoder)
    p = h.prod.Producers("glass", x, y, held, LogisticRegression(), {}, [])
    va = np.flatnonzero(p.folds["stage1"] == 0)
    tr = np.flatnonzero(p.folds["stage1"] != 0)
    result, stats = p.stage1(0, tr, va, pd.concat([x.iloc[va], held]), None)
    assert len(seen) == stats["base_fits"] == 11
    assert all(not s.intersection(va) for s in seen)
    assert len(result["stage1_proba"]) == len(va) + len(held)


def test_locked_meta_factorial_reuses_six_fits(tmp_path):
    tr = _table()
    tab = _table(240, seed=10)
    tab.index += 10000
    x = tab.drop(columns="y")
    holds = dict(full=x, average=x.copy())
    pol = h.n.load_policy()
    pol["uncertainty"]["replicates"] = 12

    def cache(stage, k, fn):
        return h.checkpoint(tmp_path, "syn", stage, k, fn)

    m, streams, locks, ledger = h.ev.lock_scores(tr, holds, SMALL[0], pol, True, cache)
    assert len(ledger) == 6 and m["weights"]
    result = h.ev.evaluate(streams, locks, tr.y, tab.y, pol)
    for arm in result.values():
        assert arm["paired"]["upstream"]["changed_retention"] == 0
        assert arm["paired"]["upstream"]["delta"]["loss"] == 0
    assert set(result["meta_xgb"]["paired"]) == {
        "upstream",
        "upstream_at_mean_meta",
        "meta_at_full_upstream",
        "meta_at_mean_upstream",
        "combined",
    }
    assert sum(v["base_fits"] for v in ledger) == 6
    # Reshuffling labels cannot change already locked numeric cuts or the prediction streams.
    changed = h.ev.evaluate(streams, locks, tr.y, 1 - tab.y, pol)
    assert changed["meta_xgb"]["locked"] == locks["meta_xgb"]
    with pytest.raises(ValueError, match="Unpaired"):
        h.ev.evaluate(streams, locks, tr.y, tab.y.iloc[::-1], pol)


def test_stage3_designs_and_gates_keep_holdout_labels_out(monkeypatch):
    # Exercise the actual block, outer/inner fits, calibration gate, full refit and fold scoring.
    rng = np.random.default_rng(48)
    x = pd.DataFrame(rng.normal(size=(160, 3)), columns=list("abc"))
    y = pd.Series(np.tile([0, 1], 80))
    held = x.iloc[:12].set_axis(range(1000, 1012))
    seen = []

    class Features:
        def __init__(self, features):
            self.features = features

        def fit_transform(self, rows, labels):
            assert rows.index.equals(labels.index) and rows.index.intersection(held.index).empty
            seen.append(set(rows.index))
            self.mean = rows.mean()
            return self.transform(rows)

        def transform(self, rows):
            return rows - self.mean

    class Factory:
        def __init__(self, **kwargs):
            self.calls = 0

        def fit(self, params, frame, labels, sample_weight):
            assert params == {"fixed": 4} and frame.index.intersection(held.index).empty
            self.calls += 1
            return LogisticRegression().fit(frame, labels, sample_weight=sample_weight)

        def positive_proba(self, model, frame):
            return model.predict_proba(frame)[:, 1]

    fac = Factory()
    monkeypatch.setattr(h.prod, "EBMFeaturePipeline", Features)
    monkeypatch.setattr(h.prod, "XGBEstimator", lambda **_: fac)
    p = h.prod.Producers("blackbox", x, y, held, None, {"fixed": 4}, list(x))
    raw = np.zeros(len(y))
    for k in [0, 1, 2, 3, 4, -1]:
        va = np.flatnonzero(p.folds["stage3"] == k)
        tr = np.flatnonzero(p.folds["stage3"] != k)
        a, info = p.stage3(k, tr, va, pd.concat([x.iloc[va], held]), raw)
        raw[va] = a["raw"][: len(va)]
        assert len(a["stage3_proba"]) == len(va) + len(held) and np.isfinite(a["stage3_proba"]).all()
    assert len(seen) == fac.calls == 31


def test_phase_receipt_verifies_population_and_checkpoint_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(h.n, "LOGS", tmp_path)
    cache = tmp_path / "holdout_cache" / "blackbox"
    ledger = {}
    for stage, fits in [
        ("stage1", [11] * 10 + [1]),
        ("stage3", [6] * 5 + [1]),
        ("meta", [1] * 6),
        ("router", [1] * 6),
    ]:
        ledger[stage] = []
        for j, num in enumerate(fits):
            k = j if j + 1 < len(fits) else -1
            values = {"base_fits": num, "router_objects": 1, "forests": 37}
            _, receipt = h.checkpoint(cache, "fixed", stage, k, lambda: ({"p": np.array([0.3])}, values))
            ledger[stage].append(receipt)
    cell = {"issued": {"kept": 6590, "deferred_positive": 0, "deferred_negative": 0, "tp": 742, "fn": 0}}
    payload = dict(
        arm="blackbox",
        step5a=h.SHA,
        policy=h.n.POLICY_SHA256,
        binding="fixed",
        split=dict(fit_rows=26360, holdout_rows=6590, fit_positive=2970, holdout_positive=742),
        ledger=ledger,
        arms={k: {"conditions": {"full": cell}} for k in ["meta_xgb", "glass_formula"]},
    )
    path = h.write("holdout_blackbox", payload)
    assert h.verify(path.parent.name, "blackbox", archives=True)["binding"] == "fixed"
    (cache / "meta_0/scores.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="damaged"):
        h.verify(path.parent.name, "blackbox", archives=True)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(h.n.records.RecordError, match="sidecar"):
        h.verify(path.parent.name, "blackbox")
