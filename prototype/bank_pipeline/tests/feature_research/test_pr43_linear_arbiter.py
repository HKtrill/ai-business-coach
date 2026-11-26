from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from feature_research import linear_arbiter as a


def table():
    rng = np.random.default_rng(57)
    n = 360
    y = np.tile([0, 0, 1], n // 3)
    p = rng.uniform(.02, .8, (n, 2)) + .15 * y[:, None]
    route = np.resize(a.STATES, n)
    b = np.linspace(.21, .36, n)
    r = np.select([route == "pass1", route == "pass2"], [.09, .68], default=np.nan)
    score = np.select([route == "pass1", route == "pass2"], [b * r, b + (1 - b) * r], default=b)
    return pd.DataFrame(dict(y=y, route=route, stage1_proba=p[:, 0], stage3_proba=p[:, 1],
                             router_score=score, router_base_rate=b), index=rng.permutation(n))


def test_evidence_scales_and_frozen_base():
    t = table().iloc[:6].copy()
    t.iloc[0, t.columns.get_loc("stage1_proba")] = 0
    t.iloc[1, t.columns.get_loc("stage3_proba")] = 1
    x, state = a.evidence(t, [.1, .2])
    assert np.isfinite(x).all()
    for k, r in [(0, .09), (1, .68)]:
        np.testing.assert_allclose(x[state == k, 2], logit(r) - logit(t.router_base_rate[state == k]))
    assert (x[state == 2, 2] == 0).all()
    assert x[0, 0] == pytest.approx(logit(1e-6) - logit(.1))


def test_optimizer_constraints_intercept_and_shrinkage(monkeypatch):
    t = table()
    x, state = a.evidence(t, [.1, .2])
    y = t.y.to_numpy()
    base, _ = a.ridge(np.zeros_like(x), state, y, 1)
    assert base[0] == pytest.approx(logit(y.mean()), abs=1e-5)
    weak, _ = a.ridge(x, state, y, .001)
    strong, _ = a.ridge(x, state, y, 1.)
    assert np.linalg.norm(strong[1:]) <= np.linalg.norm(weak[1:])
    beta, _ = a.ridge(x, state, y, .001, weak)
    assert min(beta[1:]) >= 0 and np.all(a.logits(beta, (x + .1, state)) >= a.logits(beta, (x, state)))
    assert beta[-1] == pytest.approx(weak[-1] / 1.001, abs=1e-5)
    monkeypatch.setattr(a, "minimize", lambda *args, **kw: SimpleNamespace(success=False, message="fixture"))
    with pytest.raises(RuntimeError, match="fixture"):
        a.ridge(x, state, y, .01)


def test_fold_local_calibration_and_inner_fit_budget(monkeypatch):
    t = table()
    tr, ev = t.iloc[:300], t.iloc[300:]
    touched = []
    original = a.cuts

    def observed(rows):
        touched.append(set(rows.index))
        return original(rows)
    monkeypatch.setattr(a, "cuts", observed)
    ids = a.n.FoldPlan.build(tr.y, 3, 42).fold_id
    with a.threadpool_limits(limits=1):
        out, info = a.fit_outer(tr, ev.drop(columns="y"), ids, True, a.n.load_policy())
    assert 98 <= info['counts']['ridge'] <= 99
    assert set(out) == {'route', 'global'} and len(touched) == 13
    for j in range(3):
        allowed = set(tr.index[ids != j])
        assert all(rows <= allowed for rows in touched[4 * j:4 * j + 4])
    assert touched[-1] == set(tr.index)
    for score, cfg in out.values():
        assert cfg['correction_slope'] >= 0 and score.index.equals(ev.index)
        assert np.all(np.diff(score.sort_values('raw').sigmoid) >= -1e-14)


def test_pinned_triggers_and_dispositions():
    n = a.n
    frozen = n.records.read_run_record(n.load_policy()['input_run'], 'frozen_inputs', root=n.LOGS)
    _, identity = n.verified_inputs(frozen)
    rec = n.records.read_run_record(a.DIAGNOSTIC_RUN, 'voter_diagnostics', root=n.LOGS)
    ref = {'content_sha256': rec['nested_reference']['content_sha256']}
    gates = a.eligibility(rec, ref, identity)
    assert all(gates['blackbox', j] == 'met' and gates['glass', j] == 'not met' for j in range(5))
    with pytest.raises(ValueError):
        a.eligibility(rec, {'content_sha256': 'different'}, identity)

    def result(v):
        return dict(loss_verdict=v, safeguards={'positive_retention': True})
    comparisons = {'glass_formula': result('improved'), 'meta_xgb': result('comparable')}
    assert a.disposition(comparisons) == 'provisional advance'
    comparisons['glass_formula']['safeguards']['positive_retention'] = False
    assert a.disposition(comparisons) == 'reject'


def test_reference_reconstruction_and_paired_pipeline(monkeypatch):
    from tests.feature_research.test_pr43_stage4_nested import _table
    from feature_research import research_records as rr

    # Construct synthetic reference frames through the actual Step 2C ledger.
    class Probability(a.n.MetaXGB):
        def fit(self, params, rows):
            return None

        def score(self, model, rows):
            return pd.DataFrame({"ranking": rows.stage1_proba.astype("float32").astype(float)}, index=rows.index)

    t = _table(780)
    outer, inner = a.n.nested_folds({"glass": t})
    spec = a.n.load_policy()
    arrays, records = {}, {}
    for name, arm in (("glass_formula", a.n.Formula()), ("meta_xgb", Probability([{}]))):
        _, rec, _, cache = a.n.run_nested(arm, t, outer, inner, spec)
        records[name] = rec
        arrays.update({f"glass.{name}.{col}": a.n._compact(arr) for col, arr in cache.items()})
    ref = {"procedures": {"glass": records}}
    gates = {("glass", j): "not met" for j in inner}
    with a.threadpool_limits(limits=1):
        out = a.evaluate(t, "glass", outer, inner, gates, spec, ref, arrays)
    assert out["identical_procedures"] == {"global": "selected"}
    assert [f["selected"] for f in out["selection"]] == ["global"] * 5
    assert all(c["interval"]["level"] == .9875 and c["interval"]["replicates"] == 4000
               and len(c["matched"]) == 20 for c in out["reference_comparisons"].values())
    assert len(rr.content_digest(out)) == 64
    arrays["glass.meta_xgb.outer.ranking"][0] = 0.
    with pytest.raises(ValueError, match="restored reference"):
        a.reference_frame("meta_xgb", t, outer, ref, arrays, spec, "glass")


def test_conditional_intervals_freeze_specification_and_strata(monkeypatch):
    t = table()
    outer = np.repeat(range(5), len(t) // 5)
    selected = {"form": "route", "penalty": .1}
    monkeypatch.setattr(a, "select", lambda *args: ([selected], [0], {}, {}))
    original = a.evidence
    count = []

    def design(rows, threshold):
        count.append(threshold)
        return original(rows, threshold)
    monkeypatch.setattr(a, "evidence", design)
    anchors = []

    def fixed(x, state, y, penalty, anchor=None):
        assert len(y) == len(t) and sum(y) == t.y.sum() and penalty == .1
        if anchor is None:
            anchors.append(float(x[0, 0]))
            return np.array([0, 1, 1, 1]), 1
        assert list(anchor) == [0, 1, 1, 1]
        return np.r_[0, np.ones(9)], 1
    monkeypatch.setattr(a, "ridge", fixed)
    out = a.coefficients(t, outer, True, a.n.load_policy())
    assert len(count) == 1 and len(anchors) == 501 and len(set(anchors)) > 1
    assert out["bootstrap_ridge_fits"] == 1000 and out["lower"] == out["upper"]


def test_abnormal_platt_recovers_stationarity_and_checks_the_boundary(monkeypatch):
    import shared.stage1.oof_calibration as calibration
    from scipy.optimize import check_grad

    rng = np.random.default_rng(293)
    x = rng.normal(-2, .22, 2400)
    y = rng.binomial(1, expit(x))
    # This case gave ABNORMAL on SciPy here. Exercise the same recovery path
    # deterministically on versions where the original line search succeeds.
    def abnormal(fun, initial, **kwargs):
        # Verify the actual summed objective and Jacobian, including the minus sign.
        probe = np.array([-.7, .3])
        assert check_grad(lambda v: fun(v)[0], lambda v: fun(v)[1], probe) < 1e-3
        return SimpleNamespace(x=initial, success=False, message="ABNORMAL")
    monkeypatch.setattr(calibration, "minimize", abnormal)
    for labels, boundary in [(y, False), (1-y, True)]:
        f = calibration.PlattCalibrator().fit(x, labels, nondecreasing=True)
        pos = labels.sum()
        target = np.where(labels, (pos+1)/(pos+2), 1/(len(y)-pos+2))
        residual = f.predict(x)-target
        assert abs(residual.mean()) < 1e-10
        slope_grad = np.mean(residual*(x-x.mean())/x.std())
        if boundary:
            assert f.a_ == 0 and slope_grad > 0
        else:
            assert -f.a_ > 0 and abs(slope_grad) < 1e-10
    # A solver returning a finite but nonstationary iterate cannot bypass verification.
    monkeypatch.setattr(calibration, "root", lambda *args, **kw: SimpleNamespace(x=np.array([2., 2.])))
    with pytest.raises(RuntimeError, match="score residual"):
        calibration.PlattCalibrator().fit(x, y, nondecreasing=True)


def test_failed_global_calibration_does_not_disqualify_route_anchor(monkeypatch):
    rows = table()
    ids = a.n.FoldPlan.build(rows.y, 3, 42).fold_id
    fit = a.PlattCalibrator.fit
    calls = 0
    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:  # last penalty, global form, first inner fold
            raise RuntimeError("injected calibration failure")
        return fit(self, *args, **kwargs)
    monkeypatch.setattr(a.PlattCalibrator, "fit", fail_once)
    with a.threadpool_limits(limits=1):
        trials, _, _, counts = a.select(rows, ids, True, a.n.load_policy())
    last = {s["form"]:s for s in trials if s["penalty"] == 1}
    assert last["global"]["failure"] == "injected calibration failure"
    assert "failure" not in last["route"] and last["route"]["locked"] is not None
    assert counts == {"ridge":96, "sigmoid":22}
