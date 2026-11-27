import json

import numpy as np
import pandas as pd
import pytest

from feature_research import stage4_policy as s
from stage1_calibration.evidence import stratified_bootstrap_delta


@pytest.fixture
def policy():
    return s.load_policy()


def test_loss_and_deferral_accounting(policy):
    # At cut .8: TP,TN,FP,FN once each; one correct and one incorrect decision deferred.
    y = [1, 0, 0, 1, 1, 0]
    d = [1, 0, 1, 0, 1, 1]
    confidence = [.9, .9, .8, .8, .2, .2]
    curve = s.risk_curve(y, d, confidence, policy["cost"])
    assert curve.kept.tolist() == [2, 4, 6]
    r = curve.iloc[1]
    assert (r.tp, r.tn, r.fp, r.fn) == (1, 1, 1, 1)
    assert (r.deferred_positive, r.deferred_negative, r.correct_deferred, r.wrong_deferred) == (1, 1, 1, 1)
    assert r.loss == pytest.approx(3.5/6)
    assert r.risk == .75 and r.retained_prevalence == .5
    assert r.positive_retention == r.negative_retention == 2/3
    assert r.effective_recall == 1/3
    for cost in policy["cost"]["sensitivity_defer"]:
        alt = s.risk_curve(y, d, confidence, dict(policy["cost"], defer=cost))
        assert alt.loss.iloc[1] == pytest.approx((3 + cost * 2)/6)
    # [1/2,2/3] uses .75; (2/3,1] uses 4/6.
    assert s.partial_aurc(curve, .5) == pytest.approx((.75/6 + (4/6)/3)/.5)


def test_matched_counts_and_infeasible_cut(policy):
    y, d = [1, 0, 1, 0], [1, 0, 0, 1]
    a = s.risk_curve(y, d, [.8, .8, .1, .1], policy["cost"])
    b = s.risk_curve(y, d, [.9, .8, .8, .1], policy["cost"])
    assert s.matched_risks({"a": a, "b": b}, .5) == (2, None, {"b": (1, 3)})
    assert s.matched_risks({"a": a, "b": b}, 1) == (4, {"a": .75, "b": .75}, {})
    limits = policy["coverage"]
    assert s.operating_cut(a, limits) is None  # too few observations
    assert s.operating_cut(a, dict(limits, min_retained_each_class=1)) == .8
    zeros = s.risk_curve([0, 0], [0, 1], [1, 1], policy["cost"])
    assert zeros.positive_retention.isna().all() and zeros.effective_recall.isna().all()
    assert s.operating_cut(zeros, limits) is None


def test_equal_loss_prefers_coverage_and_rejects_misalignment(policy):
    # Deferring four costs one FP; both attainable cuts have equal loss.
    y, d = [0,1,0,1,0,1,0,1], [0,1,0,1,1,1,0,1]
    c = [1]*4 + [.5]*4
    curve = s.risk_curve(y, d, c, policy["cost"])
    assert curve.loss.nunique() == 1
    assert s.operating_cut(curve, dict(policy["coverage"], min_retained_each_class=1)) == 0.
    with pytest.raises(ValueError):
        s.risk_curve(pd.Series(y), pd.Series(d, index=range(1,9)), c, policy["cost"])


def test_paired_direction_and_equivalence(policy):
    dec = policy["decision"]
    delta = np.full(12, -.01)
    ci = stratified_bootstrap_delta(delta, [0,1]*6, np.repeat([0,1,2], 4), replicates=40,
                                    seed=4302, level=.9875)
    assert s.loss_verdict(ci["lower"], ci["upper"], ci["point"], .002, dec) == "improved"
    assert s.loss_verdict(-.012, -.008, -.01, .012, dec) == "inconclusive"
    assert s.loss_verdict(-.005, .005, 0, .01, dec) == "comparable"
    assert s.loss_verdict(-.02, .01, 0, .01, dec) == "inconclusive"
    assert s.loss_verdict(.006, .01, .008, .002, dec) == "worse"


def test_registration_links_verified_inputs_and_refuses_edits(policy, tmp_path, monkeypatch):
    rr = s.records
    inp = rr.start_run(commit=2, seeds={}, root=tmp_path, run_id=policy["input_run"], notebook=s.NOTEBOOK)
    inputs = {"population": policy["population"],
              "upstreams": dict.fromkeys(("stage1.lr","stage1.mlp","stage2.glass_router",
                                          "stage2.rf_router","stage3.ebm","stage3.xgb")),
              "historical_reference": dict.fromkeys(("glass_arbiter","glass_arbiter_on_blackbox",
                                                     "meta_xgb","meta_xgb_on_glass"))}
    source = rr.write_run_record(inp, "frozen_inputs", inputs)
    receipt = s.register(tmp_path)
    saved = rr.read_run_record(receipt.parent.name, "stage4_registration", root=tmp_path)
    assert saved["frozen_inputs"]["content_sha256"] == rr.content_digest(inputs)
    assert saved["policy"]["canonical_sha256"] == rr.content_digest({"policy": policy})
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(rr.RecordError, match="sidecar"):
        s.register(tmp_path)
    monkeypatch.setattr(s, "ROOT", tmp_path)
    altered = tmp_path / s.SOURCE
    altered.parent.mkdir(parents=True)
    policy["cost"]["fn"] = 1
    altered.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="digest"):
        s.load_policy()
