from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from feature_research import voter_diagnostics as diag


def fixture():
    # Pass 1: two leaked positives (one flagged by LR alone, one missed).
    # Pass 2: a double-negative veto loses one TP and removes one FP.
    return pd.DataFrame(dict(route=["pass1","pass1","pass2","pass2","uncertain","uncertain"],
        y=[1,1,0,1,0,1], stage1_proba=[.8,.01,.01,.01,.01,.8],
        stage3_proba=[.01,.01,.01,.01,.7,.8], router_base_rate=[.2]*6,
        router_score=[.02,.02,.6,.6,.2,.2]))


def test_accounting_and_structural_bound():
    rows = fixture()
    config = dict(thresholds={"lr":.1, "ebm":.1}, weights={"lr":.3,"ebm":.3,"glass":.4},
                  abstention={"min_weighted_confidence":.15})
    result = diag.decision_structure(rows, config, {"fp":1,"fn":2})
    trade = result["router_tradeoffs"]
    assert [trade[k] for k in ("pass1_leaked_positive","removed_fp","lost_tp")] == [2,1,1]
    assert [trade["flags"][k]["n"] for k in ("lr","ebm","both","either","neither")] == [1,0,0,1,1]
    assert trade["signed_cost_change_per_population"] == pytest.approx(1/6)
    c = result["contingency"]
    assert sum(r["n"] for r in c["joint"]) == 6
    p = next(r for r in c["pairs"] if (r["pair"],r["route"],r["label"]) == ("lr/ebm","all","all"))
    assert (p["agree"], p["double_fault"], p["only_first"], p["only_second"]) == (4,2,2,0)
    u = next(r for r in c["unique"] if (r["voter"],r["route"],r["label"]) == ("lr","all","all"))
    assert (u["sole_correct"], u["correct_lone_dissent"], u["three_votes"]) == (2,1,4)
    combo = result["composition"]
    assert sum(r["n"] for r in combo if r["retained"]) == result["retained"]
    pass2 = [r for r in result["margin_bound"] if r["route"] == "pass2"]
    assert sum(r["forced_positive"] for r in pass2) == 2
    assert all(r["mean_ceiling"] == pytest.approx(.06) for r in pass2)
    assert all(r["mean_headroom"] == pytest.approx(.006) for r in pass2)
    b = result["matched"]["routed_at_pass2_count"]
    assert b["target"] == 2 and b["counts"] is None
    assert (b["brackets"]["lr"]["below"], b["brackets"]["lr"]["inclusive"]) == (1,4)


def test_logistic_diagnostic_and_unsupported_cases():
    x = np.repeat([-1., 1.], 100)
    y = np.r_[np.arange(100)<25, np.arange(100)<75].astype(int)
    fit = diag.calibration(y, expit(x))
    assert fit["intercept"] == pytest.approx(0,abs=1e-6)
    assert fit["slope"] == pytest.approx(np.log(3),abs=1e-5)
    assert fit["slope_ci95"][0] < fit["slope"] < fit["slope_ci95"][1]
    assert diag.calibration(y, y)["status"] == "separation"
    assert diag.calibration(y, y*0+.5)["status"] == "constant score"
    assert diag.calibration(y[:12], expit(x[:12]))["slope"] is None


def test_gate_magnitude_direction_support_and_outer_isolation(monkeypatch):
    def estimates(delta):
        return {(v,s): {"slope": (delta if s == "pass1" else 0.)}
                for v in ("stage1","stage3") for s in diag.STATES}
    def verdict(d, parts):
        return diag.slope_gate(estimates(d), [estimates(x) for x in parts])["status"]
    assert verdict(.5,[.1,.2,-.9]) == "met"
    assert verdict(.49,[1.,1.,1.]) == "not met"
    assert verdict(.6,[.7,None,-1.]) == "unresolved"
    assert verdict(-.5,[.1,.2,-.1]) == "not met"
    seen = []
    def estimator(rows):
        seen.append(set(rows.index))
        return estimates(.5)
    monkeypatch.setattr(diag, "state_calibration", estimator)
    outer = np.repeat(np.arange(5), 3)
    ids = {k: np.tile(np.arange(3),4) for k in range(5)}
    diag.route_trigger({"glass":pd.DataFrame(index=np.arange(15))}, outer, ids)
    for k in range(5):
        pool, *parts = seen[4*k:4*k+4]
        assert pool == set(np.flatnonzero(outer != k))
        assert set.union(*parts) == pool and all(not a & b for a,b in combinations(parts,2))


def test_exact_flag_budget_and_accepted_input_binding(monkeypatch):
    from feature_research import stage4_nested as nested, training_inputs as inputs

    table = fixture().assign(stage1_proba=np.arange(6)/6, stage3_proba=np.arange(5,-1,-1)/6)
    flags = table.assign(lr=table.stage1_proba >= .5, ebm=table.stage3_proba >= .5,
                         router=[0,0,1,1,np.nan,np.nan])
    budget = diag.flag_budgets(table, flags)["all_at_lr_count"]
    assert budget["target"] == 3 and all(v["exact"] for v in budget["brackets"].values())
    assert sum(r["n"] for r in budget["counts"]["joint"]) == 6
    # Use accepted small input/registration records; substitute only the archive reader.
    # An archive associated with other frozen inputs must fail before any upstream extraction.
    monkeypatch.setattr(inputs, "draw_training_side", lambda **kwargs: type("Train", (), {"y_train":table.y})())
    monkeypatch.setattr(nested, "load_predictions", lambda *args, **kwargs:
                        ({"inputs":dict.fromkeys(("policy","registration","frozen_inputs","checks"))}, {}))
    with pytest.raises(ValueError, match="different input or policy"):
        diag.run()
