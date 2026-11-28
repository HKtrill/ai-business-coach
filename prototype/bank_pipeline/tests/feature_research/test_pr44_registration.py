import json

import pytest

from feature_research import upstream_registration as reg


def test_design_covers_contrasts_and_counts_nested_fits():
    d = reg.policy()
    contrasts = {c["id"] for c in d["contrasts"]}
    assert len(contrasts) == len(d["contrasts"]) == 16
    assert contrasts == set("R1 R2 LR_fit MLP_fit calibration R5 R6 S3_fit EBM_fit XGB_fit R3 R4 RF_fit F10 transfer advanced".split())
    for c in d["contrasts"]:
        assert set(c["arms"]) <= d["budgets"].keys()
        assert c["rule"] in {*d["rules"], "by_stage"}
        assert c.get("space") or c.get("conditional")
        if c.get("conditional"):
            assert c.get("trigger")
    for arm, b in d["budgets"].items():
        assert 0 <= b["expected"] <= b["cap"] and 0 < b["hours_estimate"] <= b["hours_max"]
        if arm in ("LR", "MLP", "EBM", "XGB"):
            assert b["expected"] == 20 * (3*b["trials"]+2) + (40 if arm in ("LR", "MLP") else 0)


@pytest.mark.parametrize("interval,safe,support,expected", [
    ((.003, .01), True, True, "improved"), ((-.002, .002), True, True, "comparable"),
    ((-.01, -.003), True, True, "rejected"), ((0, .003), True, True, "inconclusive"),
    ((.003, .01), False, False, "rejected"), ((.003, .01), None, True, "inconclusive"),
    ((.003, .01), True, False, "inconclusive")])
def test_interval_rules(interval, safe, support, expected):
    assert reg.verdict(interval, .002, safeguards=safe, supported=support) == expected


def test_fitting_partition():
    reg.check_partition([0, 1, 2], [3], [2], [1])
    for fit, valid in [([0, 3], [2]), ([0, 1], [1]), ([0, 0], [1])]:
        with pytest.raises(ValueError):
            reg.check_partition([0, 1, 2], [3], fit, valid)


def test_receipt_replay_and_stale_parent(tmp_path, monkeypatch):
    # The accepted Windows run is external. Supply a synthetic parent only within this test.
    r = reg.records
    d = reg.policy()
    rid = d["baseline_run"]
    diag = r.read_run_record("c3_20261125_163120", "voter_diagnostics", root=r.BANK_ROOT/"research_logs/pr43")
    ref = r.read_run_record(diag["nested_reference"]["run_id"], "nested_references", root=r.BANK_ROOT/"research_logs/pr43")
    quality = {}
    for arm, counts in (("glass", (14628, 1120)), ("blackbox", (10839, 686))):
        quality[arm] = [dict(scope=scope, voter=voter, n=n, positive=p, negative=n-p,
                             prevalence=p/n, average_precision=.2, roc_auc=.7, log_loss=.3, brier=.1, ece=.01)
                        for scope, (n, p) in (("all", (32950, 3712)), ("deferred", counts))
                        for voter in ("stage1", "stage3")]
    baseline = dict(status="reproduced", pr43=diag["inputs"], population=reg.load_policy()["population"],
                    diagnostic={"run_id":diag["run"]["run_id"], "content_sha256":diag["content_sha256"]},
                    tables_sha256=ref["inputs"]["tables_sha256"], calibrated_oof=quality,
                    local_artifacts=dict.fromkeys(("stage1.lr","stage1.mlp","stage2.glass_router",
                                                  "stage2.rf_router","stage3.ebm","stage3.xgb"), True))
    run = r.start_run(commit=2, seeds={}, root=tmp_path, run_id=rid, notebook=reg.NOTEBOOK)
    r.write_run_record(run, "upstream_baselines", baseline)
    receipt = reg.register(tmp_path)
    got = r.read_run_record(receipt.parent.name, "upstream_registration", root=tmp_path)
    assert got["content_sha256"] == r.content_digest(reg._receipt(tmp_path))
    assert got["baseline"]["file_sha256"] == reg.sha256_file(run.folder/"upstream_baselines.json")
    parent = r.read_run_record(rid, "upstream_baselines", root=tmp_path)
    orig = r.read_run_record
    monkeypatch.setattr(r, "read_run_record", lambda i, *a, **k: parent if i == rid else orig(i,*a,**k))
    parent["tables_sha256"] = {}
    with pytest.raises(ValueError, match="upstream identities"):
        reg._receipt(tmp_path)
    path = tmp_path/"changed.json"
    path.write_text(json.dumps(dict(d, version="2.0.0")))
    monkeypatch.setattr(reg, "SOURCE", path)
    with pytest.raises(ValueError, match="digest|Changed"):
        reg.policy()


def test_conditional_cost_gate():
    b = reg.policy()["budgets"]["RF_router"]
    cost = dict(trials=b["trials"], fits=b["cap"], hours=b["hours_max"], conditional=True)
    assert not reg.within_budget(b, **cost)
    assert reg.within_budget(b, **cost, triggered=True)
    for key in ("trials", "fits", "hours"):
        assert not reg.within_budget(b, **{**cost, key: cost[key]+1}, triggered=True)
