"""
PR 42 Commit 3, Step 3B: past-only Stage 1 encodings, their predictive
comparison and the temporal protocol (``feature_research.temporal_validity``),
on synthetic training rows with a designed calendar.

The past-only states must equal the production encoder fitted on the same
rows (and the check must be able to fail); no unavailable label may reach an
encoding (and a rule that admits same-day labels must be caught); the
predictive runs must score identical rows for every paired comparison; the
contract, projection and record checks must apply their registered rules. The
real training rows are covered by ``tests/realdata/test_pr42_temporal_validity.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_research import encoding_audit as EA
from feature_research import temporal_validity as TV
from tests.support.synthetic_bank import TARGET, make_bank_frame

#: (year, month, contact days) in source order, the Step 3A periods with more days: the burn-in
#: (the first four periods) is longer than twice the largest lag plus the cold start, so the
#: lag-consistent arm has rows whose outcome is known at the cutoff and whose encoding exists.
CALENDAR = ((2008, 5, 14), (2008, 11, 12), (2009, 3, 12), (2009, 12, 10), (2010, 3, 8), (2010, 11, 8))


def _dated(n_per_day: int = 16, seed: int = 5) -> pd.DataFrame:
    """Bank-like rows in a designed date order; euribor3m cycles, so later days reuse earlier bins."""
    rows, day = [], 0
    for p, (y, m, k) in enumerate(CALENDAR):
        for d in range(k):
            rows += [{"month": m, "day_of_week": d % 5,
                      "euribor3m": round(3.0 + ((7 * day) % 11) * 0.1 + 0.001 * p, 3),
                      "cons.price.idx": 93.0 + 0.11 * p, "cons.conf.idx": -40.0 + m,
                      "emp.var.rate": 1.1 - 0.4 * (y - 2008), "nr.employed": 5200.0 - 30 * (y - 2008)}] * n_per_day
            day += 1
    cal = pd.DataFrame(rows)
    frame = make_bank_frame(n=len(cal), seed=seed).reset_index(drop=True)
    for c in cal:
        frame[c] = cal[c].astype(frame[c].dtype)
    frame.index = pd.Index(3 * np.arange(len(cal)) + 1, dtype="int64")
    return frame.sample(frac=1.0, random_state=seed)


@pytest.fixture(scope="module")
def dated():
    frame = _dated()
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].astype(int)
    return X, y, EA.build_time_index(X)


@pytest.fixture(scope="module")
def evaluated(dated):
    X, y, T = dated
    settings = TV.accepted_settings()
    settings = {**settings, "mlp_kwargs": {**settings["mlp_kwargs"], "max_epochs": 4, "patience": 2}}
    mp = pytest.MonkeyPatch()
    mp.setattr(TV, "MIN_WINDOW_ROWS", 100)            # two windows on the small calendar
    try:
        ev = TV.evaluate(X, y, T, settings=settings, verbose=False)
    finally:
        mp.undo()
    return ev


# ----------------------------------------------------------------------
# registration and windows
# ----------------------------------------------------------------------
def test_the_registration_fixes_every_rule_and_its_digest_is_stable():
    assert TV.PRIMARY in TV.RULES and TV.RULES[TV.PRIMARY]["lag"] == 0
    assert set(TV.REGISTRATION["rules"]["roles"]) == set(TV.RULES)
    for key in ("known_at_contact", "same_day", "outcome_timing", "cold_start", "unseen",
                "chronological_windows", "models", "mlp_early_stopping", "factorial", "contract_criteria"):
        assert TV.REGISTRATION[key]
    assert TV.registration_digest() == TV.registration_digest(dict(TV.REGISTRATION))
    changed = {**TV.REGISTRATION, "version": "9.9.9"}
    assert TV.registration_digest(changed) != TV.registration_digest()


def test_the_windows_follow_the_burn_in_and_the_row_minimum(dated, monkeypatch):
    _, y, T = dated
    monkeypatch.setattr(TV, "MIN_WINDOW_ROWS", 100)
    w = TV.chronological_windows(T)
    assert [x["periods"] for x in w] == [["2010-03"], ["2010-11"]]
    seg = TV.segments(T, w)
    assert (seg[T["period_index"].to_numpy() < TV.BURN_IN_PERIODS] == "burn-in").all()
    table = TV.window_table(T, y)
    assert table["rows"].sum() == len(T) and list(table["segment"]) == ["burn-in", "W1", "W2"]
    monkeypatch.setattr(TV, "MIN_WINDOW_ROWS", 10_000)          # remainder only: one window
    assert [x["periods"] for x in TV.chronological_windows(T)] == [["2010-03", "2010-11"]]


# ----------------------------------------------------------------------
# past-only states against production
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rule", list(TV.RULES))
def test_past_only_states_equal_the_production_encoder_on_the_same_rows(dated, rule):
    X, y, T = dated
    bank = TV.StateBank(X, y, T["day"].to_numpy(), np.arange(len(y)))
    res = TV.verify_against_production(X, y, T, bank, rule=rule, label="all")
    assert res["passed"] and res["states_compared"] > 0 and res["max_abs_diff"] == 0.0, res


def test_the_verification_fails_on_a_state_that_is_not_the_production_fit(dated):
    X, y, T = dated
    bank = TV.StateBank(X, y, T["day"].to_numpy(), np.arange(len(y)), k=50.0)   # wrong smoothing
    assert not TV.verify_against_production(X, y, T, bank)["passed"]


def test_a_state_of_one_contact_day_is_refused(dated):
    X, y, T = dated
    day = T["day"].to_numpy()
    bank = TV.StateBank(X, y, day, np.arange(len(y)))
    with pytest.raises(ValueError, match="more than one day"):
        bank.state(1)                                   # rows of day 0 only: one euribor3m value


def test_cold_start_lags_and_the_monthly_rule_decide_availability(dated):
    X, y, T = dated
    day, pidx = T["day"].to_numpy(), T["period_index"].to_numpy()
    bank = TV.StateBank(X, y, day, np.arange(len(y)))
    q = np.arange(len(y))
    for rule, spec in TV.RULES.items():
        enc = TV.rule_states(bank, X, T, rule, q)
        if spec["cadence"] == "daily":
            expect = day >= TV.COLD_START_DAYS + spec["lag"]
        else:
            expect = pidx > 0
        assert np.array_equal(enc["available"], expect), rule
        assert np.isnan(enc["subscribe_rate_by_euribor_bin"][~expect]).all()
        assert np.isfinite(enc["subscribe_rate_by_euribor_bin"][expect]).all()
        assert (enc["history_rows"][expect] == np.array([bank.size_before(t) for t in enc["state_day"][expect]])).all()


def test_values_outside_the_past_bins_and_unseen_cells_take_the_production_fallbacks(dated):
    X, y, T = dated
    day = T["day"].to_numpy()
    bank = TV.StateBank(X, y, day, np.arange(len(y)))
    st = bank.state(10)
    eur = np.array([0.5, 9.0])                          # below and above every past value
    cell = np.array([4 * 100 + 1, 4 * 100 + 2])         # January and February: never contacted
    e, d, fb, un = TV._apply_state(st, eur, cell)
    assert fb.all() and un.all()
    assert np.allclose(e, st.median) and np.allclose(d, st.global_rate)


# ----------------------------------------------------------------------
# invariance
# ----------------------------------------------------------------------
def test_unavailable_labels_never_reach_a_past_only_encoding(dated):
    X, y, T = dated
    ctx = {"all": np.arange(len(y)), "half": np.arange(0, len(y), 2)}
    checks = TV.invariance_checks(X, y, T, ctx, probes=6)
    assert checks["passed"].all(), checks.to_string()
    assert checks["check"].str.contains("accepted rule \\(contrast\\)").sum() == 2


def test_a_rule_that_admits_same_day_labels_is_caught(dated, monkeypatch):
    X, y, T = dated
    monkeypatch.setitem(TV.RULES, TV.PRIMARY, {"cadence": "daily", "lag": -1, "role": "leaky"})
    checks = TV.invariance_checks(X, y, T, {"all": np.arange(len(y))}, probes=6).set_index("check")["passed"]
    assert not checks[f"all {TV.PRIMARY}: flipping the same day's labels alone leaves them unchanged"]


# ----------------------------------------------------------------------
# predictive runs
# ----------------------------------------------------------------------
def test_every_registered_protocol_is_scored_on_the_rows_its_rule_encodes(dated, evaluated):
    X, y, T = dated
    ev = evaluated
    n = len(y)
    day = T["day"].to_numpy()
    for name in ("lr|random|accepted", "mlp|random|accepted|es_accepted", "mlp|random|accepted|es_isolated"):
        assert np.array_equal(ev["scores"][name]["rows"], np.arange(n)), name
    for rule, spec in TV.RULES.items():
        rows = ev["scores"][f"lr|random|{rule}"]["rows"]
        if spec["cadence"] == "daily":
            assert np.array_equal(rows, np.flatnonzero(day >= TV.COLD_START_DAYS + spec["lag"])), rule
    windows = np.flatnonzero(TV.segments(T, ev["windows"]) != "burn-in")
    for name, v in ev["scores"].items():
        if "|chrono|" in name:
            assert np.array_equal(v["rows"], windows), name
        assert np.isfinite(v["scores"]).all() and len(v["rows"]) == len(np.unique(v["rows"]))
    assert ev["chrono_checks"]["passed"].all(), ev["chrono_checks"].to_string()


def test_isolated_early_stopping_stops_on_the_reproduced_subset(evaluated):
    notes = evaluated["notes"]
    iso = notes[notes["early_stopping"] == "isolated"]
    assert len(iso) == 2 * (10 + len(evaluated["windows"])) and iso["split_reproduced"].all()
    assert notes[notes["early_stopping"] == "accepted"]["split_reproduced"].isna().all()
    rep = TV.reproduction_checks(evaluated).set_index("check")["passed"]
    assert rep[[c for c in rep.index if c.startswith("every isolated")]].all()


def test_paired_comparisons_use_identical_rows_and_the_factorial_is_their_contrast(dated, evaluated):
    ev = evaluated
    C = TV.comparisons(ev)
    pooled = C[C["scope"] == "pooled"]
    assert set(pooled["question"]) == {q for q, *_ in TV.COMPARISONS}
    for _, r in pooled.iterrows():
        rows, _ = TV._aligned(ev, [r["a"], r["b"]])
        assert r["rows"] == len(rows) and r["positives"] == int(ev["y"][rows].sum())
    for arm in ("lr", "mlp"):
        f = TV.factorial(ev, arm)
        auc = dict(zip(f["cells"]["cell"], f["cells"]["pooled_roc_auc"]))
        eff = dict(zip(f["effects"]["effect"], f["effects"]["pooled_diff"]))
        assert eff["past-only vs accepted protocol, random split (B - A)"] == pytest.approx(
            auc["B"] - auc["A"], abs=1e-12)
        assert eff["chronological vs random split, past-only protocol (D - B)"] == pytest.approx(
            auc["D"] - auc["B"], abs=1e-12)
        assert eff["interaction ((D - C) - (B - A))"] == pytest.approx(
            auc["D"] - auc["C"] - auc["B"] + auc["A"], abs=1e-12)
    table = TV.protocol_table(ev)
    assert set(table["protocol"]) == set(ev["scores"])
    mean = C[C["scope"] == "mean over windows"]
    assert set(mean["question"]) == set(pooled["question"])
    for _, r in mean.iterrows():
        per = C[(C["question"] == r["question"]) & C["scope"].isin([w["window"] for w in ev["windows"]])]
        assert r["diff"] == pytest.approx(per["diff"].mean(), abs=1e-12)


# ----------------------------------------------------------------------
# outcome lag in the chronological windows
# ----------------------------------------------------------------------
def test_the_lag_consistent_arm_fits_only_outcomes_known_at_the_cutoff(dated, evaluated):
    _, y, T = dated
    day = T["day"].to_numpy()
    fits = evaluated["lag_consistent_fits"]
    w = {x["window"]: x for x in evaluated["windows"]}
    for _, r in fits.iterrows():
        spec = TV.RULES[r["rule"]]
        cut = w[r["window"]]["first_day"]
        before = day < cut
        known = TV.known_at_cutoff(r["rule"], cut, day) & before
        assert r["training_rows_before_cutoff"] == before.sum() and r["outcomes_known_at_cutoff"] == known.sum()
        assert r["fit_rows"] <= r["outcomes_known_at_cutoff"]
        if spec["cadence"] == "daily" and spec["lag"] > 0:
            assert r["outcomes_known_at_cutoff"] < r["training_rows_before_cutoff"]
    checks = evaluated["chrono_checks"].set_index("check")["passed"]
    assert checks[[c for c in checks.index if "at lag 0 the lag-consistent model" in c]].all()
    regression = [c for c in checks.index if "still unavailable" in c]
    assert len(regression) == 2 * (len(TV.RULES) - 1) and checks[regression].all()
    contrast = [c for c in checks.index if "(contrast): the same flip changes the lag-0 model" in c]
    assert len(contrast) == 4 and checks[contrast].all()


def test_a_lag_arm_that_trains_on_unavailable_outcomes_is_caught(dated, monkeypatch):
    X, y, T = dated
    monkeypatch.setattr(TV, "MIN_WINDOW_ROWS", 100)
    windows = TV.chronological_windows(T)
    yv = np.asarray(y).astype(int)
    from shared.stage1.feature_engineering import LRFeatureEngineer

    cellular = TV._quiet(LRFeatureEngineer().fit_transform, X, y)["cellular_downturn"].to_numpy(float)
    settings = TV.accepted_settings()
    ok = pd.DataFrame(TV._lag_consistent_invariance(X, yv, T, windows, cellular, settings))
    assert ok["passed"].all(), ok.to_string()
    # a leaky arm: every pre-cutoff outcome counts as known, whatever the lag
    monkeypatch.setattr(TV, "known_at_cutoff", lambda rule, cutoff, day: day < cutoff)
    leaky = pd.DataFrame(TV._lag_consistent_invariance(X, yv, T, windows, cellular, settings))
    daily = leaky[leaky["check"].str.contains("past_daily") & leaky["check"].str.contains("still unavailable")]
    assert len(daily) == 4 and not daily["passed"].any()


def test_delong_matches_the_rank_auc_and_is_antisymmetric():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    a, b = rng.normal(size=400) + y, rng.normal(size=400) + 0.5 * y
    ab, ba = TV.delong_paired(y, a, b), TV.delong_paired(y, b, a)
    assert ab["auc_a"] == pytest.approx(roc_auc_score(y, a)) and ab["auc_b"] == pytest.approx(roc_auc_score(y, b))
    assert ab["diff"] == pytest.approx(-ba["diff"]) and ab["se"] == pytest.approx(ba["se"])
    same = TV.delong_paired(y, a, a)
    assert same["diff"] == 0.0 and same["se"] == 0.0 and same["p_value"] == 1.0


def test_availability_tables_cover_every_scope_and_rule(evaluated):
    A = TV.availability_tables(evaluated)
    for scope, *_ in TV._SCOPES:
        assert set(A[A["scope"] == scope]["rule"]) == set(TV.RULES), scope
    random = A[A["scope"] == "random held-out rows"].set_index("rule")
    assert (random["rows"] == len(evaluated["y"])).all()
    assert (random["available"] + random["cold_start_rows"] == random["rows"]).all()


# ----------------------------------------------------------------------
# decisions, projection, records
# ----------------------------------------------------------------------
def _decision_inputs(share=0.97, differ=500, passed=True):
    av = pd.DataFrame([{"rule": r, "available_share": share, "available": 1000,
                        "subscribe_rate_by_euribor_bin_rows_differ": differ,
                        "subscribe_rate_by_dow_month_rows_differ": 0} for r in TV.RULES])
    inv = pd.DataFrame([{"check": f"all {r}: flipping", "passed": passed} for r in TV.RULES])
    cmp_ = pd.DataFrame([{"b": f"lr|random|{r}", "scope": "pooled", "diff": -0.01} for r in TV.RULES])
    return av, inv, cmp_


def test_contract_decisions_apply_the_registered_criteria():
    d = TV.contract_decisions(*_decision_inputs()).set_index("candidate")
    assert d.loc[TV.PRIMARY, "needs_versioned_contract"]
    assert d.loc[TV.PRIMARY, "destination"].startswith("PR 44")
    assert all(d.loc[r, "destination"].startswith("PR 49") for r in TV.RULES if r != TV.PRIMARY)
    assert not d.loc["accepted (PR 39)", "needs_versioned_contract"]
    low = TV.contract_decisions(*_decision_inputs(share=0.90)).set_index("candidate")
    assert low.loc[TV.PRIMARY, "destination"].startswith("PR 49 only")
    same = TV.contract_decisions(*_decision_inputs(differ=5)).set_index("candidate")
    assert not same.loc[TV.PRIMARY, "needs_versioned_contract"]
    leaky = TV.contract_decisions(*_decision_inputs(passed=False)).set_index("candidate")
    assert not leaky.loc[TV.PRIMARY, "realizable"] and not leaky.loc[TV.PRIMARY, "needs_versioned_contract"]


def test_the_candidate_contract_is_distinct_and_leaves_the_accepted_one_frozen():
    from shared.stage1.feature_engineering import STAGE1_FEATURE_CONTRACT, stage1_feature_contract

    c = TV.candidate_contract()
    assert c["id"] == STAGE1_FEATURE_CONTRACT["id"] and c["version"].endswith("-candidate")
    assert not set(c["features"][1:]) & set(STAGE1_FEATURE_CONTRACT["features"])
    assert c["fingerprint"] != STAGE1_FEATURE_CONTRACT["fingerprint"]
    assert stage1_feature_contract(c["features"]) is None                 # not registered
    assert (STAGE1_FEATURE_CONTRACT["version"], STAGE1_FEATURE_CONTRACT["fingerprint"]) == \
        ("2.0.0", "d0d8b07f2ca5e81d")
    assert c["accepted_contract"]["fingerprint"] == STAGE1_FEATURE_CONTRACT["fingerprint"]


def test_the_prior_shift_projection_reports_like_observe_routing():
    P = TV.stage2_prevalence_projection({"low": 0.05, "high": 0.40})
    routing = P["routing"]["counts"]
    assert routing["glass"]["pass1_rows"] == 11659 and routing["glass"]["positives"] == 3712
    proj = P["projection"].set_index(["arm", "population"])
    for arm in routing:
        obs = proj.loc[(arm, "training OOF (observed)")]
        c = routing[arm]
        assert obs["pass1_npv"] == (c["pass1_rows"] - c["pass1_positives"]) / c["pass1_rows"]
        lo, hi = proj.loc[(arm, "low")], proj.loc[(arm, "high")]
        assert lo["pass1_leakage_share"] == pytest.approx(hi["pass1_leakage_share"], abs=1e-4)
        assert lo["pass1_npv"] > obs["pass1_npv"] > hi["pass1_npv"]
        assert lo["pass2_precision"] < obs["pass2_precision"] < hi["pass2_precision"]
    band = P["bands"].set_index("arm").loc["glass"]
    inside = (band["precision_holds_from_prevalence"] + band["npv_holds_up_to_prevalence"]) / 2
    edge = TV.stage2_prevalence_projection({"inside": inside, "above": band["npv_holds_up_to_prevalence"] + 0.01,
                                            "below": band["precision_holds_from_prevalence"] - 0.01})
    e = edge["projection"].set_index(["arm", "population"])
    assert e.loc[("glass", "inside"), "pass1_npv_status"] == "satisfied"
    assert e.loc[("glass", "inside"), "pass2_precision_status"] == "satisfied"
    assert e.loc[("glass", "above"), "pass1_npv_status"] == "violated"
    assert e.loc[("glass", "below"), "pass2_precision_status"] == "violated"


def test_the_protocol_and_finding_cover_the_step3b_obligations():
    for key in ("split", "fit_at_each_cutoff", "updated_after_cutoff", "prevalence_shift",
                "prior_shift_attribution", "missed_obligation_report", "pre_registration"):
        assert TV.PR49_PROTOCOL[key]
    for key in ("calibration", "thresholds", "abstention", "stage2_obligations"):
        assert TV.PR49_PROTOCOL["prevalence_shift"][key]
    d = TV.contract_decisions(*_decision_inputs())
    cmp_ = pd.DataFrame(columns=["question", "arm", "scope", "a", "b", "auc_a", "auc_b", "diff",
                                 "ci_low", "ci_high", "rows"])
    f = TV.interim_finding(cmp_, {}, d)
    for key in ("finding", "pr43", "pr44_45", "pr49_open", "contracts"):
        assert f[key]


#: The environment the accepted Step 3A run was written in (a Windows workstation).
WINDOWS_ENV = {"python": "3.13.1", "executable": r"C:\Users\someone\.venv\Scripts\python.exe",
               "platform": "Windows-11-10.0.26100-SP0", "os": "Windows",
               "packages": {"numpy": "2.5.3", "pandas": "3.0.5", "scikit-learn": "1.9.1"}}


def _accepted_step3a(root, artifacts=("passed",), environment=None):
    """A stand-in accepted Step 3A run written the production way: (accepted digests, records, environment)."""
    from feature_research import research_records as RR

    run = RR.start_run(commit=3, seeds={"split": 42}, root=root, run_id="c3_20261122_201830")
    if environment is not None:
        run.environment = dict(environment)
    roles = [{"role": f"role{i}", "status": st} for i, st in enumerate(artifacts)]
    RR.write_run_record(run, "feature_inventory", {"accepted_artifacts": roles, "content": [1, 2]})
    RR.write_run_record(run, "encoding_audit", {"content": [3]})
    recs = {n: RR.read_run_record(run.run_id, n, root=root) for n in ("feature_inventory", "encoding_audit")}
    accepted = {"run_id": run.run_id, **{n: r["content_sha256"] for n, r in recs.items()}}
    return accepted, recs, run.environment


def _this_run(recs, *, same=True, artifacts=None):
    out = {}
    for n, r in recs.items():
        rec = dict(r)
        if not same:
            rec["content_sha256"] = "0" * 64
        if artifacts is not None and n == "feature_inventory":
            rec["accepted_artifacts"] = [{"role": f"role{i}", "status": st} for i, st in enumerate(artifacts)]
        out[n] = rec
    return out


def test_step3a_checks_are_not_applicable_without_the_accepted_record(tmp_path):
    from feature_research import research_records as RR

    run = RR.start_run(commit=3, seeds={"split": 42}, root=tmp_path)
    RR.write_run_record(run, "feature_inventory", {"accepted_artifacts": [], "content": [1]})
    RR.write_run_record(run, "encoding_audit", {"content": [2]})
    recs = {n: RR.read_run_record(run.run_id, n, root=tmp_path) for n in ("feature_inventory", "encoding_audit")}
    table = TV.step3a_reproduction(recs, run.environment, root=tmp_path)
    assert (table["accepted_record"] == TV.STEP3A_ABSENT).all() and not table["reproduction_required"].any()
    assert (table["reproduction_basis"] == "accepted run directory absent").all()
    assert TV.step3a_checks(table)["passed"].all()


def test_the_matching_environment_must_reproduce_the_accepted_step3a_record(tmp_path):
    accepted, recs, env = _accepted_step3a(tmp_path)
    table = TV.step3a_reproduction(_this_run(recs), env, accepted=accepted, root=tmp_path)
    assert table["integrity_ok"].all() and table["reproduction_required"].all() and table["reproduced"].all()
    assert TV.step3a_checks(table)["passed"].all()
    differs = TV.step3a_reproduction(_this_run(recs, same=False), env, accepted=accepted, root=tmp_path)
    assert differs["reproduction_required"].all()
    assert not TV.step3a_checks(differs)["passed"].all()


def test_a_checkout_with_the_accepted_record_in_another_environment_checks_integrity_only(tmp_path):
    # the accepted run was written on Windows; this checkout (for example Linux, after the accepted
    # run is committed) holds it, and its own Step 3A digests differ
    from upstream_regeneration.records import environment

    accepted, recs, _ = _accepted_step3a(tmp_path, environment=WINDOWS_ENV)
    here = environment()
    table = TV.step3a_reproduction(_this_run(recs, same=False), here, accepted=accepted, root=tmp_path)
    assert table["integrity_ok"].all() and not table["reproduction_required"].any()
    assert table["reproduction_basis"].str.contains("platform differs").all()
    checks = TV.step3a_checks(table)
    assert checks["passed"].all(), checks.to_string()
    assert checks["check"].str.contains("exact reproduction not required").sum() == 2
    # same environment, but the local artifacts differ: reproduction is not required either
    same_env = tmp_path / "same_environment"
    accepted, recs, env = _accepted_step3a(same_env)
    other = TV.step3a_reproduction(_this_run(recs, same=False, artifacts=("not available",)), env,
                                   accepted=accepted, root=same_env)
    assert not other["reproduction_required"].any()
    assert other["reproduction_basis"].str.contains("artifact availability").all()
    assert TV.step3a_checks(other)["passed"].all()


@pytest.mark.parametrize("missing", ["encoding_audit.json", "feature_inventory.json",
                                     "encoding_audit.json.sha256"])
def test_an_incomplete_accepted_run_fails_its_integrity_check_in_another_environment(tmp_path, missing):
    from upstream_regeneration.records import environment

    accepted, recs, _ = _accepted_step3a(tmp_path, environment=WINDOWS_ENV)
    here = environment()
    valid = TV.step3a_reproduction(_this_run(recs, same=False), here, accepted=accepted, root=tmp_path)
    assert valid["integrity_ok"].all() and TV.step3a_checks(valid)["passed"].all()   # valid to start with
    (tmp_path / accepted["run_id"] / missing).unlink()
    table = TV.step3a_reproduction(_this_run(recs, same=False), here, accepted=accepted, root=tmp_path)
    bad = table.set_index("record").loc[f"{accepted['run_id']}/{missing.split('.')[0]}"]
    assert not bad["integrity_ok"] and "incomplete" in bad["accepted_record"] and missing in bad["accepted_record"]
    assert not bad["reproduction_required"]
    checks = TV.step3a_checks(table)
    assert not checks["passed"].all()
    assert checks[~checks["passed"]]["check"].str.endswith(": integrity").all()


def test_an_absent_accepted_run_directory_is_reported_unavailable(tmp_path):
    import shutil

    accepted, recs, env = _accepted_step3a(tmp_path)
    shutil.rmtree(tmp_path / accepted["run_id"])
    table = TV.step3a_reproduction(_this_run(recs, same=False), env, accepted=accepted, root=tmp_path)
    assert (table["accepted_record"] == TV.STEP3A_ABSENT).all() and not table["reproduction_required"].any()
    assert TV.step3a_checks(table)["passed"].all()


def test_a_damaged_accepted_record_fails_its_integrity_check_in_any_environment(tmp_path):
    accepted, recs, env = _accepted_step3a(tmp_path)
    path = tmp_path / accepted["run_id"] / "encoding_audit.json"
    path.write_bytes(path.read_bytes().replace(b"[\n    3\n  ]", b"[\n    4\n  ]"))
    elsewhere = {**env, "platform": "another platform"}
    table = TV.step3a_reproduction(_this_run(recs, same=False), elsewhere, accepted=accepted, root=tmp_path)
    bad = table.set_index("record").loc[f"{accepted['run_id']}/encoding_audit"]
    assert not bad["integrity_ok"] and bad["accepted_record"].startswith("integrity failed")
    assert not TV.step3a_checks(table)["passed"].all()
    wrong = {**accepted, "feature_inventory": "f" * 64}
    table = TV.step3a_reproduction(_this_run(recs), env, accepted=wrong, root=tmp_path)
    inv = table.set_index("record").loc[f"{accepted['run_id']}/feature_inventory"]
    assert not inv["integrity_ok"] and not TV.step3a_checks(table)["passed"].all()


def test_the_module_imports_no_duplicated_feature_code_and_opens_no_artifact():
    from feature_research import feature_inventory as FI

    tree = ast.parse(Path(TV.__file__).read_bytes())
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    for target in FI.TARGET_MODULES.values():
        assert not [m for m in modules if m and (m == target or m.startswith(target + "."))], target
    assert "joblib" not in modules and "pickle" not in modules
    calls = {getattr(n.func, "attr", getattr(n.func, "id", None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"load", "read_csv", "read_sql_query"}
