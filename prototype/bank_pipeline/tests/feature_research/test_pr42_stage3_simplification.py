"""
PR 42 Commit 5, Step 5A: ``feature_research.stage3_simplification`` on synthetic
data (the real training rows: ``tests/realdata/test_pr42_stage3_simplification.py``).

Covered: no new importer of the Stage 2 or research feature code; the registration
(pinned digest, the rules it fixes); the baseline's sources (features, pairs,
protocol, the transcribed hyperparameters from the pinned Git blob, unavailable and
changed sources); pair mapping after removal and replacement, and identical
configurations fitted once; the past-only density (the production kernel, same-day
exclusion, cold start and minimum history, contact ordering, held-out isolation);
the Stage 3 path on a synthetic training side (fold isolation of the whole fit,
fitted terms, out-of-fold coverage, shapes and support); the support-defined pair
cells and the post-registration centring check; the past-only fallback (later labels
move an earlier row's fallback, never its raw value) and its trace to the EBM frames;
the accepted views (missing, failing pin, failing boundary, required) and the
complementarity requirement; the complementarity counts
and their reconciliation; the nomination rules; and the notebook's Step 5A cells by
source.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_research import stage3_simplification as SS
from tests.support import stage2_crossfit_world as W

BANK = Path(SS.__file__).resolve().parents[1]
FEATURE_MODULES = ("shared.stage2.feature_engineering", "feature_research.feature_engineering")


# ======================================================================
# no new importer of the feature code
# ======================================================================
def _imported_modules(path: Path) -> set:
    out = set()
    for n in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


@pytest.mark.parametrize("path", [BANK / "feature_research" / "stage3_simplification.py", Path(__file__)])
def test_step5a_code_imports_no_stage2_or_research_feature_engineering(path):
    mods = _imported_modules(path)
    assert not [m for m in mods for t in FEATURE_MODULES if m == t or m.startswith(t + ".")]


def test_the_commit2_caller_scan_does_not_list_step5a():
    from feature_research import feature_inventory as FI
    assert "stage3_simplification" not in json.dumps(FI.duplicate_callers())


# ======================================================================
# registration and sources
# ======================================================================
def test_the_registration_is_pinned_and_any_change_moves_its_digest():
    assert SS.registration_digest() == SS.REGISTERED_SHA256
    assert SS.REGISTRATION["version"] == "1.0.0"
    changed = copy.deepcopy(SS.REGISTRATION)
    changed["nomination"]["step5b"] += " "
    assert SS.registration_digest(changed) != SS.REGISTERED_SHA256
    for key in ("baseline", "parity", "correlated_pair", "candidates", "past_only_density", "evaluation", "metrics",
                "shapes", "complementarity", "nomination"):
        assert SS.REGISTRATION[key]
    assert set(SS.CANDIDATES) <= set(SS.REGISTRATION["candidates"])
    assert "lag 0" in SS.REGISTRATION["past_only_density"]["outcome_availability"]


def test_the_baseline_sources_are_the_accepted_ones():
    checks = SS.baseline_checks()
    assert checks["passed"].all(), checks.to_string()
    assert SS.params_transcription()["status"] == "verified"
    fig = SS.accepted_figures()
    assert fig["threshold_raw"] == 0.53 and len(fig["nested_by_fold"]) == 5


def test_the_transcription_reads_only_the_pinned_notebook(tmp_path, monkeypatch):
    blob = SS._git_blob(SS.PARAMS_SOURCE["blob"], BANK)
    assert blob is not None and SS._blob_id(blob) == SS.PARAMS_SOURCE["blob"]
    monkeypatch.setattr(SS, "_git_blob", lambda b, root: None)
    assert SS.params_transcription(tmp_path)["status"] == "unavailable"            # no Git, no copy
    target = tmp_path / SS.PARAMS_SOURCE["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(blob.replace(b"\n", b"\r\n"))                              # a CRLF checkout of the blob
    assert SS.params_transcription(tmp_path)["status"] == "verified"
    target.write_bytes(blob.replace(b"learning_rate = 0.049940", b"learning_rate = 0.049941"))
    assert SS.params_transcription(tmp_path)["status"] == "unavailable"            # not the pinned blob
    monkeypatch.setattr(SS, "_git_blob", lambda b, root: blob.replace(b"max_bins = 480", b"max_bins = 448"))
    assert SS.params_transcription(tmp_path)["status"] == "mismatch"


# ======================================================================
# configurations
# ======================================================================
def test_pairs_are_resolved_by_name_and_lost_slots_stay_empty():
    c = SS.configurations()
    assert c["baseline"]["pairs"] == list(SS.BASELINE_PAIRS)
    assert c["drop_first"]["lost_pairs"] == [SS.BASELINE_PAIRS[1]] and c["drop_first"]["n_features"] == 12
    assert c["drop_second"]["lost_pairs"] == [SS.BASELINE_PAIRS[0]] and c["drop_second"]["n_features"] == 12
    for name, new in (("jed_alone", SS.JED), ("past_density", SS.PAST_FEATURE)):
        assert c[name]["lost_pairs"] == [SS.BASELINE_PAIRS[2]] and c[name]["n_features"] == 13
        assert c[name]["features"][0] == new and "decay_x_density" not in c[name]["features"]
        assert all(new not in p for p in c[name]["pairs"])                       # no inherited pair
    assert SS.configuration_checks(c)["passed"].all()


def test_identical_configurations_are_fitted_once_and_both_mapped(monkeypatch):
    reg = copy.deepcopy(SS.REGISTRATION)
    reg["candidates"]["jed_alone"] = dict(reg["candidates"]["drop_first"], question="duplicate question")
    monkeypatch.setattr(SS, "REGISTRATION", reg)
    c = SS.configurations()
    assert c["jed_alone"] == {"same_as": "drop_first", "questions": ["duplicate question"]}
    assert c["drop_first"]["questions"] == ["correlated pair, first ablation", "duplicate question"]


def test_an_added_feature_already_present_is_kept_once(monkeypatch):
    reg = copy.deepcopy(SS.REGISTRATION)
    reg["candidates"]["jed_alone"] = {"question": "q", "remove": ["decay_x_density"], "add": ["age"]}
    monkeypatch.setattr(SS, "REGISTRATION", reg)
    c = SS.configurations()["jed_alone"]
    assert c["features"].count("age") == 1 and c["n_features"] == 12


# ======================================================================
# the past-only density
# ======================================================================
def _ctx(n=900, seed=0):
    rng = np.random.default_rng(seed)
    macro = rng.normal(size=(n, 3)).round(1)
    day = np.sort(rng.integers(0, 60, n))
    y = (rng.random(n) < 0.3).astype(int)
    return macro, y, day


def test_past_density_is_the_production_kernel_on_earlier_days_only():
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import StandardScaler
    macro, y, day = _ctx()
    out = SS.past_density(macro, y, day, macro, day)
    d = 40
    q = np.flatnonzero(day == d)
    hist = day < d
    sc = StandardScaler().fit(macro[hist])
    w = np.exp(-cdist(sc.transform(macro[q]), sc.transform(macro[hist])) ** 2 / 2)
    w /= w.sum(axis=1, keepdims=True) + 1e-10
    assert np.allclose(out["value"][q], w @ y[hist], rtol=0, atol=1e-12)
    assert (out["history_rows"][q] == hist.sum()).all()


def test_same_day_and_later_labels_never_reach_a_value():
    macro, y, day = _ctx()
    base = SS.past_density(macro, y, day, macro, day)["value"]
    d = 30
    flipped = y.copy()
    flipped[day >= d] = 1 - flipped[day >= d]
    again = SS.past_density(macro, flipped, day, macro, day)["value"]
    assert np.array_equal(base[day <= d], again[day <= d], equal_nan=True)
    assert not np.allclose(base[day > d], again[day > d], equal_nan=True)


def test_cold_start_and_short_history_are_missing():
    macro, y, day = _ctx()
    out = SS.past_density(macro, y, day, macro, day)
    assert np.isnan(out["value"][day < SS.COLD_START_DAYS]).all()
    valued = ~np.isnan(out["value"])
    assert (out["history_rows"][valued] >= SS.MIN_HISTORY_ROWS).all()
    assert (out["history_positives"][valued] >= SS.MIN_HISTORY_POSITIVES).all()
    short = SS.past_density(macro[:100], y[:100], day[:100], macro, day)["value"]
    assert np.isnan(short).all()                                         # 100 rows never reach 500


def test_query_labels_never_matter():
    macro, y, day = _ctx()
    ctx, q = np.arange(0, 900, 2), np.arange(1, 900, 2)
    a = SS.past_density(macro[ctx], y[ctx], day[ctx], macro[q], day[q])["value"]
    yy = y.copy()
    yy[q] = 1 - yy[q]
    b = SS.past_density(macro[ctx], yy[ctx], day[ctx], macro[q], day[q])["value"]
    assert np.array_equal(a, b, equal_nan=True)


# ======================================================================
# the Stage 3 path on a synthetic training side
# ======================================================================
FAST = {"learning_rate": 0.05, "max_rounds": 60, "max_bins": 32, "max_interaction_bins": 16}


@pytest.fixture(scope="module")
def world():
    import feature_research.encoding_audit as EA
    mp = pytest.MonkeyPatch()
    mp.setattr(SS, "PARAMS", FAST)
    mp.setattr(SS, "MIN_HISTORY_ROWS", 50)
    mp.setattr(SS, "MIN_HISTORY_POSITIVES", 3)
    ts = W.synthetic_train_side()
    X, y = ts.X_train, ts.y_train.astype(int)
    T = EA.build_time_index(X)
    block = SS.training_block(X, y)
    past = SS.add_past_density(block, X, y, T)
    configs = SS.configurations()
    frames = SS.frames_for_fits(block, past)
    fits = {k: SS.fit_configuration(configs[k], block, frames) for k in ("baseline", "drop_second", "past_density")}
    yield dict(X=X, y=y, T=T, block=block, past=past, configs=configs, frames=frames, fits=fits, mp=mp)
    mp.undo()


def test_the_synthetic_block_keeps_rows_folds_and_the_decay_product(world):
    chk = SS.block_checks(world["block"], world["X"], world["y"])
    rows = chk.set_index("check")["passed"]
    assert rows["decay_x_density = joint_economic_decay x neighborhood_subscription_density in every frame"]
    assert rows["the outer held-out rows partition the training rows"]
    assert rows["no inner frame touches its outer fold's held-out rows"]


def test_fitted_terms_are_the_configurations_features_and_pairs(world):
    for name, r in world["fits"].items():
        assert r["terms_ok"], (name, r["terms"][0], r["expected_terms"])
        assert np.isfinite(r["proba"]).all() and np.isfinite(r["calibrated"]).all()
    assert "campaign & emp_var_rate_sigmoid_slope" not in world["fits"]["drop_second"]["expected_terms"]
    assert SS.PAST_FEATURE in world["fits"]["past_density"]["expected_terms"]


def test_flipping_a_folds_held_out_labels_leaves_its_whole_fit_unchanged(world):
    # Folds are fixed (a flip would re-stratify a new plan); every input that carries labels is flipped.
    block, frames, k = world["block"], world["frames"], 2
    held = block.tune_folds[k].val_idx
    yf = block.y_train.copy()
    yf.iloc[held] = 1 - yf.iloc[held]
    b2 = copy.copy(block)
    b2.y_train = yf
    outer = [copy.copy(f) for f in frames["outer"]]
    outer[k].y_val = 1 - outer[k].y_val
    r = world["fits"]["past_density"]
    r2 = SS.fit_configuration(world["configs"]["past_density"], b2, {**frames, "outer": outer})
    assert np.array_equal(r2["proba"], r["proba"])                       # no model reads a held-out label
    assert np.array_equal(r2["calibrated"][held], r["calibrated"][held])  # fold k's map: its inner scores only
    assert r2["nested"]["by_fold"][k] == r["nested"]["by_fold"][k]
    others = np.setdiff1d(np.arange(len(yf)), held)
    assert not np.array_equal(r2["calibrated"][others], r["calibrated"][others])   # control: other folds move
    chk = SS.isolation_checks(r, block)
    assert chk["passed"].all(), chk.to_string()


def test_metrics_paired_differences_and_shapes_on_the_synthetic_fits(world):
    fits, y = world["fits"], world["y"]
    mt = SS.operating_point_table(fits, y)
    assert set(mt["probabilities"]) == {"raw", "calibrated"} and len(mt) == 3 * len(fits)
    diff = SS.paired_differences(fits, y, world["block"].tune_plan.fold_id)
    pooled = diff["pooled"].set_index("configuration")
    assert pooled.loc["baseline", "d_pr_auc"] == 0.0 and "d_roc_auc_se" in pooled.columns
    assert set(diff["per_fold"]["fold"]) == set(range(5))
    curves = SS.shape_curves(fits, world["configs"], world["frames"])
    g = curves[(curves["configuration"] == "baseline") & (curves["feature"] == "age")]
    assert set(g["model"]) == {f"fold {k}" for k in range(5)} | {"all rows"}
    m = fits["baseline"]["full"]
    ref = world["frames"]["X_all"][list(world["configs"]["baseline"]["features"])]
    contrib = m.eval_terms(ref)[:, list(m.term_names_).index("age")]
    allrows = g[g["model"] == "all rows"]
    assert abs(contrib.mean() - allrows["shift"].iloc[0]) < 1e-12              # centred on the training rows
    sup = SS.support(world["frames"]["X_all"], world["T"], "age")
    assert sup["rows"].sum() + sup.attrs["outside_range"] == len(y)
    summ = SS.shape_summary(curves, SS.support_tables(world["frames"]["X_all"], world["T"]))
    assert {"fold_sd_dense", "sign_changes_dense", "range_dense"} <= set(summ.columns)
    pairs = SS.pair_surfaces(fits, world["configs"], world["frames"], world["T"], bins=4)
    assert {s["pair"] for s in pairs["surfaces"] if s["configuration"] == "drop_second"} == {
        "euribor3m_sigmoid_slope & economic_curvature_intensity", "prior_x_stress & decay_x_density"}


def test_pair_cells_follow_the_support_and_merge_floating_point_duplicates():
    v = np.r_[np.full(50, 0.1), np.full(50, 0.1 + 1e-12), np.full(30, 0.5)]
    idx, labels = SS.pair_bins(v)
    assert labels == ["0.1", "0.5"] and np.array_equal(np.bincount(idx), [100, 30])
    rng = np.random.default_rng(0)
    w = rng.normal(size=2000)
    idx, labels = SS.pair_bins(w, bins=10)
    assert len(labels) == 10 and len(set(labels)) == 10
    counts = np.bincount(idx, minlength=len(labels))
    assert (counts > 0).all() and counts.sum() == len(w)            # every row in one cell, no empty cell
    hi = [w[idx == j].max() for j in range(len(labels))]
    lo = [w[idx == j].min() for j in range(len(labels))]
    assert all(h < nxt for h, nxt in zip(hi[:-1], lo[1:]))         # cells are ordered, disjoint ranges
    campaign = np.r_[np.full(430, 1), np.full(260, 2), np.full(130, 3), rng.integers(4, 57, 180)]
    _, labels = SS.pair_bins(campaign)
    assert labels[:3] == ["1", "2", "3"] and labels[-1].endswith("-56")


def test_pair_cells_are_the_mean_centred_row_contributions_and_empty_cells_stay_empty(world):
    fits, configs, frames, T = world["fits"], world["configs"], world["frames"], world["T"]
    pairs = SS.pair_surfaces(fits, configs, frames, T, bins=4)
    s = next(x for x in pairs["surfaces"]
             if x["configuration"] == "baseline" and x["pair"] == "campaign & emp_var_rate_sigmoid_slope")
    X_all = frames["X_all"]
    m = fits["baseline"]["full"]
    c = m.eval_terms(X_all[list(configs["baseline"]["features"])])[:, list(m.term_names_).index(s["pair"])]
    c = c - c.mean()                                                   # the registered centring
    ia, _ = SS.pair_bins(X_all["campaign"].to_numpy(), 4)
    ib, _ = SS.pair_bins(X_all["emp_var_rate_sigmoid_slope"].to_numpy(), 4)
    for i, j in zip(*np.nonzero(s["support"])):
        rows = (ia == i) & (ib == j)
        assert s["support"][i, j] == rows.sum() and abs(s["surface"][i, j] - c[rows].mean()) < 1e-12
    assert np.isnan(s["surface"][s["support"] == 0]).all() and s["support"].sum() == len(X_all)
    assert np.nanmax(s["period_share"]) <= 1.0
    tab = SS.pair_table(pairs)
    assert len(tab) == len(pairs["surfaces"]) and (tab["cells_with_rows"] <= tab["cells"]).all()


def test_the_centring_check_is_free_of_the_centring_and_additive(world):
    fits, configs, frames, T = world["fits"], world["configs"], world["frames"], world["T"]
    sup = SS.support_tables(frames["X_all"], T)
    chk = SS.centring_check(fits, configs, frames, sup, T)
    st = chk["steps"]
    base = st[st["configuration"] == "baseline"]
    a, b = SS.CORRELATED_PAIR
    assert np.allclose(base["both slopes"], base[a] + base[b])            # the step is linear in the terms
    assert st.loc[st["configuration"] == "drop_second", b].isna().all()   # a removed feature has no step
    assert set(chk["centring"]["configuration"]) == {"baseline", "drop_second", "past_density"}
    cen = chk["centring"].set_index(["configuration", "feature"])
    assert (cen["recentring_shift_range"] >= 0).all() and (cen["fold_sd_recentred"] >= 0).all()
    for _, r in cen.iterrows():                                       # both statistics of the per-fold shifts
        assert r["recentring_shift_max_abs"] == max(abs(v) for v in r["recentring_shift_by_fold"])
        assert np.isclose(r["recentring_shift_range"], np.ptp(r["recentring_shift_by_fold"]))
    # re-centring subtracts one constant per fold curve, so the amplitude and step never depend on it
    m = fits["baseline"]["models"][0]
    X = frames["X_all"][list(configs["baseline"]["features"])]
    v = m.eval_terms(X)[:, list(m.term_names_).index(a)]
    year = T["year"].to_numpy()
    step = (v - v.mean())[year > 2008].mean() - (v - v.mean())[year == 2008].mean()
    assert abs(cen.loc[("baseline", a), "step_by_fold"][0] - step) < 1e-12
    assert len(chk["trade_off"]) == 2 and "baseline" in set(chk["trade_off"]["configuration"])


def _stated(monkeypatch, stated, **kw):
    monkeypatch.setattr(SS, "STATED", stated)
    base = dict(configs=None, corr=None, past=None, diff={"pooled": pd.DataFrame({"configuration": []}),
                "fold_differences": pd.DataFrame(columns=["configuration", "metric"])}, mtable=None,
                summary=pd.DataFrame(columns=["configuration", "feature"]), noms=pd.DataFrame(columns=["configuration"]))
    return SS.stated_checks(**{**base, **kw})


def test_the_stated_checks_cover_the_centring_check_and_the_pair_cells(world, monkeypatch):
    fits, configs, frames, T = world["fits"], world["configs"], world["frames"], world["T"]
    chk = SS.centring_check(fits, configs, frames, SS.support_tables(frames["X_all"], T), T)
    pairs = SS.pair_surfaces(fits, configs, frames, T, bins=4)
    sd = chk["step_spread"].set_index(["configuration", "statistic"]).loc[("baseline", "sd across folds"),
                                                                       "period-level terms"]
    pt = SS.pair_table(pairs).set_index(["configuration", "pair"]).loc[("baseline", "campaign & emp_var_rate_sigmoid_slope")]
    cells = tuple(int(pt[c]) for c in ("cells", "cells_with_rows", "cells_supported", "rows_in_sparse_cells"))
    p1 = ("baseline", "campaign & emp_var_rate_sigmoid_slope")
    out = _stated(monkeypatch, {
        "step_spread": {("baseline", "period-level terms"): (float(sd), 1e-9)},
        "pair_cells": {p1: cells, ("baseline", "prior_x_stress & decay_x_density"): (1, 1, 1, 1)},
        "pair_claims": {(*p1, "max_abs"): (">=", float(pt["max_abs"])),
                        (*p1, "range_supported"): ("<", 0.0)}},
        centring=chk, pairs=pairs)
    assert out["passed"].tolist() == [True, True, False, True, False]   # exact cells; claims; wrong ones fail


def test_period_level_membership_tolerates_floating_point_noise_only():
    # Rows of one day share their inputs, but a reduction can differ in the last bits by a row's position (the
    # accepted Windows run): exact nunique() then drops a period-level feature. Noise is tolerated; variation not.
    day = np.repeat(np.arange(40), 25)
    rng = np.random.default_rng(3)
    level = rng.uniform(0.0, 0.15, 40)[day]
    X = pd.DataFrame({"joint_economic_decay": level,
                      "decay_x_density": level * (1 + rng.choice([-1, 0, 1], len(day)) * 2.2e-16),
                      "age": rng.integers(18, 90, len(day)).astype(float),
                      "emp_var_rate_sigmoid_slope": level + rng.normal(0, 1e-6, len(day))})
    T = pd.DataFrame({"day": day})
    assert X.groupby(day)["decay_x_density"].nunique().max() > 1           # the exact rule would drop it
    wd = SS.within_day_spread(X, T).set_index("feature")
    assert wd.loc["decay_x_density", "period_level"] and not wd.loc["decay_x_density", "exactly_constant"]
    assert wd.loc["joint_economic_decay", "exactly_constant"]
    assert not wd.loc["age", "period_level"] and not wd.loc["emp_var_rate_sigmoid_slope", "period_level"]
    assert SS.period_level_features(X, T) == ["decay_x_density", "joint_economic_decay"]


def test_the_decay_slot_decomposes_the_period_level_step(world):
    fits, configs, frames, T = world["fits"], world["configs"], world["frames"], world["T"]
    chk = SS.centring_check(fits, configs, frames, SS.support_tables(frames["X_all"], T), T)
    st = chk["steps"]
    year = T["year"].to_numpy()
    for name, slot in (("baseline", "decay_x_density"), ("past_density", SS.PAST_FEATURE)):
        feats = list(configs[name]["features"])
        g = st[st["configuration"] == name]
        for k, m in enumerate(fits[name]["models"]):
            v = m.eval_terms(frames["X_all"][feats])[:, list(m.term_names_).index(slot)]
            v = v - v.mean()
            slot_step = v[year > 2008].mean() - v[year == 2008].mean()
            row = g[g["fold"] == k].iloc[0]
            assert abs(row["period-level terms"] - row["period-level terms except the decay slot"] - slot_step) < 1e-9
    assert "within_day_spread" in chk and chk["within_day_spread"]["period_level"].any()


def test_complementarity_statements_need_the_accepted_views_and_hold_exactly(monkeypatch):
    stated = {"complementarity": {"abstention_rows": 3, "abstention_positives": 1,
                                  "abstention_net": {"baseline": 2, "jed_alone": 1},
                                  "cascade": {"jed_alone": "nominated"}}}
    out = _stated(monkeypatch, stated, comp=None)
    assert len(out) == 1 and out["passed"].all() and "not checkable" in out["check"].iloc[0]
    table = pd.DataFrame({"ebm": ["baseline", "baseline", "jed_alone", "jed_alone"],
                          "population": ["all rows", "Stage 2 abstentions"] * 2,
                          "rows": [9, 3, 9, 3], "positives": [4, 1, 4, 1], "net": [5, 2, 4, 1]})
    noms = pd.DataFrame({"configuration": ["jed_alone"], "cascade": ["nominated"]})
    assert _stated(monkeypatch, stated, comp={"table": table}, noms=noms)["passed"].all()
    table.loc[3, "net"] = 0
    noms.loc[0, "cascade"] = "not nominated"
    out = _stated(monkeypatch, stated, comp={"table": table}, noms=noms)
    assert out["passed"].tolist() == [True, True, False, False]        # a changed count or verdict fails


# ======================================================================
# the past-only fallback
# ======================================================================
def test_later_fit_labels_move_an_earlier_rows_fallback_but_not_an_earlier_raw_value(monkeypatch):
    # The question: can a fit-side label of a contact after a cold-start row's day change that row's final
    # feature? Through the cleaner's fit-side median, yes; through the raw past-only density, never.
    monkeypatch.setattr(SS, "MIN_HISTORY_ROWS", 50)
    monkeypatch.setattr(SS, "MIN_HISTORY_POSITIVES", 3)
    macro, y, day = _ctx()
    rng = np.random.default_rng(1)
    fit = rng.random(len(day)) < 0.7
    val = ~fit
    jed = rng.uniform(0.01, 1.0, len(day))

    def final(labels):
        a = SS.past_density(macro[fit], labels, day[fit], macro[fit], day[fit])["value"]
        b = SS.past_density(macro[fit], labels, day[fit], macro[val], day[val])["value"]
        _, cb, med = SS._clean_pair(jed[fit] * a, jed[val] * b)
        return b, cb, med

    raw, col, med = final(y[fit])
    fallback = np.isnan(raw)
    assert fallback.any() and (~fallback).any()
    assert np.all(col[fallback] == med)                               # the EBM column of a fallback row
    first_day = day[val][~fallback].min()
    assert first_day > day[val][fallback].max()                       # the fallback rows are the earliest days
    flipped = y[fit].copy()
    later = day[fit] >= first_day                                     # contacts after every fallback row's day
    flipped[later] = 1 - flipped[later]
    raw2, col2, med2 = final(flipped)
    assert med2 != med and np.all(col2[fallback] == med2)             # the earlier rows' fallback moves
    first = day[val] == first_day                                     # its raw values use earlier days only
    assert np.array_equal(raw2[first], raw[first]) and np.array_equal(col2[first], col[first])
    assert np.array_equal(np.isnan(raw2), fallback)                   # which rows fall back does not change


def test_the_fallback_trace_follows_both_forms_to_the_ebm_frames(world, monkeypatch):
    tr = SS.fallback_trace(world["block"], world["X"], world["y"], world["T"], world["past"])
    chk = SS.fallback_checks(tr, world["past"])
    assert chk["passed"].all(), chk.to_string()
    t, pf = tr["totals"], tr["per_fold"]
    assert t["held_out_fallback"] > 0 and t["held_out_raw"] + t["held_out_fallback"] == len(world["y"])
    assert (pf["ebm_column_is_fallback"] & pf["ebm_column_is_raw_product"]).all()
    sv = tr["sensitivity"]
    assert set(sv["perturbation"]) == {"10% of later fit labels", "every later fit label"}
    assert sv[sv["perturbation"] == "every later fit label"]["fallback_moved"].any()
    assert sv["control_raw_values_unchanged"].all()
    te = SS.fallback_term_effect(world["fits"]["past_density"], world["configs"]["past_density"],
                                 world["frames"], tr)
    assert len(te) == len(sv)
    assert np.allclose(te["term_score_change"], te["term_score_after"] - te["term_score_before"])
    biggest = float(te.loc[te["perturbation"] == "every later fit label", "term_score_change"].abs().max())
    monkeypatch.setattr(SS, "STATED", {
        "fallback": {"held_out_fallback": t["held_out_fallback"], "held_out_raw": t["held_out_raw"] + 1,
                     "control_unchanged": True},
        "fallback_term_change": {("every later fit label", "max"): (biggest, 1e-9)}})
    stated = SS.stated_checks(configs=world["configs"], corr=None, past=None,
                              diff={"pooled": pd.DataFrame({"configuration": []}),
                                    "fold_differences": pd.DataFrame(columns=["configuration", "metric"])},
                              mtable=None, summary=pd.DataFrame(columns=["configuration", "feature"]),
                              noms=pd.DataFrame(columns=["configuration"]), fallback={**tr, "term_effect": te})
    assert stated["passed"].tolist() == [True, False, True, True]     # a wrong count fails


# ======================================================================
# the accepted views
# ======================================================================
VIEW_FILES = {"stage1.lr": "models/x/lr.joblib", "stage2.glass_router": "models/x/router.joblib",
              "stage3.ebm": "models/x/ebm.joblib"}


def _views_root(tmp_path, contents=None):
    from feature_research.frozen_references import PINS_FILE
    from shared.pinned_artifacts import PinnedFile, sha256_bytes, write_pins
    pins = {}
    for role, rel in VIEW_FILES.items():
        data = (contents or {}).get(role, f"artifact {role}".encode())
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
        pins[role] = PinnedFile(role, rel, sha256_bytes(data), len(data))
    write_pins(tmp_path / PINS_FILE, "c4_20261113_000000", pins, root=tmp_path)
    return tmp_path


def _fake_views(monkeypatch, y):
    from types import SimpleNamespace as NS
    from shared.stage2.diagnostics import boundary, operating_points
    lr = lambda *a, **k: NS(y=y)  # noqa: E731
    router = lambda *a, **k: NS(y=y, abstain=np.arange(len(y)) % 3 == 0)  # noqa: E731
    ebm = lambda *a, **k: NS(y=y, scores=pd.DataFrame({"raw_model_score": np.linspace(0, 1, len(y)),  # noqa: E731
                                                       "calibrated_probability": np.linspace(0, 1, len(y))}))
    monkeypatch.setattr(boundary, "extract_lr_view", lr)
    monkeypatch.setattr(boundary, "extract_router_view", router)
    monkeypatch.setattr(boundary, "extract_ebm_view", ebm)
    monkeypatch.setattr(operating_points, "resolve_operating_point",
                        lambda view, name: NS(flags=np.asarray(view.y) == 1, threshold=0.5, score_space="raw",
                                              field="f", selection="s", disclosure="d"))


def test_missing_artifacts_are_the_sandbox_path_and_can_be_required(tmp_path):
    root = _views_root(tmp_path)
    for rel in VIEW_FILES.values():
        (root / rel).unlink()
    v = SS.accepted_views(None, root=root)
    assert set(v["status"].values()) == {"not available locally"} and not v["available"] and not v["required"]
    with pytest.raises(SS.AcceptedViewError, match="required .*PR42_REQUIRE_ACCEPTED_VIEWS=1.* stage1.lr, "
                                                   "stage2.glass_router, stage3.ebm"):
        SS.accepted_views(None, root=root, require=True)


def test_a_present_artifact_that_fails_its_pin_raises_with_its_role(tmp_path):
    root = _views_root(tmp_path)
    (root / VIEW_FILES["stage2.glass_router"]).write_bytes(b"changed after pinning")
    with pytest.raises(SS.AcceptedViewError, match=r"^stage1\.lr: .*boundary check"):
        SS.accepted_views(None, root=root)                            # roles are read in order: LR first
    (root / VIEW_FILES["stage1.lr"]).unlink()
    with pytest.raises(SS.AcceptedViewError, match=r"^stage2\.glass_router: .*fails its pin check: PinError"):
        SS.accepted_views(None, root=root)                            # missing LR is reported, router raises


def test_a_present_artifact_that_fails_its_boundary_check_raises_with_its_role(tmp_path, monkeypatch):
    root = _views_root(tmp_path)                                      # pinned bytes that are not an artifact
    with pytest.raises(SS.AcceptedViewError, match=r"^stage1\.lr: models/x/lr\.joblib passes its pin but fails "
                                                   r"its boundary check: \w+"):
        SS.accepted_views(None, root=root)
    y = np.array([0, 1, 1, 0, 1, 0])
    _fake_views(monkeypatch, y)
    with pytest.raises(SS.AcceptedViewError, match=r"^stage1\.lr: .*not the 5 training rows' labels"):
        SS.accepted_views(None, root=root, y=y[:5])
    with pytest.raises(SS.AcceptedViewError, match=r"^stage1\.lr: .*not the 6 training rows' labels"):
        SS.accepted_views(None, root=root, y=1 - y)


def test_three_passing_views_satisfy_the_requirement_and_cell57s_checks(tmp_path, monkeypatch):
    root = _views_root(tmp_path)
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0])
    _fake_views(monkeypatch, y)
    v = SS.accepted_views(None, root=root, require=True, y=y)
    assert v["available"] and v["required"] and set(v["status"].values()) == {"passed"}
    assert {"lr", "router", "ebm"} <= set(v) and v["router"]["abstain"].sum() == 3
    dec = {"accepted": v["ebm"]["decision"], "baseline": y, "jed_alone": 1 - y}
    comp = SS.complementarity(y, v["lr"]["flags"], v["router"]["abstain"], dec)
    chk = SS.complementarity_checks(comp, len(y))
    req = SS.complementarity_requirement(v, comp, chk, list(dec))
    assert chk["passed"].all() and req["passed"].all(), req.to_string()
    assert not SS.complementarity_requirement(v, None, None, list(dec))["passed"].all()
    partial = {"table": comp["table"][comp["table"]["ebm"] != "jed_alone"], "against_reference": None}
    assert not SS.complementarity_requirement(v, partial, chk, list(dec))["passed"].all()
    sandbox = {"status": {r: "not available locally" for r in VIEW_FILES}, "required": False, "available": False}
    assert SS.complementarity_requirement(sandbox, None, None, list(dec))["passed"].all()


def test_the_views_requirement_is_an_explicit_switch():
    assert SS.views_required({}) is False and SS.views_required({SS.REQUIRE_VIEWS_ENV: "0"}) is False
    assert SS.views_required({SS.REQUIRE_VIEWS_ENV: "1"}) is True
    with pytest.raises(ValueError, match="must be 1"):
        SS.views_required({SS.REQUIRE_VIEWS_ENV: "yes"})


# ======================================================================
# complementarity and nomination
# ======================================================================
def test_the_four_counts_reconcile_and_the_net_is_against_the_baseline():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    s1 = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    base = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    cand = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    ab = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    out = SS.complementarity(y, s1, ab, {"baseline": base, "cand": cand})
    t = out["table"].set_index(["ebm", "population"])
    r = t.loc[("baseline", "Stage 2 abstentions")]
    assert (r["both_correct"], r["stage1_only_correct"], r["ebm_only_correct"], r["both_wrong"]) == (2, 0, 1, 1)
    assert t["reconciled"].all() and (t.loc[[("cand", "all rows")], "rows"] == 8).all()
    a = out["against_reference"].set_index(["ebm", "population"]).loc[("cand", "all rows")]
    assert a["corrects_stage1_errors"] == 4 and a["overturns_correct_stage1"] == 2
    assert a["net_change"] == a["net"] - a["reference_net"]
    assert SS.complementarity_checks(out, 8)["passed"].all()
    broken = {"table": out["table"].assign(reconciled=False), "against_reference": out["against_reference"]}
    assert not SS.complementarity_checks(broken, 8)["passed"].all()


def test_the_nomination_rules(monkeypatch):
    pooled = pd.DataFrame([
        {"configuration": "baseline", "d_roc_auc": None, "d_roc_auc_low": None, "d_pr_auc": 0.0, "d_nested_f2": 0.0,
         "ece_calibrated": 0.01, "brier_calibrated": 0.080},
        {"configuration": "a", "d_roc_auc": -0.002, "d_roc_auc_low": -0.006, "d_pr_auc": -0.004, "d_nested_f2": 0.0,
         "ece_calibrated": 0.01, "brier_calibrated": 0.081},
        {"configuration": "b", "d_roc_auc": -0.002, "d_roc_auc_low": -0.012, "d_pr_auc": 0.0, "d_nested_f2": 0.0,
         "ece_calibrated": 0.01, "brier_calibrated": 0.080},
    ])
    monkeypatch.setattr(SS, "INTERPRETATION", {"a": {"gain": True, "rationale": "r"}, "b": {"gain": True}})
    n = SS.nominations({"pooled": pooled}).set_index("configuration")
    assert n.loc["a", "step5b"] == "recommended" and n.loc["b", "step5b"] == "not recommended"
    assert not n.loc["b", "d_roc_auc_lower_ok"]
    monkeypatch.setattr(SS, "INTERPRETATION", {"a": {"gain": False}})
    assert SS.nominations({"pooled": pooled}).set_index("configuration").loc["a", "step5b"] == "not recommended"
    comp = {"against_reference": pd.DataFrame([{"ebm": "a", "population": "Stage 2 abstentions", "rows": 1000,
                                                "net": 10, "reference_net": 14}])}
    monkeypatch.setattr(SS, "INTERPRETATION", {"a": {"gain": True}})
    got = SS.nominations({"pooled": pooled}, comp).set_index("configuration").loc["a"]
    assert got["cascade"] == "nominated" and got["abstention_net_needed"] == 9.0


def test_figures_reach_the_notebook_as_png_under_the_agg_backend(monkeypatch):
    import matplotlib
    import matplotlib.pyplot as plt
    import IPython.display

    matplotlib.use("Agg")                       # what importing glass_arbiter.tracer selects
    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    SS.show(fig)
    assert len(shown) == 1 and shown[0].format == "png" and shown[0].data[:8] == b"\x89PNG\r\n\x1a\n"
    assert not plt.fignum_exists(fig.number)    # closed after display


# ======================================================================
# the notebook's Step 5A cells
# ======================================================================
def test_the_notebook_registers_step5a_first_and_writes_its_record_once():
    nb = json.loads((BANK / "feature_research" / "feature_research.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    headers = [next((ln for ln in s.splitlines() if ln.startswith("# CELL ")), "") for s in code]
    idx = {h.split(":")[0].replace("# CELL ", ""): i for i, h in enumerate(headers)}
    assert all(str(c) in idx for c in range(49, 60))
    assert idx["48"] < idx["49"] < idx["59"] < len(code)                 # later steps follow (5B: Cells 60-68)
    assert "SS.registration_checks()" in code[idx["49"]] and "SS.require(" in code[idx["49"]]
    first = min(i for i, s in enumerate(code) if "SS.fit_all(" in s or "SS.fit_configuration(" in s)
    assert first > idx["49"]
    views_at = min(i for i, s in enumerate(code) if "SS.accepted_views(" in s)
    assert views_at == idx["49"] < first                              # the views are checked before any fit
    joined = "\n".join(code)
    assert joined.count('"stage3_simplification"') == 3              # seed name, record written, record read
    assert code[idx["59"]].count('RR.write_run_record(RUN, "stage3_simplification"') == 1
    assert 'RR.read_run_record(RUN.run_id, "stage3_simplification")' in code[idx["59"]]
    assert "VIEWS_5A = SS.accepted_views(TRAIN.train_reference, require=REQUIRE_VIEWS_5A" in code[idx["49"]]
    assert "REQUIRE_VIEWS_5A = SS.views_required()" in code[idx["49"]]
    assert joined.count("SS.accepted_views(") == 1                    # read once, before any fit (above)
    assert "SS.fallback_trace(" in code[idx["50"]] and "SS.require(FALLBACK_CHECKS_5A" in code[idx["50"]]
    assert "SS.fallback_term_effect(" in code[idx["52"]]
    assert "SS.complementarity_requirement(" in code[idx["57"]] and "SS.require(COMP_REQUIRED_5A" in code[idx["57"]]
    assert "from feature_research import stage3_simplification as SS" in code[0]
    assert "RUN = RR.start_run(commit=5, seeds=SEEDS)" in code[idx["3"]]        # Commit 5 runs are c5_...
    for cell in ("54", "55", "56"):                                      # every figure is written as a PNG
        assert "SS.show(" in code[idx[cell]] and "display(" not in code[idx[cell]].replace("SS.show(", "")
