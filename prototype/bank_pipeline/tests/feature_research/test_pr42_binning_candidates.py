"""
PR 42 Commit 4, Step 4B: ``feature_research.binning_candidates`` on synthetic
data (the real training rows: ``tests/realdata/test_pr42_binning_candidates.py``).

Covered: no new importer of the Stage 2 feature code; the registration (pinned
digest, the 4A facts it separates from what it fixes, its checks against the
PR 39 record, the GLASS profile and the shared contract); fit and apply (state
blocks and ties, midpoint cuts, (lo, hi] intervals, the frozen cuts through
both paths); each procedure's registered rule, including empty windows, the
tail range and infeasible fits; fold isolation (held-out label flips pass for
the registered path and fail for one that reads every label); the comparator's
parity with E-B; partner counts, eligibility, ranking and overlap; the filter
partition and count reconciliation; the decision rules; and the notebook's
Step 4B cells by source.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_research import binning_candidates as BC
from feature_research import binning_drift as BD
from shared.stage2.binning import BINNING_STRATEGY, RF_FEATURES_BINARY
from stage2_binning import evidence as E
from stage2_binning import runner as R
from tests.support import stage2_crossfit_world as W

BANK = Path(BC.__file__).resolve().parents[1]
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


@pytest.mark.parametrize("path", [BANK / "feature_research" / "binning_candidates.py", Path(__file__)])
def test_step4b_code_imports_no_feature_engineering_module(path):
    mods = _imported_modules(path)
    assert not [m for m in mods for t in FEATURE_MODULES if m == t or m.startswith(t + ".")]


def test_the_commit2_caller_scan_does_not_list_step4b():
    from feature_research import feature_inventory as FI
    assert "binning_candidates" not in json.dumps(FI.duplicate_callers())


# ======================================================================
# the registration
# ======================================================================
def test_the_registration_is_the_pinned_version_and_any_change_moves_its_digest():
    assert BC.registration_digest() == BC.REGISTERED_SHA256
    assert BC.REGISTRATION["version"] == "1.0.0"
    assert BC.REGISTRATION["revisions"] == [{"version": "1.0.0",
                                             "status": "fixed before any candidate result was computed"}]
    changed = copy.deepcopy(BC.REGISTRATION)
    changed["criteria"]["A2_coverage"] += " "
    assert BC.registration_digest(changed) != BC.REGISTERED_SHA256


def test_the_registration_separates_what_4a_established_from_what_it_fixes():
    known = BC.REGISTRATION["known_before_registration"]
    for key in ("eci_warm", "eci_hot", "nsd_elevated", "both", "motivated", "not_computed_before_registration"):
        assert known[key]
    assert set(known["motivated"]) == {"share_matched", "gap_placed", "tail_selected", "eci_warm_filter"}
    assert "0.92x" in known["eci_warm"] and "1.031x" in known["eci_warm"]
    assert "1,456 rows" in known["eci_hot"] and "653 rows" in known["nsd_elevated"]
    for fam, procs in BC.CANDIDATES.items():
        for p in procs:
            assert fam in BC.REGISTRATION["procedures"][p]["families"]
    for key in ("A0_feasible", "A1_triggers", "A2_coverage", "A3_NSD", "A3_ECI", "A4_robust",
                "filter_slot", "filter_verdict"):
        assert BC.REGISTRATION["criteria"][key]
    for key in ("ties", "sparse_or_empty", "infeasible", "lift_versus_support"):
        assert BC.REGISTRATION["handling"][key]
    assert "not compliance" in BC.REGISTRATION["population"]


def test_the_registered_shares_windows_and_tail_range():
    assert BC.frozen_shares("NSD") == (0.67132, 0.878877, 0.91912)
    assert BC.frozen_shares("ECI") == (0.22434, 0.499848, 0.878938)
    for fam in BC.FAMILIES:
        share = np.asarray(BC.FROZEN_SUPPORTS[fam]) / BC.TRAINING_ROWS
        h, q = BC.window_halfwidths(fam), BC.frozen_shares(fam)
        for j in range(3):                          # windows never meet: each bin keeps half its share
            assert q[j] + h[j] <= (q[j + 1] - h[j + 1] if j < 2 else 1.0)
            assert h[j] <= 0.25 * share[j] + 1e-6 and h[j] <= 0.25 * share[j + 1] + 1e-6
    assert BC._tail_range() == [0.060531, 0.242124]
    assert BC.PARTNER_FIT_SUPPORT_MIN == 300 and BC.VARIANT_FIT_SUPPORT_MIN == 100
    assert len(BC.partner_pool()) == 288


def test_the_registration_checks_pass_on_the_accepted_inputs_and_catch_a_changed_record():
    record, _ = BD.read_pr39_record()
    checks = BC.registration_checks(record)
    assert checks["passed"].all(), checks.to_string()
    changed = copy.deepcopy(record)
    changed["eb"]["families"]["NSD"]["in_sample"]["bins"]["nsd_hot"]["pooled"]["support"] += 1
    assert not BC.registration_checks(changed)["passed"].all()


# ======================================================================
# fit and apply
# ======================================================================
def _states(values, rows, positives=None):
    positives = [0] * len(values) if positives is None else positives
    return pd.DataFrame({"value": values, "rows": rows, "positives": positives},
                        index=pd.Index(range(len(values)), name="state"))


def test_state_rows_keep_one_value_per_state_and_refuse_a_split_state():
    codes = np.array([0, 0, 1, 1, 1, 2])
    exact = BC.state_rows(codes, np.array([.1, .1, .3, .3, .3, .2]), np.array([1, 0, 0, 0, 1, 1]), exact=True)
    assert exact["rows"].tolist() == [2, 3, 1] and exact["positives"].tolist() == [1, 1, 1]
    assert exact["value"].tolist() == [.1, .3, .2]
    with pytest.raises(ValueError):
        BC.state_rows(codes, np.array([.1, .2, .3, .3, .3, .2]), np.zeros(6), exact=True)
    mean = BC.state_rows(codes, np.array([.1, .2, .3, .3, .6, .2]), np.zeros(6), exact=False)
    assert np.allclose(mean["value"], [.15, .4, .2])


def test_tied_states_form_one_block_and_cuts_lie_between_blocks():
    st = _states([0.1, 0.1 + 1e-13, 0.3, 0.2, 0.5], [5, 5, 10, 20, 60])
    blk = BC.blocks(st)
    assert blk["states"].tolist() == [2, 1, 1, 1] and blk["rows"].tolist() == [10, 20, 10, 60]
    bnd = BC.boundaries(blk)
    assert np.allclose(bnd["cut"], [0.15, 0.25, 0.4]) and np.allclose(bnd["share_below"], [.1, .3, .4])
    assert not np.isin(bnd["cut"], st["value"]).any()


def test_share_matched_takes_the_nearest_share_and_the_lower_boundary_on_a_tie(monkeypatch):
    monkeypatch.setattr(BC, "frozen_shares", lambda fam: (0.2, 0.5, 0.75))
    blk = BC.blocks(_states([1, 2, 3, 4, 5, 6], [10, 20, 20, 20, 20, 10]))   # shares .1 .3 .5 .7 .9
    fit = BC.fit_share_matched("NSD", blk)
    assert fit["feasible"]
    assert [c["boundary"] for c in fit["per_cut"]] == [0, 2, 3]          # 0.2 ties .1/.3: the lower
    assert fit["cuts"] == [1.5, 3.5, 4.5] and fit["bin_rows"] == [10, 40, 20, 30]
    assert fit["bin_states"] == [1, 2, 1, 2]


def test_gap_placed_takes_the_widest_gap_in_the_window_and_falls_back_when_it_is_empty(monkeypatch):
    monkeypatch.setattr(BC, "frozen_shares", lambda fam: (0.3, 0.6, 0.95))
    monkeypatch.setattr(BC, "window_halfwidths", lambda fam: (0.12, 0.01, 0.02))
    values = [1.0, 2.0, 2.1, 4.0, 4.2, 5.0, 9.0]
    blk = BC.blocks(_states(values, [20, 10, 10, 10, 10, 30, 10]))            # shares .2 .3 .4 .5 .6 .9
    fit = BC.fit_gap_placed("NSD", blk)
    how = [c["chosen_by"] for c in fit["per_cut"]]
    assert [c["boundary"] for c in fit["per_cut"]] == [2, 4, 5]
    assert how == ["widest gap in window", "widest gap in window", "nearest share (empty window)"]
    assert np.isclose(fit["cuts"][0], 3.05)


def test_equal_gaps_take_the_boundary_nearest_the_target(monkeypatch):
    monkeypatch.setattr(BC, "frozen_shares", lambda fam: (0.35, 0.7, 0.9))
    monkeypatch.setattr(BC, "window_halfwidths", lambda fam: (0.2, 0.01, 0.01))
    blk = BC.blocks(_states([1, 2, 3, 4, 5, 6, 7], [10, 10, 10, 10, 30, 20, 10]))  # equal gaps
    fit = BC.fit_gap_placed("NSD", blk)
    assert fit["per_cut"][0]["boundary"] == 2                    # share .3 is nearest .35 (.4 next)


def test_too_few_states_make_an_infeasible_fit_that_cannot_be_applied():
    blk = BC.blocks(_states([0.01, 0.02, 0.5], [50, 30, 20]))
    for fit in (BC.fit_share_matched("NSD", blk), BC.fit_gap_placed("NSD", blk)):
        assert not fit["feasible"] and fit["cuts"] is None and "no fit-side state" in fit["reason"]
        with pytest.raises(ValueError):
            BC.apply_fit([0.1], fit)


def test_the_frozen_cuts_bin_alike_through_both_paths_including_the_cut_values():
    for fam in BC.FAMILIES:
        cuts = BC._frozen_cuts(fam)
        v = np.r_[cuts, np.nextafter(cuts, np.inf), np.linspace(0, 0.4, 201)]
        frozen = BC.fit_frozen(fam, BC.blocks(_states([0.0, 1.0], [1, 1])))
        names = E.family_bins(fam)
        assert np.array_equal(BC.apply_fit(v, frozen), BD.assign_intervals(v, cuts, names))
        assert cuts == [hi for _, _, hi in BINNING_STRATEGY[fam]["bins"][:3]]
        assert BC.apply_fit(cuts, frozen).tolist() == names[:3]          # (lo, hi]: a cut is its lower bin


def test_tail_selected_maximizes_the_wilson_lower_bound_in_range_and_maps_it_by_share(monkeypatch):
    monkeypatch.setattr(BC, "frozen_shares", lambda fam: (0.2, 0.4, 0.85))
    monkeypatch.setattr(BC, "_tail_range", lambda: [0.05, 0.30])
    rows = [200, 200, 300, 100, 100, 50, 50]                                     # 1,000 rows
    sel = BC.blocks(_states([1, 2, 3, 4, 5, 6, 7], rows, [10, 10, 30, 10, 80, 40, 30]))
    # tails (share, rate): top1 .05 .60; top2 .10 .70; top3 .20 .75; top4 .30 .53
    fit = BC.fit_tail_selected("ECI", BC.blocks(_states([1, 2, 3, 4, 5, 6, 7], rows)), sel)
    chosen = fit["selection"]["chosen"]
    assert fit["feasible"] and chosen["tail_rows"] == 200 and chosen["tail_positives"] == 150
    lo = [BC._wilson_lower(p, n) for p, n in ((30, 50), (70, 100), (150, 200), (160, 300))]
    assert chosen["wilson_lower"] == pytest.approx(max(lo)) and max(lo) == lo[2]
    assert [c["boundary"] for c in fit["per_cut"]] == [0, 1, 3]
    assert fit["selection"]["scoring_tail_share"] == pytest.approx(0.2)
    scoring = BC.blocks(_states([1, 2, 3, 4, 5, 6, 7], [200, 200, 300, 150, 50, 50, 50]))
    mapped = BC.fit_tail_selected("ECI", scoring, sel)       # scoring tails .3 .15 .1: .15 is nearest .2
    assert mapped["per_cut"][2]["boundary"] == 3 and mapped["selection"]["scoring_tail_share"] == pytest.approx(.15)
    narrower = BC.fit_tail_selected("ECI", BC.blocks(_states([1, 2, 3, 4, 5, 6, 7], rows,
                                                             [10, 10, 30, 10, 60, 40, 30])), sel)
    assert narrower["selection"]["chosen"]["tail_rows"] == 200       # the selection uses sel, not scoring labels


def test_tail_selected_is_infeasible_without_a_tail_in_the_registered_range(monkeypatch):
    monkeypatch.setattr(BC, "_tail_range", lambda: [0.05, 0.30])
    blk = BC.blocks(_states([1, 2, 3, 4], [10, 10, 10, 970]))
    fit = BC.fit_tail_selected("ECI", blk, blk)
    assert not fit["feasible"] and "registered share range" in fit["reason"]
    with pytest.raises(ValueError):
        BC.fit_tail_selected("NSD", blk, blk)


# ======================================================================
# the synthetic world
# ======================================================================
@pytest.fixture(scope="module")
def world():
    from feature_research import feature_inventory as FI
    ts = W.synthetic_train_side()
    side = BD.stage2_side(ts.X_train, ts.y_train)
    eb = BD._canonical(R.eb_evidence(side))
    sources = BD.source_values(side, FI.build_training_frames(side.X_train, side.y_train)["stage2_sources"], eb)
    n = len(side.y_train)
    T = pd.DataFrame({"position": np.arange(n), "period": np.where(np.arange(n) < n // 2, "2008-05", "2008-06"),
                      "period_index": (np.arange(n) >= n // 2).astype(int)}, index=side.X_train.index)
    states = BD.macro_states(side.X_train, T, side.y_train)
    svals = BD.state_values(states, sources, side, BD.fold_fit_values(side, states, eb))
    fold_fits = BC.fit_folds(side, states, svals)
    deployed = BC.deployment_fits(side, states, svals, sources)
    comparison = BC.compare(side, sources, fold_fits, deployed, eb)
    return dict(side=side, eb=eb, sources=sources, T=T, states=states, svals=svals, fold_fits=fold_fits,
                deployed=deployed, comparison=comparison)


def test_each_fold_is_fitted_on_its_fit_rows_only(world):
    for ff in world["fold_fits"]:
        assert ff["checks"]["passed"].all(), ff["checks"].to_string()
        assert len(ff["inner_values"]["NSD"]) == len(world["side"].folds[ff["fold"]][0])
        assert sum(p["heldout_rows"] for p in ff["inner_provenance"]) == len(world["side"].folds[ff["fold"]][0])


def test_held_out_label_flips_leave_every_fold_fit_unchanged(world):
    checks = BC.isolation_checks(world["side"], world["states"], world["svals"], world["fold_fits"], folds=[0, 3])
    assert checks["passed"].all(), checks.to_string()
    assert len(checks) == 4


def test_held_out_label_flips_catch_a_fit_that_reads_every_label(world, monkeypatch):
    from feature_research import feature_inventory as FI
    real = BC.inner_sources

    def leaky(side, k, oof_fn=None):                        # in-sample values from all rows and labels
        src, prov, checks = real(side, k, oof_fn=oof_fn)
        every = FI.build_training_frames(side.X_train, side.y_train)["stage2_sources"]
        return every.loc[src.index], prov, checks

    monkeypatch.setattr(BC, "inner_sources", leaky)
    checks = BC.isolation_checks(world["side"], world["states"], world["svals"], world["fold_fits"], folds=[0])
    flips = checks[checks["check"].str.contains("held-out labels")]
    assert not flips["passed"].any()


def test_the_comparator_is_e_b_and_a_changed_record_fails_parity(world):
    checks = world["comparison"]["checks"]
    assert checks["passed"].all(), checks.to_string()
    changed = copy.deepcopy(world["eb"])
    changed["families"]["NSD"]["oof"]["bins"]["nsd_hot"]["pooled"]["positives"] += 1
    again = BC.compare(world["side"], world["sources"], world["fold_fits"], world["deployed"], changed)["checks"]
    assert not again.set_index("check")["passed"]["NSD: the comparator's table is E-B's out-of-fold table"]


def test_every_held_out_row_is_scored_with_its_own_folds_cuts(world):
    side, sources = world["side"], world["sources"]
    lab = world["comparison"]["labels"][("NSD", "share_matched")]
    for ff in world["fold_fits"]:
        va = side.folds[ff["fold"]][1]
        cuts = ff["fits"]["NSD"]["share_matched"]["cuts"]
        want = BD.assign_intervals(sources.values["oof"]["NSD"][va], cuts, E.family_bins("NSD"))
        assert np.array_equal(lab[va], want)


def test_candidate_tables_reconcile_with_their_folds(world):
    for (fam, proc), t in world["comparison"]["tables"].items():
        for b, e in t["bins"].items():
            assert e["pooled"]["support"] == sum(f["support"] for f in e["folds"])
            assert e["pooled"]["positives"] == sum(f["positives"] for f in e["folds"])
        assert sum(e["pooled"]["support"] for e in t["bins"].values()) == len(world["side"].y_train)


def test_migration_and_dependence_rows_are_consistent(world):
    cmp_ = world["comparison"]
    mig = cmp_["migration"].set_index(["family", "procedure"])
    for (fam, proc), lab in cmp_["labels"].items():
        dep = cmp_["deploy_labels"][(fam, proc)]
        f = BC.FOCAL[fam]
        assert mig.loc[(fam, proc), "migration_rows"] == int(((lab == f) != (dep == f)).sum())
    d = BC.dependence(world["side"], world["T"], cmp_)["removals"]
    assert set(d["removed"]) == {f"fold {k}" for k in range(5)} | {"period 2008-05", "period 2008-06"}
    r = d[(d["family"] == "NSD") & (d["procedure"] == "frozen") & (d["removed"] == "fold 2")].iloc[0]
    lab, y, fid = cmp_["labels"][("NSD", "frozen")], world["side"].y_train.to_numpy(), world["side"].fold_id
    keep = fid != 2
    assert r["support"] == int(((lab == "nsd_elevated") & keep).sum())
    assert r["lift"] == pytest.approx(y[keep & (lab == "nsd_elevated")].mean() / y[keep].mean())


# ======================================================================
# partners and the filter
# ======================================================================
def _bins(n, seed):
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(0, index=range(n), columns=RF_FEATURES_BINARY, dtype=np.int64)
    for fam in BINNING_STRATEGY:
        names = [b[0] for b in BINNING_STRATEGY[fam]["bins"]] if "bins" in BINNING_STRATEGY[fam] else \
            [BINNING_STRATEGY[fam]["passthrough"]]
        if len(names) == 1:
            out[names[0]] = rng.integers(0, 2, n)
        else:
            pick = rng.integers(0, len(names), n)
            for i, b in enumerate(names):
                out[b] = (pick == i).astype(int)
    return out


def test_partner_counts_are_the_brute_force_counts():
    bins, y = _bins(3000, 1), np.random.default_rng(2).integers(0, 2, 3000)
    st = BC.partner_stats(bins, y).set_index("partner")
    for lits in [("nsd_hot",), ("jed_warm", "cpi_cellular"), ("cci_low", "behav_hot"), ("dow_mid", "campaign_fresh")]:
        m = np.all(bins[list(lits)].to_numpy() == 1, axis=1)
        w = bins["eci_warm"].to_numpy() == 1
        r = st.loc[" & ".join(lits)]
        assert (r["support"], r["positives"]) == (m.sum(), y[m].sum())
        assert (r["warm_support"], r["warm_positives"]) == ((m & w).sum(), y[m & w].sum())
        assert r["cold_support"] == (m & ~w).sum()
    assert not [p for p in st.index if "eci_" in p]
    assert not [p for p in st.index if " & " in p and len({x.split("_")[0] for x in p.split(" & ")}) < 2]


def test_partner_selection_applies_the_gates_ranking_and_overlap():
    n = 4000
    bins = _bins(n, 3)
    rng = np.random.default_rng(4)
    y = (rng.random(n) < 0.05).astype(int)
    strong = (bins["nsd_hot"] == 1).to_numpy()
    y[strong] = (rng.random(strong.sum()) < 0.6).astype(int)               # nsd_hot rules are precise
    st, chosen = BC.select_partners(bins, y)
    el = st[st["eligible"]]
    assert (el["precision"] >= 0.2).all() and (el["support"] >= 300).all()
    assert (el["warm_support"] >= 100).all() and (el["cold_support"] >= 100).all()
    assert el["recall"].between(0.01, 0.35).all()
    assert chosen and chosen[0]["rank"] == 1 and len(chosen) <= BC.PARTNER_K
    assert [c["slot"] for c in chosen] == list(range(1, len(chosen) + 1))
    masks = [np.all(bins[c["literals"]].to_numpy() == 1, axis=1) for c in chosen]
    for i in range(len(masks)):
        for j in range(i):
            assert BC._jaccard(masks[i], masks[j]) <= 0.5
    ranked = el.sort_values("rank")
    assert ranked["wilson_lower"].is_monotonic_decreasing


def test_the_filter_variants_partition_each_partner_and_count_what_they_remove():
    n = 500
    bins = _bins(n, 5)
    y = np.random.default_rng(6).integers(0, 2, n)
    side = BD.Stage2Side(bins, pd.Series(y), ((np.arange(250, 500), np.arange(250)),
                                              (np.arange(250), np.arange(250, 500))),
                         np.r_[np.zeros(250, int), np.ones(250, int)], "two folds")
    partners = [{"slot": 1, "partner": "nsd_hot", "literals": ["nsd_hot"]},
                {"slot": 2, "partner": "jed_cold & cpi_cellular", "literals": ["jed_cold", "cpi_cellular"]}]
    fold_fits = [{"fold": 0, "partners": partners}, {"fold": 1, "partners": partners[:1]}]
    out = BC.warm_filter(side, {"bins": bins}, fold_fits)
    assert out["checks"]["passed"].all() and len(out["checks"]) == 3
    per = out["per_fold"]
    r = per[(per["fold"] == 0) & (per["slot"] == 2)].iloc[0]
    m = (bins["jed_cold"] == 1) & (bins["cpi_cellular"] == 1)
    held, w = np.arange(n) < 250, bins["eci_warm"] == 1
    assert r["alone_rows"] == (m & held).sum() and r["required_rows"] == (m & held & w).sum()
    assert r["required_removes_subscribers"] == y[m & held & ~w].sum()
    assert r["required_removes_non_subscribers"] == (m & held & ~w).sum() - y[m & held & ~w].sum()
    assert r["excluded_removes_subscribers"] == r["required_subscribers"]
    for v in ("rows", "subscribers", "non_subscribers"):
        assert (per[f"required_{v}"] + per[f"excluded_{v}"] == per[f"alone_{v}"]).all()
    slot1 = out["by_slot"].set_index("slot").loc[1]
    assert slot1["folds"] == 2 and slot1["alone_rows"] == per[per["slot"] == 1]["alone_rows"].sum()
    assert not slot1["required_effect"]                                         # fewer than 4 folds


def test_a_partition_that_does_not_add_up_fails_its_check():
    m = np.array([1, 1, 1, 0, 1], bool)
    y = np.array([1, 0, 1, 1, 0])
    w = np.array([1, 0, 0, 1, 1], bool)
    counts, ok = BC.partition_counts(m, m & w, m & ~w, y)
    assert ok and counts == {"alone": (4, 2), "required": (2, 1), "excluded": (2, 1)}
    assert not BC.partition_counts(m, m & w, m & ~w & (np.arange(5) != 1), y)[1]      # a row lost
    assert not BC.partition_counts(m, m & w, m, y)[1]                                  # overlap


# ======================================================================
# the decision rules
# ======================================================================
def _focal(rows):
    base = {"feasible_folds": 5, "triggers_fired": [], "min_support_ratio": 1.0, "migration_rows": 100,
            "lift_lower": 1.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def _removals(lifts):
    rows = []
    for (fam, proc), (lift, lower) in lifts.items():
        for name in ("fold 0", "period 2008-05"):
            rows.append({"family": fam, "procedure": proc, "removed": name, "lift": lift, "lift_lower": lower})
    return {"removals": pd.DataFrame(rows)}


def _slots(required, excluded):
    return {"by_slot": pd.DataFrame({"slot": range(1, len(required) + 1), "required_effect": required,
                                     "excluded_effect": excluded})}


def test_the_decision_advances_only_candidates_meeting_every_criterion():
    focal = _focal([
        {"family": "NSD", "procedure": "frozen", "lift": 2.5, "lift_lower": 2.2, "min_fold_lift": 0.6, "migration_rows": 650},
        {"family": "NSD", "procedure": "share_matched", "lift": 2.6, "min_fold_lift": 2.0, "migration_rows": 300},
        {"family": "NSD", "procedure": "gap_placed", "lift": 2.6, "min_fold_lift": 2.0, "migration_rows": 300},
        {"family": "ECI", "procedure": "frozen", "lift": 2.5, "min_fold_lift": 2.0},
        {"family": "ECI", "procedure": "share_matched", "lift": 2.6, "min_fold_lift": 2.0},
        {"family": "ECI", "procedure": "gap_placed", "lift": 2.9, "min_fold_lift": 2.1, "min_support_ratio": 0.4},
        {"family": "ECI", "procedure": "tail_selected", "lift": 3.0, "min_fold_lift": 2.1, "feasible_folds": 4},
    ])
    dep = _removals({("NSD", "frozen"): (2.4, 2.1), ("NSD", "share_matched"): (2.5, 2.2),
                     ("NSD", "gap_placed"): (2.5, 2.2), ("ECI", "frozen"): (2.5, 2.2),
                     ("ECI", "share_matched"): (2.6, 2.3), ("ECI", "gap_placed"): (2.9, 2.5)})
    d = BC.decide({"focal": focal}, dep, _slots([True, True, False, True, False], [False] * 5))
    v = d["verdicts"].set_index(["family", "procedure"])
    assert v.loc[("NSD", "share_matched"), "verdict"] == "advance to PR 44"     # tie: the simpler one
    assert "outranked by share_matched" in v.loc[("NSD", "gap_placed"), "reason"]
    assert v.loc[("ECI", "share_matched"), "reason"].startswith("fails A3a")    # 2.6 < 1.1 x 2.5
    assert "A2" in v.loc[("ECI", "gap_placed"), "reason"]
    assert v.loc[("ECI", "tail_selected"), "reason"].startswith("fails A0")
    assert d["families"]["NSD"]["advance"] == "share_matched" and d["families"]["ECI"]["advance"] is None
    assert d["families"]["NSD"]["revision"]["g13"] and d["revision"]["warranted"] == "pending PR 44"
    assert d["eci_warm_filter"]["verdict"] == "advance to PR 44 as a filter"
    assert d["eci_warm_filter"]["direction"] == "warm_required"


def test_a_gain_that_one_removal_erases_is_closed_out():
    focal = _focal([
        {"family": "NSD", "procedure": "frozen", "lift": 2.5, "lift_lower": 2.2, "min_fold_lift": 0.6, "migration_rows": 650},
        {"family": "NSD", "procedure": "share_matched", "lift": 2.6, "min_fold_lift": 2.0, "migration_rows": 300},
        {"family": "NSD", "procedure": "gap_placed", "lift": 2.6, "min_fold_lift": 2.0, "migration_rows": 900},
        {"family": "ECI", "procedure": "frozen", "lift": 2.5, "min_fold_lift": 2.0},
        {"family": "ECI", "procedure": "share_matched", "lift": 2.8, "min_fold_lift": 2.0},
        {"family": "ECI", "procedure": "gap_placed", "lift": 2.8, "min_fold_lift": 2.0},
        {"family": "ECI", "procedure": "tail_selected", "lift": 2.8, "min_fold_lift": 2.0, "triggers_fired": ["T3"]},
    ])
    dep = _removals({("NSD", "frozen"): (2.4, 2.1), ("NSD", "share_matched"): (2.0, 1.8),
                     ("NSD", "gap_placed"): (2.5, 2.2), ("ECI", "frozen"): (2.5, 2.2),
                     ("ECI", "share_matched"): (2.4, 2.1), ("ECI", "gap_placed"): (2.6, 2.2),
                     ("ECI", "tail_selected"): (2.6, 2.2)})
    d = BC.decide({"focal": focal}, dep, _slots([True, True, False, False, False], [True] * 5))
    c = d["criteria"].set_index(["family", "procedure"])
    assert c.loc[("NSD", "share_matched"), "failed"] == ["A4"]
    assert c.loc[("NSD", "share_matched"), "A4_failing_removals"] == ["fold 0", "period 2008-05"]
    assert c.loc[("NSD", "gap_placed"), "failed"] == ["A3c"]
    assert c.loc[("ECI", "share_matched"), "failed"] == ["A4"]
    assert c.loc[("ECI", "tail_selected"), "failed"] == ["A1"]
    assert d["families"]["ECI"]["advance"] == "gap_placed"
    assert d["eci_warm_filter"]["direction"] == "warm_excluded"
    none = BC.decide({"focal": focal}, dep, _slots([True, True, False, False, False], [False] * 5))
    assert none["eci_warm_filter"] == {**none["eci_warm_filter"], "verdict": "close out as a filter",
                                       "direction": None}


# ======================================================================
# reference inputs and the post-registration sensitivity
# ======================================================================
def test_every_reference_input_is_traced_to_the_full_training_supports():
    ref = BC.reference_inputs()
    assert set(ref["input"]) == {"frozen_shares()", "window_halfwidths()", "_tail_range()"}
    for _, r in ref.iterrows():
        assert "all 32,950 training rows" in r["derived_from"] and "held-out rows" in r["derived_from"]
        want = {"frozen_shares()": BC.frozen_shares, "window_halfwidths()": BC.window_halfwidths}
        if r["input"] in want:
            assert r["values"] == list(want[r["input"]](r["family"]))
        else:
            assert r["values"] == BC._tail_range()
    used = " ".join(ref["used_by"])
    for proc in ("share_matched", "gap_placed", "tail_selected"):
        assert proc in used
    for text in (BC.REFERENCE_SCOPE, BC.ISOLATION_SCOPE):
        assert "frozen_shares()" in text and "window_halfwidths()" in text and "_tail_range()" in text
    assert "conditional" in BC.ISOLATION_SCOPE and "PR 44" in BC.FILTER_SCOPE
    assert any("No arm of the comparison is fully nested" in x for x in BC.LIMITS)


def test_the_sensitivity_is_not_part_of_the_registration():
    assert BC.registration_digest() == BC.REGISTERED_SHA256
    assert "sensitivity" not in json.dumps(BC.REGISTRATION).lower()
    assert BC.SENSITIVITY["status"].startswith("post-registration")
    assert BC.SENSITIVITY["name"] not in BC.SIMPLICITY and BC.SENSITIVITY["name"] not in BC.CANDIDATES["NSD"]


def test_fold_local_reference_reads_the_frozen_cuts_on_the_fit_side_values():
    values = [0.01, 0.04, 0.1, 0.2, 0.21, 0.25, 0.3]
    rows = [30, 20, 10, 10, 10, 10, 10]
    blk = BC.blocks(_states(values, rows))                     # frozen NSD cuts 0.0524 / 0.192 / 0.23
    ref = BC.fold_local_reference("NSD", blk)
    assert ref["bin_rows"] == [50, 10, 20, 20]
    assert ref["shares"] == (0.5, 0.6, 0.8)
    assert ref["halfwidths"] == (0.025, 0.025, 0.05)          # 0.25 x min(.5,.1), min(.1,.2), min(.2,.2)
    fit = BC.fit_fold_local("NSD", blk)
    same = BC.fit_gap_placed("NSD", blk, shares=ref["shares"], halfwidths=ref["halfwidths"])
    assert fit["cuts"] == same["cuts"] and fit["procedure"] == BC.SENSITIVITY["name"]
    assert BC.fit_gap_placed("NSD", blk) == BC.fit_gap_placed("NSD", blk, shares=BC.frozen_shares("NSD"),
                                                              halfwidths=BC.window_halfwidths("NSD"))


def test_the_sensitivity_reports_beside_the_registered_result_without_changing_it(world):
    side, T = world["side"], world["T"]
    before = copy.deepcopy(BC._canonical(world["comparison"]["focal"].to_dict("records")))
    all_rows = BC.sensitivity_all_rows(side, world["states"], world["svals"])
    sens = BC.sensitivity(side, world["sources"], T, world["fold_fits"], all_rows, world["comparison"],
                          world["deployed"])
    assert BC._canonical(world["comparison"]["focal"].to_dict("records")) == before
    assert len(sens["per_fold"]) == 5
    for k, r in enumerate(sens["per_fold"]):
        ff = world["fold_fits"][k]
        assert r["registered_cuts"] == ff["fits"]["NSD"]["gap_placed"]["cuts"]
        assert r["registered_targets"] == list(BC.frozen_shares("NSD"))
    if sens["summary"] is not None:
        summ = sens["summary"].set_index("procedure")
        assert list(summ.index) == ["frozen", "gap_placed", BC.SENSITIVITY["name"]]
        assert summ.loc["gap_placed", "lift"] == world["comparison"]["focal"].set_index(
            ["family", "procedure"]).loc[("NSD", "gap_placed"), "lift"]
        lab = sens["labels"]
        for k, ff in enumerate(world["fold_fits"]):
            va = side.folds[k][1]
            want = BD.assign_intervals(world["sources"].values["oof"]["NSD"][va], ff["sensitivity"]["NSD"]["cuts"],
                                       E.family_bins("NSD"))
            assert np.array_equal(lab[va], want)
        assert set(sens["persists"]) >= {"fold0_repair", "zero_migration", "material_change",
                                         "all_rows_fit_equals_registered"}
        assert sens["criteria"]["note"].startswith("the registered criteria applied descriptively")


# ======================================================================
# the notebook's Step 4B cells
# ======================================================================
def test_the_notebook_registers_before_any_result_and_writes_its_record_once():
    nb = json.loads((BANK / "feature_research" / "feature_research.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    headers = [next((ln for ln in s.splitlines() if ln.startswith("# CELL ")), "") for s in code]
    idx = {h.split(":")[0].replace("# CELL ", ""): i for i, h in enumerate(headers)}
    assert all(str(c) in idx for c in range(40, 49))
    assert idx["39"] < idx["40"] < idx["48"] < len(code)             # Step 5A follows (Cells 49-59)
    reg = code[idx["40"]]
    assert "BC.registration_checks(PR39_BINNING)" in reg and "BC.require(REGISTRATION_CHECKS_4B" in reg
    assert "BC.registration_digest()" in reg and "BC.FOLD_FINGERPRINT" in reg
    first = min(i for i, s in enumerate(code) if "BC.fit_folds(" in s or "BC.compare(" in s
                or "BC.warm_filter(" in s)
    assert first > idx["40"]
    joined = "\n".join(code)
    assert joined.count('"binning_candidates"') == 3               # seed name, record written, record read
    assert code[idx["48"]].count('RR.write_run_record(RUN, "binning_candidates"') == 1
    assert 'RR.read_run_record(RUN.run_id, "binning_candidates")' in code[idx["48"]]
    assert "sens=SENSITIVITY_4B" in code[idx["48"]]
    assert "BC.require(STATED_4B" in code[idx["46"]] and "BC.decide(" in code[idx["46"]]
    assert "BC.sensitivity(" in code[idx["47"]] and "BC.require(SENSITIVITY_CHECKS_4B" in code[idx["47"]]
    assert not [s for s in code[:idx["47"]] if "BC.sensitivity(" in s]     # after the registered decision
    assert "BC.reference_inputs()" in code[idx["41"]]
    assert "from feature_research import binning_candidates as BC" in code[0]
