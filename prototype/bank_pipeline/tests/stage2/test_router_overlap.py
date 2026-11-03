"""
PR 37 Commit 2 (Step 2D): pass overlap and redundancy diagnosis of
shared.stage2.diagnostics.overlap on the synthetic Commit 1 world, plus small
hand-built counterexamples for the score-conditional helpers.

Mechanics only (counts, denominators, partitions, parity with Steps 2A-2C,
label-free strata, ties, expectations, unavailable signals, read-only
behaviour). Historical values: tests/realdata/test_router_overlap_artifacts.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import overlap as ov
from shared.stage2.diagnostics import trace as tr
from shared.stage2.diagnostics.schema import ScoreStream
from tests.support import diagnostics_world as dw

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN
ARMS = ("glass", "blackbox")
FOLDS = ["all", 0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def built():
    w = dw.world()
    ref, glass, rf, lr, ebm = dw.views(w)
    trace = d.build_router_trace(glass, rf, ref)
    refs = d.primary_reference_flags(lr, ebm)
    strata = d.score_strata(trace, lr, ebm)
    cand = d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), ref)
    return {"w": w, "ref": ref, "glass": glass, "rf": rf, "lr": lr, "ebm": ebm, "trace": trace,
            "refs": refs, "strata": strata, "cand": cand,
            "matched": d.pass2_matched_contribution(trace, lr, ebm)}


# ----------------------------------------------------------------------
def _group(t, fold):
    return np.ones(t.n, bool) if fold == "all" else np.asarray(t.fold_id) == fold


def _fold(df, fold):
    return df[df["fold"].map(lambda f: f == fold)]


def _reroute(view, rows, token):
    dec = np.array(view.source_decision, dtype=object, copy=True)
    dec[rows] = token
    return replace(view, source_decision=dec,
                   state=np.array([d.SOURCE_DECISIONS[x][0] for x in dec], dtype=object),
                   source_pred=np.array([d.SOURCE_DECISIONS[x][1] for x in dec]))


def _with_score(view, space, values):
    old = view.scores[space]
    a = np.array(values, dtype=float)
    a.setflags(write=False)
    scores = dict(view.scores)
    scores[space] = ScoreStream(old.name, a, old.space, old.field, old.provenance)
    return replace(view, scores=MappingProxyType(scores))


def _opposite_world(b):
    """A trace with GLASS Pass 1 / RF Pass 2 and GLASS Pass 2 / RF Pass 1 conflicts."""
    g, r = b["glass"], b["rf"]
    g1 = np.flatnonzero(np.asarray(g.pass1) & np.asarray(r.pass1))[:9]
    g2 = np.flatnonzero(np.asarray(g.pass2) & np.asarray(r.pass2))[:5]
    r2 = _reroute(_reroute(r, g1, "pass2"), g2, "pass1")
    return d.build_router_trace(g, r2, b["ref"]), g1, g2


# ======================================================================
# hand-built counterexamples for the score-conditional helpers
# ======================================================================
def test_quantile_bins_keep_ties_together_and_merge_duplicate_edges():
    s = np.array([0.1] * 6 + [0.2, 0.3, 0.4, 0.5])
    bins, edges = ov._quantile_bins(s, 10)
    assert len(np.unique(bins[:6])) == 1                         # tied scores share a stratum
    assert len(edges) == len(np.unique(edges)) < 9               # duplicate edges merged
    assert (np.diff(bins[np.argsort(s, kind="stable")]) >= 0).all()   # monotone in score
    one, e1 = ov._quantile_bins(np.full(7, 0.3), 10)
    assert len(set(one)) == 1 and len(e1) == 1                   # degenerate: a single stratum
    empty, e0 = ov._quantile_bins(np.zeros(0), 10)
    assert empty.size == 0


def test_flag_disagreement_without_within_stratum_information():
    """Marginal rates differ by state only because state follows the score."""
    cells = np.array([0] * 10 + [1] * 10)
    y = np.array([1] + [0] * 9 + [1] * 5 + [0] * 5)              # cell rates 10% and 50%
    state_a = np.array([True] * 10 + [False] * 10)               # 'state' == low cell
    rates = ov._row_cell_rate(y, cells)
    for group in (state_a, ~state_a):
        assert (y[group]).sum() == pytest.approx(rates[group].sum())       # O = E in strata
    assert y[state_a].mean() != y[~state_a].mean()                # yet the marginal rates differ


def test_within_stratum_separation_is_measured():
    cells = np.zeros(20, int)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [0] * 10)       # state A: 4/10, state B: 0/10
    a = np.array([True] * 10 + [False] * 10)
    rates = ov._row_cell_rate(y, cells)                           # cell rate 4/20
    assert y[a].sum() - rates[a].sum() == pytest.approx(4 - 2.0)
    assert y[~a].sum() - rates[~a].sum() == pytest.approx(0 - 2.0)


# ======================================================================
# router x reference: contingency and agreement
# ======================================================================
def test_contingency_by_hand_and_partition(built):
    t, refs = built["trace"], built["refs"]
    ct = d.router_reference_contingency(t, refs)
    y = np.asarray(t.y) == 1
    for key in d.suppression.REFERENCE_KEYS:
        f = np.asarray(refs[key].flags)
        for arm in ARMS:
            st = np.asarray(t.views[arm].state)
            for fold in FOLDS:
                k = _group(t, fold)
                sub = _fold(ct[(ct["arm"] == arm) & (ct["reference"] == key)], fold)
                assert len(sub) == 6 and sub["n"].sum() == k.sum()
                assert sub["positives"].sum() == (k & y).sum()
                for _, r in sub.iterrows():
                    m = k & (st == r["state"]) & (f == r["reference_flag"])
                    assert (r["n"], r["positives"], r["negatives"]) == (m.sum(), (m & y).sum(),
                                                                       (m & ~y).sum())


def test_contingency_parity_with_steps_2b_and_2c(built):
    t, refs = built["trace"], built["refs"]
    ct = _fold(d.router_reference_contingency(t, refs), "all")
    sup = d.pass1_suppression_report(t, built["lr"], built["ebm"])
    fz = d.pass2_frozen_contribution(t, built["lr"], built["ebm"])
    for key in d.suppression.REFERENCE_KEYS:
        for arm in ARMS:
            c = ct[(ct["arm"] == arm) & (ct["reference"] == key)].set_index(
                ["state", "reference_flag"])
            assert c.loc[(P1, True), "n"] == sup[f"{key}.{arm}.intercepted_flags"]
            assert c.loc[(P1, True), "positives"] == sup[f"{key}.{arm}.suppressed_tp"]
            assert c.loc[(P2, True), "positives"] == fz[f"{key}.{arm}.overlap_tp"]
            assert c.loc[(P2, False), "positives"] == fz[f"{key}.{arm}.unique_tp"]
            assert c.loc[(P2, False), "negatives"] == fz[f"{key}.{arm}.unique_fp"]


def test_agreement_denominators_and_parity(built):
    t, refs = built["trace"], built["refs"]
    ag = d.router_reference_agreement(t, refs)
    for key in d.suppression.REFERENCE_KEYS:
        for arm in ARMS:
            a = _fold(ag[(ag["arm"] == arm) & (ag["reference"] == key)], "all").set_index("metric")
            theirs = d.router_upstream_metrics(t.views[arm], refs[key])
            for m in ("pass1_upstream_flag_rate", "pass2_upstream_redundancy", "covered_agreement",
                      "abstain_upstream_flag_rate", "abstain_upstream_precision"):
                assert (a.loc[m, "numerator"], a.loc[m, "denominator"]) == (
                    theirs[m].numerator, theirs[m].denominator), (key, arm, m)
            v = t.views[arm]
            assert a.loc["covered_agreement", "denominator"] == (v.pass1 | v.pass2).sum() < t.n
            assert a.loc["coverage", "numerator"] == a.loc["covered_agreement", "denominator"]
            assert a.loc["pass1_upstream_negative_agreement", "numerator"] + a.loc[
                "pass1_upstream_flag_rate", "numerator"] == v.pass1.sum()
            shares = sum(a.loc[f"upstream_flags_in_{s}_share", "numerator"]
                         for s in ("pass1", "pass2", "abstain"))
            assert shares == a.loc["upstream_flags_in_pass1_share", "denominator"] == int(
                np.asarray(refs[key].flags).sum())


def test_abstention_is_a_state_not_a_negative(built):
    t, refs = built["trace"], built["refs"]
    rows = np.flatnonzero(np.asarray(built["glass"].abstain) & np.asarray(refs["union"].flags))[:4]
    t2 = d.build_router_trace(_reroute(built["glass"], rows, "pass1"), built["rf"], built["ref"])
    a1 = _fold(d.router_reference_agreement(t, refs), "all")
    a2 = _fold(d.router_reference_agreement(t2, refs), "all")
    pick = lambda a, m: a[(a["arm"] == "glass") & (a["reference"] == "union")  # noqa: E731
                          & (a["metric"] == m)].iloc[0]
    # moving abstained flagged rows into Pass 1 adds Pass 1 conflicts and routed rows
    assert pick(a2, "pass1_upstream_flag_rate")["numerator"] == pick(
        a1, "pass1_upstream_flag_rate")["numerator"] + 4
    assert pick(a2, "covered_agreement")["denominator"] == pick(
        a1, "covered_agreement")["denominator"] + 4
    assert pick(a2, "covered_agreement")["numerator"] == pick(a1, "covered_agreement")["numerator"]


# ======================================================================
# GLASS x RF
# ======================================================================
def test_crosstab_partitions_and_matches_router_router_table(built):
    t, refs = built["trace"], built["refs"]
    ct = d.router_crosstab(t, refs)
    theirs = d.router_router_table(t.views["glass"], t.views["blackbox"]).table
    pooled = _fold(ct, "all")
    for _, r in theirs.iterrows():
        mine = pooled[(pooled["glass_state"] == r["glass_state"])
                      & (pooled["rf_state"] == r["blackbox_state"])].iloc[0]
        assert (mine["n"], mine["positives"]) == (r["n"], r["positives"])
    for fold in FOLDS:
        sub = _fold(ct, fold)
        assert len(sub) == 9 and sub["n"].sum() == _group(t, fold).sum()
        assert sub["share"].sum() == pytest.approx(1.0)
        for key in refs:
            assert sub[f"{key}_flags"].sum() == (np.asarray(refs[key].flags) & _group(t, fold)).sum()
    cs = _fold(d.crosstab_summary(t), "all").set_index("metric")
    rr = d.router_router_table(t.views["glass"], t.views["blackbox"])
    assert cs.loc["cross_state_agreement", "numerator"] == rr["state_agreement"].numerator


def test_opposite_decisions_are_counted_exactly(built):
    t, g1, g2 = _opposite_world(built)
    ct = _fold(d.router_crosstab(t, built["refs"]), "all").set_index(["glass_state", "rf_state"])
    assert ct.loc[(P1, P2), "n"] == len(g1) and ct.loc[(P2, P1), "n"] == len(g2)
    cs = _fold(d.crosstab_summary(t), "all").set_index("metric")
    assert cs.loc["cross_opposite_rate", "numerator"] == len(g1) + len(g2)
    both = (t.views["glass"].state != AB) & (t.views["blackbox"].state != AB)
    assert cs.loc["cross_routed_agreement", "denominator"] == both.sum()
    groups = d.key_groups(t, built["refs"])
    assert np.flatnonzero(groups["glass_pass1_and_rf_pass2"][0]).tolist() == sorted(g1.tolist())
    for fold in FOLDS:
        s = _fold(d.crosstab_summary(t), fold).set_index("metric")
        n = _group(t, fold).sum()
        assert s.loc["cross_state_agreement", "denominator"] == n
        assert s.loc["cross_one_abstains_rate", "numerator"] + s.loc[
            "cross_both_abstain_rate", "numerator"] + s.loc[
            "cross_both_routed_rate", "numerator"] == n


def test_cross_router_pass2_overlap(built):
    t, refs = built["trace"], built["refs"]
    t2, g1, _ = _opposite_world(built)
    for trace in (t, t2):
        po = d.pass2_cross_router_overlap(trace, refs)
        fz = d.pass2_frozen_contribution(trace, built["lr"], built["ebm"])
        for _, r in po.iterrows():
            for lab in ("tp", "fp"):
                assert r[f"glass_unique_{lab}"] == r[f"shared_{lab}"] + r[f"glass_only_{lab}"]
                assert r[f"rf_unique_{lab}"] == r[f"shared_{lab}"] + r[f"rf_only_{lab}"]
            assert r["glass_only_tp"] == r["glass_only_tp_in_rf_pass1"] + r["glass_only_tp_in_rf_abstain"]
            assert r["rf_only_tp"] == r["rf_only_tp_in_glass_pass1"] + r["rf_only_tp_in_glass_abstain"]
        pooled = _fold(po, "all").set_index("reference")
        for key in refs:
            assert pooled.loc[key, "glass_unique_tp"] == fz[f"{key}.glass.unique_tp"]
            assert pooled.loc[key, "rf_unique_tp"] == fz[f"{key}.blackbox.unique_tp"]
        y = np.asarray(trace.y) == 1
        assert pooled.loc["all_pass2", "shared_tp"] == (trace.views["glass"].pass2
                                                        & trace.views["blackbox"].pass2 & y).sum()
    po2 = _fold(d.pass2_cross_router_overlap(t2, refs), "all").set_index("reference")
    y = np.asarray(t2.y) == 1
    assert po2.loc["all_pass2", "rf_only_tp_in_glass_pass1"] == (y[g1]).sum()


# ======================================================================
# final states vs signals
# ======================================================================
def test_signal_availability_never_claims_unpersisted_signals(built):
    t = built["trace"]
    sa = d.signal_availability(t, rf_candidate=built["cand"]).set_index(["arm", "signal"])
    for arm in ARMS:
        assert sa.loc[(arm, "final Pass 1 & Pass 2 rows"), "rows"] == 0
        assert "NOT evidence of independent" in sa.loc[(arm, "final Pass 1 & Pass 2 rows"), "detail"]
    for sig in ("Pass 1 candidate-rule matches per row", "Pass 2 candidate-rule matches per row",
                "outer-fold fold rule sets", "full-train selected rule set"):
        assert sa.loc[("glass", sig), "status"] == tr.NOT_VERIFIABLE
    assert "in-sample" in sa.loc[("glass", "full-train selected rule set"), "detail"]
    assert sa.loc[("glass", "fold RF passed to the GLASS fit"), "status"] == tr.CODE_LEVEL_ONLY
    assert sa.loc[("blackbox", "Pass 2 outer score"), "status"] == tr.NOT_VERIFIABLE
    assert sa.loc[("blackbox", "Pass 1 outer score (all rows)"), "status"] == \
        tr.VERIFIED_FROM_ARTIFACTS
    for sig in ("analysis: GLASS Pass 1 vs Pass 2 candidate / signal overlap",
                "analysis: RF Pass 1 vs Pass 2 signal overlap"):
        assert sa.loc[("both", sig), "status"] == tr.NOT_VERIFIABLE
    without = d.signal_availability(t).set_index(["arm", "signal"])
    assert without.loc[("blackbox", "Pass 1 outer score (all rows)"), "status"] == tr.NOT_VERIFIABLE
    assert set(sa["status"]) <= set(d.AUDIT_STATUSES)


# ======================================================================
# score strata
# ======================================================================
def test_strata_are_label_free_within_fold_and_scale_free_for_ebm(built):
    t, lr, ebm, s = built["trace"], built["lr"], built["ebm"], built["strata"]
    y2 = np.random.default_rng(3).permutation(np.asarray(t.y))
    y2.setflags(write=False)
    s2 = d.score_strata(replace(t, y=y2), replace(lr, y=y2), replace(ebm, y=y2))
    for name in ov.STRATIFIERS:
        np.testing.assert_array_equal(s.bins[name], s2.bins[name])
    raw = np.array(ebm.scores["raw_model_score"].values, copy=True)
    fold0 = np.asarray(t.fold_id) == 0
    raw[fold0] = raw[fold0] * 3 + 10                        # rescale one fold's raw scores
    thr = np.array(ebm.stored_decisions["ebm.nested_f2"].per_row_threshold, copy=True)
    thr[fold0] = thr[fold0] * 3 + 10
    sd = ebm.stored_decisions["ebm.nested_f2"]
    stored = MappingProxyType({**ebm.stored_decisions,
                               "ebm.nested_f2": replace(sd, per_row_threshold=thr)})
    s3 = d.score_strata(t, lr, replace(_with_score(ebm, "raw_model_score", raw),
                                       stored_decisions=stored))
    np.testing.assert_array_equal(s.bins["ebm_raw"], s3.bins["ebm_raw"])
    np.testing.assert_allclose(s.ebm_percentile, s3.ebm_percentile)
    np.testing.assert_allclose(s.ebm_threshold_percentile, s3.ebm_threshold_percentile)
    for name in ov.STRATIFIERS:
        assert not np.asarray(s.bins[name]).flags.writeable
    fold = np.asarray(t.fold_id)
    for k in range(5):                                      # roughly equal-sized strata per fold
        counts = np.bincount(np.asarray(s.bins["lr"])[fold == k])
        assert counts.max() - counts.min() <= 2


def test_tied_scores_share_strata_and_collapses_are_reported(built):
    t, lr, ebm = built["trace"], built["lr"], built["ebm"]
    tied = np.round(np.asarray(lr.scores["calibrated_probability"].values), 1)
    s = d.score_strata(t, _with_score(lr, "calibrated_probability", tied), ebm)
    b, fold = np.asarray(s.bins["lr"]), np.asarray(t.fold_id)
    for k in range(5):
        for v in np.unique(tied[fold == k]):
            assert len(np.unique(b[(fold == k) & (tied == v)])) == 1
    assert any(n < ov.SCORE_BINS for n in s.n_bins["lr"].values())
    assert "lr: folds" in d.overlap_notes(t, s)[-1]


def test_strata_refuse_unsupported_inputs(built):
    t, lr, ebm = built["trace"], built["lr"], built["ebm"]
    with pytest.raises(d.RouterDiagnosticError, match="rows / labels"):
        d.score_strata(t, replace(lr, index=lr.index[::-1]), ebm)
    with pytest.raises(d.RouterDiagnosticError, match="LR UpstreamOOFView"):
        d.score_strata(t, ebm, ebm)
    no_raw = replace(ebm, scores=MappingProxyType({k: v for k, v in ebm.scores.items()
                                                   if k != "raw_model_score"}))
    with pytest.raises(d.UnsupportedDiagnostic, match="raw_model_score"):
        d.score_strata(t, lr, no_raw)
    bad = np.array(lr.scores["calibrated_probability"].values, copy=True)
    bad[0] = np.nan
    with pytest.raises(d.UnsupportedDiagnostic, match="finite"):
        d.score_strata(t, _with_score(lr, "calibrated_probability", bad), ebm)
    sd = ebm.stored_decisions["ebm.nested_f2"]
    thr = np.array(sd.per_row_threshold, copy=True)
    thr[np.flatnonzero(np.asarray(t.fold_id) == 2)[0]] += 0.01
    stored = MappingProxyType({**ebm.stored_decisions,
                               "ebm.nested_f2": replace(sd, per_row_threshold=thr)})
    with pytest.raises(d.UnsupportedDiagnostic, match="not constant in fold 2"):
        d.score_strata(t, lr, replace(ebm, stored_decisions=stored))


# ======================================================================
# score-conditional tables
# ======================================================================
def test_score_conditional_counts_expectations_and_support(built):
    t, s, refs = built["trace"], built["strata"], built["refs"]
    sc = d.score_conditional_states(t, s, refs)
    y = np.asarray(t.y) == 1
    p = np.asarray(s.lr_probability)
    for name in ov.STRATIFIERS:
        bins = np.asarray(s.bins[name])
        for arm in ARMS:
            st = np.asarray(t.views[arm].state)
            for fold in FOLDS:
                k = _group(t, fold)
                sub = _fold(sc[(sc["arm"] == arm) & (sc["stratifier"] == name)], fold)
                assert sub["n"].sum() == k.sum() and sub["positives"].sum() == (k & y).sum()
                assert sub["expected_lr_calibrated"].sum() == pytest.approx(p[k].sum())
                for _, r in sub.sample(n=min(6, len(sub)), random_state=0).iterrows():
                    m = k & (bins == r["stratum"]) & (st == r["state"])
                    assert (r["n"], r["positives"]) == (m.sum(), (m & y).sum())
                    assert r["low_support"] == (m.sum() < ov.MIN_SUPPORT)
                    assert r["union_flags"] == (m & np.asarray(refs["union"].flags)).sum()
    ct = d.within_stratum_contrasts(sc)
    assert set(ct["status"]) <= {"defined", "low_support"}
    assert ct.loc[ct["status"] == "low_support", "difference_pp"].isna().all()
    row = ct[ct["status"] == "defined"].iloc[0]
    sub = sc[(sc["arm"] == row["arm"]) & (sc["stratifier"] == row["stratifier"])
             & sc["fold"].map(lambda f: f == row["fold"]) & (sc["stratum"] == row["stratum"])]
    a, b = row["contrast"].split(" - ")
    ra = sub[sub["state"] == {"pass1": P1, "pass2": P2, "abstain": AB}[a]].iloc[0]
    rb = sub[sub["state"] == {"pass1": P1, "pass2": P2, "abstain": AB}[b]].iloc[0]
    assert row["difference_pp"] == pytest.approx(100 * (ra["positive_rate"] - rb["positive_rate"]))


def test_state_standardisation_is_internally_consistent(built):
    t, s = built["trace"], built["strata"]
    ss = d.state_standardization(t, s)
    p = np.asarray(s.lr_probability)
    for arm in ARMS:
        for fold in FOLDS:
            sub = _fold(ss[ss["arm"] == arm], fold)
            for name in ov.STRATIFIERS:                     # states' expectations sum to observed
                assert sub[f"expected_stratified_{name}"].sum() == pytest.approx(
                    sub["observed"].sum())
            st = np.asarray(t.views[arm].state)
            for _, r in sub.iterrows():
                m = _group(t, fold) & (st == r["state"])
                assert r["expected_lr_calibrated"] == pytest.approx(p[m].sum())
                assert r["o_minus_e_lr_calibrated"] == pytest.approx(
                    r["observed"] - r["expected_lr_calibrated"])
                assert r["o_minus_e_stratified_lr"] == pytest.approx(
                    r["observed"] - r["expected_stratified_lr"])


def test_groups_observed_expected_and_parity(built):
    t, s, refs = built["trace"], built["strata"], built["refs"]
    groups = d.key_groups(t, refs)
    oe = _fold(d.group_observed_expected(t, s, groups), "all").set_index("group")
    sup = d.pass1_suppression_report(t, built["lr"], built["ebm"])
    fz = d.pass2_frozen_contribution(t, built["lr"], built["ebm"])
    assert oe.loc["glass_pass1_and_union_flag", "n"] == sup["union.glass.intercepted_flags"]
    assert oe.loc["glass_pass1_and_union_flag", "observed"] == sup["union.glass.suppressed_tp"]
    assert oe.loc["rf_pass2_unique_vs_union", "observed"] == fz["union.blackbox.unique_tp"]
    assert oe.loc["glass_pass2_unique_vs_lr", "observed"] == fz["lr.glass.unique_tp"]
    for name, (mask, _) in groups.items():
        assert not mask.flags.writeable
        assert oe.loc[name, "n"] == mask.sum()
    with pytest.raises(d.RouterDiagnosticError, match="does not cover"):
        d.group_observed_expected(t, s, {"bad": (np.ones(3, bool), "x")})
    with pytest.raises(d.RouterDiagnosticError, match="'union'"):
        d.key_groups(t, {"lr": refs["lr"], "ebm": refs["ebm"]})


def test_boundary_shares_partition_and_respect_flag_definitions(built):
    t, s, refs = built["trace"], built["strata"], built["refs"]
    groups = d.key_groups(t, refs)
    bd = d.boundary_diagnostics(t, s, groups)
    shares = bd[bd["status"] != "quantile"]
    for (g, score, fold), grp in shares.groupby(["group", "score", "fold"], sort=False):
        v = grp.set_index("metric")
        if v.loc["threshold_band_share", "denominator"] > 0:
            assert (v.loc["threshold_band_share", "numerator"] + v.loc["clear_above_share",
                    "numerator"] + v.loc["clear_below_share", "numerator"]) == v.loc[
                "threshold_band_share", "denominator"]
    bda = _fold(bd, "all")
    pick = lambda g, sc, m: bda[(bda["group"] == g) & (bda["score"] == sc)  # noqa: E731
                                & (bda["metric"] == m)].iloc[0]
    r = pick("glass_pass1_and_lr_flag", "lr", "above_threshold_share")
    assert r["numerator"] == r["denominator"] > 0                # LR flags are >= its threshold
    r = pick("rf_pass1_and_union_flag", "lr", "above_threshold_share")
    lrflag = np.asarray(refs["lr"].flags) & groups["rf_pass1_and_union_flag"][0]
    assert r["numerator"] == lrflag.sum()
    q = bda[(bda["group"] == "glass_pass1") & (bda["status"] == "quantile")]
    assert set(q["metric"]) >= {"lr_probability_q50", "lr_percentile_margin_q50",
                                "ebm_percentile_margin_q50"}


# ======================================================================
# arithmetic baselines
# ======================================================================
def test_chance_baselines_by_hand(built):
    t, refs, mt = built["trace"], built["refs"], built["matched"]
    cb = d.unique_positive_chance_baselines(t, refs, mt)
    y = np.asarray(t.y) == 1
    for arm in ARMS:
        v = t.views[arm]
        for key in ("lr", "ebm", "union"):
            f = np.asarray(refs[key].flags)
            r = _fold(cb[(cb["setting"] == "frozen_pass2_unique") & (cb["arm"] == arm)
                         & (cb["reference"] == key)], "all").iloc[0]
            pool = ~f & ~np.asarray(v.pass1)
            uf = np.asarray(v.pass2) & ~f
            assert (r["flags"], r["pool_rows"], r["pool_positives"]) == (uf.sum(), pool.sum(),
                                                                        (pool & y).sum())
            assert r["expected_positives"] == pytest.approx(uf.sum() * (pool & y).sum() / pool.sum())
            assert r["observed_positives"] == (uf & y).sum()
            assert r["excess_positives"] == pytest.approx(r["observed_positives"]
                                                          - r["expected_positives"])
            veto = _fold(cb[(cb["setting"] == "pass1_random_veto") & (cb["arm"] == arm)
                            & (cb["reference"] == key)], "all").iloc[0]
            assert veto["flags"] == (np.asarray(v.pass1) & f).sum()
            assert veto["expected_positives"] == pytest.approx(
                veto["flags"] * (f & y).sum() / f.sum())
        for key in ("lr_matched", "ebm_matched"):
            r = _fold(cb[(cb["setting"] == "matched_pass2_unique") & (cb["arm"] == arm)
                         & (cb["reference"] == key)], "all").iloc[0]
            assert r["observed_positives"] == mt[f"{key}.{arm}.unique_tp"]
            assert r["flags"] == mt[f"{key}.{arm}.unique_tp"] + mt[f"{key}.{arm}.unique_fp"]
            assert r["pool_rows"] == mt[f"{key}.{arm}.rows"] - mt[f"{key}.{arm}.reference_flags"]


def test_model_disagreement_baseline_uses_the_same_k(built):
    t, refs, mt = built["trace"], built["refs"], built["matched"]
    md = d.model_disagreement_baseline(t, refs, mt)
    y = np.asarray(t.y) == 1
    lr, ebm = np.asarray(refs["lr"].flags), np.asarray(refs["ebm"].flags)
    fr = _fold(md[md["setting"] == "frozen"], "all").iloc[0]
    assert (fr["lr_only_tp"], fr["ebm_only_tp"]) == ((lr & ~ebm & y).sum(), (ebm & ~lr & y).sum())
    for arm in ARMS:
        r = _fold(md[(md["setting"] == "matched_same_k") & (md["arm"] == arm)], "all").iloc[0]
        sel = mt.meta["selections"][arm]
        assert r["k"] == r["lr_flags"] == r["ebm_flags"] == sel["lr"].n_flagged == sel["ebm"].n_flagged
        ml, me = np.asarray(sel["lr"].flags), np.asarray(sel["ebm"].flags)
        assert r["ebm_only_tp"] == (me & ~ml & y).sum()
        assert r["router_unique_tp_vs_lr"] == mt[f"lr_matched.{arm}.unique_tp"]
        assert r["router_unique_tp_vs_ebm"] == mt[f"ebm_matched.{arm}.unique_tp"]


# ======================================================================
# reconciliation, misuse, non-mutation, compatibility
# ======================================================================
def test_every_reported_rate_is_registered_and_folds_reconcile(built):
    t, refs = built["trace"], built["refs"]
    tables = (d.router_reference_agreement(t, refs), d.crosstab_summary(t),
              d.boundary_diagnostics(t, built["strata"], d.key_groups(t, refs)))
    names = set()
    for table in tables:
        names |= set(table.loc[~table["status"].isin(["count", "quantile"]), "metric"])
    names |= {"set_positive_rate", "cross_cell_share", "jaccard_with_router"}
    assert names <= set(d.METRIC_DEFINITIONS)
    ag = tables[0]
    for _, grp in ag.groupby(["arm", "reference", "metric"]):
        pooled = _fold(grp, "all").iloc[0]
        folds = grp[grp["fold"].map(lambda f: f != "all")]
        assert folds["numerator"].sum() == pooled["numerator"]
        assert folds["denominator"].sum() == pooled["denominator"]
    ct = d.router_crosstab(t, refs)
    for (a, b), grp in ct.groupby(["glass_state", "rf_state"]):
        assert grp[grp["fold"].map(lambda f: f != "all")]["n"].sum() == _fold(grp, "all")["n"].iloc[0]


@pytest.mark.parametrize("fn", [
    lambda b: d.router_reference_contingency(b["glass"], b["refs"]),
    lambda b: d.crosstab_summary(b["glass"]),
    lambda b: d.score_conditional_states(b["trace"], "strata"),
    lambda b: d.router_crosstab(b["trace"], {}),
    lambda b: d.router_reference_agreement(b["trace"], {"x": np.zeros(3)}),
])
def test_misuse_is_refused(built, fn):
    with pytest.raises(d.RouterDiagnosticError):
        fn(built)


def test_nothing_is_mutated(built):
    t, lr, ebm, refs = built["trace"], built["lr"], built["ebm"], built["refs"]

    def snap():
        h = hashlib.sha256()
        for a in (lr.scores["calibrated_probability"].values, ebm.scores["raw_model_score"].values,
                  ebm.scores["calibrated_probability"].values,
                  ebm.stored_decisions["ebm.nested_f2"].per_row_threshold, t.y, t.fold_id,
                  *(refs[k].flags for k in refs)):
            h.update(np.ascontiguousarray(a).tobytes())
        for arm in ARMS:
            h.update("|".join(map(str, t.views[arm].state)).encode())
        return h.hexdigest()

    before = snap()
    s = d.score_strata(t, lr, ebm)
    groups = d.key_groups(t, refs)
    for fn in (lambda: d.router_reference_contingency(t, refs),
               lambda: d.router_reference_agreement(t, refs), lambda: d.router_crosstab(t, refs),
               lambda: d.crosstab_summary(t), lambda: d.pass2_cross_router_overlap(t, refs),
               lambda: d.signal_availability(t, rf_candidate=built["cand"]),
               lambda: d.score_conditional_states(t, s, refs), lambda: d.state_standardization(t, s),
               lambda: d.group_observed_expected(t, s, groups),
               lambda: d.boundary_diagnostics(t, s, groups),
               lambda: d.unique_positive_chance_baselines(t, refs, built["matched"]),
               lambda: d.model_disagreement_baseline(t, refs, built["matched"])):
        fn()
    assert snap() == before
    for arr in (s.lr_probability, s.ebm_raw, s.lr_percentile, s.ebm_threshold_row):
        assert not np.asarray(arr).flags.writeable
    with pytest.raises(TypeError):
        s.bins["lr"] = None
    with pytest.raises(TypeError):
        groups["both_abstain"] = None


def test_notes_and_earlier_steps_are_unaffected(built):
    t = built["trace"]
    text = " ".join(d.overlap_notes(t, built["strata"]))
    for needle in ("routing invariant", "fixed before any result", "Neither is a null distribution",
                   "not inherently complementary", "joint grid is coarse", "F17",
                   "match_glass_oof", "do not make the arms exactly rate-matched", "fold_rf",
                   "not optimised", "Stage 4"):
        assert needle in text, needle
    assert t.check_frame()["passed"].all()
    assert d.pass1_suppression_report(t, built["lr"], built["ebm"])["union_reconciles"] is True
    assert built["matched"]["lr_matched.glass.pass2_n"] == built["matched"][
        "lr_matched.glass.reference_flags"]
