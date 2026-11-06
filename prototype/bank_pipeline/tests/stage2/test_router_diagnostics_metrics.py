"""
PR 37 Commit 1 (Step 1C): operating points, rate-matched cuts and
decision-level router diagnostics of shared.stage2.diagnostics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from blackbox_pipeline.models.rf_router.evaluation.router_metrics import evaluate_router
from shared.stage2 import diagnostics as d
from tests.support import diagnostics_world as dw


@pytest.fixture(scope="module")
def built():
    w = dw.world()
    ref, glass, rf, lr, ebm = dw.views(w)
    s4 = d.extract_stage4_recall_thresholds(dw.trapped(w["meta"]), ref, lr_view=lr, ebm_view=ebm)
    return {"w": w, "ref": ref, "glass": glass, "rf": rf, "lr": lr, "ebm": ebm, "s4": s4}


def _view_for(b, name):
    return b["lr"] if name.startswith("lr.") else b["ebm"]


# ======================================================================
# Ratio and definitions
# ======================================================================
def test_ratio_with_empty_denominator_is_undefined_not_zero():
    r = d.Ratio("pass2_precision", 0, 0)
    assert r.value is None and r.status == "undefined_empty_denominator"
    assert math.isnan(r.as_float())
    assert d.Ratio("x", 3, 4).value == 0.75 and d.Ratio("x", 3, 4).status == "defined"


def test_every_reported_ratio_is_defined_in_the_registry(built):
    b = built
    flags = d.resolve_operating_point(b["lr"], "lr.in_stage_f2_cv")
    results = [d.router_metrics(b["glass"]), d.router_upstream_metrics(b["rf"], flags),
               d.router_router_table(b["glass"], b["rf"])]
    for r in results:
        for name, v in r.values.items():
            if isinstance(v, d.Ratio):
                assert name in d.METRIC_DEFINITIONS or name == "state_agreement", name


def test_empty_pass2_population_gives_undefined_precision(built):
    m = d.pass2_metrics(built["glass"], rows=built["glass"].pass1)
    assert m["pass2_n"] == 0 and m["pass2_precision"].value is None
    assert m["pass2_recall"].denominator == int(built["glass"].y[built["glass"].pass1].sum())


# ======================================================================
# parity with the repo's set-level router metrics
# ======================================================================
@pytest.mark.parametrize("arm", ["glass", "rf"])
def test_router_metrics_match_evaluate_router(built, arm):
    v = built[arm]
    ref = evaluate_router(np.asarray(v.source_pred), np.asarray(v.confidence),
                          np.asarray(v.source_decision), np.asarray(v.y))
    ours = d.router_metrics(v).values
    pairs = {"coverage": "coverage", "abstain_rate": "abstain_rate", "pass1_npv": "pass1_npv",
             "pass1_leakage_share": "pass1_leakage_rate",
             "pass1_negative_recall": "pass1_negative_recall",
             "pass2_precision": "pass2_precision", "pass2_recall": "pass2_recall",
             "abstain_positive_rate": "abstain_base_rate", "pass1_rate": "pass1_share",
             "pass2_rate": "pass2_share"}
    for mine, theirs in pairs.items():
        assert ours[mine].value == pytest.approx(ref[theirs], abs=1e-15), mine
    assert ours["pass1_positives"] == ref["pass1_subscribers_lost"]
    assert ours["pass2_positives"] == ref["pass2_positives_found"]
    assert (ours["pass1_n"], ours["pass2_n"], ours["abstain_n"]) == (
        ref["pass1_n"], ref["pass2_n"], ref["abstain_n"])


def test_state_counts_partition_the_rows(built):
    v = built["rf"]
    pooled = d.state_counts(v)
    assert pooled["n"].sum() == v.n and pooled["positives"].sum() == int(v.y.sum())
    per_fold = d.state_counts(v, by_fold=True)
    assert per_fold["n"].sum() == v.n and set(per_fold["fold"]) == set(range(5))
    table = d.router_metrics(v, by_fold=True).table
    assert set(table["fold"]) == set(range(5))


def test_eligible_recall_excludes_positives_claimed_by_pass1(built):
    v = built["glass"]
    m = d.pass2_metrics(v)
    assert m["pass2_recall_eligible"].denominator == int(v.y.sum()) - int(v.y[v.pass1].sum())


# ======================================================================
# operating points
# ======================================================================
@pytest.mark.parametrize("name", list(d.OPERATING_POINTS))
def test_every_registered_operating_point_resolves(built, name):
    view = _view_for(built, name)
    f = d.resolve_operating_point(view, name, stage4=built["s4"])
    spec = d.OPERATING_POINTS[name]
    assert f.flags.dtype == bool and not f.flags.flags.writeable and f.index.equals(view.index)
    assert f.disclosure == spec.disclosure and f.selection == spec.selection
    if spec.source == "stored_decision":
        np.testing.assert_array_equal(f.flags, view.stored_decisions[name].values == 1)
    else:
        score = view.scores[spec.score_space].values
        np.testing.assert_array_equal(f.flags, score >= f.threshold)


def test_registry_roles_and_spaces():
    assert d.PRIMARY_OPERATING_POINTS == ("lr.in_stage_f2_cv", "ebm.nested_f2")
    ops = d.OPERATING_POINTS
    assert ops["ebm.global_f2_raw"].score_space == "raw_model_score"
    assert ops["ebm.global_f2_calibrated"].score_space == "calibrated_probability"
    assert all(ops[n].score_space == "calibrated_probability"
               for n in ("lr.stage4_recall70", "ebm.stage4_recall70"))
    assert "In-selection" in ops["lr.in_stage_f2_cv"].disclosure
    assert "inner OOF" in ops["ebm.nested_f2"].disclosure


@pytest.mark.parametrize("call, exc, match", [
    (lambda b: d.resolve_operating_point(b["lr"], 0.1), d.RouterDiagnosticError, "unknown"),
    (lambda b: d.resolve_operating_point(b["lr"], "lr.f2"), d.RouterDiagnosticError, "unknown"),
    (lambda b: d.resolve_operating_point(b["lr"], "ebm.nested_f2"), d.RouterDiagnosticError,
     "ebm operating point"),
    (lambda b: d.resolve_operating_point(b["glass"], "lr.in_stage_f2_cv"), d.RouterDiagnosticError,
     "UpstreamOOFView"),
    (lambda b: d.resolve_operating_point(b["lr"], "lr.stage4_recall70"), d.UnsupportedDiagnostic,
     "stage4"),
])
def test_operating_point_misuse_is_refused(built, call, exc, match):
    with pytest.raises(exc, match=match):
        call(built)


def test_threshold_in_the_wrong_space_is_refused(built):
    import dataclasses
    lr = built["lr"]
    rec = dataclasses.replace(lr.thresholds["lr.in_stage_f2_cv"], space="raw_model_score")
    forged = dataclasses.replace(lr, thresholds={"lr.in_stage_f2_cv": rec})
    with pytest.raises(d.RouterDiagnosticError, match="space"):
        d.resolve_operating_point(forged, "lr.in_stage_f2_cv")


def test_combine_flags(built):
    a = d.resolve_operating_point(built["lr"], "lr.in_stage_f2_cv")
    b = d.resolve_operating_point(built["ebm"], "ebm.nested_f2")
    both, either = d.combine_flags(a, b, how="and"), d.combine_flags(a, b, how="or")
    np.testing.assert_array_equal(both.flags, a.flags & b.flags)
    np.testing.assert_array_equal(either.flags, a.flags | b.flags)
    assert both.components == ("lr.in_stage_f2_cv", "ebm.nested_f2")
    with pytest.raises(d.RouterDiagnosticError):
        d.combine_flags(a, b, how="xor")


# ======================================================================
# router x upstream, router x router
# ======================================================================
def test_router_upstream_metrics_by_hand(built):
    v = built["glass"]
    f = d.resolve_operating_point(built["ebm"], "ebm.nested_f2")
    r = d.router_upstream_metrics(v, f)
    y, fl = np.asarray(v.y) == 1, np.asarray(f.flags)
    p1, p2, ab = v.pass1, v.pass2, v.abstain
    expect = {
        "pass1_fp_suppression": ((p1 & fl & ~y).sum(), (fl & ~y).sum()),
        "pass1_tp_cost": ((p1 & fl & y).sum(), (fl & y).sum()),
        "pass1_leak_upstream_flagged": ((p1 & y & fl).sum(), (p1 & y).sum()),
        "pass2_upstream_redundancy": ((p2 & fl).sum(), p2.sum()),
        "pass2_unique_positive_share": ((p2 & y & ~fl).sum(), (p2 & y).sum()),
        "covered_agreement": (((p1 | p2) & (p2 == fl)).sum(), (p1 | p2).sum()),
        "abstain_upstream_flag_rate": ((ab & fl).sum(), ab.sum()),
    }
    for name, (num, den) in expect.items():
        assert (r[name].numerator, r[name].denominator) == (num, den), name
    assert (r["pass1_leak_upstream_flagged"].numerator + r["pass1_leak_upstream_missed"].numerator
            == r["pass1_leak_upstream_flagged"].denominator)
    assert (r["pass2_unique_positive_share"].numerator + r["pass2_shared_positive_share"].numerator
            == r["pass2_unique_positive_share"].denominator)
    assert d.HISTORICAL_TEST_EXPOSURE in r.notes and any(f.disclosure in n for n in r.notes)
    assert r.meta["same_fold_plan"] is True
    assert d.router_upstream_metrics(v, d.resolve_operating_point(built["lr"], "lr.in_stage_f2_cv")
                                     ).meta["same_fold_plan"] is False      # Stage 1: 10 folds


def test_router_upstream_metrics_refuse_other_rows(built):
    other = dw.world(seed=1)
    _, _, _, lr_other, _ = dw.views(other)
    f = d.resolve_operating_point(lr_other, "lr.in_stage_f2_cv")
    with pytest.raises(d.RouterDiagnosticError, match="different rows"):
        d.router_upstream_metrics(built["glass"], f)


def test_router_router_table(built):
    r = d.router_router_table(built["glass"], built["rf"])
    assert len(r.table) == 9 and r.table["n"].sum() == built["glass"].n
    diag = r.table[r.table["glass_state"] == r.table["blackbox_state"]]
    assert diag["n"].sum() == r["state_agreement"].numerator
    assert r["same_fold_plan"] is True


# ======================================================================
# rate-matched cuts
# ======================================================================
def test_matched_cut_exact_without_ties():
    s = np.array([0.1, 0.4, 0.3, 0.9, 0.7])
    c = d.matched_rate_cut(s, target_count=2)
    assert c.exact and c.cut_value == 0.7 and c.strict_count == c.inclusive_count == 2
    np.testing.assert_array_equal(c.strict, [False, False, False, True, True])
    lo = d.matched_rate_cut(s, target_count=2, ranking="lowest")
    np.testing.assert_array_equal(lo.inclusive, [True, False, True, False, False])


def test_matched_cut_reports_both_cuts_on_a_tie():
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    c = d.matched_rate_cut(s, target_count=2)
    assert not c.exact and c.cut_value == 0.5 and c.tie_block == 3
    assert (c.strict_count, c.inclusive_count) == (1, 4)
    np.testing.assert_array_equal(c.strict, [True, False, False, False, False])
    np.testing.assert_array_equal(c.inclusive, [True, True, True, True, False])


def test_matched_cut_population_zero_and_errors():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    pop = np.array([False, True, True, True])
    c = d.matched_rate_cut(s, target_count=1, population_mask=pop)
    np.testing.assert_array_equal(c.inclusive, [False, True, False, False])
    z = d.matched_rate_cut(s, target_count=0)
    assert z.exact and z.cut_value is None and not z.inclusive.any()
    for kw, match in (({"target_count": 5}, "outside"), ({"target_count": 1, "ranking": "top"},
                                                          "ranking")):
        with pytest.raises(d.RouterDiagnosticError, match=match):
            d.matched_rate_cut(s, **kw)
    with pytest.raises(d.RouterDiagnosticError, match="NaN"):
        d.matched_rate_cut(np.array([0.1, np.nan]), target_count=1)


def test_matched_cut_per_fold():
    s = np.array([0.9, 0.1, 0.5, 0.8, 0.2, 0.2])
    fold = np.array([0, 0, 0, 1, 1, 1])
    c = d.matched_rate_cut(s, target_count={0: 1, 1: 2}, fold_id=fold)
    assert not c.exact and c.by_fold[0]["exact"] and not c.by_fold[1]["exact"]
    np.testing.assert_array_equal(c.strict, [True, False, False, True, False, False])
    np.testing.assert_array_equal(c.inclusive, [True, False, False, True, True, True])
    with pytest.raises(d.RouterDiagnosticError, match="folds"):
        d.matched_rate_cut(s, target_count={0: 1}, fold_id=fold)


def test_matched_rate_comparison_against_an_upstream_score(built):
    v, lr = built["glass"], built["lr"]
    score = lr.scores["calibrated_probability"]
    r = d.matched_rate_comparison(v, d.PASS_1_NOT_SUBSCRIBE, score)
    t = r.table.set_index("set")
    assert t.loc["router", "n"] == int(v.pass1.sum()) == r["target_count"]
    assert t.loc["cut_strict", "n"] <= r["target_count"] <= t.loc["cut_inclusive", "n"]
    assert r.meta["ranking"] == "lowest"
    r2 = d.matched_rate_comparison(v, d.PASS_2_SUBSCRIBE, score, population="pass1_remainder",
                                   by_fold=True)
    assert r2.meta["ranking"] == "highest" and r2.meta["cut"].by_fold is not None
    assert r2.table.set_index("set").loc["router", "n"] == int(v.pass2.sum())
    with pytest.raises(d.RouterDiagnosticError):
        d.matched_rate_comparison(v, d.ABSTAIN, score)


# ======================================================================
# score-stream support
# ======================================================================
def test_score_streams_are_refused_where_the_artifacts_cannot_support_them(built):
    g, rf = built["glass"], built["rf"]
    assert d.require_score_stream(g, "router_transform") is g.router_score
    for view, which, match in ((g, "pass1", "rule membership"), (g, "pass2", "rule membership"),
                               (rf, "pass2", "not persisted"), (rf, "pass1", "PASSED")):
        with pytest.raises(d.UnsupportedDiagnostic, match=match):
            d.require_score_stream(view, which)
