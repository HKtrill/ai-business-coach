"""
PR 37 Commit 2 (Step 2B): Pass 1 false-positive suppression of
shared.stage2.diagnostics.suppression on the synthetic Commit 1 world.

Mechanics only (masks, numerators, denominators, union logic, alignment,
registered operating points, read-only behaviour). Historical values are
checked on the saved artifacts by
tests/realdata/test_router_suppression_artifacts.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import suppression as sp
from tests.support import diagnostics_world as dw

P1 = d.PASS_1_NOT_SUBSCRIBE
ARMS = ("glass", "blackbox")
RATIOS = [m for m in sp.SUPPRESSION_LABELS if m not in sp.SUPPRESSION_COUNTS]


@pytest.fixture(scope="module")
def built():
    w = dw.world()
    ref, glass, rf, lr, ebm = dw.views(w)
    trace = d.build_router_trace(glass, rf, ref)
    refs = d.primary_reference_flags(lr, ebm)
    return {"w": w, "ref": ref, "glass": glass, "rf": rf, "lr": lr, "ebm": ebm, "trace": trace,
            "refs": refs, "report": d.pass1_suppression_report(trace, lr, ebm)}


# ----------------------------------------------------------------------
def _row(table, arm, fold, metric, reference=None):
    m = ((table["arm"] == arm) & (table["fold"].map(lambda f: f == fold))
         & (table["metric"] == metric))
    if reference is not None:
        m &= table["reference"] == reference
    rows = table[m]
    assert len(rows) == 1, (arm, fold, metric, reference)
    return rows.iloc[0]


def _masks(b, arm, fold):
    t = b["trace"]
    keep = np.ones(t.n, bool) if fold == "all" else np.asarray(t.fold_id) == fold
    return keep, np.asarray(t.y) == 1, np.asarray(t.views[arm].pass1)


def _expected(y_pos, p1, f, keep):
    pos, neg, p1, f = y_pos & keep, ~y_pos & keep, p1 & keep, f & keep
    return {
        "rows": (keep.sum(), None), "reference_flags": (f.sum(), None),
        "reference_tp": ((f & pos).sum(), None), "reference_fp": ((f & neg).sum(), None),
        "intercepted_flags": ((p1 & f).sum(), None),
        "intercepted_fp": ((p1 & f & neg).sum(), None),
        "suppressed_tp": ((p1 & f & pos).sum(), None),
        "retained_flags": ((f & ~p1).sum(), None), "retained_fp": ((f & ~p1 & neg).sum(), None),
        "retained_tp": ((f & ~p1 & pos).sum(), None),
        "pass1_positives_unflagged": ((p1 & pos & ~f).sum(), None),
        "upstream_flag_rate": (f.sum(), keep.sum()),
        "upstream_precision": ((f & pos).sum(), f.sum()),
        "pass1_fp_suppression": ((p1 & f & neg).sum(), (f & neg).sum()),
        "pass1_tp_cost": ((p1 & f & pos).sum(), (f & pos).sum()),
        "pass1_upstream_flag_rate": ((p1 & f).sum(), p1.sum()),
        "pass1_intercepted_flag_npv": ((p1 & f & neg).sum(), (p1 & f).sum()),
        "retained_upstream_precision": ((f & ~p1 & pos).sum(), (f & ~p1).sum()),
        "pass1_leak_upstream_flagged": ((p1 & pos & f).sum(), (p1 & pos).sum()),
        "pass1_leak_upstream_missed": ((p1 & pos & ~f).sum(), (p1 & pos).sum()),
    }


def _flags_like(flags, values):
    a = np.array(values, dtype=bool)
    a.setflags(write=False)
    return replace(flags, flags=a)


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(("|".join(map(str, a)) if a.dtype == object else
                  np.ascontiguousarray(a).tobytes().hex()).encode())
    return h.hexdigest()


# ======================================================================
# 1, 2, 4. interception, suppression, exact numerators and denominators
# ======================================================================
@pytest.mark.parametrize("key", sp.REFERENCE_KEYS)
def test_every_count_and_ratio_by_hand(built, key):
    table = built["report"].table
    f_all = np.asarray(built["refs"][key].flags)
    for arm in ARMS:
        for fold in ["all", *range(built["trace"].n_folds)]:
            keep, pos, p1 = _masks(built, arm, fold)
            for metric, (num, den) in _expected(pos, p1, f_all, keep).items():
                row = _row(table, arm, fold, metric, key)
                assert row["numerator"] == num, (key, arm, fold, metric)
                if den is None:
                    assert pd.isna(row["denominator"]) and row["status"] == "count"
                else:
                    assert row["denominator"] == den, (key, arm, fold, metric)
                    if den:
                        assert row["value"] == pytest.approx(num / den)
                        assert row["status"] == "defined"


def test_interception_and_suppression_partition_the_reference_flags(built):
    table = built["report"].table
    for key in sp.REFERENCE_KEYS:
        for arm in ARMS:
            for fold in ["all", *range(built["trace"].n_folds)]:
                v = {m: int(_row(table, arm, fold, m, key)["numerator"])
                     for m in sp.SUPPRESSION_COUNTS}
                assert v["reference_flags"] == v["reference_tp"] + v["reference_fp"]
                assert v["intercepted_flags"] == v["intercepted_fp"] + v["suppressed_tp"]
                assert v["reference_fp"] == v["intercepted_fp"] + v["retained_fp"]
                assert v["reference_tp"] == v["suppressed_tp"] + v["retained_tp"]
                assert v["reference_flags"] == v["intercepted_flags"] + v["retained_flags"]


def test_suppressed_true_positives_are_never_counted_as_benefit(built):
    table = built["report"].table
    assert "error" in sp.SUPPRESSION_COUNTS["suppressed_tp"]
    assert "benefit" in sp.SUPPRESSION_COUNTS["intercepted_fp"]
    # the interception rate counts negatives only; suppressed TPs have their own rate
    for arm in ARMS:
        fp = _row(table, arm, "all", "pass1_fp_suppression", "union")
        assert fp["numerator"] == _row(table, arm, "all", "intercepted_fp", "union")["numerator"]
        tp = _row(table, arm, "all", "pass1_tp_cost", "union")
        assert tp["numerator"] == _row(table, arm, "all", "suppressed_tp", "union")["numerator"]
    assert not any("net" in m for m in table["metric"].unique())


# ======================================================================
# 3. union without double counting; intersection as context
# ======================================================================
def test_union_counts_each_row_once(built):
    refs, table = built["refs"], built["report"].table
    lr, ebm = np.asarray(refs["lr"].flags), np.asarray(refs["ebm"].flags)
    np.testing.assert_array_equal(refs["union"].flags, lr | ebm)
    np.testing.assert_array_equal(refs["intersection"].flags, lr & ebm)
    assert built["report"]["union_reconciles"] is True
    for arm in ARMS:
        for fold in ["all", *range(built["trace"].n_folds)]:
            for m in ("reference_flags", "reference_fp", "intercepted_fp", "suppressed_tp",
                      "retained_fp", "retained_tp"):
                u, a, b, both = (int(_row(table, arm, fold, m, k)["numerator"])
                                 for k in ("union", "lr", "ebm", "intersection"))
                assert u == a + b - both, (arm, fold, m)
                assert max(a, b) <= u <= a + b
    assert (lr & ebm).any() and (lr & ~ebm).any()        # the world exercises the overlap


# ======================================================================
# 5. NPV and leakage
# ======================================================================
def test_pass1_population_matches_pass1_metrics(built):
    pop, t = d.pass1_population(built["trace"]), built["trace"]
    assert list(pop.columns) == ["arm", "fold", "metric", "label", "numerator", "denominator",
                                 "value", "status"]
    for arm in ARMS:
        v = t.views[arm]
        for fold in ["all", *range(t.n_folds)]:
            rows = None if fold == "all" else np.asarray(t.fold_id) == fold
            m = d.pass1_metrics(v, rows)
            for k in ("pass1_rate", "pass1_npv", "pass1_positive_rate", "pass1_leakage_share",
                      "pass1_negative_recall"):
                r = _row(pop, arm, fold, k)
                assert (r["numerator"], r["denominator"]) == (m[k].numerator, m[k].denominator)
            keep, pos, p1 = _masks(built, arm, fold)
            assert _row(pop, arm, fold, "pass1_rows")["numerator"] == (p1 & keep).sum()
            assert _row(pop, arm, fold, "pass1_negatives")["numerator"] == (p1 & keep & ~pos).sum()
            assert _row(pop, arm, fold, "pass1_positives")["numerator"] == (p1 & keep & pos).sum()
            npv = _row(pop, arm, fold, "pass1_npv")
            assert npv["denominator"] == (p1 & keep).sum()
            leak = _row(pop, arm, fold, "pass1_leakage_share")
            assert leak["denominator"] == (keep & pos).sum()          # ALL positives of the group


def test_report_carries_pooled_npv_and_leakage(built):
    for arm in ARMS:
        m = d.pass1_metrics(built["trace"].views[arm])
        assert built["report"][f"{arm}.pass1_npv"] == m["pass1_npv"]
        assert built["report"][f"{arm}.pass1_leakage_share"] == m["pass1_leakage_share"]


# ======================================================================
# 6-7. no intersection, empty populations, undefined denominators
# ======================================================================
def test_flags_outside_every_pass1_row_give_zero_interception(built):
    t = built["trace"]
    p1_any = np.asarray(t.views["glass"].pass1) | np.asarray(t.views["blackbox"].pass1)
    flags = _flags_like(built["refs"]["lr"], np.asarray(built["refs"]["lr"].flags) & ~p1_any)
    res = d.pass1_suppression(t, flags)
    for arm in ARMS:
        assert res[f"{arm}.intercepted_flags"] == 0
        assert res[f"{arm}.pass1_fp_suppression"].numerator == 0
        assert res[f"{arm}.pass1_fp_suppression"].status == "defined"
        assert res[f"{arm}.pass1_intercepted_flag_npv"].status == "undefined_empty_denominator"
        assert res[f"{arm}.pass1_intercepted_flag_npv"].value is None
        assert res[f"{arm}.retained_flags"] == res[f"{arm}.reference_flags"]
        assert res[f"{arm}.pass1_upstream_flag_rate"].value == 0.0


def test_a_reference_with_no_flags_is_undefined_not_zero(built):
    flags = _flags_like(built["refs"]["ebm"], np.zeros(built["trace"].n, bool))
    res = d.pass1_suppression(built["trace"], flags)
    for arm in ARMS:
        for m in ("pass1_fp_suppression", "pass1_tp_cost", "upstream_precision",
                  "retained_upstream_precision", "pass1_intercepted_flag_npv"):
            assert res[f"{arm}.{m}"].status == "undefined_empty_denominator", m
        rows = res.table[(res.table["arm"] == arm)
                         & res.table["metric"].isin(["pass1_fp_suppression", "pass1_tp_cost"])]
        assert rows["value"].isna().all()
        assert res[f"{arm}.pass1_leak_upstream_missed"].value == 1.0


def test_a_fold_without_pass1_rows_is_undefined(built):
    g = built["glass"]
    rows = np.flatnonzero((np.asarray(g.state) == P1) & (np.asarray(g.fold_id) == 0))
    dec = np.array(g.source_decision, dtype=object, copy=True)
    dec[rows] = "uncertain"
    g2 = replace(g, source_decision=dec,
                 state=np.array([d.SOURCE_DECISIONS[x][0] for x in dec], dtype=object),
                 source_pred=np.array([d.SOURCE_DECISIONS[x][1] for x in dec]))
    t = d.build_router_trace(g2, built["rf"], built["ref"])
    pop = d.pass1_population(t)
    assert _row(pop, "glass", 0, "pass1_rows")["numerator"] == 0
    assert _row(pop, "glass", 0, "pass1_npv")["status"] == "undefined_empty_denominator"
    assert _row(pop, "glass", "all", "pass1_npv")["status"] == "defined"
    res = d.pass1_suppression(t, built["refs"]["lr"])
    assert _row(res.table, "glass", 0, "pass1_upstream_flag_rate")["status"] == \
        "undefined_empty_denominator"
    assert _row(res.table, "glass", 0, "pass1_fp_suppression")["numerator"] == 0
    assert _row(res.table, "glass", 0, "pass1_fp_suppression")["status"] == "defined"


# ======================================================================
# 8. pooled vs per-fold, and parity with metrics.router_upstream_metrics
# ======================================================================
def test_every_numerator_and_denominator_sums_over_folds_to_pooled(built):
    for table in (built["report"].table, d.pass1_population(built["trace"])):
        keys = ["arm", "metric"] + (["reference"] if "reference" in table else [])
        for _, grp in table.groupby(keys):
            pooled = grp[grp["fold"].map(lambda f: f == "all")].iloc[0]
            folds = grp[grp["fold"].map(lambda f: f != "all")]
            assert len(folds) == built["trace"].n_folds
            assert int(folds["numerator"].sum()) == pooled["numerator"]
            if pooled["status"] != "count":
                assert int(folds["denominator"].sum()) == pooled["denominator"]


@pytest.mark.parametrize("key", sp.REFERENCE_KEYS)
@pytest.mark.parametrize("arm", ARMS)
def test_pooled_values_match_router_upstream_metrics(built, key, arm):
    theirs = d.router_upstream_metrics(built["trace"].views[arm], built["refs"][key])
    table = built["report"].table
    for m in ("upstream_flag_rate", "pass1_upstream_flag_rate", "pass1_fp_suppression",
              "pass1_tp_cost", "pass1_leak_upstream_flagged", "pass1_leak_upstream_missed"):
        r = _row(table, arm, "all", m, key)
        assert (r["numerator"], r["denominator"]) == (theirs[m].numerator, theirs[m].denominator)
    assert _row(table, arm, "all", "intercepted_fp", key)["numerator"] == \
        theirs["pass1_fp_suppressed_n"]
    assert _row(table, arm, "all", "suppressed_tp", key)["numerator"] == theirs["pass1_tp_cost_n"]


def test_rf_minus_glass_differences(built):
    rep = built["report"]
    for key in sp.REFERENCE_KEYS:
        for m in sp._DIFF_METRICS:
            a, b = rep[f"{key}.glass.{m}"], rep[f"{key}.blackbox.{m}"]
            got = rep[f"difference_pp.{key}.{m}"]
            if a.value is None or b.value is None:
                assert np.isnan(got)
            else:
                assert got == pytest.approx(100 * (b.value - a.value))


# ======================================================================
# 9. row alignment and label integrity
# ======================================================================
def test_flags_must_cover_the_trace_rows_and_labels(built):
    t, lr = built["trace"], built["refs"]["lr"]
    y = np.array(lr.y, copy=True)
    y[0] = 1 - y[0]
    cases = {
        "reordered": replace(lr, index=lr.index[::-1]),
        "labels": replace(lr, y=y),
        "short": _flags_like(lr, np.asarray(lr.flags)[:-1]),
        "not boolean": replace(lr, flags=np.asarray(lr.flags).astype(int)),
    }
    for name, flags in cases.items():
        with pytest.raises(d.RouterDiagnosticError):
            d.pass1_suppression(t, flags)


def test_a_reference_fold_plan_may_differ_from_the_routers(built):
    t, refs = built["trace"], built["refs"]
    assert not np.array_equal(refs["lr"].fold_id, t.fold_id)      # LR: 10 folds
    meta = built["report"].meta["references"]
    assert meta["lr"]["fold_plan"].startswith("10 folds, not the routers' 5")
    assert meta["ebm"]["fold_plan"] == "same as the routers' outer folds"
    assert meta["union"]["fold_plan"].startswith("combination of")
    table = built["report"].table
    for fold in range(t.n_folds):                                  # router folds, not LR's
        assert _row(table, "glass", fold, "rows", "lr")["numerator"] == \
            int((np.asarray(t.fold_id) == fold).sum())


# ======================================================================
# 10. registered operating points only, with their semantics
# ======================================================================
def test_primary_references_are_the_registered_points(built):
    refs, lr, ebm = built["refs"], built["lr"], built["ebm"]
    assert (refs["lr"].name, refs["ebm"].name) == ("lr.in_stage_f2_cv", "ebm.nested_f2")
    assert refs["lr"].score_space == "calibrated_probability"
    thr = lr.thresholds["lr.in_stage_f2_cv"].value
    assert refs["lr"].threshold == thr
    np.testing.assert_array_equal(refs["lr"].flags,
                                  np.asarray(lr.scores["calibrated_probability"].values) >= thr)
    assert refs["ebm"].score_space == "raw_model_score" and refs["ebm"].threshold is None
    np.testing.assert_array_equal(refs["ebm"].flags,
                                  np.asarray(ebm.stored_decisions["ebm.nested_f2"].values) == 1)
    assert refs["ebm"].selection == d.OPERATING_POINTS["ebm.nested_f2"].selection
    for k in ("union", "intersection"):
        assert refs[k].components == ("lr.in_stage_f2_cv", "ebm.nested_f2")
        assert refs[k].score_space is None


def test_only_registered_operating_points_are_accepted(built):
    t, lr = built["trace"], built["refs"]["lr"]
    with pytest.raises(d.RouterDiagnosticError, match="UpstreamFlags"):
        d.pass1_suppression(t, np.asarray(lr.flags))
    for comps in (("custom.threshold",), ()):
        with pytest.raises(d.RouterDiagnosticError, match="registered"):
            d.pass1_suppression(t, replace(lr, components=comps))


def test_reference_flag_summary(built):
    s = d.reference_flag_summary(built["trace"], built["refs"]).set_index("reference")
    y = np.asarray(built["trace"].y) == 1
    for k in sp.REFERENCE_KEYS:
        f = np.asarray(built["refs"][k].flags)
        assert (s.loc[k, "flags"], s.loc[k, "tp"], s.loc[k, "fp"]) == (f.sum(), (f & y).sum(),
                                                                       (f & ~y).sum())
        assert s.loc[k, "precision"] == pytest.approx((f & y).sum() / f.sum())
    assert s.loc["lr", "score_space"] == "calibrated_probability"
    assert s.loc["ebm", "score_space"] == "raw_model_score"


def test_every_reported_rate_is_a_registered_metric(built):
    ratios = set(built["report"].table.query("status != 'count'")["metric"])
    ratios |= set(d.pass1_population(built["trace"]).query("status != 'count'")["metric"])
    assert ratios <= set(d.METRIC_DEFINITIONS)
    for new in ("upstream_precision", "pass1_intercepted_flag_npv", "retained_upstream_precision"):
        assert new in d.METRIC_DEFINITIONS
    assert "hypothetical" in d.METRIC_DEFINITIONS["retained_upstream_precision"][1]
    assert set(RATIOS) <= set(d.METRIC_DEFINITIONS)


# ======================================================================
# disclosures
# ======================================================================
def test_report_notes_carry_every_disclosure(built):
    notes = built["report"].notes
    assert notes[0] == d.HISTORICAL_TEST_EXPOSURE
    text = " ".join(notes)
    for needle in ("In-selection", "fold-nested", "is an error", "never netted",
                   "not the integrated cascade", "hypothetical Pass 1 veto", "Common reference",
                   "10 folds", "match_glass_oof", "max_subscriber_leakage_rate", "Commit 4",
                   "Step 2C", "Step 2D", "no significance test"):
        assert needle in text, needle


@pytest.mark.parametrize("source, needle", [
    ("other_source", "constraint_source='other_source'"),
    (None, "constraint_source is not recorded"),
])
def test_dependence_note_follows_the_recorded_constraint_source(built, source, needle):
    meta = dict(built["rf"].metadata)
    if source is None:
        meta.pop("constraint_source")
    else:
        meta["constraint_source"] = source
    t = d.build_router_trace(built["glass"], replace(built["rf"], metadata=MappingProxyType(meta)),
                             built["ref"])
    text = " ".join(d.pass1_suppression_report(t, built["lr"], built["ebm"]).notes)
    assert needle in text and "match_glass_oof" not in text


# ======================================================================
# 11. non-mutation
# ======================================================================
def test_no_view_or_flag_set_is_mutated(built):
    t, lr, ebm, refs = built["trace"], built["lr"], built["ebm"], built["refs"]

    def snapshot():
        return (_digest(*(getattr(t.views[a], f) for a in ARMS
                          for f in ("state", "source_decision", "y", "fold_id"))),
                _digest(lr.scores["calibrated_probability"].values, lr.y, lr.fold_id,
                        ebm.scores["raw_model_score"].values,
                        ebm.stored_decisions["ebm.nested_f2"].values),
                _digest(*(refs[k].flags for k in sp.REFERENCE_KEYS)),
                lr.thresholds["lr.in_stage_f2_cv"].value)

    before = snapshot()
    d.pass1_population(t)
    d.reference_flag_summary(t, refs)
    rep = d.pass1_suppression_report(t, lr, ebm)
    rep.table.loc[:, "numerator"] = 0
    assert snapshot() == before
    for k in sp.REFERENCE_KEYS:
        assert not np.asarray(refs[k].flags).flags.writeable
    with pytest.raises(TypeError):
        rep.meta["references"]["lr"] = None


@pytest.mark.parametrize("fn", [d.pass1_population,
                                lambda x: d.pass1_suppression(x, None),
                                lambda x: d.reference_flag_summary(x, {})])
def test_functions_refuse_anything_but_a_trace(built, fn):
    with pytest.raises(d.RouterDiagnosticError, match="RouterTrace"):
        fn(built["glass"])
