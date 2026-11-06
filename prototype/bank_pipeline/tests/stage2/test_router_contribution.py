"""
PR 37 Commit 2 (Step 2C): Pass 2 unique-positive contribution of
shared.stage2.diagnostics.contribution on the synthetic Commit 1 world.

Mechanics only: masks, numerators and denominators, the reference breakdown,
the common eligible population, the label-free rate-matched selection (exact
per-fold counts, tie bounds, tie-break rule), undefined ratios, alignment and
read-only behaviour. Historical values: tests/realdata/test_router_contribution_artifacts.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import contribution as cb
from shared.stage2.diagnostics.schema import ScoreStream
from tests.support import diagnostics_world as dw

P1, P2 = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE
ARMS = ("glass", "blackbox")
FOLDS = ["all", 0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def built():
    w = dw.world()
    ref, glass, rf, lr, ebm = dw.views(w)
    trace = d.build_router_trace(glass, rf, ref)
    return {"w": w, "ref": ref, "glass": glass, "rf": rf, "lr": lr, "ebm": ebm, "trace": trace,
            "refs": d.primary_reference_flags(lr, ebm),
            "frozen": d.pass2_frozen_contribution(trace, lr, ebm),
            "matched": d.pass2_matched_contribution(trace, lr, ebm)}


# ----------------------------------------------------------------------
def _row(table, arm, fold, metric, reference=None):
    m = ((table["arm"] == arm) & (table["fold"].map(lambda f: f == fold))
         & (table["metric"] == metric))
    if reference is not None:
        m &= table["reference"] == reference
    rows = table[m]
    assert len(rows) == 1, (arm, fold, metric, reference)
    return rows.iloc[0]


def _group(trace, fold):
    return np.ones(trace.n, bool) if fold == "all" else np.asarray(trace.fold_id) == fold


def _expected(y, p2, f):
    pos, neg, u = y, ~y, p2 | f
    q = p2 & ~f
    return {
        "rows": (len(y), None), "positives": (pos.sum(), None), "pass2_n": (p2.sum(), None),
        "pass2_tp": ((p2 & pos).sum(), None), "pass2_fp": ((p2 & neg).sum(), None),
        "reference_flags": (f.sum(), None), "reference_tp": ((f & pos).sum(), None),
        "reference_fp": ((f & neg).sum(), None), "overlap_flags": ((p2 & f).sum(), None),
        "overlap_tp": ((p2 & f & pos).sum(), None), "overlap_fp": ((p2 & f & neg).sum(), None),
        "unique_tp": ((q & pos).sum(), None), "unique_fp": ((q & neg).sum(), None),
        "reference_only_tp": ((f & ~p2 & pos).sum(), None),
        "reference_only_fp": ((f & ~p2 & neg).sum(), None),
        "union_flags": (u.sum(), None), "union_tp": ((u & pos).sum(), None),
        "union_fp": ((u & neg).sum(), None),
        "pass2_rate": (p2.sum(), len(y)), "upstream_flag_rate": (f.sum(), len(y)),
        "pass2_precision": ((p2 & pos).sum(), p2.sum()),
        "upstream_precision": ((f & pos).sum(), f.sum()),
        "pass2_recall": ((p2 & pos).sum(), pos.sum()), "upstream_recall": ((f & pos).sum(), pos.sum()),
        "pass2_upstream_redundancy": ((p2 & f).sum(), p2.sum()),
        "pass2_shared_positive_share": ((p2 & f & pos).sum(), (p2 & pos).sum()),
        "pass2_unique_positive_share": ((q & pos).sum(), (p2 & pos).sum()),
        "pass2_new_false_positive_rate": ((q & neg).sum(), p2.sum()),
        "incremental_positive_recall": ((q & pos).sum(), pos.sum()),
        "incremental_yield": ((q & pos).sum(), p2.sum()),
        "marginal_precision": ((q & pos).sum(), q.sum()),
        "union_precision": ((u & pos).sum(), u.sum()),
        "jaccard_with_router": ((p2 & f).sum(), u.sum()),
    }


def _check(row, num, den):
    assert row["numerator"] == num
    if den is None:
        assert pd.isna(row["denominator"]) and row["status"] == "count"
    else:
        assert row["denominator"] == den
        if den:
            assert row["value"] == pytest.approx(num / den) and row["status"] == "defined"
        else:
            assert np.isnan(row["value"]) and row["status"] == "undefined_empty_denominator"


def _reroute(view, rows, token):
    dec = np.array(view.source_decision, dtype=object, copy=True)
    dec[rows] = token
    return replace(view, source_decision=dec,
                   state=np.array([d.SOURCE_DECISIONS[t][0] for t in dec], dtype=object),
                   source_pred=np.array([d.SOURCE_DECISIONS[t][1] for t in dec]))


def _with_score(view, space, values):
    old = view.scores[space]
    a = np.array(values, dtype=float)
    a.setflags(write=False)
    scores = dict(view.scores)
    scores[space] = ScoreStream(old.name, a, old.space, old.field, old.provenance)
    return replace(view, scores=MappingProxyType(scores))


def _reference_top_k(score, pop, fold, target, index):
    """Independent re-implementation: per fold, sort by (-score, sha256(label)), take K."""
    key = np.array([int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:8], "big")
                    for x in index], dtype=np.uint64)
    out = np.zeros(len(score), bool)
    for k, n in target.items():
        idx = [i for i in np.flatnonzero(pop & (fold == k))]
        idx.sort(key=lambda i: (-score[i], int(key[i])))
        out[idx[:n]] = True
    return out


# ======================================================================
# 1. Pass 2 population
# ======================================================================
def test_pass2_population_by_hand_and_against_pass2_metrics(built):
    t = built["trace"]
    pop = d.pass2_population(t)
    y = np.asarray(t.y) == 1
    for arm in ARMS:
        v = t.views[arm]
        for fold in FOLDS:
            k = _group(t, fold)
            p2, p1 = np.asarray(v.pass2) & k, np.asarray(v.pass1) & k
            rem = k & ~np.asarray(v.pass1)
            exp = {"rows": (k.sum(), None), "positives": ((k & y).sum(), None),
                   "pass2_n": (p2.sum(), None), "pass2_tp": ((p2 & y).sum(), None),
                   "pass2_fp": ((p2 & ~y).sum(), None), "remainder_rows": (rem.sum(), None),
                   "remainder_positives": ((rem & y).sum(), None),
                   "pass2_rate": (p2.sum(), k.sum()), "pass2_precision": ((p2 & y).sum(), p2.sum()),
                   "pass2_recall": ((p2 & y).sum(), (k & y).sum()),
                   "pass2_remainder_rate": (p2.sum(), rem.sum()),
                   "pass2_recall_eligible": ((p2 & y).sum(), (rem & y).sum())}
            for metric, (num, den) in exp.items():
                _check(_row(pop, arm, fold, metric), num, den)
            m = d.pass2_metrics(v, None if fold == "all" else k)
            for metric in ("pass2_rate", "pass2_precision", "pass2_recall", "pass2_recall_eligible"):
                r = _row(pop, arm, fold, metric)
                assert (r["numerator"], r["denominator"]) == (m[metric].numerator,
                                                              m[metric].denominator)


# ======================================================================
# 2-3, 6-7. frozen comparison: every count and ratio by hand
# ======================================================================
@pytest.mark.parametrize("key", d.suppression.REFERENCE_KEYS)
def test_frozen_contribution_by_hand(built, key):
    t, table = built["trace"], built["frozen"].table
    f_all = np.asarray(built["refs"][key].flags)
    y_all = np.asarray(t.y) == 1
    assert set(table["population"]) == {cb.POP_ALL}
    for arm in ARMS:
        p2_all = np.asarray(t.views[arm].pass2)
        for fold in FOLDS:
            k = _group(t, fold)
            for metric, (num, den) in _expected(y_all[k], p2_all[k], f_all[k]).items():
                _check(_row(table, arm, fold, metric, key), num, den)


def test_hypothetical_or_increments_equal_the_unique_counts(built):
    table = built["frozen"].table
    for key in d.suppression.REFERENCE_KEYS:
        for arm in ARMS:
            for fold in FOLDS:
                v = {m: int(_row(table, arm, fold, m, key)["numerator"]) for m in
                     ("union_tp", "union_fp", "reference_tp", "reference_fp", "unique_tp",
                      "unique_fp", "pass2_tp", "pass2_fp", "overlap_tp", "overlap_fp",
                      "reference_only_tp", "union_flags", "reference_flags")}
                assert v["union_tp"] - v["reference_tp"] == v["unique_tp"]
                assert v["union_fp"] - v["reference_fp"] == v["unique_fp"]
                assert v["pass2_tp"] == v["overlap_tp"] + v["unique_tp"]
                assert v["pass2_fp"] == v["overlap_fp"] + v["unique_fp"]
                assert v["reference_tp"] == v["overlap_tp"] + v["reference_only_tp"]
                assert v["union_flags"] == v["reference_flags"] + v["unique_tp"] + v["unique_fp"]


def test_frozen_pooled_values_match_router_upstream_metrics(built):
    for key in d.suppression.REFERENCE_KEYS:
        for arm in ARMS:
            theirs = d.router_upstream_metrics(built["trace"].views[arm], built["refs"][key])
            mine = built["frozen"]
            for m in ("pass2_upstream_redundancy", "pass2_unique_positive_share",
                      "pass2_shared_positive_share", "pass2_new_false_positive_rate"):
                assert mine[f"{key}.{arm}.{m}"] == theirs[m], (key, arm, m)
            assert mine[f"{key}.{arm}.unique_tp"] == theirs["pass2_unique_positives_n"]


def test_rf_minus_glass_differences(built):
    fz = built["frozen"]
    for key in d.suppression.REFERENCE_KEYS:
        assert fz[f"difference.{key}.unique_tp"] == (fz[f"{key}.blackbox.unique_tp"]
                                                     - fz[f"{key}.glass.unique_tp"])
        a, b = (fz[f"{key}.{arm}.incremental_positive_recall"] for arm in ARMS)
        assert fz[f"difference_pp.{key}.incremental_positive_recall"] == pytest.approx(
            100 * (b.value - a.value))


# ======================================================================
# 4-5. positives missed by both references; no double counting
# ======================================================================
def test_breakdown_partitions_pass2_and_matches_the_union(built):
    t, refs = built["trace"], built["refs"]
    bd = d.pass2_positive_breakdown(t, refs)
    lr, ebm = np.asarray(refs["lr"].flags), np.asarray(refs["ebm"].flags)
    y = np.asarray(t.y) == 1
    for arm in ARMS:
        p2_all = np.asarray(t.views[arm].pass2)
        for fold in FOLDS:
            k = _group(t, fold)
            r = bd[(bd["arm"] == arm) & bd["fold"].map(lambda f: f == fold)].iloc[0]
            p2 = p2_all & k
            assert r["tp_both"] == (p2 & y & lr & ebm).sum()
            assert r["tp_lr_only"] == (p2 & y & lr & ~ebm).sum()
            assert r["tp_ebm_only"] == (p2 & y & ~lr & ebm).sum()
            assert r["tp_neither"] == (p2 & y & ~lr & ~ebm).sum()
            assert r["pass2_tp"] == r[["tp_both", "tp_lr_only", "tp_ebm_only", "tp_neither"]].sum()
            assert r["pass2_fp"] == r[["fp_both", "fp_lr_only", "fp_ebm_only", "fp_neither"]].sum()
            fz = built["frozen"].table
            assert r["tp_neither"] == _row(fz, arm, fold, "unique_tp", "union")["numerator"]
            assert r["fp_neither"] == _row(fz, arm, fold, "unique_fp", "union")["numerator"]
            assert r["tp_both"] == _row(fz, arm, fold, "overlap_tp", "intersection")["numerator"]


def test_unique_vs_union_never_exceeds_unique_vs_either_reference(built):
    fz = built["frozen"]
    for arm in ARMS:
        u = fz[f"union.{arm}.unique_tp"]
        assert u <= min(fz[f"lr.{arm}.unique_tp"], fz[f"ebm.{arm}.unique_tp"])
        assert fz[f"intersection.{arm}.unique_tp"] >= max(fz[f"lr.{arm}.unique_tp"],
                                                          fz[f"ebm.{arm}.unique_tp"])


# ======================================================================
# 8. frozen operating points are the Step 2B references, unchanged
# ======================================================================
def test_frozen_comparison_uses_the_step_2b_reference_flags(built):
    t, refs, fz = built["trace"], built["refs"], built["frozen"]
    summary = d.reference_flag_summary(t, refs).set_index("reference")
    for key in d.suppression.REFERENCE_KEYS:
        assert fz[f"{key}.glass.reference_flags"] == summary.loc[key, "flags"]
    assert refs["lr"].threshold == built["lr"].thresholds["lr.in_stage_f2_cv"].value
    np.testing.assert_array_equal(refs["ebm"].flags,
                                  np.asarray(built["ebm"].stored_decisions["ebm.nested_f2"].values) == 1)
    text = " ".join(fz.notes)
    for needle in ("In-selection", "fold-nested", "Hypothetical OR", "not Stage 4", "own flag rate",
                   "Common reference", "F17", "match_glass_oof", "no significance test",
                   "never across the two router"):
        assert needle in text, needle
    assert fz.notes[0] == d.HISTORICAL_TEST_EXPOSURE


# ======================================================================
# 9. common eligible population
# ======================================================================
def test_common_eligible_population(built):
    t = built["trace"]
    g, r = t.views["glass"], t.views["blackbox"]
    mask = d.common_eligible_mask(t)
    np.testing.assert_array_equal(mask, ~np.asarray(g.pass1) & ~np.asarray(r.pass1))
    assert not mask.flags.writeable
    cp = d.common_eligible_population(t).set_index("fold")
    assert cp.loc["all", "common_rows"] == mask.sum()
    assert cp.loc["all", "glass_pass2_in_common"] + cp.loc["all", "glass_pass2_outside_common"] \
        == g.pass2.sum()
    folds = cp.drop(index="all")
    for c in cp.columns:
        assert folds[c].sum() == cp.loc["all", c]


def test_pass2_rows_routed_pass1_by_the_other_router_leave_the_common_population(built):
    t0 = built["trace"]
    g = t0.views["glass"]
    moved = np.flatnonzero(np.asarray(g.pass2))[:7]
    rf2 = _reroute(t0.views["blackbox"], moved, "pass1")
    t = d.build_router_trace(g, rf2, built["ref"])
    cp = d.common_eligible_population(t).set_index("fold")
    assert cp.loc["all", "glass_pass2_outside_common"] == 7
    sel = d.matched_reference_selection(t, "glass", built["lr"])
    assert sel.n_flagged == g.pass2.sum() - 7 and not (sel.flags & ~np.asarray(sel.population)).any()
    res = d.pass2_matched_contribution(t, built["lr"], built["ebm"])
    assert res["lr_matched.glass.pass2_n"] == g.pass2.sum() - 7
    assert "7 GLASS Pass 2 rows that RF routed to Pass 1" in " ".join(res.notes)


# ======================================================================
# 10-12. label-free, exact per-fold matching, stable tie-breaking
# ======================================================================
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("model", ["lr", "ebm"])
def test_matched_selection_counts_equal_the_router_per_fold(built, arm, model):
    t = built["trace"]
    sel = d.matched_reference_selection(t, arm, built[model])
    fold = np.asarray(t.fold_id)
    router = np.asarray(t.views[arm].pass2) & np.asarray(d.common_eligible_mask(t))
    np.testing.assert_array_equal(sel.router_set, router)
    for k in range(t.n_folds):
        assert int((sel.flags & (fold == k)).sum()) == int((router & (fold == k)).sum())
        assert sel.by_fold[k]["target_count"] == int((router & (fold == k)).sum())
    assert not (sel.flags & ~np.asarray(sel.population)).any()
    assert sel.score_space == cb.MATCH_SCORES[model]
    target = {k: int((router & (fold == k)).sum()) for k in range(t.n_folds)}
    score = np.asarray(built[model].scores[cb.MATCH_SCORES[model]].values)
    np.testing.assert_array_equal(sel.flags, _reference_top_k(score, np.asarray(sel.population),
                                                              fold, target, t.index))


def test_matched_selection_never_reads_labels(built):
    t, lr = built["trace"], built["lr"]
    before = d.matched_reference_selection(t, "glass", lr)
    rng = np.random.default_rng(7)
    y2 = rng.permutation(np.asarray(t.y))
    y2.setflags(write=False)
    t2 = replace(t, y=y2)
    after = d.matched_reference_selection(t2, "glass", replace(lr, y=y2))
    np.testing.assert_array_equal(before.flags, after.flags)
    np.testing.assert_array_equal(before.strict, after.strict)
    np.testing.assert_array_equal(before.inclusive, after.inclusive)


def test_tied_scores_get_exact_counts_bounds_and_a_stable_tie_break(built):
    t, lr = built["trace"], built["lr"]
    tied = np.round(np.asarray(lr.scores["calibrated_probability"].values), 1)
    lr_t = _with_score(lr, "calibrated_probability", tied)
    a = d.matched_reference_selection(t, "glass", lr_t)
    b = d.matched_reference_selection(t, "glass", lr_t)
    np.testing.assert_array_equal(a.flags, b.flags)                       # deterministic
    assert (np.asarray(a.strict) & ~np.asarray(a.flags)).sum() == 0          # strict <= exact
    assert (np.asarray(a.flags) & ~np.asarray(a.inclusive)).sum() == 0       # exact <= inclusive
    broken = sum(int(v["tie_broken_rows"]) for v in a.by_fold.values())
    assert broken > 0 and a.n_flagged == int(np.asarray(a.router_set).sum())
    assert any(not v["exact"] for v in a.by_fold.values())
    fold = np.asarray(t.fold_id)
    target = {k: int(a.by_fold[k]["target_count"]) for k in range(t.n_folds)}
    np.testing.assert_array_equal(a.flags, _reference_top_k(tied, np.asarray(a.population), fold,
                                                            target, t.index))
    res = d.pass2_matched_contribution(t, lr_t, built["ebm"])
    tab = res.table
    for fold_ in FOLDS:
        r = {m: int(_row(tab, "glass", fold_, m, "lr_matched")["numerator"])
             for m in ("matched_strict_n", "matched_inclusive_n", "reference_flags",
                       "unique_tp_strict", "unique_tp_inclusive", "unique_tp")}
        assert r["matched_strict_n"] <= r["reference_flags"] <= r["matched_inclusive_n"]
        assert r["unique_tp_inclusive"] <= r["unique_tp"] <= r["unique_tp_strict"]


def test_tie_break_key_is_the_row_label_hash(built):
    idx = built["trace"].index
    key = cb._tie_key(idx[:5])
    for i, x in enumerate(idx[:5]):
        assert int(key[i]) == int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:8], "big")


# ======================================================================
# matched contribution values
# ======================================================================
def test_matched_contribution_by_hand(built):
    t, res = built["trace"], built["matched"]
    sels = res.meta["selections"]
    pop = np.asarray(d.common_eligible_mask(t))
    y_all = np.asarray(t.y) == 1
    assert set(res.table["population"]) == {cb.POP_COMMON}
    for arm in ARMS:
        p2_all = np.asarray(t.views[arm].pass2) & pop
        flag_sets = {"lr_matched": np.asarray(sels[arm]["lr"].flags),
                     "ebm_matched": np.asarray(sels[arm]["ebm"].flags)}
        flag_sets["lr_or_ebm_matched"] = flag_sets["lr_matched"] | flag_sets["ebm_matched"]
        for key, f_all in flag_sets.items():
            for fold in FOLDS:
                k = _group(t, fold) & pop
                for metric, (num, den) in _expected(y_all[k], p2_all[k], f_all[k]).items():
                    _check(_row(res.table, arm, fold, metric, key), num, den)
                full = _group(t, fold) & y_all
                q = p2_all & ~f_all & y_all & k
                _check(_row(res.table, arm, fold, "incremental_positive_recall_all", key),
                       q.sum(), full.sum())
            if key != "lr_or_ebm_matched":
                assert res[f"{key}.{arm}.pass2_n"] == res[f"{key}.{arm}.reference_flags"]
            assert res[f"{key}.{arm}.pass2_minus_reference_tp"] == (
                res[f"{key}.{arm}.pass2_tp"] - res[f"{key}.{arm}.reference_tp"])
    text = " ".join(res.notes)
    for needle in ("common eligible population", "not either router's fit-time",
                   "No label is read", "SHA-256", "strict / inclusive", "descriptive "
                   "operating-rate control", "Arm-specific", "size is not matched",
                   "isotonic" if "isotonic" in sels["glass"]["lr"].score_provenance else "calibration",
                   "ranked only within its own fold", "never ranked or truncated",
                   "match_glass_oof"):
        assert needle in text, needle


# ======================================================================
# 13-14. unavailable scores, empty populations, undefined ratios
# ======================================================================
def test_unavailable_or_invalid_scores_are_refused(built):
    t, lr, ebm = built["trace"], built["lr"], built["ebm"]
    no_raw = replace(ebm, scores=MappingProxyType({k: v for k, v in ebm.scores.items()
                                                   if k != "raw_model_score"}))
    with pytest.raises(d.UnsupportedDiagnostic, match="raw_model_score"):
        d.matched_reference_selection(t, "glass", no_raw)
    pop = np.flatnonzero(np.asarray(d.common_eligible_mask(t)))
    s = np.array(lr.scores["calibrated_probability"].values, copy=True)
    s[pop[0]] = np.nan
    with pytest.raises(d.UnsupportedDiagnostic, match="NaN"):
        d.matched_reference_selection(t, "glass", _with_score(lr, "calibrated_probability", s))
    with pytest.raises(d.RouterDiagnosticError):
        d.matched_reference_selection(t, "glass", built["glass"])
    with pytest.raises(d.RouterDiagnosticError, match="arm"):
        d.matched_reference_selection(t, "rf", lr)
    with pytest.raises(d.RouterDiagnosticError, match="rows / labels"):
        d.matched_reference_selection(t, "glass", replace(lr, index=lr.index[::-1]))


def test_a_fold_without_pass2_rows_matches_zero_and_is_undefined(built):
    g = built["glass"]
    rows = np.flatnonzero((np.asarray(g.state) == P2) & (np.asarray(g.fold_id) == 0))
    t = d.build_router_trace(_reroute(g, rows, "uncertain"), built["rf"], built["ref"])
    sel = d.matched_reference_selection(t, "glass", built["ebm"])
    assert sel.by_fold[0]["target_count"] == 0 and sel.by_fold[0]["cut_value"] is None
    assert int((sel.flags & (np.asarray(t.fold_id) == 0)).sum()) == 0
    res = d.pass2_matched_contribution(t, built["lr"], built["ebm"])
    for m in ("pass2_precision", "marginal_precision", "upstream_precision"):
        r = _row(res.table, "glass", 0, m, "ebm_matched")
        assert r["status"] == "undefined_empty_denominator" and np.isnan(r["value"])
    assert _row(res.table, "glass", "all", "pass2_precision", "ebm_matched")["status"] == "defined"
    pop = d.pass2_population(t)
    assert _row(pop, "glass", 0, "pass2_precision")["status"] == "undefined_empty_denominator"


def test_a_frozen_reference_with_no_flags(built):
    flags = replace(built["refs"]["lr"], flags=np.zeros(built["trace"].n, bool))
    res = d.pass2_contribution(built["trace"], flags)
    for arm in ARMS:
        assert res[f"{arm}.unique_tp"] == res[f"{arm}.pass2_tp"]
        assert res[f"{arm}.upstream_precision"].status == "undefined_empty_denominator"
        assert res[f"{arm}.pass2_upstream_redundancy"].value == 0.0


# ======================================================================
# 15-16. alignment, reconciliation
# ======================================================================
def test_frozen_flags_must_align_with_the_trace(built):
    lr = built["refs"]["lr"]
    for bad in (replace(lr, index=lr.index[::-1]), replace(lr, components=("custom",))):
        with pytest.raises(d.RouterDiagnosticError):
            d.pass2_contribution(built["trace"], bad)


def test_every_numerator_and_denominator_sums_over_folds_to_pooled(built):
    tables = (built["frozen"].table, built["matched"].table, d.pass2_population(built["trace"]))
    for table in tables:
        keys = [c for c in ("arm", "reference", "metric") if c in table]
        for _, grp in table.groupby(keys):
            pooled = grp[grp["fold"].map(lambda f: f == "all")].iloc[0]
            folds = grp[grp["fold"].map(lambda f: f != "all")]
            assert len(folds) == 5
            assert int(folds["numerator"].sum()) == pooled["numerator"], tuple(pooled[keys])
            if pooled["status"] != "count":
                assert int(folds["denominator"].sum()) == pooled["denominator"]
    bd = d.pass2_positive_breakdown(built["trace"], built["refs"])
    for arm in ARMS:
        a = bd[bd["arm"] == arm]
        pooled = a[a["fold"].map(lambda f: f == "all")].iloc[0]
        folds = a[a["fold"].map(lambda f: f != "all")]
        for c in bd.columns[2:]:
            assert folds[c].sum() == pooled[c]


def test_every_reported_rate_is_a_registered_metric(built):
    ratios = set()
    for table in (built["frozen"].table, built["matched"].table, d.pass2_population(built["trace"])):
        ratios |= set(table.loc[table["status"] != "count", "metric"])
    assert ratios <= set(d.METRIC_DEFINITIONS)
    for new in ("upstream_recall", "pass2_remainder_rate", "incremental_positive_recall",
                "incremental_positive_recall_all", "incremental_yield", "marginal_precision",
                "union_precision"):
        assert new in d.METRIC_DEFINITIONS
    for kept in ("pass2_precision", "pass2_recall", "pass2_unique_positive_share",
                 "upstream_precision", "pass1_fp_suppression", "coverage"):
        assert kept in d.METRIC_DEFINITIONS
    assert set(ratios) | {"rows"} <= set(cb.CONTRIBUTION_LABELS) | {"rows"}


# ======================================================================
# 17-18. non-mutation; Steps 2A / 2B unaffected
# ======================================================================
def test_nothing_is_mutated_and_selections_are_read_only(built):
    t, lr, ebm, refs = built["trace"], built["lr"], built["ebm"], built["refs"]

    def snap():
        h = hashlib.sha256()
        for a in (lr.scores["calibrated_probability"].values, ebm.scores["raw_model_score"].values,
                  ebm.stored_decisions["ebm.nested_f2"].values, t.y, t.fold_id,
                  *(refs[k].flags for k in refs)):
            h.update(np.ascontiguousarray(a).tobytes())
        for arm in ARMS:
            h.update("|".join(map(str, t.views[arm].state)).encode())
        return h.hexdigest()

    before = snap()
    d.pass2_population(t), d.common_eligible_population(t), d.pass2_positive_breakdown(t, refs)
    res = d.pass2_matched_contribution(t, lr, ebm)
    d.pass2_frozen_contribution(t, lr, ebm)
    assert snap() == before
    sel = res.meta["selections"]["glass"]["lr"]
    for arr in (sel.flags, sel.strict, sel.inclusive, sel.population, sel.router_set):
        assert not np.asarray(arr).flags.writeable
    with pytest.raises(TypeError):
        res.meta["selections"]["glass"] = None
    with pytest.raises(TypeError):
        sel.by_fold[0]["target_count"] = 1


def test_steps_2a_and_2b_are_unaffected(built):
    t = built["trace"]
    assert t.check_frame()["passed"].all()
    rep = d.pass1_suppression_report(t, built["lr"], built["ebm"])
    assert rep["union_reconciles"] is True
    assert d.compare_state_distributions(t)["difference.PASS_2_SUBSCRIBE.n"] == int(
        t.views["blackbox"].pass2.sum() - t.views["glass"].pass2.sum())


@pytest.mark.parametrize("fn", [d.pass2_population, d.common_eligible_population,
                                lambda x: d.pass2_matched_contribution(x, None, None),
                                lambda x: d.pass2_frozen_contribution(x, None, None)])
def test_functions_refuse_anything_but_a_trace(built, fn):
    with pytest.raises(d.RouterDiagnosticError, match="RouterTrace"):
        fn(built["glass"])
