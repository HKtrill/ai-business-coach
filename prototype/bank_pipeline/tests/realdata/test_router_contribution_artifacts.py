"""
PR 37 Commit 2 (Step 2C): Pass 2 unique-positive contribution on the SAVED
October 23 artifacts - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_router_contribution_artifacts.py

Both routers' Pass 2 against the frozen primary LR / EBM flags (and their union
and intersection) on all training rows, and against label-free rate-matched LR
/ EBM flags on the common eligible population. The anchors are the PR 37
handoff's and the validated Steps 2A / 2B results: expectations only.

Nothing is fitted, written or regenerated; no test label is read (loader:
tests.support.router_artifacts). Every artifact file read is hashed before
its first load and again at the end; with ``PR37_BASELINE_SHA256SUMS`` set,
the router, LR and EBM files must also equal the preserved baseline.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import contribution as cb
from tests.support.router_artifacts import Real

pytestmark = pytest.mark.realdata

#: Anchors (handoff, Step 2A trace, Step 2B reference flags).
N_TRAIN, N_POSITIVES = 32_950, 3_712
PASS2 = {"glass": {"n": 8_468, "tp": 2_337, "fp": 6_131},
         "blackbox": {"n": 9_427, "tp": 2_560, "fp": 6_867}}
REFERENCE_FLAGS = {"lr": 7_727, "ebm": 6_113, "union": 8_001, "intersection": 5_839}
ARMS = ("glass", "blackbox")


@pytest.fixture(scope="module")
def run():
    """Each file is hashed before its first load; everything is built once."""
    real = Real()
    before: dict = {}

    def remember(p) -> Path:
        p = Path(p)
        before.setdefault(p.resolve(), d.file_sha256(p))
        return p

    for k in ("meta_ebm", "meta_ebm_on_blackbox"):
        remember(real.roots()[k])
    native = {s: remember(real.paths("meta_ebm")[s]) for s in ("stage1", "stage2", "stage3")}
    rf_p = remember(real.paths("meta_ebm_on_blackbox")["stage2"])
    ref = real.ref()
    glass, rf, lr, ebm = (real.view(w) for w in ("glass", "rf", "lr", "ebm"))
    trace = d.build_router_trace(glass, rf, ref)
    return SimpleNamespace(
        real=real, ref=ref, glass=glass, rf=rf, lr=lr, ebm=ebm, trace=trace,
        refs=d.primary_reference_flags(lr, ebm), population=d.pass2_population(trace),
        frozen=d.pass2_frozen_contribution(trace, lr, ebm),
        matched=d.pass2_matched_contribution(trace, lr, ebm),
        before=before, attested=(*native.values(), rf_p))


def _row(table, arm, fold, metric, reference=None):
    m = ((table["arm"] == arm) & (table["fold"].map(lambda f: f == fold))
         & (table["metric"] == metric))
    if reference is not None:
        m &= table["reference"] == reference
    rows = table[m]
    assert len(rows) == 1, (arm, fold, metric, reference)
    return rows.iloc[0]


# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS)
def test_pass2_population_matches_the_baseline(run, arm):
    pop = run.population
    got = {k: int(_row(pop, arm, "all", f"pass2_{k}")["numerator"]) for k in ("n", "tp", "fp")}
    assert got == PASS2[arm]
    prec = _row(pop, arm, "all", "pass2_precision")
    assert (prec["numerator"], prec["denominator"]) == (PASS2[arm]["tp"], PASS2[arm]["n"])
    rec = _row(pop, arm, "all", "pass2_recall")
    assert (rec["numerator"], rec["denominator"]) == (PASS2[arm]["tp"], N_POSITIVES)
    m = d.pass2_metrics(run.trace.views[arm])
    elig = _row(pop, arm, "all", "pass2_recall_eligible")
    assert (elig["numerator"], elig["denominator"]) == (m["pass2_recall_eligible"].numerator,
                                                        m["pass2_recall_eligible"].denominator)


def test_frozen_references_are_the_step_2b_flags(run):
    for key, n in REFERENCE_FLAGS.items():
        for arm in ARMS:
            assert run.frozen[f"{key}.{arm}.reference_flags"] == n, (key, arm)


def test_frozen_counts_reconcile_and_match_router_upstream_metrics(run):
    fz = run.frozen
    for key in REFERENCE_FLAGS:
        for arm in ARMS:
            g = lambda m: fz[f"{key}.{arm}.{m}"]  # noqa: E731
            assert g("pass2_tp") == g("overlap_tp") + g("unique_tp") == PASS2[arm]["tp"]
            assert g("pass2_fp") == g("overlap_fp") + g("unique_fp") == PASS2[arm]["fp"]
            assert g("union_tp") - g("reference_tp") == g("unique_tp")
            assert g("union_fp") - g("reference_fp") == g("unique_fp")
            theirs = d.router_upstream_metrics(run.trace.views[arm], run.refs[key])
            for m in ("pass2_upstream_redundancy", "pass2_unique_positive_share",
                      "pass2_new_false_positive_rate"):
                assert g(m) == theirs[m]
            assert g("unique_tp") == theirs["pass2_unique_positives_n"]


def test_breakdown_partitions_pass2_and_matches_the_union(run):
    bd = d.pass2_positive_breakdown(run.trace, run.refs)
    for arm in ARMS:
        r = bd[(bd["arm"] == arm) & bd["fold"].map(lambda f: f == "all")].iloc[0]
        assert r["pass2_tp"] == PASS2[arm]["tp"]
        assert r[["tp_both", "tp_lr_only", "tp_ebm_only", "tp_neither"]].sum() == r["pass2_tp"]
        assert r[["fp_both", "fp_lr_only", "fp_ebm_only", "fp_neither"]].sum() == r["pass2_fp"]
        assert r["tp_neither"] == run.frozen[f"union.{arm}.unique_tp"]
        assert r["tp_both"] == run.frozen[f"intersection.{arm}.overlap_tp"]


def test_common_eligible_population(run):
    t = run.trace
    cp = d.common_eligible_population(t).set_index("fold")
    g, r = t.views["glass"], t.views["blackbox"]
    assert cp.loc["all", "common_rows"] == int((~g.pass1 & ~r.pass1).sum())
    assert cp.loc["all", "glass_pass2_in_common"] + cp.loc["all", "glass_pass2_outside_common"] \
        == PASS2["glass"]["n"]
    assert cp.loc["all", "rf_pass2_in_common"] + cp.loc["all", "rf_pass2_outside_common"] \
        == PASS2["blackbox"]["n"]
    for c in cp.columns:
        assert cp.drop(index="all")[c].sum() == cp.loc["all", c]


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("model", ["lr", "ebm"])
def test_matched_selection_on_the_saved_scores(run, arm, model):
    sel = run.matched.meta["selections"][arm][model]
    fold = np.asarray(run.trace.fold_id)
    router = np.asarray(sel.router_set)
    for k in range(run.trace.n_folds):
        assert int((sel.flags & (fold == k)).sum()) == int((router & (fold == k)).sum())
    assert not (np.asarray(sel.strict) & ~np.asarray(sel.flags)).any()
    assert not (np.asarray(sel.flags) & ~np.asarray(sel.inclusive)).any()
    assert sel.score_space == cb.MATCH_SCORES[model]
    view = run.lr if model == "lr" else run.ebm
    y2 = np.random.default_rng(11).permutation(np.asarray(run.trace.y))
    y2.setflags(write=False)
    again = d.matched_reference_selection(replace(run.trace, y=y2), arm, replace(view, y=y2))
    np.testing.assert_array_equal(again.flags, sel.flags)              # labels are never read


def test_matched_counts_reconcile_per_fold(run):
    tab = run.matched.table
    for _, grp in tab.groupby(["arm", "reference", "metric"]):
        pooled = grp[grp["fold"].map(lambda f: f == "all")].iloc[0]
        folds = grp[grp["fold"].map(lambda f: f != "all")]
        assert int(folds["numerator"].sum()) == pooled["numerator"]
        if pooled["status"] != "count":
            assert int(folds["denominator"].sum()) == pooled["denominator"]
    for arm in ARMS:
        for key in ("lr_matched", "ebm_matched"):
            assert run.matched[f"{key}.{arm}.pass2_n"] == run.matched[f"{key}.{arm}.reference_flags"]
            lo = run.matched[f"{key}.{arm}.unique_tp_inclusive"]
            hi = run.matched[f"{key}.{arm}.unique_tp_strict"]
            assert lo <= run.matched[f"{key}.{arm}.unique_tp"] <= hi


def test_disclosures(run):
    assert run.rf.metadata.get("constraint_source") == "match_glass_oof"
    for res in (run.frozen, run.matched):
        assert res.notes[0] == d.HISTORICAL_TEST_EXPOSURE
        text = " ".join(res.notes)
        for needle in ("match_glass_oof", "F17"):
            assert needle in text, needle
    text = " ".join(run.matched.notes)
    for needle in ("No label is read", "SHA-256", "Arm-specific", "descriptive operating-rate"):
        assert needle in text, needle


# keep last: every file read above must be unchanged
def test_no_artifact_file_was_changed(run):
    after = {p: d.file_sha256(p) for p in run.before}
    assert after == run.before
    if run.real.sums:
        for p in run.attested:
            assert run.before[p.resolve()] == run.real.sha(p), p
