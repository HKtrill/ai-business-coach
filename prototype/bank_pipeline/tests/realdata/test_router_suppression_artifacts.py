"""
PR 37 Commit 2 (Step 2B): Pass 1 false-positive suppression on the SAVED
October 23 artifacts - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_router_suppression_artifacts.py

Both routers' Pass 1 against the GLASS arm's Stage 1 LR (lr.in_stage_f2_cv)
and Stage 3 EBM (ebm.nested_f2) primary operating points, their union and
their intersection. The anchors below are the PR 37 handoff's and the
validated Step 2A results: expectations, never used by the implementation.

Nothing is fitted, written or regenerated; no test label is read (loader:
tests.support.router_artifacts). Every artifact file read is hashed before its
first load and again at the end. All files are read from the preserved October
23 archive (``PR37_BASELINE_SHA256SUMS``; tests.support.archived_artifacts),
never from the live models/ folders, and each must equal its recorded SHA-256.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import suppression as sp
from tests.support.archived_artifacts import HistoricalReal

pytestmark = pytest.mark.realdata

#: Anchors (handoff section 11 and the validated Step 2A trace).
N_TRAIN, N_POSITIVES = 32_950, 3_712
REFERENCE_FLAGS = {"lr": 7_727, "ebm": 6_113}
EBM_NESTED_THRESHOLDS = (0.56, 0.56, 0.54, 0.49, 0.49)
PASS1 = {"glass": {"rows": 11_489, "negatives": 10_976, "positives": 513},
         "blackbox": {"rows": 13_074, "negatives": 12_545, "positives": 529}}
RF_CONSTRAINT_SOURCE = "match_glass_oof"


@pytest.fixture(scope="module")
def run():
    """Each file is hashed before its first load; everything is built once."""
    real = HistoricalReal()
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
    refs = d.primary_reference_flags(lr, ebm)
    return SimpleNamespace(real=real, ref=ref, glass=glass, rf=rf, lr=lr, ebm=ebm, trace=trace,
                           refs=refs, report=d.pass1_suppression_report(trace, lr, ebm),
                           population=d.pass1_population(trace), before=before,
                           attested=(*native.values(), rf_p))


def _row(table, arm, fold, metric, reference=None):
    m = ((table["arm"] == arm) & (table["fold"].map(lambda f: f == fold))
         & (table["metric"] == metric))
    if reference is not None:
        m &= table["reference"] == reference
    rows = table[m]
    assert len(rows) == 1, (arm, fold, metric, reference)
    return rows.iloc[0]


# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", ["glass", "blackbox"])
def test_pass1_population_matches_the_step_2a_baseline(run, arm):
    pop = run.population
    got = {k: int(_row(pop, arm, "all", f"pass1_{k}")["numerator"])
           for k in ("rows", "negatives", "positives")}
    assert got == PASS1[arm]
    assert int(_row(pop, arm, "all", "positives")["numerator"]) == N_POSITIVES
    npv = _row(pop, arm, "all", "pass1_npv")
    assert (npv["numerator"], npv["denominator"]) == (PASS1[arm]["negatives"], PASS1[arm]["rows"])
    leak = _row(pop, arm, "all", "pass1_leakage_share")
    assert (leak["numerator"], leak["denominator"]) == (PASS1[arm]["positives"], N_POSITIVES)


def test_primary_reference_flags_match_the_registered_counts(run):
    s = d.reference_flag_summary(run.trace, run.refs).set_index("reference")
    for k, n in REFERENCE_FLAGS.items():
        assert int(s.loc[k, "flags"]) == n, k
    assert s.loc["lr", "score_space"] == "calibrated_probability"
    assert s.loc["ebm", "score_space"] == "raw_model_score"
    assert run.report["union_reconciles"] is True
    assert int(s.loc["union", "flags"]) == (REFERENCE_FLAGS["lr"] + REFERENCE_FLAGS["ebm"]
                                            - int(s.loc["intersection", "flags"]))
    nested = run.ebm.stored_decisions["ebm.nested_f2"].by_fold
    assert tuple(nested) == pytest.approx(EBM_NESTED_THRESHOLDS)


def test_reference_fold_plans(run):
    meta = run.report.meta["references"]
    assert meta["ebm"]["fold_plan"] == "same as the routers' outer folds"
    assert meta["lr"]["fold_plan"].startswith("10 folds, not the routers' 5")


def test_counts_partition_and_folds_reconcile(run):
    table = run.report.table
    for key in sp.REFERENCE_KEYS:
        for arm in ("glass", "blackbox"):
            for fold in ["all", *range(run.trace.n_folds)]:
                v = {m: int(_row(table, arm, fold, m, key)["numerator"])
                     for m in sp.SUPPRESSION_COUNTS}
                assert v["reference_fp"] == v["intercepted_fp"] + v["retained_fp"]
                assert v["reference_tp"] == v["suppressed_tp"] + v["retained_tp"]
                assert v["intercepted_flags"] == v["intercepted_fp"] + v["suppressed_tp"]
            pass1_pos = int(_row(table, arm, "all", "suppressed_tp", key)["numerator"]) + int(
                _row(table, arm, "all", "pass1_positives_unflagged", key)["numerator"])
            assert pass1_pos == PASS1[arm]["positives"]
    for _, grp in table.groupby(["reference", "arm", "metric"]):
        pooled = grp[grp["fold"].map(lambda f: f == "all")].iloc[0]
        folds = grp[grp["fold"].map(lambda f: f != "all")]
        assert int(folds["numerator"].sum()) == pooled["numerator"]
        if pooled["status"] != "count":
            assert int(folds["denominator"].sum()) == pooled["denominator"]


@pytest.mark.parametrize("key", sp.REFERENCE_KEYS)
@pytest.mark.parametrize("arm", ["glass", "blackbox"])
def test_pooled_values_match_router_upstream_metrics(run, key, arm):
    theirs = d.router_upstream_metrics(run.trace.views[arm], run.refs[key])
    for m in ("pass1_fp_suppression", "pass1_tp_cost", "pass1_upstream_flag_rate",
              "pass1_leak_upstream_flagged", "pass1_leak_upstream_missed"):
        r = _row(run.report.table, arm, "all", m, key)
        assert (r["numerator"], r["denominator"]) == (theirs[m].numerator, theirs[m].denominator)


def test_report_discloses_dependence_and_scope(run):
    assert run.rf.metadata.get("constraint_source") == RF_CONSTRAINT_SOURCE
    text = " ".join(run.report.notes)
    assert run.report.notes[0] == d.HISTORICAL_TEST_EXPOSURE
    for needle in (RF_CONSTRAINT_SOURCE, "never netted", "not the integrated cascade",
                   "Common reference", "In-selection", "fold-nested", "Step 2C"):
        assert needle in text, needle


# keep last: every file read above must be unchanged
def test_no_artifact_file_was_changed(run):
    after = {p: d.file_sha256(p) for p in run.before}
    assert after == run.before
    if run.real.sums:
        for p in run.attested:
            assert run.before[p.resolve()] == run.real.sha(p), p
