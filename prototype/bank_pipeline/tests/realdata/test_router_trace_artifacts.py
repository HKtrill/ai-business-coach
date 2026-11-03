"""
PR 37 Commit 2 (Step 2A): the three-state router trace on the SAVED October 23
artifacts - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_router_trace_artifacts.py

Builds the trace from the GLASS Router (recorded by the native Meta-EBM) and
the RF Router (recorded by the black-box-upstream Meta-EBM), audits the
routing policy with the GLASS-arm RF result's Pass 1 candidate, and checks the
historical anchors below. The anchors are expectations recorded in the PR 37
handoff, not logic: the implementation never uses them.

Nothing is fitted, written or regenerated, and no test label is read (the
loader is tests.support.router_artifacts, shared with the Commit 1 tests).
Every artifact file the trace reads is hashed before it is first loaded and
again at the end: all must be unchanged. With ``PR37_BASELINE_SHA256SUMS`` set,
the router files and the RF result must also equal the preserved baseline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import trace as tr
from shared.stage2.diagnostics.boundary import RF_PASS2_REMAINDER_PROTOCOL
from tests.support.router_artifacts import (
    ROUTER_SPLIT_FP,
    STAGE_IO_SPLIT_FP,
    TRAIN_LABEL_FP,
    Real,
)

pytestmark = pytest.mark.realdata

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN

#: Historical anchors of the October 23 baseline (PR 37 handoff).
N_TRAIN, N_POSITIVES, N_FOLDS, FOLD_ROWS = 32_950, 3_712, 5, 6_590
HISTORICAL_STATE_COUNTS = {
    "glass": {P1: 11_489, P2: 8_468, AB: 12_993},
    "blackbox": {P1: 13_074, P2: 9_427, AB: 10_449},
}
#: RF Pass 1 rows per outer fold, as reproduced by the Commit 1 equivalence run.
RF_PASS1_BY_FOLD = (2_584, 2_487, 2_809, 2_552, 2_642)
#: The RF Router's recorded constraint source (Step 1B audit finding).
RF_CONSTRAINT_SOURCE = "match_glass_oof"
DECISION_CLAIMS = ("decision.pass1_precedence", "decision.pass2_eligibility",
                   "decision.abstain_not_pass1")


@pytest.fixture(scope="module")
def run():
    """Each file is hashed before its first load; everything is built once."""
    real = Real()
    before: dict = {}

    def remember(p) -> Path:
        p = Path(p)
        before.setdefault(p.resolve(), d.file_sha256(p))
        return p

    roots = {k: remember(real.roots()[k]) for k in ("meta_ebm", "meta_ebm_on_blackbox")}
    glass_p = remember(real.paths("meta_ebm")["stage2"])
    rf_p = remember(real.paths("meta_ebm_on_blackbox")["stage2"])
    ref = real.ref()
    glass, rf = real.view("glass"), real.view("rf")
    rf_result_p = remember(real.rf_result_path())
    cand = d.extract_rf_pass1_candidate(rf_result_p, ref, expected_sha256=real.sha(rf_result_p))
    trace = d.build_router_trace(glass, rf, ref)
    return SimpleNamespace(real=real, ref=ref, glass=glass, rf=rf, cand=cand, trace=trace,
                           before=before, attested=(roots["meta_ebm"], glass_p, rf_p, rf_result_p))


def _pooled(frame):
    return frame[frame["fold"].map(lambda f: f == "all")]


def _per_fold(frame):
    return frame[frame["fold"].map(lambda f: f != "all")]


# ----------------------------------------------------------------------
def test_trace_builds_with_every_check_passing_and_the_recorded_identities(run):
    checks = run.trace.check_frame()
    assert checks["passed"].all(), checks.loc[~checks["passed"], ["stage", "check", "detail"]]
    t = run.trace
    assert t.router_split_fingerprint == ROUTER_SPLIT_FP
    assert t.train_label_fingerprint == TRAIN_LABEL_FP
    assert t.stage_io_split_fingerprint == STAGE_IO_SPLIT_FP     # attested, not recomputed
    assert t.index.equals(run.ref.index_train)


def test_training_population_and_outer_folds_match_the_baseline(run):
    t = run.trace
    assert (t.n, t.n_positives, t.n_folds) == (N_TRAIN, N_POSITIVES, N_FOLDS)
    assert np.bincount(t.fold_id).tolist() == [FOLD_ROWS] * N_FOLDS
    pos = np.bincount(t.fold_id, weights=t.y).astype(int)
    assert pos.sum() == N_POSITIVES and pos.max() - pos.min() <= 1      # stratified plan


@pytest.mark.parametrize("arm", ["glass", "blackbox"])
def test_pooled_state_counts_match_the_historical_baseline(run, arm):
    sd = _pooled(d.state_distribution(run.trace))
    got = {s: int(n) for s, n in zip(sd.loc[sd["arm"] == arm, "state"],
                                     sd.loc[sd["arm"] == arm, "n"])}
    assert got == HISTORICAL_STATE_COUNTS[arm]
    assert sum(got.values()) == N_TRAIN


def test_rf_pass1_per_fold_counts_match_the_commit1_equivalence_run(run):
    sd = _per_fold(d.state_distribution(run.trace))
    rf = sd[(sd["arm"] == "blackbox") & (sd["state"] == P1)].sort_values("fold")
    assert tuple(int(n) for n in rf["n"]) == RF_PASS1_BY_FOLD


def test_class_composition_per_state_is_complete_and_defined(run):
    sd = d.state_distribution(run.trace)
    assert ((sd["positives"] + sd["negatives"]) == sd["n"]).all()
    for arm in ("glass", "blackbox"):
        pooled = _pooled(sd[sd["arm"] == arm])
        assert int(pooled["positives"].sum()) == N_POSITIVES
        assert (pooled["positive_rate_status"] == "defined").all()
        np.testing.assert_allclose(pooled["positive_rate"] + pooled["negative_rate"], 1.0)
        np.testing.assert_allclose(pooled["share"].sum(), 1.0)


def test_per_fold_values_reconcile_with_the_pooled_values(run):
    sd = d.state_distribution(run.trace)
    for arm in ("glass", "blackbox"):
        a = sd[sd["arm"] == arm]
        summed = _per_fold(a).groupby("state", observed=False)[
            ["n", "positives", "negatives"]].sum()
        pooled = _pooled(a).set_index("state")[["n", "positives", "negatives"]]
        assert summed.to_numpy().tolist() == pooled.to_numpy().tolist()
    rr = d.routing_rates(run.trace)
    for (arm, metric), grp in rr.groupby(["arm", "metric"]):
        assert int(_per_fold(grp)["numerator"].sum()) == int(_pooled(grp)["numerator"].iloc[0])
        assert int(_pooled(grp)["denominator"].iloc[0]) == N_TRAIN


def test_routing_rates_match_router_metrics(run):
    rr = _pooled(d.routing_rates(run.trace))
    for arm in ("glass", "blackbox"):
        theirs = d.router_metrics(run.trace.views[arm])
        mine = rr[rr["arm"] == arm].set_index("metric")
        for m in ("pass1_rate", "pass2_rate", "coverage", "abstain_rate"):
            assert (int(mine.loc[m, "numerator"]), int(mine.loc[m, "denominator"])) == (
                theirs[m].numerator, theirs[m].denominator), (arm, m)
        counts = HISTORICAL_STATE_COUNTS[arm]
        assert int(mine.loc["coverage", "numerator"]) == counts[P1] + counts[P2]


def test_routing_policy_audit_on_the_saved_artifacts(run):
    rep = d.check_rf_pass1_score_equivalence(run.rf, run.cand)
    assert rep.passed, rep.conclusion
    audit = d.routing_policy_audit(run.trace, rf_candidate=run.cand)
    status = {(a, c): s for a, c, s in zip(audit["arm"], audit["claim"], audit["status"])}
    for claim in DECISION_CLAIMS:
        assert status[("blackbox", claim)] == tr.VERIFIED_FROM_ARTIFACTS, claim
        assert status[("glass", claim)] == tr.CODE_LEVEL_ONLY, claim
    for arm in ("glass", "blackbox"):
        assert status[(arm, "states.one_per_row")] == tr.VERIFIED_FROM_ARTIFACTS
        for claim in ("decision.pass2_condition", "decision.abstain_not_pass2",
                      "fit.pass2_population_realised"):
            assert status[(arm, claim)] == tr.NOT_VERIFIABLE, (arm, claim)
        assert status[(arm, "fit.pass2_population_declared")] == tr.DECLARED
    assert (audit.loc[audit["status"] == tr.VERIFIED_FROM_ARTIFACTS, "violations"] == 0).all()
    declared = audit[audit["claim"] == "fit.pass2_population_declared"].set_index("arm")
    assert "'full'" in declared.loc["glass", "evidence"]
    assert RF_PASS2_REMAINDER_PROTOCOL in declared.loc["blackbox", "evidence"]


def test_comparison_matches_the_historical_counts_and_discloses_dependence(run):
    res = d.compare_state_distributions(run.trace)
    for s in (P1, P2, AB):
        assert res[f"difference.{s}.n"] == (HISTORICAL_STATE_COUNTS["blackbox"][s]
                                            - HISTORICAL_STATE_COUNTS["glass"][s])
    g, b = HISTORICAL_STATE_COUNTS["glass"], HISTORICAL_STATE_COUNTS["blackbox"]
    assert res["difference_pp.coverage"] == pytest.approx(
        100 * ((b[P1] + b[P2]) - (g[P1] + g[P2])) / N_TRAIN)
    assert run.rf.metadata.get("constraint_source") == RF_CONSTRAINT_SOURCE
    text = " ".join(res.notes)
    assert res.notes[0] == d.HISTORICAL_TEST_EXPOSURE
    for needle in (RF_CONSTRAINT_SOURCE, "'full'", RF_PASS2_REMAINDER_PROTOCOL, "No ranking",
                   "fold_rule_sets", "pass2_outer_scores"):
        assert needle in text, needle


# keep last: every file read above must be unchanged
def test_no_artifact_file_was_changed(run):
    after = {p: d.file_sha256(p) for p in run.before}
    assert after == run.before
    if run.real.sums:
        for p in run.attested:
            assert run.before[p.resolve()] == run.real.sha(p), p
