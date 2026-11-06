"""
PR 37 Commit 2 (Step 2D): pass overlap and redundancy diagnosis on the SAVED
October 23 artifacts - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_router_overlap_artifacts.py

Checks the overlap tables against the validated Steps 2A-2C results (parity
anchors from the PR 37 handoff and the Step 2B / 2C reports), the signal
availability classification, the label-free score strata and the internal
consistency of the score-conditional expectations. Expectations only: the
implementation never uses these numbers.

Nothing is fitted, written or regenerated; no test label is read (loader:
tests.support.router_artifacts). Every artifact file read is hashed before its
first load and again at the end. All files are read from the preserved October
23 archive (``PR37_BASELINE_SHA256SUMS``; tests.support.archived_artifacts),
never from the live models/ folders, and each must equal its recorded SHA-256.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import overlap as ov
from shared.stage2.diagnostics import trace as tr
from tests.support.archived_artifacts import HistoricalReal

pytestmark = pytest.mark.realdata

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN
N_TRAIN, N_POSITIVES = 32_950, 3_712
STATES = {"glass": {P1: 11_489, P2: 8_468, AB: 12_993},
          "blackbox": {P1: 13_074, P2: 9_427, AB: 10_449}}
#: Step 2B: Pass 1 rows the reference flags (and flagged positives among them).
PASS1_FLAGGED = {("glass", "lr"): (586, 74), ("glass", "ebm"): (250, 53),
                 ("glass", "union"): (644, 82), ("blackbox", "lr"): (5, 0),
                 ("blackbox", "ebm"): (9, 5), ("blackbox", "union"): (14, 5)}
#: Step 2C: Pass 2 positives / negatives the union does not flag.
UNIQUE_VS_UNION = {"glass": (131, 1_862), "blackbox": (117, 1_669)}
#: Step 2A: Pass 2 true positives.
PASS2_TP = {"glass": 2_337, "blackbox": 2_560}
#: Step 2C common-population table: GLASS Pass 1 & RF Pass 2 and GLASS Pass 2 & RF Pass 1 by fold.
G1_R2_BY_FOLD = (200, 92, 189, 82, 250)
G2_R1_BY_FOLD = (74, 0, 0, 0, 0)
#: Frozen LR-only / EBM-only true positives (Step 2C reference flags).
LR_ONLY_TP, EBM_ONLY_TP = 170, 56
EBM_NESTED_THRESHOLDS = (0.56, 0.56, 0.54, 0.49, 0.49)


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
    rf_result_p = remember(real.rf_result_path())
    cand = d.extract_rf_pass1_candidate(rf_result_p, ref, expected_sha256=real.sha(rf_result_p))
    trace = d.build_router_trace(glass, rf, ref)
    refs = d.primary_reference_flags(lr, ebm)
    strata = d.score_strata(trace, lr, ebm)
    return SimpleNamespace(real=real, ref=ref, lr=lr, ebm=ebm, trace=trace, refs=refs,
                           strata=strata, cand=cand, groups=d.key_groups(trace, refs),
                           matched=d.pass2_matched_contribution(trace, lr, ebm), before=before,
                           attested=(*native.values(), rf_p, rf_result_p))


def _fold(df, fold):
    return df[df["fold"].map(lambda f: f == fold)]


# ----------------------------------------------------------------------
def test_contingency_matches_steps_2a_to_2c(run):
    ct = _fold(d.router_reference_contingency(run.trace, run.refs), "all")
    for arm, states in STATES.items():
        for key in ("lr", "ebm", "union", "intersection"):
            c = ct[(ct["arm"] == arm) & (ct["reference"] == key)].set_index(
                ["state", "reference_flag"])
            assert c["n"].sum() == N_TRAIN and c["positives"].sum() == N_POSITIVES
            for s, n in states.items():
                assert c.loc[(s, False), "n"] + c.loc[(s, True), "n"] == n
            if (arm, key) in PASS1_FLAGGED:
                assert (c.loc[(P1, True), "n"], c.loc[(P1, True), "positives"]) == \
                    PASS1_FLAGGED[(arm, key)]
        u = ct[(ct["arm"] == arm) & (ct["reference"] == "union")].set_index(
            ["state", "reference_flag"])
        assert (u.loc[(P2, False), "positives"], u.loc[(P2, False), "negatives"]) == \
            UNIQUE_VS_UNION[arm]


def test_agreement_matches_router_upstream_metrics(run):
    ag = _fold(d.router_reference_agreement(run.trace, run.refs), "all")
    for key in run.refs:
        for arm in STATES:
            a = ag[(ag["arm"] == arm) & (ag["reference"] == key)].set_index("metric")
            theirs = d.router_upstream_metrics(run.trace.views[arm], run.refs[key])
            for m in ("pass1_upstream_flag_rate", "pass2_upstream_redundancy",
                      "covered_agreement", "abstain_upstream_flag_rate",
                      "abstain_upstream_precision"):
                assert (a.loc[m, "numerator"], a.loc[m, "denominator"]) == (
                    theirs[m].numerator, theirs[m].denominator), (key, arm, m)


def test_crosstab_opposite_cells_and_margins(run):
    ct = d.router_crosstab(run.trace, run.refs)
    pooled = _fold(ct, "all")
    assert pooled.groupby("glass_state")["n"].sum().to_dict() == STATES["glass"]
    assert pooled.groupby("rf_state")["n"].sum().to_dict() == STATES["blackbox"]
    for k in range(5):
        cells = _fold(ct, k).set_index(["glass_state", "rf_state"])
        assert cells.loc[(P1, P2), "n"] == G1_R2_BY_FOLD[k]
        assert cells.loc[(P2, P1), "n"] == G2_R1_BY_FOLD[k]
    rr = d.router_router_table(run.trace.views["glass"], run.trace.views["blackbox"])
    cs = _fold(d.crosstab_summary(run.trace), "all").set_index("metric")
    assert cs.loc["cross_state_agreement", "numerator"] == rr["state_agreement"].numerator
    assert cs.loc["cross_opposite_rate", "numerator"] == sum(G1_R2_BY_FOLD) + sum(G2_R1_BY_FOLD)


def test_cross_router_pass2_overlap_partitions(run):
    po = d.pass2_cross_router_overlap(run.trace, run.refs)
    for _, r in po.iterrows():
        for lab in ("tp", "fp"):
            assert r[f"glass_unique_{lab}"] == r[f"shared_{lab}"] + r[f"glass_only_{lab}"]
            assert r[f"rf_unique_{lab}"] == r[f"shared_{lab}"] + r[f"rf_only_{lab}"]
        assert r["glass_only_tp"] == r["glass_only_tp_in_rf_pass1"] + r["glass_only_tp_in_rf_abstain"]
        assert r["rf_only_tp"] == r["rf_only_tp_in_glass_pass1"] + r["rf_only_tp_in_glass_abstain"]
    pooled = _fold(po, "all").set_index("reference")
    assert pooled.loc["union", "glass_unique_tp"] == UNIQUE_VS_UNION["glass"][0]
    assert pooled.loc["union", "rf_unique_tp"] == UNIQUE_VS_UNION["blackbox"][0]
    assert pooled.loc["all_pass2", "glass_unique_tp"] == PASS2_TP["glass"]
    assert pooled.loc["all_pass2", "rf_unique_tp"] == PASS2_TP["blackbox"]


def test_signal_availability_on_the_saved_artifacts(run):
    sa = d.signal_availability(run.trace, rf_candidate=run.cand).set_index(["arm", "signal"])
    for arm in ("glass", "blackbox"):
        assert sa.loc[(arm, "final Pass 1 & Pass 2 rows"), "rows"] == 0
    assert sa.loc[("blackbox", "Pass 1 outer score (all rows)"), "status"] == \
        tr.VERIFIED_FROM_ARTIFACTS
    for key in (("glass", "Pass 1 candidate-rule matches per row"),
                ("glass", "outer-fold fold rule sets"), ("blackbox", "Pass 2 outer score"),
                ("both", "analysis: GLASS Pass 1 vs Pass 2 candidate / signal overlap"),
                ("both", "analysis: RF Pass 1 vs Pass 2 signal overlap")):
        assert sa.loc[key, "status"] == tr.NOT_VERIFIABLE, key


def test_strata_are_label_free_and_use_fold_thresholds(run):
    s, t = run.strata, run.trace
    fold = np.asarray(t.fold_id)
    thr = np.asarray(s.ebm_threshold_row)
    assert tuple(float(thr[fold == k][0]) for k in range(5)) == pytest.approx(EBM_NESTED_THRESHOLDS)
    y2 = np.random.default_rng(5).permutation(np.asarray(t.y))
    y2.setflags(write=False)
    s2 = d.score_strata(replace(t, y=y2), replace(run.lr, y=y2), replace(run.ebm, y=y2))
    for name in ov.STRATIFIERS:
        np.testing.assert_array_equal(s.bins[name], s2.bins[name])
        assert all(1 <= n <= s.requested_bins[name] for n in s.n_bins[name].values())


def test_score_conditional_expectations_are_consistent(run):
    t, s = run.trace, run.strata
    p = np.asarray(s.lr_probability)
    ss = d.state_standardization(t, s)
    for arm in STATES:
        sub = _fold(ss[ss["arm"] == arm], "all")
        assert sub.set_index("state")["n"].to_dict() == STATES[arm]
        assert sub["expected_lr_calibrated"].sum() == pytest.approx(p.sum())
        for name in ov.STRATIFIERS:
            assert sub[f"expected_stratified_{name}"].sum() == pytest.approx(N_POSITIVES)
    sc = d.score_conditional_states(t, s, run.refs)
    for (arm, name), grp in _fold(sc, "all").groupby(["arm", "stratifier"]):
        assert grp["n"].sum() == N_TRAIN and grp["positives"].sum() == N_POSITIVES
    oe = _fold(d.group_observed_expected(t, s, run.groups), "all").set_index("group")
    assert (oe.loc["glass_pass1_and_union_flag", "n"],
            oe.loc["glass_pass1_and_union_flag", "observed"]) == PASS1_FLAGGED[("glass", "union")]
    assert oe.loc["glass_pass1_and_rf_pass2", "n"] == sum(G1_R2_BY_FOLD)


def test_boundary_and_baselines(run):
    bd = _fold(d.boundary_diagnostics(run.trace, run.strata, run.groups), "all")
    r = bd[(bd["group"] == "glass_pass1_and_lr_flag") & (bd["score"] == "lr")
           & (bd["metric"] == "above_threshold_share")].iloc[0]
    assert r["numerator"] == r["denominator"] == PASS1_FLAGGED[("glass", "lr")][0]
    cb = _fold(d.unique_positive_chance_baselines(run.trace, run.refs, run.matched), "all")
    fz = cb[(cb["setting"] == "frozen_pass2_unique") & (cb["reference"] == "union")].set_index("arm")
    for arm, (tp, fp) in UNIQUE_VS_UNION.items():
        assert (fz.loc[arm, "observed_positives"], fz.loc[arm, "flags"]) == (tp, tp + fp)
    md = _fold(d.model_disagreement_baseline(run.trace, run.refs, run.matched), "all")
    fr = md[md["setting"] == "frozen"].iloc[0]
    assert (fr["lr_only_tp"], fr["ebm_only_tp"]) == (LR_ONLY_TP, EBM_ONLY_TP)
    for arm in STATES:
        m = md[(md["setting"] == "matched_same_k") & (md["arm"] == arm)].iloc[0]
        assert m["k"] == run.matched[f"lr_matched.{arm}.pass2_n"]
        assert m["router_unique_tp_vs_lr"] == run.matched[f"lr_matched.{arm}.unique_tp"]


def test_disclosures(run):
    text = " ".join(d.overlap_notes(run.trace, run.strata))
    for needle in ("routing invariant", "match_glass_oof", "fold_rf", "F17", "Stage 4",
                   "Neither is a null distribution"):
        assert needle in text, needle


# keep last: every file read above must be unchanged
def test_no_artifact_file_was_changed(run):
    after = {p: d.file_sha256(p) for p in run.before}
    assert after == run.before
    if run.real.sums:
        for p in run.attested:
            assert run.before[p.resolve()] == run.real.sha(p), p
