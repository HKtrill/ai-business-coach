"""
PR 42 Commit 4, Step 4B on the real training rows - opt-in, like every
realdata test (it reads ``data/raw`` and ``data/db``):

    pytest -m realdata tests/realdata/test_pr42_binning_candidates.py

The notebook's Step 4B path (Cells 40-46), without writing a record:

1. The registration is the pinned 1.0.0 and its inputs (PR 39 in-sample
   supports, GLASS profile, shared contract, frozen binning) are unchanged.
2. Every fold fit uses only its fit rows; flipping fold 0's held-out labels
   leaves its inner values, cuts, tail choice and partners unchanged (with its
   control).
3. The comparator is E-B's out-of-fold binning, row by row and table by table.
4. Each partner's two filter variants partition its rows and reconcile with
   its counts, per fold and pooled.
5. Every verdict and figure the review document states holds, and so does
   every statement about the post-registration fold-local sensitivity (Cell 47),
   which leaves the registered verdicts unchanged.

Only training rows are bound; no artifact is opened.
"""

from __future__ import annotations

import numpy as np
import pytest

from feature_research import binning_candidates as BC
from feature_research import binning_drift as BD
from feature_research import encoding_audit as EA
from feature_research import feature_inventory as FI
from feature_research.training_side import draw_training_side

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def step4b():
    side = draw_training_side(verbose=False)
    X, y = side.X_train, side.y_train
    frames = FI.build_training_frames(X, y)
    T = EA.build_time_index(X)
    record, _ = BD.read_pr39_record()
    s2 = BD.stage2_side(X, y)
    repro = BD.reproduce_pr39(s2, record)
    sources = BD.source_values(s2, frames["stage2_sources"], repro["eb"])
    states = BD.macro_states(X, T, y)
    svals = BD.state_values(states, sources, s2, BD.fold_fit_values(s2, states, repro["eb"]))
    reg = BC.registration_checks(record)
    heldout = BC.heldout_sources(s2, sources)
    fold_fits = BC.fit_folds(s2, states, svals)
    deployed = BC.deployment_fits(s2, states, svals, sources)
    comparison = BC.compare(s2, sources, fold_fits, deployed, repro["eb"])
    dep = BC.dependence(s2, T, comparison)
    filt = BC.warm_filter(s2, heldout, fold_fits)
    decision = BC.decide(comparison, dep, filt)
    sens = BC.sensitivity(s2, sources, T, fold_fits, BC.sensitivity_all_rows(s2, states, svals), comparison,
                          deployed)
    return dict(side=s2, states=states, svals=svals, reg=reg, heldout=heldout, fold_fits=fold_fits,
                comparison=comparison, dep=dep, filt=filt, decision=decision, sens=sens)


def test_the_registration_and_its_inputs_are_unchanged(step4b):
    assert step4b["reg"]["passed"].all(), step4b["reg"].to_string()
    assert step4b["side"].fold_fingerprint == BC.FOLD_FINGERPRINT


def test_fold_fits_use_only_their_fit_rows_and_ignore_held_out_labels(step4b):
    for ff in step4b["fold_fits"]:
        assert ff["checks"]["passed"].all(), ff["checks"].to_string()
    flips = BC.isolation_checks(step4b["side"], step4b["states"], step4b["svals"], step4b["fold_fits"], folds=[0])
    assert flips["passed"].all(), flips.to_string()


def test_the_comparator_is_e_b_and_every_fit_is_feasible(step4b):
    assert step4b["heldout"]["checks"]["passed"].all(), step4b["heldout"]["checks"].to_string()
    assert step4b["comparison"]["checks"]["passed"].all(), step4b["comparison"]["checks"].to_string()
    assert all(all(v) for v in step4b["comparison"]["feasible"].values())


def test_the_filter_variants_reconcile_per_fold_and_pooled(step4b):
    filt = step4b["filt"]
    assert filt["checks"]["passed"].all(), filt["checks"].to_string()
    per, slots = filt["per_fold"], filt["by_slot"].set_index("slot")
    for v in ("rows", "subscribers", "non_subscribers"):
        assert (per[f"required_{v}"] + per[f"excluded_{v}"] == per[f"alone_{v}"]).all()
    for slot, g in per.groupby("slot"):
        assert slots.loc[slot, "alone_rows"] == g["alone_rows"].sum()
        assert slots.loc[slot, "required_removes_subscribers"] == g["excluded_subscribers"].sum()
    assert set(per["fold"]) == set(range(BC.N_FOLDS)) and (per.groupby("fold").size() == BC.PARTNER_K).all()
    assert not [p for p in per["partner"] if "eci_" in p]


def test_every_statement_of_the_review_document_holds(step4b):
    stated = BC.stated_checks(step4b["decision"], step4b["comparison"], step4b["filt"], step4b["dep"])
    assert stated["passed"].all(), stated[~stated["passed"]].to_string()
    lab = step4b["comparison"]["labels"]
    tail, frozen = lab[("ECI", "tail_selected")] == "eci_hot", lab[("ECI", "frozen")] == "eci_hot"
    assert not (tail & ~frozen).any() and int(tail.sum()) == 2532


def test_the_post_registration_sensitivity_and_its_statements(step4b):
    checks = BC.sensitivity_checks(step4b["sens"])
    assert checks["passed"].all(), checks[~checks["passed"]].to_string()
    assert not step4b["sens"]["persists"]["fold0_repair"] and not step4b["sens"]["persists"]["zero_migration"]
    v = step4b["decision"]["verdicts"].set_index(["family", "procedure"])["verdict"]
    assert v[("NSD", "gap_placed")] == "advance to PR 44"              # the registered verdict is unchanged
