"""
PR 37 Commit 1 (Step 1C): the RF Pass 1 score-equivalence check. Exactly zero
disagreements are required; boundary cases are classified, never accepted;
a failure makes the RF Pass 1 score stream unsupported without affecting
decision-level diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.stage2 import diagnostics as d
from tests.support import diagnostics_world as dw


@pytest.fixture
def w():
    return dw.world()


def run(w, ref=None):
    ref = ref or dw.reference(w)
    rf = d.extract_router_view(dw.trapped(w["rf"]), ref, expected_arm="blackbox")
    cand = d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), ref)
    return rf, cand, d.check_rf_pass1_score_equivalence(rf, cand)


def _t1(w):
    return {r["fold"]: r["t1"] for r in w["rf"]["extra"]["oof_fold_summary"]}


def _row(w, state, fold=0):
    dec = np.asarray(w["rf"]["extra"]["train_decisions"])
    return int(np.flatnonzero((dec == state) & (w["fold"] == fold))[0])


def test_consistent_candidate_is_equivalent_and_usable(w):
    rf, cand, rep = run(w)
    assert rep.passed and rep.structural_passed
    assert (rep.n_disagreements, rep.n_boundary_disagreements, rep.pass2_precedence_violations) == (0, 0, 0)
    assert rep.n_rows == rf.n and rep.conclusion.startswith("EQUIVALENT")
    assert list(rep.fold_summary["disagreements"]) == [0] * 5
    s = d.require_score_stream(rf, "pass1", equivalence=rep, candidate=cand)
    assert s.space == "weighted_forest_score" and s.values is cand.values


def test_ulp_noise_away_from_t1_still_passes(w):
    p = np.asarray(w["rf_result"]["oof_proba"], dtype=float)
    w["rf_result"]["oof_proba"] = np.nextafter(p, 1.0)
    assert run(w)[2].passed


def test_one_far_disagreement_fails_and_blocks_the_score_stream_only(w):
    i = _row(w, "uncertain")
    w["rf_result"]["oof_proba"][i] = 0.0                      # far below t1, but stored not Pass 1
    rf, cand, rep = run(w)
    assert not rep.passed and rep.structural_passed
    assert (rep.n_disagreements, rep.n_boundary_disagreements) == (1, 0)
    assert rep.disagreements.iloc[0]["row"] == rf.index[i] and not rep.disagreements.iloc[0]["boundary"]
    assert rep.conclusion.startswith("UNSUPPORTED")
    with pytest.raises(d.UnsupportedDiagnostic, match="did not pass"):
        d.require_score_stream(rf, "pass1", equivalence=rep, candidate=cand)
    assert d.router_metrics(rf)["pass1_npv"].value is not None     # decisions unaffected


def test_boundary_disagreement_is_classified_but_not_accepted(w):
    i = _row(w, "pass1", fold=2)
    w["rf_result"]["oof_proba"][i] = _t1(w)[2] + 1e-13       # just above t1: would not route
    _, _, rep = run(w)
    assert not rep.passed
    assert (rep.n_disagreements, rep.n_boundary_disagreements) == (1, 1)
    assert "boundary cases are documented, not accepted" in rep.conclusion


def test_pass2_row_below_t1_is_a_precedence_violation(w):
    i = _row(w, "pass2", fold=1)
    w["rf_result"]["oof_proba"][i] = _t1(w)[1] / 2
    _, _, rep = run(w)
    assert not rep.passed and rep.pass2_precedence_violations == 1 and rep.n_disagreements == 1


@pytest.mark.parametrize("mutate, check", [
    (lambda w: w["rf_result"]["params"].__setitem__("min_samples_leaf", 10), "same forest parameters"),
    (lambda w: w["rf_result"]["params"].__setitem__("max_features", "log2"), "same forest parameters"),
    (lambda w: w["rf"]["config"].__setitem__("random_state", 7), "same forest seed"),
])
def test_structural_mismatch_fails_even_with_zero_row_disagreements(w, mutate, check):
    mutate(w)
    _, _, rep = run(w)
    assert rep.n_disagreements == 0 and not rep.passed
    assert not rep.structural[check][0] and check in rep.conclusion


def test_param_representation_does_not_matter(w):
    w["rf_result"]["params"]["n_estimators"] = 500.0
    assert run(w)[2].passed


def test_report_is_bound_to_its_candidate(w):
    rf, cand, rep = run(w)
    other = dw.world()
    other["rf_result"]["oof_proba"] = np.nextafter(np.asarray(other["rf_result"]["oof_proba"]), 1.0)
    cand2 = d.extract_rf_pass1_candidate(dw.trapped(other["rf_result"]), dw.reference(w))
    with pytest.raises(d.RouterDiagnosticError, match="another candidate"):
        d.verified_rf_pass1_score(rep, cand2)


def test_glass_view_is_not_a_two_pass_rf_router(w):
    ref = dw.reference(w)
    glass = d.extract_router_view(dw.trapped(w["glass"]), ref, expected_arm="glass")
    cand = d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), ref)
    rep = d.check_rf_pass1_score_equivalence(glass, cand)
    assert not rep.passed and not rep.structural["view is the two-pass RF Router"][0]
