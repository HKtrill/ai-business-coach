"""
Arm-neutral Stage 2 obligation checks (PR 37 Commit 4, Step 4A).

Contract
1. Limits are the decimals the contract writes, compared exactly: at a
   boundary (NPV exactly 0.95, leakage exactly 0.15, precision exactly 0.30) the
   obligation holds; one count beyond it, it does not. Leakage is a ceiling.
2. At fit time an empty set, a set below its support floor or an undefined
   ratio is infeasible; a population without both classes is rejected.
3. At evaluation an undefined ratio is reported as undefined (never 0, never
   compliant), an observed violation is a violation however small, and pooled
   metrics come from summed counts, not averaged fold ratios.
4. The counts and ratios match the diagnostics registry and the shared
   ``evaluate_router``; both arms' decision tokens are accepted.
5. The RF solver's feasibility agrees with the checker band by band, and its
   chosen bands are the checker's largest feasible ones (lexicographic
   coverage). GLASS's existing union leakage cap is the checker's integer cap.
6. Fit outcomes are classified, and a claim the obligation check contradicts
   is refused.
7. The checker imports neither router.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import shared.stage2.obligations as ob
from blackbox_pipeline.models.rf_router.contract import rf_constraints_from_contract_v2
from blackbox_pipeline.models.rf_router.evaluation.router_metrics import evaluate_router
from blackbox_pipeline.models.rf_router.router.decisions import PASS1, PASS2, UNCERTAIN
from blackbox_pipeline.models.rf_router.threshhold import ThresholdSolver
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import ILPRuleSelector
from shared.stage2.contract import (
    ContractViolation, SharedOperatingContract, SharedOperatingContractV2, working_profile_path,
)
from shared.stage2.diagnostics.metrics import METRIC_DEFINITIONS, UNDEFINED
from shared.stage2.obligations import (
    OUTCOME_FEASIBLE_GREEDY, OUTCOME_FEASIBLE_OPTIMAL, OUTCOME_FEASIBLE_UNPROVEN,
    OUTCOME_INFEASIBLE, OUTCOME_INSUFFICIENT, OUTCOME_NO_SOLUTION, SATISFIED, UNDEFINED_STATUS,
    VIOLATED, ObligationInfeasible, PassCounts, RoutingCounts,
)

BANK_ROOT = Path(__file__).resolve().parents[2]


def _record() -> dict:
    return json.loads(working_profile_path().read_text(encoding="utf-8"))


def _contract(**overrides) -> SharedOperatingContractV2:
    """The working profile, with synthetic-scale overrides where a test needs them."""
    return SharedOperatingContractV2.from_dict({**_record(), **overrides})


C = _contract()                                     # 0.95 / 0.15 / None / 0.30 / 125 / 75
SMALL = _contract(min_pass1_support=5, min_pass2_support=3)


def _p1(pop_rows, pop_pos, rows, pos, contract=SMALL):
    return ob.check_pass1_fit(PassCounts(pop_rows, pop_pos, rows, pos), contract)


def _p2(pop_rows, pop_pos, rows, pos, contract=SMALL):
    return ob.check_pass2_fit(PassCounts(pop_rows, pop_pos, rows, pos), contract)


# ----------------------------------------------------------------------
# 1. exact limits and boundaries
# ----------------------------------------------------------------------
def test_limits_are_the_written_decimals_exactly():
    assert ob.exact_limit(0.15) == Fraction(3, 20) != Fraction(0.15)
    assert ob.exact_limit(0.95) == Fraction(19, 20)
    assert ob.exact_limit(0.30) == Fraction(3, 10)
    for bad in (float("nan"), float("inf"), True, "0.15", None):
        with pytest.raises(ContractViolation):
            ob.exact_limit(bad)


def test_the_leakage_limit_is_a_ceiling_met_exactly_at_the_boundary():
    # 20 positives in the population: 3 leaked is exactly 0.15.
    assert _p1(400, 20, 100, 3).result("pass1_leakage_ceiling").status == SATISFIED
    r = _p1(400, 20, 100, 4).result("pass1_leakage_ceiling")
    assert r.status == VIOLATED and r.shortfall == Fraction(4, 20) - Fraction(3, 20)
    assert _p1(400, 20, 100, 0).result("pass1_leakage_ceiling").status == SATISFIED
    assert _p1(400, 20, 100, 3).result("pass1_leakage_ceiling").shortfall == 0
    assert ob.leakage_cap_count(C, 20) == 3 and ob.leakage_cap_count(C, 3712) == 556


def test_the_npv_and_precision_floors_are_met_exactly_at_the_boundary():
    assert _p1(400, 40, 20, 1).result("pass1_npv_floor").status == SATISFIED       # 19/20
    assert _p1(400, 40, 20, 2).result("pass1_npv_floor").status == VIOLATED        # 18/20
    assert _p1(400, 40, 60, 3).result("pass1_npv_floor").status == SATISFIED       # 57/60
    assert _p2(400, 100, 10, 3).result("pass2_precision_floor").status == SATISFIED  # 3/10
    assert _p2(400, 100, 10, 2).result("pass2_precision_floor").status == VIOLATED
    r = _p2(400, 100, 100, 29).result("pass2_precision_floor")
    assert r.status == VIOLATED and r.shortfall == Fraction(1, 100)
    assert _p2(400, 100, 100, 30).result("pass2_precision_floor").shortfall == 0


def test_support_floors_bind_at_fit_time():
    assert _p1(5000, 600, 124, 2, C).result("pass1_support_floor").status == VIOLATED
    assert _p1(5000, 600, 125, 2, C).all_satisfied
    assert _p2(5000, 600, 74, 40, C).result("pass2_support_floor").status == VIOLATED
    assert _p2(5000, 600, 75, 40, C).all_satisfied
    with pytest.raises(ObligationInfeasible, match="pass1_support_floor=violated"):
        ob.require_fit(_p1(5000, 600, 124, 2, C))


def test_a_count_cap_is_checked_only_when_the_contract_sets_one():
    names = {r.obligation for r in _p1(400, 20, 100, 3).results}
    assert "pass1_leakage_count_cap" not in names
    capped = _contract(min_pass1_support=5, min_pass2_support=3, max_pass1_leakage_count=2)
    r = _p1(400, 20, 100, 3, capped)
    assert r.result("pass1_leakage_ceiling").status == SATISFIED
    assert r.result("pass1_leakage_count_cap").status == VIOLATED and not r.all_satisfied
    assert ob.leakage_cap_count(capped, 20) == 2


# ----------------------------------------------------------------------
# 2. zero support at fit time
# ----------------------------------------------------------------------
def test_an_empty_routed_set_is_infeasible_not_vacuously_satisfied():
    r1 = _p1(400, 20, 0, 0)
    assert r1.result("pass1_npv_floor").status == UNDEFINED_STATUS
    assert r1.result("pass1_support_floor").status == VIOLATED and not r1.all_satisfied
    r2 = _p2(400, 20, 0, 0)
    assert r2.result("pass2_precision_floor").status == UNDEFINED_STATUS
    assert r2.result("pass2_precision_floor").shortfall is None and not r2.all_satisfied
    with pytest.raises(ObligationInfeasible):
        ob.require_fit(r2)


@pytest.mark.parametrize("rows, pos, status", [
    (0, 0, ob.POPULATION_EMPTY), (100, 0, ob.POPULATION_NO_POSITIVES),
    (100, 100, ob.POPULATION_NO_NEGATIVES),
])
def test_a_population_without_both_classes_is_rejected(rows, pos, status):
    r = _p1(rows, pos, 0, 0)
    assert r.population_status == status and not r.all_satisfied
    if pos == 0:
        assert r.result("pass1_leakage_ceiling").status == UNDEFINED_STATUS
    with pytest.raises(ObligationInfeasible, match=f"population={status}"):
        ob.require_fit(r)


@pytest.mark.parametrize("counts", [
    (10, 2, 11, 0), (10, 2, 5, 3), (10, 12, 5, 1), (10, 9, 5, 0), (10, True, 5, 1),
    (10, 2, -1, 0), (10.0, 2, 5, 1),
])
def test_inconsistent_pass_counts_are_refused(counts):
    with pytest.raises(ContractViolation):
        PassCounts(*counts)


def test_the_checker_needs_a_schema_2_contract():
    v1 = SharedOperatingContract(contract_id="synthetic", min_pass1_npv=0.9,
                                 max_pass1_leakage_rate=0.2, max_pass1_leakage_count=None,
                                 min_pass2_precision=0.4, provenance="unit-test fixture")
    with pytest.raises(ContractViolation, match="schema-2"):
        ob.check_pass1_fit(PassCounts(10, 2, 5, 0), v1)
    with pytest.raises(ContractViolation, match="fit-time"):
        ob.require_fit(ob.observe_routing(RoutingCounts(10, 2, 3, 0, 2, 1), C))


# ----------------------------------------------------------------------
# 3. evaluation: observed, undefined, pooled
# ----------------------------------------------------------------------
def test_an_observed_violation_is_a_violation_however_small():
    # 3,712 positives: 557 leaked is 0.15005, one positive over.
    rep = ob.observe_routing(RoutingCounts(32950, 3712, 13555, 557, 8000, 2440), C)
    r = rep.result("pass1_leakage_ceiling")
    assert r.status == VIOLATED and 0 < float(r.shortfall) < 0.0003
    assert not rep.all_satisfied


def test_undefined_evaluation_metrics_stay_undefined():
    rep = ob.observe_routing(RoutingCounts(100, 10, 30, 1, 0, 0), C)
    prec = rep.result("pass2_precision_floor")
    assert prec.status == UNDEFINED_STATUS and prec.shortfall is None and prec.value is None
    assert rep.ratio("pass2_precision").value is None
    assert rep.ratio("pass2_precision").status == UNDEFINED
    assert not rep.all_satisfied
    d = rep.to_dict()
    assert d["ratios"]["pass2_precision"]["value"] is None
    assert d["results"][-1]["status"] == "undefined"
    no_pos = ob.observe_routing(RoutingCounts(100, 0, 30, 0, 10, 0), C)
    assert no_pos.population_status == ob.POPULATION_NO_POSITIVES
    assert no_pos.result("pass1_leakage_ceiling").status == UNDEFINED_STATUS


def test_evaluation_reports_support_but_does_not_judge_it():
    rep = ob.observe_routing(RoutingCounts(400, 40, 20, 1, 10, 3), C)   # far below 125 / 75
    assert {r.obligation for r in rep.results} == {
        "pass1_npv_floor", "pass1_leakage_ceiling", "pass2_precision_floor"}
    assert rep.counts.pass1_rows == 20 and rep.counts.pass2_rows == 10 and rep.all_satisfied


def test_pooled_metrics_sum_counts_and_never_average_fold_ratios():
    # Fold a: NPV 99/100; fold b: NPV 900/1000. Mean of ratios 0.945; pooled 999/1100.
    a = RoutingCounts(rows=200, positives=40, pass1_rows=100, pass1_positives=1,
                      pass2_rows=50, pass2_positives=20)
    b = RoutingCounts(rows=2000, positives=400, pass1_rows=1000, pass1_positives=100,
                      pass2_rows=300, pass2_positives=90)
    folds, pooled = ob.observe_folds({0: a, 1: b}, C)
    npv = pooled.result("pass1_npv_floor")
    assert (npv.numerator, npv.denominator) == (999, 1100)
    assert pooled.ratio("pass1_npv").value == pytest.approx(999 / 1100)
    assert pooled.ratio("pass1_npv").value != pytest.approx((0.99 + 0.90) / 2)
    assert pooled.result("pass2_precision_floor").denominator == 350
    assert ob.pool_counts([a, b]) == a + b == pooled.counts
    assert folds[0].result("pass1_npv_floor").status == SATISFIED
    assert folds[1].result("pass1_npv_floor").status == VIOLATED
    with pytest.raises(ContractViolation):
        ob.pool_counts([])


def test_routing_ratios_use_the_registry_definitions():
    c = RoutingCounts(rows=1000, positives=120, pass1_rows=400, pass1_positives=15,
                      pass2_rows=200, pass2_positives=64)
    r = ob.routing_ratios(c)
    assert set(r) == set(ob.CONTRACT_METRICS) and all(m in METRIC_DEFINITIONS for m in r)
    assert {k: (v.numerator, v.denominator) for k, v in r.items()} == {
        "pass1_npv": (385, 400), "pass1_leakage_share": (15, 120), "pass2_precision": (64, 200),
        "pass1_rate": (400, 1000), "pass2_rate": (200, 1000), "coverage": (600, 1000),
        "abstain_rate": (400, 1000), "pass2_remainder_rate": (200, 600)}
    defs = ob.contract_metric_definitions(C)
    assert defs["pass1_leakage_share"] == ("positives routed by Pass 1", "all positives")
    assert defs["coverage"] == ("rows routed Pass 1 or Pass 2", "all rows")
    assert list(defs) == list(ob.CONTRACT_METRICS)


def test_counts_from_both_arms_tokens_match_evaluate_router():
    rng = np.random.default_rng(7)
    y = (rng.random(3000) < 0.12).astype(int)
    tokens = rng.choice([PASS1, PASS2, UNCERTAIN], size=3000, p=[0.4, 0.25, 0.35])
    counts = ob.RoutingCounts.from_decisions(tokens, y)
    schema_tokens = np.array([{"pass1": "PASS_1_NOT_SUBSCRIBE", "pass2": "PASS_2_SUBSCRIBE",
                               "uncertain": "ABSTAIN"}[t] for t in tokens], dtype=object)
    assert ob.RoutingCounts.from_decisions(schema_tokens, y) == counts
    preds = np.select([tokens == PASS1, tokens == PASS2], [0, 1], default=-1)
    ref = evaluate_router(preds, np.zeros(3000), tokens, y)
    r = ob.routing_ratios(counts)
    assert r["pass1_npv"].value == pytest.approx(ref["pass1_npv"])
    assert r["pass1_leakage_share"].value == pytest.approx(ref["pass1_leakage_rate"])
    assert r["pass2_precision"].value == pytest.approx(ref["pass2_precision"])
    assert r["coverage"].value == pytest.approx(ref["coverage"])
    assert r["abstain_rate"].value == pytest.approx(ref["abstain_rate"])
    assert r["pass1_rate"].value == pytest.approx(ref["pass1_share"])
    assert r["pass2_rate"].value == pytest.approx(ref["pass2_share"])


@pytest.mark.parametrize("decisions, y", [
    (["pass1", "skip"], [0, 1]), (["pass1", "pass2"], [0, 2]), (["pass1"], [0, 1]), ([], []),
])
def test_bad_decision_inputs_are_refused(decisions, y):
    with pytest.raises(ContractViolation):
        ob.RoutingCounts.from_decisions(decisions, y)


def test_inconsistent_routing_counts_are_refused():
    for args in [(10, 2, 6, 0, 5, 1), (10, 2, 3, 2, 3, 1), (10, 2, 3, 4, 3, 0),
                 (10, 9, 2, 0, 2, 0)]:
        with pytest.raises(ContractViolation):
            RoutingCounts(*args)


# ----------------------------------------------------------------------
# 4. equivalence with the arms' existing enforcement
# ----------------------------------------------------------------------
def _boundary_population(seed):
    """Scores and labels with ties, and bands sitting exactly on the limits."""
    rng = np.random.default_rng(seed)
    n = 400
    y = (rng.random(n) < 0.15).astype(int)
    y[:20], y[20:60] = 0, 0
    y[[5, 30, 59]] = 1                     # the 60 lowest rows: NPV 57/60 = 0.95 exactly
    p = np.sort(rng.random(n)) * 0.9
    p[100:110] = p[100]                    # a tie block no threshold can split
    return p, y


@pytest.mark.parametrize("seed", range(6))
def test_the_rf_pass1_solver_agrees_with_the_checker_band_by_band(seed):
    p, y = _boundary_population(seed)
    solver = ThresholdSolver(rf_constraints_from_contract_v2(SMALL))
    res = solver.solve_pass1(p, y)
    P = int(y.sum())
    feasible_sizes = []
    for row in res.sweep.itertuples():
        rep = _p1(len(y), P, int(row.band_size), int(row.subscribers_routed))
        assert rep.all_satisfied == bool(row.feasible), (row.band_size, row.binding)
        if rep.all_satisfied:
            feasible_sizes.append(int(row.band_size))
    if feasible_sizes:
        assert res.feasible and res.band_size == max(feasible_sizes)   # max Pass 1 rows
    else:
        assert not res.feasible


@pytest.mark.parametrize("seed", range(6))
def test_the_rf_pass2_solver_agrees_with_the_checker_band_by_band(seed):
    rng = np.random.default_rng(100 + seed)
    n = 300
    y = (rng.random(n) < 0.2).astype(int)
    p = np.round(rng.random(n), 2)         # many ties
    solver = ThresholdSolver(rf_constraints_from_contract_v2(SMALL))
    res = solver.solve_pass2(p, y, n_positives_global=int(y.sum()) + 50)
    feasible_sizes = []
    for row in res.sweep.itertuples():
        rep = _p2(n, int(y.sum()), int(row.band_size), int(row.positives_in_band))
        assert rep.all_satisfied == bool(row.feasible), (row.band_size, row.binding)
        if rep.all_satisfied:
            feasible_sizes.append(int(row.band_size))
    if feasible_sizes:
        assert res.feasible and res.band_size == max(feasible_sizes)   # max Pass 2 rows
    else:
        assert not res.feasible


def test_the_rf_solver_meets_exact_boundaries_like_the_checker():
    """One band sits exactly on both Pass 1 limits: NPV 57/60 = 0.95 and leakage 3/20 = 0.15."""
    y = np.zeros(400, dtype=int)
    y[[0, 30, 59]] = 1                     # 3 of the 60 lowest-scoring rows
    y[-17:] = 1                            # 20 positives in all
    p = np.arange(len(y), dtype=float) / len(y)
    on_both = _p1(len(y), 20, 60, 3)
    assert on_both.all_satisfied
    assert on_both.result("pass1_npv_floor").shortfall == 0
    assert on_both.result("pass1_leakage_ceiling").shortfall == 0
    res = ThresholdSolver(rf_constraints_from_contract_v2(SMALL)).solve_pass1(p, y)
    sweep = res.sweep.set_index("band_size")
    assert int(sweep.loc[60, "subscribers_routed"]) == 3 and bool(sweep.loc[60, "feasible"])
    # The largest feasible band stops just before the 4th positive.
    assert res.band_size == len(y) - 17 and res.positives_in_band == 3
    assert not _p1(len(y), 20, len(y) - 16, 4).all_satisfied


def _glass_union_cap(rate, positives):
    """GLASS's existing Pass 1 union leakage cap, read back from the real selector."""
    class _Validator:
        def extract_base_feature(self, feature):
            return feature.split("__")[0]

        def has_duplicate_base_features(self, segment):
            return False

    class _Rule:
        def __init__(self, rule_id, rows, n):
            self.rule_id = rule_id
            self.segment = ((f"f{rule_id}__{rule_id}", 1),)
            self.segment_frozen = frozenset(self.segment)
            self._cached_covered_idx = set(rows)
            self.support, self.coverage, self.precision = len(rows), len(rows) / n, 1.0
            self.recall, self.predicted_class = 0.0, 0
            self.rf_confidence, self.rf_alignment = 0.5, 0.0

    n = positives + 40
    y = np.zeros(n, dtype=int)
    y[40:] = 1
    sel = ILPRuleSelector(
        min_pass1_rules=1, max_pass1_rules=2, min_precision_pass1=0.9, max_precision_pass1=1.0,
        max_subscriber_leakage_rate_pass1=1.0, max_subscriber_leakage_absolute_pass1=10 ** 9,
        min_pass2_rules=1, max_pass2_rules=2, min_precision_pass2=0.0, max_precision_pass2=1.0,
        min_recall_pass2=0.0, max_recall_pass2=1.0, min_novelty_ratio_pass1=0.2,
        min_novelty_ratio_pass2=0.2, enable_novelty_constraints=True, diversity_weight=0.35,
        lambda_rf_uncertainty=0.0, lambda_rf_misalignment=0.0)
    stub = _Validator()
    sel.validator = stub
    sel.diversity_analyzer.validator = stub
    rules = [_Rule("A", range(0, 20), n), _Rule("B", range(20, 40), n)]
    sel._optimize_pass(candidates=rules, min_rules=1, max_rules=2,
                       pass_name="Pass 1 (NOT_SUBSCRIBE)", scoring_mode="precision_first",
                       y_val=y, min_novelty_ratio=0.2, max_union_leakage_rate=rate)
    return sel.selection_report_["Pass 1 (NOT_SUBSCRIBE)"]["union_leakage_cap"]


@pytest.mark.parametrize("positives", [20, 37, 100, 2969, 2970, 3712])
def test_glass_union_leakage_cap_is_the_checkers_integer_cap(positives):
    """The one set-level obligation GLASS enforces today (Step 4B adds the others)."""
    assert _glass_union_cap(C.max_pass1_leakage_rate, positives) == ob.leakage_cap_count(
        C, positives)


# ----------------------------------------------------------------------
# 5. fit outcomes
# ----------------------------------------------------------------------
def test_fit_outcomes_carry_the_approved_dispositions():
    ok = _p1(5000, 600, 400, 20, C)
    assert ok.all_satisfied
    opt = ob.classify_fit_outcome(OUTCOME_FEASIBLE_OPTIMAL, scope="pass1", reason="exhaustive scan",
                                  report=ok)
    assert opt.disposition == ob.DISPOSITION_PRIMARY and opt.primary_eligible
    inc = ob.classify_fit_outcome(OUTCOME_FEASIBLE_UNPROVEN, scope="pass1", reason="time limit",
                                  report=ok, solver={"status": "Optimal", "sol_status": 2})
    assert inc.disposition == ob.DISPOSITION_PRIMARY_IF_DOCUMENTED
    assert not inc.documentation_complete and not inc.primary_eligible
    documented = ob.classify_fit_outcome(
        OUTCOME_FEASIBLE_UNPROVEN, scope="pass1", reason="time limit", report=ok,
        solver={"status": "Optimal", "gap": 0.004, "reproduced": True})
    assert documented.documentation_complete and documented.primary_eligible
    greedy = ob.classify_fit_outcome(OUTCOME_FEASIBLE_GREEDY, scope="pass1", reason="no incumbent",
                                     report=ok)
    assert greedy.disposition == ob.DISPOSITION_REQUIRES_REVIEW and not greedy.primary_eligible
    for outcome in (OUTCOME_INFEASIBLE, OUTCOME_NO_SOLUTION):
        stop = ob.classify_fit_outcome(outcome, scope="pass2", reason="nothing feasible")
        assert stop.disposition == ob.DISPOSITION_STOP and not stop.primary_eligible
    short = ob.classify_fit_outcome(OUTCOME_INSUFFICIENT, scope="pass2", reason="tiny remainder",
                                    report=_p2(100, 0, 0, 0))
    assert short.disposition == ob.DISPOSITION_STOP


def test_claims_the_obligation_check_contradicts_are_refused():
    ok, bad = _p1(5000, 600, 400, 20, C), _p1(5000, 600, 400, 100, C)
    with pytest.raises(ContractViolation, match="contradicted"):
        ob.classify_fit_outcome(OUTCOME_FEASIBLE_UNPROVEN, scope="pass1", reason="x", report=bad)
    with pytest.raises(ContractViolation, match="needs the fit-time obligation report"):
        ob.classify_fit_outcome(OUTCOME_FEASIBLE_GREEDY, scope="pass1", reason="x")
    with pytest.raises(ContractViolation, match="every obligation holds"):
        ob.classify_fit_outcome(OUTCOME_INFEASIBLE, scope="pass1", reason="x", report=ok)
    with pytest.raises(ContractViolation, match="adequate population and support"):
        ob.classify_fit_outcome(OUTCOME_INSUFFICIENT, scope="pass1", reason="x", report=bad)
    with pytest.raises(ContractViolation, match="does not match"):
        ob.classify_fit_outcome(OUTCOME_FEASIBLE_OPTIMAL, scope="pass2", reason="x", report=ok)
    with pytest.raises(ContractViolation, match="outcome must be one of"):
        ob.classify_fit_outcome("relaxed_and_retried", scope="pass1", reason="x")


# ----------------------------------------------------------------------
# 6. arm neutrality
# ----------------------------------------------------------------------
def test_the_checker_imports_neither_router():
    tree = ast.parse(Path(ob.__file__).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m.startswith(("glass_pipeline", "blackbox_pipeline"))]
    code = ("import sys, shared.stage2.obligations\n"
            "print(sorted(m for m in sys.modules "
            "if m.startswith(('glass_pipeline', 'blackbox_pipeline'))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=BANK_ROOT, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]"
