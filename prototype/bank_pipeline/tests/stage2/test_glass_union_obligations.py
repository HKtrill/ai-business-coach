"""
GLASS ILP selector: set-level union obligations (PR 37 Commit 4, Step 4B).

Contract
1. Off by default. With no Step 4B argument the selector, its ILP, its reports
   and its selections are the historical ones (Pass 2 maximises union
   positives; no solver log is captured).
2. Each obligation holds on the SELECTED UNION, every row counted once:
   Pass 1 union NPV and union support, Pass 2 union precision and union
   support. A union can fail a floor every member rule meets; the ILP refuses
   it. Ratios are exact integer comparisons on the decimal the limit is
   written as, never float products.
3. The atom indicators are exact in both directions: a covered atom is
   counted even when it holds no "good" row (zero-positive atoms under a
   precision floor), and an uncovered atom is never counted.
4. The ILP optimum equals exhaustive enumeration, for Pass 1 and for Pass 2
   under both objectives, and infeasibility is proven where enumeration
   finds no feasible set.
5. The opt-in Pass 2 "rows" objective maximises flagged rows and is valid only
   with a union precision floor.
6. No silent fallback: empty unions, unmet floors and unusable solver results
   either raise or return a selection validated against every obligation.
   Incumbents and greedy results are checked independently.
7. Solves are deterministic for fixed inputs; under obligations the solver
   log's objective and bound are recorded (the gap PuLP does not return).
8. The overlap diagnostics' optimisation is exact: identical Jaccard values,
   identical printed output, one match per rule instead of one per pair.
9. (slow, opt-in) A real fit under the obligations meets every one of them
   on its training rows.

Rules are the duck-typed fakes of test_ilp_rule_selector_union.py.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import time
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector as ilp_mod
from glass_pipeline.glass_router.core.config import (
    PASS2_OBJECTIVE_POSITIVES, PASS2_OBJECTIVE_ROWS, union_obligation_errors,
)
from glass_pipeline.glass_router.ilp_rule_selector.ilp_builder import ILPBuilder, parse_cbc_log
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import (
    INCUMBENT, PROVEN, REJECTED, SelectorInfeasibleError,
)
from glass_pipeline.glass_router.ilp_rule_selector.utils import (
    UnionObligations, exact_fraction, get_covered_array, meets_novelty_floor, unique_counts,
    union_obligation_violations,
)
from glass_pipeline.glass_router.rule_evaluator.overlap_analyzer import OverlapAnalyzer
from glass_pipeline.glass_router.rule_evaluator.rule_matcher import RuleMatcher
from shared.stage2.obligations import exact_limit
from tests.stage2.test_ilp_rule_selector_union import (
    P1, P2, FakeRule, _StubSelected, _StubValidator, ids, make_selector, no_solution,
    real_but_time_limited, rng, script_solver, zero_incumbent,
)

STEP_4B_REPORT_KEYS = {"union_objective", "union_ratio", "union_ratio_floor",
                       "union_support_floor", "union_indicators", "union_rows", "union_good",
                       "selected_rule_ids"}


def run(sel, rules, y, lo, hi, pass_name=P1, ratio=None, support=None, objective=None,
        novelty=0.20, leak_rate=None, max_base_reuse=None):
    mode = "precision_first" if pass_name == P1 else "recall_first"
    return sel._optimize_pass(
        candidates=rules, min_rules=lo, max_rules=hi, pass_name=pass_name, scoring_mode=mode,
        y_val=y, max_base_reuse=max_base_reuse, min_novelty_ratio=novelty,
        max_union_leakage_rate=leak_rate, min_union_ratio=ratio, min_union_support=support,
        union_objective=objective,
    )


def union_of(rules, n):
    covered = np.zeros(n, dtype=bool)
    for r in rules:
        covered[get_covered_array(r, n)] = True
    return covered


def good_ratio(rules, y, good_label):
    covered = union_of(rules, len(y))
    return Fraction(int((y[covered] == good_label).sum()), int(covered.sum()))


def p2(rule_id, rows, n, **kw):
    return FakeRule(rule_id, rows, n, predicted_class=1, **kw)


# ----------------------------------------------------------------------------
# shared instances
# ----------------------------------------------------------------------------
def _npv_case():
    """A and B each have NPV >= 0.75; their union has NPV 9/13 < 0.75."""
    n = 40
    y = np.zeros(n, int)
    y[[8, 9, 10, 11]] = 1
    A = FakeRule("A", rng(0, 9), n)                  # 8/10 = 0.80
    B = FakeRule("B", rng(0, 7) + rng(10, 12), n)    # 9/11 = 0.818
    return n, y, A, B


def _precision_case():
    """Pass 2: C holds no positive at all (a zero-positive atom)."""
    n = 200
    y = np.zeros(n, int)
    y[0:6] = 1
    y[100:103] = 1
    return n, y, [p2("A", rng(0, 19), n), p2("B", rng(100, 119), n), p2("C", rng(150, 189), n)]


# ============================================================================
# 1. off by default
# ============================================================================
def test_defaults_are_off_and_the_historical_model_is_built():
    sel = make_selector()
    assert (sel.min_union_npv_pass1, sel.min_union_support_pass1, sel.min_union_precision_pass2,
            sel.min_union_support_pass2) == (None, None, None, None)
    assert sel.pass2_union_objective == PASS2_OBJECTIVE_POSITIVES
    assert sel.ilp_builder.record_solver_log is False
    n, y, A, B = _npv_case()
    chosen = run(sel, [A, B], y, 1, 2)
    rep = sel.selection_report_[P1]
    assert ids(chosen) == {"A", "B"}
    assert not STEP_4B_REPORT_KEYS & set(rep)
    assert "solver_log" not in rep["phase1"] and "solver_log" not in rep["phase2"]


def test_explicit_none_arguments_equal_the_historical_call():
    n, y, rules = _precision_case()
    hist = make_selector()
    a = run(hist, rules, y, 1, 3, pass_name=P2)
    explicit = make_selector(min_union_npv_pass1=None, min_union_support_pass1=None,
                             min_union_precision_pass2=None, min_union_support_pass2=None,
                             pass2_union_objective=PASS2_OBJECTIVE_POSITIVES)
    b = run(explicit, rules, y, 1, 3, pass_name=P2, objective=PASS2_OBJECTIVE_POSITIVES)
    assert [r.rule_id for r in a] == [r.rule_id for r in b]
    assert hist.selection_report_[P2] == explicit.selection_report_[P2]
    assert hist.selection_report_[P2]["union_weight"] == "positives"


def test_the_historical_pass2_objective_maximises_positives_not_rows():
    """Under a precision floor, "positives" and "rows" can choose different sets."""
    n = 300
    y = np.zeros(n, int)
    y[0:12] = 1                                       # P: 12 positives in 30 rows (0.40)
    y[100:105] = 1                                    # Q: 5 positives in 60 rows (0.083)
    y[200:217] = 1                                    # R: 17 positives in 40 rows (0.425)
    P = p2("P", rng(0, 29), n)
    Q = p2("Q", rng(100, 159), n)
    R = p2("R", rng(200, 239), n)
    S = p2("S", rng(200, 215) + rng(240, 299), n)     # 16 positives in 76 rows (0.21)
    floor = 0.25
    by_pos = run(make_selector(min_union_precision_pass2=floor), [P, Q, R, S], y, 1, 1,
                 pass_name=P2, ratio=floor, objective=PASS2_OBJECTIVE_POSITIVES)
    by_rows = run(make_selector(min_union_precision_pass2=floor,
                                pass2_union_objective=PASS2_OBJECTIVE_ROWS),
                  [P, Q, R, S], y, 1, 1, pass_name=P2, ratio=floor,
                  objective=PASS2_OBJECTIVE_ROWS)
    assert ids(by_pos) == {"R"}       # most positives (17) among feasible singletons
    assert ids(by_rows) == {"R"}      # 40 rows; S (76 rows) fails the floor
    by_rows_lo = run(make_selector(min_union_precision_pass2=0.20,
                                   pass2_union_objective=PASS2_OBJECTIVE_ROWS),
                     [P, Q, R, S], y, 1, 1, pass_name=P2, ratio=0.20,
                     objective=PASS2_OBJECTIVE_ROWS)
    by_pos_lo = run(make_selector(min_union_precision_pass2=0.20), [P, Q, R, S], y, 1, 1,
                    pass_name=P2, ratio=0.20)
    assert ids(by_rows_lo) == {"S"}   # 76 rows at 0.21
    assert ids(by_pos_lo) == {"R"}    # 17 positives beat S's 16 under "positives"


# ============================================================================
# 2. obligations hold on the union
# ============================================================================
def test_pass1_union_npv_floor_refuses_a_union_its_rules_individually_meet():
    n, y, A, B = _npv_case()
    assert good_ratio([A], y, 0) >= Fraction(3, 4) and good_ratio([B], y, 0) >= Fraction(3, 4)
    assert good_ratio([A, B], y, 0) == Fraction(9, 13) < Fraction(3, 4)
    sel = make_selector(min_union_npv_pass1=0.75)
    chosen = run(sel, [A, B], y, 1, 2, ratio=0.75)
    rep = sel.selection_report_[P1]
    assert ids(chosen) == {"B"}
    assert (rep["union_rows"], rep["union_good"]) == (11, 9)
    assert rep["union_ratio"] == "NPV" and rep["union_ratio_floor"] == "3/4"
    assert rep["lexicographic_guarantee"] == "proven"


def test_pass2_union_precision_floor_counts_a_zero_positive_atom():
    n, y, rules = _precision_case()
    sel = make_selector(min_union_precision_pass2=0.2, min_union_support_pass2=5,
                        pass2_union_objective=PASS2_OBJECTIVE_ROWS)
    chosen = run(sel, rules, y, 1, 3, pass_name=P2, ratio=0.2, support=5,
                 objective=PASS2_OBJECTIVE_ROWS)
    rep = sel.selection_report_[P2]
    assert ids(chosen) == {"A", "B"}                   # adding C: 9/80 < 1/5
    assert (rep["union_rows"], rep["union_good"]) == (40, 9)
    assert rep["union_objective"] == "rows" and rep["union_ratio"] == "precision"


def test_upper_links_stop_an_unselected_atom_from_lending_its_positives():
    """H1 and H2 cover the same 40 positives (one multi-rule atom); C has 100
    negatives. Without y_a <= sum x_j, selecting C alone could 'count' H's
    positives and pass the floor; the true optimum is one H rule."""
    n = 200
    y = np.zeros(n, int)
    y[0:40] = 1
    H1 = p2("H1", rng(0, 39), n, base="h1")
    H2 = p2("H2", rng(0, 39), n, base="h2")
    C = p2("C", rng(40, 139), n, base="c")
    sel = make_selector(min_union_precision_pass2=0.2, pass2_union_objective=PASS2_OBJECTIVE_ROWS)
    chosen = run(sel, [H1, H2, C], y, 1, 1, pass_name=P2, ratio=0.2,
                 objective=PASS2_OBJECTIVE_ROWS)
    assert len(chosen) == 1 and ids(chosen) <= {"H1", "H2"}
    assert sel.selection_report_[P2]["union_rows"] == 40
    assert sel.selection_report_[P2]["union_indicators"] >= 1


def test_lower_links_count_a_covered_atom_with_no_positives():
    """C1 and C2 share 100 negatives (a multi-rule zero-positive atom). Without
    y_a >= x_j, A + C1 could hide those negatives and pass the floor."""
    n = 200
    y = np.zeros(n, int)
    y[0:20] = 1
    A = p2("A", rng(0, 19), n, base="a")
    C1 = p2("C1", rng(20, 119), n, base="c1")
    C2 = p2("C2", rng(20, 119), n, base="c2")
    sel = make_selector(min_union_precision_pass2=0.2, pass2_union_objective=PASS2_OBJECTIVE_ROWS)
    chosen = run(sel, [A, C1, C2], y, 1, 2, pass_name=P2, ratio=0.2,
                 objective=PASS2_OBJECTIVE_ROWS)
    assert ids(chosen) == {"A"}
    assert good_ratio(chosen, y, 1) >= Fraction(1, 5)


def test_ratio_floors_are_exact_not_float_products():
    """7 positives in 50 rows meets 0.14 exactly, though 0.14 * 50 > 7 in floats."""
    assert 7 < 0.14 * 50
    n = 100
    y = np.zeros(n, int)
    y[0:7] = 1
    A = p2("A", rng(0, 49), n)
    sel = make_selector(min_union_precision_pass2=0.14)
    assert ids(run(sel, [A], y, 1, 1, pass_name=P2, ratio=0.14)) == {"A"}
    assert sel.selection_report_[P2]["union_ratio_floor"] == "7/50"
    y[6] = 0                                          # 6/50 < 7/50
    with pytest.raises(SelectorInfeasibleError) as err:
        run(make_selector(min_union_precision_pass2=0.14), [A], y, 1, 1, pass_name=P2, ratio=0.14)
    assert err.value.proven is True


@pytest.mark.parametrize("positives, feasible", [(1, True), (2, False)])
def test_npv_floor_boundary(positives, feasible):
    """20 rows: 1 positive is NPV 19/20 = 0.95 exactly (feasible); 2 is 0.90."""
    n = 60
    y = np.zeros(n, int)
    y[:positives] = 1
    A = FakeRule("A", rng(0, 19), n)
    sel = make_selector(min_union_npv_pass1=0.95)
    if feasible:
        assert ids(run(sel, [A], y, 1, 1, ratio=0.95)) == {"A"}
    else:
        with pytest.raises(SelectorInfeasibleError) as err:
            run(sel, [A], y, 1, 1, ratio=0.95)
        assert err.value.proven is True and "union NPV" in str(err.value)


@pytest.mark.parametrize("support, feasible", [(30, True), (31, False)])
def test_union_support_floor_boundary(support, feasible):
    n = 100
    y = np.zeros(n, int)
    A = FakeRule("A", rng(0, 19), n)
    B = FakeRule("B", rng(20, 29), n)
    sel = make_selector(min_union_support_pass1=support)
    if feasible:
        chosen = run(sel, [A, B], y, 1, 2, support=support)
        assert ids(chosen) == {"A", "B"} and sel.selection_report_[P1]["union_rows"] == 30
    else:
        with pytest.raises(SelectorInfeasibleError) as err:
            run(sel, [A, B], y, 1, 2, support=support)
        assert err.value.proven is True and "union rows >= 31" in str(err.value)


def test_obligations_combine_with_the_union_leakage_cap():
    n = 1000
    y = np.zeros(n, int)
    y[0:4] = 1
    y[100:104] = 1
    y[900:942] = 1                                    # 50 positives -> cap 5 at 0.10
    A = FakeRule("A", rng(0, 99), n)
    B = FakeRule("B", rng(100, 199), n)
    C = FakeRule("C", rng(200, 259), n)
    sel = make_selector(max_union_leakage_rate_pass1=0.10, min_union_npv_pass1=0.95,
                        min_union_support_pass1=150)
    chosen = run(sel, [A, B, C], y, 1, 3, ratio=0.95, support=150, leak_rate=0.10)
    rep = sel.selection_report_[P1]
    assert ids(chosen) == {"A", "C"}
    assert rep["union_positives"] == 4 <= rep["union_leakage_cap"] == 5
    assert (rep["union_rows"], rep["union_good"]) == (160, 156)


# ============================================================================
# 3. empty unions, infeasibility and no silent fallback
# ============================================================================
def test_an_empty_union_never_meets_set_level_obligations():
    n = 50
    y = np.zeros(n, int)
    weak = FakeRule("W", rng(0, 9), n, precision=0.5)  # fails the 0.90 per-rule gate
    hist = make_selector()
    assert run(hist, [weak], y, 0, 2) == []
    assert hist.selection_report_[P1]["source"].startswith("empty")
    for kwargs in (dict(ratio=0.9), dict(support=1)):
        sel = make_selector()
        with pytest.raises(SelectorInfeasibleError) as err:
            run(sel, [weak], y, 0, 2, **kwargs)
        assert err.value.proven is True
        assert sel.selection_report_[P1]["stage"] == "quality gates"
    with pytest.raises(SelectorInfeasibleError, match="candidate generation"):
        run(make_selector(), [], y, 0, 2, support=10)


def test_greedy_fallback_respects_union_floors():
    n, y, A, B = _npv_case()
    sel = make_selector(min_union_npv_pass1=0.75)
    script_solver(sel, [no_solution(sel.ilp_builder)])
    chosen = run(sel, [A, B], y, 1, 2, ratio=0.75)
    rep = sel.selection_report_[P1]
    assert rep["source"] == "greedy_fallback"
    assert good_ratio(chosen, y, 0) >= Fraction(3, 4)
    assert rep["selected_rule_ids"] == [r.rule_id for r in chosen]


def test_greedy_fallback_below_the_support_floor_raises_unproven():
    n, y, A, B = _npv_case()
    sel = make_selector(min_union_support_pass1=12)
    script_solver(sel, [no_solution(sel.ilp_builder)])
    with pytest.raises(SelectorInfeasibleError) as err:
        run(sel, [A, B], y, 1, 2, ratio=0.75, support=12)  # 11 rows is the best feasible union
    assert err.value.proven is False
    assert "support floor 12" in str(err.value)
    assert sel.selection_report_[P1]["source"] == "infeasible"


def test_an_unproven_incumbent_violating_a_union_floor_is_rejected():
    n, y, A, B = _npv_case()
    sel = make_selector(min_union_npv_pass1=0.75)

    def all_selected(prob, real):
        for v in prob.variables():
            v.varValue = 1.0
        sel.ilp_builder.last_sol_status = 2
        return "Optimal"

    script_solver(sel, [all_selected, no_solution(sel.ilp_builder)])
    chosen = run(sel, [A, B], y, 1, 2, ratio=0.75)
    rep = sel.selection_report_[P1]
    assert rep["phase1"]["state"] == REJECTED
    assert any("union NPV 9/13" in p for p in rep["phase1"]["problems"])
    assert rep["source"] == "greedy_fallback" and ids(chosen) == {"B"}


def test_a_validated_time_limited_incumbent_is_kept_but_marked_unproven():
    n, y, rules = _precision_case()
    sel = make_selector(min_union_precision_pass2=0.2, pass2_union_objective=PASS2_OBJECTIVE_ROWS)
    script_solver(sel, [real_but_time_limited(sel.ilp_builder), "real"])
    chosen = run(sel, rules, y, 1, 3, pass_name=P2, ratio=0.2, objective=PASS2_OBJECTIVE_ROWS)
    rep = sel.selection_report_[P2]
    assert rep["phase1"]["state"] == INCUMBENT and rep["lexicographic_guarantee"] == "UNPROVEN"
    assert not sel._hard_constraint_violations(
        chosen, 1, 3, None, (0.2, 1), n, None,
        set_level=sel._set_level_spec("recall_first", y, 0.2, None))


def test_a_proven_optimum_violating_a_union_floor_is_a_formulation_error():
    n, y, A, B = _npv_case()
    sel = make_selector(min_union_npv_pass1=0.75)

    def proven_all(prob, real):
        for v in prob.variables():
            v.varValue = 1.0
        sel.ilp_builder.last_sol_status = 1
        return "Optimal"

    script_solver(sel, [proven_all])
    with pytest.raises(RuntimeError, match="fails independent validation"):
        run(sel, [A, B], y, 1, 2, ratio=0.75)


# ============================================================================
# 4. exhaustive-search equivalence
# ============================================================================
def _brute_force(rules, y, lo, hi, novelty, reuse, *, good_label, weight, floor=None,
                 support=1, leak_cap=None):
    n = len(y)
    arrays = {r.rule_id: get_covered_array(r, n) for r in rules}
    validator = _StubValidator()
    best = None
    for k in range(lo, hi + 1):
        for combo in itertools.combinations(rules, k):
            bases = [validator.extract_base_feature(f) for r in combo for f, _ in r.segment]
            if any(bases.count(b) > reuse for b in set(bases)):
                continue
            arrs = [arrays[r.rule_id] for r in combo]
            if not all(meets_novelty_floor(u, a.size, novelty, 1)
                       for u, a in zip(unique_counts(arrs, n), arrs)):
                continue
            covered = union_of(combo, n)
            rows = int(covered.sum())
            good = int((y[covered] == good_label).sum())
            if rows < support or (floor is not None and good * floor.denominator
                                  < floor.numerator * rows):
                continue
            if leak_cap is not None and int(y[covered].sum()) > leak_cap:
                continue
            w = weight(covered)
            raw = sum(weight(np.isin(np.arange(n), a)) for a in arrs)
            key = (w, -(raw - w))
            if best is None or key > best:
                best = key
    return best


def _random_rules(r, n, k, predicted_class):
    rules = []
    for j in range(k):
        start = r.randint(0, n - 15)
        rows = set(range(start, start + r.randint(5, 16))) | set(r.randint(0, n, r.randint(0, 6)))
        rules.append(FakeRule(f"r{j}", rows, n, precision=float(r.uniform(0.93, 1.0)),
                              predicted_class=predicted_class, base=f"b{r.randint(0, 3)}"))
    return rules


def test_pass1_npv_support_and_leakage_match_exhaustive_search():
    n = 60
    seen = {"feasible": 0, "infeasible": 0}
    for seed in range(36):
        r = np.random.RandomState(900 + seed)
        y = (r.rand(n) < 0.12).astype(int)
        rules = _random_rules(r, n, 8, 0)
        npv = [0.85, 0.90, 0.95][seed % 3]
        support = [1, 15, 25, 35][seed % 4]
        rate = [None, 0.5][seed % 2]
        cap = None if rate is None else (Fraction(rate).numerator * int(y.sum())) // Fraction(rate).denominator
        lo, hi = [(1, 3), (2, 4)][seed % 2]
        expected = _brute_force(rules, y, lo, hi, 0.2, 2, good_label=0,
                                weight=lambda c: int(c.sum()), floor=exact_fraction(npv),
                                support=support, leak_cap=cap)
        sel = make_selector(min_union_npv_pass1=npv, min_union_support_pass1=support)
        if expected is None:
            with pytest.raises(SelectorInfeasibleError) as err:
                run(sel, rules, y, lo, hi, ratio=npv, support=support, leak_rate=rate,
                    max_base_reuse=2)
            assert err.value.proven is True, f"seed {seed}"
            seen["infeasible"] += 1
            continue
        chosen = run(sel, rules, y, lo, hi, ratio=npv, support=support, leak_rate=rate,
                     max_base_reuse=2)
        covered = union_of(chosen, n)
        raw = sum(get_covered_array(c, n).size for c in chosen)
        assert (int(covered.sum()), -(raw - int(covered.sum()))) == expected, f"seed {seed}"
        assert sel.selection_report_[P1]["lexicographic_guarantee"] == "proven"
        seen["feasible"] += 1
    assert seen["feasible"] >= 8 and seen["infeasible"] >= 4, seen


@pytest.mark.parametrize("objective", [PASS2_OBJECTIVE_POSITIVES, PASS2_OBJECTIVE_ROWS])
def test_pass2_precision_and_support_match_exhaustive_search(objective):
    n = 60
    seen = {"feasible": 0, "infeasible": 0}
    for seed in range(36):
        r = np.random.RandomState(1700 + seed)
        y = (r.rand(n) < 0.3).astype(int)
        rules = _random_rules(r, n, 8, 1)
        prec = [0.25, 0.35, 0.45][seed % 3]
        support = [1, 12, 22][seed % 3]
        lo, hi = [(1, 3), (2, 4)][seed % 2]
        if objective == PASS2_OBJECTIVE_ROWS:
            weight = lambda c: int(c.sum())
        else:
            weight = lambda c, y=y: int(y[c].sum())
        expected = _brute_force(rules, y, lo, hi, 0.2, 2, good_label=1, weight=weight,
                                floor=exact_fraction(prec), support=support)
        sel = make_selector(min_union_precision_pass2=prec, min_union_support_pass2=support,
                            pass2_union_objective=objective)
        if expected is None:
            with pytest.raises(SelectorInfeasibleError) as err:
                run(sel, rules, y, lo, hi, pass_name=P2, ratio=prec, support=support,
                    objective=objective, max_base_reuse=2)
            assert err.value.proven is True, f"seed {seed}"
            seen["infeasible"] += 1
            continue
        chosen = run(sel, rules, y, lo, hi, pass_name=P2, ratio=prec, support=support,
                     objective=objective, max_base_reuse=2)
        covered = union_of(chosen, n)
        raw = sum(weight(np.isin(np.arange(n), get_covered_array(c, n))) for c in chosen)
        assert (weight(covered), -(raw - weight(covered))) == expected, f"seed {seed}"
        seen["feasible"] += 1
    assert seen["feasible"] >= 8 and seen["infeasible"] >= 3, seen


# ============================================================================
# 5. configuration
# ============================================================================
@pytest.mark.parametrize("kwargs, match", [
    (dict(min_union_npv_pass1=0.0), "min_union_npv_pass1"),
    (dict(min_union_npv_pass1=1.5), "min_union_npv_pass1"),
    (dict(min_union_precision_pass2=float("nan")), "min_union_precision_pass2"),
    (dict(min_union_precision_pass2=True), "min_union_precision_pass2"),
    (dict(min_union_support_pass1=0), "min_union_support_pass1"),
    (dict(min_union_support_pass2=2.5), "min_union_support_pass2"),
    (dict(min_union_support_pass2=True), "min_union_support_pass2"),
    (dict(pass2_union_objective="coverage"), "pass2_union_objective"),
    (dict(pass2_union_objective=PASS2_OBJECTIVE_ROWS), "needs min_union_precision_pass2"),
])
def test_invalid_union_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_selector(**kwargs)
    full = dict(min_union_npv_pass1=None, min_union_support_pass1=None,
                min_union_precision_pass2=None, min_union_support_pass2=None,
                pass2_union_objective=PASS2_OBJECTIVE_POSITIVES)
    assert any(match in e for e in union_obligation_errors(**{**full, **kwargs}))


def test_the_rows_objective_is_a_pass2_option_with_a_precision_floor():
    n, y, A, B = _npv_case()
    with pytest.raises(ValueError, match="'rows' objective"):
        run(make_selector(), [A, B], y, 1, 2, ratio=0.75, objective=PASS2_OBJECTIVE_ROWS)
    n, y, rules = _precision_case()
    with pytest.raises(ValueError, match="'rows' objective"):
        run(make_selector(), rules, y, 1, 3, pass_name=P2, objective=PASS2_OBJECTIVE_ROWS)
    with pytest.raises(ValueError, match="union_objective"):
        run(make_selector(), rules, y, 1, 3, pass_name=P2, ratio=0.2, objective="flagged")


def test_select_rules_passes_each_pass_its_own_obligations(monkeypatch):
    monkeypatch.setattr(ilp_mod, "SelectedRule", _StubSelected)
    n, y, A, B = _npv_case()
    n2, y2, rules2 = _precision_case()
    sel = make_selector(min_union_npv_pass1=0.75, min_union_precision_pass2=0.2,
                        min_union_support_pass2=5, pass2_union_objective=PASS2_OBJECTIVE_ROWS)
    out = sel.select_rules([A, B], y_val=y, passes=("pass1",))
    assert ids(out["pass1_rules"]) == {"B"}
    out2 = sel.select_rules(rules2, y_val=y2, passes=("pass2",))
    rep = sel.selection_report_[P2]
    assert ids(out2["pass2_rules"]) == {"A", "B"}
    assert (rep["union_ratio_floor"], rep["union_support_floor"], rep["union_objective"]) == (
        "1/5", 5, "rows")


# ============================================================================
# 6. exact arithmetic shared with the arm-neutral checker
# ============================================================================
def test_exact_fraction_reads_limits_like_the_contract_checker():
    for v in (0.95, 0.15, 0.3, 0.30, 0.7, 0.125, 1.0, 1e-3, 0.333, 0.9999):
        assert exact_fraction(v) == exact_limit(v)
    assert exact_fraction(0.15) == Fraction(3, 20) and exact_fraction(0.95) == Fraction(19, 20)


def test_union_leakage_cap_formula_equals_the_exact_cap_at_the_contract_rate():
    p = np.arange(0, 500_001, dtype=np.int64)
    float_cap = np.floor(0.15 * p + 1e-9).astype(np.int64)
    assert np.array_equal(float_cap, (3 * p) // 20)


def test_union_obligation_violations_is_exact_and_refuses_empty_unions():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    spec = UnionObligations(good=y == 1, floor=Fraction(3, 10), support=10, ratio_name="precision")
    assert union_obligation_violations(np.ones(10, bool), spec) == []
    short = UnionObligations(good=y == 1, floor=Fraction(3, 10), support=11, ratio_name="precision")
    assert union_obligation_violations(np.ones(10, bool), short) == [
        "union rows 10 below the support floor 11"]
    assert "empty union" in union_obligation_violations(np.zeros(10, bool), spec)[-1]


# ============================================================================
# 7. determinism, tie-breaking and the solver log
# ============================================================================
def test_repeated_solves_are_identical():
    n = 60
    r = np.random.RandomState(3)
    y = (r.rand(n) < 0.3).astype(int)
    rules = _random_rules(r, n, 8, 1)
    out = []
    for _ in range(3):
        sel = make_selector(min_union_precision_pass2=0.3, min_union_support_pass2=10,
                            pass2_union_objective=PASS2_OBJECTIVE_ROWS)
        chosen = run(sel, rules, y, 1, 4, pass_name=P2, ratio=0.3, support=10,
                     objective=PASS2_OBJECTIVE_ROWS, max_base_reuse=2)
        rep = sel.selection_report_[P2]
        log = rep["phase1"]["solver_log"]
        out.append(([c.rule_id for c in chosen], rep["union_rows"], rep["union_good"],
                    log["objective"], log["bound"]))
    assert out[0] == out[1] == out[2]


def test_exact_ties_resolve_to_a_lexicographic_optimum():
    """Twins with different bases tie on every objective term but quality;
    the higher-quality twin is chosen, whatever the candidate order."""
    n = 50
    y = np.zeros(n, int)
    lo_q = FakeRule("L", rng(0, 19), n, precision=0.94, base="l")
    hi_q = FakeRule("H", rng(0, 19), n, precision=0.99, base="h")
    for order in ([lo_q, hi_q], [hi_q, lo_q]):
        sel = make_selector(min_union_npv_pass1=0.9)
        assert ids(run(sel, order, y, 1, 2, ratio=0.9)) == {"H"}


def test_the_solver_log_records_objective_and_bound_under_obligations():
    n, y, rules = _precision_case()
    sel = make_selector(min_union_precision_pass2=0.2, pass2_union_objective=PASS2_OBJECTIVE_ROWS)
    assert sel.ilp_builder.record_solver_log is True
    run(sel, rules, y, 1, 3, pass_name=P2, ratio=0.2, objective=PASS2_OBJECTIVE_ROWS)
    rep = sel.selection_report_[P2]
    log = rep["phase1"]["solver_log"]
    assert rep["phase1"]["state"] == PROVEN
    assert log["source"] == "cbc_log" and log["result"] == "Optimal solution found"
    assert log["objective"] == rep["union_rows"] == 40
    assert log["bound"] == 40 and log["bound_basis"] == "proven_optimal" and log["gap_rel"] == 0.0
    n, y, A, B = _npv_case()                     # a model CBC solves in preprocessing
    sel = make_selector(min_union_npv_pass1=0.75)
    run(sel, [A, B], y, 1, 2, ratio=0.75)
    log = sel.selection_report_[P1]["phase1"]["solver_log"]
    assert (log["objective"], log["bound"], log["gap_rel"]) == (11.0, 11.0, 0.0)


def test_a_scripted_solve_leaves_no_stale_solver_log():
    n, y, A, B = _npv_case()
    sel = make_selector(min_union_npv_pass1=0.75)
    run(sel, [A, B], y, 1, 2, ratio=0.75)                       # a real solve fills the log
    script_solver(sel, [no_solution(sel.ilp_builder)])
    run(sel, [A, B], y, 1, 2, ratio=0.75)
    assert sel.selection_report_[P1]["phase1"]["solver_log"] is None


def test_parse_cbc_log_reads_only_what_the_log_states():
    partial = ("Cbc0005I Partial search - best objective -38919 (best possible -39292.167), "
               "took 225732 iterations and 25173 nodes (2.02 seconds)\n"
               "Result - Stopped on time limit\n")
    got = parse_cbc_log(partial, maximize=True)
    assert (got["objective"], got["bound"], got["result"], got["bound_basis"]) == (
        38919.0, 39292.167, "Stopped on time limit", "search")
    assert got["gap_abs"] == pytest.approx(373.167) and got["gap_rel"] == pytest.approx(373.167 / 38919)
    done = parse_cbc_log("Cbc0001I Search completed - best objective 12, took 3 iterations\n"
                         "Result - Optimal solution found\n", maximize=False)
    assert (done["objective"], done["bound"], done["gap_rel"]) == (12.0, 12.0, 0.0)
    none = parse_cbc_log("Cbc0005I Partial search - best objective 1e+50 (best possible -40), "
                         "took 1\n", maximize=True)
    assert none["objective"] is None and none["gap_rel"] is None and none["bound"] == 40.0
    assert parse_cbc_log("", maximize=True)["bound"] is None
    presolved = parse_cbc_log("Result - Optimal solution found\n\nObjective value:   11.0\n",
                              maximize=True)
    assert (presolved["objective"], presolved["bound"], presolved["bound_basis"]) == (
        11.0, 11.0, "proven_optimal")
    stopped = parse_cbc_log("Result - Stopped on time limit\n\nObjective value:   9.0\n",
                            maximize=True)
    assert stopped["objective"] is None and stopped["bound"] is None   # no bound stated
    infeasible = parse_cbc_log("Result - Problem proven infeasible\n", maximize=True)
    assert infeasible["result"] == "Problem proven infeasible" and infeasible["objective"] is None


def test_record_solver_log_is_off_for_the_builder_by_default():
    assert ILPBuilder(0.0, 0.0).record_solver_log is False


# ============================================================================
# 8. overlap diagnostics: one match per rule, identical values
# ============================================================================
class _CountingMatcher(RuleMatcher):
    def __init__(self):
        super().__init__(segment_builder=None)
        self.calls = 0

    def match_rule(self, rule, segments_df):
        self.calls += 1
        return super().match_rule(rule, segments_df)


class _SegRule:
    def __init__(self, segment, predicted_class=0):
        self.segment = tuple(segment)
        self.predicted_class = predicted_class


def _legacy_pairwise(matcher, rules, segments_df):
    """The pre-Step-4B loop of compute_overlap_diagnostics, verbatim."""
    overlaps = []
    for i, r1 in enumerate(rules):
        mask1 = matcher.match_rule(r1, segments_df)
        for r2 in rules[i + 1:]:
            mask2 = matcher.match_rule(r2, segments_df)
            intersection = (mask1 & mask2).sum()
            union = (mask1 | mask2).sum()
            overlaps.append(intersection / union if union > 0 else 0)
    return overlaps


def _overlap_world(seed, n_rows=400, n_rules=30):
    r = np.random.RandomState(seed)
    df = pd.DataFrame({f"f{k}": r.randint(0, 3, n_rows) for k in range(5)},
                      index=r.permutation(n_rows) + 1000)
    rules = []
    for j in range(n_rules):
        k = r.randint(1, 4)
        feats = r.choice(5, k, replace=False)
        rules.append(_SegRule([(f"f{f}", int(r.randint(0, 3))) for f in feats],
                              predicted_class=j % 2))
    rules.append(_SegRule([("missing_feature", 1)]))        # matches nothing
    rules.append(_SegRule([("f0", 9)]))                      # matches nothing
    rules.append(_SegRule([("f0", 9)]))                      # empty twin: union 0 -> 0
    rules.append(rules[0])                                   # duplicate: Jaccard 1
    return df, rules


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pairwise_jaccards_equal_the_historical_loop_exactly(seed):
    df, rules = _overlap_world(seed)
    analyzer = OverlapAnalyzer(matcher=_CountingMatcher())
    legacy = _legacy_pairwise(RuleMatcher(None), rules, df)
    new = analyzer.pairwise_jaccards(rules, df)
    assert len(new) == len(legacy) == len(rules) * (len(rules) - 1) // 2
    assert all(type(a) is type(b) and a == b for a, b in zip(new, legacy))
    assert np.mean(new).tobytes() == np.mean(legacy).tobytes()
    assert analyzer.matcher.calls == len(rules)


def test_overlap_diagnostics_print_the_same_report():
    df, rules = _overlap_world(5)
    new_out, old_out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(new_out):
        OverlapAnalyzer(matcher=RuleMatcher(None)).compute_overlap_diagnostics(rules, df)
    with contextlib.redirect_stdout(old_out):
        for name, cls in (("Pass 1", 0), ("Pass 2", 1)):
            subset = [r for r in rules if r.predicted_class == cls]
            if len(subset) < 2:
                continue
            ov = _legacy_pairwise(RuleMatcher(None), subset, df)
            print(f"\n{name} overlap diagnostics:")
            print(f"  Average Jaccard: {np.mean(ov):.3f}")
            print(f"  Maximum Jaccard: {np.max(ov):.3f}")
            print(f"  Rules with >0.5 overlap: {sum(1 for o in ov if o > 0.5)} / {len(ov)}")
    assert new_out.getvalue() == old_out.getvalue() != ""


def test_pairwise_jaccards_of_fewer_than_two_rules_is_empty():
    df, rules = _overlap_world(0)
    analyzer = OverlapAnalyzer(matcher=_CountingMatcher())
    assert analyzer.pairwise_jaccards(rules[:1], df) == [] and analyzer.matcher.calls == 0


# ============================================================================
# 9. one real fit under the obligations (opt-in: pytest -m slow, ~2 min)
# ============================================================================
@pytest.mark.slow
def test_a_real_fit_meets_every_union_obligation_on_its_training_rows():
    """Rule search, evaluation and CBC are real. A 30 s limit per solve is
    shorter than production (300 s), so Pass 1 may be an unproven incumbent
    or a validated greedy result; whatever is returned must meet every
    obligation, recomputed from the training rows."""
    from shared.stage2.feature_engineering import engineer_features
    from tests.stage2.conftest import legacy_config, new_router
    from tests.support.synthetic import make_bank_frame, split_frame

    n_rows = 1200
    split = split_frame(make_bank_frame(n=n_rows))
    floors = dict(max_union_leakage_rate_pass1=0.15, min_union_npv_pass1=0.95,
                  min_union_support_pass1=60, min_union_precision_pass2=0.30,
                  min_union_support_pass2=20, pass2_union_objective=PASS2_OBJECTIVE_ROWS)
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
        mp.setattr(ilp_mod.ILPRuleSelector, "TIME_LIMIT_PER_SOLVE", 30)
        X_tr, _ = engineer_features(split["X_train"], split["X_test"], split["y_train"])
        router = new_router(legacy_config(min_support_pass1=n_rows // 40,
                                          min_support_pass2=n_rows // 120, **floors)).fit(
            X_tr, split["y_train"])
    y = split["y_train"].to_numpy()

    def routed(rules):
        return np.logical_or.reduce([np.logical_and.reduce(
            [X_tr[f].to_numpy() == v for f, v in r.segment]) for r in rules])

    p1 = routed(router.pass1_rules)
    tn, n1 = int((y[p1] == 0).sum()), int(p1.sum())
    assert n1 >= 60 and tn * 20 >= 19 * n1
    assert int(y[p1].sum()) <= (3 * int(y.sum())) // 20
    p2_rows = routed(router.pass2_rules)
    tp, n2 = int(y[p2_rows].sum()), int(p2_rows.sum())
    assert n2 >= 20 and tp * 10 >= 3 * n2
    reports = router.selection_reports()
    for key, rows, good in (("pass1", n1, tn), ("pass2", n2, tp)):
        assert reports[key]["source"] in ("ilp", "greedy_fallback")
        assert (reports[key]["union_rows"], reports[key]["union_good"]) == (rows, good)


def benchmark(n_rows=2000, n_rules=120, seed=0):
    """Lightweight before/after timing of the overlap diagnostics (not a test)."""
    df, rules = _overlap_world(seed, n_rows=n_rows, n_rules=n_rules)
    t = time.perf_counter()
    legacy = _legacy_pairwise(RuleMatcher(None), rules, df)
    t_old = time.perf_counter() - t
    t = time.perf_counter()
    new = OverlapAnalyzer(matcher=RuleMatcher(None)).pairwise_jaccards(rules, df)
    t_new = time.perf_counter() - t
    assert new == legacy
    print(f"overlap diagnostics, {len(rules)} rules x {n_rows} rows, {len(new)} pairs: "
          f"historical {t_old:.2f}s, Step 4B {t_new:.3f}s (x{t_old / max(t_new, 1e-9):.0f}); "
          f"values identical")


if __name__ == "__main__":   # python -m tests.stage2.test_glass_union_obligations
    benchmark()
