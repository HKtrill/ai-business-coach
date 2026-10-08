"""
Deterministic regression tests for the Stage 2 ILP rule selector
(glass_pipeline/glass_router/ilp_rule_selector).

Location: bank_pipeline/tests/test_ilp_rule_selector_union.py. Run from
bank_pipeline/ (e.g. `python -m pytest tests/test_ilp_rule_selector_union.py`)
so that `import glass_pipeline` resolves.

Most tests drive ILPRuleSelector._optimize_pass (one pass, no SelectedRule
conversion); the select_rules tests replace SelectedRule with a stub. The
components are also tested directly. Rules are duck-typed fakes whose covered
row sets are fixed. The repo's FeatureValidator is constructed, then replaced
by a stub whose base feature is the part of the feature name before "__".
"""
import itertools

import numpy as np
import pytest

import glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector as ilp_mod
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import (
    ILPRuleSelector, SelectorInfeasibleError, PROVEN, INCUMBENT, INFEASIBLE, NO_SOLUTION,
)
from glass_pipeline.glass_router.ilp_rule_selector.novelty_analyzer import NoveltyAnalyzer
from glass_pipeline.glass_router.ilp_rule_selector.quality_gate_filter import QualityGateFilter
from glass_pipeline.glass_router.ilp_rule_selector.utils import (
    get_covered_array, unique_counts, union_row_weights, meets_novelty_floor,
    novelty_floor_rows, compute_rule_quality, membership_atoms,
)

P1, P2 = "Pass 1 (NOT_SUBSCRIBE)", "Pass 2 (SUBSCRIBE)"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
class _StubValidator:
    def extract_base_feature(self, feature):
        return feature.split("__")[0]

    def has_duplicate_base_features(self, segment):
        return False


class FakeRule:
    """Duck-typed EvaluatedRule with a fixed covered row set."""

    def __init__(self, rule_id, rows, n, precision=0.95, recall=0.0, predicted_class=0,
                 rf_confidence=0.5, rf_alignment=0.0, base=None):
        rows = {int(i) for i in rows}
        self.rule_id = rule_id
        feature = f"{base or 'feat_' + str(rule_id)}__{rule_id}"
        self.segment = ((feature, 1),)
        self.segment_frozen = frozenset(self.segment)
        self._cached_covered_idx = rows
        self.support = len(rows)
        self.coverage = len(rows) / n
        self.precision = precision
        self.recall = recall
        self.predicted_class = predicted_class
        self.rf_confidence = rf_confidence
        self.rf_alignment = rf_alignment

    def __repr__(self):
        return f"FakeRule({self.rule_id})"


def rng(a, b):
    """Inclusive integer range."""
    return list(range(a, b + 1))


def make_selector(**overrides):
    params = dict(
        min_pass1_rules=1, max_pass1_rules=8,
        min_precision_pass1=0.90, max_precision_pass1=1.0,
        max_subscriber_leakage_rate_pass1=1.0, max_subscriber_leakage_absolute_pass1=10**9,
        min_pass2_rules=1, max_pass2_rules=8,
        min_precision_pass2=0.0, max_precision_pass2=1.0,
        min_recall_pass2=0.0, max_recall_pass2=1.0,
        min_novelty_ratio_pass1=0.20, min_novelty_ratio_pass2=0.20,
        enable_novelty_constraints=True, diversity_weight=0.35,
        lambda_rf_uncertainty=0.0, lambda_rf_misalignment=0.0,
    )
    params.update(overrides)
    sel = ILPRuleSelector(**params)
    stub = _StubValidator()
    sel.validator = stub
    sel.diversity_analyzer.validator = stub
    return sel


def run_pass1(sel, rules, y, min_rules, max_rules, ratio=0.20, max_base_reuse=None,
              leak_rate=None):
    return sel._optimize_pass(
        candidates=rules, min_rules=min_rules, max_rules=max_rules, pass_name=P1,
        scoring_mode="precision_first", y_val=y, max_base_reuse=max_base_reuse,
        min_novelty_ratio=ratio, max_union_leakage_rate=leak_rate,
    )


def run_greedy(sel, rules, y, max_rules, scoring_mode="precision_first", floor=(0.20, 1),
               base_limit=40):
    return sel.greedy_selector.greedy_select(
        rules, max_rules, scoring_mode, union_row_weights(y, scoring_mode),
        floor=floor, base_reuse_limit=base_limit, verbose=False,
    )


def ids(rules):
    return {r.rule_id for r in rules}


def union_size(rules):
    return len(set().union(*(r._cached_covered_idx for r in rules))) if rules else 0


def unique_shares(rules, n):
    arrays = [get_covered_array(r, n) for r in rules]
    return {r.rule_id: u / a.size for r, u, a in zip(rules, unique_counts(arrays, n), arrays)}


def script_solver(sel, steps):
    """Replace ilp_builder.solve with scripted steps; 'real' runs CBC."""
    real = sel.ilp_builder.solve
    calls = []

    def fake(prob, time_limit=300, warm_start=False, gap_rel=None, gap_abs=None):
        step = steps[min(len(calls), len(steps) - 1)]
        calls.append(step)
        if step == "real":
            return real(prob, time_limit=time_limit, warm_start=warm_start,
                        gap_rel=gap_rel, gap_abs=gap_abs)
        return step(prob, real)

    sel.ilp_builder.solve = fake
    return calls


def no_solution(builder):
    def step(prob, real):
        builder.last_sol_status = 0          # LpSolutionNoSolutionFound
        return "Not Solved"
    return step


def zero_incumbent(builder, sol_status):
    """All-zero 'solution' - what a time-limited CBC run returned in the sandbox."""
    def step(prob, real):
        for v in prob.variables():
            v.varValue = 0.0
        builder.last_sol_status = sol_status
        return "Optimal"
    return step


def real_but_time_limited(builder):
    """Real CBC solution, reported the way PuLP reports a time-limit stop."""
    def step(prob, real):
        status = real(prob, time_limit=300, warm_start=False, gap_rel=0.0)
        builder.last_sol_status = 2          # LpSolutionIntegerFeasible
        return status
    return step


# ----------------------------------------------------------------------------
# shared instances
# ----------------------------------------------------------------------------
def _abc():
    n = 20
    return n, np.zeros(n, int), [FakeRule("A", rng(0, 4), n), FakeRule("B", rng(0, 3), n),
                                 FakeRule("C", rng(5, 7), n)]


def _zero_marginal_case(c_half=50, c_precision=1.00):
    n = 400
    y = np.zeros(n, dtype=int)
    A = FakeRule("A", rng(0, 99), n, precision=0.95)
    B = FakeRule("B", rng(100, 199), n, precision=0.95)
    # C lies inside A ∪ B but is a subset of neither (50% overlap with each)
    # and has the best standalone quality -> the old objective always wanted it.
    C = FakeRule("C", rng(0, c_half - 1) + rng(100, 100 + c_half - 1), n, precision=c_precision)
    return n, y, A, B, C


def _floor_case():
    n = 400
    y = np.zeros(n, dtype=int)
    A = FakeRule("A", rng(0, 99), n)
    B = FakeRule("B", rng(100, 199), n)
    # D: 45 rows in A, 45 in B, 10 new -> 10% unique given A and B;
    # pairwise overlaps are only 45%, so no pairwise rule can see this.
    D = FakeRule("D", rng(0, 44) + rng(100, 144) + rng(200, 209), n)
    # E: 8 new rows, 100% novel, less union gain than D (8 < 10).
    E = FakeRule("E", rng(300, 307), n)
    return n, y, A, B, D, E


# ============================================================================
# 1. classic A/B/C overlap regression
# ============================================================================
@pytest.mark.parametrize("lambdas", [(0.0, 0.0), (0.15, 0.08)])
def test_abc_union_beats_redundant_raw_coverage(lambdas):
    """A={0..4}, B={0..3}, C={5..7}. Raw: A+B = 9 > A+C = 8. Union: 5 < 8."""
    n, y, rules = _abc()
    sel = make_selector(lambda_rf_uncertainty=lambdas[0], lambda_rf_misalignment=lambdas[1])
    chosen = run_pass1(sel, rules, y, min_rules=2, max_rules=2)
    assert ids(chosen) == {"A", "C"}
    assert union_size(chosen) == 8
    assert sel.selection_report_[P1]["lexicographic_guarantee"] == "proven"


# ============================================================================
# 2. rule contained in the union of multiple selected rules
# ============================================================================
@pytest.mark.parametrize("novelty_on", [True, False])
@pytest.mark.parametrize("order", ["ABC", "CAB", "BCA"])
def test_rule_inside_union_of_others_never_selected(novelty_on, order):
    n, y, A, B, C = _zero_marginal_case()
    by = {"A": A, "B": B, "C": C}
    sel = make_selector(enable_novelty_constraints=novelty_on)
    chosen = run_pass1(sel, [by[k] for k in order], y, min_rules=2, max_rules=3)
    assert ids(chosen) == {"A", "B"}


# ============================================================================
# 3. no zero-marginal selected rules where alternatives exist
# ============================================================================
def test_no_zero_marginal_rule_with_default_rf_penalties():
    """The old model's negative weights picked the top-min_rules standalone
    rules (C + A/B, union 150); the union objective picks A + B (union 200)."""
    n, y, A, B, C = _zero_marginal_case()
    sel = make_selector(lambda_rf_uncertainty=0.15, lambda_rf_misalignment=0.08)
    chosen = run_pass1(sel, [A, B, C], y, min_rules=2, max_rules=2)
    assert ids(chosen) == {"A", "B"} and union_size(chosen) == 200
    assert all(s > 0 for s in unique_shares(chosen, n).values())


def test_novelty_floor_is_set_level_not_pairwise():
    n, y, A, B, D, E = _floor_case()
    chosen = run_pass1(make_selector(), [A, B, D, E], y, min_rules=2, max_rules=3)
    assert ids(chosen) == {"A", "B", "E"}
    assert all(s >= 0.20 for s in unique_shares(chosen, n).values())
    # control: without the floor the union optimum takes D (10 new rows > 8)
    chosen_off = run_pass1(make_selector(enable_novelty_constraints=False),
                           [A, B, D, E], y, min_rules=2, max_rules=3)
    assert ids(chosen_off) == {"A", "B", "D"}


# ============================================================================
# 4. ILP union coverage accounting
# ============================================================================
def test_union_coverage_expression_counts_each_row_once():
    n, y, rules = _abc()
    sel = make_selector()
    b = sel.ilp_builder
    arrays = {r.rule_id: get_covered_array(r, n) for r in rules}
    prob = b.create_problem("accounting")
    x = b.create_decision_variables(rules)
    cov, raw, n_weighted, z_links = b.add_union_coverage(prob, rules, x, arrays, np.ones(n, int))
    assert raw == {"A": 5, "B": 4, "C": 3}
    prob += x["A"] == 1
    prob += x["B"] == 1
    prob += x["C"] == 0
    prob.setObjective(cov)
    assert b.solve(prob, time_limit=60, gap_rel=0.0) == "Optimal"
    assert round(cov.value()) == 5          # not 5 + 4


def test_atoms_are_exact_membership_patterns():
    arrays = {"A": np.array([0, 1, 2]), "B": np.array([2, 3])}
    members, atom_of_row = membership_atoms(["A", "B"], arrays, 5)
    assert members == [("A",), ("A", "B"), ("B",)]
    assert atom_of_row.tolist() == [0, 0, 1, 2, -1]


def test_selection_report_union_matches_true_union():
    n, y, A, B, D, E = _floor_case()
    sel = make_selector()
    chosen = run_pass1(sel, [A, B, D, E], y, 2, 3)
    rep = sel.selection_report_[P1]
    assert rep["union_coverage"] == union_size(chosen) == 208
    assert rep["phase1"]["state"] == PROVEN and rep["phase2"]["state"] == PROVEN


# ============================================================================
# 5. greedy union-gain behaviour
# ============================================================================
def test_greedy_prefers_union_gain_over_standalone_quality():
    n = 400
    y = np.zeros(n, int)
    big = FakeRule("BIG", rng(0, 99), n, precision=0.95)
    redundant = FakeRule("RED", rng(0, 89), n, precision=1.00)   # best quality, 0 gain
    new = FakeRule("NEW", rng(200, 239), n, precision=0.93)       # worst quality, +40
    chosen = run_greedy(make_selector(), [big, redundant, new], y, 3, floor=None)
    assert [r.rule_id for r in chosen] == ["BIG", "NEW"]


def test_greedy_rechecks_earlier_picks_against_floor():
    """X first; Y and Z together would cover X completely. The set-level floor
    must reject Z. The ILP (optimal) is never worse than the greedy."""
    n = 400
    y = np.zeros(n, int)
    X = FakeRule("X", rng(0, 99), n)
    Y = FakeRule("Y", rng(0, 49) + rng(100, 139), n)
    Z = FakeRule("Z", rng(50, 99) + rng(200, 239), n)
    W = FakeRule("W", rng(300, 319), n)
    sel = make_selector()
    g = run_greedy(sel, [X, Y, Z, W], y, 3)
    assert ids(g) == {"X", "Y", "W"}
    ilp = run_pass1(sel, [X, Y, Z, W], y, 1, 3)
    assert ids(ilp) == {"Y", "Z", "W"} and union_size(ilp) == 200 >= union_size(g)


# ============================================================================
# 6. greedy / ILP hard-constraint parity
# ============================================================================
def test_greedy_and_ilp_satisfy_identical_hard_constraints():
    cases = []
    n, y, rules = _abc()
    cases.append((n, y, rules, 2, 2))
    n, y, A, B, C = _zero_marginal_case(c_half=40, c_precision=0.95)
    cases.append((n, y, [A, B, C], 2, 3))
    n, y, A, B, D, E = _floor_case()
    cases.append((n, y, [A, B, D, E], 2, 3))
    for n, y, rules, lo, hi in cases:
        sel = make_selector()
        ilp = run_pass1(sel, rules, y, lo, hi)
        g = run_greedy(sel, rules, y, hi)
        for chosen in (ilp, g):
            assert sel._meets_hard_constraints(chosen, lo, hi, None, (0.20, 1), n)
        assert ids(g) == ids(ilp)          # greedy is optimal on these instances


# ============================================================================
# 7. base-feature reuse in the fallback (and the ILP)
# ============================================================================
def _reuse_case():
    n = 400
    y = np.zeros(n, int)
    A = FakeRule("A", rng(0, 99), n, base="euribor")
    B = FakeRule("B", rng(100, 189), n, base="euribor")      # disjoint, same base
    C = FakeRule("C", rng(200, 249), n, base="campaign")
    return n, y, [A, B, C]


def test_base_feature_reuse_enforced_by_ilp_and_fallback():
    n, y, rules = _reuse_case()
    ilp = run_pass1(make_selector(), rules, y, 1, 3, max_base_reuse=1)
    assert ids(ilp) == {"A", "C"}

    sel = make_selector()
    script_solver(sel, [no_solution(sel.ilp_builder)])
    fb = run_pass1(sel, rules, y, 1, 3, max_base_reuse=1)
    assert sel.selection_report_[P1]["source"] == "greedy_fallback"
    assert ids(fb) == {"A", "C"}
    assert sel._meets_hard_constraints(fb, 1, 3, 1, (0.20, 1), n)


# ============================================================================
# 8. pass-specific novelty in the fallback
# ============================================================================
@pytest.mark.parametrize("ratio, expected", [(0.40, {"A"}), (0.20, {"A", "D"})])
def test_fallback_uses_pass_floor_not_greedy_defaults(ratio, expected):
    """D keeps 30% unique rows next to A. The superseded greedy default
    (min_novelty_greedy=0.15) would accept it; the pass floor decides."""
    n = 400
    y = np.zeros(n, int)
    A = FakeRule("A", rng(0, 199), n)
    D = FakeRule("D", rng(0, 69) + rng(300, 329), n)
    sel = make_selector(min_novelty_greedy=0.15, min_absolute_new_samples=30)
    script_solver(sel, [no_solution(sel.ilp_builder)])
    fb = run_pass1(sel, [A, D], y, 1, 2, ratio=ratio)
    assert sel.selection_report_[P1]["source"] == "greedy_fallback"
    assert ids(fb) == expected


# ============================================================================
# 9. infeasible / invalid incumbent rejection
# ============================================================================
def test_proven_infeasible_raises_with_constraint_details():
    n, y, A, B, C = _zero_marginal_case()
    sel = make_selector()
    calls = script_solver(sel, ["real"])
    with pytest.raises(SelectorInfeasibleError) as err:
        run_pass1(sel, [A, B, C], y, min_rules=3, max_rules=3)
    assert err.value.proven is True
    msg = str(err.value)
    for needle in ("min_rules", "= 3", "novelty floor", "20%", "base-feature", "PROVED"):
        assert needle in msg
    assert len(calls) == 1                  # proven by a single exact solve


def test_fallback_that_cannot_meet_min_rules_raises_unproven():
    n, y, A, B, C = _zero_marginal_case()
    sel = make_selector()
    script_solver(sel, [no_solution(sel.ilp_builder)])
    with pytest.raises(SelectorInfeasibleError) as err:
        run_pass1(sel, [A, B, C], y, min_rules=3, max_rules=3)
    assert err.value.proven is False
    assert "greedy" in str(err.value) and "outside cardinality" in str(err.value)


def test_invalid_unproven_incumbent_is_rejected():
    """PuLP labels a time-limited all-zero incumbent 'Optimal' (sol_status 2).
    It must be rejected, and the validated fallback used instead."""
    n, y, A, B, D, E = _floor_case()
    sel = make_selector()
    script_solver(sel, [zero_incumbent(sel.ilp_builder, sol_status=2)])
    chosen = run_pass1(sel, [A, B, D, E], y, 2, 3)
    rep = sel.selection_report_[P1]
    assert rep["phase1"]["state"] == "rejected_incumbent"
    assert rep["source"] == "greedy_fallback"
    assert ids(chosen) == {"A", "B", "E"}


def test_proven_optimum_failing_validation_raises():
    n, y, A, B, D, E = _floor_case()
    sel = make_selector()
    script_solver(sel, [zero_incumbent(sel.ilp_builder, sol_status=1)])
    with pytest.raises(RuntimeError, match="fails independent validation"):
        run_pass1(sel, [A, B, D, E], y, 2, 3)


# ============================================================================
# 10. time-limited solver status handling
# ============================================================================
@pytest.mark.parametrize("status, sol_status, expected", [
    ("Optimal", 1, PROVEN),
    ("Optimal", 2, INCUMBENT),          # PuLP's label for a time-limit stop
    ("Not Solved", 2, INCUMBENT),
    ("Not Solved", 0, NO_SOLUTION),
    ("Infeasible", -1, INFEASIBLE),
    ("Infeasible", 0, INFEASIBLE),
    ("Optimal", None, INCUMBENT),       # no sol_status available -> never "proven"
    ("Undefined", 0, NO_SOLUTION),
])
def test_solution_classification(status, sol_status, expected):
    assert ILPRuleSelector._classify_solution(status, sol_status) == expected


def test_time_limited_incumbent_accepted_but_marked_unproven(capsys):
    n, y, A, B, D, E = _floor_case()
    sel = make_selector()
    script_solver(sel, [real_but_time_limited(sel.ilp_builder), "real"])
    chosen = run_pass1(sel, [A, B, D, E], y, 2, 3)
    rep = sel.selection_report_[P1]
    assert rep["phase1"]["state"] == INCUMBENT
    assert rep["phase2"]["state"] == PROVEN
    assert rep["lexicographic_guarantee"] == "UNPROVEN"
    assert rep["source"] == "ilp"
    assert ids(chosen) == {"A", "B", "E"}
    assert "NOT proven optimal" in capsys.readouterr().out


# ============================================================================
# 11. Pass 2 positive-union objective
# ============================================================================
def test_pass2_union_recall_counts_positives_once():
    n = 1000
    y = np.zeros(n, dtype=int)
    y[0:50] = 1        # positives shared by A and B
    y[300:330] = 1     # positives only C reaches
    y[900:1000] = 1    # other positives
    P = int(y.sum())
    A = FakeRule("A", rng(0, 99), n, precision=0.50, recall=50 / P, predicted_class=1)
    B = FakeRule("B", rng(0, 49) + rng(100, 149), n, precision=0.40, recall=50 / P, predicted_class=1)
    C = FakeRule("C", rng(300, 399), n, precision=0.30, recall=30 / P, predicted_class=1)
    sel = make_selector()
    chosen = sel._optimize_pass(
        candidates=[A, B, C], min_rules=2, max_rules=2, pass_name=P2,
        scoring_mode="recall_first", y_val=y, max_base_reuse=None, min_novelty_ratio=0.20,
    )
    assert ids(chosen) == {"A", "C"}
    assert sel.selection_report_[P2]["union_coverage"] == 80


def test_recall_first_score_is_a_function_of_true_positives_only():
    """recall^3 * precision * coverage = TP^4 / (P^3 * N)."""
    N, P = 1000, 200
    for tp, n_j in [(40, 80), (40, 400), (10, 50)]:
        r = FakeRule("R", range(n_j), N, precision=tp / n_j, recall=tp / P, predicted_class=1)
        assert compute_rule_quality(r, "recall_first") == pytest.approx(tp ** 4 / (P ** 3 * N))


# ============================================================================
# 12. quality-gate semantics preserved (per-rule gates, not union constraints)
# ============================================================================
def test_per_rule_gates_are_hard_and_unchanged():
    n = 1000
    y = np.zeros(n, dtype=int)
    y[900:1000] = 1
    H = FakeRule("H", rng(0, 799), n, precision=0.80)       # fails precision
    L = FakeRule("L", rng(850, 999), n, precision=0.95)     # 100 subscribers -> leakage
    K1 = FakeRule("K1", rng(0, 199), n, precision=0.97)
    K2 = FakeRule("K2", rng(300, 449), n, precision=0.96)
    sel = make_selector(min_precision_pass1=0.93, max_subscriber_leakage_absolute_pass1=20)
    chosen = run_pass1(sel, [H, L, K1, K2], y, min_rules=1, max_rules=4)
    assert ids(chosen) == {"K1", "K2"}

    gate = QualityGateFilter(min_precision_pass1=0.93, max_precision_pass1=1.0,
                             max_subscriber_leakage_rate_pass1=1.0,
                             max_subscriber_leakage_absolute_pass1=20)
    valid, rejected = gate.apply_quality_gates_pass1([H, L, K1, K2], y)
    assert ids(valid) == {"K1", "K2"}
    assert {r.rule_id: sub for r, _, sub in rejected} == {"H": 0, "L": 100}


def test_leakage_cap_is_per_rule_not_union():
    """Each rule leaks 4% of subscribers (cap 5%); together 8%. Per-rule gates
    allow both - the selector adds no union-level leakage constraint."""
    n = 1000
    y = np.zeros(n, dtype=int)
    y[0:4] = 1
    y[100:104] = 1
    y[900:992] = 1                          # 100 subscribers in total
    A = FakeRule("A", rng(0, 99), n, precision=0.96)
    B = FakeRule("B", rng(100, 199), n, precision=0.96)
    sel = make_selector(max_subscriber_leakage_rate_pass1=0.05)
    chosen = run_pass1(sel, [A, B], y, 1, 2)
    assert ids(chosen) == {"A", "B"}
    assert int(y[list(set().union(*(r._cached_covered_idx for r in chosen)))].sum()) == 8


def test_union_precision_and_leakage_diagnostics_use_union():
    n = 200
    y = np.zeros(n, dtype=int)
    y[[0, 1, 2, 3]] = 1
    y[[60, 61, 62]] = 1          # in the overlap
    y[[140, 141, 142, 143]] = 1
    A = FakeRule("A", rng(0, 99), n, precision=0.93)
    B = FakeRule("B", rng(50, 149), n, precision=0.93)
    stats = NoveltyAnalyzer().analyze_selection_novelty([A, B], "Pass 1", y=y)
    assert stats["summed_rule_positives"] == 14 and stats["union_positives"] == 11
    assert stats["union_negatives"] / stats["union_rows"] == pytest.approx(139 / 150)
    assert stats["union_negatives"] / stats["union_rows"] < 0.93


# ============================================================================
# constraint-aware equivalence merge
# ============================================================================
def test_merge_keeps_best_of_truly_equivalent_twins():
    n = 400
    y = np.zeros(n, int)
    lo = FakeRule("LO", rng(0, 99), n, precision=0.94, base="cci")
    hi = FakeRule("HI", rng(0, 99), n, precision=0.99, base="cci")   # same rows, same base
    other = FakeRule("O", rng(200, 249), n, base="dow")
    sel = make_selector()
    chosen = run_pass1(sel, [lo, hi, other], y, 1, 3)
    assert sel.selection_report_[P1]["candidates_after_merge"] == 2
    assert ids(chosen) == {"HI", "O"}


def test_merge_does_not_collapse_twins_with_different_bases():
    """T1 and T2 cover the same rows; T1 has better quality but shares S's base.
    With reuse <= 1 only S + T2 is feasible. A coverage-only merge would keep
    T1 and lose the union-150 solution."""
    n = 400
    y = np.zeros(n, int)
    S = FakeRule("S", rng(0, 99), n, precision=0.97, base="euribor")
    T1 = FakeRule("T1", rng(100, 149), n, precision=0.99, base="euribor")
    T2 = FakeRule("T2", rng(100, 149), n, precision=0.93, base="campaign")
    sel = make_selector()
    chosen = run_pass1(sel, [S, T1, T2], y, 1, 2, max_base_reuse=1)
    assert sel.selection_report_[P1]["candidates_after_merge"] == 3
    assert ids(chosen) == {"S", "T2"} and union_size(chosen) == 150


def test_no_merge_when_novelty_disabled():
    n = 400
    y = np.zeros(n, int)
    a = FakeRule("a", rng(0, 99), n, base="cci")
    b = FakeRule("b", rng(0, 99), n, base="cci")
    sel = make_selector(enable_novelty_constraints=False)
    run_pass1(sel, [a, b], y, 1, 2)
    assert sel.selection_report_[P1]["candidates_after_merge"] == 2


# ============================================================================
# compact floor formulation vs exhaustive enumeration
# ============================================================================
def _brute_force(rules, n, lo, hi, ratio, reuse, validator, y=None, leak_cap=None):
    arrays = {r.rule_id: get_covered_array(r, n) for r in rules}
    best = None
    for k in range(lo, hi + 1):
        for combo in itertools.combinations(rules, k):
            bases = [validator.extract_base_feature(f) for r in combo for f, _ in r.segment]
            if any(bases.count(b) > reuse for b in set(bases)):
                continue
            arrs = [arrays[r.rule_id] for r in combo]
            uniq = unique_counts(arrs, n)
            if not all(meets_novelty_floor(u, a.size, ratio, 1) for u, a in zip(uniq, arrs)):
                continue
            covered = set().union(*(r._cached_covered_idx for r in combo))
            if leak_cap is not None and int(y[list(covered)].sum()) > leak_cap:
                continue
            union = len(covered)
            overlap = sum(a.size for a in arrs) - union
            key = (union, -overlap)
            if best is None or key > best:
                best = key
    return best


def test_compact_floor_matches_exhaustive_search():
    n = 60
    y = np.zeros(n, int)
    outcomes = {"feasible": 0, "infeasible": 0}
    for seed in range(60):
        r = np.random.RandomState(seed)
        rules = []
        for j in range(9):
            start = r.randint(0, 45)
            rows = set(range(start, start + r.randint(5, 16))) | set(r.randint(0, n, r.randint(0, 6)))
            rules.append(FakeRule(f"r{j}", rows, n, precision=float(r.uniform(0.93, 1.0)),
                                  base=f"b{r.randint(0, 3)}"))
        ratio = [0.2, 0.35, 0.5, 0.7][seed % 4]
        lo, hi = [(1, 3), (2, 4), (3, 4), (4, 5), (5, 6)][seed % 5]
        reuse = 2
        expected = _brute_force(rules, n, lo, hi, ratio, reuse, _StubValidator())
        sel = make_selector()
        if expected is None:
            with pytest.raises(SelectorInfeasibleError) as err:
                run_pass1(sel, rules, y, lo, hi, ratio=ratio, max_base_reuse=reuse)
            assert err.value.proven is True
            outcomes["infeasible"] += 1
            continue
        chosen = run_pass1(sel, rules, y, lo, hi, ratio=ratio, max_base_reuse=reuse)
        arrs = [get_covered_array(c, n) for c in chosen]
        union = union_size(chosen)
        overlap = sum(a.size for a in arrs) - union
        assert (union, -overlap) == expected, f"seed {seed}"
        assert sel._meets_hard_constraints(chosen, lo, hi, reuse, (ratio, 1), n)
        outcomes["feasible"] += 1
    print(outcomes)
    assert outcomes["feasible"] >= 10 and outcomes["infeasible"] >= 3, outcomes


# ============================================================================
# misc contract checks
# ============================================================================
def test_unknown_scoring_mode_rejected():
    n, y, rules = _abc()
    sel = make_selector()
    with pytest.raises(ValueError, match="scoring_mode"):
        sel._optimize_pass(rules, 1, 2, P1, "precision", y_val=y)
    with pytest.raises(ValueError, match="scoring_mode"):
        union_row_weights(y, "recall")
    with pytest.raises(ValueError, match="scoring_mode"):
        compute_rule_quality(rules[0], "")


def test_ordered_novelty_can_hide_redundancy_unique_share_cannot():
    n = 60
    A, B, C = FakeRule("A", rng(0, 9), n), FakeRule("B", rng(5, 14), n), FakeRule("C", rng(10, 19), n)
    stats = NoveltyAnalyzer().analyze_selection_novelty([A, B, C], "Pass 1", y=np.zeros(n, int))
    assert stats["novelty"]["B"] == pytest.approx(0.5)
    assert stats["unique_rows"]["B"] == 0 and stats["redundant_rule_ids"] == ["B"]


def test_selected_set_is_independent_of_candidate_order():
    n, y, A, B, D, E = _floor_case()
    base = [A, B, D, E]
    results = {
        frozenset(ids(run_pass1(make_selector(), [base[i] for i in perm], y, 2, 3)))
        for perm in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2])
    }
    assert results == {frozenset({"A", "B", "E"})}


def test_floor_boundaries():
    assert novelty_floor_rows(100, 0.20, 1) == 20
    assert novelty_floor_rows(3, 0.20, 1) == 1
    assert meets_novelty_floor(20, 100, 0.20, 1)
    assert not meets_novelty_floor(19, 100, 0.20, 1)
    assert not meets_novelty_floor(0, 100, 0.0, 1)       # zero-marginal always fails


# ============================================================================
# A1 edge cases: the configured min_rules is a hard contract
# ============================================================================
def test_min_rules_above_candidates_after_quality_gates_raises():
    n = 400
    y = np.zeros(n, int)
    ok1 = FakeRule("K1", rng(0, 99), n, precision=0.97)
    ok2 = FakeRule("K2", rng(100, 199), n, precision=0.97)
    bad = FakeRule("BAD", rng(200, 299), n, precision=0.80)     # fails the 0.93 gate
    sel = make_selector(min_precision_pass1=0.93)
    calls = script_solver(sel, ["real"])
    with pytest.raises(SelectorInfeasibleError) as err:
        run_pass1(sel, [ok1, ok2, bad], y, min_rules=3, max_rules=4)
    assert err.value.proven is True
    msg = str(err.value)
    assert "min_rules = 3" in msg and "only 2" in msg and "quality gates" in msg
    assert calls == []                                          # no solve needed


def test_min_rules_above_candidates_after_equivalent_merge_raises():
    n = 400
    y = np.zeros(n, int)
    t1 = FakeRule("T1", rng(0, 99), n, precision=0.97, base="cci")
    t2 = FakeRule("T2", rng(0, 99), n, precision=0.95, base="cci")   # identical twin
    o = FakeRule("O", rng(200, 249), n, base="dow")
    with pytest.raises(SelectorInfeasibleError) as err:
        run_pass1(make_selector(), [t1, t2, o], y, min_rules=3, max_rules=3)
    assert err.value.proven is True
    assert "only 2" in str(err.value) and "constraint-equivalent merge" in str(err.value)
    # control: without the floor there is no merge and twins may coexist
    chosen = run_pass1(make_selector(enable_novelty_constraints=False), [t1, t2, o], y, 3, 3)
    assert ids(chosen) == {"T1", "T2", "O"}


def test_zero_candidates_with_positive_min_rules_raises():
    sel = make_selector()
    with pytest.raises(SelectorInfeasibleError, match="only 0 candidate"):
        sel._optimize_pass([], 2, 4, P1, "precision_first", y_val=np.zeros(10, int))
    # min_rules == 0 explicitly allows an empty selection
    assert sel._optimize_pass([], 0, 4, P1, "precision_first", y_val=np.zeros(10, int)) == []


def test_zero_valid_rules_after_gates_with_positive_min_rules_raises():
    n = 200
    y = np.zeros(n, int)
    rules = [FakeRule("L1", rng(0, 49), n, precision=0.80),
             FakeRule("L2", rng(50, 99), n, precision=0.85)]
    sel = make_selector(min_precision_pass1=0.93)
    with pytest.raises(SelectorInfeasibleError) as err:
        run_pass1(sel, rules, y, min_rules=1, max_rules=3)
    assert "only 0" in str(err.value) and "quality gates" in str(err.value)
    assert run_pass1(make_selector(min_precision_pass1=0.93), rules, y, 0, 3) == []


@pytest.mark.parametrize("lo, hi", [(5, 3), (-1, 3), (0, 0), (2.0, 4)])
def test_invalid_cardinality_configuration_is_a_config_error(lo, hi):
    with pytest.raises(ValueError, match="invalid rule-count configuration"):
        make_selector(min_pass1_rules=lo, max_pass1_rules=hi)
    with pytest.raises(ValueError, match="invalid rule-count configuration"):
        make_selector(min_pass2_rules=lo, max_pass2_rules=hi)
    n, y, rules = _abc()
    with pytest.raises(ValueError, match="invalid rule-count configuration"):
        make_selector()._optimize_pass(rules, lo, hi, P1, "precision_first", y_val=y)


def test_greedy_fallback_below_min_rules_is_rejected():
    """Forced fallback; the greedy can only reach 2 floor-feasible rules but
    min_rules = 3 -> raise instead of returning 2."""
    n, y, A, B, C = _zero_marginal_case()
    sel = make_selector()
    script_solver(sel, [no_solution(sel.ilp_builder)])
    with pytest.raises(SelectorInfeasibleError) as err:
        run_pass1(sel, [A, B, C], y, min_rules=3, max_rules=3)
    assert err.value.proven is False
    assert sel.selection_report_[P1]["source"] == "infeasible"


def test_feasible_selection_with_exactly_min_rules_succeeds():
    n, y, A, B, D, E = _floor_case()
    sel = make_selector()
    chosen = run_pass1(sel, [A, B, D, E], y, min_rules=3, max_rules=3)
    assert ids(chosen) == {"A", "B", "E"} and len(chosen) == 3
    # min_rules equal to the number of candidates: max is bounded, min is not lowered
    n = 300
    y = np.zeros(n, int)
    three = [FakeRule(f"R{k}", rng(100 * k, 100 * k + 49), n) for k in range(3)]
    sel = make_selector()
    chosen = run_pass1(sel, three, y, min_rules=3, max_rules=8)
    assert len(chosen) == 3
    rep = sel.selection_report_[P1]
    assert rep["min_rules"] == 3 and rep["max_rules"] == 3


# ============================================================================
# select_rules(passes=...) - single-pass callers without violating A1
# ============================================================================
class _StubSelected:
    @classmethod
    def from_evaluated(cls, rule, pass_assignment):
        return rule


def test_select_rules_single_pass_does_not_run_the_other_pass(monkeypatch):
    monkeypatch.setattr(ilp_mod, "SelectedRule", _StubSelected)
    n, y, A, B, D, E = _floor_case()
    sel = make_selector(min_pass1_rules=2, max_pass1_rules=3, min_pass2_rules=5, max_pass2_rules=8)
    out = sel.select_rules([A, B, D, E], y_val=y, passes=("pass1",))
    assert ids(out["pass1_rules"]) == {"A", "B", "E"} and out["pass2_rules"] == []
    # default (both passes): Pass 2 has no candidates and min_pass2_rules = 5 -> raise
    with pytest.raises(SelectorInfeasibleError, match="Pass 2"):
        make_selector(min_pass1_rules=2, max_pass1_rules=3, min_pass2_rules=5,
                      max_pass2_rules=8).select_rules([A, B, D, E], y_val=y)


def test_select_rules_empty_input_obeys_min_rules(monkeypatch):
    monkeypatch.setattr(ilp_mod, "SelectedRule", _StubSelected)
    with pytest.raises(SelectorInfeasibleError, match="only 0 candidate"):
        make_selector(min_pass1_rules=2).select_rules([], y_val=np.zeros(5, int))
    out = make_selector(min_pass1_rules=0, min_pass2_rules=0).select_rules([], y_val=np.zeros(5, int))
    assert out == {"pass1_rules": [], "pass2_rules": []}


def test_select_rules_rejects_unknown_passes():
    with pytest.raises(ValueError, match="passes"):
        make_selector().select_rules([], y_val=np.zeros(5, int), passes=("pass3",))


# ============================================================================
# optional Pass 1 union leakage cap
# ============================================================================
def _leak_case():
    """A and B each hold 4 of the 50 positives; C holds none. rate 0.10 -> cap 5."""
    n = 1000
    y = np.zeros(n, int)
    y[0:4] = 1          # in A
    y[100:104] = 1      # in B
    y[900:942] = 1      # elsewhere -> 50 positives in total
    A = FakeRule("A", rng(0, 99), n, precision=0.97)
    B = FakeRule("B", rng(100, 199), n, precision=0.96)
    C = FakeRule("C", rng(200, 259), n, precision=0.99)
    return n, y, [A, B, C]


def test_union_leakage_cap_off_changes_nothing():
    n, y, A, B, D, E = _floor_case()
    base = run_pass1(make_selector(), [A, B, D, E], y, 2, 3)
    sel = make_selector()
    capped_off = run_pass1(sel, [A, B, D, E], y, 2, 3, leak_rate=None)
    assert ids(base) == ids(capped_off) == {"A", "B", "E"}
    assert sel.selection_report_[P1]["union_leakage_cap"] is None


def test_union_leakage_cap_binds_on_the_union():
    n, y, rules = _leak_case()
    assert ids(run_pass1(make_selector(), rules, y, 1, 3)) == {"A", "B", "C"}   # uncapped: 8 leaked
    sel = make_selector()
    chosen = run_pass1(sel, rules, y, 1, 3, leak_rate=0.10)
    rep = sel.selection_report_[P1]
    assert ids(chosen) == {"A", "C"}
    assert rep["union_leakage_cap"] == 5 and rep["union_positives"] == 4
    assert rep["lexicographic_guarantee"] == "proven"


def test_union_leakage_counts_shared_positives_once():
    """A and B share the same 4 positives: union leakage 4 <= 5 although the
    per-rule sum is 8 - a summed (non-union) constraint would wrongly forbid B."""
    n = 1000
    y = np.zeros(n, int)
    y[0:4] = 1
    y[900:946] = 1                                   # 50 positives in total
    A = FakeRule("A", rng(0, 99), n, precision=0.96)
    B = FakeRule("B", rng(0, 3) + rng(100, 195), n, precision=0.96)
    sel = make_selector()
    chosen = run_pass1(sel, [A, B], y, 1, 2, leak_rate=0.10)
    assert ids(chosen) == {"A", "B"}
    assert sel.selection_report_[P1]["union_positives"] == 4


def test_union_leakage_cap_infeasible_raises():
    n, y, rules = _leak_case()
    with pytest.raises(SelectorInfeasibleError) as err:
        run_pass1(make_selector(), rules[:2], y, 1, 2, leak_rate=0.0)
    assert err.value.proven is True
    assert "union leakage" in str(err.value) and "<= 0 of 50" in str(err.value)


def test_fallback_respects_union_leakage_cap():
    n, y, rules = _leak_case()
    sel = make_selector()
    script_solver(sel, [no_solution(sel.ilp_builder)])
    chosen = run_pass1(sel, rules, y, 1, 3, leak_rate=0.10)
    assert sel.selection_report_[P1]["source"] == "greedy_fallback"
    assert ids(chosen) == {"A", "C"}
    leakage = (y == 1, 5)
    assert sel._meets_hard_constraints(chosen, 1, 3, None, (0.20, 1), n, leakage)
    assert not sel._meets_hard_constraints(rules, 1, 3, None, (0.20, 1), n, leakage)


def test_union_leakage_cap_matches_exhaustive_search():
    n = 60
    checked = {"feasible": 0, "infeasible": 0}
    for seed in range(40):
        r = np.random.RandomState(500 + seed)
        y = (r.rand(n) < 0.15).astype(int)
        total = int(y.sum())
        rules = []
        for j in range(9):
            start = r.randint(0, 45)
            rows = set(range(start, min(n, start + r.randint(5, 16)))) | set(r.randint(0, n, r.randint(0, 6)))
            rules.append(FakeRule(f"r{j}", rows, n, precision=float(r.uniform(0.93, 1.0)),
                                  base=f"b{r.randint(0, 3)}"))
        rate = [0.1, 0.2, 0.35, 0.5][seed % 4]
        lo, hi = [(1, 3), (2, 4), (3, 5)][seed % 3]
        cap = int(np.floor(rate * total + 1e-9))
        expected = _brute_force(rules, n, lo, hi, 0.2, 2, _StubValidator(), y=y, leak_cap=cap)
        sel = make_selector()
        if expected is None:
            with pytest.raises(SelectorInfeasibleError):
                run_pass1(sel, rules, y, lo, hi, max_base_reuse=2, leak_rate=rate)
            checked["infeasible"] += 1
            continue
        chosen = run_pass1(sel, rules, y, lo, hi, max_base_reuse=2, leak_rate=rate)
        arrs = [get_covered_array(c, n) for c in chosen]
        union = union_size(chosen)
        assert (union, -(sum(a.size for a in arrs) - union)) == expected, f"seed {seed}"
        covered = list(set().union(*(c._cached_covered_idx for c in chosen)))
        assert int(y[covered].sum()) <= cap
        checked["feasible"] += 1
    print(checked)
    assert checked["feasible"] >= 10 and checked["infeasible"] >= 3, checked


def test_union_leakage_rate_validation():
    with pytest.raises(ValueError, match="max_union_leakage_rate_pass1"):
        make_selector(max_union_leakage_rate_pass1=1.5)
    from glass_pipeline.glass_router.core.config import GlassRouterConfig
    base = dict(
        mode="strict", min_support_pass1=120, min_support_pass2=25,
        max_leakage_rate_depth2=0.9, max_leakage_fraction_depth2=0.9,
        max_jaccard_overlap=0.5, max_high_overlap_rules=20,
        min_pass1_rules=2, max_pass1_rules=8,
        min_precision_not_subscribe=0.93, max_precision_not_subscribe=1.0,
        max_subscriber_leakage_rate=0.05, max_subscriber_leakage_absolute=800,
        min_pass2_rules=2, max_pass2_rules=8,
        min_precision_subscribe=0.2, max_precision_subscribe=1.0,
        min_recall_subscribe=0.01, max_recall_subscribe=0.35,
        min_novelty_ratio_pass1=0.2, min_novelty_ratio_pass2=0.25,
        enable_novelty_constraints=True, max_complexity=3, diversity_weight=0.35,
    )
    assert GlassRouterConfig(**base).max_union_leakage_rate_pass1 is None      # optional
    assert GlassRouterConfig(**base, max_union_leakage_rate_pass1=0.15).max_union_leakage_rate_pass1 == 0.15
    with pytest.raises(ValueError, match="max_union_leakage_rate_pass1"):
        GlassRouterConfig(**base, max_union_leakage_rate_pass1=1.2)


def test_select_rules_applies_cap_to_pass1(monkeypatch):
    monkeypatch.setattr(ilp_mod, "SelectedRule", _StubSelected)
    n, y, rules = _leak_case()
    sel = make_selector(min_pass1_rules=1, max_pass1_rules=3, max_union_leakage_rate_pass1=0.10)
    out = sel.select_rules(rules, y_val=y, passes=("pass1",))
    assert ids(out["pass1_rules"]) == {"A", "C"}
    assert sel.selection_report_[P1]["union_leakage_cap"] == 5
