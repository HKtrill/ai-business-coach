# ============================================================
# GLASS ROUTER: ILP BUILDER MODULE
# ============================================================
# Construct the Integer Linear Programming optimization problem
# Works with EvaluatedRule objects
# ============================================================

from typing import List, Dict, Tuple

import numpy as np
from pulp import (
    LpProblem, LpVariable, LpMaximize, LpAffineExpression, lpSum, PULP_CBC_CMD, LpStatus,
)

from glass_pipeline.glass_router.core.rule import EvaluatedRule
from .utils import compute_rule_quality, membership_atoms


class ILPBuilder:
    """Build and solve Integer Linear Programming problems for rule selection."""

    def __init__(
        self,
        lambda_rf_uncertainty: float,
        lambda_rf_misalignment: float,
    ):
        self.lambda_rf_uncertainty = lambda_rf_uncertainty
        self.lambda_rf_misalignment = lambda_rf_misalignment
        self.last_sol_status = None

    def adjusted_quality(self, rule: EvaluatedRule, scoring_mode: str) -> float:
        """Standalone per-rule quality minus RF penalties (secondary criterion only)."""
        return (
            compute_rule_quality(rule, scoring_mode)
            - self.lambda_rf_uncertainty * (1.0 - rule.rf_confidence)
            - self.lambda_rf_misalignment * (1.0 - rule.rf_alignment)
        )

    def build_objective(
        self,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable],
        scoring_mode: str
    ) -> List:
        """
        Per-rule quality terms  sum_j q_j * x_j.

        This is NOT a coverage objective: overlapping rows are credited once
        per rule. Not used by the current selection path:
        ILPRuleSelector._optimize_pass builds its Phase 2 quality tie-break
        inline from adjusted_quality. Kept for compatibility.
        """
        return [
            decision_vars[rule.rule_id] * self.adjusted_quality(rule, scoring_mode)
            for rule in rules
        ]

    def add_union_coverage(
        self,
        prob: LpProblem,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable],
        arrays: Dict[int, np.ndarray],
        row_weight: np.ndarray,
        atoms=None,
    ) -> Tuple[LpAffineExpression, Dict[int, int], int, List[Tuple[LpVariable, List[int]]]]:
        """
        Add union-coverage variables and return the union-coverage expression.

        z_a = 1 iff atom a (rows with one membership pattern) is covered by at
        least one selected rule:
            z_a <= sum_{j covers a} x_j,   0 <= z_a <= 1
        Maximising sum_a w_a z_a counts each covered row ONCE, however many
        selected rules match it (w_a = summed row weight of the atom).
        Atoms matched by a single rule need no z variable (z = x_j at the
        optimum), so they contribute w_a * x_j directly.

        Args:
            atoms: (members, atom_of_row) from utils.membership_atoms; computed
                   here when not supplied.

        Returns:
            (coverage_expr, raw_weight_by_rule_id, n_weighted_atoms, z_links)
            z_links: [(z_var, [rule_ids covering that atom]), ...]
        """
        n_rows = len(row_weight)
        ids = [r.rule_id for r in rules]
        raw_weight = {rid: int(row_weight[arrays[rid]].sum()) for rid in ids}
        members, atom_of_row = atoms if atoms is not None else membership_atoms(ids, arrays, n_rows)
        if not members:
            return lpSum([]), raw_weight, 0, []

        covered = atom_of_row >= 0
        weight = np.bincount(
            atom_of_row[covered], weights=row_weight[covered], minlength=len(members)
        ).round().astype(np.int64)

        terms = []
        z_links = []
        n_weighted = 0
        for a, member_ids in enumerate(members):
            w = int(weight[a])
            if w <= 0:
                continue
            n_weighted += 1
            if len(member_ids) == 1:
                terms.append(w * decision_vars[member_ids[0]])
                continue
            z = LpVariable(f"z_{len(z_links)}", lowBound=0, upBound=1)
            prob += z <= lpSum(decision_vars[rid] for rid in member_ids), f"link_z_{len(z_links)}"
            terms.append(w * z)
            z_links.append((z, list(member_ids)))

        return lpSum(terms), raw_weight, n_weighted, z_links

    def add_union_leakage_cap(
        self,
        prob: LpProblem,
        decision_vars: Dict[int, LpVariable],
        atoms,
        pos_mask: np.ndarray,
        cap: int,
    ) -> int:
        """
        Cap the positives (subscribers) routed by the selected UNION:
            sum_a p_a * u_a <= cap,   u_a >= x_j for every rule j covering atom a
        p_a = positives in atom a. The lower-bound links force u_a = 1 whenever
        any selected rule covers the atom, so every covered positive is counted
        exactly once (the coverage variables z_a are only upper-bounded and
        could not be used here). Single-rule atoms use p_a * x_j directly.

        Returns:
            number of atoms carrying positives
        """
        members, atom_of_row = atoms
        covered = atom_of_row >= 0
        pos_counts = np.bincount(
            atom_of_row[covered], weights=pos_mask[covered].astype(float), minlength=len(members)
        ).round().astype(np.int64)
        terms = []
        n_pos_atoms = 0
        for a, member_ids in enumerate(members):
            p_a = int(pos_counts[a])
            if p_a == 0:
                continue
            n_pos_atoms += 1
            if len(member_ids) == 1:
                terms.append(p_a * decision_vars[member_ids[0]])
                continue
            u = LpVariable(f"leak_{a}", lowBound=0, upBound=1)
            for k, rid in enumerate(member_ids):
                prob += u >= decision_vars[rid], f"leak_link_{a}_{k}"
            terms.append(p_a * u)
        prob += lpSum(terms) <= int(cap), "union_leakage_cap"
        return n_pos_atoms

    def add_cardinality_constraints(
        self,
        prob: LpProblem,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable],
        min_rules: int,
        max_rules: int
    ):
        """
        Add constraints on the number of rules selected.

        Args:
            prob: PuLP problem object
            rules: List of EvaluatedRule objects
            decision_vars: Dict mapping rule_id to LpVariable
            min_rules: Minimum number of rules to select
            max_rules: Maximum number of rules to select
        """
        total_vars = lpSum(decision_vars[r.rule_id] for r in rules)
        prob += total_vars >= min_rules, "min_rules"
        prob += total_vars <= max_rules, "max_rules"

    def solve(
        self,
        prob: LpProblem,
        time_limit: int = 300,
        warm_start: bool = False,
        gap_rel: float = None,
        gap_abs: float = None,
    ) -> str:
        """
        Solve the ILP problem.

        Args:
            prob: PuLP problem object
            time_limit: Time limit in seconds
            warm_start: pass current variable values to CBC as a MIP start
            gap_rel: stop once the solution is proven within this relative gap
            gap_abs: stop once the solution is proven within this absolute gap

        Returns:
            Status string ("Optimal", "Infeasible", etc.). PuLP can report
            "Optimal" for a run stopped by the time limit (even with an unusable
            incumbent); self.last_sol_status records PuLP's solution status
            (1 = proven optimal, 2 = feasible incumbent only, 0 = none,
            -1 = infeasible). Callers classify with both values and validate
            any returned solution.
        """
        solver = PULP_CBC_CMD(
            msg=0, timeLimit=max(1, int(time_limit)), warmStart=warm_start,
            gapRel=gap_rel, gapAbs=gap_abs,
        )
        prob.solve(solver)
        self.last_sol_status = getattr(prob, "sol_status", None)
        return LpStatus[prob.status]

    def extract_selected_rules(
        self,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable]
    ) -> List[EvaluatedRule]:
        """
        Extract rules that were selected in the solution.

        Args:
            rules: List of EvaluatedRule objects
            decision_vars: Dict mapping rule_id to LpVariable

        Returns:
            List of selected EvaluatedRule objects
        """
        return [
            r for r in rules
            if decision_vars[r.rule_id].varValue is not None
            and decision_vars[r.rule_id].varValue > 0.5
        ]

    def create_problem(self, pass_name: str) -> LpProblem:
        problem_name = f"GLASS_ROUTER_{pass_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        return LpProblem(problem_name, LpMaximize)

    def create_decision_variables(self, rules: List[EvaluatedRule]) -> Dict[int, LpVariable]:
        return {
            r.rule_id: LpVariable(f"x_{r.rule_id}", cat="Binary")
            for r in rules
        }
