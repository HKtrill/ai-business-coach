# ============================================================
# GLASS-BRW: ILP BUILDER MODULE
# ============================================================
# Construct the Integer Linear Programming optimization problem
# Works with EvaluatedRule objects
# ============================================================

from typing import List, Dict
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD, LpStatus

from glass_brw.core.rule import EvaluatedRule
from .utils import compute_rule_quality


class ILPBuilder:
    """Build and solve Integer Linear Programming problems for rule selection."""

    def __init__(
        self,
        lambda_rf_uncertainty: float,
        lambda_rf_misalignment: float,
    ):
        self.lambda_rf_uncertainty = lambda_rf_uncertainty
        self.lambda_rf_misalignment = lambda_rf_misalignment

    def build_objective(
        self,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable],
        scoring_mode: str
    ) -> List:
        terms = []
        for rule in rules:
            quality = compute_rule_quality(rule, scoring_mode)
            adjusted_quality = (
                quality
                - self.lambda_rf_uncertainty * (1.0 - rule.rf_confidence)
                - self.lambda_rf_misalignment * (1.0 - rule.rf_alignment)
            )
            terms.append(decision_vars[rule.rule_id] * adjusted_quality)
        return terms
    
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
    
    def solve(self, prob: LpProblem, time_limit: int = 300) -> str:
        """
        Solve the ILP problem.
        
        Args:
            prob: PuLP problem object
            time_limit: Time limit in seconds
            
        Returns:
            Status string ("Optimal", "Infeasible", etc.)
        """
        solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        prob.solve(solver)
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
        problem_name = f"GLASS_BRW_{pass_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        return LpProblem(problem_name, LpMaximize)

    def create_decision_variables(self, rules: List[EvaluatedRule]) -> Dict[int, LpVariable]:
        return {
            r.rule_id: LpVariable(f"x_{r.rule_id}", cat="Binary")
            for r in rules
        }