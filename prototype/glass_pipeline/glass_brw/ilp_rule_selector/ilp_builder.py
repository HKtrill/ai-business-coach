# ============================================================
# GLASS-BRW: ILP BUILDER MODULE
# ============================================================
# Construct the Integer Linear Programming optimization problem
# ============================================================
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD, LpStatus


class ILPBuilder:
    """Build and solve Integer Linear Programming problems for rule selection."""
    
    def __init__(
        self,
        lambda_rf_uncertainty: float = 0.15,
        lambda_rf_misalignment: float = 0.08,
    ):
        """
        Initialize ILP builder.
        
        Args:
            lambda_rf_uncertainty: Penalty weight for RF uncertainty
            lambda_rf_misalignment: Penalty weight for RF misalignment
        """
        self.lambda_rf_uncertainty = lambda_rf_uncertainty
        self.lambda_rf_misalignment = lambda_rf_misalignment
    
    def build_objective(
        self,
        rules: list,
        decision_vars: dict,
        scoring_mode: str
    ) -> list:
        """
        Build objective function terms for ILP.
        
        Args:
            rules: List of Rule objects
            decision_vars: Dict mapping rule_id to LpVariable
            scoring_mode: "precision_first" or "recall_first"
            
        Returns:
            List of objective function terms
        """
        terms = []
        
        for rule in rules:
            # Base quality score
            if scoring_mode == "precision_first":
                quality = (rule.precision ** 3) * rule.coverage
            else:  # recall_first
                quality = (rule.recall ** 3) * rule.precision * rule.coverage
            
            # RF penalties
            rf_uncertainty = 1.0 - getattr(rule, 'rf_confidence', 0.5)
            rf_misalignment = 1.0 - getattr(rule, 'rf_alignment', 0.0)
            
            # Adjusted quality
            adjusted_quality = (
                quality
                - self.lambda_rf_uncertainty * rf_uncertainty
                - self.lambda_rf_misalignment * rf_misalignment
            )
            
            terms.append(decision_vars[rule.rule_id] * adjusted_quality)
        
        return terms
    
    def create_problem(self, pass_name: str):
        """
        Create a new ILP problem.
        
        Args:
            pass_name: Name of the pass (for problem naming)
            
        Returns:
            PuLP LpProblem object
        """
        problem_name = f"GLASS_BRW_{pass_name.replace(' ', '_')}"
        return LpProblem(problem_name, LpMaximize)
    
    def create_decision_variables(self, rules: list) -> dict:
        """
        Create binary decision variables for each rule.
        
        Args:
            rules: List of Rule objects
            
        Returns:
            Dict mapping rule_id to LpVariable
        """
        return {
            r.rule_id: LpVariable(f"x_{r.rule_id}", cat="Binary")
            for r in rules
        }
    
    def add_cardinality_constraints(
        self,
        prob,
        rules: list,
        decision_vars: dict,
        min_rules: int,
        max_rules: int
    ):
        """
        Add constraints on the number of rules selected.
        
        Args:
            prob: PuLP problem object
            rules: List of Rule objects
            decision_vars: Dict mapping rule_id to LpVariable
            min_rules: Minimum number of rules to select
            max_rules: Maximum number of rules to select
        """
        total_vars = lpSum(decision_vars[r.rule_id] for r in rules)
        prob += total_vars >= min_rules, "min_rules"
        prob += total_vars <= max_rules, "max_rules"
    
    def solve(self, prob, time_limit: int = 300) -> str:
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
        rules: list,
        decision_vars: dict
    ) -> list:
        """
        Extract rules that were selected in the solution.
        
        Args:
            rules: List of Rule objects
            decision_vars: Dict mapping rule_id to LpVariable
            
        Returns:
            List of selected Rule objects
        """
        return [
            r for r in rules
            if decision_vars[r.rule_id].varValue is not None
            and decision_vars[r.rule_id].varValue > 0.5
        ]