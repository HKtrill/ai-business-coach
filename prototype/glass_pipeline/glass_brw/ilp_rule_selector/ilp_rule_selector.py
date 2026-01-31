# ============================================================
# GLASS-BRW: ILP RULE SELECTOR MODULE (REFACTORED)
# ============================================================
# Main orchestrator for ILP-based rule selection using modular components
# 
# Input:  List[EvaluatedRule] from RuleEvaluator
# Output: Dict with List[SelectedRule] for each pass
# ============================================================

from typing import List, Dict, Tuple, Optional

from .quality_gate_filter import QualityGateFilter
from .novelty_analyzer import NoveltyAnalyzer
from .diversity_analyzer import DiversityAnalyzer
from .ilp_builder import ILPBuilder
from .greedy_selector import GreedySelector
from .ilp_deduplicator import ILPDeduplicator
from glass_brw.rule_generator.feature_validator import FeatureValidator
from glass_brw.core.rule import EvaluatedRule, SelectedRule


class ILPRuleSelector:
    """
    Selects optimal rule subsets via Integer Linear Programming.
    
    Input:  List[EvaluatedRule] from RuleEvaluator
    Output: Dict with List[SelectedRule] for pass1_rules and pass2_rules
    
    Key Features:
    - All tuning parameters must be explicitly passed (no hidden defaults)
    - Greedy fallback uses HARD novelty cutoff by default
    - Coverage caps in quality gates
    - Modular architecture with reusable components
    """
    
    def __init__(
        self,
        # ============================================================
        # PASS 1 CONSTRAINTS - REQUIRED (tuning params)
        # ============================================================
        min_pass1_rules: Optional[int] = None,
        max_pass1_rules: Optional[int] = None,
        min_precision_pass1: Optional[float] = None,
        max_precision_pass1: Optional[float] = None,
        max_subscriber_leakage_rate_pass1: Optional[float] = None,
        max_subscriber_leakage_absolute_pass1: Optional[int] = None,
        max_base_reuse_pass1: Optional[int] = None,  # Optional - can be None
        
        # ============================================================
        # PASS 2 CONSTRAINTS - REQUIRED (tuning params)
        # ============================================================
        min_pass2_rules: Optional[int] = None,
        max_pass2_rules: Optional[int] = None,
        min_precision_pass2: Optional[float] = None,
        max_precision_pass2: Optional[float] = None,
        min_recall_pass2: Optional[float] = None,
        max_recall_pass2: Optional[float] = None,
        max_base_reuse_pass2: Optional[int] = None,  # Optional - can be None
        
        # ============================================================
        # NOVELTY CONSTRAINTS - REQUIRED (tuning params)
        # ============================================================
        min_novelty_ratio_pass1: Optional[float] = None,
        min_novelty_ratio_pass2: Optional[float] = None,
        enable_novelty_constraints: Optional[bool] = None,
        
        # ============================================================
        # SHARED TUNING - REQUIRED
        # ============================================================
        diversity_weight: Optional[float] = None,
        
        # ============================================================
        # STRUCTURAL/BEHAVIORAL - KEEP DEFAULTS (rarely tuned)
        # ============================================================
        max_feature_usage: int = 40,
        lambda_rf_uncertainty: float = 0.15,
        lambda_rf_misalignment: float = 0.08,
        
        # Greedy fallback controls
        min_novelty_greedy: float = 0.15,
        greedy_novelty_weight: float = 0.5,
        greedy_hard_novelty_cutoff: bool = True,
        min_absolute_new_samples: int = 30,
        
        # Feature validation
        tier1_prefixes: Tuple[str, ...] = (
            'previous', 'nr_employed', 'euribor', 'emp_var', 'cpi', 'cci',
            'month', 'contact', 'age', 'campaign', 'job', 'marital',
            'education', 'dow', 'default', 'housing', 'loan',
            'econ', 'prospect',
        ),
    ):
        """
        Initialize ILP rule selector with modular components.
        
        Raises:
            ValueError: If any required tuning parameter is None
        """
        # ============================================================
        # VALIDATE REQUIRED PARAMETERS
        # ============================================================
        required_params = {
            'min_pass1_rules': min_pass1_rules,
            'max_pass1_rules': max_pass1_rules,
            'min_precision_pass1': min_precision_pass1,
            'max_precision_pass1': max_precision_pass1,
            'max_subscriber_leakage_rate_pass1': max_subscriber_leakage_rate_pass1,
            'max_subscriber_leakage_absolute_pass1': max_subscriber_leakage_absolute_pass1,
            'min_pass2_rules': min_pass2_rules,
            'max_pass2_rules': max_pass2_rules,
            'min_precision_pass2': min_precision_pass2,
            'max_precision_pass2': max_precision_pass2,
            'min_recall_pass2': min_recall_pass2,
            'max_recall_pass2': max_recall_pass2,
            'min_novelty_ratio_pass1': min_novelty_ratio_pass1,
            'min_novelty_ratio_pass2': min_novelty_ratio_pass2,
            'enable_novelty_constraints': enable_novelty_constraints,
            'diversity_weight': diversity_weight,
        }
        
        missing = [k for k, v in required_params.items() if v is None]
        if missing:
            raise ValueError(
                f"ILPRuleSelector missing required parameters: {missing}\n"
                f"All tuning parameters must be explicitly passed from GLASSBRWConfig."
            )
        
        # ============================================================
        # STORE PASS-LEVEL PARAMETERS
        # ============================================================
        self.min_pass1_rules = min_pass1_rules
        self.max_pass1_rules = max_pass1_rules
        self.max_base_reuse_pass1 = max_base_reuse_pass1
        
        self.min_pass2_rules = min_pass2_rules
        self.max_pass2_rules = max_pass2_rules
        self.max_base_reuse_pass2 = max_base_reuse_pass2
        
        self.min_novelty_ratio_pass1 = min_novelty_ratio_pass1
        self.min_novelty_ratio_pass2 = min_novelty_ratio_pass2
        
        # ============================================================
        # INITIALIZE MODULAR COMPONENTS
        # ============================================================
        self.validator = FeatureValidator(tier1_prefixes=tier1_prefixes)
        
        self.quality_filter = QualityGateFilter(
            min_precision_pass1=min_precision_pass1,
            max_precision_pass1=max_precision_pass1,
            max_subscriber_leakage_rate_pass1=max_subscriber_leakage_rate_pass1,
            max_subscriber_leakage_absolute_pass1=max_subscriber_leakage_absolute_pass1,
            min_precision_pass2=min_precision_pass2,
            max_precision_pass2=max_precision_pass2,
            min_recall_pass2=min_recall_pass2,
            max_recall_pass2=max_recall_pass2,
        )
        
        self.novelty_analyzer = NoveltyAnalyzer(
            enable_novelty_constraints=enable_novelty_constraints
        )
        
        self.diversity_analyzer = DiversityAnalyzer(
            validator=self.validator,
            max_feature_usage=max_feature_usage
        )
        
        self.ilp_builder = ILPBuilder(
            lambda_rf_uncertainty=lambda_rf_uncertainty,
            lambda_rf_misalignment=lambda_rf_misalignment,
        )
        
        self.greedy_selector = GreedySelector(
            diversity_analyzer=self.diversity_analyzer,
            min_novelty_greedy=min_novelty_greedy,
            greedy_novelty_weight=greedy_novelty_weight,
            greedy_hard_novelty_cutoff=greedy_hard_novelty_cutoff,
            min_absolute_new_samples=min_absolute_new_samples,
            diversity_weight=diversity_weight,
        )
        
        self.deduplicator = ILPDeduplicator()
        
        print("✅ ILPRuleSelector initialized (all tuning params validated)")
    
    # ============================================================
    # STRUCTURAL FILTERING
    # ============================================================
    
    def _filter_invalid_rules(
        self, 
        candidates: List[EvaluatedRule]
    ) -> Tuple[List[EvaluatedRule], List[EvaluatedRule]]:
        """
        Filter out structurally invalid rules.
        
        Args:
            candidates: List of EvaluatedRule objects
            
        Returns:
            (valid_rules, rejected_rules) tuple
        """
        valid, rejected = [], []
        
        for rule in candidates:
            if self.validator.has_duplicate_base_features(rule.segment):
                rejected.append(rule)
            else:
                valid.append(rule)
        
        return valid, rejected
    
    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================
    
    def select_rules(
        self, 
        evaluated_rules: List[EvaluatedRule], 
        y_val=None,
        X_val=None,
        segment_builder=None,
    ) -> Dict[str, List[SelectedRule]]:
        """
        Public entry point for ILP rule selection.
        
        Args:
            evaluated_rules: List of EvaluatedRule objects from RuleEvaluator
            y_val: Validation labels (required for Pass 1 leakage calculation)
            X_val: Validation features (for computing covered indices)
            segment_builder: Segment builder (for computing covered indices)
            
        Returns:
            Dict with {"pass1_rules": List[SelectedRule], "pass2_rules": List[SelectedRule]}
        """
        if not evaluated_rules:
            print("⚠️  No evaluated rules provided to ILP selector.")
            return {"pass1_rules": [], "pass2_rules": []}
        
        print("\n" + "=" * 80)
        print("🧮 ILP RULE SELECTION")
        print("=" * 80)
        
        # Precompute covered indices if X_val provided
        if X_val is not None:
            print("  Computing covered indices for novelty analysis...")
            self._precompute_covered_indices(evaluated_rules, X_val, segment_builder)
        
        self._print_configuration()
        
        # Filter invalid rules
        valid_rules, rejected = self._filter_invalid_rules(evaluated_rules)
        if rejected:
            print(f"  🚫 Rejected {len(rejected)} structurally invalid rules")
        
        if not valid_rules:
            print("⚠️  No valid rules remain after structural filtering.")
            return {"pass1_rules": [], "pass2_rules": []}
        
        # Split by pass
        pass1_candidates = [r for r in valid_rules if r.predicted_class == 0]
        pass2_candidates = [r for r in valid_rules if r.predicted_class == 1]
        
        print(f"\nCandidate split:")
        print(f"  Pass 1 candidates: {len(pass1_candidates)}")
        print(f"  Pass 2 candidates: {len(pass2_candidates)}")
        
        # Optimize each pass (returns EvaluatedRule lists)
        pass1_evaluated = self._optimize_pass(
            candidates=pass1_candidates,
            min_rules=self.min_pass1_rules,
            max_rules=self.max_pass1_rules,
            pass_name="Pass 1 (NOT_SUBSCRIBE)",
            scoring_mode="precision_first",
            y_val=y_val,
            max_base_reuse=self.max_base_reuse_pass1,
            min_novelty_ratio=self.min_novelty_ratio_pass1,
        )
        
        pass2_evaluated = self._optimize_pass(
            candidates=pass2_candidates,
            min_rules=self.min_pass2_rules,
            max_rules=self.max_pass2_rules,
            pass_name="Pass 2 (SUBSCRIBE)",
            scoring_mode="recall_first",
            y_val=None,
            max_base_reuse=self.max_base_reuse_pass2,
            min_novelty_ratio=self.min_novelty_ratio_pass2,
        )
        
        # Post-selection analysis
        self.novelty_analyzer.analyze_selection_novelty(pass1_evaluated, "Pass 1")
        self.novelty_analyzer.analyze_selection_novelty(pass2_evaluated, "Pass 2")
        
        # Convert EvaluatedRule → SelectedRule
        pass1_selected = [
            SelectedRule.from_evaluated(r, pass_assignment="pass1") 
            for r in pass1_evaluated
        ]
        pass2_selected = [
            SelectedRule.from_evaluated(r, pass_assignment="pass2") 
            for r in pass2_evaluated
        ]
        
        print("\n✅ ILP SELECTION COMPLETE")
        print(f"  Pass 1 selected: {len(pass1_selected)} rules")
        print(f"  Pass 2 selected: {len(pass2_selected)} rules")
        
        return {
            "pass1_rules": pass1_selected,
            "pass2_rules": pass2_selected,
        }
    
    def _precompute_covered_indices(
        self,
        rules: List[EvaluatedRule],
        X_val,
        segment_builder,
    ):
        """
        Precompute and cache covered indices for novelty calculations.
        
        Stores results in a temporary dict on each rule for ILP use.
        """
        for rule in rules:
            covered = rule.compute_covered_indices(X_val, segment_builder)
            # Store temporarily for ILP (will be discarded after selection)
            rule._cached_covered_idx = covered
    
    # ============================================================
    # PASS OPTIMIZATION
    # ============================================================
    
    def _optimize_pass(
        self,
        candidates: List[EvaluatedRule],
        min_rules: int,
        max_rules: int,
        pass_name: str,
        scoring_mode: str,
        y_val=None,
        max_base_reuse: Optional[int] = None,
        min_novelty_ratio: float = 0.20,
    ) -> List[EvaluatedRule]:
        """
        Optimize a single pass using ILP or greedy fallback.
        
        Returns List[EvaluatedRule] (conversion to SelectedRule happens in select_rules)
        """
        if not candidates:
            print(f"⚠️  No candidates for {pass_name}")
            return []
        
        print(f"\n🧪 ENTERING ILP OPTIMIZATION: {pass_name}")
        print("-" * 80)
        print(f"Total incoming candidates: {len(candidates)}")
        
        # Apply quality gates
        if "Pass 1" in pass_name and y_val is not None:
            valid, rejected = self.quality_filter.apply_quality_gates_pass1(candidates, y_val)
        else:
            valid, rejected = self.quality_filter.apply_quality_gates_pass2(candidates)
        
        print(f"  ✅ Passed gates: {len(valid)}/{len(candidates)}")
        print(f"  ❌ Rejected: {len(rejected)}/{len(candidates)}")
        
        if len(valid) == 0:
            print(f"  ⚠️  No valid candidates after quality gates!")
            return []
        
        # Deduplicate
        valid = self.deduplicator.deduplicate_by_segment(valid, scoring_mode)
        print(f"  After deduplication: {len(valid)} unique rules")
        
        # Adjust cardinality bounds
        n_valid = len(valid)
        actual_min = max(0, min(min_rules, n_valid))
        actual_max = max(1, min(max_rules, n_valid))
        if actual_min > actual_max:
            actual_min = actual_max
        
        # Build ILP
        prob = self.ilp_builder.create_problem(pass_name)
        decision_vars = self.ilp_builder.create_decision_variables(valid)
        
        # Set objective
        objective_terms = self.ilp_builder.build_objective(valid, decision_vars, scoring_mode)
        prob += sum(objective_terms)
        
        # Add constraints
        self.ilp_builder.add_cardinality_constraints(
            prob, valid, decision_vars, actual_min, actual_max
        )
        
        self.diversity_analyzer.add_diversity_constraints(
            prob, valid, decision_vars, max_base_reuse
        )
        
        n_constraints = self.novelty_analyzer.add_novelty_constraints(
            prob, valid, decision_vars, min_novelty_ratio
        )
        
        # Solve
        print(f"\nSolving ILP for {pass_name}...")
        print(f"  Variables: {len(decision_vars)}")
        print(f"  Cardinality: [{actual_min}, {actual_max}]")
        print(f"  Novelty constraints: {n_constraints}")
        
        if n_constraints > len(valid) * (len(valid) - 1) // 4:
            print(f"  ⚠️  High constraint ratio - ILP may be infeasible")
        
        status = self.ilp_builder.solve(prob, time_limit=300)
        print(f"  Status: {status}")
        
        if status != "Optimal":
            return self.greedy_selector.greedy_select(valid, actual_max, scoring_mode)
        
        # Extract solution
        selected = self.ilp_builder.extract_selected_rules(valid, decision_vars)
        print(f"  ✅ Selected {len(selected)} rules")
        
        return selected
    
    # ============================================================
    # CONFIGURATION PRINTING
    # ============================================================
    
    def _print_configuration(self):
        """Print ILP selector configuration."""
        print(f"\n📊 Configuration:")
        print(f"   Novelty constraints enabled: {self.novelty_analyzer.enable_novelty_constraints}")
        if self.novelty_analyzer.enable_novelty_constraints:
            print(f"   Pass 1 min novelty (ILP): {self.min_novelty_ratio_pass1:.0%}")
            print(f"   Pass 2 min novelty (ILP): {self.min_novelty_ratio_pass2:.0%}")
        print(f"   Greedy min novelty: {self.greedy_selector.min_novelty_greedy:.0%}")
        print(f"   Greedy hard cutoff: {self.greedy_selector.greedy_hard_novelty_cutoff}")
        print(f"   Min absolute new samples: {self.greedy_selector.min_absolute_new_samples}")
        print(f"   Pass 1 coverage: [{self.quality_filter.min_coverage_pass1}, {self.quality_filter.max_coverage_pass1}]")
        print(f"   Pass 2 coverage: [{self.quality_filter.min_coverage_pass2}, {self.quality_filter.max_coverage_pass2}]")
    
    def __repr__(self):
        return (
            f"ILPRuleSelector("
            f"pass1=[{self.min_pass1_rules}-{self.max_pass1_rules}], "
            f"pass2=[{self.min_pass2_rules}-{self.max_pass2_rules}], "
            f"novelty={self.novelty_analyzer.enable_novelty_constraints})"
        )