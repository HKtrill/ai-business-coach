# ============================================================
# GLASS-BRW: ILP RULE SELECTOR MODULE (REFACTORED)
# ============================================================
# Main orchestrator for ILP-based rule selection using modular components
# ============================================================

from .quality_gate_filter import QualityGateFilter
from .novelty_analyzer import NoveltyAnalyzer
from .diversity_analyzer import DiversityAnalyzer
from .ilp_builder import ILPBuilder
from .greedy_selector import GreedySelector
from .ilp_deduplicator import ILPDeduplicator
from glass_brw.rule_generator.feature_validator import FeatureValidator


class ILPRuleSelector:
    """
    Selects optimal rule subsets via Integer Linear Programming.
    
    Key Features:
    - Lower default novelty thresholds (20% instead of 50%)
    - Greedy fallback uses HARD novelty cutoff by default
    - Coverage caps in quality gates
    - Modular architecture with reusable components
    """
    
    def __init__(
        self,
        # Pass 1 constraints
        min_pass1_rules=5,
        max_pass1_rules=12,
        min_precision_pass1=0.25,
        max_precision_pass1=1.00,
        max_subscriber_leakage_rate_pass1=0.15,
        max_subscriber_leakage_absolute_pass1=150,
        max_base_reuse_pass1=None,
        min_coverage_pass1=0.005,
        max_coverage_pass1=0.75,
        # Pass 2 constraints
        min_pass2_rules=5,
        max_pass2_rules=12,
        min_precision_pass2=0.05,
        max_precision_pass2=1.00,
        min_recall_pass2=0.10,
        max_recall_pass2=0.99,
        max_base_reuse_pass2=None,
        min_coverage_pass2=0.005,
        max_coverage_pass2=0.35,
        # Shared parameters
        max_feature_usage=40,
        lambda_rf_uncertainty=0.15,
        lambda_rf_misalignment=0.08,
        diversity_weight=0.33,
        # Novelty constraints
        min_novelty_ratio_pass1=0.20,
        min_novelty_ratio_pass2=0.20,
        enable_novelty_constraints=True,
        # Greedy fallback controls
        min_novelty_greedy=0.15,
        greedy_novelty_weight=0.5,
        greedy_hard_novelty_cutoff=True,
        min_absolute_new_samples=30,
        # Feature validation
        tier1_prefixes=(
            'previous', 'nr_employed', 'euribor', 'emp_var', 'cpi', 'cci',
            'month', 'contact', 'age', 'campaign', 'job', 'marital',
            'education', 'dow', 'default', 'housing', 'loan',
            'econ', 'prospect',
        ),
    ):
        """Initialize ILP rule selector with modular components."""
        
        # Store pass-level parameters
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
        
        # Feature validation (REUSED from rule_generator)
        self.validator = FeatureValidator(tier1_prefixes=tier1_prefixes)
        
        # Quality gate filtering
        self.quality_filter = QualityGateFilter(
            min_precision_pass1=min_precision_pass1,
            max_precision_pass1=max_precision_pass1,
            max_subscriber_leakage_rate_pass1=max_subscriber_leakage_rate_pass1,
            max_subscriber_leakage_absolute_pass1=max_subscriber_leakage_absolute_pass1,
            min_coverage_pass1=min_coverage_pass1,
            max_coverage_pass1=max_coverage_pass1,
            min_precision_pass2=min_precision_pass2,
            max_precision_pass2=max_precision_pass2,
            min_recall_pass2=min_recall_pass2,
            max_recall_pass2=max_recall_pass2,
            min_coverage_pass2=min_coverage_pass2,
            max_coverage_pass2=max_coverage_pass2,
        )
        
        # Novelty analysis
        self.novelty_analyzer = NoveltyAnalyzer(
            enable_novelty_constraints=enable_novelty_constraints
        )
        
        # Diversity analysis
        self.diversity_analyzer = DiversityAnalyzer(
            validator=self.validator,
            max_feature_usage=max_feature_usage
        )
        
        # ILP building
        self.ilp_builder = ILPBuilder(
            lambda_rf_uncertainty=lambda_rf_uncertainty,
            lambda_rf_misalignment=lambda_rf_misalignment,
        )
        
        # Greedy fallback
        self.greedy_selector = GreedySelector(
            diversity_analyzer=self.diversity_analyzer,
            min_novelty_greedy=min_novelty_greedy,
            greedy_novelty_weight=greedy_novelty_weight,
            greedy_hard_novelty_cutoff=greedy_hard_novelty_cutoff,
            min_absolute_new_samples=min_absolute_new_samples,
            diversity_weight=diversity_weight,
        )
        
        # Deduplication
        self.deduplicator = ILPDeduplicator()
    
    # ============================================================
    # STRUCTURAL FILTERING
    # ============================================================
    
    def _filter_invalid_rules(self, candidates: list) -> tuple:
        """
        Filter out structurally invalid rules.
        
        Args:
            candidates: List of Rule objects
            
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
    
    def select_rules(self, evaluated_rules: list, y_val=None) -> dict:
        """
        Public entry point for ILP rule selection.
        
        Args:
            evaluated_rules: List of evaluated Rule objects
            y_val: Validation labels (required for Pass 1)
            
        Returns:
            Dict with {"pass1_rules": [...], "pass2_rules": [...]}
        """
        if not evaluated_rules:
            print("⚠️  No evaluated rules provided to ILP selector.")
            return {"pass1_rules": [], "pass2_rules": []}
        
        print("\n" + "=" * 80)
        print("🧮 ILP RULE SELECTION")
        print("=" * 80)
        
        # Print configuration
        self._print_configuration()
        
        # Filter invalid rules
        valid_rules, rejected = self._filter_invalid_rules(evaluated_rules)
        if rejected:
            print(f"  🚫 Rejected {len(rejected)} structurally invalid rules")
        
        if not valid_rules:
            print("⚠️  No valid rules remain after structural filtering.")
            return {"pass1_rules": [], "pass2_rules": []}
        
        # Split by pass
        pass1_candidates = [r for r in valid_rules if getattr(r, "predicted_class", None) == 0]
        pass2_candidates = [r for r in valid_rules if getattr(r, "predicted_class", None) == 1]
        
        print(f"\nCandidate split:")
        print(f"  Pass 1 candidates: {len(pass1_candidates)}")
        print(f"  Pass 2 candidates: {len(pass2_candidates)}")
        
        # Optimize each pass
        pass1_selected = self._optimize_pass(
            candidates=pass1_candidates,
            min_rules=self.min_pass1_rules,
            max_rules=self.max_pass1_rules,
            pass_name="Pass 1 (NOT_SUBSCRIBE)",
            scoring_mode="precision_first",
            y_val=y_val,
            max_base_reuse=self.max_base_reuse_pass1,
            min_novelty_ratio=self.min_novelty_ratio_pass1,
        )
        
        pass2_selected = self._optimize_pass(
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
        self.novelty_analyzer.analyze_selection_novelty(pass1_selected, "Pass 1")
        self.novelty_analyzer.analyze_selection_novelty(pass2_selected, "Pass 2")
        
        print("\n✅ ILP SELECTION COMPLETE")
        print(f"  Pass 1 selected: {len(pass1_selected)} rules")
        print(f"  Pass 2 selected: {len(pass2_selected)} rules")
        
        return {
            "pass1_rules": pass1_selected,
            "pass2_rules": pass2_selected,
        }
    
    # ============================================================
    # PASS OPTIMIZATION
    # ============================================================
    
    def _optimize_pass(
        self,
        candidates: list,
        min_rules: int,
        max_rules: int,
        pass_name: str,
        scoring_mode: str,
        y_val=None,
        max_base_reuse: int = None,
        min_novelty_ratio: float = 0.20,
    ) -> list:
        """Optimize a single pass using ILP or greedy fallback."""
        
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