# ============================================================
# GLASS-BRW: RULE EVALUATOR MODULE (REFACTORED)
# ============================================================
# Main orchestrator for rule evaluation using modular components
# ============================================================
import pandas as pd
import numpy as np

from .rule_matcher import RuleMatcher
from .evaluation_metrics import EvaluationMetrics
from .stability_estimator import StabilityEstimator
from .rf_integration import RFIntegration
from .rule_deduplicator import RuleDeduplicator
from .overlap_analyzer import OverlapAnalyzer
from glass_brw.rule_generator.feature_validator import FeatureValidator


class RuleEvaluator:
    """
    Validates and scores candidate rules using held-out validation data.
    Works exclusively with BINARY features (int8 values 0/1).
    """
    
    def __init__(
        self,
        segment_builder,
        min_support: int = 30,
        n_bootstrap: int = 3,
        tier1_prefixes: tuple = (
            'previous', 'nr_employed', 'euribor', 'emp_var', 'cpi', 'cci',
            'month', 'contact', 'age', 'campaign', 'job', 'marital',
            'education', 'dow', 'default', 'housing', 'loan',
            'econ', 'prospect',
        ),
    ):
        """
        Initialize rule evaluator with modular components.
        
        Args:
            segment_builder: SegmentBuilder instance for discretization
            min_support: Minimum support threshold for filtering rules
            n_bootstrap: Number of bootstrap samples for stability estimation
            tier1_prefixes: Tuple of tier1 feature prefixes
        """
        self.segment_builder = segment_builder
        self.min_support = min_support
        
        # ============================================================
        # INITIALIZE MODULAR COMPONENTS
        # ============================================================
        
        # Feature validation (reused from rule_generator)
        self.validator = FeatureValidator(tier1_prefixes=tier1_prefixes)
        
        # Rule matching
        self.matcher = RuleMatcher(segment_builder=segment_builder)
        
        # Metrics computation
        self.metrics = EvaluationMetrics()
        
        # Stability estimation
        self.stability = StabilityEstimator(
            segment_builder=segment_builder,
            n_bootstrap=n_bootstrap
        )
        
        # RF integration
        self.rf_integration = RFIntegration()
        
        # Deduplication
        self.deduplicator = RuleDeduplicator(validator=self.validator)
        
        # Overlap analysis
        self.overlap_analyzer = OverlapAnalyzer(matcher=self.matcher)
    
    # ============================================================
    # BACKWARD COMPATIBILITY METHODS (OPTIONAL)
    # ============================================================
    
    def match_rule(self, rule, segments_df):
        """Backward compatibility wrapper for matcher.match_rule()."""
        return self.matcher.match_rule(rule, segments_df)
    
    def _compute_precision(self, mask, y_val, predicted_class):
        """Backward compatibility wrapper for metrics.compute_precision()."""
        return self.metrics.compute_precision(mask, y_val, predicted_class)
    
    def _compute_recall(self, mask, y_val, predicted_class):
        """Backward compatibility wrapper for metrics.compute_recall()."""
        return self.metrics.compute_recall(mask, y_val, predicted_class)
    
    def _compute_coverage(self, mask, N):
        """Backward compatibility wrapper for metrics.compute_coverage()."""
        return self.metrics.compute_coverage(mask, N)
    
    # ============================================================
    # MAIN EVALUATION PIPELINE
    # ============================================================
    
    def evaluate_candidates(
        self,
        candidates: list,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        rf_model=None
    ) -> list:
        """
        Evaluate candidate rules on held-out validation data.
        
        Pipeline:
            1. Discretize validation features
            2. For each rule:
               - Match rule to samples (boolean mask)
               - Compute precision, recall, coverage
               - Estimate stability (bootstrap)
               - Compute RF metrics (if model provided)
            3. Filter low-coverage rules (< min_support/2)
            4. Deduplicate (structural + exact segment)
            5. Compute overlap diagnostics
        
        Args:
            candidates: Rule objects from RuleGenerator
            X_val: Validation features (raw)
            y_val: Validation labels (0 or 1)
            rf_model: RandomForestClassifier (optional) for confidence metrics
            
        Returns:
            Evaluated Rule objects with populated metrics:
                - precision, recall, coverage
                - stability (bootstrap-based)
                - rf_confidence, rf_alignment (if RF provided)
                - covered_idx (sample indices)
        
        Example:
            >>> evaluated = evaluator.evaluate_candidates(
            ...     candidates, X_val, y_val, rf_model
            ... )
            Evaluating 453 candidate rules...
            Evaluated 387 rules (filtered 66 low-coverage)
            Deduplicated rules: 387 → 352
        """
        # ============================================================
        # STEP 1: DISCRETIZE VALIDATION DATA
        # ============================================================
        segments_val = self.segment_builder.assign_segments(X_val)
        N = len(X_val)
        
        # ============================================================
        # STEP 2: GET RF PREDICTIONS (IF PROVIDED)
        # ============================================================
        rf_proba = self.rf_integration.get_rf_predictions(rf_model, X_val)
        
        print(f"\nEvaluating {len(candidates)} candidate rules...")
        
        # ============================================================
        # STEP 3: EVALUATE EACH RULE
        # ============================================================
        evaluated_rules = []
        
        for rule in candidates:
            # Match rule to samples
            mask = self.matcher.match_rule(rule, segments_val)
            n_matches = mask.sum()
            
            # Store covered sample indices (for ILP)
            rule.covered_idx = self.matcher.get_covered_indices(rule, segments_val)
            
            # Quality gate: minimum support
            if n_matches < self.min_support // 2:
                continue
            
            # Compute quality metrics
            rule_metrics = self.metrics.compute_all_metrics(
                mask, y_val, rule.predicted_class, N
            )
            rule.precision = rule_metrics['precision']
            rule.recall = rule_metrics['recall']
            rule.coverage = rule_metrics['coverage']
            
            # Estimate stability (SUBSCRIBE rules only)
            if rule.predicted_class == 1:
                rule.stability = self.stability.estimate_stability(
                    rule, X_val, y_val, self.matcher, self.metrics
                )
            else:
                rule.stability = 1.0
            
            # Compute RF metrics (if model provided)
            if rf_proba is not None:
                rule.rf_confidence, rule.rf_alignment = \
                    self.rf_integration.compute_rf_metrics(mask, rf_proba)
            else:
                rule.rf_confidence = 0.5
                rule.rf_alignment = 0.0
            
            evaluated_rules.append(rule)
        
        # ============================================================
        # STEP 4: LOG EVALUATION SUMMARY
        # ============================================================
        print(f"Evaluated {len(evaluated_rules)} rules "
              f"(filtered {len(candidates) - len(evaluated_rules)} low-coverage)")
        
        # ============================================================
        # STEP 5: DEDUPLICATE RULES
        # ============================================================
        before = len(evaluated_rules)
        evaluated_rules = self.deduplicator.deduplicate_rules(evaluated_rules)
        after = len(evaluated_rules)
        
        if after < before:
            print(f"Deduplicated rules: {before} → {after}")
        
        # ============================================================
        # STEP 6: COMPUTE OVERLAP DIAGNOSTICS
        # ============================================================
        self.overlap_analyzer.compute_overlap_diagnostics(
            evaluated_rules, segments_val
        )
        
        return evaluated_rules