# ============================================================
# GLASS-BRW: RULE EVALUATOR MODULE (REFACTORED)
# ============================================================
# Main orchestrator for rule evaluation using modular components
#
# Input:  List[CandidateRule] from RuleGenerator
# Output: List[EvaluatedRule] with validation metrics
# ============================================================

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from .rule_matcher import RuleMatcher
from .evaluation_metrics import EvaluationMetrics
from .stability_estimator import StabilityEstimator
from .rf_integration import RFIntegration
from .rule_deduplicator import RuleDeduplicator
from .overlap_analyzer import OverlapAnalyzer
from glass_brw.rule_generator.feature_validator import FeatureValidator
from glass_brw.core.rule import CandidateRule, EvaluatedRule


class RuleEvaluator:
    """
    Validates and scores candidate rules using held-out validation data.
    
    Input:  List[CandidateRule] from RuleGenerator
    Output: List[EvaluatedRule] with validation metrics and RF diagnostics
    """
    
    def __init__(
        self,
        segment_builder,
        min_support: int = 30,
        n_bootstrap: int = 3,
        tier1_prefixes: Tuple[str, ...] = (
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
        
        # Initialize modular components
        self.validator = FeatureValidator(tier1_prefixes=tier1_prefixes)
        self.matcher = RuleMatcher(segment_builder=segment_builder)
        self.metrics = EvaluationMetrics()
        self.stability = StabilityEstimator(
            segment_builder=segment_builder,
            n_bootstrap=n_bootstrap
        )
        self.rf_integration = RFIntegration()
        self.deduplicator = RuleDeduplicator(validator=self.validator)
        self.overlap_analyzer = OverlapAnalyzer(matcher=self.matcher)
    
    # ============================================================
    # BACKWARD COMPATIBILITY METHODS
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
        candidates: List[CandidateRule],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        rf_model=None
    ) -> List[EvaluatedRule]:
        """
        Evaluate candidate rules on held-out validation data.
        
        Args:
            candidates: CandidateRule objects from RuleGenerator
            X_val: Validation features (raw)
            y_val: Validation labels (0 or 1)
            rf_model: RandomForestClassifier (optional) for confidence metrics
            
        Returns:
            List of EvaluatedRule objects with populated metrics
        """
        # Discretize validation data
        segments_val = self.segment_builder.assign_segments(X_val)
        N = len(X_val)
        
        # Get RF predictions if model provided
        rf_proba = self.rf_integration.get_rf_predictions(rf_model, X_val)
        
        print(f"\nEvaluating {len(candidates)} candidate rules...")
        
        # Evaluate each rule and convert to EvaluatedRule
        evaluated_rules: List[EvaluatedRule] = []
        
        for candidate in candidates:
            # Match rule to samples
            mask = self.matcher.match_rule(candidate, segments_val)
            n_matches = mask.sum()
            
            # Quality gate: minimum support
            if n_matches < self.min_support // 2:
                continue
            
            # Compute quality metrics
            rule_metrics = self.metrics.compute_all_metrics(
                mask, y_val, candidate.predicted_class, N
            )
            
            # Compute RF metrics (if model provided)
            if rf_proba is not None:
                rf_confidence, rf_alignment = self.rf_integration.compute_rf_metrics(
                    mask, rf_proba
                )
            else:
                rf_confidence = 0.5
                rf_alignment = 0.0
            
            # Convert CandidateRule → EvaluatedRule
            evaluated = EvaluatedRule(
                rule_id=candidate.rule_id,
                segment=candidate.segment,
                predicted_class=candidate.predicted_class,
                complexity=candidate.complexity,
                precision=rule_metrics['precision'],
                recall=rule_metrics['recall'],
                coverage=rule_metrics['coverage'],
                support=n_matches,
                rf_confidence=rf_confidence,
                rf_alignment=rf_alignment,
            )
            
            evaluated_rules.append(evaluated)
        
        # Log evaluation summary
        print(f"Evaluated {len(evaluated_rules)} rules "
              f"(filtered {len(candidates) - len(evaluated_rules)} low-coverage)")
        
        # Deduplicate rules
        before = len(evaluated_rules)
        evaluated_rules = self._deduplicate_evaluated_rules(evaluated_rules)
        after = len(evaluated_rules)
        
        if after < before:
            print(f"Deduplicated rules: {before} → {after}")
        
        # Compute overlap diagnostics
        self.overlap_analyzer.compute_overlap_diagnostics(
            evaluated_rules, segments_val
        )
        
        return evaluated_rules
    
    def _deduplicate_evaluated_rules(
        self, 
        rules: List[EvaluatedRule]
    ) -> List[EvaluatedRule]:
        """
        Deduplicate EvaluatedRule objects by segment.
        
        Keeps rule with highest precision for ties.
        """
        unique = {}
        
        for rule in rules:
            sig = (rule.segment_frozen, rule.predicted_class)
            
            if sig not in unique:
                unique[sig] = rule
            elif rule.precision > unique[sig].precision:
                unique[sig] = rule
        
        return list(unique.values())