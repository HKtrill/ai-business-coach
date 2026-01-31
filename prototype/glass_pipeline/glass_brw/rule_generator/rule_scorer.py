# ============================================================
# GLASS-BRW: RULE SCORER MODULE
# ============================================================
# Scores rules for beam search and tracks feature diversity
# Works with frozenset segments
# ============================================================

from typing import Dict, List, Tuple, Set, FrozenSet, Union, Optional
import numpy as np

SegmentType = Union[FrozenSet[Tuple[str, int]], Set[Tuple[str, int]]]


class RuleScorer:
    """Score rules for beam search ranking and track feature diversity."""
    
    def __init__(
        self,
        diversity_penalty: float = 0.3,
        max_feature_reuse_pass1: Optional[int] = 3,
        max_feature_reuse_pass2: Optional[int] = 3,
    ):
        """
        Initialize rule scorer.
        
        Args:
            diversity_penalty: Penalty multiplier for feature reuse
            max_feature_reuse_pass1: Max times a feature can be reused (Pass 1)
            max_feature_reuse_pass2: Max times a feature can be reused (Pass 2)
        """
        self.diversity_penalty = diversity_penalty
        self.max_feature_reuse_pass1 = max_feature_reuse_pass1
        self.max_feature_reuse_pass2 = max_feature_reuse_pass2
        
        # Feature usage tracking (reset per generation)
        self.pass1_feature_usage: Dict[str, int] = {}
        self.pass2_feature_usage: Dict[str, int] = {}
    
    def reset_tracking(self):
        """Reset feature usage tracking for new generation."""
        self.pass1_feature_usage = {}
        self.pass2_feature_usage = {}
    
    def compute_base_score(
        self,
        precision: float,
        recall: float,
        coverage: float,
        predicted_class: int,
    ) -> float:
        """
        Compute base beam search score for rule ranking.
        
        Args:
            precision: Rule precision
            recall: Rule recall
            coverage: Rule coverage
            predicted_class: Target class (0 or 1)
            
        Returns:
            Base score (before diversity adjustment)
        """
        if predicted_class == 0:
            # Pass 1: Prioritize precision and coverage
            return (precision ** 3) * np.sqrt(coverage)
        else:
            # Pass 2: Prioritize recall, then precision and coverage
            return (recall ** 3) * precision * coverage
    
    def get_diversity_penalty(
        self,
        segment: SegmentType,
        predicted_class: int,
        validator,
    ) -> float:
        """
        Compute diversity penalty based on feature reuse.
        
        Args:
            segment: Set or frozenset of (feature, level) tuples
            predicted_class: Target class (0 or 1)
            validator: FeatureValidator instance
            
        Returns:
            Diversity multiplier (0.1 to 1.0)
        """
        if predicted_class == 0:
            feature_usage = self.pass1_feature_usage
        else:
            feature_usage = self.pass2_feature_usage
        
        penalty = 0.0
        for feature, _ in segment:
            base = validator.extract_base_feature(feature)
            usage_count = feature_usage.get(base, 0)
            if usage_count > 0:
                penalty += self.diversity_penalty * (usage_count ** 1.2)
        
        return max(0.1, 1.0 - penalty)
    
    def score_rule(
        self,
        precision: float,
        recall: float,
        coverage: float,
        predicted_class: int,
        segment: Optional[SegmentType] = None,
        validator=None,
    ) -> float:
        """
        Compute final score with diversity adjustment.
        
        Args:
            precision: Rule precision
            recall: Rule recall
            coverage: Rule coverage
            predicted_class: Target class (0 or 1)
            segment: Optional segment for diversity penalty
            validator: Optional FeatureValidator for diversity penalty
            
        Returns:
            Final score
        """
        base_score = self.compute_base_score(precision, recall, coverage, predicted_class)
        
        if segment is not None and validator is not None:
            diversity_mult = self.get_diversity_penalty(segment, predicted_class, validator)
            return base_score * diversity_mult
        
        return base_score
    
    def update_feature_usage(
        self,
        segment: SegmentType,
        predicted_class: int,
        validator,
    ):
        """
        Track feature usage for diversity enforcement.
        
        Args:
            segment: Set or frozenset of (feature, level) tuples
            predicted_class: Target class (0 or 1)
            validator: FeatureValidator instance
        """
        if predicted_class == 0:
            feature_usage = self.pass1_feature_usage
        else:
            feature_usage = self.pass2_feature_usage
        
        for feature, _ in segment:
            base = validator.extract_base_feature(feature)
            feature_usage[base] = feature_usage.get(base, 0) + 1
    
    def get_usage_summary(self) -> Dict[str, any]:
        """
        Get summary of feature usage across both passes.
        
        Returns:
            Dict with usage statistics
        """
        return {
            'pass1_features_used': len(self.pass1_feature_usage),
            'pass1_top_features': sorted(
                self.pass1_feature_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'pass2_features_used': len(self.pass2_feature_usage),
            'pass2_top_features': sorted(
                self.pass2_feature_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }