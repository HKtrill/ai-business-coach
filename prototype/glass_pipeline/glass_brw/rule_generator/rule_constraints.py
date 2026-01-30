# ============================================================
# GLASS-BRW: RULE CONSTRAINTS MODULE
# ============================================================
# Depth-staged constraint checking:
#   Depth 1: Structural validity ONLY
#   Depth 2: Light pruning (extreme leakage guardrail)
#   Depth 3: Quality constraints
# ============================================================
from collections import defaultdict


class RuleConstraints:
    """Check depth-staged constraints for rule validation."""
    
    def __init__(
        self,
        # Pass 1 constraints (Depth 3)
        min_support_pass1: int = 270,
        max_coverage_not_subscribe: float = 1.00,
        min_precision_not_subscribe: float = 0.25,
        max_precision_not_subscribe: float = 1.00,
        max_subscriber_leakage_rate: float = 0.99,
        max_subscriber_leakage_absolute: int = 999,
        # Pass 2 constraints (Depth 3)
        min_support_pass2: int = 100,
        max_coverage_subscribe: float = 0.99,
        min_precision_subscribe: float = 0.25,
        max_precision_subscribe: float = 1.00,
        min_recall_subscribe: float = 0.15,
        max_recall_subscribe: float = 0.99,
        # Depth 2 light pruning
        max_leakage_rate_depth2: float = 0.75,
        max_leakage_fraction_depth2: float = 0.65,
    ):
        # Pass 1 (NOT_SUBSCRIBE) thresholds
        self.min_support_pass1 = min_support_pass1
        self.max_coverage_not_subscribe = max_coverage_not_subscribe
        self.min_precision_not_subscribe = min_precision_not_subscribe
        self.max_precision_not_subscribe = max_precision_not_subscribe
        self.max_subscriber_leakage_rate = max_subscriber_leakage_rate
        self.max_subscriber_leakage_absolute = max_subscriber_leakage_absolute
        
        # Pass 2 (SUBSCRIBE) thresholds
        self.min_support_pass2 = min_support_pass2
        self.max_coverage_subscribe = max_coverage_subscribe
        self.min_precision_subscribe = min_precision_subscribe
        self.max_precision_subscribe = max_precision_subscribe
        self.min_recall_subscribe = min_recall_subscribe
        self.max_recall_subscribe = max_recall_subscribe
        
        # Depth 2 light pruning
        self.max_leakage_rate_depth2 = max_leakage_rate_depth2
        self.max_leakage_fraction_depth2 = max_leakage_fraction_depth2
        self.max_leakage_absolute_depth2 = None  # Set dynamically
    
    def set_total_subscribers(self, total_subscribers: int):
        """Set total subscriber count for computing absolute leakage threshold."""
        self.max_leakage_absolute_depth2 = int(
            total_subscribers * self.max_leakage_fraction_depth2
        )
    
    def check_depth1_constraints(self, segment: set, validator) -> tuple:
        """
        Check Depth 1 constraints (structural validity only).
        
        Args:
            segment: Set of (feature, level) tuples
            validator: FeatureValidator instance
            
        Returns:
            (is_valid, rejection_reason) tuple
        """
        # Only check structural validity
        if validator.has_duplicate_base_features(segment):
            return False, "duplicate_base_features"
        
        return True, None
    
    def check_depth2_constraints(
        self,
        segment: set,
        depth: int,
        predicted_class: int,
        metrics: dict,
        validator,
    ) -> tuple:
        """
        Check Depth 2 constraints (light structural pruning).
        
        Args:
            segment: Set of (feature, level) tuples
            depth: Current depth
            predicted_class: Target class (0 or 1)
            metrics: Dict with computed metrics
            validator: FeatureValidator instance
            
        Returns:
            (is_valid, rejection_reason) tuple
        """
        # Structural validity (all depths)
        if validator.has_duplicate_base_features(segment):
            return False, "duplicate_base_features"
        
        # Tier1 requirement for Pass 2 at depth >= 2
        if predicted_class == 1 and depth >= 2:
            if not validator.has_tier1_feature(segment):
                return False, "missing_tier1"
        
        # Depth 2 specific: Light pruning
        if depth == 2:
            # Pass 1: Extreme leakage guardrail (BLOCKS EXPANSION)
            if predicted_class == 0:
                leakage_rate = metrics['leakage_rate']
                subscribers_caught = metrics['subscribers_caught']
                
                if (leakage_rate > self.max_leakage_rate_depth2 or
                    subscribers_caught > self.max_leakage_absolute_depth2):
                    return False, f"extreme_leakage_depth2"
            
            # Pass 2: Precision floor + Coverage cap (BLOCKS EXPANSION)
            if predicted_class == 1:
                min_prec_depth2 = 0.15
                max_cov_depth2 = 0.50
                
                if metrics['precision'] < min_prec_depth2:
                    return False, "precision_floor_depth2"
                
                if metrics['coverage'] > max_cov_depth2:
                    return False, "coverage_cap_depth2"
        
        return True, None
    
    def check_depth3_constraints(
        self,
        segment: set,
        depth: int,
        predicted_class: int,
        metrics: dict,
        validator,
    ) -> tuple:
        """
        Check Depth 3+ constraints (quality gates).
        
        Args:
            segment: Set of (feature, level) tuples
            depth: Current depth
            predicted_class: Target class (0 or 1)
            metrics: Dict with computed metrics
            validator: FeatureValidator instance
            
        Returns:
            (is_valid, rejection_reason) tuple
        """
        # Structural validity
        if validator.has_duplicate_base_features(segment):
            return False, "duplicate_base_features"
        
        # Tier1 requirement for Pass 2
        if predicted_class == 1 and depth >= 2:
            if not validator.has_tier1_feature(segment):
                return False, "missing_tier1"
        
        # Minimum support
        min_support = (self.min_support_pass1 if predicted_class == 0
                      else self.min_support_pass2)
        if metrics['support'] < min_support:
            return False, "insufficient_support"
        
        # Coverage caps
        if predicted_class == 0:
            if metrics['coverage'] > self.max_coverage_not_subscribe:
                return False, "coverage_cap"
        else:
            if metrics['coverage'] > self.max_coverage_subscribe:
                return False, "coverage_cap"
        
        # Pass-specific quality gates
        if predicted_class == 0:  # Pass 1: Precision + leakage
            if (metrics['precision'] < self.min_precision_not_subscribe or
                metrics['precision'] > self.max_precision_not_subscribe):
                return False, "precision_range"
            
            if metrics['leakage_rate'] > self.max_subscriber_leakage_rate:
                return False, "leakage_rate"
            
            if metrics['subscribers_caught'] > self.max_subscriber_leakage_absolute:
                return False, "leakage_absolute"
        
        else:  # Pass 2: Precision + recall
            if (metrics['precision'] < self.min_precision_subscribe or
                metrics['precision'] > self.max_precision_subscribe):
                return False, "precision_range"
            
            if metrics['recall'] < self.min_recall_subscribe:
                return False, "recall_min"
            
            if metrics['recall'] > self.max_recall_subscribe:
                return False, "recall_cap"
        
        return True, None
    
    def get_threshold_info(self, constraint_name: str, predicted_class: int) -> str:
        """Get human-readable threshold info for a constraint."""
        thresholds = {
            'insufficient_support': (
                f"min_support={'pass1=' + str(self.min_support_pass1) if predicted_class == 0 else 'pass2=' + str(self.min_support_pass2)}"
            ),
            'coverage_cap': (
                f"max_coverage={self.max_coverage_not_subscribe if predicted_class == 0 else self.max_coverage_subscribe}"
            ),
            'precision_range': (
                f"precision_range=[{self.min_precision_not_subscribe if predicted_class == 0 else self.min_precision_subscribe}, "
                f"{self.max_precision_not_subscribe if predicted_class == 0 else self.max_precision_subscribe}]"
            ),
            'leakage_rate': f"max_leakage_rate={self.max_subscriber_leakage_rate}",
            'leakage_absolute': f"max_leakage_absolute={self.max_subscriber_leakage_absolute}",
            'recall_min': f"min_recall={self.min_recall_subscribe}",
            'recall_cap': f"max_recall={self.max_recall_subscribe}",
        }
        return thresholds.get(constraint_name, "")