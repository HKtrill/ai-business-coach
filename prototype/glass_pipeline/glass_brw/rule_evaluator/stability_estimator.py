# ============================================================
# GLASS-BRW: STABILITY ESTIMATOR MODULE
# ============================================================
# Bootstrap-based stability estimation for rules
# ============================================================
import numpy as np
import pandas as pd


class StabilityEstimator:
    """Estimate rule precision stability via bootstrap resampling."""
    
    def __init__(self, segment_builder, n_bootstrap: int = 3):
        """
        Initialize stability estimator.
        
        Args:
            segment_builder: SegmentBuilder instance for discretization
            n_bootstrap: Number of bootstrap samples (default: 3)
        """
        self.segment_builder = segment_builder
        self.n_bootstrap = n_bootstrap
    
    def estimate_stability(
        self,
        rule,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        matcher,
        metrics_computer,
    ) -> float:
        """
        Estimate rule precision stability via bootstrap resampling.
        
        Method:
            1. Draw n_bootstrap samples with replacement
            2. Compute precision on each sample
            3. Measure variance across samples
            4. Stability = 1 - (std / mean) ∈ [0, 1]
        
        Interpretation:
            1.0 = perfectly stable (zero variance)
            0.0 = highly unstable (variance == mean)
        
        Args:
            rule: Rule to evaluate
            X_val: Validation features (raw)
            y_val: Validation labels
            matcher: RuleMatcher instance
            metrics_computer: EvaluationMetrics instance
            
        Returns:
            Stability score ∈ [0, 1]
        """
        precisions = []
        n = len(X_val)
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            X_sample = X_val.iloc[idx]
            y_sample = y_val.iloc[idx] if hasattr(y_val, 'iloc') else y_val[idx]
            
            # Discretize and match
            segments = self.segment_builder.assign_segments(X_sample)
            mask = matcher.match_rule(rule, segments)
            
            # Skip if insufficient coverage
            if mask.sum() < 10:
                continue
            
            # Compute precision
            precision = metrics_computer.compute_precision(
                mask, y_sample, rule.predicted_class
            )
            precisions.append(precision)
        
        # Compute stability from variance
        if len(precisions) < 2:
            return 0.0
        
        mean_prec = np.mean(precisions)
        std_prec = np.std(precisions)
        
        if mean_prec > 0:
            stability = 1.0 - (std_prec / mean_prec)
            return max(0.0, stability)
        else:
            return 0.0