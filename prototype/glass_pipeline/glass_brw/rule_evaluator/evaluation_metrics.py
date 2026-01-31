# ============================================================
# GLASS-BRW: EVALUATION METRICS MODULE
# ============================================================
# Computes precision, recall, and coverage for rule evaluation
# ============================================================
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score


class EvaluationMetrics:
    """Compute quality metrics for rule evaluation."""
    
    def compute_precision(
        self,
        mask: pd.Series,
        y_val: pd.Series,
        predicted_class: int
    ) -> float:
        """
        Compute class-specific precision for rule.
        
        Definition:
            precision = correct_predictions / total_predictions
        
        Class-Specific:
            Pass 1 (y_r = 0): p = TN / |segment|  [fraction correct]
            Pass 2 (y_r = 1): p = TP / (TP + FP)  [standard precision]
        
        Args:
            mask: Boolean mask of matched samples
            y_val: True labels
            predicted_class: Rule's predicted class (0 or 1)
            
        Returns:
            Precision ∈ [0, 1]
        """
        n_matches = mask.sum()
        if n_matches == 0:
            return 0.0
        
        y_true = y_val[mask]
        
        if predicted_class == 1:  # SUBSCRIBE rules
            y_pred = np.full(n_matches, 1)
            return precision_score(y_true, y_pred, zero_division=0.0)
        else:  # NOT_SUBSCRIBE rules
            return (y_true == 0).sum() / len(y_true)
    
    def compute_recall(
        self,
        mask: pd.Series,
        y_val: pd.Series,
        predicted_class: int
    ) -> float:
        """
        Compute class-specific recall for rule.
        
        Definition:
            recall = true_positives / total_class_samples
        
        Class-Specific:
            Pass 1 (y_r = 0): r = TN / N_0  [fraction negatives captured]
            Pass 2 (y_r = 1): r = TP / N_1  [fraction positives captured]
        
        Args:
            mask: Boolean mask of matched samples
            y_val: True labels
            predicted_class: Rule's predicted class (0 or 1)
            
        Returns:
            Recall ∈ [0, 1]
        """
        if predicted_class == 1:  # SUBSCRIBE rules
            total_positives = (y_val == 1).sum()
            if total_positives == 0:
                return 0.0
            true_positives = ((y_val == 1) & mask).sum()
            return true_positives / total_positives
        else:  # NOT_SUBSCRIBE rules
            total_negatives = (y_val == 0).sum()
            if total_negatives == 0:
                return 0.0
            true_negatives = ((y_val == 0) & mask).sum()
            return true_negatives / total_negatives
    
    def compute_coverage(self, mask: pd.Series, N: int) -> float:
        """
        Compute fraction of samples matched by rule.
        
        Formula:
            coverage = |samples matched| / |total samples|
        
        Args:
            mask: Boolean mask of matched samples
            N: Total number of samples
            
        Returns:
            Coverage ∈ [0, 1]
        """
        return mask.sum() / N
    
    def compute_all_metrics(
        self,
        mask: pd.Series,
        y_val: pd.Series,
        predicted_class: int,
        N: int
    ) -> dict:
        """
        Compute all quality metrics at once.
        
        Args:
            mask: Boolean mask of matched samples
            y_val: True labels
            predicted_class: Rule's predicted class (0 or 1)
            N: Total number of samples
            
        Returns:
            Dict with precision, recall, coverage, support
        """
        return {
            'precision': self.compute_precision(mask, y_val, predicted_class),
            'recall': self.compute_recall(mask, y_val, predicted_class),
            'coverage': self.compute_coverage(mask, N),
            'support': mask.sum(),
        }