# ============================================================
# GLASS-BRW: RULE METRICS MODULE
# ============================================================
# Computes precision, recall, coverage, and leakage metrics for rules
# Works with frozenset segments
# ============================================================

from typing import Dict, Tuple, Set, FrozenSet, Union
import pandas as pd
import numpy as np

SegmentType = Union[FrozenSet[Tuple[str, int]], Set[Tuple[str, int]]]


class RuleMetrics:
    """Compute quality metrics for rule segments."""
    
    def __init__(self, segments_df: pd.DataFrame, y: pd.Series):
        """
        Initialize metrics computer.
        
        Args:
            segments_df: DataFrame with segment features
            y: Target labels
        """
        self.segments_df = segments_df
        self.y = y
        self.N = len(segments_df)
        self.total_subscribers = (y == 1).sum()
        self.total_non_subscribers = (y == 0).sum()
    
    def compute_rule_mask(self, segment: SegmentType) -> pd.Series:
        """
        Compute boolean mask for samples matching rule segment.
        
        Args:
            segment: Set or frozenset of (feature, level) tuples
            
        Returns:
            Boolean mask indicating which samples match the rule
        """
        mask = pd.Series(True, index=self.segments_df.index)
        for feature, level in segment:
            if feature in self.segments_df.columns:
                mask &= (self.segments_df[feature] == level)
            else:
                # Feature not in data - return all False
                return pd.Series(False, index=self.segments_df.index)
        return mask
    
    def compute_support(self, mask: pd.Series) -> int:
        """Compute absolute support (number of matching samples)."""
        return int(mask.sum())
    
    def compute_coverage(self, mask: pd.Series) -> float:
        """Compute coverage as fraction of total population."""
        return mask.sum() / self.N
    
    def compute_precision(self, mask: pd.Series, predicted_class: int) -> float:
        """
        Compute class-specific precision.
        
        Args:
            mask: Boolean mask of samples matching rule
            predicted_class: Target class (0 or 1)
            
        Returns:
            Precision (TP / (TP + FP))
        """
        if mask.sum() == 0:
            return 0.0
        return float((self.y[mask] == predicted_class).mean())
    
    def compute_recall(self, mask: pd.Series, predicted_class: int) -> float:
        """
        Compute class-specific recall.
        
        Args:
            mask: Boolean mask of samples matching rule
            predicted_class: Target class (0 or 1)
            
        Returns:
            Recall (TP / (TP + FN))
        """
        total_class = (self.y == predicted_class).sum()
        if total_class == 0:
            return 0.0
        true_positives = ((self.y == predicted_class) & mask).sum()
        return float(true_positives / total_class)
    
    def compute_subscriber_leakage(self, mask: pd.Series) -> Tuple[float, int]:
        """
        Compute subscriber leakage for Pass 1 rules.
        
        Returns:
            (leakage_rate, subscribers_caught) tuple
        """
        if self.total_subscribers == 0:
            return 0.0, 0
        
        subscriber_mask = (self.y == 1)
        subscribers_caught = int((mask & subscriber_mask).sum())
        leakage_rate = subscribers_caught / self.total_subscribers
        
        return leakage_rate, subscribers_caught
    
    def compute_all_metrics(
        self, 
        segment: SegmentType, 
        predicted_class: int
    ) -> Dict[str, any]:
        """
        Compute all metrics for a rule segment.
        
        Args:
            segment: Set or frozenset of (feature, level) tuples
            predicted_class: Target class (0 or 1)
            
        Returns:
            Dictionary with all computed metrics
        """
        mask = self.compute_rule_mask(segment)
        support = self.compute_support(mask)
        coverage = self.compute_coverage(mask)
        precision = self.compute_precision(mask, predicted_class)
        recall = self.compute_recall(mask, predicted_class)
        leakage_rate, subscribers_caught = self.compute_subscriber_leakage(mask)
        
        return {
            'mask': mask,
            'support': support,
            'coverage': coverage,
            'precision': precision,
            'recall': recall,
            'leakage_rate': leakage_rate,
            'subscribers_caught': subscribers_caught,
        }