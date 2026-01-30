# ============================================================
# GLASS-BRW: RULE MATCHER MODULE
# ============================================================
# Matches rules to samples and creates boolean masks
# ============================================================
import pandas as pd
import numpy as np


class RuleMatcher:
    """Match rules to samples and create boolean masks."""
    
    def __init__(self, segment_builder):
        """
        Initialize rule matcher.
        
        Args:
            segment_builder: SegmentBuilder instance for discretization
        """
        self.segment_builder = segment_builder
    
    def match_rule(self, rule, segments_df: pd.DataFrame) -> pd.Series:
        """
        Create boolean mask for samples matching rule segment.
        
        Logic: mask[i] = True iff sample i satisfies ALL conditions
        
        Args:
            rule: Rule object with .segment attribute
            segments_df: Discretized validation features
            
        Returns:
            Boolean mask (True = sample matches rule)
            
        Complexity: O(k × n) where k = |segment|, n = |segments_df|
        
        Example:
            >>> mask = matcher.match_rule(rule, segments_df)
            >>> mask.sum()  # Count matching samples
            127
        """
        mask = pd.Series(True, index=segments_df.index)
        
        for feature, level in rule.segment:
            if feature in segments_df.columns:
                mask &= (segments_df[feature] == level)
            else:
                # Feature missing → no matches possible
                return pd.Series(False, index=segments_df.index)
        
        return mask
    
    def get_covered_indices(self, rule, segments_df: pd.DataFrame) -> set:
        """
        Get set of sample indices covered by rule.
        
        Args:
            rule: Rule object
            segments_df: Discretized features
            
        Returns:
            Set of integer indices
        """
        mask = self.match_rule(rule, segments_df)
        return set(np.flatnonzero(mask.values))