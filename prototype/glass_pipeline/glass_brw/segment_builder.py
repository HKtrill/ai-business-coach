"""
Binary Segment Builder for GLASS-BRW
=====================================

EBM-aligned segment builder with strict binary contract enforcement.

Ensures all segment features are {0,1} for rule learning.
Catches bitwise NOT artifacts and other invalid values.

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import pandas as pd
import warnings

from typing import Dict

warnings.filterwarnings('ignore')


class BankSegmentBuilder:
    """
    EBM-aligned binary segment builder for GLASS-BRW.
    
    Enforces STRICT binary contract {0,1} on all segment features.
    Any upstream bitwise artifacts (-1, -2, etc.) are caught explicitly.
    """
    from glass_brw.rf.binning import RF_FEATURES_BINARY
    SEGMENT_FEATURES = RF_FEATURES_BINARY
    
    def assign_segments(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and validate binary segment features.
        
        Guarantees:
        - Output ∈ {0,1}
        - dtype = int8
        - Raises explicit errors on contract violation
        
        Parameters
        ----------
        X : pd.DataFrame
            Engineered features (from BRW feature engineering)
            
        Returns
        -------
        segments : pd.DataFrame
            Validated binary segment features
            
        Raises
        ------
        ValueError
            If any feature contains non-binary values or negative values
        """
        # Drop labels if present
        X = X.drop(columns=["y", "y_bin"], errors="ignore")
        
        # Keep only declared segment features that exist
        cols = [c for c in self.SEGMENT_FEATURES if c in X.columns]
        
        if not cols:
            raise ValueError("No valid segment features found in input DataFrame")
        
        segments = X[cols].copy()
        
        # ============================================================
        # STRICT BINARY ENFORCEMENT
        # ============================================================
        for col in segments.columns:
            raw_vals = set(segments[col].dropna().unique())
            
            # Explicitly catch bitwise NOT artifacts
            if any(v < 0 for v in raw_vals):
                raise ValueError(
                    f"Invalid negative values in segment feature '{col}': {raw_vals}\n"
                    f"Likely cause: bitwise (~) used on int instead of boolean logic."
                )
            
            if not raw_vals.issubset({0, 1}):
                raise ValueError(
                    f"Non-binary segment feature '{col}': {raw_vals}"
                )
        
        return segments.astype("int8")
    
    def validate_segments(self, X: pd.DataFrame) -> bool:
        """
        Validate segment features without raising errors.
        
        Parameters
        ----------
        X : pd.DataFrame
            Segment features to validate
            
        Returns
        -------
        is_valid : bool
            True if all features are binary, False otherwise
        """
        try:
            self.assign_segments(X)
            return True
        except ValueError:
            return False
    
    def get_feature_coverage(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get coverage statistics for each segment feature.
        
        Parameters
        ----------
        X : pd.DataFrame
            Segment features
            
        Returns
        -------
        coverage : pd.DataFrame
            Coverage statistics (count, percentage) for each feature
        """
        segments = self.assign_segments(X)
        
        coverage = pd.DataFrame({
            'Feature': segments.columns,
            'Count': segments.sum(),
            'Percentage': (segments.sum() / len(segments) * 100).round(2)
        }).sort_values('Count', ascending=False)
        
        return coverage
    
    def get_segment_summary(self, X: pd.DataFrame) -> Dict:
        """
        Get comprehensive summary of segment features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Segment features
            
        Returns
        -------
        summary : dict
            Summary statistics
        """
        segments = self.assign_segments(X)
        coverage = self.get_feature_coverage(X)
        
        # Identify low/high coverage features
        low_coverage = coverage[coverage['Percentage'] < 1.0]
        high_coverage = coverage[coverage['Percentage'] > 99.0]
        
        return {
            'n_features': len(segments.columns),
            'n_samples': len(segments),
            'coverage': coverage,
            'low_coverage_features': low_coverage.to_dict('records'),
            'high_coverage_features': high_coverage.to_dict('records'),
            'features': list(segments.columns)
        }


# Convenience function
def prepare_segments(
    X: pd.DataFrame,
    validate: bool = True,
    print_summary: bool = True
) -> pd.DataFrame:
    """
    Prepare binary segments from engineered features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Engineered features
    validate : bool
        Whether to validate segments
    print_summary : bool
        Whether to print summary
        
    Returns
    -------
    segments : pd.DataFrame
        Binary segment features
    """
    builder = BankSegmentBuilder()
    segments = builder.assign_segments(X)
    
    if print_summary:
        print("=" * 80)
        print("📋 SEGMENT FEATURES PREPARED")
        print("=" * 80)
        print(f"   Features: {segments.shape[1]}")
        print(f"   Samples: {segments.shape[0]}")
        
        coverage = builder.get_feature_coverage(X)
        
        low_cov = coverage[coverage['Percentage'] < 1.0]
        if len(low_cov) > 0:
            print(f"\n⚠️  Low coverage features ({len(low_cov)}):")
            for _, row in low_cov.iterrows():
                print(f"   {row['Feature']}: {row['Percentage']:.2f}%")
        
        high_cov = coverage[coverage['Percentage'] > 99.0]
        if len(high_cov) > 0:
            print(f"\n⚠️  High coverage features ({len(high_cov)}):")
            for _, row in high_cov.iterrows():
                print(f"   {row['Feature']}: {row['Percentage']:.2f}%")
        
        print("\n✅ Segment features validated")
    
    return segments


