"""
Correlation Analysis & Multicollinearity Reduction
===================================================

Identifies and removes highly correlated features to reduce multicollinearity
while preserving features most correlated with the target.

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings("ignore")


class CorrelationAnalyzer:
    """
    Analyzes feature correlations and removes redundant features.
    
    Strategy: For pairs of features with correlation above threshold,
    keeps the feature with higher absolute correlation with target.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        threshold : float, default=0.7
            Correlation threshold above which features are considered redundant
        """
        self.threshold = threshold
        self.features_to_drop = []
        self.correlation_pairs = []
        self.fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> 'CorrelationAnalyzer':
        """
        Identify redundant features based on training data.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
            
        Returns
        -------
        self : CorrelationAnalyzer
            Fitted analyzer
        """
        print("\n" + "="*80)
        print("🔍 CORRELATION ANALYSIS & MULTICOLLINEARITY REDUCTION")
        print("="*80)
        
        # Compute correlation matrix (with target)
        X_with_y = X_train.copy()
        X_with_y['y'] = y_train.values
        corr_matrix = X_with_y.corr()
        
        # Correlation with target
        target_corr = corr_matrix['y'].drop('y').abs().sort_values(ascending=False)
        
        print(f"\n📈 Top 10 features by |correlation| with target:")
        for feat, corr in target_corr.head(10).items():
            print(f"  {feat}: {corr:.4f}")
        
        # Feature-feature correlations (exclude target)
        feature_corr = corr_matrix.drop('y', axis=0).drop('y', axis=1)
        
        # Find highly correlated pairs
        self.correlation_pairs = []
        
        for i in range(len(feature_corr.columns)):
            for j in range(i + 1, len(feature_corr.columns)):
                feat1 = feature_corr.columns[i]
                feat2 = feature_corr.columns[j]
                corr_val = feature_corr.iloc[i, j]
                
                if abs(corr_val) > self.threshold:
                    # Get correlation with target for each feature
                    corr_with_y_1 = target_corr[feat1]
                    corr_with_y_2 = target_corr[feat2]
                    
                    # Feature to drop = lower correlation with target
                    if corr_with_y_1 >= corr_with_y_2:
                        keep = feat1
                        drop = feat2
                    else:
                        keep = feat2
                        drop = feat1
                    
                    self.correlation_pairs.append({
                        'feature_1': feat1,
                        'feature_2': feat2,
                        'correlation': corr_val,
                        'corr_with_y_1': corr_with_y_1,
                        'corr_with_y_2': corr_with_y_2,
                        'keep': keep,
                        'drop': drop
                    })
        
        # Display results
        if self.correlation_pairs:
            pairs_df = pd.DataFrame(self.correlation_pairs)
            print(f"\n⚠️  Found {len(pairs_df)} highly correlated pairs (|r| > {self.threshold}):")
            print(pairs_df[['feature_1', 'feature_2', 'correlation', 
                           'corr_with_y_1', 'corr_with_y_2', 'keep', 'drop']].to_string(index=False))
            
            # Get unique features to drop
            self.features_to_drop = list(set(pairs_df['drop'].tolist()))
            
            print(f"\n🗑️  Features to drop (lower |corr| with target):")
            for feat in self.features_to_drop:
                corr_y = target_corr[feat]
                print(f"    • {feat} (|corr with y| = {corr_y:.4f})")
        else:
            print(f"\n✅ No feature pairs with |r| > {self.threshold}")
            self.features_to_drop = []
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove redundant features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to transform
            
        Returns
        -------
        X_pruned : pd.DataFrame
            Features with redundant columns dropped
        """
        if not self.fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
        
        if not self.features_to_drop:
            return X.copy()
        
        X_pruned = X.drop(columns=self.features_to_drop)
        
        print(f"\n🔧 Dropping {len(self.features_to_drop)} redundant features")
        print(f"   Before: {X.shape[1]} features")
        print(f"   After:  {X_pruned.shape[1]} features")
        
        return X_pruned
    
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
            
        Returns
        -------
        X_pruned : pd.DataFrame
            Pruned training features
        """
        return self.fit(X_train, y_train).transform(X_train)
    
    def plot_correlation_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = "Feature Correlation Matrix",
        figsize: Tuple[int, int] = (16, 14)
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Create correlation matrix with target
        X_with_y = X.copy()
        X_with_y['y'] = y.values
        corr_matrix = X_with_y.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            annot_kws={"size": 8}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def get_summary(self) -> Dict:
        """
        Get summary of correlation analysis.
        
        Returns
        -------
        summary : dict
            Summary statistics
        """
        if not self.fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
        
        return {
            'threshold': self.threshold,
            'n_high_corr_pairs': len(self.correlation_pairs),
            'n_features_dropped': len(self.features_to_drop),
            'features_dropped': self.features_to_drop,
            'correlation_pairs': self.correlation_pairs
        }


def analyze_and_prune_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = 0.7,
    plot: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, CorrelationAnalyzer]:
    """
    Convenience function to analyze correlations and prune redundant features.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training target
    threshold : float, default=0.7
        Correlation threshold
    plot : bool, default=True
        Whether to generate plots
        
    Returns
    -------
    X_train_pruned : pd.DataFrame
        Pruned training features
    X_test_pruned : pd.DataFrame
        Pruned test features
    analyzer : CorrelationAnalyzer
        Fitted analyzer object
    """
    analyzer = CorrelationAnalyzer(threshold=threshold)
    
    # Plot original correlations
    if plot:
        print("\n📊 Generating correlation heatmap (original features)...")
        fig = analyzer.plot_correlation_matrix(
            X_train, y_train,
            title="Correlation Matrix (Original Features)"
        )
        plt.show()
    
    # Fit and transform
    X_train_pruned = analyzer.fit_transform(X_train, y_train)
    X_test_pruned = analyzer.transform(X_test)
    
    # Verify no remaining high correlations
    if analyzer.features_to_drop:
        print("\n🔍 Verifying no remaining high correlations...")
        feature_corr_pruned = X_train_pruned.corr()
        remaining_high = []
        
        for i in range(len(feature_corr_pruned.columns)):
            for j in range(i + 1, len(feature_corr_pruned.columns)):
                corr_val = feature_corr_pruned.iloc[i, j]
                if abs(corr_val) > threshold:
                    remaining_high.append((
                        feature_corr_pruned.columns[i],
                        feature_corr_pruned.columns[j],
                        corr_val
                    ))
        
        if remaining_high:
            print(f"\n⚠️  Still have {len(remaining_high)} pairs above threshold:")
            for f1, f2, r in remaining_high:
                print(f"    {f1} ↔ {f2}: {r:.3f}")
        else:
            print(f"\n✅ All feature pairs now below |r| = {threshold}")
    
    # Plot pruned correlations
    if plot and analyzer.features_to_drop:
        print("\n📊 Generating correlation heatmap (pruned features)...")
        fig = analyzer.plot_correlation_matrix(
            X_train_pruned, y_train,
            title=f"Correlation Matrix (Pruned: {X_train_pruned.shape[1]} features)"
        )
        plt.show()
    
    # Summary
    print("\n" + "="*80)
    print("📋 MULTICOLLINEARITY REDUCTION SUMMARY")
    print("="*80)
    summary = analyzer.get_summary()
    print(f"  Correlation threshold: |r| > {summary['threshold']}")
    print(f"  High-corr pairs found: {summary['n_high_corr_pairs']}")
    print(f"  Features dropped: {summary['n_features_dropped']}")
    print(f"  Original feature count: {X_train.shape[1]}")
    print(f"  Final feature count: {X_train_pruned.shape[1]}")
    print("="*80)
    
    return X_train_pruned, X_test_pruned, analyzer
