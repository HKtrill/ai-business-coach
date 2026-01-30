"""
RF Lift Analysis for Optimal Binning
=====================================

Uses Random Forest to identify:
1. Feature importance (Gini + Permutation)
2. Natural breakpoints for continuous features via lift analysis
3. Categorical feature lift patterns

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class RFLiftAnalyzer:
    """
    Random Forest lift analyzer for binning guidance.
    
    Analyzes feature importance and lift patterns to guide
    GLASS-BRW feature engineering decisions.
    """
    
    CONTINUOUS_FEATURES = [
        'age', 'campaign', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
        'euribor3m', 'nr.employed'
    ]
    
    CATEGORICAL_FEATURES = [
        'job', 'marital', 'education', 'default',
        'housing', 'loan', 'contact', 'month', 'day_of_week'
    ]
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_leaf: int = 50,
        random_state: int = 42
    ):
        """Initialize RF lift analyzer."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.rf = None
        self.overall_rate = None
        self.gini_importance = None
        self.perm_importance = None
        self.continuous_lift = {}
        self.categorical_lift = {}
        self.fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> 'RFLiftAnalyzer':
        """
        Fit RF and compute lift analysis.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        X_test : pd.DataFrame
            Test features  
        y_test : pd.Series
            Test labels
            
        Returns
        -------
        self : RFLiftAnalyzer
            Fitted analyzer
        """
        print("=" * 80)
        print("🌲 RF LIFT ANALYSIS")
        print("=" * 80)
        
        # Train RF
        print("\n🌲 Training Random Forest...")
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf.fit(X_train, y_train)
        
        # Quick performance check
        y_pred = self.rf.predict(X_test)
        y_proba = self.rf.predict_proba(X_test)[:, 1]
        print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"   ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
        
        # Store overall conversion rate
        self.overall_rate = y_train.mean()
        
        # Feature importance
        self._compute_feature_importance(X_train, X_test, y_test)
        
        # Lift analysis
        self._compute_lift_analysis(X_train, y_train)
        
        self.fitted = True
        return self
    
    def _compute_feature_importance(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """Compute Gini and permutation importance."""
        print("\n" + "="*60)
        print("📊 FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Gini importance
        self.gini_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Gini_Importance': self.rf.feature_importances_
        }).sort_values('Gini_Importance', ascending=False)
        
        print("\n🔝 Gini Importance (Top 15):")
        print(self.gini_importance.head(15).to_string(index=False))
        
        # Permutation importance
        print("\n🔀 Computing Permutation Importance...")
        perm_imp = permutation_importance(
            self.rf, X_test, y_test,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1,
            scoring='roc_auc'
        )
        
        self.perm_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Perm_Importance': perm_imp.importances_mean,
            'Perm_Std': perm_imp.importances_std
        }).sort_values('Perm_Importance', ascending=False)
        
        print("\n🔝 Permutation Importance (Top 15):")
        print(self.perm_importance.head(15).to_string(index=False))
        
        # Combined ranking
        combined = self.gini_importance.merge(self.perm_importance, on='Feature')
        combined['Avg_Rank'] = (
            combined['Gini_Importance'].rank(ascending=False) +
            combined['Perm_Importance'].rank(ascending=False)
        ) / 2
        self.combined_importance = combined.sort_values('Avg_Rank')
        
        print("\n🏆 Combined Ranking (Gini + Permutation):")
        print(self.combined_importance[['Feature', 'Gini_Importance', 'Perm_Importance', 'Avg_Rank']].head(15).to_string(index=False))
    
    def _compute_lift_analysis(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):
        """Compute lift by feature decile."""
        print("\n" + "="*60)
        print("📈 LIFT ANALYSIS")
        print("="*60)
        
        # Continuous features
        continuous_present = [f for f in self.CONTINUOUS_FEATURES if f in X_train.columns]
        
        print("\n📊 Continuous Feature Lift:")
        for feat in continuous_present:
            lift_df = self._calculate_lift_by_decile(X_train, y_train, feat)
            self.continuous_lift[feat] = lift_df
            
            print(f"\n🔹 {feat}:")
            print(f"   Range: [{X_train[feat].min():.2f}, {X_train[feat].max():.2f}]")
            print(f"   Best bin lift: {lift_df['lift'].max():.2f}x")
            print(f"   Worst bin lift: {lift_df['lift'].min():.2f}x")
        
        # Categorical features
        categorical_present = [f for f in self.CATEGORICAL_FEATURES if f in X_train.columns]
        
        print("\n📊 Categorical Feature Lift:")
        for feat in categorical_present:
            lift_df = self._calculate_categorical_lift(X_train, y_train, feat)
            self.categorical_lift[feat] = lift_df
            
            print(f"\n🔹 {feat}:")
            print(lift_df.sort_values('lift', ascending=False).to_string(index=False))
    
    def _calculate_lift_by_decile(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature: str,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """Calculate conversion rate and lift by decile."""
        df_temp = pd.DataFrame({feature: X[feature], 'y': y})
        
        # Handle features with few unique values
        n_unique = df_temp[feature].nunique()
        if n_unique < n_bins:
            df_temp['bin'] = df_temp[feature]
        else:
            df_temp['bin'] = pd.qcut(df_temp[feature], q=n_bins, duplicates='drop')
        
        lift_df = df_temp.groupby('bin').agg(
            count=('y', 'count'),
            conversions=('y', 'sum'),
            conv_rate=('y', 'mean')
        ).reset_index()
        
        lift_df['lift'] = lift_df['conv_rate'] / self.overall_rate
        lift_df['feature'] = feature
        
        return lift_df
    
    def _calculate_categorical_lift(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature: str
    ) -> pd.DataFrame:
        """Calculate lift for categorical feature."""
        df_temp = pd.DataFrame({feature: X[feature], 'y': y})
        
        lift_df = df_temp.groupby(feature).agg(
            count=('y', 'count'),
            conversions=('y', 'sum'),
            conv_rate=('y', 'mean')
        ).reset_index()
        
        lift_df['lift'] = lift_df['conv_rate'] / self.overall_rate
        lift_df['pct_of_total'] = lift_df['count'] / len(df_temp) * 100
        
        return lift_df
    
    def plot_importance(self, figsize: Tuple[int, int] = (16, 8)):
        """Plot Gini and Permutation importance."""
        if not self.fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Gini
        ax1 = axes[0]
        top_gini = self.gini_importance.head(15)
        ax1.barh(top_gini['Feature'], top_gini['Gini_Importance'], color='forestgreen')
        ax1.set_xlabel('Gini Importance')
        ax1.set_title('Top 15 Features (Gini Importance)', fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Permutation
        ax2 = axes[1]
        top_perm = self.perm_importance.head(15)
        ax2.barh(top_perm['Feature'], top_perm['Perm_Importance'],
                 xerr=top_perm['Perm_Std'], color='steelblue')
        ax2.set_xlabel('Permutation Importance (ROC-AUC drop)')
        ax2.set_title('Top 15 Features (Permutation Importance)', fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_lift_curves(self, figsize: Tuple[int, int] = (18, 15)):
        """Plot lift curves for continuous features."""
        if not self.fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
        
        n_features = len(self.continuous_lift)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (feat, lift_df) in enumerate(self.continuous_lift.items()):
            ax = axes[idx]
            
            x_labels = [str(b)[:20] for b in lift_df['bin']]
            bars = ax.bar(range(len(lift_df)), lift_df['conv_rate'], color='steelblue', alpha=0.7)
            
            # Color bars by lift
            for i, (bar, lift) in enumerate(zip(bars, lift_df['lift'])):
                if lift > 1.5:
                    bar.set_color('darkgreen')
                elif lift < 0.7:
                    bar.set_color('darkred')
            
            # Overall rate line
            ax.axhline(y=self.overall_rate, color='red', linestyle='--', linewidth=2,
                      label=f'Overall ({self.overall_rate:.3f})')
            
            ax.set_xticks(range(len(lift_df)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Conversion Rate')
            ax.set_title(f'{feat}\n(Max Lift: {lift_df["lift"].max():.2f}x)', fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.continuous_lift), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Conversion Rate by Feature Decile (Green=High Lift, Red=Low Lift)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of lift analysis."""
        if not self.fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
        
        return {
            'model': self.rf,
            'overall_conversion_rate': self.overall_rate,
            'gini_importance': self.gini_importance,
            'perm_importance': self.perm_importance,
            'combined_importance': self.combined_importance,
            'continuous_lift': self.continuous_lift,
            'categorical_lift': self.categorical_lift
        }


def analyze_rf_lift(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    global_split: Dict,
    plot: bool = True
) -> Tuple[RFLiftAnalyzer, Dict]:
    """
    Convenience function to run RF lift analysis.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    global_split : dict
        Global split dict (for indices)
    plot : bool
        Whether to generate plots
        
    Returns
    -------
    analyzer : RFLiftAnalyzer
        Fitted analyzer
    analysis : dict
        Analysis summary
    """
    analyzer = RFLiftAnalyzer()
    analyzer.fit(X_train, y_train, X_test, y_test)
    
    if plot:
        analyzer.plot_importance()
        analyzer.plot_lift_curves()
    
    analysis = analyzer.get_analysis_summary()
    
    print("\n" + "=" * 80)
    print("🎯 Use these insights to define GLASS-BRW binning")
    print("=" * 80)
    
    return analyzer, analysis
