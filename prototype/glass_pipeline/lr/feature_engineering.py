"""
Feature Engineering for Logistic Regression Stage
==================================================

Comprehensive feature engineering focusing on economic indicators:
- PCA on economic features
- Economic health index (composite indicator)
- Polynomial and interaction terms
- Cyclical encoding for temporal features
- Campaign feature transformations

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class EconomicFeatureEngineer:
    """
    Feature engineer specialized in economic indicators and interactions.
    
    Creates interpretable and predictive features from:
    - Employment variation rate
    - Consumer price index
    - Consumer confidence index
    - Euribor 3-month rate
    - Number of employees
    """
    
    ECONOMIC_COLS = [
        'emp.var.rate',
        'cons.price.idx',
        'cons.conf.idx',
        'euribor3m',
        'nr.employed'
    ]
    
    def __init__(self, n_pca_components: int = 3):
        """
        Initialize feature engineer.
        
        Parameters
        ----------
        n_pca_components : int, default=3
            Number of PCA components to extract from economic indicators
        """
        self.n_pca_components = n_pca_components
        self.econ_scaler = None
        self.pca = None
        self.euribor_median = None
        self.fitted = False
        
    def fit(self, X_train: pd.DataFrame) -> 'EconomicFeatureEngineer':
        """
        Fit transformers on training data.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
            
        Returns
        -------
        self : EconomicFeatureEngineer
            Fitted engineer
        """
        print("\n" + "="*80)
        print("🔧 FITTING ECONOMIC FEATURE ENGINEER")
        print("="*80)
        
        # 1. Fit scaler on economic features
        self.econ_scaler = StandardScaler()
        X_econ = X_train[self.ECONOMIC_COLS].values
        self.econ_scaler.fit(X_econ)
        
        # 2. Fit PCA
        X_econ_scaled = self.econ_scaler.transform(X_econ)
        self.pca = PCA(n_components=self.n_pca_components)
        self.pca.fit(X_econ_scaled)
        
        print(f"\n📐 PCA on Economic Indicators:")
        print(f"  Components: {self.n_pca_components}")
        print(f"  Explained variance: {self.pca.explained_variance_ratio_}")
        print(f"  Total variance captured: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Show loadings
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_pca_components)],
            index=self.ECONOMIC_COLS
        )
        print(f"\n  Component loadings:")
        print(loadings.round(3).to_string())
        
        # 3. Store euribor median for regime detection
        self.euribor_median = X_train['euribor3m'].median()
        print(f"\n📊 Euribor median: {self.euribor_median:.4f}")
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted engineer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to transform
            
        Returns
        -------
        X_transformed : pd.DataFrame
            Features with engineered columns added
        """
        if not self.fitted:
            raise ValueError("Engineer not fitted. Call fit() first.")
        
        X_eng = X.copy()
        
        print("\n" + "="*80)
        print("🔧 ECONOMIC FEATURE ENGINEERING")
        print("="*80)
        
        # ==============================
        # 1. PCA FEATURES
        # ==============================
        X_econ = X[self.ECONOMIC_COLS].values
        X_econ_scaled = self.econ_scaler.transform(X_econ)
        X_pca = self.pca.transform(X_econ_scaled)
        
        for i in range(self.n_pca_components):
            X_eng[f'econ_pc{i+1}'] = X_pca[:, i]
        
        print(f"✅ Added: {self.n_pca_components} PCA components")
        
        # ==============================
        # 2. ECONOMIC HEALTH INDEX
        # Composite indicator: higher = better economic conditions
        # ==============================
        econ_z = pd.DataFrame(
            X_econ_scaled,
            columns=[f'{c}_z' for c in self.ECONOMIC_COLS],
            index=X.index
        )
        
        X_eng['econ_health'] = (
            econ_z['emp.var.rate_z'] * 0.25 +      # Employment growth
            -econ_z['euribor3m_z'] * 0.30 +        # Lower rates better (inverted)
            econ_z['cons.conf.idx_z'] * 0.25 +     # Consumer confidence
            econ_z['nr.employed_z'] * 0.20         # Employment level
        )
        print(f"✅ Added: econ_health (weighted composite)")
        print(f"  Range: [{X_eng['econ_health'].min():.3f}, {X_eng['econ_health'].max():.3f}]")
        
        # ==============================
        # 3. RATE ENVIRONMENT (binary regime indicator)
        # ==============================
        X_eng['low_rate_env'] = (X['euribor3m'] < self.euribor_median).astype(int)
        print(f"✅ Added: low_rate_env (euribor < {self.euribor_median:.2f})")
        
        # ==============================
        # 4. POLYNOMIAL FEATURES
        # ==============================
        X_eng['euribor3m_sq'] = X['euribor3m'] ** 2
        print(f"✅ Added: euribor3m_sq (quadratic term)")
        
        # ==============================
        # 5. INTERACTION TERMS
        # ==============================
        X_eng['conf_x_rate'] = X['cons.conf.idx'] * X['euribor3m']
        print(f"✅ Added: conf_x_rate (interaction)")
        
        # ==============================
        # 6. TEMPORAL FEATURES (cyclical encoding)
        # ==============================
        if 'month' in X.columns:
            X_eng['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
            X_eng['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
            print(f"✅ Added: month_sin, month_cos")
        
        if 'day_of_week' in X.columns:
            X_eng['dow_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 5)
            X_eng['dow_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 5)
            print(f"✅ Added: dow_sin, dow_cos")
        
        # ==============================
        # 7. CAMPAIGN FEATURES
        # ==============================
        if 'campaign' in X.columns:
            X_eng['log_campaign'] = np.log1p(X['campaign'])
            print(f"✅ Added: log_campaign")
        
        if 'previous' in X.columns:
            X_eng['is_cold'] = (X['previous'] == 0).astype(int)
            print(f"✅ Added: is_cold (never contacted)")
        
        # ==============================
        # 8. DROP ABSORBED FEATURES
        # ==============================
        drop_cols = []
        if 'month' in X.columns:
            drop_cols.append('month')
        if 'day_of_week' in X.columns:
            drop_cols.append('day_of_week')
        if 'campaign' in X.columns:
            drop_cols.append('campaign')
        
        if drop_cols:
            X_eng = X_eng.drop(columns=drop_cols)
            print(f"\n🗑️ Dropped (absorbed into engineered features): {drop_cols}")
        
        print(f"\n✅ Feature engineering complete: {X_eng.shape[1]} features")
        return X_eng
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
            
        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed training features
        """
        return self.fit(X_train).transform(X_train)
    
    def get_engineered_feature_names(self) -> list:
        """Get list of all engineered feature names."""
        engineered = [
            f'econ_pc{i+1}' for i in range(self.n_pca_components)
        ] + [
            'econ_health',
            'low_rate_env',
            'euribor3m_sq',
            'conf_x_rate',
            'month_sin',
            'month_cos',
            'dow_sin',
            'dow_cos',
            'log_campaign',
            'is_cold'
        ]
        return engineered


def engineer_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_pca_components: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, EconomicFeatureEngineer]:
    """
    Convenience function to engineer features on train and test sets.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    n_pca_components : int, default=3
        Number of PCA components
        
    Returns
    -------
    X_train_eng : pd.DataFrame
        Engineered training features
    X_test_eng : pd.DataFrame
        Engineered test features
    engineer : EconomicFeatureEngineer
        Fitted engineer object
    """
    engineer = EconomicFeatureEngineer(n_pca_components=n_pca_components)
    
    # Fit and transform train
    X_train_eng = engineer.fit_transform(X_train)
    
    # Transform test
    X_test_eng = engineer.transform(X_test)
    
    # Summary
    print("\n" + "="*80)
    print("📋 FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Engineered features: {X_train_eng.shape[1]}")
    print(f"  Features added: {X_train_eng.shape[1] - X_train.shape[1]}")
    print(f"\n  Engineered feature names:")
    for feat in engineer.get_engineered_feature_names():
        if feat in X_train_eng.columns:
            print(f"    • {feat}")
    print("="*80)
    
    return X_train_eng, X_test_eng, engineer
