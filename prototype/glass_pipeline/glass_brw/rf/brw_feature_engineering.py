"""
GLASS-BRW Feature Engineering
==============================

Lift-based binning for binary rule generation.

Transforms continuous and categorical features into binary features
based on RF lift analysis insights.

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


def engineer_features_brw(df_proc: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer GLASS-BRW binary features from preprocessed data.
    
    Based on RF lift analysis, creates binary features with:
    - Lift-guided binning thresholds
    - Composite indicators
    - All binary (0/1) for rule learning
    
    Parameters
    ----------
    df_proc : pd.DataFrame
        Preprocessed dataframe (from BankPreprocessor)
        
    Returns
    -------
    df_eng : pd.DataFrame
        Engineered binary features with target 'y'
    """
    print("=" * 80)
    print("🔧 GLASS-BRW FEATURE ENGINEERING (Lift-Based)")
    print("=" * 80)
    
    df = df_proc.copy()
    
    # =========================================================
    # PREVIOUS (contact history)
    # =========================================================
    df["previous_zero"] = (df["previous"] == 0).astype("int8")
    df["previous_low"]  = ((df["previous"] >= 1) & (df["previous"] <= 2)).astype("int8")
    df["previous_mid"]  = ((df["previous"] >= 3) & (df["previous"] <= 4)).astype("int8")
    df["previous_high"] = (df["previous"] >= 5).astype("int8")
    
    # =========================================================
    # ECONOMIC FEATURES
    # =========================================================
    df["nr_employed_low"]  = (df["nr.employed"] <= 5076).astype("int8")
    df["nr_employed_mid"]  = ((df["nr.employed"] > 5076) & (df["nr.employed"] <= 5100)).astype("int8")
    df["nr_employed_high"] = (df["nr.employed"] > 5100).astype("int8")
    
    df["euribor_low"]  = (df["euribor3m"] <= 1.044).astype("int8")
    df["euribor_mid"]  = ((df["euribor3m"] > 1.044) & (df["euribor3m"] <= 1.5)).astype("int8")
    df["euribor_high"] = (df["euribor3m"] > 1.5).astype("int8")
    
    df["emp_var_very_neg"] = (df["emp.var.rate"] <= -1.7).astype("int8")
    df["emp_var_neg"]      = ((df["emp.var.rate"] > -1.7) & (df["emp.var.rate"] <= 0)).astype("int8")
    df["emp_var_pos"]      = (df["emp.var.rate"] > 0).astype("int8")
    
    # =========================================================
    # CPI (Consumer Price Index)
    # =========================================================
    df["cpi_low"]        = (df["cons.price.idx"] <= 93.2).astype("int8")
    df["cpi_sweet_spot"] = ((df["cons.price.idx"] > 93.4) & (df["cons.price.idx"] <= 93.8)).astype("int8")
    df["cpi_high"]       = (df["cons.price.idx"] > 94.4).astype("int8")
    df["cpi_mid"] = (
        (df["cpi_low"] == 0) &
        (df["cpi_sweet_spot"] == 0) &
        (df["cpi_high"] == 0)
    ).astype("int8")
    
    # =========================================================
    # CCI (Consumer Confidence Index)
    # =========================================================
    df["cci_very_low"]   = (df["cons.conf.idx"] <= -46).astype("int8")
    df["cci_sweet_spot"] = ((df["cons.conf.idx"] > -42) & (df["cons.conf.idx"] <= -36)).astype("int8")
    df["cci_high"]       = (df["cons.conf.idx"] > -36).astype("int8")
    df["cci_mid"] = (
        (df["cci_very_low"] == 0) &
        (df["cci_sweet_spot"] == 0) &
        (df["cci_high"] == 0)
    ).astype("int8")
    
    # =========================================================
    # MONTH
    # =========================================================
    df["month_hot"]     = df["month"].isin([3, 9, 10, 12]).astype("int8")
    df["month_warm"]    = df["month"].isin([4]).astype("int8")
    df["month_neutral"] = df["month"].isin([6, 8, 11]).astype("int8")
    df["month_cold"]    = df["month"].isin([5, 7]).astype("int8")
    
    # =========================================================
    # CONTACT
    # =========================================================
    df["contact_cellular"]  = (df["contact"] == 0).astype("int8")
    df["contact_telephone"] = (df["contact"] == 1).astype("int8")
    
    # =========================================================
    # AGE
    # =========================================================
    df["age_young"]  = (df["age"] <= 28).astype("int8")
    df["age_prime"]  = ((df["age"] > 28) & (df["age"] <= 38)).astype("int8")
    df["age_mid"]    = ((df["age"] > 38) & (df["age"] <= 55)).astype("int8")
    df["age_senior"] = (df["age"] > 55).astype("int8")
    
    # =========================================================
    # CAMPAIGN
    # =========================================================
    df["campaign_first"]    = (df["campaign"] <= 2).astype("int8")
    df["campaign_moderate"] = ((df["campaign"] > 2) & (df["campaign"] <= 5)).astype("int8")
    df["campaign_heavy"]    = (df["campaign"] > 5).astype("int8")
    
    # =========================================================
    # JOB
    # =========================================================
    df["job_high_lift"]  = df["job"].isin([8, 5]).astype("int8")
    df["job_above_avg"]  = df["job"].isin([0, 10, 11]).astype("int8")
    df["job_neutral"]    = df["job"].isin([9, 6, 3, 4, 2]).astype("int8")
    df["job_low_lift"]   = df["job"].isin([1, 7]).astype("int8")
    
    # =========================================================
    # MARITAL
    # =========================================================
    df["marital_single"]   = (df["marital"] == 2).astype("int8")
    df["marital_married"]  = (df["marital"] == 1).astype("int8")
    df["marital_divorced"] = (df["marital"] == 0).astype("int8")
    df["marital_unknown"]  = (df["marital"] == -1).astype("int8")
    
    # =========================================================
    # EDUCATION
    # =========================================================
    df["education_high"] = df["education"].isin([6, -1]).astype("int8")
    df["education_mid"]  = df["education"].isin([4, 5]).astype("int8")
    df["education_low"]  = df["education"].isin([0, 1, 2, 3]).astype("int8")
    
    # =========================================================
    # DAY OF WEEK
    # =========================================================
    df["dow_midweek"] = df["day_of_week"].isin([1, 2, 3]).astype("int8")
    df["dow_edges"]   = df["day_of_week"].isin([0, 4]).astype("int8")
    
    # =========================================================
    # BINARY (default, housing, loan)
    # =========================================================
    df["default_no"]      = (df["default"] == 0).astype("int8")
    df["default_unknown"] = (df["default"] == -1).astype("int8")
    df["housing_yes"]     = (df["housing"] == 1).astype("int8")
    df["housing_no"]      = (df["housing"] == 0).astype("int8")
    df["loan_yes"]        = (df["loan"] == 1).astype("int8")
    df["loan_no"]         = (df["loan"] == 0).astype("int8")
    
    # =========================================================
    # COMPOSITE INDICATORS
    # =========================================================
    df["econ_favorable"] = (
        (df["euribor_low"] == 1) |
        (df["nr_employed_low"] == 1) |
        (df["emp_var_very_neg"] == 1)
    ).astype("int8")
    
    df["econ_unfavorable"] = (
        (df["euribor_high"] == 1) &
        (df["nr_employed_high"] == 1) &
        (df["emp_var_pos"] == 1)
    ).astype("int8")
    
    df["prospect_hot"] = (
        (df["previous_high"] == 1) |
        (df["previous_mid"] == 1)
    ).astype("int8")
    
    df["prospect_warm"] = (
        (df["previous_low"] == 1) &
        (df["job_high_lift"] == 1)
    ).astype("int8")
    
    df["prospect_cold"] = (
        (df["previous_zero"] == 1) &
        (df["job_low_lift"] == 1)
    ).astype("int8")
    
    # =========================================================
    # OUTPUT: Only engineered features + target
    # =========================================================
    feature_cols = [c for c in df.columns if c not in df_proc.columns or c == "y"]
    df_eng = df[feature_cols].copy()
    df_eng["y"] = df["y"].astype("int8")
    
    print(f"\n✅ Engineered {df_eng.shape[1] - 1} binary features")
    print(f"   Features: {list(df_eng.drop(columns=['y']).columns)}")
    
    return df_eng


def prepare_brw_data(
    df_proc: pd.DataFrame,
    global_split: Dict
) -> Dict:
    """
    Engineer BRW features and split into train/test aligned with global split.
    
    Parameters
    ----------
    df_proc : pd.DataFrame
        Preprocessed dataframe
    global_split : dict
        Global split dictionary with train_idx, test_idx
        
    Returns
    -------
    brw_data : dict
        Dictionary containing engineered train/test splits
    """
    print("\n" + "="*60)
    print("📐 PREPARING BRW DATA")
    print("="*60)
    
    # Engineer features
    df_eng = engineer_features_brw(df_proc)
    
    # Split using global indices
    train_idx = global_split['train_idx']
    test_idx = global_split['test_idx']
    
    df_eng_train = df_eng.loc[train_idx]
    df_eng_test = df_eng.loc[test_idx]
    
    X_eng_train = df_eng_train.drop(columns=['y'])
    y_eng_train = df_eng_train['y']
    X_eng_test = df_eng_test.drop(columns=['y'])
    y_eng_test = df_eng_test['y']
    
    print(f"\n   Train: {X_eng_train.shape}, positives: {y_eng_train.sum()} ({y_eng_train.mean():.4f})")
    print(f"   Test:  {X_eng_test.shape}, positives: {y_eng_test.sum()} ({y_eng_test.mean():.4f})")
    
    # Validate feature coverage
    print("\n✅ VALIDATION: Feature Coverage")
    low_coverage = []
    high_coverage = []
    for col in X_eng_train.columns:
        coverage = X_eng_train[col].sum() / len(X_eng_train) * 100
        if coverage < 1:
            low_coverage.append((col, coverage))
        elif coverage > 99:
            high_coverage.append((col, coverage))
    
    if low_coverage:
        print(f"   ⚠️  Low coverage features:")
        for feat, cov in low_coverage:
            print(f"      {feat}: {cov:.2f}%")
    
    if high_coverage:
        print(f"   ⚠️  High coverage features:")
        for feat, cov in high_coverage:
            print(f"      {feat}: {cov:.2f}%")
    
    brw_data = {
        'df_eng': df_eng,
        'df_eng_train': df_eng_train,
        'df_eng_test': df_eng_test,
        'X_eng_train': X_eng_train,
        'y_eng_train': y_eng_train,
        'X_eng_test': X_eng_test,
        'y_eng_test': y_eng_test,
        'feature_names': list(X_eng_train.columns),
    }
    
    print(f"\n📦 BRW_DATA created with {len(brw_data['feature_names'])} features")
    print("✅ Ready for GLASS-BRW rule generation")
    
    return brw_data