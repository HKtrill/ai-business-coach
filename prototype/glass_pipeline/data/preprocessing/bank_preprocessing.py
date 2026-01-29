"""
Bank Marketing Dataset Preprocessing
====================================

Handles all preprocessing for the UCI Bank Marketing dataset including:
- Target encoding (yes/no → 1/0)
- Binary feature encoding with unknown handling
- Ordinal encoding for education, month, day_of_week
- Nominal encoding for job, marital
- Feature validation and type conversion

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class BankPreprocessor:
    """
    Comprehensive preprocessor for Bank Marketing dataset.
    
    Handles encoding, validation, and feature type conversion while
    preserving information in 'unknown' values where appropriate.
    """
    
    # Class-level mapping dictionaries
    BINARY_MAP = {'no': 0, 'yes': 1, 'unknown': -1}
    CONTACT_MAP = {'cellular': 0, 'telephone': 1}
    
    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    DAY_MAP = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
    
    POUTCOME_MAP = {
        'nonexistent': 0,  # Never contacted before
        'failure': 1,      # Previous campaign failed
        'success': 2       # Previous campaign succeeded
    }
    
    EDUCATION_MAP = {
        'illiterate': 0,
        'basic.4y': 1,
        'basic.6y': 2,
        'basic.9y': 3,
        'high.school': 4,
        'professional.course': 5,
        'university.degree': 6,
        'unknown': -1
    }
    
    MARITAL_MAP = {'divorced': 0, 'married': 1, 'single': 2, 'unknown': -1}
    
    # Feature groups for reference
    FEATURE_GROUPS = {
        'target': ['y'],
        'binary': ['default', 'housing', 'loan'],
        'ordinal': ['education', 'poutcome', 'contact', 'month', 'day_of_week'],
        'nominal': ['job', 'marital'],
        'numeric_campaign': ['age', 'duration', 'campaign', 'pdays', 'previous'],
        'numeric_economic': ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                            'euribor3m', 'nr.employed'],
    }
    
    # Known leaky features (should be dropped for production models)
    LEAKY_FEATURES = ['duration', 'pdays', 'poutcome']
    
    def __init__(self, drop_leaky: bool = True):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        drop_leaky : bool, default=True
            Whether to drop features that leak information from the future
            (duration, pdays, poutcome)
        """
        self.drop_leaky = drop_leaky
        self.job_map = None  # Will be created from training data
        
    def fit(self, df: pd.DataFrame) -> 'BankPreprocessor':
        """
        Fit preprocessor on training data.
        
        Creates job category mapping from unique values in training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw training dataframe
            
        Returns
        -------
        self : BankPreprocessor
            Fitted preprocessor
        """
        # Create job mapping from training data
        job_categories = sorted(df['job'].unique())
        self.job_map = {cat: i for i, cat in enumerate(job_categories)}
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe using fitted encodings.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe to transform
            
        Returns
        -------
        df_proc : pd.DataFrame
            Preprocessed dataframe
        """
        df_proc = df.copy()
        
        print("\n" + "="*80)
        print("🔧 PREPROCESSING: Bank Marketing Dataset")
        print("="*80)
        
        # 1. Target encoding
        df_proc['y'] = df_proc['y'].map({'yes': 1, 'no': 0})
        assert df_proc['y'].notna().all(), "❌ Unexpected values in target y"
        df_proc['y'] = df_proc['y'].astype('int8')
        print(f"✅ Target 'y': {df_proc['y'].value_counts().to_dict()}")
        
        # 2. Binary columns (with unknown=-1)
        binary_cols = ['default', 'housing', 'loan']
        for col in binary_cols:
            df_proc[col] = df_proc[col].map(self.BINARY_MAP)
            assert df_proc[col].notna().all(), f"❌ Unmapped values in {col}"
            df_proc[col] = df_proc[col].astype('int8')
        print(f"✅ Binary cols (unknown=-1): {binary_cols}")
        
        # 3. Contact
        df_proc['contact'] = df_proc['contact'].map(self.CONTACT_MAP)
        assert df_proc['contact'].notna().all(), "❌ Unmapped contact values"
        df_proc['contact'] = df_proc['contact'].astype('int8')
        print(f"✅ contact: {df_proc['contact'].value_counts().to_dict()}")
        
        # 4. Month (ordinal 1-12)
        df_proc['month'] = df_proc['month'].map(self.MONTH_MAP)
        assert df_proc['month'].notna().all(), "❌ Unmapped month values"
        df_proc['month'] = df_proc['month'].astype('int8')
        print(f"✅ month: {sorted(df_proc['month'].unique())}")
        
        # 5. Day of week (ordinal 0-4)
        df_proc['day_of_week'] = df_proc['day_of_week'].map(self.DAY_MAP)
        assert df_proc['day_of_week'].notna().all(), "❌ Unmapped day_of_week"
        df_proc['day_of_week'] = df_proc['day_of_week'].astype('int8')
        print(f"✅ day_of_week: {sorted(df_proc['day_of_week'].unique())}")
        
        # 6. Poutcome
        df_proc['poutcome'] = df_proc['poutcome'].map(self.POUTCOME_MAP)
        assert df_proc['poutcome'].notna().all(), "❌ Unmapped poutcome"
        df_proc['poutcome'] = df_proc['poutcome'].astype('int8')
        print(f"✅ poutcome: {df_proc['poutcome'].value_counts().sort_index().to_dict()}")
        
        # 7. Education (ordinal with unknown=-1)
        df_proc['education'] = df_proc['education'].map(self.EDUCATION_MAP)
        assert df_proc['education'].notna().all(), "❌ Unmapped education"
        df_proc['education'] = df_proc['education'].astype('int8')
        print(f"✅ education: ordinal 0-6, unknown=-1")
        
        # 8. Job (nominal, label encoded)
        if self.job_map is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        df_proc['job'] = df_proc['job'].map(self.job_map)
        # Handle unknown jobs in test set
        df_proc['job'] = df_proc['job'].fillna(-1).astype('int8')
        print(f"✅ job: {len(self.job_map)} categories label-encoded")
        
        # 9. Marital
        df_proc['marital'] = df_proc['marital'].map(self.MARITAL_MAP)
        assert df_proc['marital'].notna().all(), "❌ Unmapped marital"
        df_proc['marital'] = df_proc['marital'].astype('int8')
        print(f"✅ marital: {df_proc['marital'].value_counts().to_dict()}")
        
        # 10. Economic features (already numeric)
        economic_cols = self.FEATURE_GROUPS['numeric_economic']
        print(f"\n📊 Economic features (already numeric):")
        for col in economic_cols:
            df_proc[col] = df_proc[col].astype('float64')
            print(f"  {col}: [{df_proc[col].min():.3f}, {df_proc[col].max():.3f}]")
        
        # 11. Campaign features (already numeric)
        campaign_cols = self.FEATURE_GROUPS['numeric_campaign']
        print(f"\n📊 Campaign features (already numeric):")
        for col in campaign_cols:
            print(f"  {col}: [{df_proc[col].min()}, {df_proc[col].max()}]")
        
        # 12. Drop leaky features if requested
        if self.drop_leaky:
            leaky_present = [f for f in self.LEAKY_FEATURES if f in df_proc.columns]
            if leaky_present:
                df_proc = df_proc.drop(columns=leaky_present)
                print(f"⚠️  Dropped leaky features: {leaky_present}")
        
        # Final validation
        non_numeric = df_proc.select_dtypes(exclude=['number']).columns.tolist()
        assert len(non_numeric) == 0, f"❌ NON-NUMERIC: {non_numeric}"
        assert df_proc.isnull().sum().sum() == 0, "❌ Nulls detected"
        
        print(f"\n✅ df_proc ready: {df_proc.shape}")
        print(f"Memory: {df_proc.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Feature groups for reference
        print(f"\n📋 Feature Groups:")
        for group_name, features in self.FEATURE_GROUPS.items():
            print(f"  {group_name}: {features}")
        
        return df_proc
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe
            
        Returns
        -------
        df_proc : pd.DataFrame
            Preprocessed dataframe
        """
        return self.fit(df).transform(df)
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature groups dictionary."""
        return self.FEATURE_GROUPS.copy()


def inspect_categorical_columns(df: pd.DataFrame) -> None:
    """
    Inspect all categorical columns in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("📌 CATEGORICAL COLUMNS\n" + "-"*60)
    
    for col in categorical_cols:
        print(f"\n🔎 Column: {col}")
        print("-" * 40)
        values = (
            df[col]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
        )
        print(values)


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../raw/bank-additional-full.csv", sep=";")
    print(f"Loaded data: {df.shape}")
    
    # Inspect categorical columns
    inspect_categorical_columns(df)
    
    # Preprocess
    preprocessor = BankPreprocessor(drop_leaky=True)
    df_processed = preprocessor.fit_transform(df)
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Final shape: {df_processed.shape}")
    print(f"Columns: {list(df_processed.columns)}")