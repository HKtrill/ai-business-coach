"""
Global Train/Test Split Manager
================================

Creates and manages the canonical train/test split for the Bank Marketing
dataset, ensuring consistency across all stages of the pipeline.

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, Tuple
import pickle
import os


class GlobalSplitManager:
    """
    Manages the global train/test split for the pipeline.
    
    Ensures that all stages use the same split to prevent data leakage
    and enable fair comparison across models.
    """
    
    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        n_cv_folds: int = 10
    ):
        """
        Initialize split manager.
        
        Parameters
        ----------
        test_size : float, default=0.20
            Proportion of dataset to include in test split
        random_state : int, default=42
            Random state for reproducibility
        n_cv_folds : int, default=10
            Number of cross-validation folds for hyperparameter tuning
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_cv_folds = n_cv_folds
        self.split_dict = None
        
    def create_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'y'
    ) -> Dict:
        """
        Create stratified train/test split.
        
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe
        target_col : str, default='y'
            Name of target column
            
        Returns
        -------
        split_dict : dict
            Dictionary containing:
            - X_train, X_test: feature matrices
            - y_train, y_test: target vectors
            - train_idx, test_idx: index arrays
            - cv_10: StratifiedKFold object
            - random_state: random state used
        """
        print("\n" + "="*80)
        print("📐 CREATING GLOBAL TRAIN / TEST SPLIT")
        print("="*80)
        
        # Separate features and target
        X_all = df.drop(columns=[target_col])
        y_all = df[target_col]
        
        print(f"\nTotal samples: {len(df)}")
        print(f"Positive rate: {y_all.mean():.4f}")
        print(f"Class counts: {y_all.value_counts().to_dict()}")
        print(f"Features: {X_all.shape[1]}")
        
        # Create stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=self.test_size,
            stratify=y_all,
            random_state=self.random_state
        )
        
        # Get indices
        train_idx = X_train.index.values
        test_idx = X_test.index.values
        
        print(f"\n✅ Split created:")
        print(f"  Train: {X_train.shape} | positives: {y_train.sum()} ({y_train.mean():.4f})")
        print(f"  Test:  {X_test.shape} | positives: {y_test.sum()} ({y_test.mean():.4f})")
        
        # Create CV object for hyperparameter tuning
        cv = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Sanity check: no overlap
        assert len(set(train_idx) & set(test_idx)) == 0, "❌ Train/Test overlap!"
        print("🔒 No train/test leakage")
        
        # Create canonical split dictionary
        self.split_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'cv': cv,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'n_cv_folds': self.n_cv_folds,
            'feature_names': list(X_train.columns),
            'n_features': X_train.shape[1]
        }
        
        print("\n📦 GLOBAL_SPLIT ready")
        return self.split_dict
    
    def save_split(self, filepath: str) -> None:
        """
        Save split to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save split dictionary (will save as .pkl)
        """
        if self.split_dict is None:
            raise ValueError("No split created. Call create_split() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.split_dict, f)
        
        print(f"💾 Split saved to: {filepath}")
    
    @staticmethod
    def load_split(filepath: str) -> Dict:
        """
        Load split from disk.
        
        Parameters
        ----------
        filepath : str
            Path to saved split dictionary
            
        Returns
        -------
        split_dict : dict
            Loaded split dictionary
        """
        with open(filepath, 'rb') as f:
            split_dict = pickle.load(f)
        
        print(f"📂 Split loaded from: {filepath}")
        print(f"   Train: {split_dict['X_train'].shape}")
        print(f"   Test:  {split_dict['X_test'].shape}")
        print(f"   Features: {split_dict['n_features']}")
        
        return split_dict
    
    def get_split_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of the split.
        
        Returns
        -------
        summary : pd.DataFrame
            Summary table with split statistics
        """
        if self.split_dict is None:
            raise ValueError("No split created. Call create_split() first.")
        
        s = self.split_dict
        
        summary = pd.DataFrame({
            'Split': ['Train', 'Test', 'Total'],
            'Samples': [
                len(s['y_train']),
                len(s['y_test']),
                len(s['y_train']) + len(s['y_test'])
            ],
            'Positives': [
                s['y_train'].sum(),
                s['y_test'].sum(),
                s['y_train'].sum() + s['y_test'].sum()
            ],
            'Positive_Rate': [
                s['y_train'].mean(),
                s['y_test'].mean(),
                (s['y_train'].sum() + s['y_test'].sum()) / 
                (len(s['y_train']) + len(s['y_test']))
            ]
        })
        
        return summary


def create_and_save_global_split(
    df_processed: pd.DataFrame,
    output_dir: str = "./global_splits",
    split_name: str = "bank_marketing_split.pkl"
) -> Dict:
    """
    Convenience function to create and save global split.
    
    Parameters
    ----------
    df_processed : pd.DataFrame
        Preprocessed dataframe with target column 'y'
    output_dir : str
        Directory to save split
    split_name : str
        Filename for split
        
    Returns
    -------
    split_dict : dict
        Created split dictionary
    """
    manager = GlobalSplitManager()
    split_dict = manager.create_split(df_processed)
    
    filepath = os.path.join(output_dir, split_name)
    manager.save_split(filepath)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SPLIT SUMMARY")
    print("="*80)
    summary = manager.get_split_summary()
    print(summary.to_string(index=False))
    print("="*80)
    
    return split_dict


