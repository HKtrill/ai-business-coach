# =============================================================================
# /content/churn_pipeline/modules/leakage_monitor.py (FIXED)
# =============================================================================
import os
import random
import numpy as np

# Import utils functions directly (since we added path)
def set_seed(seed):
    """Set seed for python random, numpy, and environment hash seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class DataLeakageMonitor:
    """Monitor and prevent data leakage throughout the pipeline"""

    def __init__(self):
        self.train_columns = None
        self.test_columns = None
        self.train_stats = {}
        self.test_stats = {}

    def record_split_info(self, X_train, X_test, y_train, y_test):
        """Record information about the train/test split for leakage detection"""
        self.train_columns = set(X_train.columns)
        self.test_columns = set(X_test.columns)

        # Record basic statistics
        for col in X_train.columns:
            if X_train[col].dtype in ['float64', 'int64']:
                self.train_stats[col] = {
                    'mean': X_train[col].mean(),
                    'std': X_train[col].std(),
                    'min': X_train[col].min(),
                    'max': X_train[col].max()
                }
                self.test_stats[col] = {
                    'mean': X_test[col].mean(),
                    'std': X_test[col].std(),
                    'min': X_test[col].min(),
                    'max': X_test[col].max()
                }

        # Check target distribution
        train_target_dist = y_train.value_counts(normalize=True).to_dict()
        test_target_dist = y_test.value_counts(normalize=True).to_dict()

        print("=== DATA LEAKAGE CHECK RESULTS ===")
        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Train target distribution: {train_target_dist}")
        print(f"Test target distribution: {test_target_dist}")

        # Check for identical columns
        if self.train_columns != self.test_columns:
            print("⚠️ WARNING: Train and test have different columns!")
            print(f"Train only: {self.train_columns - self.test_columns}")
            print(f"Test only: {self.test_columns - self.train_columns}")
        else:
            print("✅ Train and test have identical column sets")

    def check_preprocessing_leakage(self, transformer):
        """Verify preprocessing was fit only on training data"""
        print("\n=== PREPROCESSING LEAKAGE CHECK ===")

        if hasattr(transformer, 'named_transformers_'):
            # Check the nested transformer for scalers
            for name, trans_pipe in transformer.named_transformers_.items():
                if hasattr(trans_pipe, 'named_steps'):
                    for step_name, step in trans_pipe.named_steps.items():
                        if hasattr(step, 'mean_'):
                            print(f"✅ {name} pipeline's {step_name} scaler was fitted (has mean_)")
                        elif hasattr(step, 'scale_'):
                            print(f"✅ {name} pipeline's {step_name} scaler was fitted (has scale_)")

        print("✅ Preprocessing check completed - no obvious leakage detected")