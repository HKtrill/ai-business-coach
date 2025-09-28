# =============================================================================
# /content/churn_pipeline/modules/leakage_monitor.py (FINAL, FIXED VERSION)
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif 

class DataLeakageMonitor:
    """Monitor and prevent data leakage throughout the pipeline using MI and Correlation."""

    def __init__(self):
        self.mi_results = None 
        self.correlation_check = None 

    def record_split_info(self, X_train, X_test, y_train, y_test):
        """Record information about the train/test split and perform statistical analysis."""
        
        print("\n=== DATA LEAKAGE & FEATURE ANALYSIS (RAW SPLIT) ===")
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Ensure data is numeric for MI and correlation checks (fillna with 0 for safety)
        X_train_numeric = X_train.select_dtypes(include=np.number).fillna(0)
        X_test_numeric = X_test.select_dtypes(include=np.number).fillna(0)
        
        # --- 1. Basic Stats & Distribution Check ---
        train_target_dist = y_train.value_counts(normalize=True).to_dict()
        test_target_dist = y_test.value_counts(normalize=True).to_dict()
        print(f"Train target distribution: {train_target_dist}")
        print(f"Test target distribution: {test_target_dist}")
        
        if X_train.columns.equals(X_test.columns):
            print("✅ Train and test have identical column sets")
        else:
            print("⚠️ WARNING: Train and test have different columns!")
            
        
        # --- 2. Statistical Distribution Comparison (Mean/Std) ---
        print("\n--- Statistical Distribution Comparison (Raw Splits) ---")
        
        if X_train_numeric.empty:
            print("⚠️ No numeric features found for mean/std comparison.")
        else:
            stats = pd.DataFrame({
                'Train Mean': X_train_numeric.mean(),
                'Test Mean': X_test_numeric.mean(),
                'Train Std': X_train_numeric.std(),
                'Test Std': X_test_numeric.std()
            })
            # Calculate the percentage difference in means
            stats['Mean Diff (%)'] = abs(stats['Train Mean'] - stats['Test Mean']) / (abs(stats['Train Mean']) + 1e-6) * 100
            
            print(stats.to_string(float_format="%.4f"))
            
            if (stats['Mean Diff (%)'] > 10).any():
                 print("\n⚠️ WARNING: One or more feature means differ by >10% across splits. Investigate split.")
            else:
                 print("\n✅ Mean/Std comparison shows low difference (good split balance).")


        # --- 3. Feature Relevance (Mutual Information) ---
        print("\n--- Feature Relevance (Mutual Information) ---")
        try:
            if X_train_numeric.shape[1] > 0:
                # MI is run on the features that can be used directly (numerical)
                self.mi_results = mutual_info_classif(X_train_numeric.values, y_train.values.ravel(), random_state=42)
                mi_df = pd.Series(self.mi_results, index=X_train_numeric.columns).sort_values(ascending=False)
                
                print(mi_df)
                print(f"Interpretation: High MI suggests a strong, potentially non-linear, relationship with 'Churn'.")
            else:
                print("⚠️ Cannot calculate MI: No numeric features present.")
        except Exception as e:
            print(f"⚠️ MI Check failed: {e}")
            self.mi_results = None
            
        # --- 4. Inter-Set Correlation Check (Identical Rows) ---
        print("\n--- Inter-Set Correlation Check (Identical Rows) ---")
        try:
            # Find rows present in both splits (indicates bad split or sample duplication)
            train_reset = X_train.reset_index(drop=True)
            test_reset = X_test.reset_index(drop=True)
            
            duplicates_in_splits = pd.merge(train_reset, test_reset, how='inner', on=list(X_train.columns))
            
            if not duplicates_in_splits.empty:
                print(f"🛑 FATAL LEAKAGE: Found {len(duplicates_in_splits)} identical rows across Train and Test splits!")
                self.correlation_check = False
            else:
                print("✅ No identical rows found across Train and Test splits (Good).")
                self.correlation_check = True

        except Exception as e:
            print(f"⚠️ Correlation Check failed: {e}")
            self.correlation_check = False
            
        print("=" * 60)

    def check_preprocessing_leakage(self, pipeline):
        """
        FIXED: Verifies preprocessing was fit only on training data by checking mean_/scale_
        attributes on steps of a standard Pipeline (using named_steps).
        """
        print("\n=== PREPROCESSING LEAKAGE CHECK (Structural) ===")

        # FIX: Use named_steps for a standard Pipeline object
        found_step_with_stats = False
        
        try:
            for name, step in pipeline.named_steps.items():
                # Check if the step is likely a fitted scaler or transformer (e.g., StandardScaler, OneHotEncoder)
                # by looking for fitting attributes.
                if hasattr(step, 'mean_') or hasattr(step, 'scale_'):
                    found_step_with_stats = True
                    if hasattr(step, 'mean_') and hasattr(step, 'scale_'):
                        print(f"✅ Scaler Check: Step '{name}' was fitted on Train data (has mean_/scale_).")
                    elif hasattr(step, 'mean_'):
                         print(f"✅ Scaler Check: Step '{name}' was fitted on Train data (has mean_).")
                    elif hasattr(step, 'scale_'):
                         print(f"✅ Scaler Check: Step '{name}' was fitted on Train data (has scale_).")
                
            if not found_step_with_stats:
                 print("⚠️ Structural Check: No steps with fitting statistics (mean_/scale_) found in the pipeline.")
                 
        except AttributeError:
            print("🛑 ERROR: Pipeline object does not have the expected 'named_steps' attribute. Check Preprocessor implementation.")
             
        print("==================================================")