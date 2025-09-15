# =============================================================================
# 6. /content/churn_pipeline/modules/cascade_model.py (FIXED)
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss

class CascadeModel:
    """
    Handles the cascade ensemble model training and prediction.
    Assumes data is already preprocessed and ready for modeling.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.stage1_model = None
        self.stage2_model = None
        self.stage3_model = None

    def train_cascade_pipeline(self, X_train_preprocessed, y_train, X_test_preprocessed, y_test):
        """
        Train the cascade pipeline with multiple stages.
        Assumes data is already preprocessed.
        """
        print(f"\n=== TRAINING CASCADE PIPELINE ===")
        
        # ============================================
        # DEBUGGING: Print data BEFORE anything else
        # ============================================
        print("\n=== Data just before SMOTE (cannot be skipped) ===")
        print("X_train_preprocessed (first 5 rows):")
        print(pd.DataFrame(X_train_preprocessed).head().to_string())
        
        print("\ny_train (first 5 rows):")
        print(pd.DataFrame(y_train).head().to_string())
        print("===============================\n")
        # ============================================
        
        # Apply SMOTE for class balancing on the already preprocessed data
        smote = SMOTE(random_state=self.random_state)
        
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)
        except Exception as e:
            print(f"Error during SMOTE resampling: {e}")
            print("\nThe SMOTE operation is failing because the input data contains non-numeric values. This means the preprocessor is not working correctly.")
            return None, None, None

        # Initialize models with different complexities
        self.stage1_model = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight='balanced', random_state=self.random_state
        )
        self.stage2_model = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight='balanced', random_state=self.random_state
        )
        self.stage3_model = RandomForestClassifier(
            n_estimators=300, max_depth=16,
            class_weight='balanced', random_state=self.random_state
        )
        
        print("Training Stage 1...")
        self.stage1_model.fit(X_train_balanced, y_train_balanced)
        
        print("Training Stage 2...")
        self.stage2_model.fit(X_train_balanced, y_train_balanced)

        print("Training Stage 3...")
        self.stage3_model.fit(X_train_balanced, y_train_balanced)
        
        print("\nMaking predictions on the test set...")
        y_proba1 = self.stage1_model.predict_proba(X_test_preprocessed)[:, 1]
        y_proba2 = self.stage2_model.predict_proba(X_test_preprocessed)[:, 1]
        y_proba3 = self.stage3_model.predict_proba(X_test_preprocessed)[:, 1]
        
        y_proba_final = (y_proba1 * 0.2) + (y_proba2 * 0.4) + (y_proba3 * 0.4)
        
        y_pred = (y_proba_final > 0.5).astype(int)
        
        return y_test, y_pred, y_proba_final
