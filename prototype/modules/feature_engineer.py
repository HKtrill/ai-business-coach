import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    SIMPLIFIED Feature engineering transformer. 
    Focuses only on 3 core, essential features to combat over-engineering.
    """

    def fit(self, X, y=None):
        # Store column names for transforming
        self.input_features_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
        return self

    def transform(self, X):
        # Ensure we can handle NumPy array input from the pipeline
        if isinstance(X, np.ndarray):
            # We must use the column names stored in fit()
            df = pd.DataFrame(X.copy(), columns=self.input_features_)
        else:
            df = X.copy()
            self.input_features_ = df.columns.tolist() # Update if initial fit was skipped

        print("\nApplying SIMPLIFIED Feature Engineering (3 Core Features)...")

        # 1. CLEANING/PREP
        # Ensure numeric conversion (TotalCharges handling is critical here)
        numeric_cols_to_convert = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill TotalCharges NaNs (which come from 0-tenure customers) with 0
        df.fillna({'TotalCharges': 0}, inplace=True) 

        # --- CORE ENGINEERED FEATURES (2-3 MAX) ---

        # 2. CORE RISK SCORE: Monthly Rate per Tenure Month (High score = high risk)
        tenure = df.get('tenure', pd.Series(0)).replace(0, 1e-6)
        monthly_charges = df.get('MonthlyCharges', pd.Series(0)).replace(0, 1e-6)
        df['risk_score'] = monthly_charges / tenure
        print("Created feature: risk_score")

        # 3. TENURE RATIO: Simple Inverse
        df['tenure_monthly_ratio'] = tenure / monthly_charges
        print("Created feature: tenure_monthly_ratio")

        # 4. STABILITY: Based on Tenure (The most critical non-financial stability factor)
        df['tenure_stability'] = np.where(tenure > 24, 2, np.where(tenure > 12, 1, 0))
        print("Created feature: tenure_stability")

        # --- FINAL CLEANUP ---
        
        # We must keep all original columns that were passed to this transformer 
        # (plus the new engineered ones) so the ColumnTransformer can run next.
        
        print(f"Simplified Feature Engineering complete. New shape: {df.shape}")
        
        return df