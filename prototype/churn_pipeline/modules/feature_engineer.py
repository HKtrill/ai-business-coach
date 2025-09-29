import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Simplified feature engineering with combined categorical features for better generalization
    """

    def fit(self, X, y=None):
        # Store column names for transforming
        self.input_features_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Store categorical mappings for consistent encoding across train/test
        df = self._prepare_dataframe(X)
        
        # Learn categorical encodings
        self.categorical_mappings_ = {}
        
        # Contract: Natural ordinal ordering (longer = more stable)
        if 'Contract' in df.columns:
            contract_order = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
            self.categorical_mappings_['Contract'] = contract_order
        
        # InternetService: Quality-based ordering (No < DSL < Fiber optic)
        if 'InternetService' in df.columns:
            internet_order = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
            self.categorical_mappings_['InternetService'] = internet_order
        
        # PaymentMethod: Convenience-based ordering (less convenient = higher churn risk)
        if 'PaymentMethod' in df.columns:
            payment_order = {
                'Bank transfer (automatic)': 0,      # Most convenient
                'Credit card (automatic)': 1,        # Second most convenient  
                'Mailed check': 2,                   # Manual but traditional
                'Electronic check': 3                # Least convenient/stable
            }
            self.categorical_mappings_['PaymentMethod'] = payment_order
        
        return self

    def transform(self, X):
        df = self._prepare_dataframe(X)
        
        print("\nApplying Simplified Feature Engineering...")
        
        # 1. Risk Score - higher means more likely to churn
        df['risk_score'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        print("Created feature: risk_score")
        
        # REMOVED: customer_stability (high correlation but zero coefficient - redundant)
        # df['customer_stability'] = (df['tenure'] + 1) / df['MonthlyCharges']
        # print("Created feature: customer_stability")
        
        # 3. Spend Efficiency - total value vs tenure
        df['spend_efficiency'] = df['TotalCharges'] / (df['tenure'] + 1)
        print("Created feature: spend_efficiency")
        
        # 4. Tenure Value Ratio - how long they've stayed relative to what they've paid
        df['tenure_value_ratio'] = (df['tenure'] + 1) / (df['TotalCharges'] + 1)
        print("Created feature: tenure_value_ratio")
        
        # Keep existing engineered features
        if 'Partner' in df.columns and 'Dependents' in df.columns:
            df['family_plan'] = df['Partner'] * df['Dependents']
            print("Created feature: family_plan")
        
        # REMOVED: streaming_services (low predictive value)
        # if 'StreamingTV' in df.columns and 'StreamingMovies' in df.columns:
        #     df['streaming_services'] = df['StreamingTV'] * df['StreamingMovies']
        #     print("Created feature: streaming_services")
        
        # Log transform charges to prevent skewing (keep these)
        df['log_monthly_charges'] = np.log1p(df['MonthlyCharges'])
        df['log_total_charges'] = np.log1p(df['TotalCharges'])
        print("Created features: log_monthly_charges, log_total_charges")
        
        # CATEGORICAL ENCODING: Convert to ordinal based on learned mappings
        for col, mapping in self.categorical_mappings_.items():
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].map(mapping)
                print(f"Encoded categorical: {col} -> {col}_encoded")
                
                # Handle any unmapped values (new categories in test data)
                if df[f'{col}_encoded'].isna().any():
                    print(f"  Warning: Found unmapped values in {col}, filling with median")
                    median_val = int(np.median(list(mapping.values())))
                    df[f'{col}_encoded'].fillna(median_val, inplace=True)
        
        print(f"Feature Engineering complete. Final shape: {df.shape}")
        print(f"Final columns: {df.columns.tolist()}")
        
        return df

    def _prepare_dataframe(self, X):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X.copy(), columns=self.input_features_)
        else:
            df = X.copy()
            self.input_features_ = df.columns.tolist()
        
        print(f"Input columns to feature engineering: {df.columns.tolist()}")
        
        # Ensure numeric conversion
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values
        df.fillna({'TotalCharges': 0}, inplace=True)
        
        return df

    def get_feature_mappings(self):
        """
        Returns the categorical mappings for interpretability
        """
        return getattr(self, 'categorical_mappings_', {})
    
    def get_categorical_features(self):
        """
        Returns list of encoded categorical feature names
        """
        if hasattr(self, 'categorical_mappings_'):
            return [f'{col}_encoded' for col in self.categorical_mappings_.keys()]
        return []