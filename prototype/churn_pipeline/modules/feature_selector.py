# feature_selector.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessor import Preprocessor
from feature_engineer import FeatureEngineer # 🚨 ADDED

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects only the recommended features from a dataset to reduce noise
    in modeling. This is useful after feature engineering.
    """
    
    def __init__(self, recommended_features=None, verbose=True):
        if recommended_features is None:
            # Actual feature names after your preprocessor runs
            self.recommended_features = [
                'tenure', 'MonthlyCharges', 'TotalCharges', 
                'InternetService_Fiber optic', 'Contract_One year', 'Contract_Two year',
                'PaymentMethod_Electronic check', 'PaymentMethod_Credit card (automatic)',
                'Partner', 'Dependents', 'PaperlessBilling'
            ]
        else:
            self.recommended_features = recommended_features
        self.verbose = verbose
        self.selected_features_ = []
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer() # 🚨 ADDED

    def fit(self, X, y=None):
        """Fit the selector: preprocess -> engineer data first, then check which features exist."""
        # First preprocess the data
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        # 🚨 FEATURE ENGINEERING STEP
        X_engineered = self.feature_engineer.fit_transform(X_preprocessed, y)
        
        # Convert to DataFrame if it's a numpy array
        if not isinstance(X_engineered, pd.DataFrame):
            # The X_preprocessed from the external pipeline is what is expected here
            # But since this class runs it internally, we use the engineered result
            X_transformed = X_engineered
        else:
            X_transformed = X_engineered
            
        # Check which recommended features exist in the preprocessed data
        self.selected_features_ = [f for f in self.recommended_features if f in X_transformed.columns]
        missing_features = set(self.recommended_features) - set(self.selected_features_)
        
        if self.verbose and missing_features:
            print(f"[FeatureSelector] Warning: Missing recommended features: {missing_features}")
        
        if self.verbose:
            print(f"[FeatureSelector] Selected {len(self.selected_features_)} features out of {X_transformed.shape[1]}")
        
        return self

    def transform(self, X):
        """Transform the dataset: preprocess -> engineer -> select features."""
        if not hasattr(self, 'selected_features_'):
            raise ValueError("FeatureSelector is not fitted yet.")
        
        # Preprocess the data
        X_preprocessed = self.preprocessor.transform(X)
        
        # 🚨 FEATURE ENGINEERING STEP
        X_engineered = self.feature_engineer.transform(X_preprocessed)
        
        # Convert to DataFrame if it's a numpy array
        if not isinstance(X_engineered, pd.DataFrame):
            # Ensure it can be treated like a DataFrame for selection
            X_transformed = X_engineered
        else:
            X_transformed = X_engineered
        
        # Select only the recommended features
        if len(self.selected_features_) == 0:
            print("[FeatureSelector] Warning: No features selected. Using all features.")
            X_selected = X_transformed
        else:
            X_selected = X_transformed[self.selected_features_].copy()
        
        if self.verbose:
            print(f"[FeatureSelector] Selected {len(self.selected_features_)} features")
            
        return X_selected

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def get_support(self):
        """Return list of selected features."""
        return self.selected_features_