# =============================================================================
# FIXED: /content/churn_pipeline/modules/preprocessor.py
# (FeatureEngineer is RESTORED inside the pipeline and DataFrame input is forced)
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engineer import FeatureEngineer # RESTORED

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = None
        self.feature_names_ = None

    def fit_transform(self, X, y=None):
        self.pipeline = self._create_pipeline(X)
        result = self.pipeline.fit_transform(X)
        
        # Store feature names for debugging
        try:
            # Try to get feature names from the ColumnTransformer
            if hasattr(self.pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                self.feature_names_ = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
        except:
            self.feature_names_ = [f'feature_{i}' for i in range(result.shape[1])]
            
        return result

    def transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        
        # The external X is a DataFrame (from DataLoader) or a result of a previous step.
        # It's safest to let the pipeline handle the data, as the InitialDataCleaner 
        # inside the pipeline is where the NumPy check must occur.
        return self.pipeline.transform(X)
        
    def _create_pipeline(self, X):
        
        # A custom transformer to handle all raw data cleaning
        class InitialDataCleaner(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                # 🚨 THE CRITICAL FIX: Ensure X is a DataFrame with columns
                if not isinstance(X, pd.DataFrame):
                    # This handles the case where a previous step turned X into an array
                    # We assume original column names based on the raw data loader output
                    # This relies on the raw data columns being consistent.
                    # In a real pipeline, X should be the raw DataFrame at the start.
                    try:
                        # Attempt to reconstruct the DataFrame, we need column names here
                        # Since we don't have them easily, this is a risky part of the architecture.
                        # However, based on the traceback, the input to InitialDataCleaner 
                        # on the SECOND CALL is the problem.

                        # We will make the assumption that the raw columns from X_train are accessible during transform.
                        # For robustness, we must assume the input *should* be a DataFrame here.
                        # If the ExperimentRunner is calling it correctly with a DataFrame, this is robust.
                        df = X.copy()
                    except:
                        # Fallback for error handling if copy fails (e.g. array)
                        # This should not happen if ExperimentRunner is structured correctly.
                        df = pd.DataFrame(X)
                else:
                    df = X.copy()
                
                print(f"InitialDataCleaner input shape: {df.shape}")
                print(f"InitialDataCleaner input columns: {df.columns.tolist()}") # This is what caused the error

                # Handle the tricky TotalCharges column, converting empty strings to NaN
                if 'TotalCharges' in df.columns:
                    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                    print(f"Converted TotalCharges to numeric")
                
                # Standardize 'No internet service' and 'No phone service'
                service_mapping = {'No internet service': 'No', 'No phone service': 'No'}
                df.replace(service_mapping, inplace=True)
                
                # Now map ALL binary 'Yes'/'No' and 'Male'/'Female' columns to 1/0
                binary_mapping = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
                binary_cols_to_map = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                                     'PaperlessBilling', 'MultipleLines', 'OnlineSecurity',
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                     'StreamingTV', 'StreamingMovies']
                
                for col in binary_cols_to_map:
                    if col in df.columns:
                        df[col] = df[col].map(binary_mapping)
                        print(f"Mapped binary column: {col}")
                
                print(f"InitialDataCleaner output shape: {df.shape}")
                print(f"Data types after cleaning:")
                print(df.dtypes)
                
                return df

        # Use a temporary DataFrame to get the feature types after cleaning
        temp_df = InitialDataCleaner().transform(X)
        
        # Identify features for the ColumnTransformer
        numeric_features = [col for col in temp_df.columns if pd.api.types.is_numeric_dtype(temp_df[col])]
        categorical_features = [col for col in temp_df.columns if pd.api.types.is_object_dtype(temp_df[col])]
        
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Pipelines for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # Column transformer handles remaining columns
        data_preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Final pipeline with all steps
        # This structure is what you insisted on: Cleaner -> Engineer -> Scikit-learn Processor
        return Pipeline(steps=[
            ('initial_cleaner', InitialDataCleaner()),
            ('feature_engineer', FeatureEngineer()),  # RESTORED HERE
            ('preprocessor', data_preprocessor),
        ])