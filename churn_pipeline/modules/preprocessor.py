# =============================================================================
# FIXED: /content/churn_pipeline/modules/preprocessor.py
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Import the fixed FeatureEngineer
from feature_engineer import FeatureEngineer

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
        return self.pipeline.transform(X)
        
    def _create_pipeline(self, X):
        
        # A custom transformer to handle all raw data cleaning
        class InitialDataCleaner(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                df = X.copy()
                
                print(f"InitialDataCleaner input shape: {df.shape}")
                print(f"InitialDataCleaner input columns: {df.columns.tolist()}")
                    
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
        return Pipeline(steps=[
            ('initial_cleaner', InitialDataCleaner()),
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', data_preprocessor),
        ])

# =============================================================================
# ALTERNATIVE: SIMPLIFIED PREPROCESSOR (Use this if above still has issues)
# =============================================================================
class SimplifiedPreprocessor(BaseEstimator, TransformerMixin):
    """Simplified preprocessor that focuses on getting the pipeline working"""
    
    def __init__(self):
        self.pipeline = None
        self.feature_names_ = None

    def fit_transform(self, X, y=None):
        self.pipeline = self._create_pipeline(X)
        result = self.pipeline.fit_transform(X)
        return result

    def transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.pipeline.transform(X)
        
    def _create_pipeline(self, X):
        
        # A custom transformer to handle all raw data cleaning
        class InitialDataCleaner(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                df = X.copy()
                    
                # Handle the tricky TotalCharges column, converting empty strings to NaN
                if 'TotalCharges' in df.columns:
                    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                
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
                
                return df

        # Use a temporary DataFrame to get the feature types after cleaning
        temp_df = InitialDataCleaner().transform(X)
        
        # Identify features for the ColumnTransformer
        numeric_features = [col for col in temp_df.columns if pd.api.types.is_numeric_dtype(temp_df[col])]
        categorical_features = [col for col in temp_df.columns if pd.api.types.is_object_dtype(temp_df[col])]
        
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
        
        # inside Preprocessor._create_pipeline()
        return Pipeline(steps=[
            ('initial_cleaner', InitialDataCleaner()),
            ('feature_engineer', FeatureEngineer()),   # <-- moved BEFORE
            ('preprocessor', data_preprocessor)        # <-- runs last
        ])

