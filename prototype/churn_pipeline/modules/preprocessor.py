import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engineer import FeatureEngineer

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = None
        self.feature_names_ = None
        self.clean_feature_names_ = None

    def fit_transform(self, X, y=None):
        self.pipeline = self._create_pipeline(X)
        result = self.pipeline.fit_transform(X)
        
        # Store both original and clean feature names
        try:
            if hasattr(self.pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                self.feature_names_ = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
                # Clean the feature names by removing prefixes
                self.clean_feature_names_ = [name.split('__')[-1] for name in self.feature_names_]
        except:
            self.feature_names_ = [f'feature_{i}' for i in range(result.shape[1])]
            self.clean_feature_names_ = self.feature_names_
            
        return result

    def transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        
        return self.pipeline.transform(X)
    
    def get_clean_feature_names(self):
        """Return feature names without ColumnTransformer prefixes"""
        return self.clean_feature_names_ if self.clean_feature_names_ is not None else self.feature_names_
        
    def _create_pipeline(self, X):
        
        # A custom transformer to handle all raw data cleaning
        class InitialDataCleaner(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    try:
                        df = X.copy()
                    except:
                        df = pd.DataFrame(X)
                else:
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

        # Create a temporary pipeline to see what columns exist AFTER feature engineering
        temp_pipeline = Pipeline(steps=[
            ('initial_cleaner', InitialDataCleaner()),
            ('feature_engineer', FeatureEngineer())
        ])
        
        # Transform a small sample to see the final column structure
        temp_df = temp_pipeline.fit_transform(X.head())
        
        # NOW identify features based on what exists AFTER feature engineering
        numeric_features = [col for col in temp_df.columns if pd.api.types.is_numeric_dtype(temp_df[col])]
        categorical_features = [col for col in temp_df.columns if pd.api.types.is_object_dtype(temp_df[col])]
        
        print(f"POST-ENGINEERING Numeric features: {numeric_features}")
        print(f"POST-ENGINEERING Categorical features: {categorical_features}")
        
        # Pipelines for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # Column transformer with verbose_feature_names_out=False to reduce prefixes
        data_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),  # Shorter prefix
                ('cat', categorical_transformer, categorical_features)  # Shorter prefix
            ],
            remainder='drop',
            verbose_feature_names_out=False  # This reduces the prefixing
        )
        
        # CORRECT ORDER: Cleaner -> Engineer -> Processor
        return Pipeline(steps=[
            ('initial_cleaner', InitialDataCleaner()),
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', data_preprocessor),
        ])