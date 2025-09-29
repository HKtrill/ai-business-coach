# =============================================================================
# 7. /content/churn_pipeline/modules/experiment_runner.py
# FIXED VERSION - COLUMNS DROPPED BEFORE MODEL TRAINING
# =============================================================================
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif 
# --- NEW IMPORTS FOR ELASTIC NET ---
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler # Ensure data is scaled for L1 penalty
# -----------------------------------

from data_loader import DataLoader
from leakage_monitor import DataLeakageMonitor
from preprocessor import Preprocessor
from feature_engineer import FeatureEngineer
from cascade_model import CascadeModel


# Utility: set reproducible seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class ExperimentRunner:
    """Complete churn prediction pipeline with strict anti-leakage measures"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data_loader = DataLoader()
        self.leakage_monitor = DataLeakageMonitor()
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.cascade_model = CascadeModel(random_state=random_state)
        self.artifacts_dir = "artifacts"

        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(f"{self.artifacts_dir}/models", exist_ok=True)
        os.makedirs(f"{self.artifacts_dir}/preprocessors", exist_ok=True)
        os.makedirs(f"{self.artifacts_dir}/plots", exist_ok=True)
        os.makedirs("data_splits", exist_ok=True)
        
        # Define columns to drop for final analysis (UPDATED to remove duplicates)
        self.cols_to_drop = [
            # Original categorical columns (replaced by encoded versions)
            'Contract', 'InternetService', 'PaymentMethod',
            
            # Duplicate dummy variables (replaced by encoded versions)  
            'Contract_One year', 'Contract_Two year',
            'InternetService_Fiber optic', 'InternetService_No',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'PaymentMethod_Credit card (automatic)',
            
            # Low-value features identified by feature selection
            'Partner', 'Dependents', 'gender', 'family_plan',
            'StreamingTV', 'StreamingMovies', 'streaming_services',
            'customer_stability',  # Redundant with other features
            
            # Raw charge features (replaced by log versions)
            'MonthlyCharges', 'TotalCharges', 'log_monthly_charges',
            
            # Other low-importance features
            'PhoneService', 'MultipleLines', 'OnlineBackup', 
            'DeviceProtection', 'OnlineProtection',
            'tenure', 'spend_efficiency', 'risk_score',
            'TechSupport', 'SeniorCitizen', #'tenure_value_ratio'
        ]
        
    def run_elastic_net_feature_selection(self, X_train, y_train, feature_names):
        """
        Uses Logistic Regression with Elastic Net (L1/Lasso dominant) penalty 
        to identify redundant features by driving their coefficients to zero.
        """
        print("\n=== ELASTIC NET FEATURE REDUNDANCY TEST ===")
        
        # 1. Standardize data (L1 regularization requires scaled data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 2. Train Elastic Net Logistic Regression
        # C is inverse of regularization strength (smaller C = stronger penalty)
        # l1_ratio=0.9 ensures a strong L1 (Lasso) penalty to zero out coefficients
        enet = LogisticRegression(
            penalty='elasticnet',
            solver='saga', # 'saga' is required for elasticnet/L1
            C=0.1,         # Moderate to strong regularization
            l1_ratio=0.9,
            max_iter=500,
            random_state=self.random_state
        )
        enet.fit(X_train_scaled, y_train)
        
        # 3. Analyze Coefficients
        coefs = pd.Series(enet.coef_[0], index=feature_names).sort_values(key=lambda x: np.abs(x), ascending=False)
        
        print("\n--- Elastic Net Coefficients (Magnitude) ---")
        print("Features with coefficients near 0 are highly redundant and can be removed.")
        print(coefs)
        print("=" * 50)
        return coefs

    def _drop_columns_from_features(self, X_df, feature_names):
        """Drop specified columns from feature set for final analysis"""
        # Get columns that actually exist in the current feature set
        existing_cols_to_drop = [col for col in self.cols_to_drop if col in X_df.columns]
        
        if existing_cols_to_drop:
            print(f"\nDropping {len(existing_cols_to_drop)} columns for final analysis:")
            print(f"  Categorical originals: {[col for col in existing_cols_to_drop if col in ['Contract', 'InternetService', 'PaymentMethod']]}")
            print(f"  Duplicate dummies: {[col for col in existing_cols_to_drop if any(cat in col for cat in ['Contract_', 'InternetService_', 'PaymentMethod_'])]}")
            print(f"  Low-value features: {[col for col in existing_cols_to_drop if col not in ['Contract', 'InternetService', 'PaymentMethod'] and not any(cat in col for cat in ['Contract_', 'InternetService_', 'PaymentMethod_'])]}")
            
            X_clean = X_df.drop(columns=existing_cols_to_drop)
            
            # Update feature names to match
            clean_feature_names = [name for name in feature_names if name not in existing_cols_to_drop]
            
            print(f"\nFeatures after cleaning: {len(clean_feature_names)}")
            print(f"Remaining features: {clean_feature_names}")
            return X_clean, clean_feature_names
        else:
            print("\nNo columns to drop found in current feature set")
            return X_df, feature_names

    def run_experiment(self, data_path, split_seed=None):
        """Run a single experiment with the given data and seed"""
        print("=" * 60)
        print("CHURN PREDICTION PIPELINE - FIXED VERSION")
        print("=" * 60)

        # Load raw data
        data = self.data_loader.load_raw_data(data_path)
        if data is None:
            return None

        TARGET_COLUMN = 'Churn'
        X = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN].map({'Yes': 1, 'No': 0})

        # Train/test split
        current_seed = split_seed if split_seed is not None else random.randint(1, 10000)
        set_seed(current_seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=current_seed, stratify=y
        )
        print(f"\nInitial splits created. Train: {len(X_train)}, Test: {len(X_test)}")

        # Preprocessing
        print("Preprocessing data for modeling...")
        X_train_preprocessed = self.preprocessor.fit_transform(X_train)
        X_test_preprocessed = self.preprocessor.transform(X_test)
        self.leakage_monitor.check_preprocessing_leakage(self.preprocessor.pipeline)

        clean_feature_names = self.preprocessor.get_clean_feature_names()
        if clean_feature_names is None:
            clean_feature_names = [f'feature_{i}' for i in range(X_train_preprocessed.shape[1])]

        print(f"\nX_train_preprocessed shape: {X_train_preprocessed.shape}")
        print(f"X_test_preprocessed shape: {X_test_preprocessed.shape}")

        # ====================================================================
        # DROP COLUMNS BEFORE MODEL TRAINING (FIXED!)
        # ====================================================================
        print("\n=== CLEANING FEATURE SET (Removing Duplicates & Low-Value Features) ===")
        
        # Convert to DataFrames first
        X_train_final = pd.DataFrame(X_train_preprocessed, columns=clean_feature_names)
        X_test_final = pd.DataFrame(X_test_preprocessed, columns=clean_feature_names)

        # Drop columns BEFORE model training
        X_train_clean, clean_feature_names_final = self._drop_columns_from_features(X_train_final, clean_feature_names)
        X_test_clean, _ = self._drop_columns_from_features(X_test_final, clean_feature_names)

        print(f"\nFinal feature set for MODEL TRAINING: {len(clean_feature_names_final)} features")

        # Sample of final preprocessed features (AFTER dropping)
        print("\nSample of cleaned data (first 3 rows):")
        sample_df = X_train_clean.head(3)
        print(sample_df)
        print("=" * 50)
        
        # ====================================================================
        # ELASTIC NET TEST (on CLEANED data)
        # ====================================================================
        self.run_elastic_net_feature_selection(
             X_train_clean.values, y_train, clean_feature_names_final
        )

        # ====================================================================
        # Train Cascade Model (on CLEANED data)
        # ====================================================================
        y_true, y_pred, y_proba = self.cascade_model.train_cascade_pipeline(
            X_train_clean.values, y_train, X_test_clean.values, y_test
        )
        if y_true is None:
            print("Experiment failed due to data preprocessing error.")
            return None

        # ====================================================================
        # FINAL MODEL EVALUATION
        # ====================================================================
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("\n=== FINAL MODEL EVALUATION ===")
        print(f"Confusion Matrix:\n  TP: {tp}, FP: {fp}\n  FN: {fn}, TN: {tn}")

        business_cost = (fn * 750) + (fp * 75)
        metrics = {
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'ROC AUC': roc_auc_score(y_true, y_proba),
            'BusinessCost': business_cost
        }
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"\n  Total Business Cost: ${business_cost:.2f}")

        # ====================================================================
        # STATISTICAL ANALYSIS ON FINAL FEATURES (MOVED TO BOTTOM)
        # ====================================================================
        print("\n=== STATISTICAL ANALYSIS ON FINAL FEATURES (POST-MODEL) ===")
        
        print(f"\nFinal feature set for analysis: {len(clean_feature_names_final)} features")
        print(f"Features: {clean_feature_names_final}")

        # Simple descriptive stats on CLEANED data
        stats_train = X_train_clean.describe().T
        stats_test = X_test_clean.describe().T
        stats_train['Test Mean'] = stats_test['mean']
        stats_train['Test Std'] = stats_test['std']
        stats_train['Mean Diff (%)'] = ((stats_train['mean'] - stats_train['Test Mean']).abs() / (stats_train['mean'].replace(0, 1))) * 100

        print("\n--- Descriptive Statistics (After Cleaning) ---")
        print(stats_train)
        print("=" * 50)

        # ====================================================================
        # FEATURE RELEVANCE ANALYSIS (MOVED LOGISTIC REGRESSION HERE)
        # ====================================================================
        
        # 1. Mutual Information (feature relevance) on CLEANED data
        print("\n--- Feature Relevance (Mutual Information - Cleaned Data) ---")
        mi = mutual_info_classif(X_train_clean, y_train, discrete_features='auto', random_state=self.random_state)
        mi_series = pd.Series(mi, index=clean_feature_names_final).sort_values(ascending=False)
        print(mi_series)
        print("=" * 50)

        # 2. Elastic Net Analysis on CLEANED data (MOVED HERE)
        print("\n--- Elastic Net Feature Analysis (Cleaned Data) ---")
        enet_coefs = self.run_elastic_net_feature_selection(
            X_train_clean.values, y_train, clean_feature_names_final
        )
        
        # 3. Combine both analyses for comprehensive feature ranking
        print("\n--- COMPREHENSIVE FEATURE RANKING ---")
        feature_analysis = pd.DataFrame({
            'feature': clean_feature_names_final,
            'mutual_info': [mi_series[feature] for feature in clean_feature_names_final],
            'elastic_net_coef': [enet_coefs[feature] for feature in clean_feature_names_final],
            'elastic_net_abs': [abs(enet_coefs[feature]) for feature in clean_feature_names_final]
        })
        
        # Rank features by both metrics
        feature_analysis['mi_rank'] = feature_analysis['mutual_info'].rank(ascending=False)
        feature_analysis['enet_rank'] = feature_analysis['elastic_net_abs'].rank(ascending=False)
        feature_analysis['combined_rank'] = (feature_analysis['mi_rank'] + feature_analysis['enet_rank']) / 2
        
        feature_analysis = feature_analysis.sort_values('combined_rank')
        print(feature_analysis[['feature', 'mutual_info', 'elastic_net_coef', 'combined_rank']].round(4))
        print("=" * 50)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return metrics

    def run_multiple_experiments(self, data_path, n_splits=5, seeds=None, results_path=None):
        """Run multiple experiments with different seeds"""
        if seeds is None:
            seeds = [random.randint(1, 10000) for _ in range(n_splits)]

        results = []
        for i, seed in enumerate(seeds):
            print(f"\n{'=' * 60}")
            print(f"RUNNING EXPERIMENT {i+1}/{len(seeds)} WITH SEED {seed}")
            print(f"{'=' * 60}")

            result = self.run_experiment(data_path, split_seed=seed)
            if result is not None:
                result['seed'] = seed
                result['experiment'] = i + 1
                results.append(result)

        results_df = pd.DataFrame(results)
        if results_path:
            results_df.to_csv(results_path, index=False)
            print(f"\nResults saved to {results_path}")

        return results_df