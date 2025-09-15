# =============================================================================
# 7. /content/churn_pipeline/modules/experiment_runner.py (FIXED)
# =============================================================================
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from data_loader import DataLoader
from leakage_monitor import DataLeakageMonitor
from preprocessor import Preprocessor
from cascade_model import CascadeModel

# Import utils functions directly
def set_seed(seed):
    """Set seed for python random, numpy, and environment hash seed for reproducibility."""
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
        self.cascade_model = CascadeModel(random_state=random_state)
        self.artifacts_dir = "artifacts"
        
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(f"{self.artifacts_dir}/models", exist_ok=True)
        os.makedirs(f"{self.artifacts_dir}/preprocessors", exist_ok=True)
        os.makedirs(f"{self.artifacts_dir}/plots", exist_ok=True)
        os.makedirs("data_splits", exist_ok=True)
    
    def run_experiment(self, data_path, split_seed=None):
        """Run a single experiment with the given data and seed"""
        print("="*60)
        print("CHURN PREDICTION PIPELINE - FINAL VERSION")
        print("="*60)
        
        data = self.data_loader.load_raw_data(data_path)
        if data is None:
            return None
        
        current_seed = split_seed if split_seed is not None else random.randint(1,10000)
        set_seed(current_seed)
        
        X = data.drop(columns=['Churn'])
        y = data['Churn']
        
        # === THIS IS THE FIX: Map the y variable before the split ===
        y = y.map({'Yes': 1, 'No': 0})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=current_seed, stratify=y
        )

        print(f"\nInitial splits created. Train: {len(X_train)}, Test: {len(X_test)}")
        
        self.leakage_monitor.record_split_info(X_train, X_test, y_train, y_test)
        
        print("Preprocessing data for modeling...")
        X_train_preprocessed = self.preprocessor.fit_transform(X_train)
        X_test_preprocessed = self.preprocessor.transform(X_test)
        
        # ============================================
        # DEBUGGING: Print the preprocessed data here
        # ============================================
        print("\nHead of X_train_preprocessed (as DataFrame):")
        print(pd.DataFrame(X_train_preprocessed).head())
        print("\nHead of y_train:")
        print(y_train.head())
        print("\nHead of X_test_preprocessed (as DataFrame):")
        print(pd.DataFrame(X_test_preprocessed).head())
        print("\nHead of y_test:")
        print(y_test.head())
        # ============================================

        # 4. Train Cascade Model
        y_true, y_pred, y_proba = self.cascade_model.train_cascade_pipeline(
            X_train_preprocessed, y_train, X_test_preprocessed, y_test
        )
        
        if y_true is None:
            print("Experiment failed due to data preprocessing error.")
            return None
        
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
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return metrics

    def run_multiple_experiments(self, data_path, n_splits=5, seeds=None, results_path=None):
        """Run multiple experiments with different seeds"""
        if seeds is None:
            seeds = [random.randint(1, 10000) for _ in range(n_splits)]
        
        results = []
        for i, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"RUNNING EXPERIMENT {i+1}/{len(seeds)} WITH SEED {seed}")
            print(f"{'='*60}")
            
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