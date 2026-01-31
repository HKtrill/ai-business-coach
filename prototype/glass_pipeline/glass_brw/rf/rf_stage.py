"""
Random Forest Stage (BRW Features)
===================================

Hyperparameter tuning and training on GLASS-BRW binary features.

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, fbeta_score, confusion_matrix,
    precision_recall_curve, roc_curve, make_scorer
)
from sklearn.inspection import permutation_importance
from scipy.stats import randint
from typing import Dict, Tuple
import pickle
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class RFStage:
    """
    Random Forest stage for GLASS-BRW pipeline.
    
    Trains on binary engineered features with hyperparameter tuning.
    """
    
    def __init__(
        self,
        n_iter: int = 60,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize RF stage.
        
        Parameters
        ----------
        n_iter : int
            Number of RandomizedSearchCV iterations
        cv_folds : int
            Number of CV folds
        random_state : int
            Random state
        """
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.model = None
        self.best_params = {}
        self.best_cv_score = None
        self.optimal_threshold = 0.5
        self.train_metrics = {}
        self.test_metrics = {}
        self.gini_importance = None
        self.perm_importance = None
        self.threshold_df = None
        self.fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> 'RFStage':
        """
        Fit RF with hyperparameter tuning.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (BRW binary)
        y_train : pd.Series
            Training labels
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
            
        Returns
        -------
        self : RFStage
            Fitted stage
        """
        print("=" * 80)
        print("🌲 RF STAGE: Hyperparameter Tuning (BRW Features)")
        print("=" * 80)
        
        print(f"\n📊 Data:")
        print(f"   Train: {X_train.shape} | Positives: {y_train.sum()} ({y_train.mean():.4f})")
        print(f"   Test:  {X_test.shape} | Positives: {y_test.sum()} ({y_test.mean():.4f})")
        
        # Define search space
        param_distributions = {
            'n_estimators': randint(100, 400),
            'max_depth': randint(5, 15),
            'min_samples_split': randint(20, 100),
            'min_samples_leaf': randint(10, 50),
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True],
            'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 3}, {0: 1, 1: 5}],
            'criterion': ['gini', 'entropy'],
        }
        
        print("\n🔧 Hyperparameter Search:")
        print(f"   Iterations: {self.n_iter}")
        print(f"   CV: {self.cv_folds}-fold stratified")
        print(f"   Scoring: F2 (recall-favoring)")
        
        # F2 scorer
        f2_scorer = make_scorer(fbeta_score, beta=2)
        
        # CV splitter
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # RandomizedSearchCV
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, verbose=0)
        
        random_search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            scoring=f2_scorer,
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        print("\n   Running search...")
        random_search.fit(X_train, y_train)
        
        # Store results
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_cv_score = random_search.best_score_
        
        print(f"\n✅ Best CV F2 Score: {self.best_cv_score:.4f}")
        print(f"\n📊 Best Hyperparameters:")
        for param, value in sorted(self.best_params.items()):
            print(f"   {param:20s}: {value}")
        
        # Compute metrics
        self._compute_metrics(X_train, y_train, X_test, y_test)
        
        # Feature importance
        self._compute_feature_importance(X_train, X_test, y_test)
        
        # Threshold optimization
        self._optimize_threshold(X_test, y_test)
        
        self.fitted = True
        return self
    
    def _compute_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """Compute train and test metrics."""
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Training metrics
        self.train_metrics = {
            'Accuracy': accuracy_score(y_train, y_train_pred),
            'Precision': precision_score(y_train, y_train_pred),
            'Recall': recall_score(y_train, y_train_pred),
            'F1-Score': f1_score(y_train, y_train_pred),
            'F2-Score': fbeta_score(y_train, y_train_pred, beta=2),
            'ROC-AUC': roc_auc_score(y_train, y_train_proba)
        }
        
        # Test metrics
        self.test_metrics = {
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred),
            'Recall': recall_score(y_test, y_test_pred),
            'F1-Score': f1_score(y_test, y_test_pred),
            'F2-Score': fbeta_score(y_test, y_test_pred, beta=2),
            'ROC-AUC': roc_auc_score(y_test, y_test_proba)
        }
        
        print("\n📊 Performance:")
        print("\n🔵 TRAINING SET:")
        for k, v in self.train_metrics.items():
            print(f"   {k}: {v:.4f}")
        
        print("\n🟢 TEST SET:")
        for k, v in self.test_metrics.items():
            print(f"   {k}: {v:.4f}")
        
        # Overfitting analysis
        print("\n🔍 OVERFITTING ANALYSIS:")
        for metric in ['Accuracy', 'F1-Score', 'F2-Score']:
            gap = self.train_metrics[metric] - self.test_metrics[metric]
            status = "✅" if gap < 0.03 else "⚠️" if gap < 0.07 else "❌"
            print(f"   {metric}: Gap={gap:.4f} {status}")
    
    def _compute_feature_importance(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """Compute Gini and permutation importance."""
        print("\n📊 FEATURE IMPORTANCE:")
        
        # Gini
        self.gini_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Gini': self.model.feature_importances_
        }).sort_values('Gini', ascending=False)
        
        print("\n🔝 Top 15 Features (Gini):")
        print(self.gini_importance.head(15).to_string(index=False))
        
        # Permutation
        print("\n🔀 Computing permutation importance...")
        perm_imp = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1,
            scoring='roc_auc'
        )
        
        self.perm_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Perm_Imp': perm_imp.importances_mean,
            'Perm_Std': perm_imp.importances_std
        }).sort_values('Perm_Imp', ascending=False)
        
        print("\n🔝 Top 15 Features (Permutation):")
        print(self.perm_importance.head(15).to_string(index=False))
    
    def _optimize_threshold(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """Optimize decision threshold for F2-score."""
        print("\n🎯 THRESHOLD OPTIMIZATION:")
        
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
        
        threshold_results = []
        for i, thresh in enumerate(thresholds):
            y_pred_t = (y_test_proba >= thresh).astype(int)
            if y_pred_t.sum() == 0:
                continue
            threshold_results.append({
                'threshold': thresh,
                'precision': precision_score(y_test, y_pred_t),
                'recall': recall_score(y_test, y_pred_t),
                'f1': f1_score(y_test, y_pred_t),
                'f2': fbeta_score(y_test, y_pred_t, beta=2),
            })
        
        self.threshold_df = pd.DataFrame(threshold_results)
        best_f2_idx = self.threshold_df['f2'].idxmax()
        self.optimal_threshold = self.threshold_df.loc[best_f2_idx, 'threshold']
        
        print(f"   Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"   F2-score: {self.threshold_df.loc[best_f2_idx, 'f2']:.4f}")
    
    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Predict class labels."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if threshold is None:
            threshold = self.optimal_threshold
        
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def plot_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Generate evaluation plots."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        y_test_pred = self.predict(X_test)
        y_test_proba = self.predict_proba(X_test)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Gini Importance
        ax1 = axes[0, 0]
        top_gini = self.gini_importance.head(15)
        ax1.barh(top_gini['Feature'], top_gini['Gini'], color='forestgreen')
        ax1.set_xlabel('Gini Importance')
        ax1.set_title('Top 15 Features (Gini)', fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Permutation Importance
        ax2 = axes[0, 1]
        top_perm = self.perm_importance.head(15)
        ax2.barh(top_perm['Feature'], top_perm['Perm_Imp'],
                 xerr=top_perm['Perm_Std'], color='steelblue')
        ax2.set_xlabel('Permutation Importance')
        ax2.set_title('Top 15 Features (Permutation)', fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Precision-Recall vs Threshold
        ax3 = axes[0, 2]
        ax3.plot(self.threshold_df['threshold'], self.threshold_df['precision'], 
                'b-', linewidth=2, label='Precision')
        ax3.plot(self.threshold_df['threshold'], self.threshold_df['recall'],
                'r-', linewidth=2, label='Recall')
        ax3.plot(self.threshold_df['threshold'], self.threshold_df['f2'],
                'g--', linewidth=2, label='F2')
        ax3.axvline(x=self.optimal_threshold, color='purple', linestyle=':',
                   linewidth=2, label=f'Optimal ({self.optimal_threshold:.3f})')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Score')
        ax3.set_title('Metrics vs Threshold', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. ROC Curve
        ax4 = axes[1, 0]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        ax4.plot(fpr, tpr, 'b-', linewidth=2, 
                label=f'ROC (AUC={self.test_metrics["ROC-AUC"]:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Precision-Recall Curve
        ax5 = axes[1, 1]
        precisions, recalls, _ = precision_recall_curve(y_test, y_test_proba)
        ax5.plot(recalls[:-1], precisions[:-1], 'b-', linewidth=2)
        # Mark optimal point
        y_pred_opt = self.predict(X_test, self.optimal_threshold)
        opt_recall = recall_score(y_test, y_pred_opt)
        opt_precision = precision_score(y_test, y_pred_opt)
        ax5.scatter([opt_recall], [opt_precision], color='red', s=150,
                   zorder=5, marker='*', label=f'Optimal (t={self.optimal_threshold:.3f})')
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title('Precision-Recall Curve', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Confusion Matrix
        ax6 = axes[1, 2]
        cm = confusion_matrix(y_test, y_test_pred)
        im = ax6.imshow(cm, cmap='Blues')
        ax6.set_xticks([0, 1])
        ax6.set_yticks([0, 1])
        ax6.set_xticklabels(['No', 'Yes'])
        ax6.set_yticklabels(['No', 'Yes'])
        ax6.set_xlabel('Predicted')
        ax6.set_ylabel('Actual')
        ax6.set_title(f'Confusion Matrix (t={self.optimal_threshold:.3f})', fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                ax6.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        plt.tight_layout()
        plt.show()
    
    def save(self, output_dir: str = "./models/rf") -> Dict[str, str]:
        """Save model and metadata."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        rf_ensemble_dict = {
            'model': self.model,
            'model_name': 'random_forest_brw',
            'feature_names': list(self.gini_importance['Feature']),
            'best_params': self.best_params,
            'best_cv_score': self.best_cv_score,
            'optimal_threshold': self.optimal_threshold,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'gini_importance': self.gini_importance.to_dict('records'),
            'perm_importance': self.perm_importance.to_dict('records'),
            'threshold_analysis': self.threshold_df.to_dict('records'),
            'training_date': timestamp
        }
        
        model_path = os.path.join(output_dir, f"rf_model_{timestamp}.pkl")
        ensemble_path = os.path.join(output_dir, f"rf_ensemble_{timestamp}.joblib")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        joblib.dump(rf_ensemble_dict, ensemble_path)
        
        print(f"\n💾 Saved:")
        print(f"   Model:    {model_path}")
        print(f"   Ensemble: {ensemble_path}")
        
        return {'model_path': model_path, 'ensemble_path': ensemble_path}


def train_rf_stage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_iter: int = 60,
    plot: bool = True,
    save: bool = True,
    output_dir: str = "./models/rf"
) -> Tuple[RFStage, Dict]:
    """
    Convenience function to train RF stage.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (BRW binary)
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels
    n_iter : int
        RandomizedSearchCV iterations
    plot : bool
        Whether to generate plots
    save : bool
        Whether to save model
    output_dir : str
        Output directory
        
    Returns
    -------
    stage : RFStage
        Trained stage
    metrics : dict
        Test metrics
    """
    stage = RFStage(n_iter=n_iter)
    stage.fit(X_train, y_train, X_test, y_test)
    
    if plot:
        stage.plot_evaluation(X_test, y_test)
    
    if save:
        stage.save(output_dir)
    
    print("\n🎯 RF Stage complete")
    return stage, stage.test_metrics
