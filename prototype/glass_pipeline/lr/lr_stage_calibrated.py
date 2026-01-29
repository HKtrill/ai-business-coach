"""
Logistic Regression Stage with Calibration & Tuning
====================================================

Advanced LR stage with:
- Grid search hyperparameter tuning (F2-score optimized for recall)
- Probability calibration (Platt scaling / Isotonic regression)
- Threshold optimization
- Comprehensive evaluation and visualization

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, fbeta_score,
    brier_score_loss, log_loss,
    precision_recall_curve, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    make_scorer
)
from typing import Dict, Tuple
import pickle
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class CalibratedLRStage:
    """
    Advanced Logistic Regression Stage with tuning and calibration.
    
    Optimized for recall (F2-score) with well-calibrated probabilities.
    """
    
    def __init__(
        self,
        tune_hyperparameters: bool = True,
        calibration_method: str = 'auto',  # 'sigmoid', 'isotonic', or 'auto'
        optimize_threshold: bool = True,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize calibrated LR stage.
        
        Parameters
        ----------
        tune_hyperparameters : bool, default=True
            Whether to perform grid search
        calibration_method : str, default='auto'
            Calibration method ('sigmoid', 'isotonic', 'auto' = choose best)
        optimize_threshold : bool, default=True
            Whether to optimize decision threshold for F2-score
        cv_folds : int, default=5
            Number of CV folds for tuning/calibration
        random_state : int, default=42
            Random state
        """
        self.tune_hyperparameters = tune_hyperparameters
        self.calibration_method = calibration_method
        self.optimize_threshold = optimize_threshold
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.model = None
        self.calibrated_model = None
        self.scaler = None
        self.feature_names = None
        self.best_params = {}
        self.optimal_threshold = 0.5
        self.metrics = {}
        self.calibration_metrics = {}
        self.fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> 'CalibratedLRStage':
        """
        Fit tuned and calibrated LR model.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
            
        Returns
        -------
        self : CalibratedLRStage
            Fitted stage
        """
        print("\n" + "="*80)
        print("🔧 CALIBRATED LR STAGE: Hyperparameter Tuning + Calibration")
        print("="*80)
        
        self.feature_names = list(X_train.columns)
        
        # Scale features
        print("\n📏 Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # ==============================
        # 1. HYPERPARAMETER TUNING
        # ==============================
        if self.tune_hyperparameters:
            print("\n" + "="*60)
            print("🔍 HYPERPARAMETER TUNING (F2-Score: Recall-Favoring)")
            print("="*60)
            
            # F2 scorer: weights recall 2x more than precision
            f2_scorer = make_scorer(fbeta_score, beta=2)
            
            param_grid = {
                'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'class_weight': [
                    'balanced',
                    {0: 1, 1: 2},
                    {0: 1, 1: 3},
                    {0: 1, 1: 4},
                ],
                'max_iter': [1000]
            }
            
            n_combinations = (
                len(param_grid['C']) *
                len(param_grid['penalty']) *
                len(param_grid['class_weight'])
            )
            print(f"   Testing {n_combinations} combinations...")
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                random_state=self.random_state)
            
            grid_search = GridSearchCV(
                estimator=LogisticRegression(random_state=self.random_state),
                param_grid=param_grid,
                cv=cv,
                scoring=f2_scorer,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"\n✅ Best Parameters:")
            for param, value in self.best_params.items():
                print(f"   {param}: {value}")
            print(f"   Best CV F2-Score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            self.model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )
            self.model.fit(X_train_scaled, y_train)
            print("\n✅ Model trained with default parameters")
        
        # ==============================
        # 2. CALIBRATION
        # ==============================
        print("\n" + "="*60)
        print("🔧 PROBABILITY CALIBRATION")
        print("="*60)
        
        # Get uncalibrated metrics
        y_proba_uncal = self.model.predict_proba(X_train_scaled)[:, 1]
        brier_uncal = brier_score_loss(y_train, y_proba_uncal)
        logloss_uncal = log_loss(y_train, y_proba_uncal)
        ece_uncal = self._calculate_ece(y_train.values, y_proba_uncal)
        
        print(f"\n📊 Uncalibrated Metrics:")
        print(f"   Brier: {brier_uncal:.6f}")
        print(f"   Log Loss: {logloss_uncal:.6f}")
        print(f"   ECE: {ece_uncal:.6f}")
        
        if self.calibration_method in ['sigmoid', 'auto']:
            # Platt scaling
            print("\n   Fitting Platt Scaling (sigmoid)...")
            calibrated_sigmoid = CalibratedClassifierCV(
                estimator=self.model,
                method='sigmoid',
                cv=self.cv_folds,
                n_jobs=-1
            )
            calibrated_sigmoid.fit(X_train_scaled, y_train)
            y_proba_sigmoid = calibrated_sigmoid.predict_proba(X_train_scaled)[:, 1]
            brier_sigmoid = brier_score_loss(y_train, y_proba_sigmoid)
            logloss_sigmoid = log_loss(y_train, y_proba_sigmoid)
            ece_sigmoid = self._calculate_ece(y_train.values, y_proba_sigmoid)
        
        if self.calibration_method in ['isotonic', 'auto']:
            # Isotonic regression
            print("   Fitting Isotonic Regression...")
            calibrated_isotonic = CalibratedClassifierCV(
                estimator=self.model,
                method='isotonic',
                cv=self.cv_folds,
                n_jobs=-1
            )
            calibrated_isotonic.fit(X_train_scaled, y_train)
            y_proba_isotonic = calibrated_isotonic.predict_proba(X_train_scaled)[:, 1]
            brier_isotonic = brier_score_loss(y_train, y_proba_isotonic)
            logloss_isotonic = log_loss(y_train, y_proba_isotonic)
            ece_isotonic = self._calculate_ece(y_train.values, y_proba_isotonic)
        
        # Select best calibration method
        if self.calibration_method == 'auto':
            if brier_sigmoid <= brier_isotonic:
                self.calibrated_model = calibrated_sigmoid
                self.calibration_method = 'sigmoid'
                best_brier = brier_sigmoid
                best_ece = ece_sigmoid
            else:
                self.calibrated_model = calibrated_isotonic
                self.calibration_method = 'isotonic'
                best_brier = brier_isotonic
                best_ece = ece_isotonic
        elif self.calibration_method == 'sigmoid':
            self.calibrated_model = calibrated_sigmoid
            best_brier = brier_sigmoid
            best_ece = ece_sigmoid
        else:  # isotonic
            self.calibrated_model = calibrated_isotonic
            best_brier = brier_isotonic
            best_ece = ece_isotonic
        
        print(f"\n🏆 Selected Calibration: {self.calibration_method}")
        print(f"   Brier: {brier_uncal:.6f} → {best_brier:.6f} " +
              f"({(best_brier-brier_uncal)/brier_uncal*100:+.1f}%)")
        print(f"   ECE: {ece_uncal:.6f} → {best_ece:.6f} " +
              f"({(best_ece-ece_uncal)/ece_uncal*100:+.1f}%)")
        
        self.calibration_metrics = {
            'brier_uncalibrated': brier_uncal,
            'brier_calibrated': best_brier,
            'ece_uncalibrated': ece_uncal,
            'ece_calibrated': best_ece
        }
        
        self.fitted = True
        return self
    
    def optimize_threshold_on_validation(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """
        Optimize decision threshold on validation set.
        
        Parameters
        ----------
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation labels
            
        Returns
        -------
        optimal_threshold : float
            Threshold that maximizes F2-score
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not self.optimize_threshold:
            return 0.5
        
        print("\n🎯 Optimizing threshold on validation set...")
        
        X_val_scaled = self.scaler.transform(X_val)
        y_proba = self.calibrated_model.predict_proba(X_val_scaled)[:, 1]
        
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        
        # Calculate F2 for each threshold
        f2_scores = []
        for i, thresh in enumerate(thresholds):
            y_pred = (y_proba >= thresh).astype(int)
            if y_pred.sum() == 0:
                continue
            f2 = fbeta_score(y_val, y_pred, beta=2)
            f2_scores.append((thresh, f2))
        
        # Find optimal
        optimal_threshold, best_f2 = max(f2_scores, key=lambda x: x[1])
        self.optimal_threshold = optimal_threshold
        
        print(f"   Optimal threshold: {optimal_threshold:.4f}")
        print(f"   F2-score: {best_f2:.4f}")
        
        return optimal_threshold
    
    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        threshold : float, optional
            Decision threshold (uses optimal if None)
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted labels
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if threshold is None:
            threshold = self.optimal_threshold
        
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        y_proba : np.ndarray
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.calibrated_model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        plot: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        plot : bool, default=True
            Whether to generate plots
            
        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        print("\n" + "="*80)
        print(f"📈 EVALUATING CALIBRATED LR (threshold={self.optimal_threshold:.4f})")
        print("="*80)
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        self.metrics = {
            'threshold': self.optimal_threshold,
            'calibration': self.calibration_method,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'f2': fbeta_score(y_test, y_pred, beta=2),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'brier': brier_score_loss(y_test, y_proba),
            'ece': self._calculate_ece(y_test.values, y_proba)
        }
        
        print("\n📊 Test Performance:")
        for k, v in self.metrics.items():
            if k not in ['threshold', 'calibration']:
                print(f"   {k.upper()}: {v:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        if plot:
            self._plot_evaluation(X_test, y_test)
        
        return self.metrics
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                avg_confidence = y_prob[in_bin].mean()
                avg_accuracy = y_true[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        return ece
    
    def _plot_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Generate evaluation plots."""
        y_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Calibration curve
        ax1 = axes[0, 0]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        ax1.plot(prob_pred, prob_true, 'b-', marker='o', linewidth=2, markersize=8)
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Calibration Curve ({self.calibration_method})')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. ROC curve
        ax2 = axes[0, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={self.metrics["roc_auc"]:.3f}')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Precision-Recall curve
        ax3 = axes[1, 0]
        precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
        ax3.plot(recalls, precisions, 'b-', linewidth=2)
        ax3.scatter([self.metrics['recall']], [self.metrics['precision']],
                   color='red', s=150, zorder=5, marker='*',
                   label=f'Selected (t={self.optimal_threshold:.3f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Confusion matrix
        ax4 = axes[1, 1]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
        disp.plot(ax=ax4, cmap='Blues', values_format='d')
        ax4.set_title(f'Confusion Matrix (t={self.optimal_threshold:.3f})')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from base model."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get coefficients from base model (before calibration)
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance
    
    def save(self, output_dir: str = "./models/lr") -> Dict[str, str]:
        """Save model and metadata."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ensemble dict
        ensemble_dict = {
            'base_model': self.model,
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'optimal_threshold': self.optimal_threshold,
            'calibration_method': self.calibration_method,
            'performance_metrics': self.metrics,
            'calibration_metrics': self.calibration_metrics,
            'training_date': timestamp
        }
        
        ensemble_path = os.path.join(output_dir, f"lr_calibrated_{timestamp}.joblib")
        joblib.dump(ensemble_dict, ensemble_path)
        
        print(f"\n💾 Saved: {ensemble_path}")
        return {'ensemble_path': ensemble_path}


def train_calibrated_lr_stage(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    save: bool = True,
    output_dir: str = "./models/lr"
) -> Tuple[CalibratedLRStage, Dict[str, float]]:
    """
    Convenience function to train calibrated LR stage.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features (for threshold optimization)
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training labels
    y_val : pd.Series
        Validation labels
    y_test : pd.Series
        Test labels
    save : bool, default=True
        Whether to save model
    output_dir : str
        Output directory
        
    Returns
    -------
    stage : CalibratedLRStage
        Trained stage
    metrics : dict
        Test metrics
    """
    # Train
    stage = CalibratedLRStage(
        tune_hyperparameters=True,
        calibration_method='auto',
        optimize_threshold=True
    )
    stage.fit(X_train, y_train)
    
    # Optimize threshold on validation
    stage.optimize_threshold_on_validation(X_val, y_val)
    
    # Evaluate on test
    metrics = stage.evaluate(X_test, y_test, plot=True)
    
    # Feature importance
    print("\n📊 Feature Importance (Top 10):")
    importance = stage.get_feature_importance()
    print(importance.head(10).to_string(index=False))
    
    # Save
    if save:
        stage.save(output_dir)
    
    print("\n🎯 Calibrated LR Stage complete")
    return stage, metrics
