"""
Logistic Regression Stage with Calibration & Tuning
====================================================
FIXED: Now takes GLOBAL_SPLIT directly, no internal splitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, fbeta_score,
    brier_score_loss, log_loss,
    precision_recall_curve, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    make_scorer
)
from typing import Dict, Tuple, Optional
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class CalibratedLRStage:
    """
    Advanced Logistic Regression Stage with tuning and calibration.
    
    FIXED: Uses CV for threshold optimization, saves FULL GLOBAL_SPLIT predictions.
    """
    
    def __init__(
        self,
        tune_hyperparameters: bool = True,
        calibration_method: str = 'auto',
        optimize_threshold: bool = True,
        cv_folds: int = 5,
        random_state: int = 42
    ):
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
        
        # Cache for FULL GLOBAL_SPLIT (set via save or fit_from_global_split)
        self._X_train_full = None
        self._y_train_full = None
        self._X_test_full = None
        self._y_test_full = None
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> 'CalibratedLRStage':
        """Fit tuned and calibrated LR model."""
        print("\n" + "="*80)
        print("🔧 CALIBRATED LR STAGE: Hyperparameter Tuning + Calibration")
        print("="*80)

        # Cache training data
        self._X_train_full = X_train.copy()
        self._y_train_full = y_train.copy()
        
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
        
        y_proba_uncal = self.model.predict_proba(X_train_scaled)[:, 1]
        brier_uncal = brier_score_loss(y_train, y_proba_uncal)
        ece_uncal = self._calculate_ece(y_train.values, y_proba_uncal)
        
        print(f"\n📊 Uncalibrated Metrics:")
        print(f"   Brier: {brier_uncal:.6f}")
        print(f"   ECE: {ece_uncal:.6f}")
        
        # Try both calibration methods
        calibrated_sigmoid = None
        calibrated_isotonic = None
        
        if self.calibration_method in ['sigmoid', 'auto']:
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
            ece_sigmoid = self._calculate_ece(y_train.values, y_proba_sigmoid)
        
        if self.calibration_method in ['isotonic', 'auto']:
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
            ece_isotonic = self._calculate_ece(y_train.values, y_proba_isotonic)
        
        # Select best
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
        else:
            self.calibrated_model = calibrated_isotonic
            best_brier = brier_isotonic
            best_ece = ece_isotonic
        
        print(f"\n🏆 Selected Calibration: {self.calibration_method}")
        print(f"   Brier: {brier_uncal:.6f} → {best_brier:.6f}")
        print(f"   ECE: {ece_uncal:.6f} → {best_ece:.6f}")
        
        self.calibration_metrics = {
            'brier_uncalibrated': brier_uncal,
            'brier_calibrated': best_brier,
            'ece_uncalibrated': ece_uncal,
            'ece_calibrated': best_ece
        }
        
        # ==============================
        # 3. THRESHOLD OPTIMIZATION (CV-BASED, NO SEPARATE VAL SET)
        # ==============================
        if self.optimize_threshold:
            print("\n🎯 Optimizing threshold via cross-validation...")
            self._optimize_threshold_cv(X_train_scaled, y_train)
        
        self.fitted = True
        return self
    
    def _optimize_threshold_cv(self, X_scaled: np.ndarray, y: pd.Series):
        """Optimize threshold using cross-validation (no separate val set needed)."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        # Get OOF predictions
        y_proba_oof = cross_val_predict(
            self.calibrated_model, X_scaled, y,
            cv=cv, method='predict_proba', n_jobs=-1
        )[:, 1]
        
        # Find optimal threshold
        best_f2 = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.05, 0.50, 0.01):
            y_pred = (y_proba_oof >= thresh).astype(int)
            if y_pred.sum() == 0:
                continue
            f2 = fbeta_score(y, y_pred, beta=2)
            if f2 > best_f2:
                best_f2 = f2
                best_thresh = thresh
        
        self.optimal_threshold = best_thresh
        print(f"   Optimal threshold (CV): {best_thresh:.4f}")
        print(f"   CV F2-score: {best_f2:.4f}")
    
    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted.")
        if threshold is None:
            threshold = self.optimal_threshold
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.calibrated_model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        plot: bool = True
    ) -> Dict[str, float]:
        """Evaluate on test set."""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        # Cache test data for save()
        self._X_test_full = X_test.copy()
        self._y_test_full = y_test.copy()
        
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
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        if plot:
            self._plot_evaluation(X_test, y_test)
        
        return self.metrics
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      n_bins: int = 10) -> float:
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
        y_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        ax1 = axes[0, 0]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        ax1.plot(prob_pred, prob_true, 'b-', marker='o', linewidth=2, markersize=8)
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Calibration Curve ({self.calibration_method})')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2 = axes[0, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={self.metrics["roc_auc"]:.3f}')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
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
        
        ax4 = axes[1, 1]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
        disp.plot(ax=ax4, cmap='Blues', values_format='d')
        ax4.set_title(f'Confusion Matrix (t={self.optimal_threshold:.3f})')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Model not fitted.")
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        return importance
    
    def save(self, output_dir: str = "./models/lr") -> Dict[str, str]:
        """Save model + FULL predictions for GLOBAL_SPLIT alignment."""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        if self._X_train_full is None or self._X_test_full is None:
            raise ValueError("No cached data. Call fit() and evaluate() first.")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Compute predictions on FULL cached data
        train_probs = self.predict_proba(self._X_train_full)
        test_probs = self.predict_proba(self._X_test_full)
        
        print(f"\n💾 Saving LR artifact...")
        print(f"   train_predictions: {len(train_probs)} samples")
        print(f"   test_predictions: {len(test_probs)} samples")
        
        artifact = {
            # Models
            'base_model': self.model,
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,

            # FULL predictions (aligned with GLOBAL_SPLIT)
            'train_predictions_calibrated': train_probs,
            'test_predictions_calibrated': test_probs,
            'train_predictions': train_probs,
            'test_predictions': test_probs,

            # FULL labels
            'train_labels': np.array(self._y_train_full),
            'test_labels': np.array(self._y_test_full),
            'y_train': np.array(self._y_train_full),
            'y_test': np.array(self._y_test_full),

            # Thresholds
            'optimal_threshold_calibrated': self.optimal_threshold,
            'optimal_threshold': self.optimal_threshold,

            # Diagnostics
            'best_params': self.best_params,
            'performance_metrics': self.metrics,
            'calibration_metrics': self.calibration_metrics,

            # Metadata
            'training_date': timestamp
        }

        path = os.path.join(output_dir, f"lr_calibrated_{timestamp}.joblib")
        joblib.dump(artifact, path)

        print(f"   ✅ Saved → {path}")
        return {'path': path}


def train_lr_stage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save: bool = True,
    output_dir: str = "./models/lr"
) -> Tuple[CalibratedLRStage, Dict[str, float]]:
    """
    Train calibrated LR stage using FULL GLOBAL_SPLIT data.
    
    NO INTERNAL SPLITTING - uses CV for threshold optimization.
    Saves predictions for the FULL train/test sets.
    """
    print("\n" + "="*80)
    print("🚀 TRAINING LR STAGE (FULL GLOBAL_SPLIT, NO INTERNAL SPLIT)")
    print("="*80)
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    
    # Train (uses CV for threshold optimization, no separate val set)
    stage = CalibratedLRStage(
        tune_hyperparameters=True,
        calibration_method='auto',
        optimize_threshold=True
    )
    stage.fit(X_train, y_train)
    
    # Evaluate on test
    metrics = stage.evaluate(X_test, y_test, plot=True)
    
    # Feature importance
    print("\n📊 Feature Importance (Top 10):")
    importance = stage.get_feature_importance()
    print(importance.head(10).to_string(index=False))
    
    # Save with FULL data
    if save:
        stage.save(output_dir)
    
    print("\n🎯 LR Stage complete")
    return stage, metrics


# Backward compatibility alias
train_calibrated_lr_stage = train_lr_stage