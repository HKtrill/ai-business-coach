"""
Logistic Regression Stage (Baseline)
=====================================

Basic logistic regression model with standard preprocessing.
Serves as Stage 1 in the Glass Cascade pipeline.

Author: Glass Pipeline Team
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple
import pickle
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class LRStage:
    """
    Logistic Regression Stage for Glass Cascade Pipeline.
    
    Implements fit/predict/abstain contract for cascade integration.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        solver: str = 'liblinear',
        class_weight: str = 'balanced',
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize LR stage.
        
        Parameters
        ----------
        C : float, default=1.0
            Inverse of regularization strength
        penalty : str, default='l2'
            Regularization penalty
        solver : str, default='liblinear'
            Solver algorithm
        class_weight : str, default='balanced'
            Class weights for imbalanced data
        max_iter : int, default=1000
            Maximum iterations
        random_state : int, default=42
            Random state for reproducibility
        """
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}
        self.fitted = False
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> 'LRStage':
        """
        Fit logistic regression model.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
            
        Returns
        -------
        self : LRStage
            Fitted stage
        """
        print("\n" + "="*80)
        print("🤖 LOGISTIC REGRESSION STAGE 1: Baseline")
        print("="*80)
        print(f"\nTrain: {X_train.shape} | Positive rate: {y_train.mean():.4f}")
        print(f"Features: {list(X_train.columns)}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Scale features
        print("\n📏 Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        print("✅ Features scaled")
        
        # Train model
        print("\n🤖 Training Logistic Regression...")
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Training metrics
        y_train_pred = self.model.predict(X_train_scaled)
        y_train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        }
        
        print("\n📊 Training Performance:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted labels
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        y_proba : np.ndarray
            Predicted probabilities for positive class
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
            
        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        print("\n" + "="*80)
        print("📊 EVALUATING LR STAGE ON TEST SET")
        print("="*80)
        
        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        print("\n📊 Test Performance:")
        for k, v in self.metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficients).
        
        Returns
        -------
        importance : pd.DataFrame
            Sorted feature importance
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance
    
    def save(self, output_dir: str = "./models/lr") -> Dict[str, str]:
        """
        Save baseline LR model and metadata.
        NOTE: Baseline LR does NOT persist predictions or calibration.
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_dict = {
            'model': self.model,
            'model_name': 'logistic_regression_baseline',
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'hyperparameters': {
                'C': self.C,
                'penalty': self.penalty,
                'solver': self.solver,
                'class_weight': self.class_weight
            },
            'performance_metrics': self.metrics,
            'training_date': timestamp
        }

        path = os.path.join(output_dir, f"lr_baseline_{timestamp}.joblib")
        joblib.dump(ensemble_dict, path)

        print(f"\n💾 Saved baseline LR: {path}")
        return {'ensemble_path': path}


    @staticmethod
    def load(ensemble_path: str) -> 'LRStage':
        """
        Load saved model.
        
        Parameters
        ----------
        ensemble_path : str
            Path to saved ensemble dict
            
        Returns
        -------
        stage : LRStage
            Loaded stage
        """
        ensemble_dict = joblib.load(ensemble_path)
        
        # Recreate stage
        hyperparams = ensemble_dict['hyperparameters']
        stage = LRStage(
            C=hyperparams['C'],
            penalty=hyperparams['penalty'],
            solver=hyperparams['solver'],
            class_weight=hyperparams['class_weight']
        )
        
        # Restore fitted components
        stage.model = ensemble_dict['model']
        stage.scaler = ensemble_dict['scaler']
        stage.feature_names = ensemble_dict['feature_names']
        stage.metrics = ensemble_dict['performance_metrics']
        stage.fitted = True
        
        print(f"📂 Loaded model from: {ensemble_path}")
        return stage


def train_lr_stage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save: bool = True,
    output_dir: str = "./models/lr"
) -> Tuple[LRStage, Dict[str, float]]:
    """
    Convenience function to train and evaluate LR stage.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels
    save : bool, default=True
        Whether to save model
    output_dir : str
        Output directory
        
    Returns
    -------
    stage : LRStage
        Trained stage
    metrics : dict
        Test metrics
    """
    # Train
    stage = LRStage()
    stage.fit(X_train, y_train)
    
    # Evaluate
    metrics = stage.evaluate(X_test, y_test)
    
    # Feature importance
    print("\n📊 Feature Importance (Top 10):")
    importance = stage.get_feature_importance()
    print(importance.head(10).to_string(index=False))
    
    # Save
    if save:
        stage.save(output_dir)
    
    print("\n🎯 LR Stage 1 complete")
    return stage, metrics
