# ============================================================
# GLASS-BRW: RF INTEGRATION MODULE
# ============================================================
# Random Forest confidence and alignment metrics for rules
# ============================================================
import numpy as np
import pandas as pd


class RFIntegration:
    """Compute Random Forest confidence metrics for rule segments."""
    
    def compute_rf_metrics(
        self,
        mask: pd.Series,
        rf_proba: np.ndarray
    ) -> tuple:
        """
        Compute RF confidence and alignment for rule segment.
        
        Metrics:
            RF Confidence: Mean distance from decision boundary (0.5)
            RF Alignment: Fraction of samples with high confidence (>0.20)
        
        Args:
            mask: Boolean mask of matched samples
            rf_proba: RF probability estimates for SUBSCRIBE class
            
        Returns:
            (rf_confidence, rf_alignment) tuple
        """
        rf_conf_in_segment = np.abs(rf_proba[mask] - 0.5)
        rf_confidence = rf_conf_in_segment.mean()
        rf_alignment = (rf_conf_in_segment > 0.20).mean()
        
        return rf_confidence, rf_alignment
    
    def get_rf_predictions(
        self,
        rf_model,
        X_val: pd.DataFrame
    ) -> np.ndarray:
        """
        Get RF probability predictions for validation data.
        
        Args:
            rf_model: Trained RandomForestClassifier
            X_val: Validation features
            
        Returns:
            Array of probabilities for SUBSCRIBE class
        """
        if rf_model is None:
            return None
        
        return rf_model.predict_proba(X_val)[:, 1]