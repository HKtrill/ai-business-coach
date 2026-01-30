# ============================================================
# GLASS-BRW: MODEL SAVER MODULE
# ============================================================
# Save GLASS-BRW model, predictions, and metadata
# ============================================================
import joblib
import numpy as np
from datetime import datetime
from glass_brw.segment_builder import BankSegmentBuilder


class ModelSaver:
    """Save GLASS-BRW model artifacts and metadata."""
    
    def __init__(self, glass_model, segment_builder_class=BankSegmentBuilder):
        """
        Initialize model saver.
        
        Args:
            glass_model: Fitted GLASS_BRW instance
            segment_builder_class: SegmentBuilder class for feature names
        """
        self.glass = glass_model
        self.segment_builder_class = segment_builder_class
    
    def save_model(
        self,
        train_out: dict,
        test_out: dict,
        train_proba: np.ndarray,
        test_proba: np.ndarray,
        global_split: dict,
        output_path: str = None,
        y_test=None,
    ) -> str:
        """
        Save complete GLASS-BRW bundle.
        
        Args:
            train_out: Training predictions dict
            test_out: Test predictions dict
            train_proba: Training probabilities
            test_proba: Test probabilities
            global_split: Dict with train_idx, test_idx
            output_path: Custom output path (default: auto-generated)
            y_test: Test labels (for sample predictions)
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = f"./models/glass_brw/glass_brw_{timestamp}.joblib"
        
        # Build bundle
        glass_bundle = {
            "features": self.segment_builder_class.SEGMENT_FEATURES,
            "train_idx": global_split.get("train_idx"),
            "test_idx": global_split.get("test_idx"),
            # Train outputs
            "train_pred": train_out["pred"],
            "train_confidence": train_out.get("confidence"),
            "train_covered": train_out["covered"],
            "train_abstained": train_out["abstained"],
            "train_decisions": train_out["decisions"],
            "train_proba": train_proba,
            # Test outputs
            "test_pred": test_out["pred"],
            "test_confidence": test_out.get("confidence"),
            "test_covered": test_out["covered"],
            "test_abstained": test_out["abstained"],
            "test_decisions": test_out["decisions"],
            "test_proba": test_proba,
            # Model metadata
            "rules": self.glass.get_rule_summary(),
            "config": self._build_config(),
            "timestamp": timestamp,
        }
        
        # Save
        joblib.dump(glass_bundle, output_path)
        
        # Print summary
        self._print_save_summary(output_path, glass_bundle, y_test)
        
        return output_path
    
    def _build_config(self) -> dict:
        """Build configuration dict from glass model."""
        return {
            "mode": self.glass.mode,
            "execution_type": "sequential_dual_focus",
            "support_pass1": self.glass.min_support_pass1,
            "support_pass2": self.glass.min_support_pass2,
            "jaccard_overlap_max": getattr(self.glass, "max_jaccard_overlap", None),
            "pass1_focus": "precision",
            "pass2_focus": "recall_with_diversity",
            "pass1_min_precision": self.glass.min_precision_not_subscribe,
            "pass1_max_leakage_rate": self.glass.max_subscriber_leakage_rate,
            "pass2_min_precision": self.glass.min_precision_subscribe,
            "pass2_min_recall": self.glass.min_recall_subscribe,
            "max_complexity": self.glass.max_complexity,
            "min_novelty_ratio_pass1": self.glass.min_novelty_ratio_pass1,
            "min_novelty_ratio_pass2": self.glass.min_novelty_ratio_pass2,
            "enable_novelty_constraints": self.glass.enable_novelty_constraints,
        }
    
    def _print_save_summary(self, output_path: str, bundle: dict, y_test=None):
        """Print save summary and verification."""
        print("\n" + "="*80)
        print("💾 SAVING ARTIFACTS")
        print("="*80)
        print(f"\n✅ SAVED: {output_path}")
        
        print(f"\n🔍 VERIFICATION:")
        print(f"   train_proba shape: {bundle['train_proba'].shape}")
        print(f"   test_proba shape: {bundle['test_proba'].shape}")
        print(f"   Rules: {len(bundle['rules'])}")
        
        # Sample predictions
        if y_test is not None and len(bundle['test_proba']) > 0:
            self._print_sample_predictions(bundle, y_test)
    
    def _print_sample_predictions(self, bundle: dict, y_test, n_samples: int = 10):
        """Print sample predictions for verification."""
        print(f"\n📊 Sample predictions (first {n_samples} test samples):")
        print(f"{'Sample':<8} {'Pass':<12} {'P(NOT_SUB)':<12} {'P(SUB)':<12} {'Pred':<12} {'True':<6}")
        print("-" * 70)
        
        for i in range(min(n_samples, len(bundle['test_proba']))):
            p_not_sub = bundle['test_proba'][i, 0]
            p_sub = bundle['test_proba'][i, 1]
            pred = bundle['test_pred'][i]
            true = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            pass_dec = bundle['test_decisions'][i]
            
            pred_str = "NOT_SUB(0)" if pred == 0 else ("SUB(1)" if pred == 1 else "ABSTAIN(-1)")
            true_str = "SUB(1)" if true == 1 else "NOT(0)"
            correct = "✓" if (pred == true or pred == -1) else "✗"
            
            print(f"{i:<8} {pass_dec:<12} {p_not_sub:<12.6f} {p_sub:<12.6f} {pred_str:<12} {true_str:<6} {correct}")
    
    def print_final_summary(
        self,
        test_out: dict,
        pass1_blocked_subscribers: int,
        total_subscribers: int,
        overall_recall: float,
        covered_precision: float,
        covered_recall: float,
    ):
        """Print final model summary."""
        print("\n" + "="*80)
        print("✅ GLASS-BRW COMPLETE")
        print("="*80)
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Architecture: Depth-Staged Beam Search + ILP Selection")
        print(f"   Pass 1: {len(self.glass.pass1_rules)} precision-focused filters")
        print(f"   Pass 2: {len(self.glass.pass2_rules)} recall-focused detectors")
        print(f"   Test Coverage: {test_out['covered'].mean():.1%}")
        print(f"   Subscriber Leakage (Pass 1): {pass1_blocked_subscribers/total_subscribers:.1%}")
        print(f"   Overall Subscriber Recall: {overall_recall:.1%}")
        print(f"   Covered Precision: {covered_precision:.3f}")
        print(f"   Covered Recall: {covered_recall:.3f}")
        print("="*80)