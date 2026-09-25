# ============================================================
# GLASS ROUTER: MODEL SAVER MODULE
# ============================================================
# Save GLASS Router model, predictions, and metadata
# Works with models containing SelectedRule objects
# ============================================================

from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import joblib
import numpy as np

from glass_pipeline.glass_router.core.segment_builder import BankSegmentBuilder
from glass_pipeline.glass_router.core.rule import SelectedRule


class ModelSaver:
    """
    Save GLASS Router model artifacts and metadata.
    
    Works with fitted models that contain SelectedRule objects
    in pass1_rules and pass2_rules attributes.
    """
    
    def __init__(
        self, 
        glass_model, 
        segment_builder_class=BankSegmentBuilder
    ):
        """
        Initialize model saver.
        
        Args:
            glass_model: Fitted GlassRouterPipeline instance
            segment_builder_class: SegmentBuilder class for feature names
        """
        self.glass = glass_model
        self.segment_builder_class = segment_builder_class
    
    def save_model(
        self,
        train_out: Optional[Dict[str, Any]],
        test_out: Dict[str, Any],
        train_proba: Optional[np.ndarray],
        test_proba: np.ndarray,
        global_split: Dict[str, Any],
        output_path: Optional[str] = None,
        y_test=None,
        train_outputs_oof: bool = False,
        split_fingerprint: Optional[str] = None,
        oof=None,
        train_insample_proba: Optional[np.ndarray] = None,
    ) -> str:
        """
        Save complete GLASS Router bundle.
        
        Args:
            train_out: Training predictions dict
            test_out: Test predictions dict
            train_proba: Training probabilities
            test_proba: Test probabilities
            global_split: Dict with train_idx, test_idx
            output_path: Custom output path (default: auto-generated)
            y_test: Test labels (for sample predictions)
            train_outputs_oof: True only if train_out / train_proba were produced
                out-of-fold (cross-fitted). Default False = in-sample routing;
                downstream stages must not treat them as honest features.
            split_fingerprint: Fingerprint of GLOBAL_SPLIT from the shared split
                helper (the same one the XGBoost/EBM artifacts use). Falls back to
                global_split['split_fingerprint']. Required for split-identity checks.
            oof: Stage2OOF from crossfit_stage2(). When given, the train-side block
                is taken from it (OOF predictions, confidence, decisions, proba) and
                train_outputs_oof / oof_protocol / train_fold_id are set from it.
                Pass train_out=None and train_proba=None in that case.
            train_insample_proba: optional in-sample train proba from the
                full-train model. Stored for diagnostics only (optimism check);
                Stage 4 must read train_proba, never this.
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Train-side block: OOF (cross-fitted) or in-sample
        oof_protocol, train_fold_id = None, None
        if oof is not None:
            if train_out is not None or train_proba is not None:
                raise ValueError("Pass either oof= or train_out/train_proba, not both.")
            train_out = oof.glass_train_out()
            train_proba = oof.glass_proba
            train_outputs_oof = True
            oof_protocol = oof.protocol
            train_fold_id = oof.fold_id
        elif train_outputs_oof:
            raise ValueError(
                "train_outputs_oof=True requires oof= (a Stage2OOF), so the fold "
                "assignment and protocol are saved with the outputs."
            )
        if train_out is None or train_proba is None:
            raise ValueError("Train-side outputs missing: pass oof= or train_out + train_proba.")
        
        if output_path is None:
            output_path = f"./models/glass_router/glass_router_{timestamp}.joblib"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Split provenance (PR 34 contract: split_fingerprint + index_test)
        if split_fingerprint is None:
            split_fingerprint = global_split.get("split_fingerprint")
        if split_fingerprint is None:
            raise ValueError(
                "split_fingerprint is required: pass it explicitly or include it in "
                "global_split. Artifacts without it cannot be split-verified."
            )
        index_test = global_split.get("test_idx")
        index_train = global_split.get("train_idx")
        # GLOBAL_SPLIT from GlobalSplitManager carries the frames, not index arrays
        if index_test is None and "X_test" in global_split:
            index_test = global_split["X_test"].index
        if index_train is None and "X_train" in global_split:
            index_train = global_split["X_train"].index
        if index_test is None or index_train is None:
            raise ValueError("global_split must provide train_idx/test_idx or X_train/X_test.")
        if len(index_test) != len(test_out["pred"]):
            raise ValueError(
                f"index_test length {len(index_test)} != test outputs "
                f"{len(test_out['pred'])} — outputs are not aligned to GLOBAL_SPLIT."
            )
        if len(index_train) != len(train_out["pred"]):
            raise ValueError(
                f"index_train length {len(index_train)} != train outputs "
                f"{len(train_out['pred'])} — outputs are not aligned to GLOBAL_SPLIT."
            )

        # Build bundle
        glass_bundle = {
            "features": self.segment_builder_class.SEGMENT_FEATURES,
            "train_idx": index_train,
            "test_idx": index_test,
            "split_fingerprint": split_fingerprint,
            "index_test": index_test,
            # Train outputs
            "train_pred": train_out["pred"],
            "train_confidence": train_out.get("confidence"),
            "train_covered": train_out["covered"],
            "train_abstained": train_out["abstained"],
            "train_decisions": train_out["decisions"],
            "train_proba": train_proba,
            "train_outputs_oof": bool(train_outputs_oof),
            "train_fold_id": train_fold_id,
            "oof_protocol": oof_protocol,
            "train_insample_proba_diagnostic": train_insample_proba,
            # Test outputs
            "test_pred": test_out["pred"],
            "test_confidence": test_out.get("confidence"),
            "test_covered": test_out["covered"],
            "test_abstained": test_out["abstained"],
            "test_decisions": test_out["decisions"],
            "test_proba": test_proba,
            # Model metadata
            "rules": self._serialize_rules(),
            "config": self._build_config(),
            "config_full": (
                self.glass.config.to_dict() if hasattr(self.glass, "config") else None
            ),
            "training_base_rate": getattr(self.glass, "training_base_rate", None),
            "fit_report": getattr(self.glass, "fit_report_", None),
            "timestamp": timestamp,
        }
        
        # Save
        joblib.dump(glass_bundle, output_path)
        
        # Print summary
        self._print_save_summary(output_path, glass_bundle, y_test)
        
        return output_path
    
    def _serialize_rules(self) -> List[Dict[str, Any]]:
        """Serialize SelectedRule objects to dicts for storage."""
        rules = []
        
        # Pass 1 rules
        pass1_rules: List[SelectedRule] = self.glass.pass1_rules
        for r in pass1_rules:
            rules.append({
                "rule_id": r.rule_id,
                "pass": "Pass 1",
                "pass_assignment": r.pass_assignment,
                "class": "NOT_SUBSCRIBE (0)",
                "predicted_class": r.predicted_class,
                "segment": list(r.segment),
                "segment_str": r.segment_str if hasattr(r, 'segment_str') else str(r.segment),
                "precision": r.precision,
                "recall": r.recall,
                "coverage": r.coverage,
                "complexity": r.complexity,
                "rf_alignment": r.rf_alignment,
            })
        
        # Pass 2 rules
        pass2_rules: List[SelectedRule] = self.glass.pass2_rules
        for r in pass2_rules:
            rules.append({
                "rule_id": r.rule_id,
                "pass": "Pass 2",
                "pass_assignment": r.pass_assignment,
                "class": "SUBSCRIBE (1)",
                "predicted_class": r.predicted_class,
                "segment": list(r.segment),
                "segment_str": r.segment_str if hasattr(r, 'segment_str') else str(r.segment),
                "precision": r.precision,
                "recall": r.recall,
                "coverage": r.coverage,
                "complexity": r.complexity,
                "rf_alignment": r.rf_alignment,
            })
        
        return rules
    
    def _build_config(self) -> Dict[str, Any]:
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
    
    def _print_save_summary(
        self, 
        output_path: str, 
        bundle: Dict[str, Any], 
        y_test=None
    ):
        """Print save summary and verification."""
        print("\n" + "="*80)
        print("💾 SAVING ARTIFACTS")
        print("="*80)
        print(f"\n✅ SAVED: {output_path}")
        
        print(f"\n🔍 VERIFICATION:")
        print(f"   train_proba shape: {bundle['train_proba'].shape}")
        print(f"   test_proba shape: {bundle['test_proba'].shape}")
        print(f"   Rules: {len(bundle['rules'])}")
        print(f"   training_base_rate: {bundle['training_base_rate']}")
        print(f"   split_fingerprint: {bundle['split_fingerprint']}")
        print(f"   index_test: {len(bundle['index_test']):,} rows")
        print(f"   train_outputs_oof: {bundle['train_outputs_oof']}"
              + ("" if bundle['train_outputs_oof'] else "  ⚠️  train-side outputs are IN-SAMPLE"))
        if bundle.get("oof_protocol"):
            print(f"   fold_fingerprint: {bundle['oof_protocol'].get('fold_fingerprint')}")
        
        # Sample predictions
        if y_test is not None and len(bundle['test_proba']) > 0:
            self._print_sample_predictions(bundle, y_test)
    
    def _print_sample_predictions(
        self, 
        bundle: Dict[str, Any], 
        y_test, 
        n_samples: int = 10
    ):
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
        test_out: Dict[str, Any],
        pass1_blocked_subscribers: int,
        total_subscribers: int,
        overall_recall: float,
        covered_precision: float,
        covered_recall: float,
    ):
        """Print final model summary."""
        print("\n" + "="*80)
        print("✅ GLASS ROUTER COMPLETE")
        print("="*80)
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Architecture: Depth-Staged Beam Search + ILP Selection")
        print(f"   Pass 1: {len(self.glass.pass1_rules)} precision-focused filters")
        print(f"   Pass 2: {len(self.glass.pass2_rules)} recall-focused detectors")
        print(f"   Test Coverage: {test_out['covered'].mean():.1%}")
        if total_subscribers > 0:
            print(f"   Subscriber Leakage (Pass 1): {pass1_blocked_subscribers/total_subscribers:.1%}")
        print(f"   Overall Subscriber Recall: {overall_recall:.1%}")
        print(f"   Covered Precision: {covered_precision:.3f}")
        print(f"   Covered Recall: {covered_recall:.3f}")
        print("="*80)