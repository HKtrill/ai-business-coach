# ============================================================
# GLASS-BRW: CLASSIFIER MODULE
# ============================================================
# Inference + diagnostics for the GLASS-BRW two-pass rule system
# ============================================================

from typing import List, Tuple
import numpy as np
import pandas as pd

from glass_brw.core.segment_builder import BankSegmentBuilder
from glass_brw.core.rule import (
    SelectedRule,
    SUBSCRIBE,
    NOT_SUBSCRIBE,
    ABSTAIN,
)


class GLASSBRWClassifier:
    """
    GLASS-BRW classifier: inference and diagnostics.
    
    Uses SelectedRule objects for two-pass prediction:
      Pass 1: Route NOT_SUBSCRIBE cases (precision-focused)
      Pass 2: Detect SUBSCRIBE cases (recall-focused)
    
    Attributes:
        pass1_rules: List[SelectedRule] - routing rules
        pass2_rules: List[SelectedRule] - detection rules
        segment_builder: BankSegmentBuilder for feature binning
        training_base_rate: Prior probability of SUBSCRIBE
    """

    def __init__(
        self,
        pass1_rules: List[SelectedRule],
        pass2_rules: List[SelectedRule],
        segment_builder: BankSegmentBuilder,
        training_base_rate: float,
    ):
        self.pass1_rules = pass1_rules
        self.pass2_rules = pass2_rules
        self.segment_builder = segment_builder
        self.training_base_rate = training_base_rate
        self.is_fitted = True

    # ================================================================
    # FACTORY
    # ================================================================

    @classmethod
    def from_selected_rules(
        cls,
        selected_rules: dict,
        segment_builder: BankSegmentBuilder,
        training_base_rate: float,
    ) -> 'GLASSBRWClassifier':
        """
        Create classifier from ILP selector output.
        
        Args:
            selected_rules: Dict with "pass1_rules" and "pass2_rules" lists
            segment_builder: Segment builder for feature assignment
            training_base_rate: Base rate of SUBSCRIBE in training data
            
        Returns:
            Configured GLASSBRWClassifier instance
        """
        pass1 = []
        for r in selected_rules.get("pass1_rules", []):
            if isinstance(r, SelectedRule):
                pass1.append(r)
            else:
                pass1.append(SelectedRule.from_evaluated(r, pass_assignment="pass1"))
        
        pass2 = []
        for r in selected_rules.get("pass2_rules", []):
            if isinstance(r, SelectedRule):
                pass2.append(r)
            else:
                pass2.append(SelectedRule.from_evaluated(r, pass_assignment="pass2"))
        
        pass1_sorted = sorted(pass1, key=lambda r: r.precision, reverse=True)
        pass2_sorted = sorted(pass2, key=lambda r: r.recall, reverse=True)
        
        return cls(
            pass1_rules=pass1_sorted,
            pass2_rules=pass2_sorted,
            segment_builder=segment_builder,
            training_base_rate=training_base_rate,
        )

    # ================================================================
    # INFERENCE
    # ================================================================

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate predictions using sequential rule application.
        
        Args:
            X: Features DataFrame (raw or segmented)
            
        Returns:
            Tuple of:
                - preds: np.ndarray (0, 1, or -1 for abstain)
                - conf: np.ndarray of confidence scores (rule precision)
                - decisions: List[str] ("pass1", "pass2", or "abstain")
        """
        X = pd.DataFrame(X)
        segments = self.segment_builder.assign_segments(X)
        n = len(X)

        preds = np.full(n, ABSTAIN, dtype=int)
        conf = np.zeros(n)
        decisions = ["abstain"] * n

        for i in range(n):
            row = segments.iloc[i]

            # Pass 1: Route NOT_SUBSCRIBE cases
            for rule in self.pass1_rules:
                if rule.matches(row):
                    preds[i] = NOT_SUBSCRIBE
                    conf[i] = rule.precision
                    decisions[i] = "pass1"
                    break

            if decisions[i] == "pass1":
                continue

            # Pass 2: Detect SUBSCRIBE cases
            for rule in self.pass2_rules:
                if rule.matches(row):
                    preds[i] = SUBSCRIBE
                    conf[i] = rule.precision
                    decisions[i] = "pass2"
                    break

        return preds, conf, decisions

    def predict_proba(
        self,
        X: pd.DataFrame,
        base_rate: float = None,
    ) -> np.ndarray:
        """
        Generate probabilistic predictions.
        
        Args:
            X: Features DataFrame
            base_rate: Override base rate (defaults to training_base_rate)
            
        Returns:
            Array of shape (n, 2) with [P(NOT_SUBSCRIBE), P(SUBSCRIBE)]
        """
        preds, conf, decisions = self.predict(X)
        n = len(preds)

        if base_rate is None:
            base_rate = self.training_base_rate

        probas = np.full((n, 2), [1 - base_rate, base_rate], dtype=float)

        for i in range(n):
            if decisions[i] == "pass1":
                probas[i, 1] = base_rate * (1 - conf[i])
                probas[i, 0] = 1 - probas[i, 1]
            elif decisions[i] == "pass2":
                probas[i, 1] = base_rate + (1 - base_rate) * conf[i]
                probas[i, 0] = 1 - probas[i, 1]

        return probas

    # ================================================================
    # RULE SUMMARY
    # ================================================================

    def get_rule_summary(self) -> List[dict]:
        """Generate human-readable summary of rules."""
        summary = []

        for r in self.pass1_rules:
            summary.append({
                "rule_id": r.rule_id,
                "pass": "Pass 1",
                "class": "NOT_SUBSCRIBE (0)",
                "segment": r.segment_str,
                "precision": r.precision,
                "recall": r.recall,
                "coverage": r.coverage,
                "complexity": r.complexity,
            })

        for r in self.pass2_rules:
            summary.append({
                "rule_id": r.rule_id,
                "pass": "Pass 2",
                "class": "SUBSCRIBE (1)",
                "segment": r.segment_str,
                "precision": r.precision,
                "recall": r.recall,
                "coverage": r.coverage,
                "complexity": r.complexity,
            })

        return summary

    def get_rules_dataframe(self) -> pd.DataFrame:
        """Return rules as a pandas DataFrame for analysis."""
        return pd.DataFrame(self.get_rule_summary())

    # ================================================================
    # DIAGNOSTICS
    # ================================================================

    def analyze_rule_false_negatives(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_pred: np.ndarray,
        test_decisions: List[str],
    ) -> pd.DataFrame:
        """
        Analyze which rules are causing false negatives.
        
        Args:
            X_test: Test features (binned BRW features)
            y_test: Test labels
            test_pred: Test predictions
            test_decisions: Test pass decisions
            
        Returns:
            DataFrame with rule-level FN statistics
        """
        print("\n" + "=" * 80)
        print("🔎 RULE-LEVEL FALSE NEGATIVE ANALYSIS (TEST SET)")
        print("=" * 80)

        df = X_test.copy()
        df["y_true"] = y_test.values
        df["pred"] = test_pred
        df["pass_decision"] = test_decisions

        rule_stats = []

        # Pass 1 rules — any subscriber matched is a false negative
        for rule in self.pass1_rules:
            mask = df.apply(lambda r, _rule=rule: _rule.matches(r), axis=1)
            total = mask.sum()
            fn = (mask & (df["y_true"] == SUBSCRIBE)).sum()
            tn = (mask & (df["y_true"] == NOT_SUBSCRIBE)).sum()

            rule_stats.append({
                "pass": "PASS 1",
                "rule_id": rule.rule_id,
                "segment": rule.segment_str,
                "fires": int(total),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "fn_rate": fn / total if total > 0 else 0.0,
                "precision_train": rule.precision,
                "recall_train": rule.recall,
            })

        # Pass 2 rules
        for rule in self.pass2_rules:
            mask = df.apply(lambda r, _rule=rule: _rule.matches(r), axis=1)
            total = mask.sum()
            fn = (mask & (df["y_true"] == SUBSCRIBE) & (df["pred"] != SUBSCRIBE)).sum()
            tp = (mask & (df["y_true"] == SUBSCRIBE) & (df["pred"] == SUBSCRIBE)).sum()

            rule_stats.append({
                "pass": "PASS 2",
                "rule_id": rule.rule_id,
                "segment": rule.segment_str,
                "fires": int(total),
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "fn_rate": fn / total if total > 0 else 0.0,
                "precision_train": rule.precision,
                "recall_train": rule.recall,
            })

        rule_df = pd.DataFrame(rule_stats)
        rule_df = rule_df.sort_values(
            by=["pass", "false_negatives", "fires"],
            ascending=[True, False, False],
        )

        print(rule_df.to_string(index=False))
        return rule_df

    def analyze_abstain_samples(
        self,
        y_test: pd.Series,
        test_pred: np.ndarray,
    ):
        """Analyze abstain sample composition."""
        print("\n" + "=" * 80)
        print("🚨 ABSTAIN SAMPLE ANALYSIS (TEST SET)")
        print("=" * 80)

        abstain_mask = test_pred == ABSTAIN
        n_abstain = abstain_mask.sum()
        subs = (y_test[abstain_mask] == SUBSCRIBE).sum()
        notsubs = (y_test[abstain_mask] == NOT_SUBSCRIBE).sum()

        print(f"Total abstained samples: {n_abstain:,}")
        print(f"  Subscribers (FN risk): {subs:,}")
        print(f"  Not-subscribers:       {notsubs:,}")

        if n_abstain > 0:
            print(f"  Subscribe rate among abstains: {subs / n_abstain:.2%}")

        print(f"\nBase subscribe rate (test): {y_test.mean():.2%}")

    def print_sanity_checks(
        self,
        y_test: pd.Series,
        test_pred: np.ndarray,
    ):
        """Print sanity check counts."""
        print("\n" + "=" * 80)
        print("🧮 SANITY CHECK COUNTS")
        print("=" * 80)

        print(f"Total test samples: {len(y_test):,}")
        print(f"Predicted SUBSCRIBE: {(test_pred == SUBSCRIBE).sum():,}")
        print(f"Predicted NOT_SUB:   {(test_pred == NOT_SUBSCRIBE).sum():,}")
        print(f"Abstained:           {(test_pred == ABSTAIN).sum():,}")

    def run_full_diagnostics(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_pred: np.ndarray,
        test_decisions: List[str],
    ) -> pd.DataFrame:
        """
        Run all diagnostic analyses.
        
        Returns:
            DataFrame with rule-level FN statistics
        """
        rule_df = self.analyze_rule_false_negatives(
            X_test, y_test, test_pred, test_decisions
        )
        self.analyze_abstain_samples(y_test, test_pred)
        self.print_sanity_checks(y_test, test_pred)
        return rule_df

    # ================================================================
    # REPR
    # ================================================================

    def __repr__(self):
        return (
            f"GLASSBRWClassifier("
            f"pass1={len(self.pass1_rules)}, "
            f"pass2={len(self.pass2_rules)}, "
            f"base_rate={self.training_base_rate:.3f})"
        )