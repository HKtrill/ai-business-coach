# ============================================================
# GLASS-BRW: CLASSIFIER MODULE
# ============================================================
# Inference-only classifier using SelectedRule objects
# ============================================================

from typing import List, Tuple
import numpy as np
import pandas as pd

from glass_brw.segment_builder import BankSegmentBuilder
from glass_brw.core.rule import (
    SelectedRule,
    EvaluatedRule,
    SUBSCRIBE,
    NOT_SUBSCRIBE,
    ABSTAIN,
)


class GLASSBRWClassifier:
    """
    Stage-2 GLASS-BRW classifier (INFERENCE ONLY).
    
    Uses SelectedRule objects for prediction.
    Rules are applied sequentially: Pass 1 first, then Pass 2.
    
    Attributes:
        pass1_rules: List[SelectedRule] - routing rules (predict NOT_SUBSCRIBE)
        pass2_rules: List[SelectedRule] - detection rules (predict SUBSCRIBE)
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
        """
        Initialize classifier with selected rules.
        
        Args:
            pass1_rules: Routing rules (sorted by precision, descending)
            pass2_rules: Detection rules (sorted by recall, descending)
            segment_builder: Segment builder for feature assignment
            training_base_rate: Base rate of SUBSCRIBE in training data
        """
        self.pass1_rules = pass1_rules
        self.pass2_rules = pass2_rules
        self.segment_builder = segment_builder
        self.training_base_rate = training_base_rate
        self.is_fitted = True

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
        # Convert EvaluatedRule to SelectedRule if needed
        pass1 = []
        for r in selected_rules.get("pass1_rules", []):
            if isinstance(r, SelectedRule):
                pass1.append(r)
            elif isinstance(r, EvaluatedRule):
                pass1.append(SelectedRule.from_evaluated(r, pass_assignment="pass1"))
            else:
                # Legacy Rule object - duck type it
                pass1.append(SelectedRule.from_evaluated(r, pass_assignment="pass1"))
        
        pass2 = []
        for r in selected_rules.get("pass2_rules", []):
            if isinstance(r, SelectedRule):
                pass2.append(r)
            elif isinstance(r, EvaluatedRule):
                pass2.append(SelectedRule.from_evaluated(r, pass_assignment="pass2"))
            else:
                pass2.append(SelectedRule.from_evaluated(r, pass_assignment="pass2"))
        
        # Sort rules
        pass1_sorted = sorted(pass1, key=lambda r: r.precision, reverse=True)
        pass2_sorted = sorted(pass2, key=lambda r: r.recall, reverse=True)
        
        return cls(
            pass1_rules=pass1_sorted,
            pass2_rules=pass2_sorted,
            segment_builder=segment_builder,
            training_base_rate=training_base_rate,
        )

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate predictions using sequential rule application.
        
        Args:
            X: Features DataFrame (raw or segmented)
            
        Returns:
            Tuple of:
                - preds: np.ndarray of predictions (0, 1, or -1 for abstain)
                - conf: np.ndarray of confidence scores (rule precision)
                - decisions: List[str] of decision sources ("pass1", "pass2", "abstain")
        """
        X = pd.DataFrame(X)
        segments = self.segment_builder.assign_segments(X)
        n = len(X)

        preds = np.full(n, ABSTAIN, dtype=int)
        conf = np.zeros(n)
        decisions = ["abstain"] * n

        # Pass 1: Route NOT_SUBSCRIBE cases
        for i in range(n):
            for rule in self.pass1_rules:
                if self._rule_matches(rule, segments.iloc[i]):
                    preds[i] = NOT_SUBSCRIBE
                    conf[i] = rule.precision
                    decisions[i] = "pass1"
                    break

            # Skip Pass 2 if already routed
            if decisions[i] == "pass1":
                continue

            # Pass 2: Detect SUBSCRIBE cases
            for rule in self.pass2_rules:
                if self._rule_matches(rule, segments.iloc[i]):
                    preds[i] = SUBSCRIBE
                    conf[i] = rule.precision
                    decisions[i] = "pass2"
                    break

        return preds, conf, decisions

    def predict_proba(
        self, 
        X: pd.DataFrame, 
        base_rate: float = None
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
                # Pass 1 routes away from SUBSCRIBE
                probas[i, 1] = base_rate * (1 - conf[i])
                probas[i, 0] = 1 - probas[i, 1]
            elif decisions[i] == "pass2":
                # Pass 2 detects SUBSCRIBE
                probas[i, 1] = base_rate + (1 - base_rate) * conf[i]
                probas[i, 0] = 1 - probas[i, 1]

        return probas

    def _rule_matches(self, rule: SelectedRule, row: pd.Series) -> bool:
        """Check if a rule matches a row."""
        return all(row[f] == v for f, v in rule.segment)

    def get_rule_summary(self) -> List[dict]:
        """Generate human-readable summary of rules."""
        summary = []
        
        for i, r in enumerate(self.pass1_rules, 1):
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
        
        for i, r in enumerate(self.pass2_rules, 1):
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

    def __repr__(self):
        return (
            f"GLASSBRWClassifier("
            f"pass1={len(self.pass1_rules)}, "
            f"pass2={len(self.pass2_rules)}, "
            f"base_rate={self.training_base_rate:.3f})"
        )