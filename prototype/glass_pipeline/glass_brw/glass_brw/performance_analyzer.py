# ============================================================
# GLASS-BRW: RULE DIAGNOSTICS MODULE
# ============================================================
# Analyze rule-level false negatives and abstain patterns
# Works with models containing SelectedRule objects
# ============================================================

from typing import List
import pandas as pd
import numpy as np

from glass_brw.core.rule import SelectedRule, SUBSCRIBE, NOT_SUBSCRIBE, ABSTAIN


class RuleDiagnostics:
    """
    Analyze rule-level false negatives and abstain patterns.
    
    Works with fitted GLASS-BRW models that contain SelectedRule objects.
    """
    
    def __init__(
        self, 
        glass_model, 
        subscribe_label: int = SUBSCRIBE, 
        not_subscribe_label: int = NOT_SUBSCRIBE, 
        abstain_label: int = ABSTAIN
    ):
        """
        Initialize rule diagnostics.
        
        Args:
            glass_model: Fitted GLASS_BRW instance
            subscribe_label: Label for SUBSCRIBE class
            not_subscribe_label: Label for NOT_SUBSCRIBE class
            abstain_label: Label for ABSTAIN
        """
        self.glass = glass_model
        self.SUBSCRIBE = subscribe_label
        self.NOT_SUBSCRIBE = not_subscribe_label
        self.ABSTAIN = abstain_label
    
    def analyze_rule_false_negatives(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_pred: np.ndarray,
        test_decisions: List[str]
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
        print("\n" + "="*80)
        print("🔎 RULE-LEVEL FALSE NEGATIVE ANALYSIS (TEST SET)")
        print("="*80)
        
        # Build analysis dataframe
        df_test_analysis = X_test.copy()
        df_test_analysis["y_true"] = y_test.values
        df_test_analysis["pred"] = test_pred
        df_test_analysis["pass_decision"] = test_decisions
        
        rule_stats = []
        
        # ======================
        # PASS 1 RULES (DANGEROUS ONES)
        # ======================
        pass1_rules: List[SelectedRule] = self.glass.pass1_rules
        for ridx, rule in enumerate(pass1_rules, 1):
            mask = df_test_analysis.apply(lambda r: self._rule_matches(rule, r), axis=1)
            total = mask.sum()
            fn = ((mask) & (df_test_analysis["y_true"] == self.SUBSCRIBE)).sum()
            tn = ((mask) & (df_test_analysis["y_true"] == self.NOT_SUBSCRIBE)).sum()
            
            rule_stats.append({
                "pass": "PASS 1",
                "rule_id": rule.rule_id,
                "segment": rule.segment_str if hasattr(rule, 'segment_str') else " AND ".join(f"{f}={v}" for f, v in rule.segment),
                "fires": int(total),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "fn_rate": fn / total if total > 0 else 0.0,
                "precision_train": rule.precision,
                "recall_train": rule.recall
            })
        
        # ======================
        # PASS 2 RULES
        # ======================
        pass2_rules: List[SelectedRule] = self.glass.pass2_rules
        for ridx, rule in enumerate(pass2_rules, 1):
            mask = df_test_analysis.apply(lambda r: self._rule_matches(rule, r), axis=1)
            total = mask.sum()
            fn = ((mask) & (df_test_analysis["y_true"] == self.SUBSCRIBE) & 
                  (df_test_analysis["pred"] != self.SUBSCRIBE)).sum()
            tp = ((mask) & (df_test_analysis["y_true"] == self.SUBSCRIBE) & 
                  (df_test_analysis["pred"] == self.SUBSCRIBE)).sum()
            
            rule_stats.append({
                "pass": "PASS 2",
                "rule_id": rule.rule_id,
                "segment": rule.segment_str if hasattr(rule, 'segment_str') else " AND ".join(f"{f}={v}" for f, v in rule.segment),
                "fires": int(total),
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "fn_rate": fn / total if total > 0 else 0.0,
                "precision_train": rule.precision,
                "recall_train": rule.recall
            })
        
        # Create DataFrame and sort
        rule_df = pd.DataFrame(rule_stats)
        rule_df = rule_df.sort_values(
            by=["pass", "false_negatives", "fires"],
            ascending=[True, False, False]
        )
        
        print(rule_df.to_string(index=False))
        
        return rule_df
    
    def analyze_abstain_samples(
        self,
        y_test: pd.Series,
        test_pred: np.ndarray
    ):
        """
        Analyze abstain samples.
        
        Args:
            y_test: Test labels
            test_pred: Test predictions
        """
        print("\n" + "="*80)
        print("🚨 ABSTAIN SAMPLE ANALYSIS (TEST SET)")
        print("="*80)
        
        abstain_mask = test_pred == self.ABSTAIN
        n_abstain = abstain_mask.sum()
        subs_in_abstain = (y_test[abstain_mask] == self.SUBSCRIBE).sum()
        notsubs_in_abstain = (y_test[abstain_mask] == self.NOT_SUBSCRIBE).sum()
        
        print(f"Total abstained samples: {n_abstain:,}")
        print(f"  Subscribers (FN risk): {subs_in_abstain:,}")
        print(f"  Not-subscribers:       {notsubs_in_abstain:,}")
        
        if n_abstain > 0:
            print(f"  Subscribe rate among abstains: {subs_in_abstain / n_abstain:.2%}")
        
        # Compare to base rate
        print(f"\nBase subscribe rate (test): {y_test.mean():.2%}")
    
    def print_sanity_checks(
        self,
        y_test: pd.Series,
        test_pred: np.ndarray
    ):
        """
        Print sanity check counts.
        
        Args:
            y_test: Test labels
            test_pred: Test predictions
        """
        print("\n" + "="*80)
        print("🧮 SANITY CHECK COUNTS")
        print("="*80)
        
        print(f"Total test samples: {len(y_test):,}")
        print(f"Predicted SUBSCRIBE: {(test_pred == self.SUBSCRIBE).sum():,}")
        print(f"Predicted NOT_SUB:   {(test_pred == self.NOT_SUBSCRIBE).sum():,}")
        print(f"Abstained:           {(test_pred == self.ABSTAIN).sum():,}")
    
    def run_full_diagnostics(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_pred: np.ndarray,
        test_decisions: List[str]
    ) -> pd.DataFrame:
        """
        Run all diagnostic analyses.
        
        Args:
            X_test: Test features (binned BRW features)
            y_test: Test labels
            test_pred: Test predictions
            test_decisions: Test pass decisions
            
        Returns:
            DataFrame with rule-level FN statistics
        """
        # Rule-level FN analysis
        rule_df = self.analyze_rule_false_negatives(
            X_test, y_test, test_pred, test_decisions
        )
        
        # Abstain analysis
        self.analyze_abstain_samples(y_test, test_pred)
        
        # Sanity checks
        self.print_sanity_checks(y_test, test_pred)
        
        return rule_df
    
    def _rule_matches(self, rule: SelectedRule, row: pd.Series) -> bool:
        """Check if a rule matches a row."""
        return all(row[f] == v for f, v in rule.segment)