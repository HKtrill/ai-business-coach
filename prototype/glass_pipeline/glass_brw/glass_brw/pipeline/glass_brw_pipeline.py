# ============================================================
# GLASS-BRW PIPELINE MODULE
# ============================================================
# Main orchestrator for the GLASS-BRW two-pass rule system
# ============================================================

import numpy as np
import pandas as pd

from glass_brw.segment_builder import BankSegmentBuilder
from glass_brw.ilp_rule_selector.ilp_rule_selector import ILPRuleSelector
from glass_brw.rule_generator.rule_generator import RuleGenerator
from glass_brw.rule_evaluator.rule_evaluator import RuleEvaluator


class GLASSBRWPipeline:
    """
    GLASS-BRW Pipeline: Two-pass rule-based classification system.
    
    Pass 1: High-precision rules to route NOT_SUBSCRIBE cases
    Pass 2: Recall-focused rules to detect SUBSCRIBE cases
    
    Attributes exposed directly for ModelSaver compatibility:
        mode, min_support_pass1, min_support_pass2, max_jaccard_overlap,
        min_precision_not_subscribe, max_subscriber_leakage_rate,
        min_precision_subscribe, min_recall_subscribe, max_complexity,
        min_novelty_ratio_pass1, min_novelty_ratio_pass2, enable_novelty_constraints
    """
    
    def __init__(self, config, rf_model=None, segment_builder=None):
        self.config = config
        self.rf_model = rf_model
        self.segment_builder = segment_builder or BankSegmentBuilder()
        
        # ============================================================
        # EXPOSE CONFIG ATTRIBUTES DIRECTLY (ModelSaver compatibility)
        # ============================================================
        self.mode = config.mode
        self.min_support_pass1 = config.min_support_pass1
        self.min_support_pass2 = config.min_support_pass2
        self.max_jaccard_overlap = getattr(config, 'max_jaccard_overlap', None)
        self.min_precision_not_subscribe = config.min_precision_not_subscribe
        self.max_precision_not_subscribe = config.max_precision_not_subscribe
        self.max_subscriber_leakage_rate = config.max_subscriber_leakage_rate
        self.max_subscriber_leakage_absolute = config.max_subscriber_leakage_absolute
        self.min_precision_subscribe = config.min_precision_subscribe
        self.max_precision_subscribe = config.max_precision_subscribe
        self.min_recall_subscribe = config.min_recall_subscribe
        self.max_recall_subscribe = config.max_recall_subscribe
        self.max_complexity = config.max_complexity
        self.min_novelty_ratio_pass1 = config.min_novelty_ratio_pass1
        self.min_novelty_ratio_pass2 = config.min_novelty_ratio_pass2
        self.enable_novelty_constraints = config.enable_novelty_constraints
        self.diversity_weight = config.diversity_weight
        self.max_leakage_rate_depth2 = config.max_leakage_rate_depth2
        self.max_leakage_fraction_depth2 = config.max_leakage_fraction_depth2
        self.max_feature_reuse_pass1 = config.max_feature_reuse_pass1
        self.max_feature_reuse_pass2 = config.max_feature_reuse_pass2
        
        # ============================================================
        # RULE GENERATOR
        # ============================================================
        self.rule_generator = RuleGenerator(
            segment_builder=self.segment_builder,
            rf_model=rf_model,
            min_support_pass1=config.min_support_pass1,
            min_support_pass2=config.min_support_pass2,
            max_complexity=config.max_complexity,
            min_precision_not_subscribe=config.min_precision_not_subscribe,
            max_precision_not_subscribe=config.max_precision_not_subscribe,
            max_subscriber_leakage_rate=config.max_subscriber_leakage_rate,
            max_subscriber_leakage_absolute=config.max_subscriber_leakage_absolute,
            min_precision_subscribe=config.min_precision_subscribe,
            max_precision_subscribe=config.max_precision_subscribe,
            min_recall_subscribe=config.min_recall_subscribe,
            max_recall_subscribe=config.max_recall_subscribe,
            max_leakage_rate_depth2=config.max_leakage_rate_depth2,
            max_leakage_fraction_depth2=config.max_leakage_fraction_depth2,
            diversity_penalty=config.diversity_weight,
            max_feature_reuse_pass1=config.max_feature_reuse_pass1,
            max_feature_reuse_pass2=config.max_feature_reuse_pass2,
        )
        
        # ============================================================
        # RULE EVALUATOR
        # ============================================================
        self.rule_evaluator = RuleEvaluator(segment_builder=self.segment_builder)
        
        # ============================================================
        # ILP SELECTOR
        # ============================================================
        self.ilp_selector = ILPRuleSelector(
            min_pass1_rules=config.min_pass1_rules,
            max_pass1_rules=config.max_pass1_rules,
            min_precision_pass1=config.min_precision_not_subscribe,
            max_precision_pass1=config.max_precision_not_subscribe,
            max_subscriber_leakage_rate_pass1=config.max_subscriber_leakage_rate,
            max_subscriber_leakage_absolute_pass1=config.max_subscriber_leakage_absolute,
            min_pass2_rules=config.min_pass2_rules,
            max_pass2_rules=config.max_pass2_rules,
            min_precision_pass2=config.min_precision_subscribe,
            max_precision_pass2=config.max_precision_subscribe,
            min_recall_pass2=config.min_recall_subscribe,
            max_recall_pass2=config.max_recall_subscribe,
            min_novelty_ratio_pass1=config.min_novelty_ratio_pass1,
            min_novelty_ratio_pass2=config.min_novelty_ratio_pass2,
            enable_novelty_constraints=config.enable_novelty_constraints,
            diversity_weight=config.diversity_weight,
        )
        
        # ============================================================
        # STATE
        # ============================================================
        self.is_fitted = False
        self.pass1_rules = []
        self.pass2_rules = []
        self.training_base_rate = None
        
        print("✅ GLASSBRWPipeline initialized")
    
    # ================================================================
    # FIT
    # ================================================================

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the GLASS-BRW pipeline.
        """
        if X_val is None:
            X_val, y_val = X_train, y_train
    
        # Store training base rate for predict_proba
        self.training_base_rate = (
            y_train.mean() if hasattr(y_train, 'mean') else np.mean(y_train)
        )
    
        # Generate candidate rules
        candidates = self.rule_generator.generate_candidates(X_train, y_train)
    
        # Evaluate candidates on validation set
        evaluated = self.rule_evaluator.evaluate_candidates(
            candidates, X_val, y_val
        )
    
        # ------------------------------------------------------------
        # 🔑 THIS IS THE FIX
        # Forward notebook inputs to ILP selector
        # ------------------------------------------------------------
        selection = self.ilp_selector.select_rules(
            evaluated_rules=evaluated,
            y_val=y_val,
            X_val=X_val,
            segment_builder=self.segment_builder,
        )
    
        self.pass1_rules = selection["pass1_rules"]
        self.pass2_rules = selection["pass2_rules"]
        self.is_fitted = True
    
        print(f"\n✅ Pass 1: {len(self.pass1_rules)} | Pass 2: {len(self.pass2_rules)}")
        return self

    # ================================================================
    # PREDICT
    # ================================================================
    
    def predict(self, X):
        """
        Generate predictions using the two-pass rule system.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Tuple of (predictions, confidence, decisions)
            - predictions: 0 (NOT_SUBSCRIBE), 1 (SUBSCRIBE), or 0 (uncertain)
            - confidence: Rule precision for matched rules
            - decisions: "pass1", "pass2", or "uncertain"
        """
        if not self.is_fitted:
            raise ValueError("Call fit() first")
        
        n = len(X)
        preds = np.zeros(n, dtype=int)
        conf = np.zeros(n)
        decisions = np.array(["uncertain"] * n, dtype=object)
        
        # Pass 1: Route NOT_SUBSCRIBE cases
        for rule in self.pass1_rules:
            mask = np.ones(n, dtype=bool)
            for f, v in rule.segment:
                if f in X.columns:
                    mask &= (X[f] == v)
            apply = mask & (decisions == "uncertain")
            preds[apply] = 0
            conf[apply] = getattr(rule, 'precision', 0.9)
            decisions[apply] = "pass1"
        
        # Pass 2: Detect SUBSCRIBE cases
        for rule in self.pass2_rules:
            mask = np.ones(n, dtype=bool)
            for f, v in rule.segment:
                if f in X.columns:
                    mask &= (X[f] == v)
            apply = mask & (decisions == "uncertain")
            preds[apply] = 1
            conf[apply] = getattr(rule, 'precision', 0.5)
            decisions[apply] = "pass2"
        
        return preds, conf, decisions
    
    # ================================================================
    # PREDICT_PROBA
    # ================================================================
    
    def predict_proba(self, X, base_rate=None):
        """
        Generate probabilistic predictions.
        
        Args:
            X: Features DataFrame
            base_rate: Prior probability of SUBSCRIBE (defaults to training rate)
            
        Returns:
            Array of shape (n, 2) with [P(NOT_SUBSCRIBE), P(SUBSCRIBE)]
        """
        preds, conf, decisions = self.predict(X)
        n = len(preds)

        if base_rate is None:
            base_rate = self.training_base_rate if self.training_base_rate else 0.117

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
    
    # ================================================================
    # RULE SUMMARY
    # ================================================================
    
    def get_rule_summary(self):
        """
        Generate human-readable summary of learned rules.
        
        Returns:
            List of dicts with rule details
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        summary = []
        
        # Pass 1 rules
        for i, r in enumerate(self.pass1_rules, 1):
            summary.append({
                "rule_id": i,
                "pass": "Pass 1",
                "class": "NOT_SUBSCRIBE (0)",
                "segment": " AND ".join(f"{f}={v}" for f, v in r.segment),
                "precision": r.precision,
                "recall": r.recall,
                "coverage": r.coverage,
                "complexity": r.complexity,
                "rf_alignment": getattr(r, "rf_alignment", 0.0),
            })
        
        # Pass 2 rules
        for i, r in enumerate(self.pass2_rules, 1):
            summary.append({
                "rule_id": i,
                "pass": "Pass 2",
                "class": "SUBSCRIBE (1)",
                "segment": " AND ".join(f"{f}={v}" for f, v in r.segment),
                "precision": r.precision,
                "recall": r.recall,
                "coverage": r.coverage,
                "complexity": r.complexity,
                "rf_alignment": getattr(r, "rf_alignment", 0.0),
            })
        
        return summary
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def get_rules_dataframe(self):
        """Return rules as a pandas DataFrame for analysis."""
        return pd.DataFrame(self.get_rule_summary())
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        p1 = len(self.pass1_rules) if self.is_fitted else 0
        p2 = len(self.pass2_rules) if self.is_fitted else 0
        return f"GLASSBRWPipeline(mode='{self.mode}', {status}, pass1={p1}, pass2={p2})"