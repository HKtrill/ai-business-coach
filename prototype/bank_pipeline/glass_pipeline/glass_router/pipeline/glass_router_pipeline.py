# ============================================================
# GLASS ROUTER PIPELINE MODULE
# ============================================================
# Main orchestrator for the GLASS Router two-pass rule system
# ============================================================

import contextlib
import hashlib
import io

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from glass_pipeline.glass_router.core.config import INNER_RANKER_FOLD_NESTED_RF, INNER_RANKER_OUTER_RF
from glass_pipeline.glass_router.core.segment_builder import BankSegmentBuilder
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import ILPRuleSelector
from glass_pipeline.glass_router.rule_generator.rule_generator import RuleGenerator
from glass_pipeline.glass_router.rule_evaluator.rule_evaluator import RuleEvaluator
from glass_pipeline.glass_router.core.rule import SUBSCRIBE, NOT_SUBSCRIBE, ABSTAIN
from shared.stage_io import label_fingerprint

#: ``_new_generator(rf_model=...)`` default: use the pipeline's own ``rf_model``.
_OWN_RF = object()


def feature_order_fingerprint(estimator, columns) -> str:
    """
    SHA-256 (16 hex) of the feature order a ranker induces.

    The order mirrors ``RuleGenerator._get_feature_order``: importance
    descending, ties in column order (Python's sort is stable).
    """
    imp = np.asarray(estimator.feature_importances_, dtype=float)
    order = sorted(range(len(columns)), key=lambda i: imp[i], reverse=True)
    return hashlib.sha256("\n".join(str(columns[i]) for i in order).encode("utf-8")).hexdigest()[:16]


class GlassRouterPipeline:
    """
    GLASS Router Pipeline: Two-pass rule-based classification system.

    Pass 1: High-precision rules to route NOT_SUBSCRIBE cases
    Pass 2: Recall-focused rules to detect SUBSCRIBE cases

    Pass 2 population is set by ``config.pass2_population``:
        "full"          Pass 2 is fitted on the whole training population
                        (original behaviour).
        "oof_remainder" Pass 2 is fitted on the rows Pass 1 does not route,
                        where routing comes from OUT-OF-FOLD Pass 1 decisions
                        (config.pass1_oof_folds inner folds). This is the same
                        construction as the black-box RF router, so a row's
                        own Pass 1 fit never decides whether it joins the Pass 2
                        training population. Pass 2 recall stays on the global
                        denominator (all training positives) in both modes.

    Inner-fold feature ranking (``config.inner_feature_ranker``, oof_remainder
    only; PR 37 Commit 3, F17):
        "outer_rf"       (default, historical) every inner-fold Pass 1 search is
                        ordered by ``rf_model``. In the cross-fit that RF was
                        fitted on every row the inner folds hold out.
        "fold_nested_rf" (opt-in) each inner fold's search is ordered by a fresh
                        ranker, ``feature_ranker_factory(X_inner_train,
                        y_inner_train)``, so it is never trained on the rows
                        it routes. Router-level isolation only, conditional on
                        the engineered features (fitted on the whole outer-
                        training partition; PR 37 F17 L1, PR 38 gate G13).
                        The full-population Pass 1 and the Pass 2 search keep
                        ``rf_model``. No factory -> fit() refuses (no fallback).

    Attributes exposed directly for ModelSaver compatibility:
        mode, min_support_pass1, min_support_pass2, max_jaccard_overlap,
        min_precision_not_subscribe, max_subscriber_leakage_rate,
        min_precision_subscribe, min_recall_subscribe, max_complexity,
        min_novelty_ratio_pass1, min_novelty_ratio_pass2, enable_novelty_constraints
    """

    def __init__(self, config, rf_model=None, segment_builder=None, rule_logger=None,
                 feature_ranker_factory=None):
        self.config = config
        self.rf_model = rf_model
        self.segment_builder = segment_builder or BankSegmentBuilder()
        self.rule_logger = rule_logger
        # Opt-in (inner_feature_ranker="fold_nested_rf"): (X, y) -> a fitted
        # estimator exposing feature_importances_, fitted on exactly (X, y).
        self.feature_ranker_factory = feature_ranker_factory

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
        self.pass2_population = getattr(config, "pass2_population", "full")
        self.inner_feature_ranker = getattr(config, "inner_feature_ranker", INNER_RANKER_OUTER_RF)

        # Components used by the main (full-population) fit
        self.rule_generator = self._new_generator()
        self.rule_evaluator = self._new_evaluator()
        self.ilp_selector = self._new_selector()

        # ============================================================
        # STATE
        # ============================================================
        self.is_fitted = False
        self.pass1_rules = []
        self.pass2_rules = []
        self.training_base_rate = None
        self.train_label_fingerprint_ = None
        self.fit_report_ = {}

        print("✅ GlassRouterPipeline initialized")

    # ================================================================
    # COMPONENT FACTORIES
    # ================================================================
    # Fresh instances per fit so no state (e.g. a generator's lazily trained
    # RF) carries from one population to another.

    def _new_generator(self, recall_scale: float = 1.0, rf_model=_OWN_RF):
        """
        RuleGenerator for this config.

        recall_scale = remainder_positives / total_positives. The generator
        measures recall against the population it is given; when that is the
        Pass 1 remainder, the global recall bounds are divided by this factor so
        the depth-3 gates keep their global meaning.

        rf_model: the feature ranker; default = this pipeline's ``rf_model``.
        """
        c = self.config
        return RuleGenerator(
            segment_builder=self.segment_builder,
            rf_model=self.rf_model if rf_model is _OWN_RF else rf_model,
            min_support_pass1=c.min_support_pass1,
            min_support_pass2=c.min_support_pass2,
            max_complexity=c.max_complexity,
            min_precision_not_subscribe=c.min_precision_not_subscribe,
            max_precision_not_subscribe=c.max_precision_not_subscribe,
            max_subscriber_leakage_rate=c.max_subscriber_leakage_rate,
            max_subscriber_leakage_absolute=c.max_subscriber_leakage_absolute,
            min_precision_subscribe=c.min_precision_subscribe,
            max_precision_subscribe=c.max_precision_subscribe,
            min_recall_subscribe=min(1.0, c.min_recall_subscribe / recall_scale),
            max_recall_subscribe=min(1.0, c.max_recall_subscribe / recall_scale),
            max_leakage_rate_depth2=c.max_leakage_rate_depth2,
            max_leakage_fraction_depth2=c.max_leakage_fraction_depth2,
            diversity_penalty=c.diversity_weight,
            max_feature_reuse_pass1=c.max_feature_reuse_pass1,
            max_feature_reuse_pass2=c.max_feature_reuse_pass2,
            rule_logger=self.rule_logger,
        )

    def _new_evaluator(self):
        c = self.config
        return RuleEvaluator(
            segment_builder=self.segment_builder,
            min_support=min(c.min_support_pass1, c.min_support_pass2),
        )

    def _new_selector(self):
        c = self.config
        return ILPRuleSelector(
            min_pass1_rules=c.min_pass1_rules,
            max_pass1_rules=c.max_pass1_rules,
            min_precision_pass1=c.min_precision_not_subscribe,
            max_precision_pass1=c.max_precision_not_subscribe,
            max_subscriber_leakage_rate_pass1=c.max_subscriber_leakage_rate,
            max_subscriber_leakage_absolute_pass1=c.max_subscriber_leakage_absolute,
            min_pass2_rules=c.min_pass2_rules,
            max_pass2_rules=c.max_pass2_rules,
            min_precision_pass2=c.min_precision_subscribe,
            max_precision_pass2=c.max_precision_subscribe,
            min_recall_pass2=c.min_recall_subscribe,
            max_recall_pass2=c.max_recall_subscribe,
            min_novelty_ratio_pass1=c.min_novelty_ratio_pass1,
            min_novelty_ratio_pass2=c.min_novelty_ratio_pass2,
            enable_novelty_constraints=c.enable_novelty_constraints,
            diversity_weight=c.diversity_weight,
            lambda_rf_uncertainty=c.lambda_rf_uncertainty,
            lambda_rf_misalignment=c.lambda_rf_misalignment,
            max_union_leakage_rate_pass1=getattr(c, "max_union_leakage_rate_pass1", None),
        )

    # ================================================================
    # FIT
    # ================================================================

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the GLASS Router pipeline.

        Rules are evaluated on X_val / y_val, which default to the training
        data. In "oof_remainder" mode X_val must be the training data (or
        omitted), because the Pass 2 remainder is defined over training rows.
        """
        if X_val is None:
            X_val, y_val = X_train, y_train

        self.training_base_rate = (
            y_train.mean() if hasattr(y_train, 'mean') else np.mean(y_train)
        )
        # PR 36 (audit F12): the rows and labels this fit used (test-side outputs).
        rows = getattr(y_train, "index", getattr(X_train, "index", range(len(y_train))))
        self.train_label_fingerprint_ = label_fingerprint(rows, y_train)

        if self.pass2_population == "full":
            self._fit_full(X_train, y_train, X_val, y_val)
        else:
            if not (X_val is X_train or X_val.index.equals(X_train.index)):
                raise ValueError(
                    "pass2_population='oof_remainder' defines the Pass 2 remainder "
                    "over training rows; pass X_val=None (or the training data)."
                )
            if (self.inner_feature_ranker == INNER_RANKER_FOLD_NESTED_RF
                    and self.feature_ranker_factory is None):
                raise ValueError(
                    "inner_feature_ranker='fold_nested_rf' needs a feature_ranker_factory "
                    "(crossfit_stage2 supplies one); refusing to fall back to the "
                    "outer rf_model."
                )
            self._fit_oof_remainder(X_train, y_train)

        self.is_fitted = True
        print(f"\n✅ Pass 1: {len(self.pass1_rules)} | Pass 2: {len(self.pass2_rules)}")
        return self

    # ----------------------------------------------------------------
    def _fit_full(self, X_train, y_train, X_val, y_val):
        """Original behaviour: both passes selected on the same population."""
        candidates = self.rule_generator.generate_candidates(X_train, y_train)
        evaluated = self.rule_evaluator.evaluate_candidates(candidates, X_val, y_val)
        selection = self.ilp_selector.select_rules(
            evaluated_rules=evaluated,
            y_val=y_val,
            X_val=X_val,
            segment_builder=self.segment_builder,
        )
        self.pass1_rules = selection["pass1_rules"]
        self.pass2_rules = selection["pass2_rules"]
        self.fit_report_ = {"pass2_population": "full"}

    # ----------------------------------------------------------------
    def _fit_oof_remainder(self, X, y):
        """
        Pass 1 on the full population; Pass 2 on the OOF Pass 1 remainder.

        1. Pass 1 rules: generated, evaluated, selected on all training rows
           (unchanged Pass 1 semantics).
        2. OOF Pass 1 routing: for each inner fold, a Pass-1-only fit on the
           other folds routes the held-out fold.
        3. Remainder = training rows NOT routed by their OOF Pass 1 decision.
        4. Pass 2 rules: generated, evaluated and selected on the remainder.
           Recall and coverage are rescaled to the global denominators
           (all training positives / all training rows) before the ILP gates.
        """
        c = self.config
        y = pd.Series(np.asarray(y), index=X.index) if not isinstance(y, pd.Series) else y
        y_arr = np.asarray(y).astype(int)
        n, total_pos = len(y_arr), int(y_arr.sum())

        # ---- 1. Pass 1 on the full population ---------------------------
        print("\n[oof_remainder] Pass 1 on the full training population")
        self.pass1_rules = self._fit_pass1_only(
            X, y, self.rule_generator, self.rule_evaluator, self.ilp_selector
        )

        # ---- 2. OOF Pass 1 routing ----------------------------------------
        print(f"[oof_remainder] OOF Pass 1 routing ({c.pass1_oof_folds} inner folds)")
        routed_oof = np.zeros(n, dtype=bool)
        inner_rules = []
        nested_ranker = self.inner_feature_ranker == INNER_RANKER_FOLD_NESTED_RF
        ranker_provenance, ranker_objs = [], []
        skf = StratifiedKFold(
            n_splits=c.pass1_oof_folds, shuffle=True, random_state=c.random_state
        )
        for k, (tr, va) in enumerate(skf.split(np.zeros(n), y_arr)):
            ranker = (self._fit_inner_ranker(k, X.iloc[tr], y.iloc[tr],
                                             ranker_provenance, ranker_objs)
                      if nested_ranker else None)
            with contextlib.redirect_stdout(io.StringIO()):
                generator = (self._new_generator(rf_model=ranker) if nested_ranker
                             else self._new_generator())
                rules_k = self._fit_pass1_only(
                    X.iloc[tr], y.iloc[tr],
                    generator, self._new_evaluator(), self._new_selector(),
                )
            routed_oof[va] = self._rules_mask(rules_k, X.iloc[va])
            inner_rules.append(len(rules_k))
            print(f"   inner fold {k}: {len(rules_k)} Pass 1 rules | "
                  f"routes {routed_oof[va].mean():.1%} of held-out rows")

        # ---- 3. Remainder --------------------------------------------------
        remainder = ~routed_oof
        n_rem = int(remainder.sum())
        rem_pos = int(y_arr[remainder].sum())
        if n_rem < c.remainder_min_size:
            raise ValueError(
                f"OOF Pass 1 leaves only {n_rem} training rows "
                f"(remainder_min_size={c.remainder_min_size})."
            )
        if rem_pos == 0 or rem_pos == n_rem:
            raise ValueError(f"Remainder is single-class ({rem_pos}/{n_rem}); cannot fit Pass 2.")

        recall_scale = rem_pos / total_pos
        coverage_scale = n_rem / n
        print(f"[oof_remainder] remainder: {n_rem:,} rows ({n_rem / n:.1%}), "
              f"{rem_pos:,} positives ({recall_scale:.1%} of all positives)")

        # ---- 4. Pass 2 on the remainder -----------------------------------
        X_rem, y_rem = X.iloc[np.flatnonzero(remainder)], y.iloc[np.flatnonzero(remainder)]
        gen2 = self._new_generator(recall_scale=recall_scale)
        cands2 = [r for r in gen2.generate_candidates(X_rem, y_rem) if r.predicted_class == SUBSCRIBE]
        ev2 = self._new_evaluator().evaluate_candidates(cands2, X_rem, y_rem)
        for r in ev2:
            # Remainder-relative -> global denominators (all positives / all rows)
            r.recall = r.recall * recall_scale
            r.coverage = r.coverage * coverage_scale
        sel2 = self._new_selector().select_rules(
            evaluated_rules=ev2, y_val=y_rem, X_val=X_rem,
            segment_builder=self.segment_builder, passes=("pass2",),
        )
        self.pass2_rules = sel2["pass2_rules"]

        self.fit_report_ = {
            "pass2_population": "oof_remainder",
            "pass1_oof_folds": c.pass1_oof_folds,
            "random_state": c.random_state,
            "inner_pass1_rule_counts": inner_rules,
            "remainder_rows": n_rem,
            "remainder_fraction": n_rem / n,
            "remainder_positives": rem_pos,
            "remainder_base_rate": rem_pos / n_rem,
            "recall_scale": recall_scale,
            "coverage_scale": coverage_scale,
            "recall_denominator": "global",
            "pass2_precision_measured_on": "remainder",
            "remainder_mask": remainder,
            # PR 37 Commit 3 (F17): what ordered the inner-fold Pass 1 searches.
            "inner_feature_ranker": self.inner_feature_ranker,
        }
        if nested_ranker:
            # Metadata only in the report (it is saved with the artifact); the
            # fitted rankers stay on the instance for in-memory audits.
            self.fit_report_["inner_ranker_provenance"] = ranker_provenance
            self.inner_rankers_ = ranker_objs

    # ----------------------------------------------------------------
    def _fit_inner_ranker(self, k, X_tr, y_tr, provenance, seen):
        """
        Fit inner fold k's feature ranker on its training rows only (opt-in).

        Refuses a ranker that is not a fresh, fitted estimator with one
        importance per column: reusing the outer ``rf_model`` or another fold's
        ranker would defeat the nesting.
        """
        ranker = self.feature_ranker_factory(X_tr, y_tr)
        imp = getattr(ranker, "feature_importances_", None)
        if imp is None or len(imp) != X_tr.shape[1]:
            raise ValueError(
                f"inner fold {k}: feature_ranker_factory must return a fitted estimator "
                f"with {X_tr.shape[1]} feature_importances_"
            )
        if ranker is self.rf_model or any(ranker is r for r in seen):
            raise ValueError(
                f"inner fold {k}: feature_ranker_factory returned an estimator already in "
                "use (the outer rf_model or another fold's ranker); each inner fold needs "
                "its own ranker fitted on its own training rows"
            )
        seen.append(ranker)
        provenance.append({
            "fold": int(k),
            "n_train": int(len(X_tr)),
            "train_label_fingerprint": label_fingerprint(X_tr.index, y_tr),
            "estimator": type(ranker).__name__,
            "random_state": getattr(ranker, "random_state", None),
            "feature_order_fingerprint": feature_order_fingerprint(ranker, list(X_tr.columns)),
        })
        return ranker

    # ----------------------------------------------------------------
    def _fit_pass1_only(self, X, y, generator, evaluator, selector):
        """Generate, evaluate and select Pass 1 (NOT_SUBSCRIBE) rules only."""
        cands = [r for r in generator.generate_candidates(X, y) if r.predicted_class == NOT_SUBSCRIBE]
        ev = evaluator.evaluate_candidates(cands, X, y)
        sel = selector.select_rules(
            evaluated_rules=ev, y_val=y, X_val=X, segment_builder=self.segment_builder,
            passes=("pass1",),
        )
        return sel["pass1_rules"]

    @staticmethod
    def _rules_mask(rules, X):
        """Rows matched by ANY of the rules."""
        hit = np.zeros(len(X), dtype=bool)
        for rule in rules:
            m = np.ones(len(X), dtype=bool)
            for f, v in rule.segment:
                if f not in X.columns:
                    raise ValueError(f"Missing required segment feature in X: {f}")
                m &= (X[f].to_numpy() == v)
            hit |= m
        return hit

    def _get_rule_precision(self, rule):
        """Return rule precision, failing loudly if missing."""
        if not hasattr(rule, "precision") or rule.precision is None:
            rule_id = getattr(rule, "rule_id", "<unknown>")
            raise ValueError(f"Rule {rule_id} is missing precision")
        return rule.precision

    # ================================================================
    # PREDICT
    # ================================================================

    def predict(self, X):
        """
        Generate predictions using the two-pass rule system.

        Returns:
            Tuple of (predictions, confidence, decisions)
            - predictions: NOT_SUBSCRIBE (0), SUBSCRIBE (1), or ABSTAIN (-1)
            - confidence: Rule precision for matched rules (0.0 for abstains)
            - decisions: "pass1", "pass2", or "uncertain" (abstain)
        """
        if not self.is_fitted:
            raise ValueError("Call fit() first")

        n = len(X)
        preds = np.full(n, ABSTAIN, dtype=int)
        conf = np.zeros(n)
        decisions = np.array(["uncertain"] * n, dtype=object)

        # Pass 1: Route NOT_SUBSCRIBE cases
        for rule in self.pass1_rules:
            mask = np.ones(n, dtype=bool)
            for f, v in rule.segment:
                if f not in X.columns:
                    raise ValueError(f"Missing required segment feature in X: {f}")
                mask &= (X[f] == v)
            apply = mask & (decisions == "uncertain")
            preds[apply] = NOT_SUBSCRIBE
            conf[apply] = self._get_rule_precision(rule)
            decisions[apply] = "pass1"

        # Pass 2: Detect SUBSCRIBE cases among what Pass 1 left
        for rule in self.pass2_rules:
            mask = np.ones(n, dtype=bool)
            for f, v in rule.segment:
                if f not in X.columns:
                    raise ValueError(f"Missing required segment feature in X: {f}")
                mask &= (X[f] == v)
            apply = mask & (decisions == "uncertain")
            preds[apply] = SUBSCRIBE
            conf[apply] = self._get_rule_precision(rule)
            decisions[apply] = "pass2"

        return preds, conf, decisions

    # ================================================================
    # PREDICT_PROBA
    # ================================================================

    def predict_proba(self, X, base_rate=None):
        """
        Generate probabilistic predictions.

        Returns:
            Array of shape (n, 2) with [P(NOT_SUBSCRIBE), P(SUBSCRIBE)]
        """
        preds, conf, decisions = self.predict(X)
        n = len(preds)

        if base_rate is None:
            if self.training_base_rate is None:
                raise ValueError(
                    "training_base_rate is not set; call fit() before predict_proba "
                    "or pass base_rate explicitly."
                )
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

    def get_rule_summary(self):
        """List of dicts with rule details."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        summary = []
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

    def get_rules_dataframe(self):
        """Return rules as a pandas DataFrame for analysis."""
        return pd.DataFrame(self.get_rule_summary())

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        p1 = len(self.pass1_rules) if self.is_fitted else 0
        p2 = len(self.pass2_rules) if self.is_fitted else 0
        return (f"GlassRouterPipeline(mode='{self.mode}', {status}, pass1={p1}, "
                f"pass2={p2}, pass2_population='{self.pass2_population}')")