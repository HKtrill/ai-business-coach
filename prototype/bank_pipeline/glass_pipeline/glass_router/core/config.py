# ============================================================
# GLASS ROUTER: CONFIGURATION MODULE
# ============================================================
# Single source of truth for Stage 2 experiment parameters.
# Required parameters default to None and must be set explicitly
# by the notebook. Optional parameters may remain None.
# ============================================================

import math
import numbers
from dataclasses import dataclass, fields
from typing import Optional

#: Feature ranker for the inner-fold Pass 1 searches of
#: ``pass2_population="oof_remainder"`` (PR 37 Commit 3, F17).
#: "outer_rf"       historical: the caller's ``rf_model`` (in the cross-fit, the
#:                  outer fold's RF, fitted on every row the inner folds hold out).
#: "fold_nested_rf" opt-in: a fresh ranker fitted only on each inner fold's
#:                  training rows, by ``GlassRouterPipeline.feature_ranker_factory``.
INNER_RANKER_OUTER_RF = "outer_rf"
INNER_RANKER_FOLD_NESTED_RF = "fold_nested_rf"
INNER_FEATURE_RANKERS = (INNER_RANKER_OUTER_RF, INNER_RANKER_FOLD_NESTED_RF)

#: Pass 2 union objective of the ILP (PR 37 Commit 4, Step 4B).
#: "positives" historical: maximise the positives the selected union captures.
#: "rows"      opt-in: maximise the rows the selected union flags (routed-set
#:             coverage, the shared contract's objective). Only valid together
#:             with a union precision floor, which stops it trading precision
#:             for coverage.
PASS2_OBJECTIVE_POSITIVES = "positives"
PASS2_OBJECTIVE_ROWS = "rows"
PASS2_UNION_OBJECTIVES = (PASS2_OBJECTIVE_POSITIVES, PASS2_OBJECTIVE_ROWS)


def union_obligation_errors(*, min_union_npv_pass1, min_union_support_pass1,
                            min_union_precision_pass2, min_union_support_pass2,
                            pass2_union_objective) -> list:
    """Validation shared by GlassRouterConfig and ILPRuleSelector (Step 4B)."""
    errors = []
    for name, v in (("min_union_npv_pass1", min_union_npv_pass1),
                    ("min_union_precision_pass2", min_union_precision_pass2)):
        if v is None:
            continue
        if (isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v)
                or not 0.0 < float(v) <= 1.0):
            errors.append(f"{name} must be None or a number in (0, 1], got {v!r}")
    for name, v in (("min_union_support_pass1", min_union_support_pass1),
                    ("min_union_support_pass2", min_union_support_pass2)):
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
            errors.append(f"{name} must be None or an integer >= 1, got {v!r}")
    if pass2_union_objective not in PASS2_UNION_OBJECTIVES:
        errors.append(f"pass2_union_objective must be one of {list(PASS2_UNION_OBJECTIVES)}, got "
                      f"{pass2_union_objective!r}")
    elif pass2_union_objective == PASS2_OBJECTIVE_ROWS and min_union_precision_pass2 is None:
        errors.append("pass2_union_objective='rows' needs min_union_precision_pass2: without a union "
                      "precision floor, maximising flagged rows is degenerate")
    return errors


@dataclass
class GlassRouterConfig:
    """
    Configuration for the GLASS two-pass symbolic rule router.

    Required experiment parameters must be explicitly set so the
    notebook remains the single source of truth for each run.
    Optional parameters may remain None.

    Usage:
        config = GlassRouterConfig(
            mode="strict",
            min_support_pass1=125,
            min_support_pass2=75,
            ...
        )

    Raises:
        ValueError:
            If a required parameter is missing or a parameter value
            violates the configured constraints.
    """
    
    # ============================================================
    # EXECUTION MODE - REQUIRED
    # ============================================================
    mode: Optional[str] = None  # "strict", "relaxed", "exploratory"
    
    # ============================================================
    # SUPPORT THRESHOLDS - REQUIRED
    # ============================================================
    min_support_pass1: Optional[int] = None
    min_support_pass2: Optional[int] = None
    
    # ============================================================
    # DEPTH 2 PRUNING - REQUIRED
    # ============================================================
    max_leakage_rate_depth2: Optional[float] = None
    max_leakage_fraction_depth2: Optional[float] = None
    
    # ============================================================
    # OVERLAP CONTROL - REQUIRED
    # ============================================================
    max_jaccard_overlap: Optional[float] = None
    max_high_overlap_rules: Optional[int] = None
    
    # ============================================================
    # PASS 1: ROUTING (NOT_SUBSCRIBE) - REQUIRED
    # ============================================================
    min_pass1_rules: Optional[int] = None
    max_pass1_rules: Optional[int] = None
    min_precision_not_subscribe: Optional[float] = None
    max_precision_not_subscribe: Optional[float] = None
    max_subscriber_leakage_rate: Optional[float] = None
    max_subscriber_leakage_absolute: Optional[int] = None
    
    # ============================================================
    # PASS 2: DETECTION (SUBSCRIBE) - REQUIRED
    # ============================================================
    min_pass2_rules: Optional[int] = None
    max_pass2_rules: Optional[int] = None
    min_precision_subscribe: Optional[float] = None
    max_precision_subscribe: Optional[float] = None
    min_recall_subscribe: Optional[float] = None
    max_recall_subscribe: Optional[float] = None
    
    # ============================================================
    # NOVELTY CONSTRAINTS - REQUIRED
    # ============================================================
    min_novelty_ratio_pass1: Optional[float] = None
    min_novelty_ratio_pass2: Optional[float] = None
    enable_novelty_constraints: Optional[bool] = None
    
    # ============================================================
    # SHARED PARAMETERS - REQUIRED
    # ============================================================
    max_complexity: Optional[int] = None
    diversity_weight: Optional[float] = None
    
    # ============================================================
    # OPTIONAL PARAMETERS (can remain None)
    # ============================================================
    max_feature_reuse_pass1: Optional[int] = None
    max_feature_reuse_pass2: Optional[int] = None
    lambda_rf_uncertainty: Optional[float] = None
    lambda_rf_misalignment: Optional[float] = None
    # Pass 1 UNION leakage cap: subscribers routed by the selected Pass 1 rule
    # set (each counted once) <= this share of all subscribers. None = off; the
    # per-rule max_subscriber_leakage_rate / _absolute gates apply either way.
    max_union_leakage_rate_pass1: Optional[float] = None

    # ============================================================
    # PASS 2 POPULATION (optional; default = original behaviour)
    # ============================================================
    # "full"          Pass 2 rules are generated, evaluated and selected on the
    #                 whole training population (original GLASS behaviour).
    # "oof_remainder" Pass 2 is fitted on the rows Pass 1 does NOT route, where
    #                 "routed" comes from out-of-fold Pass 1 decisions
    #                 (pass1_oof_folds inner folds). Same construction as the
    #                 black-box RF router. Recall stays on the global
    #                 denominator (all training positives) in both modes.
    pass2_population: str = "full"
    pass1_oof_folds: int = 5
    random_state: int = 42
    remainder_min_size: int = 500
    # Inner-fold Pass 1 feature ranker (oof_remainder only); see
    # INNER_FEATURE_RANKERS above. Default = historical behaviour.
    inner_feature_ranker: str = INNER_RANKER_OUTER_RF

    # ============================================================
    # SET-LEVEL (UNION) OBLIGATIONS (optional; PR 37 Commit 4, Step 4B)
    # ============================================================
    # Checked by the ILP on the SELECTED UNION of rules, each row counted once,
    # on the population the pass is selected on (all fitting rows for Pass 1,
    # the Pass 1 remainder for Pass 2 under "oof_remainder"). None = off: the
    # historical per-rule gates and objectives are unchanged. The per-rule
    # gates above still apply as candidate filters either way.
    min_union_npv_pass1: Optional[float] = None        # union negatives / union rows >= this
    min_union_support_pass1: Optional[int] = None      # union rows >= this
    min_union_precision_pass2: Optional[float] = None  # union positives / union rows >= this
    min_union_support_pass2: Optional[int] = None      # union rows >= this
    pass2_union_objective: str = PASS2_OBJECTIVE_POSITIVES
    # Provenance block of the shared contract this config was built from
    # (glass_pipeline.glass_router.contract.glass_config_from_contract). None
    # means the config claims no shared-contract compliance.
    contract_provenance: Optional[dict] = None

    def __post_init__(self):
        """Validate that all required parameters are set."""
        optional_params = {
            'max_feature_reuse_pass1',
            'max_feature_reuse_pass2',
            'lambda_rf_uncertainty',
            'lambda_rf_misalignment',
            'max_union_leakage_rate_pass1',
            'pass2_population',
            'pass1_oof_folds',
            'random_state',
            'remainder_min_size',
            'inner_feature_ranker',
            'min_union_npv_pass1',
            'min_union_support_pass1',
            'min_union_precision_pass2',
            'min_union_support_pass2',
            'pass2_union_objective',
            'contract_provenance',
        }
        
        missing = []
        for f in fields(self):
            if f.name not in optional_params:
                if getattr(self, f.name) is None:
                    missing.append(f.name)
        
        if missing:
            raise ValueError(
                f"GlassRouterConfig missing required parameters: {missing}\n"
                f"All required parameters must be explicitly set in the notebook."
        )
        
        valid_modes = {"strict", "relaxed", "exploratory"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {valid_modes}"
            )
        
        self._validate_ranges()
    
    def _validate_ranges(self):
        """Validate parameter ranges and cross-parameter constraints."""
        errors = []
        
        if self.min_support_pass1 <= 0:
            errors.append("min_support_pass1 must be > 0")
        if self.min_support_pass2 <= 0:
            errors.append("min_support_pass2 must be > 0")
        
        for param in ['min_precision_not_subscribe', 'max_precision_not_subscribe',
                      'min_precision_subscribe', 'max_precision_subscribe',
                      'min_recall_subscribe', 'max_recall_subscribe',
                      'max_subscriber_leakage_rate', 'diversity_weight',
                      'min_novelty_ratio_pass1', 'min_novelty_ratio_pass2']:
            val = getattr(self, param)
            if not 0 <= val <= 1:
                errors.append(f"{param} must be in [0, 1], got {val}")
        
        if self.max_union_leakage_rate_pass1 is not None and not 0 <= self.max_union_leakage_rate_pass1 <= 1:
            errors.append(
                f"max_union_leakage_rate_pass1 must be in [0, 1] or None, "
                f"got {self.max_union_leakage_rate_pass1}"
            )
        if self.min_pass1_rules > self.max_pass1_rules:
            errors.append("min_pass1_rules > max_pass1_rules")
        if self.min_pass2_rules > self.max_pass2_rules:
            errors.append("min_pass2_rules > max_pass2_rules")
        if self.min_precision_not_subscribe > self.max_precision_not_subscribe:
            errors.append("min_precision_not_subscribe > max_precision_not_subscribe")
        if self.min_precision_subscribe > self.max_precision_subscribe:
            errors.append("min_precision_subscribe > max_precision_subscribe")
        if self.min_recall_subscribe > self.max_recall_subscribe:
            errors.append("min_recall_subscribe > max_recall_subscribe")
        if self.max_complexity <= 0:
            errors.append("max_complexity must be > 0")
        if self.pass2_population not in ("full", "oof_remainder"):
            errors.append(
                f"pass2_population must be 'full' or 'oof_remainder', got {self.pass2_population!r}"
            )
        if self.pass1_oof_folds < 2:
            errors.append("pass1_oof_folds must be >= 2")
        if self.remainder_min_size < 1:
            errors.append("remainder_min_size must be >= 1")
        if self.inner_feature_ranker not in INNER_FEATURE_RANKERS:
            errors.append(
                f"inner_feature_ranker must be one of {list(INNER_FEATURE_RANKERS)}, "
                f"got {self.inner_feature_ranker!r}"
            )
        errors.extend(union_obligation_errors(
            min_union_npv_pass1=self.min_union_npv_pass1,
            min_union_support_pass1=self.min_union_support_pass1,
            min_union_precision_pass2=self.min_union_precision_pass2,
            min_union_support_pass2=self.min_union_support_pass2,
            pass2_union_objective=self.pass2_union_objective,
        ))
        if self.contract_provenance is not None:
            cp = self.contract_provenance
            if not isinstance(cp, dict) or cp.get("constraint_source") != "shared_contract":
                errors.append("contract_provenance must be the shared contract's provenance block "
                              "(constraint_source 'shared_contract')")
            unset = [n for n in ("max_union_leakage_rate_pass1", "min_union_npv_pass1",
                                 "min_union_support_pass1", "min_union_precision_pass2",
                                 "min_union_support_pass2") if getattr(self, n) is None]
            if unset or self.pass2_union_objective != PASS2_OBJECTIVE_ROWS:
                errors.append("a config that records shared-contract provenance must enforce every "
                              f"union obligation (unset: {unset}) with "
                              f"pass2_union_objective='{PASS2_OBJECTIVE_ROWS}'; per-rule gates alone "
                              "cannot claim shared compliance")
        
        if errors:
            raise ValueError(
                f"GlassRouterConfig validation errors:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GlassRouterConfig':
        """Create config from dictionary."""
        return cls(**d)
    
    def __repr__(self):
        return (
            f"GlassRouterConfig(mode='{self.mode}', "
            f"pass1=[{self.min_pass1_rules}-{self.max_pass1_rules}], "
            f"pass2=[{self.min_pass2_rules}-{self.max_pass2_rules}], "
            f"complexity={self.max_complexity}, "
            f"pass2_population='{self.pass2_population}')"
        )
