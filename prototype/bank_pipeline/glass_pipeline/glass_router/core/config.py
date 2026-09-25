# ============================================================
# GLASS ROUTER: CONFIGURATION MODULE
# ============================================================
# Single source of truth for Stage 2 experiment parameters.
# Required parameters default to None and must be set explicitly
# by the notebook. Optional parameters may remain None.
# ============================================================

from dataclasses import dataclass, fields
from typing import Optional


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

    def __post_init__(self):
        """Validate that all required parameters are set."""
        optional_params = {
            'max_feature_reuse_pass1',
            'max_feature_reuse_pass2',
            'lambda_rf_uncertainty',
            'lambda_rf_misalignment',
            'pass2_population',
            'pass1_oof_folds',
            'random_state',
            'remainder_min_size',
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