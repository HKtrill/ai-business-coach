# ============================================================
# GLASS-BRW: CONFIGURATION MODULE
# ============================================================
# Single source of truth for all pipeline tuning parameters.
# All tuning params default to None - must be set explicitly in notebook.
# ============================================================

from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class GLASSBRWConfig:
    """
    Configuration for GLASS-BRW pipeline.
    
    All tuning parameters must be explicitly set (no hidden defaults).
    This ensures the notebook is the single source of truth for experiments.
    
    Usage:
        config = GLASSBRWConfig(
            mode="strict",
            min_support_pass1=125,
            min_support_pass2=75,
            ...
        )
    
    Raises:
        ValueError: If any required parameter is None after initialization
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
    
    def __post_init__(self):
        """Validate that all required parameters are set."""
        # Parameters that are allowed to be None
        optional_params = {
            'max_feature_reuse_pass1',
            'max_feature_reuse_pass2',
        }
        
        # Check all required params
        missing = []
        for f in fields(self):
            if f.name not in optional_params:
                if getattr(self, f.name) is None:
                    missing.append(f.name)
        
        if missing:
            raise ValueError(
                f"GLASSBRWConfig missing required parameters: {missing}\n"
                f"All tuning parameters must be explicitly set in your notebook."
            )
        
        # Validate mode
        valid_modes = {"strict", "relaxed", "exploratory"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {valid_modes}"
            )
        
        # Validate ranges
        self._validate_ranges()
    
    def _validate_ranges(self):
        """Validate parameter ranges make sense."""
        errors = []
        
        # Support must be positive
        if self.min_support_pass1 <= 0:
            errors.append("min_support_pass1 must be > 0")
        if self.min_support_pass2 <= 0:
            errors.append("min_support_pass2 must be > 0")
        
        # Precision/recall in [0, 1]
        for param in ['min_precision_not_subscribe', 'max_precision_not_subscribe',
                      'min_precision_subscribe', 'max_precision_subscribe',
                      'min_recall_subscribe', 'max_recall_subscribe',
                      'max_subscriber_leakage_rate', 'diversity_weight',
                      'min_novelty_ratio_pass1', 'min_novelty_ratio_pass2']:
            val = getattr(self, param)
            if not 0 <= val <= 1:
                errors.append(f"{param} must be in [0, 1], got {val}")
        
        # Min <= Max checks
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
        
        # Complexity must be positive
        if self.max_complexity <= 0:
            errors.append("max_complexity must be > 0")
        
        if errors:
            raise ValueError(
                f"GLASSBRWConfig validation errors:\n" + 
                "\n".join(f"  - {e}" for e in errors)
            )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GLASSBRWConfig':
        """Create config from dictionary."""
        return cls(**d)
    
    def __repr__(self):
        return (
            f"GLASSBRWConfig(mode='{self.mode}', "
            f"pass1=[{self.min_pass1_rules}-{self.max_pass1_rules}], "
            f"pass2=[{self.min_pass2_rules}-{self.max_pass2_rules}], "
            f"complexity={self.max_complexity})"
        )