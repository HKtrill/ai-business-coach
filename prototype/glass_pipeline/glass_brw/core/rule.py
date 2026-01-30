"""
Rule Data Structure for GLASS-BRW
==================================

Atomic rule representation for sequential execution.
Single source of truth for all rule state.

Author: Glass Pipeline Team
Date: 2026-01-29
"""

from dataclasses import dataclass, field
from typing import Set, Tuple


@dataclass
class Rule:
    """
    Atomic rule representation for GLASS-BRW sequential execution.
    
    Single source of truth for all rule state. No logic - just data.
    
    Lifecycle:
    1. RuleGenerator: Creates rule, sets identity + base metrics
    2. RuleEvaluator: Enriches with validation metrics + RF diagnostics
    3. ILPRuleSelector: Reads metrics to select optimal subset
    4. GLASS_BRW: Executes rules sequentially using pass_assignment
    """
    
    # ============================================================
    # IDENTITY & STRUCTURE
    # ============================================================
    rule_id: int
    segment: Set[Tuple[str, str]]  # Conjunction of (feature, value) conditions
    predicted_class: int  # 0=NOT_SUBSCRIBE, 1=SUBSCRIBE
    complexity: int  # Number of conditions (k ≤ 3)
    pass_assignment: str = "pass2"  # "pass1" or "pass2" for sequential execution
    
    # ============================================================
    # QUALITY METRICS
    # ============================================================
    # Set by: RuleGenerator (initial), RuleEvaluator (validation-based)
    precision: float = 0.0  # P(y=predicted_class | rule matches)
    recall: float = 0.0     # P(rule matches | y=predicted_class)
    coverage: float = 0.0   # Fraction of population covered
    
    # ============================================================
    # RF-SPECIFIC DIAGNOSTICS
    # ============================================================
    # Set by: RuleEvaluator
    rf_confidence: float = 0.5  # Mean |RF_proba - 0.5| in segment
    rf_alignment: float = 0.0   # Fraction of segment with RF_conf > 0.2
    
    # ============================================================
    # EXECUTION METADATA
    # ============================================================
    # Set by: RuleEvaluator (match_rule)
    covered_idx: Set[int] = field(default_factory=set)  # Sample indices matched by rule
    
    # Set by: RuleGenerator (beam search scoring)
    beam_search_score: float = 0.0
    
    def __hash__(self):
        return hash((self.rule_id, frozenset(self.segment), self.predicted_class))
    
    def __eq__(self, other):
        return self.rule_id == other.rule_id


# ============================================================
# CLASS CONSTANTS (module-level for system-wide use)
# ============================================================
SUBSCRIBE = 1
NOT_SUBSCRIBE = 0
ABSTAIN = -1  # Does not meet any rule params