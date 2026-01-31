# ============================================================
# GLASS-BRW: CORE RULE DATA STRUCTURES
# ============================================================
# Stage-typed rule classes for the GLASS-BRW pipeline:
#   CandidateRule → EvaluatedRule → SelectedRule
# ============================================================

from dataclasses import dataclass, field
from typing import Tuple, FrozenSet, Set, Union, Optional, List
import pandas as pd
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================
SUBSCRIBE = 1
NOT_SUBSCRIBE = 0
ABSTAIN = -1

# Type alias for segment
SegmentType = Union[FrozenSet[Tuple[str, int]], Set[Tuple[str, int]]]


def _normalize_segment(segment: SegmentType) -> FrozenSet[Tuple[str, int]]:
    """Convert segment to frozenset for immutability."""
    if isinstance(segment, frozenset):
        return segment
    return frozenset(segment)


# ============================================================
# CANDIDATE RULE (from RuleGenerator)
# ============================================================
@dataclass
class CandidateRule:
    """
    Rule candidate from RuleGenerator (beam search).
    
    Contains identity and initial training metrics.
    Does NOT contain pass_assignment (determined by ILPRuleSelector).
    
    Attributes:
        rule_id: Unique identifier
        segment: Frozenset of (feature, value) conditions
        predicted_class: 0 (NOT_SUBSCRIBE) or 1 (SUBSCRIBE)
        complexity: Number of conditions (depth)
        precision: Training precision
        recall: Training recall
        coverage: Training coverage
        support: Number of samples matched
    """
    rule_id: int
    segment: FrozenSet[Tuple[str, int]]
    predicted_class: int
    complexity: int
    precision: float
    recall: float
    coverage: float
    support: int
    
    # Internal beam score (not persisted, used during generation)
    _beam_score: float = field(default=0.0, repr=False, compare=False)
    
    def __post_init__(self):
        """Normalize segment to frozenset."""
        object.__setattr__(self, 'segment', _normalize_segment(self.segment))
    
    @property
    def segment_frozen(self) -> FrozenSet[Tuple[str, int]]:
        """Return segment as frozenset (for deduplication)."""
        return self.segment
    
    @property
    def segment_str(self) -> str:
        """Human-readable segment string."""
        sorted_seg = sorted(self.segment, key=lambda x: x[0])
        return " AND ".join(f"{f}={v}" for f, v in sorted_seg)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "rule_id": self.rule_id,
            "segment": list(self.segment),
            "predicted_class": self.predicted_class,
            "complexity": self.complexity,
            "precision": self.precision,
            "recall": self.recall,
            "coverage": self.coverage,
            "support": self.support,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CandidateRule':
        """Deserialize from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            segment=frozenset(tuple(x) for x in data["segment"]),
            predicted_class=data["predicted_class"],
            complexity=data["complexity"],
            precision=data["precision"],
            recall=data["recall"],
            coverage=data["coverage"],
            support=data["support"],
        )


# ============================================================
# EVALUATED RULE (from RuleEvaluator)
# ============================================================
@dataclass
class EvaluatedRule:
    """
    Rule evaluated on validation data by RuleEvaluator.
    
    Adds RF diagnostics and validation metrics.
    covered_idx is computed on-demand, not stored.
    
    Attributes:
        rule_id: Unique identifier
        segment: Frozenset of (feature, value) conditions
        predicted_class: 0 or 1
        complexity: Number of conditions
        precision: Validation precision
        recall: Validation recall
        coverage: Validation coverage
        support: Validation support
        rf_confidence: RF confidence (mean distance from 0.5)
        rf_alignment: RF alignment (fraction with high confidence)
    """
    rule_id: int
    segment: FrozenSet[Tuple[str, int]]
    predicted_class: int
    complexity: int
    precision: float
    recall: float
    coverage: float
    support: int
    rf_confidence: float = 0.5
    rf_alignment: float = 0.0
    
    # Cached covered indices (set by ILPRuleSelector, not persisted)
    _cached_covered_idx: Optional[Set[int]] = field(default=None, repr=False, compare=False)
    
    def __post_init__(self):
        """Normalize segment to frozenset."""
        object.__setattr__(self, 'segment', _normalize_segment(self.segment))
    
    @property
    def segment_frozen(self) -> FrozenSet[Tuple[str, int]]:
        """Return segment as frozenset."""
        return self.segment
    
    @property
    def segment_str(self) -> str:
        """Human-readable segment string."""
        sorted_seg = sorted(self.segment, key=lambda x: x[0])
        return " AND ".join(f"{f}={v}" for f, v in sorted_seg)
    
    def compute_covered_indices(
        self,
        X: pd.DataFrame,
        segment_builder
    ) -> Set[int]:
        """
        Compute indices of samples covered by this rule.
        
        Args:
            X: Features DataFrame
            segment_builder: SegmentBuilder for discretization
            
        Returns:
            Set of covered sample indices
        """
        segments_df = segment_builder.assign_segments(X)
        mask = pd.Series(True, index=segments_df.index)
        
        for feature, level in self.segment:
            if feature in segments_df.columns:
                mask &= (segments_df[feature] == level)
            else:
                return set()
        
        return set(np.flatnonzero(mask.values))
    
    @classmethod
    def from_candidate(
        cls,
        candidate: CandidateRule,
        rf_confidence: float = 0.5,
        rf_alignment: float = 0.0,
    ) -> 'EvaluatedRule':
        """Create from CandidateRule with RF metrics."""
        return cls(
            rule_id=candidate.rule_id,
            segment=candidate.segment,
            predicted_class=candidate.predicted_class,
            complexity=candidate.complexity,
            precision=candidate.precision,
            recall=candidate.recall,
            coverage=candidate.coverage,
            support=candidate.support,
            rf_confidence=rf_confidence,
            rf_alignment=rf_alignment,
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "rule_id": self.rule_id,
            "segment": list(self.segment),
            "predicted_class": self.predicted_class,
            "complexity": self.complexity,
            "precision": self.precision,
            "recall": self.recall,
            "coverage": self.coverage,
            "support": self.support,
            "rf_confidence": self.rf_confidence,
            "rf_alignment": self.rf_alignment,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluatedRule':
        """Deserialize from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            segment=frozenset(tuple(x) for x in data["segment"]),
            predicted_class=data["predicted_class"],
            complexity=data["complexity"],
            precision=data["precision"],
            recall=data["recall"],
            coverage=data["coverage"],
            support=data["support"],
            rf_confidence=data.get("rf_confidence", 0.5),
            rf_alignment=data.get("rf_alignment", 0.0),
        )


# ============================================================
# SELECTED RULE (from ILPRuleSelector)
# ============================================================
@dataclass
class SelectedRule:
    """
    Final rule selected by ILPRuleSelector for inference.
    
    Lightweight structure optimized for prediction.
    
    Attributes:
        rule_id: Unique identifier
        segment: Frozenset of (feature, value) conditions
        predicted_class: 0 or 1
        complexity: Number of conditions
        precision: Final precision
        recall: Final recall
        coverage: Final coverage
        pass_assignment: "pass1" or "pass2"
        rf_alignment: RF alignment score
    """
    rule_id: int
    segment: FrozenSet[Tuple[str, int]]
    predicted_class: int
    complexity: int
    precision: float
    recall: float
    coverage: float
    pass_assignment: str  # "pass1" or "pass2"
    rf_alignment: float = 0.0
    
    def __post_init__(self):
        """Normalize segment and validate pass_assignment."""
        object.__setattr__(self, 'segment', _normalize_segment(self.segment))
        if self.pass_assignment not in ("pass1", "pass2"):
            raise ValueError(f"pass_assignment must be 'pass1' or 'pass2', got {self.pass_assignment}")
    
    @property
    def segment_frozen(self) -> FrozenSet[Tuple[str, int]]:
        """Return segment as frozenset."""
        return self.segment
    
    @property
    def segment_str(self) -> str:
        """Human-readable segment string."""
        sorted_seg = sorted(self.segment, key=lambda x: x[0])
        return " AND ".join(f"{f}={v}" for f, v in sorted_seg)
    
    def matches(self, row: pd.Series) -> bool:
        """
        Check if this rule matches a sample.
        
        Args:
            row: Series with segment features
            
        Returns:
            True if all conditions are satisfied
        """
        return all(row[f] == v for f, v in self.segment)
    
    @classmethod
    def from_evaluated(
        cls,
        evaluated: EvaluatedRule,
        pass_assignment: str
    ) -> 'SelectedRule':
        """Create from EvaluatedRule with pass assignment."""
        return cls(
            rule_id=evaluated.rule_id,
            segment=evaluated.segment,
            predicted_class=evaluated.predicted_class,
            complexity=evaluated.complexity,
            precision=evaluated.precision,
            recall=evaluated.recall,
            coverage=evaluated.coverage,
            pass_assignment=pass_assignment,
            rf_alignment=evaluated.rf_alignment,
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "rule_id": self.rule_id,
            "segment": list(self.segment),
            "predicted_class": self.predicted_class,
            "complexity": self.complexity,
            "precision": self.precision,
            "recall": self.recall,
            "coverage": self.coverage,
            "pass_assignment": self.pass_assignment,
            "rf_alignment": self.rf_alignment,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SelectedRule':
        """Deserialize from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            segment=frozenset(tuple(x) for x in data["segment"]),
            predicted_class=data["predicted_class"],
            complexity=data["complexity"],
            precision=data["precision"],
            recall=data["recall"],
            coverage=data["coverage"],
            pass_assignment=data["pass_assignment"],
            rf_alignment=data.get("rf_alignment", 0.0),
        )


# ============================================================
# CONVERSION UTILITIES
# ============================================================

def candidates_to_evaluated(
    candidates: List[CandidateRule],
    evaluator_fn,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    rf_model=None,
) -> List[EvaluatedRule]:
    """
    Batch convert CandidateRule to EvaluatedRule.
    
    Args:
        candidates: List of CandidateRule objects
        evaluator_fn: Function that evaluates candidates
        X_val: Validation features
        y_val: Validation labels
        rf_model: Optional RF model for confidence metrics
        
    Returns:
        List of EvaluatedRule objects
    """
    return evaluator_fn(candidates, X_val, y_val, rf_model)


def evaluated_to_selected(
    evaluated: List[EvaluatedRule],
    pass_assignments: Optional[List[str]] = None
) -> List[SelectedRule]:
    """
    Batch convert EvaluatedRule to SelectedRule.
    
    Args:
        evaluated: List of EvaluatedRule objects
        pass_assignments: Optional list of pass assignments.
                         If None, inferred from predicted_class.
        
    Returns:
        List of SelectedRule objects
    """
    if pass_assignments is None:
        pass_assignments = [
            "pass1" if r.predicted_class == 0 else "pass2"
            for r in evaluated
        ]
    
    return [
        SelectedRule.from_evaluated(r, pa)
        for r, pa in zip(evaluated, pass_assignments)
    ]


# ============================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================
# During migration, existing code can use `Rule` as alias
Rule = EvaluatedRule