# ============================================================
# GLASS-BRW: QUALITY GATE FILTER MODULE
# ============================================================
# Apply quality thresholds for Pass 1 and Pass 2 rules
# Works with EvaluatedRule objects
# ============================================================

from typing import List, Tuple, Any, Optional
from collections import defaultdict

from glass_brw.core.rule import EvaluatedRule


class QualityGateFilter:
    """
    Apply quality gate constraints to filter rules.
    
    Works with EvaluatedRule objects.
    Uses _cached_covered_idx for subscriber leakage calculations.
    """
    
    def __init__(
        self,
        # Pass 1 thresholds
        min_precision_pass1: Optional[float] = None,
        max_precision_pass1: Optional[float] = None,
        max_subscriber_leakage_rate_pass1: Optional[float] = None,
        max_subscriber_leakage_absolute_pass1: Optional[int] = None,
        min_coverage_pass1: Optional[float] = None,
        max_coverage_pass1: Optional[float] = None,
        # Pass 2 thresholds
        min_precision_pass2: Optional[float] = None,
        max_precision_pass2: Optional[float] = None,
        min_recall_pass2: Optional[float] = None,
        max_recall_pass2: Optional[float] = None,
        min_coverage_pass2: Optional[float] = None,
        max_coverage_pass2: Optional[float] = None,
    ):
        # Pass 1 constraints
        self.min_precision_pass1 = min_precision_pass1
        self.max_precision_pass1 = max_precision_pass1
        self.max_subscriber_leakage_rate_pass1 = max_subscriber_leakage_rate_pass1
        self.max_subscriber_leakage_absolute_pass1 = max_subscriber_leakage_absolute_pass1
        self.min_coverage_pass1 = min_coverage_pass1
        self.max_coverage_pass1 = max_coverage_pass1
        
        # Pass 2 constraints
        self.min_precision_pass2 = min_precision_pass2
        self.max_precision_pass2 = max_precision_pass2
        self.min_recall_pass2 = min_recall_pass2
        self.max_recall_pass2 = max_recall_pass2
        self.min_coverage_pass2 = min_coverage_pass2
        self.max_coverage_pass2 = max_coverage_pass2
    
    def _get_covered_indices(self, rule: EvaluatedRule) -> set:
        """Get covered indices from cached attribute or empty set."""
        if hasattr(rule, '_cached_covered_idx'):
            return rule._cached_covered_idx
        return set()
    
    def apply_quality_gates_pass1(
        self, 
        candidates: List[EvaluatedRule], 
        y_val: Any
    ) -> Tuple[List[EvaluatedRule], List[Tuple]]:
        valid, rejected = [], []
        total_subscribers = (y_val == 1).sum()
        reject_reasons = defaultdict(int)
        
        for rule in candidates:
            covered_idx = self._get_covered_indices(rule)
            
            if covered_idx:
                subscribers_in_rule = sum(
                    1 for idx in covered_idx
                    if (y_val.iloc[idx] if hasattr(y_val, "iloc") else y_val[idx]) == 1
                )
            else:
                subscribers_in_rule = int(rule.support * (1 - rule.precision))
            
            leakage_rate = subscribers_in_rule / total_subscribers if total_subscribers > 0 else 0.0
            
            failed = False
            
            if rule.precision < self.min_precision_pass1:
                reject_reasons['precision_low'] += 1
                failed = True
            if rule.precision > self.max_precision_pass1:
                reject_reasons['precision_high'] += 1
                failed = True
            if leakage_rate > self.max_subscriber_leakage_rate_pass1:
                reject_reasons['leakage_rate'] += 1
                failed = True
            if subscribers_in_rule > self.max_subscriber_leakage_absolute_pass1:
                reject_reasons['leakage_abs'] += 1
                failed = True
            if self.min_coverage_pass1 is not None and rule.coverage < self.min_coverage_pass1:
                reject_reasons['coverage_low'] += 1
                failed = True
            if self.max_coverage_pass1 is not None and rule.coverage > self.max_coverage_pass1:
                reject_reasons['coverage_high'] += 1
                failed = True
            
            if failed:
                rejected.append((rule, leakage_rate, subscribers_in_rule))
            else:
                valid.append(rule)
        
        self._print_diagnostics_pass1(candidates, valid, rejected, reject_reasons)
        
        return valid, rejected
    
    def apply_quality_gates_pass2(
        self, 
        candidates: List[EvaluatedRule]
    ) -> Tuple[List[EvaluatedRule], List[EvaluatedRule]]:
        valid, rejected = [], []
        reject_reasons = defaultdict(int)
        
        for rule in candidates:
            failed = False
            
            if rule.precision < self.min_precision_pass2:
                reject_reasons['precision_low'] += 1
                failed = True
            if rule.precision > self.max_precision_pass2:
                reject_reasons['precision_high'] += 1
                failed = True
            if rule.recall < self.min_recall_pass2:
                reject_reasons['recall_low'] += 1
                failed = True
            if rule.recall > self.max_recall_pass2:
                reject_reasons['recall_high'] += 1
                failed = True
            if self.min_coverage_pass2 is not None and rule.coverage < self.min_coverage_pass2:
                reject_reasons['coverage_low'] += 1
                failed = True
            if self.max_coverage_pass2 is not None and rule.coverage > self.max_coverage_pass2:
                reject_reasons['coverage_high'] += 1
                failed = True
            
            if failed:
                rejected.append(rule)
            else:
                valid.append(rule)
        
        self._print_diagnostics_pass2(candidates, valid, rejected, reject_reasons)
        
        return valid, rejected
    
    def _print_diagnostics_pass1(self, candidates, valid, rejected, reject_reasons):
        print(f"\n🔎 Pass 1 Quality Gate Diagnostics")
        print(f"   Candidates:              {len(candidates)}")
        print(f"   Passed:                  {len(valid)}")
        print(f"   Rejected:                {len(rejected)}")
        print(f"   Constraints: prec=[{self.min_precision_pass1}, {self.max_precision_pass1}], "
              f"cov=[{self.min_coverage_pass1}, {self.max_coverage_pass1}]")
        
        if reject_reasons:
            print(f"   Rejection breakdown:")
            for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
                print(f"     {reason}: {count}")
    
    def _print_diagnostics_pass2(self, candidates, valid, rejected, reject_reasons):
        print("\n🔎 Pass 2 Quality Gate Diagnostics")
        print(f"   Candidates:              {len(candidates)}")
        print(f"   Passed:                  {len(valid)}")
        print(f"   Rejected:                {len(rejected)}")
        print(f"   Constraints: prec=[{self.min_precision_pass2}, {self.max_precision_pass2}], "
              f"recall=[{self.min_recall_pass2}, {self.max_recall_pass2}], "
              f"cov=[{self.min_coverage_pass2}, {self.max_coverage_pass2}]")
        
        if reject_reasons:
            print(f"   Rejection breakdown:")
            for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
                print(f"     {reason}: {count}")
        
        if valid:
            print(f"\n   Sample VALID rules (first 5):")
            for r in sorted(valid, key=lambda x: -x.precision)[:5]:
                seg_str = r.segment_str if hasattr(r, 'segment_str') else str(list(r.segment)[:2])
                print(f"     prec={r.precision:.3f}, recall={r.recall:.3f}, "
                      f"cov={r.coverage:.3f}, segment={seg_str}")