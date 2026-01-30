# ============================================================
# GLASS-BRW: RULE LOGGER MODULE
# ============================================================
# Diagnostic logging for accepted/rejected rules with detailed metrics
# ============================================================
from collections import defaultdict


class RuleLogger:
    """Comprehensive diagnostic logging for rule generation."""
    
    def __init__(
        self,
        verbose_rejection_logging: bool = True,
        max_rejection_logs_per_constraint: int = 50,
        verbose_acceptance_logging: bool = True,
        max_acceptance_logs_per_depth: int = 25,
    ):
        """
        Initialize rule logger.
        
        Args:
            verbose_rejection_logging: Enable detailed rejection logs
            max_rejection_logs_per_constraint: Max rejection logs per constraint type
            verbose_acceptance_logging: Enable detailed acceptance logs
            max_acceptance_logs_per_depth: Max acceptance logs per depth
        """
        self.verbose_rejection_logging = verbose_rejection_logging
        self.max_rejection_logs_per_constraint = max_rejection_logs_per_constraint
        self.verbose_acceptance_logging = verbose_acceptance_logging
        self.max_acceptance_logs_per_depth = max_acceptance_logs_per_depth
        
        # Tracking counters (reset per depth)
        self.rejection_log_counts = defaultdict(lambda: {0: 0, 1: 0})
        self.acceptance_log_counts = {0: 0, 1: 0}
    
    def reset_depth_counters(self):
        """Reset logging counters for new depth."""
        self.rejection_log_counts = defaultdict(lambda: {0: 0, 1: 0})
        self.acceptance_log_counts = {0: 0, 1: 0}
    
    def log_accepted_rule(
        self,
        seg: set,
        depth: int,
        predicted_class: int,
        precision: float,
        recall: float,
        coverage: float,
        support: int,
        parent_precision: float = None,
        parent_recall: float = None,
        parent_coverage: float = None,
        parent_segment: set = None,
        leakage_rate: float = None,
        subscribers_caught: int = None,
    ):
        """Log comprehensive metrics for an accepted rule."""
        # Check if we should log this rule
        if not self.verbose_acceptance_logging:
            return
        
        if self.acceptance_log_counts[predicted_class] >= self.max_acceptance_logs_per_depth:
            return
        
        self.acceptance_log_counts[predicted_class] += 1
        
        pass_name = "Pass1" if predicted_class == 0 else "Pass2"
        seg_sorted = sorted(seg, key=lambda x: x[0])
        seg_str = "{" + ", ".join([f"({f!r}, {v})" for f, v in seg_sorted]) + "}"
        
        print(f"\n{'─'*70}")
        print(f"✅ ACCEPTED: {seg_str}")
        print(f"   Depth: {depth} | {pass_name}")
        
        # Show parent segment if available
        if parent_segment is not None:
            parent_sorted = sorted(parent_segment, key=lambda x: x[0])
            parent_str = "{" + ", ".join([f"({f!r}, {v})" for f, v in parent_sorted]) + "}"
            
            # Find the new feature added
            new_features = seg - parent_segment
            if new_features:
                new_feat_str = ", ".join([f"({f!r}, {v})" for f, v in new_features])
                print(f"   Parent: {parent_str}")
                print(f"   Added:  {new_feat_str}")
        
        print(f"{'─'*70}")
        
        # Precision
        print(f"  Precision: {precision:.4f}", end="")
        if parent_precision is not None:
            prec_delta = precision - parent_precision
            print(f"  (parent: {parent_precision:.4f}, Δ = {prec_delta:+.4f})")
        else:
            print()
        
        # Recall
        print(f"  Recall:    {recall:.4f}", end="")
        if parent_recall is not None:
            rec_delta = recall - parent_recall
            print(f"  (parent: {parent_recall:.4f}, Δ = {rec_delta:+.4f})")
        else:
            print()
        
        # Coverage
        print(f"  Coverage:  {coverage:.4f}", end="")
        if parent_coverage is not None:
            cov_delta = coverage - parent_coverage
            print(f"  (parent: {parent_coverage:.4f}, Δ = {cov_delta:+.4f})")
        else:
            print()
        
        print(f"  Support:   {support}")
        
        # Leakage/Subscribers
        if leakage_rate is not None:
            if predicted_class == 0:
                print(f"  Leakage:   {subscribers_caught} absolute (rate: {leakage_rate:.4f})")
            else:
                print(f"  Subscribers caught: {subscribers_caught} (recall: {leakage_rate:.4f})")
        
        # Score info
        if predicted_class == 0:
            import numpy as np
            score = (precision ** 3) * np.sqrt(coverage)
            print(f"  Score:     {score:.6f}  (prec³ × √cov)")
        else:
            score = (recall ** 3) * precision * coverage
            print(f"  Score:     {score:.6f}  (rec³ × prec × cov)")
        
        # F1 proxy
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        print(f"  F1 proxy:  {f1:.4f}")
        print(f"{'─'*70}")
    
    def log_rejected_rule(
        self,
        seg: set,
        depth: int,
        rejection_reason: str,
        predicted_class: int,
        precision: float,
        recall: float,
        coverage: float,
        support: int,
        parent_precision: float,
        parent_recall: float,
        parent_coverage: float,
        parent_segment: set = None,
        leakage_rate: float = None,
        subscribers_caught: int = None,
        threshold_info: str = None,
    ):
        """Log comprehensive metrics for a rejected rule."""
        # Check if we should log this rule
        if not self.verbose_rejection_logging:
            return
        
        if self.rejection_log_counts[rejection_reason][predicted_class] >= self.max_rejection_logs_per_constraint:
            return
        
        self.rejection_log_counts[rejection_reason][predicted_class] += 1
        
        pass_name = "Pass1" if predicted_class == 0 else "Pass2"
        seg_sorted = sorted(seg, key=lambda x: x[0])
        seg_str = "{" + ", ".join([f"({f!r}, {v})" for f, v in seg_sorted]) + "}"
        
        print(f"\n{'─'*70}")
        print(f"🔍 REJECTED: {seg_str}")
        print(f"   Reason: {rejection_reason} | Depth: {depth} | {pass_name}")
        
        # Show parent segment if available
        if parent_segment is not None:
            parent_sorted = sorted(parent_segment, key=lambda x: x[0])
            parent_str = "{" + ", ".join([f"({f!r}, {v})" for f, v in parent_sorted]) + "}"
            new_features = seg - parent_segment
            if new_features:
                new_feat_str = ", ".join([f"({f!r}, {v})" for f, v in new_features])
                print(f"   Parent: {parent_str}")
                print(f"   Added:  {new_feat_str}")
        
        print(f"{'─'*70}")
        
        # Metrics with deltas
        prec_delta = precision - parent_precision
        rec_delta = recall - parent_recall
        cov_delta = coverage - parent_coverage
        
        print(f"  Precision: {precision:.4f}  (parent: {parent_precision:.4f}, Δ = {prec_delta:+.4f})")
        print(f"  Recall:    {recall:.4f}  (parent: {parent_recall:.4f}, Δ = {rec_delta:+.4f})")
        print(f"  Coverage:  {coverage:.4f}  (parent: {parent_coverage:.4f}, Δ = {cov_delta:+.4f})")
        print(f"  Support:   {support}")
        
        # Leakage/Subscribers with thresholds
        if leakage_rate is not None:
            if predicted_class == 0:
                print(f"  Leakage:   {subscribers_caught} absolute (rate: {leakage_rate:.4f})")
            else:
                print(f"  Subscribers: {subscribers_caught} caught (recall: {leakage_rate:.4f})")
        
        if threshold_info:
            print(f"  Threshold: {threshold_info}")
        
        # Derived metrics
        print(f"  ── Derived ──")
        if prec_delta < 0 and rec_delta > 0:
            tradeoff_ratio = abs(rec_delta) / abs(prec_delta) if prec_delta != 0 else float('inf')
            print(f"  P-R Tradeoff: {tradeoff_ratio:.2f}x recall gain per precision loss")
        elif prec_delta > 0 and rec_delta > 0:
            print(f"  P-R Tradeoff: BOTH improved (precision +{prec_delta:.4f}, recall +{rec_delta:.4f})")
        elif prec_delta < 0 and rec_delta < 0:
            print(f"  P-R Tradeoff: BOTH degraded (precision {prec_delta:.4f}, recall {rec_delta:.4f})")
        
        # F1 comparison
        parent_f1 = 2 * parent_precision * parent_recall / (parent_precision + parent_recall + 1e-10)
        child_f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_delta = child_f1 - parent_f1
        print(f"  F1 proxy:  {child_f1:.4f}  (parent: {parent_f1:.4f}, Δ = {f1_delta:+.4f})")
        print(f"{'─'*70}")
    
    def get_log_summary(self) -> dict:
        """Get summary of logs emitted for current depth."""
        return {
            'rejection_counts': dict(self.rejection_log_counts),
            'acceptance_counts': self.acceptance_log_counts.copy(),
        }