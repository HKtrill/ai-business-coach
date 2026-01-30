# ============================================================
# GLASS-BRW: DIVERSITY ANALYZER MODULE
# ============================================================
# Compute diversity scores and add diversity constraints
# ============================================================
from collections import defaultdict
from pulp import lpSum


class DiversityAnalyzer:
    """Compute diversity metrics and constraints for rule selection."""
    
    def __init__(self, validator, max_feature_usage: int = 40):
        """
        Initialize diversity analyzer.
        
        Args:
            validator: FeatureValidator instance for extracting base features
            max_feature_usage: Maximum times a base feature can be used
        """
        self.validator = validator
        self.max_feature_usage = max_feature_usage
    
    def compute_diversity_score(self, rule, selected_rules: list) -> float:
        """
        Compute diversity score for a rule given already-selected rules.
        
        Diversity is measured as feature overlap with existing rules.
        Higher score = more diverse (less overlap).
        
        Args:
            rule: Rule object to evaluate
            selected_rules: List of already-selected Rule objects
            
        Returns:
            Diversity score ∈ [0, 1]
        """
        if not selected_rules:
            return 1.0
        
        rule_bases = {self.validator.extract_base_feature(f) for f, _ in rule.segment}
        overlaps = []
        
        for selected in selected_rules:
            selected_bases = {self.validator.extract_base_feature(f) for f, _ in selected.segment}
            overlap = len(rule_bases & selected_bases) / max(len(rule_bases), 1)
            overlaps.append(overlap)
        
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        return 1.0 - avg_overlap
    
    def add_diversity_constraints(
        self,
        prob,
        rules: list,
        decision_vars: dict,
        max_base_reuse: int = None
    ):
        """
        Add constraints to limit feature reuse across rules.
        
        Args:
            prob: PuLP problem object
            rules: List of Rule objects
            decision_vars: Dict mapping rule_id to LpVariable
            max_base_reuse: Max times a base feature can be used (overrides default)
        """
        # Track which rules use each base feature
        base_feature_usage = defaultdict(list)
        
        for rule in rules:
            for feature, _ in rule.segment:
                base = self.validator.extract_base_feature(feature)
                base_feature_usage[base].append(rule.rule_id)
        
        # Use provided limit or default
        limit = max_base_reuse if max_base_reuse is not None else self.max_feature_usage
        
        print(f"  Applying diversity constraint: max {limit} rules per base feature")
        
        # Add constraint for each base feature
        for base, rule_ids in base_feature_usage.items():
            prob += lpSum(decision_vars[rid] for rid in rule_ids) <= limit