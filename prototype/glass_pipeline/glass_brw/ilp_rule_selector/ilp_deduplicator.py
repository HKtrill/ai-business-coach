# ============================================================
# GLASS-BRW: ILP DEDUPLICATOR MODULE
# ============================================================
# Deduplicate rules for ILP selection (with scoring mode support)
# Works with EvaluatedRule objects
# ============================================================

from typing import List, Dict, FrozenSet, Tuple

from glass_brw.core.rule import EvaluatedRule


class ILPDeduplicator:
    """
    Deduplicate rules for ILP with scoring-mode-aware quality ranking.
    
    Works with EvaluatedRule objects.
    """
    
    def deduplicate_by_segment(
        self, 
        rules: List[EvaluatedRule], 
        scoring_mode: str
    ) -> List[EvaluatedRule]:
        """
        Remove exact segment duplicates, keeping highest quality rule.
        
        Quality ranking depends on scoring mode:
        - "precision_first": Precision primary, then coverage
        - "recall_first": Recall primary, then precision
        
        Args:
            rules: List of EvaluatedRule objects
            scoring_mode: "precision_first" or "recall_first"
            
        Returns:
            List of deduplicated EvaluatedRule objects
        """
        unique: Dict[FrozenSet[Tuple[str, str]], EvaluatedRule] = {}
        
        for rule in rules:
            # Create signature from segment (order-independent)
            sig = rule.segment_frozen
            
            if sig not in unique:
                unique[sig] = rule
            else:
                # Keep rule with better quality based on scoring mode
                existing = unique[sig]
                if self._is_better(rule, existing, scoring_mode):
                    unique[sig] = rule
        
        return list(unique.values())
    
    def _is_better(
        self, 
        rule: EvaluatedRule, 
        existing: EvaluatedRule, 
        scoring_mode: str
    ) -> bool:
        """
        Check if rule is better than existing based on scoring mode.
        
        Args:
            rule: New rule to compare
            existing: Existing rule
            scoring_mode: "precision_first" or "recall_first"
            
        Returns:
            True if rule is better than existing
        """
        if scoring_mode == "precision_first":
            if rule.precision > existing.precision:
                return True
            elif rule.precision == existing.precision:
                return rule.coverage > existing.coverage
            return False
        else:  # recall_first
            if rule.recall > existing.recall:
                return True
            elif rule.recall == existing.recall:
                return rule.precision > existing.precision
            return False
    
    def deduplicate_by_rule_id(
        self, 
        rules: List[EvaluatedRule]
    ) -> List[EvaluatedRule]:
        """
        Remove duplicate rule IDs (keep first occurrence).
        
        Args:
            rules: List of EvaluatedRule objects
            
        Returns:
            List of deduplicated EvaluatedRule objects
        """
        seen_ids = set()
        unique = []
        
        for rule in rules:
            if rule.rule_id not in seen_ids:
                seen_ids.add(rule.rule_id)
                unique.append(rule)
        
        return unique