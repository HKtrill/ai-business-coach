# ============================================================
# GLASS-BRW: ILP DEDUPLICATOR MODULE
# ============================================================
# Deduplicate rules for ILP selection (with scoring mode support)
# ============================================================


class ILPDeduplicator:
    """Deduplicate rules for ILP with scoring-mode-aware quality ranking."""
    
    def deduplicate_by_segment(self, rules: list, scoring_mode: str) -> list:
        """
        Remove exact segment duplicates, keeping highest quality rule.
        
        Quality ranking depends on scoring mode:
        - "precision_first": Precision primary, then coverage
        - "recall_first": Recall primary, then precision
        
        Args:
            rules: List of Rule objects
            scoring_mode: "precision_first" or "recall_first"
            
        Returns:
            List of deduplicated Rule objects
        """
        unique = {}
        
        for rule in rules:
            # Create signature from segment (order-independent)
            sig = frozenset(rule.segment)
            
            if sig not in unique:
                unique[sig] = rule
            else:
                # Keep rule with better quality based on scoring mode
                if scoring_mode == "precision_first":
                    if rule.precision > unique[sig].precision:
                        unique[sig] = rule
                    elif rule.precision == unique[sig].precision:
                        if rule.coverage > unique[sig].coverage:
                            unique[sig] = rule
                else:  # recall_first
                    if rule.recall > unique[sig].recall:
                        unique[sig] = rule
                    elif rule.recall == unique[sig].recall:
                        if rule.precision > unique[sig].precision:
                            unique[sig] = rule
        
        return list(unique.values())