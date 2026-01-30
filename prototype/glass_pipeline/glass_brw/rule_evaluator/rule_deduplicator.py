# ============================================================
# GLASS-BRW: RULE DEDUPLICATOR MODULE
# ============================================================
# Remove duplicate rules via structural validation and segment matching
# ============================================================


class RuleDeduplicator:
    """Remove duplicate rules and structurally invalid rules."""
    
    def __init__(self, validator):
        """
        Initialize rule deduplicator.
        
        Args:
            validator: FeatureValidator instance for structural validation
        """
        self.validator = validator
    
    def deduplicate_rules(self, rules: list) -> list:
        """
        Remove duplicate rules via structural validation and segment matching.
        
        Two-Phase Process:
            Phase 1: Remove structurally invalid rules (duplicate base features)
            Phase 2: Remove exact segment duplicates (keep highest quality)
        
        Deduplication Key: (predicted_class, frozenset(segment))
        
        Quality Ranking:
            1. Precision (primary)
            2. Coverage (secondary)
            3. Recall (tertiary)
        
        Args:
            rules: Rule objects to deduplicate
            
        Returns:
            Deduplicated rules (unique, valid, highest-quality)
        """
        # ============================================================
        # PHASE 1: STRUCTURAL VALIDATION
        # ============================================================
        valid_rules = []
        
        for rule in rules:
            if self.validator.has_duplicate_base_features(rule.segment):
                # Format segment for logging
                seg_str = ' AND '.join([f"{f}={v}" for f, v in list(rule.segment)[:3]])
                if len(rule.segment) > 3:
                    seg_str += f" ... (+{len(rule.segment)-3})"
                print(f"  ⚠️  Filtering rule with duplicate base features: {seg_str}")
                continue
            valid_rules.append(rule)
        
        if len(valid_rules) < len(rules):
            print(f"  Filtered {len(rules) - len(valid_rules)} rules with duplicate base features")
        
        # ============================================================
        # PHASE 2: EXACT SEGMENT DEDUPLICATION
        # ============================================================
        unique = {}
        
        for rule in valid_rules:
            # Create deduplication key (order-invariant, class-specific)
            key = (rule.predicted_class, frozenset(rule.segment))
            
            if key not in unique:
                unique[key] = rule
            else:
                # Keep higher-quality rule
                existing = unique[key]
                if (rule.precision > existing.precision or
                    (rule.precision == existing.precision and rule.coverage > existing.coverage) or
                    (rule.precision == existing.precision and
                     rule.coverage == existing.coverage and
                     rule.recall > existing.recall)):
                    unique[key] = rule
        
        return list(unique.values())