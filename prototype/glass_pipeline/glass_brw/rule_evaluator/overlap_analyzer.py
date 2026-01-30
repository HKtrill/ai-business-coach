# ============================================================
# GLASS-BRW: OVERLAP ANALYZER MODULE
# ============================================================
# Compute and log pairwise Jaccard overlap between rules
# ============================================================
import numpy as np


class OverlapAnalyzer:
    """Compute overlap diagnostics for rule sets."""
    
    def __init__(self, matcher):
        """
        Initialize overlap analyzer.
        
        Args:
            matcher: RuleMatcher instance
        """
        self.matcher = matcher
    
    def compute_overlap_diagnostics(self, rules: list, segments_df):
        """
        Compute and log pairwise Jaccard overlap between rules.
        
        Jaccard Similarity:
            J(A, B) = |A ∩ B| / |A ∪ B|
        
        Interpretation:
            J = 0.0 → No overlap (complementary)
            J = 1.0 → Complete overlap (redundant)
        
        Computed separately per pass (Pass 1 / Pass 2).
        
        Args:
            rules: Evaluated Rule objects
            segments_df: Discretized validation features
        """
        # Split rules by pass
        pass1_rules = [r for r in rules if r.predicted_class == 0]
        pass2_rules = [r for r in rules if r.predicted_class == 1]
        
        for pass_name, pass_rules in [("Pass 1", pass1_rules), ("Pass 2", pass2_rules)]:
            if len(pass_rules) < 2:
                continue
            
            overlaps = []
            
            # Compute pairwise Jaccard
            for i, r1 in enumerate(pass_rules):
                mask1 = self.matcher.match_rule(r1, segments_df)
                for r2 in pass_rules[i+1:]:
                    mask2 = self.matcher.match_rule(r2, segments_df)
                    intersection = (mask1 & mask2).sum()
                    union = (mask1 | mask2).sum()
                    jaccard = intersection / union if union > 0 else 0
                    overlaps.append(jaccard)
            
            # Log statistics
            if overlaps:
                avg_overlap = np.mean(overlaps)
                max_overlap = np.max(overlaps)
                print(f"\n{pass_name} overlap diagnostics:")
                print(f"  Average Jaccard: {avg_overlap:.3f}")
                print(f"  Maximum Jaccard: {max_overlap:.3f}")
                print(f"  Rules with >0.5 overlap: {sum(1 for o in overlaps if o > 0.5)} / {len(overlaps)}")
    
    def compute_pairwise_overlap(self, rule1, rule2, segments_df) -> float:
        """
        Compute Jaccard overlap between two rules.
        
        Args:
            rule1: First Rule object
            rule2: Second Rule object
            segments_df: Discretized features
            
        Returns:
            Jaccard similarity ∈ [0, 1]
        """
        mask1 = self.matcher.match_rule(rule1, segments_df)
        mask2 = self.matcher.match_rule(rule2, segments_df)
        
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        
        return intersection / union if union > 0 else 0.0