# ============================================================
# GLASS ROUTER: OVERLAP ANALYZER MODULE
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
            
            # Compute pairwise Jaccard
            overlaps = self.pairwise_jaccards(pass_rules, segments_df)
            
            # Log statistics
            if overlaps:
                avg_overlap = np.mean(overlaps)
                max_overlap = np.max(overlaps)
                print(f"\n{pass_name} overlap diagnostics:")
                print(f"  Average Jaccard: {avg_overlap:.3f}")
                print(f"  Maximum Jaccard: {max_overlap:.3f}")
                print(f"  Rules with >0.5 overlap: {sum(1 for o in overlaps if o > 0.5)} / {len(overlaps)}")
    
    def pairwise_jaccards(self, rules: list, segments_df) -> list:
        """
        Jaccard similarity of every rule pair (i, j), i < j, in row-major order.

        Each rule is matched once (len(rules) calls to match_rule) instead of
        once per pair. Intersection and union sizes are exact integer counts
        (|A u B| = |A| + |B| - |A n B|), so every value equals the historical
        pairwise loop bit for bit (PR 37 Commit 4, Step 4B).
        """
        if len(rules) < 2:
            return []
        masks = np.vstack([
            np.asarray(self.matcher.match_rule(rule, segments_df), dtype=bool)
            for rule in rules
        ])
        sizes = np.count_nonzero(masks, axis=1).astype(np.int64)
        overlaps = []
        for i in range(len(rules) - 1):
            inter = np.count_nonzero(masks[i + 1:] & masks[i], axis=1).astype(np.int64)
            union = sizes[i] + sizes[i + 1:] - inter
            for intersection, size in zip(inter, union):
                overlaps.append(intersection / size if size > 0 else 0)
        return overlaps

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