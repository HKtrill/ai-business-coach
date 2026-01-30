# ============================================================
# GLASS-BRW: NOVELTY ANALYZER MODULE
# ============================================================
# Compute pairwise overlap and novelty metrics for rules
# ============================================================
from pulp import lpSum


class NoveltyAnalyzer:
    """Compute novelty and overlap metrics for rule selection."""
    
    def __init__(self, enable_novelty_constraints: bool = True):
        """
        Initialize novelty analyzer.
        
        Args:
            enable_novelty_constraints: Whether to enable novelty constraints
        """
        self.enable_novelty_constraints = enable_novelty_constraints
    
    def compute_pairwise_overlap(self, rule_i, rule_j) -> tuple:
        """
        Compute overlap ratios between two rules.
        
        Args:
            rule_i: First Rule object
            rule_j: Second Rule object
            
        Returns:
            (overlap_ratio_i, overlap_ratio_j, jaccard) tuple
        """
        covered_i = set(rule_i.covered_idx)
        covered_j = set(rule_j.covered_idx)
        
        if not covered_i or not covered_j:
            return 0.0, 0.0, 0.0
        
        intersection = len(covered_i & covered_j)
        union = len(covered_i | covered_j)
        
        overlap_ratio_i = intersection / len(covered_i)
        overlap_ratio_j = intersection / len(covered_j)
        jaccard = intersection / union if union > 0 else 0.0
        
        return overlap_ratio_i, overlap_ratio_j, jaccard
    
    def add_novelty_constraints(
        self,
        prob,
        rules: list,
        decision_vars: dict,
        min_novelty_ratio: float
    ) -> int:
        """
        Add pairwise constraints for rules with extremely high overlap.
        
        Logic: Only constrain pairs where BOTH rules have overlap > max_overlap_ratio.
        This is less restrictive than constraining when EITHER has high overlap.
        
        Args:
            prob: PuLP problem object
            rules: List of Rule objects
            decision_vars: Dict mapping rule_id to LpVariable
            min_novelty_ratio: Minimum novelty ratio required
            
        Returns:
            Number of constraints added
        """
        if not self.enable_novelty_constraints:
            print("  ⚠️  Novelty constraints DISABLED")
            return 0
        
        max_overlap_ratio = 1.0 - min_novelty_ratio
        
        print(f"\n  📊 Computing pairwise overlaps for novelty constraints...")
        print(f"     Min novelty ratio: {min_novelty_ratio:.0%}")
        print(f"     Max overlap ratio: {max_overlap_ratio:.0%}")
        print(f"     Logic: Constrain if BOTH rules have overlap > {max_overlap_ratio:.0%}")
        
        n_rules = len(rules)
        constraints_added = 0
        high_overlap_pairs = []
        
        for i in range(n_rules):
            for j in range(i + 1, n_rules):
                rule_i = rules[i]
                rule_j = rules[j]
                
                overlap_i, overlap_j, jaccard = self.compute_pairwise_overlap(rule_i, rule_j)
                
                # Only constrain if BOTH have high overlap (more permissive)
                if overlap_i > max_overlap_ratio and overlap_j > max_overlap_ratio:
                    prob += decision_vars[rule_i.rule_id] + decision_vars[rule_j.rule_id] <= 1
                    constraints_added += 1
                    
                    if len(high_overlap_pairs) < 10:
                        high_overlap_pairs.append({
                            'rule_i': rule_i.rule_id,
                            'rule_j': rule_j.rule_id,
                            'overlap_i': overlap_i,
                            'overlap_j': overlap_j,
                            'jaccard': jaccard,
                        })
        
        print(f"     Pairs evaluated: {n_rules * (n_rules - 1) // 2}")
        print(f"     Constraints added: {constraints_added}")
        
        if high_overlap_pairs:
            print(f"\n     Sample high-overlap pairs (first 10):")
            for pair in high_overlap_pairs[:10]:
                print(f"       Rule {pair['rule_i']} ↔ Rule {pair['rule_j']}: "
                      f"overlap_i={pair['overlap_i']:.1%}, overlap_j={pair['overlap_j']:.1%}, "
                      f"jaccard={pair['jaccard']:.1%}")
        
        return constraints_added
    
    def analyze_selection_novelty(self, selected_rules: list, pass_name: str):
        """
        Analyze the novelty of selected rules.
        
        Args:
            selected_rules: List of selected Rule objects
            pass_name: Name of the pass (for logging)
        """
        if len(selected_rules) < 2:
            return
        
        print(f"\n📈 {pass_name} Selection Novelty Analysis:")
        
        all_covered = set()
        novelty_ratios = []
        
        for i, rule in enumerate(selected_rules):
            rule_covered = set(rule.covered_idx)
            
            if i == 0:
                novelty = 1.0
            else:
                new_samples = rule_covered - all_covered
                novelty = len(new_samples) / len(rule_covered) if rule_covered else 0.0
            
            novelty_ratios.append(novelty)
            all_covered |= rule_covered
            
            print(f"   Rule {i+1} (id={rule.rule_id}): "
                  f"covers {len(rule_covered)}, "
                  f"novelty={novelty:.1%}, "
                  f"cumulative={len(all_covered)}")
        
        avg_novelty = sum(novelty_ratios[1:]) / len(novelty_ratios[1:]) if len(novelty_ratios) > 1 else 1.0
        print(f"   Average novelty (excluding first rule): {avg_novelty:.1%}")