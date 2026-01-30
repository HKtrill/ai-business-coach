# ============================================================
# GLASS-BRW: GREEDY SELECTOR MODULE
# ============================================================
# Greedy fallback selection when ILP fails
# ============================================================
import numpy as np


class GreedySelector:
    """Greedy rule selection with novelty and diversity constraints."""
    
    def __init__(
        self,
        diversity_analyzer,
        min_novelty_greedy: float = 0.15,
        greedy_novelty_weight: float = 0.5,
        greedy_hard_novelty_cutoff: bool = True,
        min_absolute_new_samples: int = 30,
        diversity_weight: float = 0.33,
    ):
        """
        Initialize greedy selector.
        
        Args:
            diversity_analyzer: DiversityAnalyzer instance
            min_novelty_greedy: Minimum novelty ratio for selection
            greedy_novelty_weight: Weight for novelty in scoring
            greedy_hard_novelty_cutoff: Whether to enforce hard novelty cutoff
            min_absolute_new_samples: Minimum new samples required
            diversity_weight: Weight for diversity in scoring
        """
        self.diversity_analyzer = diversity_analyzer
        self.min_novelty_greedy = min_novelty_greedy
        self.greedy_novelty_weight = greedy_novelty_weight
        self.greedy_hard_novelty_cutoff = greedy_hard_novelty_cutoff
        self.min_absolute_new_samples = min_absolute_new_samples
        self.diversity_weight = diversity_weight
    
    def greedy_select(
        self,
        rules: list,
        max_rules: int,
        scoring_mode: str
    ) -> list:
        """
        Perform greedy selection with HARD novelty cutoff by default.
        
        Each selected rule MUST cover new samples.
        
        Args:
            rules: List of Rule objects
            max_rules: Maximum number of rules to select
            scoring_mode: "precision_first" or "recall_first"
            
        Returns:
            List of selected Rule objects
        """
        print(f"\n  ⚠️  ILP failed - using greedy fallback with diversity + novelty")
        print(f"      Greedy min novelty: {self.min_novelty_greedy:.0%}")
        print(f"      Hard cutoff: {self.greedy_hard_novelty_cutoff}")
        print(f"      Min new samples: {self.min_absolute_new_samples}")
        print(f"      Novelty weight: {self.greedy_novelty_weight}")
        
        selected = []
        remaining = rules.copy()
        covered_samples = set()
        
        while len(selected) < max_rules and remaining:
            best_rule = None
            best_score = -np.inf
            best_novelty = 0
            best_new_samples = 0
            
            for rule in remaining:
                rule_covered = set(rule.covered_idx)
                
                # Compute novelty
                if not covered_samples:
                    new_samples = len(rule_covered)
                    novelty = 1.0
                else:
                    new_samples_set = rule_covered - covered_samples
                    new_samples = len(new_samples_set)
                    novelty = new_samples / len(rule_covered) if rule_covered else 0.0
                
                # HARD cutoff mode (default)
                if self.greedy_hard_novelty_cutoff and selected:
                    if novelty < self.min_novelty_greedy:
                        continue
                    if new_samples < self.min_absolute_new_samples:
                        continue
                
                # Compute quality score
                if scoring_mode == "precision_first":
                    quality = (rule.precision ** 3) * rule.coverage
                else:  # recall_first
                    quality = (rule.recall ** 3) * rule.precision * rule.coverage
                
                # Apply diversity penalty (feature-based)
                diversity = self.diversity_analyzer.compute_diversity_score(rule, selected)
                
                # Novelty bonus in scoring
                novelty_factor = 1.0 + self.greedy_novelty_weight * novelty
                
                # Combined score
                score = quality * (1 + self.diversity_weight * diversity) * novelty_factor
                
                if score > best_score:
                    best_score = score
                    best_rule = rule
                    best_novelty = novelty
                    best_new_samples = new_samples
            
            if best_rule is None:
                print(f"      No more rules meet novelty threshold - stopping at {len(selected)} rules")
                break
            
            # Log selection
            print(f"      Selected rule {best_rule.rule_id}: "
                  f"prec={best_rule.precision:.3f}, "
                  f"recall={best_rule.recall:.3f}, "
                  f"cov={best_rule.coverage:.3f}, "
                  f"novelty={best_novelty:.1%} (+{best_new_samples} new)")
            
            selected.append(best_rule)
            remaining.remove(best_rule)
            covered_samples |= set(best_rule.covered_idx)
        
        print(f"\n  ✅ Selected {len(selected)} rules via greedy")
        print(f"     Total unique samples covered: {len(covered_samples)}")
        
        return selected