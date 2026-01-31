# ============================================================
# GLASS-BRW: BEAM SEARCH MODULE
# ============================================================
# Manages beam search expansion with depth-staged constraint checking
# Creates CandidateRule objects
# ============================================================

from typing import List, Dict, Tuple, Set
from collections import defaultdict

from glass_brw.core.rule import CandidateRule


class BeamSearch:
    """Manage beam search expansion for rule generation."""
    
    def __init__(
        self,
        beam_width: int = 100,
        min_complexity: int = 1,
        max_complexity: int = 3,
    ):
        """
        Initialize beam search.
        
        Args:
            beam_width: Number of top rules to keep per depth
            min_complexity: Minimum rule complexity to include in candidates
            max_complexity: Maximum rule depth to explore
        """
        self.beam_width = beam_width
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
    
    def deduplicate_rules(self, rules: List[CandidateRule]) -> List[CandidateRule]:
        """
        Remove duplicate rules via segment matching.
        
        Args:
            rules: List of CandidateRule objects
            
        Returns:
            Deduplicated list of rules
        """
        unique: Dict[frozenset, CandidateRule] = {}
        for rule in rules:
            seg_signature = rule.segment_frozen
            if seg_signature not in unique:
                unique[seg_signature] = rule
            else:
                # Keep rule with higher beam score
                if rule._beam_score > unique[seg_signature]._beam_score:
                    unique[seg_signature] = rule
        return list(unique.values())
    
    def prune_beam(
        self, 
        rules: List[CandidateRule], 
        scorer, 
        validator
    ) -> List[CandidateRule]:
        """
        Sort and prune beam to width limit.
        
        Args:
            rules: List of CandidateRule objects
            scorer: RuleScorer instance
            validator: FeatureValidator instance
            
        Returns:
            Pruned list of top rules
        """
        # Sort by beam score (highest first)
        rules = sorted(rules, key=lambda r: r._beam_score, reverse=True)
        
        # Keep top beam_width rules
        return rules[:self.beam_width]
    
    def expand_rule(
        self,
        parent_rule: CandidateRule,
        available_features: List[str],
        depth: int,
        predicted_class: int,
        validator,
        metrics_computer,
        constraints,
        scorer,
        logger,
        rule_id_counter: Dict[str, int],
    ) -> Tuple[List[CandidateRule], Dict[str, int]]:
        """
        Expand a single parent rule by adding one feature.
        
        Args:
            parent_rule: Parent CandidateRule to expand
            available_features: List of features to try adding
            depth: Current depth
            predicted_class: Target class (0 or 1)
            validator: FeatureValidator instance
            metrics_computer: RuleMetrics instance
            constraints: RuleConstraints instance
            scorer: RuleScorer instance
            logger: RuleLogger instance
            rule_id_counter: Dict to track rule IDs
            
        Returns:
            (accepted_rules, stats) tuple
        """
        accepted_rules: List[CandidateRule] = []
        stats: Dict[str, int] = defaultdict(int)
        
        # Track seen segments to avoid order-dependent duplicates
        seen_segments: Set[frozenset] = set()
        
        # Get features and base features already used in parent
        used_features = {f for f, _ in parent_rule.segment}
        used_base_features = validator.get_used_base_features(parent_rule.segment)
        
        # Try adding each available feature
        for feature in available_features:
            # Skip if feature already used
            if feature in used_features:
                continue
            
            # Skip if base feature already used
            base_feature = validator.extract_base_feature(feature)
            if base_feature in used_base_features:
                continue
            
            # Try both levels (1 and 0)
            for level in (1, 0):
                stats['total_considered'] += 1
                
                # Create candidate segment (frozenset for immutability)
                seg = frozenset(parent_rule.segment | {(feature, level)})
                
                # Check for duplicate segment (order-independent)
                if seg in seen_segments:
                    stats['rejected_duplicate_segment'] += 1
                    continue
                
                seen_segments.add(seg)
                
                # Compute metrics
                metrics = metrics_computer.compute_all_metrics(seg, predicted_class)
                
                # Skip if zero support
                if metrics['support'] == 0:
                    continue
                
                # Apply depth-appropriate constraints
                if depth == 2:
                    is_valid, rejection_reason = constraints.check_depth2_constraints(
                        seg, depth, predicted_class, metrics, validator
                    )
                else:  # depth >= 3
                    is_valid, rejection_reason = constraints.check_depth3_constraints(
                        seg, depth, predicted_class, metrics, validator
                    )
                
                if not is_valid:
                    stats[f'rejected_{rejection_reason}'] += 1
                    
                    # Log rejection if enabled
                    if logger.verbose_rejection_logging:
                        threshold_info = constraints.get_threshold_info(
                            rejection_reason, predicted_class
                        )
                        logger.log_rejected_rule(
                            seg=seg,
                            depth=depth,
                            rejection_reason=rejection_reason,
                            predicted_class=predicted_class,
                            precision=metrics['precision'],
                            recall=metrics['recall'],
                            coverage=metrics['coverage'],
                            support=metrics['support'],
                            parent_precision=parent_rule.precision,
                            parent_recall=parent_rule.recall,
                            parent_coverage=parent_rule.coverage,
                            parent_segment=parent_rule.segment,
                            leakage_rate=metrics['leakage_rate'],
                            subscribers_caught=metrics['subscribers_caught'],
                            threshold_info=threshold_info,
                        )
                    continue
                
                # All checks passed - create rule
                stats['accepted'] += 1
                
                # Log acceptance if enabled
                logger.log_accepted_rule(
                    seg=seg,
                    depth=depth,
                    predicted_class=predicted_class,
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    coverage=metrics['coverage'],
                    support=metrics['support'],
                    parent_precision=parent_rule.precision,
                    parent_recall=parent_rule.recall,
                    parent_coverage=parent_rule.coverage,
                    parent_segment=parent_rule.segment,
                    leakage_rate=metrics['leakage_rate'],
                    subscribers_caught=metrics['subscribers_caught'],
                )
                
                # Create CandidateRule
                rule = CandidateRule(
                    rule_id=rule_id_counter['current'],
                    segment=seg,
                    predicted_class=predicted_class,
                    complexity=depth,
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    coverage=metrics['coverage'],
                    support=metrics['support'],
                )
                
                # Store beam score for internal use
                rule._beam_score = scorer.score_rule(
                    metrics['precision'],
                    metrics['recall'],
                    metrics['coverage'],
                    predicted_class,
                    seg,
                    validator,
                )
                
                accepted_rules.append(rule)
                rule_id_counter['current'] += 1
        
        return accepted_rules, dict(stats)
    
    def expand_depth(
        self,
        current_rules: List[CandidateRule],
        available_features: List[str],
        depth: int,
        predicted_class: int,
        validator,
        metrics_computer,
        constraints,
        scorer,
        logger,
        rule_id_counter: Dict[str, int],
    ) -> Tuple[List[CandidateRule], Dict[str, int]]:
        """
        Expand all rules at current depth to next depth.
        
        Args:
            current_rules: List of rules at current depth
            available_features: List of features to try adding
            depth: Next depth to expand to
            predicted_class: Target class (0 or 1)
            validator: FeatureValidator instance
            metrics_computer: RuleMetrics instance
            constraints: RuleConstraints instance
            scorer: RuleScorer instance
            logger: RuleLogger instance
            rule_id_counter: Dict to track rule IDs
            
        Returns:
            (next_rules, combined_stats) tuple
        """
        next_rules: List[CandidateRule] = []
        combined_stats: Dict[str, int] = defaultdict(int)
        
        # Reset logger counters for this depth
        logger.reset_depth_counters()
        
        # Expand each parent rule
        for parent_rule in current_rules:
            accepted, stats = self.expand_rule(
                parent_rule=parent_rule,
                available_features=available_features,
                depth=depth,
                predicted_class=predicted_class,
                validator=validator,
                metrics_computer=metrics_computer,
                constraints=constraints,
                scorer=scorer,
                logger=logger,
                rule_id_counter=rule_id_counter,
            )
            
            next_rules.extend(accepted)
            
            # Aggregate stats
            for key, value in stats.items():
                combined_stats[key] += value
        
        return next_rules, dict(combined_stats)