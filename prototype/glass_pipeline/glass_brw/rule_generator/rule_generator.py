# ============================================================
# GLASS-BRW: RULE GENERATOR MODULE (REFACTORED)
# ============================================================
# Main orchestrator for rule generation using modular components
# ============================================================
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

from .rule_metrics import RuleMetrics
from .feature_validator import FeatureValidator
from .rule_constraints import RuleConstraints
from .rule_scorer import RuleScorer
from .rule_logger import RuleLogger
from .beam_search import BeamSearch
from glass_brw.core.rule import Rule


class RuleGenerator:
    """
    Generate candidate rules with DEPTH-STAGED CONSTRAINTS.
    
    Depth Strategy:
        Depth 1: Structural validity ONLY (no support/precision/recall)
        Depth 2: Light pruning (extreme leakage guardrail, precision floor, coverage cap)
        Depth 3: Quality checks (support, precision, recall, coverage)
                 NO OVERLAP CHECK - ILP handles via novelty constraints
                 NO NOVEL COVERAGE CHECK - ILP handles diversity
    """
    
    def __init__(
        self,
        # ============================================================
        # SUPPORT THRESHOLDS - DEPTH 3 ONLY
        # ============================================================
        min_support_pass1=270,
        min_support_pass2=100,
        # ============================================================
        # COMPLEXITY BOUNDS
        # ============================================================
        max_complexity=3,
        min_complexity=1,
        # ============================================================
        # PASS 1: NOT_SUBSCRIBE (Precision-Focused Filters)
        # ============================================================
        min_complexity_not_subscribe=1,
        max_coverage_not_subscribe=1.00,
        min_precision_not_subscribe=0.25,
        max_precision_not_subscribe=1.00,
        max_subscriber_leakage_rate=0.99,
        max_subscriber_leakage_absolute=999,
        min_seed_precision_not_subscribe=0.05,
        # ============================================================
        # PASS 2: SUBSCRIBE (Recall-Focused Detectors)
        # ============================================================
        min_complexity_subscribe=1,
        max_coverage_subscribe=0.99,
        min_precision_subscribe=0.25,
        max_precision_subscribe=1.00,
        min_recall_subscribe=0.15,
        max_recall_subscribe=0.99,
        min_seed_precision_subscribe=0.05,
        # ============================================================
        # DEPTH 2: LIGHT STRUCTURAL PRUNING
        # ============================================================
        max_leakage_rate_depth2=0.75,
        max_leakage_fraction_depth2=0.65,
        # ============================================================
        # BEAM SEARCH PARAMETERS
        # ============================================================
        mode="strict",
        beam_width=100,
        # ============================================================
        # DIVERSITY ENFORCEMENT
        # ============================================================
        diversity_penalty=0.3,
        max_feature_reuse_pass1=3,
        max_feature_reuse_pass2=3,
        max_jaccard_overlap=0.35,
        max_high_overlap_rules=None,
        # ============================================================
        # SEGMENT ONTOLOGY
        # ============================================================
        segment_builder=None,
        # ============================================================
        # FEATURE SEMANTICS
        # ============================================================
        tier1_prefixes=(
            'previous', 'nr_employed', 'euribor', 'emp_var', 'cpi', 'cci',
            'month', 'contact', 'age', 'campaign', 'job', 'marital',
            'education', 'dow', 'default', 'housing', 'loan',
            'econ', 'prospect',
        ),
        # ============================================================
        # RANDOM FOREST INTEGRATION
        # ============================================================
        rf_model=None,
        feature_importance_threshold=0.00,
        # ============================================================
        # DIAGNOSTIC LOGGING
        # ============================================================
        verbose_rejection_logging=True,
        max_rejection_logs_per_constraint=50,
        verbose_acceptance_logging=True,
        max_acceptance_logs_per_depth=25,
    ):
        """Initialize rule generator with modular components."""
        
        # Store basic parameters
        self.max_complexity = max_complexity
        self.min_complexity = min_complexity
        self.mode = mode
        self.beam_width = beam_width
        
        # Segment builder
        if segment_builder is None:
            raise ValueError("RuleGenerator requires a SegmentBuilder instance")
        self.segment_builder = segment_builder
        self.segment_features = list(self.segment_builder.SEGMENT_FEATURES)
        
        # Random Forest
        self.rf_model = rf_model
        self.feature_importance_threshold = feature_importance_threshold
        self.feature_importances_ = None
        
        # ============================================================
        # INITIALIZE MODULAR COMPONENTS
        # ============================================================
        
        # Feature validation
        self.validator = FeatureValidator(tier1_prefixes=tier1_prefixes)
        
        # Constraint checking
        self.constraints = RuleConstraints(
            min_support_pass1=min_support_pass1,
            min_support_pass2=min_support_pass2,
            max_coverage_not_subscribe=max_coverage_not_subscribe,
            min_precision_not_subscribe=min_precision_not_subscribe,
            max_precision_not_subscribe=max_precision_not_subscribe,
            max_subscriber_leakage_rate=max_subscriber_leakage_rate,
            max_subscriber_leakage_absolute=max_subscriber_leakage_absolute,
            max_coverage_subscribe=max_coverage_subscribe,
            min_precision_subscribe=min_precision_subscribe,
            max_precision_subscribe=max_precision_subscribe,
            min_recall_subscribe=min_recall_subscribe,
            max_recall_subscribe=max_recall_subscribe,
            max_leakage_rate_depth2=max_leakage_rate_depth2,
            max_leakage_fraction_depth2=max_leakage_fraction_depth2,
        )
        
        # Rule scoring and diversity
        self.scorer = RuleScorer(
            diversity_penalty=diversity_penalty,
            max_feature_reuse_pass1=max_feature_reuse_pass1,
            max_feature_reuse_pass2=max_feature_reuse_pass2,
        )
        
        # Diagnostic logging
        self.logger = RuleLogger(
            verbose_rejection_logging=verbose_rejection_logging,
            max_rejection_logs_per_constraint=max_rejection_logs_per_constraint,
            verbose_acceptance_logging=verbose_acceptance_logging,
            max_acceptance_logs_per_depth=max_acceptance_logs_per_depth,
        )
        
        # Beam search
        self.beam_search = BeamSearch(
            beam_width=beam_width,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
        )
        
        print(f"🧩 RuleGenerator initialized with {len(self.segment_features)} segment features")
    
    # ============================================================
    # RANDOM FOREST INTEGRATION
    # ============================================================
    
    def _compute_rf_feature_importance(self, segments_df, y):
        """Compute or retrieve RF feature importances."""
        if self.rf_model is None:
            print("Training RF for feature importance...")
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=25,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(segments_df, y)
        
        importances = dict(zip(
            segments_df.columns,
            self.rf_model.feature_importances_
        ))
        self.feature_importances_ = importances
        
        top_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print("\nTop 10 RF features by importance:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.4f}")
        
        return importances
    
    def _get_feature_order(self, segments_df, y):
        """Get features ordered by RF importance."""
        importances = self._compute_rf_feature_importance(segments_df, y)
        
        important_features = [
            f for f, imp in importances.items()
            if imp >= self.feature_importance_threshold
        ]
        
        important_features.sort(
            key=lambda f: importances[f],
            reverse=True
        )
        
        print(f"\nUsing {len(important_features)}/{len(segments_df.columns)} "
              f"features above threshold")
        
        return important_features
    
    # ============================================================
    # SEED GENERATION (DEPTH 1)
    # ============================================================
    
    def _generate_seeds(self, metrics_computer, rule_id_counter):
        """Generate depth-1 seeds with structural validity only."""
        print(f"\n{'='*60}")
        print(f"DEPTH 1: SEED GENERATION")
        print(f"{'='*60}")
        print(f"  🚫 NO support constraints at this depth")
        print(f"  🚫 NO precision/recall constraints at this depth")
        
        seed_features = list(self.segment_features)
        current = {0: [], 1: []}
        
        debug_stats = {
            'total_considered': 0,
            'rejected_duplicate_base': 0,
            'accepted': {0: 0, 1: 0}
        }
        
        for feature in seed_features:
            for level in (1, 0):
                seg = {(feature, level)}
                debug_stats['total_considered'] += 1
                
                # Depth 1 constraint: Structural validity only
                is_valid, rejection_reason = self.constraints.check_depth1_constraints(
                    seg, self.validator
                )
                
                if not is_valid:
                    debug_stats['rejected_duplicate_base'] += 1
                    continue
                
                # Compute metrics
                metrics = metrics_computer.compute_all_metrics(seg, 0)  # Use class 0 for mask
                
                # Skip if zero support
                if metrics['support'] == 0:
                    continue
                
                # Generate rule for both classes
                for predicted_class in (0, 1):
                    metrics_cls = metrics_computer.compute_all_metrics(seg, predicted_class)
                    
                    debug_stats['accepted'][predicted_class] += 1
                    
                    # Create rule
                    rule = Rule(
                        rule_id=rule_id_counter['current'],
                        segment=seg,
                        predicted_class=predicted_class,
                        complexity=1,
                        pass_assignment="pass1" if predicted_class == 0 else "pass2"
                    )
                    
                    rule._p = metrics_cls['precision']
                    rule._r = metrics_cls['recall']
                    rule._c = metrics_cls['coverage']
                    rule._s = self.scorer.score_rule(
                        metrics_cls['precision'],
                        metrics_cls['recall'],
                        metrics_cls['coverage'],
                        predicted_class,
                        seg,
                        self.validator,
                    )
                    rule._cls = predicted_class
                    
                    current[predicted_class].append(rule)
                    rule_id_counter['current'] += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print("DEPTH 1 SUMMARY:")
        print(f"{'='*60}")
        print(f"Total feature-label pairs considered: {debug_stats['total_considered']}")
        print(f"Rejected - duplicate base features: {debug_stats['rejected_duplicate_base']}")
        for cls in (0, 1):
            pass_name = "Pass 1 (NOT_SUBSCRIBE)" if cls == 0 else "Pass 2 (SUBSCRIBE)"
            print(f"\n{pass_name}:")
            print(f"  Accepted: {debug_stats['accepted'][cls]}")
        
        # Deduplicate
        for cls in (0, 1):
            before = len(current[cls])
            current[cls] = self.beam_search.deduplicate_rules(current[cls])
            after = len(current[cls])
            if after < before:
                print(f"  Pass {2 if cls == 1 else 1}: Deduplicated seeds {before} → {after}")
        
        print(f"\nDepth 1 seeds (structural validity only):")
        print(f"  Pass 1 (NOT_SUBSCRIBE): {len(current[0])} rules")
        print(f"  Pass 2 (SUBSCRIBE): {len(current[1])} rules")
        
        return current
    
    # ============================================================
    # MAIN GENERATION PIPELINE
    # ============================================================
    
    def generate_candidates(self, segments_df, y):
        """
        Generate candidate rules with DEPTH-STAGED CONSTRAINTS.
        """
        # ============================================================
        # INITIALIZATION
        # ============================================================
        features = self._get_feature_order(segments_df, y)
        N = len(segments_df)
        
        # Initialize metrics computer
        metrics_computer = RuleMetrics(segments_df, y)
        
        # Set total subscribers for constraints
        total_subscribers = (y == 1).sum()
        self.constraints.set_total_subscribers(total_subscribers)
        
        # Reset tracking
        self.scorer.reset_tracking()
        rule_id_counter = {'current': 0}
        
        # Print generation summary
        print(f"\n{'='*60}")
        print("SEQUENTIAL RULE GENERATION WITH DEPTH-STAGED CONSTRAINTS")
        print(f"{'='*60}")
        print(f"  Beam width: {self.beam_width}")
        print(f"  Complexity: {self.min_complexity}–{self.max_complexity}")
        print(f"  Total samples: {N}")
        print(f"  Total subscribers: {total_subscribers}")
        print(f"\n  ✅ DEPTH-STAGED CONSTRAINT STRATEGY:")
        print(f"    • Depth 1: ONLY structural validity")
        print(f"    • Depth 2: Light pruning ONLY")
        print(f"    • Depth 3: Quality constraints (NO OVERLAP - ILP handles)")
        
        all_candidates = []
        
        # ============================================================
        # DEPTH 1: SEED GENERATION
        # ============================================================
        current = self._generate_seeds(metrics_computer, rule_id_counter)
        
        # ============================================================
        # DEPTH 2+: BEAM SEARCH EXPANSION
        # ============================================================
        for depth in range(2, self.max_complexity + 1):
            print(f"\n{'='*60}")
            if depth == 2:
                print(f"DEPTH {depth}: Light Structural Pruning")
            else:
                print(f"DEPTH {depth}: Quality Constraints")
            print(f"{'='*60}")
            
            next_rules = {0: [], 1: []}
            combined_stats = {0: defaultdict(int), 1: defaultdict(int)}
            
            # Expand each class separately
            for predicted_class in (0, 1):
                print(f"\n  Expanding {len(current[predicted_class])} "
                      f"Pass {2 if predicted_class == 1 else 1} rules...")
                
                expanded, stats = self.beam_search.expand_depth(
                    current_rules=current[predicted_class],
                    available_features=features,
                    depth=depth,
                    predicted_class=predicted_class,
                    validator=self.validator,
                    metrics_computer=metrics_computer,
                    constraints=self.constraints,
                    scorer=self.scorer,
                    logger=self.logger,
                    rule_id_counter=rule_id_counter,
                )
                
                next_rules[predicted_class] = expanded
                combined_stats[predicted_class] = stats
            
            # Print depth summary
            self._print_depth_summary(depth, combined_stats)
            
            # Check if expansion produced any rules
            total_expanded = len(next_rules[0]) + len(next_rules[1])
            if total_expanded == 0:
                print(f"\nNo valid expansions — stopping early at depth {depth-1}")
                break
            
            # ============================================================
            # DEDUPLICATION & BEAM PRUNING
            # ============================================================
            for cls in (0, 1):
                before = len(next_rules[cls])
                next_rules[cls] = self.beam_search.deduplicate_rules(next_rules[cls])
                after_dedup = len(next_rules[cls])
                
                if after_dedup < before:
                    print(f"  Pass {2 if cls == 1 else 1}: Deduplicated {before} → {after_dedup} rules")
                
                next_rules[cls] = self.beam_search.prune_beam(
                    next_rules[cls], self.scorer, self.validator
                )
                
                # Update feature usage
                for rule in next_rules[cls]:
                    self.scorer.update_feature_usage(rule.segment, cls, self.validator)
                
                # Add to candidates if depth >= min_complexity
                if depth >= self.min_complexity:
                    all_candidates.extend(next_rules[cls])
            
            # Print diversity summary
            usage_summary = self.scorer.get_usage_summary()
            print(f"\nDiversity tracking after depth {depth}:")
            print(f"  Pass 1 base features used: {usage_summary['pass1_features_used']}")
            if usage_summary['pass1_top_features']:
                top_str = ', '.join([f'{k}({v})' for k, v in usage_summary['pass1_top_features']])
                print(f"    Top: {top_str}")
            print(f"  Pass 2 base features used: {usage_summary['pass2_features_used']}")
            if usage_summary['pass2_top_features']:
                top_str = ', '.join([f'{k}({v})' for k, v in usage_summary['pass2_top_features']])
                print(f"    Top: {top_str}")
            
            print(f"\nAfter deduplication + beam prune:")
            print(f"  Pass 1 (NOT_SUBSCRIBE): {len(next_rules[0])} rules")
            print(f"  Pass 2 (SUBSCRIBE): {len(next_rules[1])} rules")
            
            current = next_rules
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        pass1_count = sum(1 for r in all_candidates if r.predicted_class == 0)
        pass2_count = sum(1 for r in all_candidates if r.predicted_class == 1)
        
        print(f"\n{'='*60}")
        print(f"✅ Generated {len(all_candidates)} candidate rules (k ≥ {self.min_complexity})")
        print(f"   Pass 1 (NOT_SUBSCRIBE): {pass1_count} rules")
        print(f"   Pass 2 (SUBSCRIBE): {pass2_count} rules")
        print(f"   ⚠️  Diversity handling delegated to ILP")
        
        return all_candidates
    
    def _print_depth_summary(self, depth, stats):
        """Print summary statistics for a depth."""
        print(f"\n{'='*60}")
        print(f"DEPTH {depth} SUMMARY:")
        print(f"{'='*60}")
        
        for cls in (0, 1):
            pass_name = "Pass 1 (NOT_SUBSCRIBE)" if cls == 0 else "Pass 2 (SUBSCRIBE)"
            cls_stats = stats[cls]
            
            print(f"\n{pass_name}:")
            print(f"  Total considered: {cls_stats['total_considered']}")
            print(f"  Accepted: {cls_stats['accepted']}")
            
            # Print rejection reasons
            rejection_keys = [k for k in cls_stats.keys() if k.startswith('rejected_')]
            for key in sorted(rejection_keys):
                reason = key.replace('rejected_', '')
                print(f"  Rejected - {reason}: {cls_stats[key]}")