# ============================================================
# GLASS-BRW: PERFORMANCE ANALYZER MODULE
# ============================================================
# Extract analysis and reporting logic from main pipeline
# ============================================================
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class PerformanceAnalyzer:
    """Analyze and report GLASS-BRW performance metrics."""
    
    def __init__(self, glass_model, subscribe_label=1, abstain_label=-1):
        """
        Initialize performance analyzer.
        
        Args:
            glass_model: Fitted GLASS_BRW instance
            subscribe_label: Label for SUBSCRIBE class (default: 1)
            abstain_label: Label for ABSTAIN (default: -1)
        """
        self.glass = glass_model
        self.SUBSCRIBE = subscribe_label
        self.ABSTAIN = abstain_label
    
    # ============================================================
    # RULE QUALITY ANALYSIS
    # ============================================================
    
    def print_rule_quality(self):
        """Print rule quality and diversity metrics."""
        print("\n" + "="*80)
        print("📊 RULE QUALITY & DIVERSITY ANALYSIS")
        print("="*80)
        
        self._print_pass1_quality()
        self._print_pass2_quality()
    
    def _print_pass1_quality(self):
        """Print Pass 1 (routing) rule quality."""
        print(f"\n🧭 Pass 1 Routing Rules:")
        print(f"   Count: {len(self.glass.pass1_rules)}")
        
        if not self.glass.pass1_rules:
            return
        
        pass1_precs = [r.precision for r in self.glass.pass1_rules]
        pass1_covs = [r.coverage for r in self.glass.pass1_rules]
        
        print(f"   Precision: {np.mean(pass1_precs):.3f} ± {np.std(pass1_precs):.3f}")
        print(f"   Coverage:  {np.mean(pass1_covs):.3f} ± {np.std(pass1_covs):.3f}")
        
        print(f"\n   Top 3 by precision:")
        for i, rule in enumerate(
            sorted(self.glass.pass1_rules, key=lambda r: r.precision, reverse=True)[:3], 1
        ):
            seg_str = self._format_segment(rule.segment, max_features=2)
            print(f"   {i}. {seg_str}")
            print(f"      Prec: {rule.precision:.3f}, Cov: {rule.coverage:.3f}")
    
    def _print_pass2_quality(self):
        """Print Pass 2 (detection) rule quality."""
        print(f"\n🎯 Pass 2 Detection Rules:")
        print(f"   Count: {len(self.glass.pass2_rules)}")
        
        if not self.glass.pass2_rules:
            return
        
        pass2_recalls = [r.recall for r in self.glass.pass2_rules]
        pass2_precs = [r.precision for r in self.glass.pass2_rules]
        pass2_covs = [r.coverage for r in self.glass.pass2_rules]
        
        print(f"   Recall:    {np.mean(pass2_recalls):.3f} ± {np.std(pass2_recalls):.3f}")
        print(f"   Precision: {np.mean(pass2_precs):.3f} ± {np.std(pass2_precs):.3f}")
        print(f"   Coverage:  {np.mean(pass2_covs):.3f} ± {np.std(pass2_covs):.3f}")
        
        print(f"\n   Top 3 by recall:")
        for i, rule in enumerate(
            sorted(self.glass.pass2_rules, key=lambda r: r.recall, reverse=True)[:3], 1
        ):
            seg_str = self._format_segment(rule.segment, max_features=2)
            print(f"   {i}. {seg_str}")
            print(f"      Recall: {rule.recall:.3f}, Prec: {rule.precision:.3f}, Cov: {rule.coverage:.3f}")
        
        # Diversity analysis
        print(f"\n   Diversity Analysis:")
        base_features_used = set()
        for rule in self.glass.pass2_rules:
            for feature, _ in rule.segment:
                base = feature.split('_')[0] if '_' in feature else feature
                base_features_used.add(base)
        print(f"   Unique base features: {len(base_features_used)}")
        print(f"   Features: {', '.join(sorted(base_features_used))}")
    
    def _format_segment(self, segment, max_features=2):
        """Format segment for display."""
        seg_list = list(segment)
        seg_str = ' AND '.join([f"{f}={v}" for f, v in seg_list[:max_features]])
        if len(segment) > max_features:
            seg_str += f" ... (+{len(segment)-max_features})"
        return seg_str
    
    # ============================================================
    # PREDICTION ANALYSIS
    # ============================================================
    
    def analyze_predictions(self, X_train, y_train, X_test, y_test):
        """
        Analyze predictions and print comprehensive metrics.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict with train and test outputs
        """
        print("\n" + "="*80)
        print("📊 GENERATING PREDICTIONS")
        print("="*80)
        
        # Get predictions
        train_pred, train_conf, train_decisions = self.glass.predict(X_train)
        test_pred, test_conf, test_decisions = self.glass.predict(X_test)
        
        train_proba = self.glass.predict_proba(X_train)
        test_proba = self.glass.predict_proba(X_test)
        
        # Package outputs
        train_out = {
            "pred": train_pred,
            "confidence": train_conf,
            "covered": (train_pred != self.ABSTAIN),
            "abstained": (train_pred == self.ABSTAIN),
            "decisions": train_decisions,
            "proba": train_proba,
        }
        
        test_out = {
            "pred": test_pred,
            "confidence": test_conf,
            "covered": (test_pred != self.ABSTAIN),
            "abstained": (test_pred == self.ABSTAIN),
            "decisions": test_decisions,
            "proba": test_proba,
        }
        
        # Print metrics
        self._print_coverage_stats(train_out, test_out, len(train_pred), len(test_pred))
        self._print_decision_flow(train_out, test_out, len(train_pred), len(test_pred))
        self._print_survivor_diagnostics(test_out, y_test)
        self._print_covered_performance(test_out, test_pred, test_conf, y_test)
        
        return train_out, test_out
    
    def _print_coverage_stats(self, train_out, test_out, n_train, n_test):
        """Print coverage statistics."""
        print(f"\n📊 Coverage Statistics:")
        print(f"   Train: {train_out['covered'].sum():5,}/{n_train} ({train_out['covered'].mean():.1%})")
        print(f"   Test:  {test_out['covered'].sum():5,}/{n_test} ({test_out['covered'].mean():.1%})")
        
        train_pass1 = sum(1 for d in train_out['decisions'] if d == 'pass1')
        test_pass1 = sum(1 for d in test_out['decisions'] if d == 'pass1')
        print(f"   Train - Pass 1 blocked: {train_pass1:5,} ({train_pass1/n_train:.1%})")
        print(f"   Test  - Pass 1 blocked: {test_pass1:5,} ({test_pass1/n_test:.1%})")
    
    def _print_decision_flow(self, train_out, test_out, n_train, n_test):
        """Print sequential decision flow."""
        train_pass2 = sum(1 for d in train_out['decisions'] if d == 'pass2')
        test_pass2 = sum(1 for d in test_out['decisions'] if d == 'pass2')
        
        print(f"\n📊 Sequential Decision Flow:")
        print(f"   Train - Pass 2 detections: {train_pass2:5,} ({train_pass2/n_train:.1%})")
        print(f"   Train - Abstain:           {train_out['abstained'].sum():5,} ({train_out['abstained'].mean():.1%})")
        print(f"\n   Test  - Pass 2 detections: {test_pass2:5,} ({test_pass2/n_test:.1%})")
        print(f"   Test  - Abstain:           {test_out['abstained'].sum():5,} ({test_out['abstained'].mean():.1%})")
    
    def _print_survivor_diagnostics(self, test_out, y_test):
        """Print Pass survivor diagnostics."""
        # Boolean masks
        test_pass1_mask = np.array([d == "pass1" for d in test_out["decisions"]])
        test_pass2_mask = np.array([d == "pass2" for d in test_out["decisions"]])
        
        # Subscriber masks
        is_subscriber = (y_test == self.SUBSCRIBE).values
        
        # Counts
        total_subscribers = is_subscriber.sum()
        pass1_blocked_subscribers = np.sum(test_pass1_mask & is_subscriber)
        pass2_eligible_subscribers = np.sum((~test_pass1_mask) & is_subscriber)
        pass2_detected_subscribers = np.sum(test_pass2_mask & is_subscriber)
        
        # Rates
        pass1_leakage_rate = (
            pass1_blocked_subscribers / total_subscribers
            if total_subscribers > 0 else 0.0
        )
        eligible_recall = (
            pass2_detected_subscribers / pass2_eligible_subscribers
            if pass2_eligible_subscribers > 0 else 0.0
        )
        overall_recall = (
            pass2_detected_subscribers / total_subscribers
            if total_subscribers > 0 else 0.0
        )
        
        print("\n" + "="*80)
        print("🧬 PASS SURVIVOR DIAGNOSTICS")
        print("="*80)
        print(f"Total subscribers:                     {total_subscribers:,}")
        print(f"Blocked by Pass 1 (FN leakage):        {pass1_blocked_subscribers:,} "
              f"({pass1_leakage_rate:.1%})")
        print(f"Eligible for Pass 2 (survivors):       {pass2_eligible_subscribers:,} "
              f"({pass2_eligible_subscribers/total_subscribers:.1%})")
        print(f"Detected by Pass 2:                   {pass2_detected_subscribers:,}")
        print(f"\n🎯 Eligible Recall (Pass 2 only):      {eligible_recall:.1%}")
        print(f"🎯 Overall Subscriber Recall:          {overall_recall:.1%}")
    
    def _print_covered_performance(self, test_out, test_pred, test_conf, y_test):
        """Print performance on covered cases."""
        covered = test_out['covered']
        test_conf = np.asarray(test_conf, dtype=float)
        
        print(f"\n📊 Performance on Covered Cases (EXCLUDES ABSTENTIONS):")
        print(f"   Samples: {covered.sum():,}")
        
        if not covered.any():
            print("   No covered samples.")
            return
        
        covered_precision = precision_score(
            y_test[covered],
            test_pred[covered],
            pos_label=self.SUBSCRIBE,
            zero_division=0
        )
        covered_recall = recall_score(
            y_test[covered],
            test_pred[covered],
            pos_label=self.SUBSCRIBE,
            zero_division=0
        )
        covered_f1 = f1_score(
            y_test[covered],
            test_pred[covered],
            pos_label=self.SUBSCRIBE,
            zero_division=0
        )
        
        print(f"   Precision: {covered_precision:.3f}")
        print(f"   Recall:    {covered_recall:.3f}")
        print(f"   F1-Score:  {covered_f1:.3f}")
        print(f"   Avg Confidence (Pass 2 only): {test_conf[covered].mean():.3f}")