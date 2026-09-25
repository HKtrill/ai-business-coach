# ============================================================
# GLASS ROUTER: FEATURE VALIDATOR MODULE
# ============================================================
# Validates feature composition and structural constraints
# Works with frozenset segments
# ============================================================

from typing import Tuple, Set

from glass_pipeline.glass_router.core.rule import SegmentType
from glass_pipeline.glass_router.rf.binning import BINNING_STRATEGY


def _build_bin_family_map() -> dict:
    """
    Exact bin -> family map from the locked BINNING_STRATEGY.

    Every bin of one source feature maps to the same family (e.g. nsd_cold,
    nsd_warm, nsd_elevated, nsd_hot -> "nsd"), so a rule can never combine two
    bins of the same feature. The passthrough cpi_cellular maps to "cpi".
    """
    fam = {}
    for group_name, group_def in BINNING_STRATEGY.items():
        family = group_name.lower()
        if "passthrough" in group_def:
            fam[group_def["passthrough"]] = family
        else:
            for bin_name, _lo, _hi in group_def["bins"]:
                fam[bin_name] = family
    return fam


_BIN_FAMILY = _build_bin_family_map()


class FeatureValidator:
    """Validate feature usage and structural constraints for rules."""

    def __init__(self, rule_prefixes: Tuple[str, ...]):
        """
        Initialize feature validator.

        Args:
            rule_prefixes: Tuple of prefixes identifying rule-eligible features
        """
        self.rule_prefixes = rule_prefixes

        # Define base prefixes for feature family mapping
        self.base_prefixes = [
            "previous_", "nr_employed_", "euribor_", "emp_var_",
            "cpi_", "cci_", "month_", "contact_", "age_",
            "campaign_", "job_", "marital_", "education_",
            "dow_", "default_", "housing_", "loan_",
            "econ_", "prospect_",
        ]

        # Define bin suffixes for reverse lookup
        self.bin_suffixes = [
            "_zero", "_low", "_mid", "_high",
            "_very_neg", "_neg", "_pos",
            "_sweet_spot",
            "_hot", "_warm", "_neutral", "_cold",
            "_cellular", "_telephone",
            "_young", "_prime", "_senior",
            "_first", "_moderate", "_heavy",
            "_high_lift", "_above_avg", "_low_lift",
            "_single", "_married", "_divorced", "_unknown",
            "_midweek", "_edges",
            "_no", "_yes",
            "_favorable", "_unfavorable",
        ]

    def extract_base_feature(self, feature_name: str) -> str:
        """
        Map discretized feature to its base variable family.

        Args:
            feature_name: Feature name, e.g. 'age_young', 'euribor_low'

        Returns:
            Base feature name, e.g. 'age', 'euribor'
        """
        # Current 29 bins: exact lookup (covers nsd_elevated, jed_transition,
        # behav_baseline, which the prefix/suffix lists below missed)
        if feature_name in _BIN_FAMILY:
            return _BIN_FAMILY[feature_name]

        # Legacy feature names (prefix / suffix heuristics)
        # Composite features
        if feature_name == "is_cold":
            return "is_cold"
        if feature_name.startswith("prospect_tier_"):
            return "prospect_tier"

        # Standard binned features
        for prefix in self.base_prefixes:
            if feature_name.startswith(prefix):
                return prefix.rstrip("_")

        # Bin suffixes for reverse lookup if needed
        for suffix in self.bin_suffixes:
            if feature_name.endswith(suffix):
                return feature_name[:-len(suffix)]

        # Binary features with no prefix/suffix
        return feature_name

    def has_duplicate_base_features(self, segment: SegmentType) -> bool:
        """
        Validate rule structure: no multiple bins from the same base feature.

        Args:
            segment: Set or frozenset of (feature, level) tuples

        Returns:
            True if segment contains duplicate base features
        """
        observed_bases: Set[str] = set()

        for feature, _ in segment:
            base = self.extract_base_feature(feature)

            if base in observed_bases:
                return True

            observed_bases.add(base)

        return False

    def has_rule_feature(self, segment: SegmentType) -> bool:
        """
        Check if segment contains at least one rule-eligible feature.

        Args:
            segment: Set or frozenset of (feature, level) tuples

        Returns:
            True if segment has at least one rule-eligible feature
        """
        features = {f for f, _ in segment}

        return any(
            any(f.startswith(prefix) for prefix in self.rule_prefixes)
            for f in features
        )

    def get_used_base_features(self, segment: SegmentType) -> Set[str]:
        """
        Get set of base features used in segment.

        Args:
            segment: Set or frozenset of (feature, level) tuples

        Returns:
            Set of base feature names
        """
        return {self.extract_base_feature(f) for f, _ in segment}