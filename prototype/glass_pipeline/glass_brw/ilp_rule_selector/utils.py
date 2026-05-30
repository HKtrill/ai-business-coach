from glass_brw.core.rule import EvaluatedRule


def get_covered_indices(rule):
    idx = getattr(rule, '_cached_covered_idx', None)
    if idx is None:
        raise ValueError(f"covered indices not precomputed for rule {rule.rule_id}")
    return idx


def compute_rule_quality(rule: EvaluatedRule, scoring_mode: str) -> float:
    """
    Compute base quality score for a rule.

    precision_first: (precision^3) * coverage
    recall_first:    (recall^3) * precision * coverage
    """
    if scoring_mode == "precision_first":
        return (rule.precision ** 3) * rule.coverage
    return (rule.recall ** 3) * rule.precision * rule.coverage


def get_base_features(rule: EvaluatedRule, validator) -> set:
    """Set of base features used by a rule (deduplicated)."""
    return {validator.extract_base_feature(f) for f, _ in rule.segment}