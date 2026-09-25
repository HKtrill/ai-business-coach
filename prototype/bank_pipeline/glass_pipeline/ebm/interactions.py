"""
glass_pipeline.ebm.interactions
================================
EBM interaction term definitions for Stage 3.

Replaces the prototype macro_index / month_sin / log_campaign pairs with
validated interactions derived from the research feature importance analysis.

Interaction philosophy
----------------------
Many EBM_FEATURES are already compound interactions:
  - cpi_high_cellular          = cons.price.idx × contact (overlap.py)
  - overlap_default_clean      = low_stress_zone × default_clean
  - overlap_behavioral_score   = low_stress_zone × behavioral_favorability
  - prior_x_stress             = has_prior_contact × economic_stress_integral
  - decay_x_density            = joint_economic_decay × neighborhood_subscription_density

These are handled in feature engineering and do not need to appear here.

Explicit EBM interaction terms (pairwise shape functions) are reserved for
relationships that cross feature boundaries not captured by the compounds above.

Validated interactions
----------------------
TIER 1 (importance=0.205, research confirmed):
    campaign × emp_var_rate_sigmoid_slope
    — Campaign contact pressure modulated by economic momentum.
    — NOTE: if emp_var_rate_sigmoid_slope is dropped in the 13→12 pruning
      experiment this interaction collapses entirely. define_ebm_interactions()
      guards for feature presence — it will simply not be added.

TIER 2 (exploratory, consistent with research findings):
    euribor3m_sigmoid_slope × economic_curvature_intensity
    — Rate-environment slope vs. curvature inflection zone interaction.
    — Both are economic derivative features; their joint surface may capture
      non-linearities not visible in main effects alone.

    prior_x_stress × decay_x_density
    — Prior-contact value amplified under economic stress, crossed with
      neighbourhood density signal under decay conditions.

All interactions are conditional on feature presence so pruning experiments
(EBM_FEATURES_13, EBM_FEATURES_12) do not require a separate interactions file.
"""


def define_ebm_interactions(X) -> list[tuple[str, str]]:
    """
    Return the list of explicit pairwise interaction terms for the EBM.

    Interactions are added only when both members are present in X,
    making this safe to call after any pruning variant of select_ebm_features().

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix passed to EBM training (post-select_ebm_features).

    Returns
    -------
    list[tuple[str, str]]
        Pairs of column names for EBM to model as explicit 2D interactions.
    """
    cols = set(X.columns)
    interactions = []

    # ------------------------------------------------------------------
    # TIER 1 — Research-validated, importance=0.205
    # Campaign contact pressure × economic employment momentum slope.
    # Highest-ranked interaction in the research EBM round.
    # Will be absent if emp_var_rate_sigmoid_slope is dropped (cut 2).
    # ------------------------------------------------------------------
    if 'campaign' in cols and 'emp_var_rate_sigmoid_slope' in cols:
        interactions.append(('campaign', 'emp_var_rate_sigmoid_slope'))

    # ------------------------------------------------------------------
    # TIER 2 — Exploratory; consistent with research derivative analysis
    # Rate environment (slope) vs. curvature inflection crossing.
    # ------------------------------------------------------------------
    if 'euribor3m_sigmoid_slope' in cols and 'economic_curvature_intensity' in cols:
        interactions.append(('euribor3m_sigmoid_slope', 'economic_curvature_intensity'))

    # ------------------------------------------------------------------
    # TIER 2 — Prior contact quality under stress, crossed with
    # neighbourhood density decay.
    # ------------------------------------------------------------------
    if 'prior_x_stress' in cols and 'decay_x_density' in cols:
        interactions.append(('prior_x_stress', 'decay_x_density'))

    return interactions