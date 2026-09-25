""""
feature_research.feature_engineering
=====================================
Subpackage for Glass Cascade feature engineering.

Modules (in DAG order)
-----------------------
crisis      — Cell 10A: cellular_crisis + intermediates
integrals   — Cell 10B: economic_stress_integral, neighborhood_subscription_density
derivatives — Cell 10C: sigmoid slopes, local rate, curvature, decay composites
temporal    — Cell 10D: dow_month_encoded
prior       — Cell 10E: has_prior_contact, prior_x_stress
overlap     — Cell 10F: cpi_high_cellular, behavioral_favorability,
              overlap_default_clean, overlap_behavioral_score
pipeline    — build_features() orchestrator

Public API
----------
Primary entry point (research notebook):
    build_features(df, target_col, random_state, n_bins, smoothing_factor) -> df

Production engineers (glass_cascade pipeline — leakage-free):
    IntegralFeatureEngineer     fit/transform for neighborhood_subscription_density
    DerivativeFeatureEngineer   fit/transform for euribor3m_local_rate + curvature
    TemporalFeatureEngineer     fit/transform for dow_month_encoded

Stage feature registries:
    LIVE_FEATURES  dict[str, list[str]]   keyed by 'lr', 'rf', 'ebm'
    LR_FEATURES, RF_FEATURES, EBM_FEATURES
"""

from feature_research.feature_engineering.crisis import add_crisis_features
from feature_research.feature_engineering.integrals import (
    add_integral_features,
    IntegralFeatureEngineer,
)
from feature_research.feature_engineering.derivatives import (
    add_derivative_features,
    DerivativeFeatureEngineer,
)
from feature_research.feature_engineering.temporal import (
    add_temporal_features,
    TemporalFeatureEngineer,
)
from feature_research.feature_engineering.prior import add_prior_features
from feature_research.feature_engineering.overlap import add_overlap_features
from feature_research.feature_engineering.pipeline import (
    build_features,
    finalize_features,
    get_stage_df,
)

# ---------------------------------------------------------------------------
# Stage feature registries (from Cell 10G)
# ---------------------------------------------------------------------------

LR_FEATURES: list[str] = [
    'cellular_crisis',        # crisis.py
    'euribor3m_local_rate',   # derivatives.py
    'dow_month_encoded',      # temporal.py
]

RF_FEATURES: list[str] = [
    'campaign',                          # raw UCI
    'neighborhood_subscription_density', # integrals.py
    'cons.conf.idx',                     # raw UCI
    'economic_curvature_intensity',      # derivatives.py
    'joint_economic_decay',              # derivatives.py
    'dow_month_encoded',                 # temporal.py
    'behavioral_favorability',           # overlap.py
    'cpi_high_cellular',                 # overlap.py
]

EBM_FEATURES: list[str] = [
    'decay_x_density',              # derivatives.py
    'euribor3m_sigmoid_slope',      # derivatives.py
    'economic_curvature_intensity', # derivatives.py
    'cons.conf.idx',                # raw UCI
    'age',                          # raw UCI
    'dow_month_encoded',            # temporal.py
    'default',                      # raw UCI
    'prior_x_stress',               # prior.py
    'campaign',                     # raw UCI
    'overlap_default_clean',        # overlap.py
    'overlap_behavioral_score',     # overlap.py
    'cpi_high_cellular',            # overlap.py
    'behavioral_favorability',      # overlap.py
    'emp_var_rate_sigmoid_slope',   # derivatives.py
]

LIVE_FEATURES: dict[str, list[str]] = {
    'lr':  LR_FEATURES,
    'rf':  RF_FEATURES,
    'ebm': EBM_FEATURES,
}

__all__ = [
    # pipeline
    'build_features',
    'finalize_features',
    'get_stage_df',
    # individual adders (DAG order)
    'add_crisis_features',
    'add_integral_features',
    'add_derivative_features',
    'add_temporal_features',
    'add_prior_features',
    'add_overlap_features',
    # production engineers (fit/transform — leakage-free)
    'IntegralFeatureEngineer',
    'DerivativeFeatureEngineer',
    'TemporalFeatureEngineer',
    # registries
    'LR_FEATURES',
    'RF_FEATURES',
    'EBM_FEATURES',
    'LIVE_FEATURES',
]