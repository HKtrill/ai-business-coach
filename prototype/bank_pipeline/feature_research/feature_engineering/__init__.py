"""
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
pipeline    — FeaturePipeline orchestrator

Public API
----------
    FeaturePipeline().fit_transform(X_train, y_train) / .transform(X_test)

Fit-on-train engineers (used by FeaturePipeline):
    IntegralFeatureEngineer     neighborhood_subscription_density
    DerivativeFeatureEngineer   euribor3m_local_rate + curvature
    TemporalFeatureEngineer     dow_month_encoded

Deterministic adders: add_crisis_features, add_prior_features, add_overlap_features

Stage feature registries:
    LIVE_FEATURES  dict[str, list[str]]   keyed by 'lr', 'rf', 'ebm'
    LR_FEATURES, RF_FEATURES, EBM_FEATURES
"""

from feature_research.feature_engineering.crisis import add_crisis_features
from feature_research.feature_engineering.integrals import IntegralFeatureEngineer
from feature_research.feature_engineering.derivatives import DerivativeFeatureEngineer
from feature_research.feature_engineering.temporal import TemporalFeatureEngineer
from feature_research.feature_engineering.prior import add_prior_features
from feature_research.feature_engineering.overlap import add_overlap_features
from feature_research.feature_engineering.pipeline import (
    FeaturePipeline,
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
    'FeaturePipeline',
    'finalize_features',
    'get_stage_df',
    # deterministic adders
    'add_crisis_features',
    'add_prior_features',
    'add_overlap_features',
    # fit-on-train engineers
    'IntegralFeatureEngineer',
    'DerivativeFeatureEngineer',
    'TemporalFeatureEngineer',
    # registries
    'LR_FEATURES',
    'RF_FEATURES',
    'EBM_FEATURES',
    'LIVE_FEATURES',
]