"""
glass_pipeline.ebm.feature_engineering
=======================================
EBM Stage 3 feature engineering — production (leakage-free).

Replaces the prototype cyclic/log/macro_index approach with the validated
research DAG from feature_research.feature_engineering.

Full DAG (dependency order):
    crisis      → cellular_crisis, high_conversion_month,
                  cellular_contact, default_clean              [leakage-free]
    integrals   → economic_stress_integral,
                  neighborhood_subscription_density            [fit/transform]
    derivatives → sigmoid slopes, local_rate, curvature,
                  decay composites, decay_x_density            [fit/transform]
    temporal    → dow_month_encoded                            [fit/transform]
    prior       → has_prior_contact, prior_x_stress            [leakage-free]
    overlap     → cpi_high_cellular, behavioral_favorability,
                  overlap_default_clean, overlap_behavioral_score [leakage-free]

Leakage contract:
    IntegralFeatureEngineer, DerivativeFeatureEngineer, and
    TemporalFeatureEngineer all fit exclusively on X_train + y_train.
    engineer_ebm_features() requires y_train for this reason.

Pruning experiment (correlation / interpretability pass):
    overlap_behavioral_score  — drop candidate (|r|=0.716 with
                                economic_curvature_intensity; safer cut)
    emp_var_rate_sigmoid_slope — drop candidate (|r|=0.724 with
                                euribor3m_sigmoid_slope; riskier — anchors
                                TIER 1 interaction campaign×emp_var_rate_sigmoid_slope,
                                importance=0.205)
    See EBM_FEATURES_13 and EBM_FEATURES_12 registries below.
"""

import pandas as pd

from feature_research.feature_engineering.crisis import add_crisis_features
from feature_research.feature_engineering.integrals import IntegralFeatureEngineer
from feature_research.feature_engineering.derivatives import DerivativeFeatureEngineer
from feature_research.feature_engineering.temporal import TemporalFeatureEngineer
from feature_research.feature_engineering.prior import add_prior_features
from feature_research.feature_engineering.overlap import add_overlap_features

# ---------------------------------------------------------------------------
# Leaky raw features — removed before any engineering
# ---------------------------------------------------------------------------
LEAKY_FEATURES = ['poutcome', 'pdays', 'duration']

# ---------------------------------------------------------------------------
# EBM feature registries
# ---------------------------------------------------------------------------

# Baseline 14-feature set (validated in research)
EBM_FEATURES: list[str] = [
    'decay_x_density',              # derivatives — joint decay × neighborhood density
    'euribor3m_sigmoid_slope',      # derivatives — normalised sigmoid slope
    'economic_curvature_intensity', # derivatives — mean |d²P/dx²| across economic features
    'cons.conf.idx',                # raw UCI
    'age',                          # raw UCI
    'dow_month_encoded',            # temporal — smoothed P(subscribe | dow × month)
    'default',                      # raw UCI
    'prior_x_stress',               # prior — has_prior_contact × economic_stress_integral
    'campaign',                     # raw UCI
    'overlap_default_clean',        # overlap — low_stress_zone × default_clean
 #   'overlap_behavioral_score',     # overlap — low_stress_zone × behavioral_favorability
                                    #           DROP CANDIDATE (cut 1 — safer)
    'cpi_high_cellular',            # overlap — cons.price.idx ≥ 93.5 AND contact==0
    'behavioral_favorability',      # overlap — weighted behavioral composite
    'emp_var_rate_sigmoid_slope',   # derivatives — normalised sigmoid slope
                                    #           DROP CANDIDATE (cut 2 — riskier;
                                    #           anchors campaign×emp_var_rate_sigmoid_slope)
]

# Pruning experiment — cut 1: drop overlap_behavioral_score
# Retrain, compare AUC / Recall / F2 against 14-feature baseline.
EBM_FEATURES_13: list[str] = [
    f for f in EBM_FEATURES if f != 'overlap_behavioral_score'
]

# Pruning experiment — cut 2: also drop emp_var_rate_sigmoid_slope
# Only proceed if F2 holds after cut 1. Collapses campaign×emp_var_rate_sigmoid_slope.
EBM_FEATURES_12: list[str] = [
    f for f in EBM_FEATURES_13 if f != 'emp_var_rate_sigmoid_slope'
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def drop_leaky_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop known leaky features before any engineering.

    Leaky: poutcome, pdays, duration.
    Safe to call even if columns are already absent.
    """
    present = [f for f in LEAKY_FEATURES if f in X_train.columns]
    if present:
        X_train = X_train.drop(columns=present)
        X_test  = X_test.drop(columns=present)
    return X_train, X_test


def engineer_ebm_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Apply the full validated research DAG in fit-on-train / transform order.

    DAG: crisis → integrals → derivatives → temporal → prior → overlap

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (leaky columns already removed).
    X_test : pd.DataFrame
        Test features (leaky columns already removed).
    y_train : pd.Series
        Training labels. Used only to fit IntegralFeatureEngineer,
        DerivativeFeatureEngineer, and TemporalFeatureEngineer.
        Never applied to X_test.

    Returns
    -------
    X_train : pd.DataFrame
        Training split with all engineered features appended.
    X_test : pd.DataFrame
        Test split with all engineered features appended.
    features_added : list[str]
        Names of the EBM feature columns produced (EBM_FEATURES).

    Notes
    -----
    Call select_ebm_features() after this to positive-select down to
    EBM_FEATURES (or a pruning variant) before model training.
    """
    # ------------------------------------------------------------------
    # Step 1 — Crisis features (leakage-free; no target dependency)
    # ------------------------------------------------------------------
    X_train = add_crisis_features(X_train)
    X_test  = add_crisis_features(X_test)

    # ------------------------------------------------------------------
    # Step 2 — Integral features (fit on train only)
    # Produces: economic_stress_integral, neighborhood_subscription_density
    # ------------------------------------------------------------------
    integral_eng = IntegralFeatureEngineer()
    X_train = integral_eng.fit_transform(X_train, y_train)
    X_test  = integral_eng.transform(X_test)

    # ------------------------------------------------------------------
    # Step 3 — Derivative features (fit on train only)
    # Requires neighborhood_subscription_density from Step 2.
    # Produces: euribor3m_sigmoid_slope, emp_var_rate_sigmoid_slope,
    #           euribor3m_local_rate, economic_curvature_intensity,
    #           joint_economic_decay, decay_x_density
    # ------------------------------------------------------------------
    derivative_eng = DerivativeFeatureEngineer()
    X_train = derivative_eng.fit_transform(X_train, y_train)
    X_test  = derivative_eng.transform(X_test)

    # ------------------------------------------------------------------
    # Step 4 — Temporal features (fit on train only)
    # Produces: dow_month_encoded
    # ------------------------------------------------------------------
    temporal_eng = TemporalFeatureEngineer()
    X_train = temporal_eng.fit_transform(X_train, y_train)
    X_test  = temporal_eng.transform(X_test)

    # ------------------------------------------------------------------
    # Step 5 — Prior features (leakage-free; needs economic_stress_integral)
    # Produces: has_prior_contact, prior_x_stress
    # ------------------------------------------------------------------
    X_train = add_prior_features(X_train)
    X_test  = add_prior_features(X_test)

    # ------------------------------------------------------------------
    # Step 6 — Overlap features (leakage-free; needs all prior steps)
    # Produces: low_stress_zone, cpi_high_cellular, behavioral_favorability,
    #           overlap_default_clean, overlap_behavioral_score
    # ------------------------------------------------------------------
    X_train = add_overlap_features(X_train)
    X_test  = add_overlap_features(X_test)

    return X_train, X_test, list(EBM_FEATURES)


def select_ebm_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features: list[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Positive-select to EBM feature columns only.

    Strips all intermediate scaffolding (economic_crisis_score,
    low_stress_zone, has_prior_contact, etc.) and any UCI columns
    not consumed by the EBM stage.

    Parameters
    ----------
    features : list[str] or None
        Override the feature list for pruning experiments.
        Defaults to EBM_FEATURES (14 features).
        Pass EBM_FEATURES_13 or EBM_FEATURES_12 for pruning runs.

    Returns
    -------
    X_train, X_test restricted to `features` columns.
    """
    if features is None:
        features = EBM_FEATURES

    missing = [f for f in features if f not in X_train.columns]
    if missing:
        raise KeyError(
            f"select_ebm_features: columns missing from X_train: {missing}\n"
            "Ensure engineer_ebm_features() has been called first."
        )

    return X_train[features], X_test[features]


def prune_redundant_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DEPRECATED — retained as passthrough to avoid breaking ebm_stage.py.

    The original prototype version dropped raw UCI columns that had been
    replaced by engineered surrogates (age, default, campaign, cons.conf.idx,
    etc.). In the validated pipeline these columns ARE the EBM features and
    must not be removed here.

    Feature selection is now handled by select_ebm_features() which
    positive-selects to EBM_FEATURES after engineering is complete.

    TODO: Remove this function and its call-site in ebm_stage.py once
    the stage file is updated.
    """
    return X_train, X_test