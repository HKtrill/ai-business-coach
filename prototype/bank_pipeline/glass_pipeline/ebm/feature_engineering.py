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
    TemporalFeatureEngineer are TARGET-DEPENDENT: they fit on (X, y).
    Fitting them once on all of X_train and then cross-validating on the
    result leaks every validation row's label into its own features.

    Since PR 33 Stage 3 uses ``EBMFeaturePipeline`` through
    ``shared.stage_runner.Stage3Block``, which refits the pipeline inside every
    fold (fit on the fold's training rows, transform its validation rows and
    the test split). The full-train fit is used only for the refit model and
    the test split, and is saved in the artifact for serving.

    ``engineer_ebm_features()`` (fit on all of X_train) is kept for research
    code and the legacy path; Stage 3 training must not cross-validate on it.

Pruning experiment (correlation / interpretability pass):
    overlap_behavioral_score  — drop candidate (|r|=0.716 with
                                economic_curvature_intensity; safer cut)
    emp_var_rate_sigmoid_slope — drop candidate (|r|=0.724 with
                                euribor3m_sigmoid_slope; riskier — anchors
                                TIER 1 interaction campaign×emp_var_rate_sigmoid_slope,
                                importance=0.205)
    overlap_behavioral_score is already OUT of the production list, so
    EBM_FEATURES and EBM_FEATURES_13 are the same 13 features.
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

# Production 13-feature set. The research baseline had 14; overlap_behavioral_score
# (pruning cut 1) is commented out below, so EBM_FEATURES == EBM_FEATURES_13.
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

# Pruning experiment — cut 1: drop overlap_behavioral_score.
# Identical to EBM_FEATURES while that feature is commented out above; kept so
# the research baseline (EBM_FEATURES + overlap_behavioral_score) stays nameable.
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


def _drop_leaky(X: pd.DataFrame) -> pd.DataFrame:
    present = [f for f in LEAKY_FEATURES if f in X.columns]
    return X.drop(columns=present) if present else X


def _run_dag(X: pd.DataFrame, engineers: dict, y: pd.Series | None) -> pd.DataFrame:
    """
    Apply the DAG to one frame.

    ``y`` given → fit_transform each target-dependent engineer (training rows).
    ``y`` None  → transform with already-fitted engineers (any other rows).
    """
    fit = y is not None
    X = add_crisis_features(X)
    for name in ("integral", "derivative", "temporal"):
        eng = engineers[name]
        X = eng.fit_transform(X, y) if fit else eng.transform(X)
    X = add_prior_features(X)
    X = add_overlap_features(X)
    return X


def _new_engineers() -> dict:
    return {
        "integral": IntegralFeatureEngineer(),     # economic_stress_integral,
                                                   # neighborhood_subscription_density
        "derivative": DerivativeFeatureEngineer(), # sigmoid slopes, local_rate,
                                                   # curvature, decay composites
        "temporal": TemporalFeatureEngineer(),     # dow_month_encoded
    }


def engineer_ebm_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Apply the full validated research DAG in fit-on-train / transform order.

    DAG: crisis → integrals → derivatives → temporal → prior → overlap

    The target-dependent engineers are fitted on ALL of ``X_train``. Do not
    cross-validate on the output — Stage 3 uses ``EBMFeaturePipeline`` per
    fold instead (see module docstring).

    Returns
    -------
    X_train, X_test : frames with all engineered columns appended.
    features_added  : EBM_FEATURES.
    """
    engineers = _new_engineers()
    X_train = _run_dag(X_train, engineers, y_train)
    X_test = _run_dag(X_test, engineers, None)
    return X_train, X_test, list(EBM_FEATURES)


class EBMFeaturePipeline:
    """
    The Stage 3 feature pipeline as one fit/transform object.

    drop leaky → crisis → integrals → derivatives → temporal → prior →
    overlap → positive-select ``features`` (default ``EBM_FEATURES``).

    ``fit_transform(X, y)`` fits the three target-dependent engineers on
    ``(X, y)`` only; ``transform(X)`` reuses them. Both preserve the index.
    Pass the class itself as ``pipeline_factory`` to ``Stage3Block`` so a fresh
    instance is fitted per fold. A fitted instance is saved in every Stage 3
    artifact (``refit["feature_pipeline"]``) to score new raw rows.
    """

    def __init__(self, features: list[str] | None = None):
        self.features = list(features) if features is not None else list(EBM_FEATURES)
        self.engineers_: dict | None = None
        self.n_fit_rows_: int = 0

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if y is None:
            raise ValueError("EBMFeaturePipeline.fit_transform needs y")
        y = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)
        if not y.index.equals(X.index):
            raise ValueError("y index must match X index")
        self.engineers_ = _new_engineers()
        out = _run_dag(_drop_leaky(X.copy()), self.engineers_, y)
        self.n_fit_rows_ = int(len(X))
        return self._select(out)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EBMFeaturePipeline":
        self.fit_transform(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.engineers_ is None:
            raise ValueError("EBMFeaturePipeline: call fit_transform() first")
        return self._select(_run_dag(_drop_leaky(X.copy()), self.engineers_, None))

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise KeyError(f"EBMFeaturePipeline: DAG did not produce {missing}")
        return X[self.features]


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
        Defaults to EBM_FEATURES (13 features).
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
