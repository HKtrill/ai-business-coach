"""
feature_research.feature_engineering.pipeline
==============================================
FeaturePipeline — the full feature DAG with every target-dependent statistic
fitted on training rows only:

    crisis       add_crisis_features        deterministic
    integrals    IntegralFeatureEngineer    fit on train
    derivatives  DerivativeFeatureEngineer  fit on train (needs integrals)
    temporal     TemporalFeatureEngineer    fit on train
    prior        add_prior_features         deterministic (needs integrals)
    overlap      add_overlap_features       deterministic (needs all above)

finalize_features / get_stage_df select the model-ready columns.
"""

import pandas as pd

from feature_research.config import RANDOM_SEED
from feature_research.feature_engineering._checks import check_xy_aligned
from feature_research.feature_engineering.crisis import add_crisis_features
from feature_research.feature_engineering.integrals import IntegralFeatureEngineer
from feature_research.feature_engineering.derivatives import DerivativeFeatureEngineer
from feature_research.feature_engineering.temporal import TemporalFeatureEngineer
from feature_research.feature_engineering.prior import add_prior_features
from feature_research.feature_engineering.overlap import add_overlap_features

__all__ = ["FeaturePipeline", "finalize_features", "get_stage_df"]


# ---------------------------------------------------------------------------
# Fit-on-train / transform orchestrator (use this in the research notebook)
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Full feature DAG with every target-dependent statistic fitted on the
    rows passed to fit_transform() only.

    DAG order:
        crisis -> integrals -> derivatives -> temporal -> prior -> overlap

    Deterministic modules (crisis, prior, overlap) are applied as-is; the
    three target-dependent modules use their fit/transform engineers.

    Usage
    -----
        fp = FeaturePipeline()
        X_train_fe = fp.fit_transform(X_train, y_train)
        X_test_fe  = fp.transform(X_test)

    X must NOT contain the target column. transform() output must have
    exactly the same columns, in the same order and with the same dtypes,
    as fit_transform() output.
    """

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        n_bins: int = 20,
        smoothing_factor: int = 100,
    ) -> None:
        self.random_state = random_state
        self.n_bins = n_bins
        self.smoothing_factor = smoothing_factor
        self._integral = IntegralFeatureEngineer(random_state=random_state)
        self._derivative = DerivativeFeatureEngineer(n_bins=n_bins)
        self._temporal = TemporalFeatureEngineer(smoothing_factor=smoothing_factor)
        self.input_columns_: list = None
        self.output_columns_: list = None
        self.output_dtypes_: dict = None
        self.fitted = False

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        check_xy_aligned(X, y, "FeaturePipeline.fit_transform")
        self.input_columns_ = list(X.columns)
        X = add_crisis_features(X)
        X = self._integral.fit_transform(X, y)
        X = self._derivative.fit_transform(X, y)
        X = self._temporal.fit_transform(X, y)
        X = add_prior_features(X)
        X = add_overlap_features(X)
        self.output_columns_ = list(X.columns)
        self.output_dtypes_ = X.dtypes.astype(str).to_dict()
        self.fitted = True
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("FeaturePipeline not fitted. Call fit_transform() first.")
        if list(X.columns) != self.input_columns_:
            raise ValueError(
                "FeaturePipeline.transform: input columns differ from fit "
                f"(names or order).\n  fit:  {self.input_columns_}\n  got:  {list(X.columns)}"
            )
        X = add_crisis_features(X)
        X = self._integral.transform(X)
        X = self._derivative.transform(X)
        X = self._temporal.transform(X)
        X = add_prior_features(X)
        X = add_overlap_features(X)
        if list(X.columns) != self.output_columns_:
            raise RuntimeError("FeaturePipeline.transform: output columns differ from fit.")
        dtypes = X.dtypes.astype(str).to_dict()
        changed = {c: (self.output_dtypes_[c], dtypes[c])
                   for c in dtypes if dtypes[c] != self.output_dtypes_[c]}
        if changed:
            raise TypeError(f"FeaturePipeline.transform: dtypes differ from fit (fit, got): {changed}")
        return X


# ---------------------------------------------------------------------------
# Helpers for model-ready feature selection
# ---------------------------------------------------------------------------

def finalize_features(
    df: pd.DataFrame,
    live_features: dict,
    target_col: str = 'y',
) -> pd.DataFrame:
    """
    Positive-select to only model-consumed columns: union of all stage
    features + target.  Everything else (original UCI columns not used
    by any model, all intermediate scaffolding) is dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Output of FeaturePipeline (plus target column).
    live_features : dict[str, list[str]]
        LIVE_FEATURES registry, e.g. from feature_engineering.__init__.
    target_col : str
        Target column to retain alongside features.

    Returns
    -------
    pd.DataFrame
        Shape: (n_rows, n_unique_model_features + 1)
        Columns ordered as they appear in df.
    """
    all_model_cols: set = set()
    for features in live_features.values():
        all_model_cols.update(features)
    if target_col in df.columns:
        all_model_cols.add(target_col)
    # Preserve df column order
    keep = [c for c in df.columns if c in all_model_cols]
    return df[keep]


def get_stage_df(
    df: pd.DataFrame,
    stage: str,
    live_features: dict,
    target_col: str = 'y',
) -> pd.DataFrame:
    """
    Return a model-ready dataframe for a single cascade stage.

    Parameters
    ----------
    df : pd.DataFrame
        Output of finalize_features().
    stage : str
        One of 'lr', 'rf', 'ebm'.
    live_features : dict[str, list[str]]
        LIVE_FEATURES registry.
    target_col : str
        Target column to include.

    Returns
    -------
    pd.DataFrame
        Contains only that stage's feature columns + target.
        LR → 3+1, RF → 8+1, EBM → 14+1 columns.
    """
    if stage not in live_features:
        raise ValueError(f"stage must be one of {list(live_features)}; got {stage!r}")
    cols = live_features[stage] + (
        [target_col] if target_col in df.columns else []
    )
    if len(set(cols)) != len(cols):
        raise ValueError(f"get_stage_df: duplicated columns in {stage!r} registry: {cols}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"get_stage_df: columns missing from df: {missing}")
    return df[cols]