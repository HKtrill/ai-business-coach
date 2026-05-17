"""
feature_research.feature_engineering.overlap
=============================================
Overlap-zone interaction features extracted from the original Cell 10F.

Survivors after dead-feature audit (~30 → 5):
  - low_stress_zone          INTERMEDIATE — explicit column; Cell 10G references
                              it by name for overlap_default_clean and
                              overlap_behavioral_score.  economic_stress_integral < 0.2.
  - cpi_high_cellular        LIVE (RF Stage 2, EBM Stage 3)
  - behavioral_favorability  LIVE (RF Stage 2, EBM Stage 3)
  - overlap_default_clean    LIVE (EBM Stage 3)
  - overlap_behavioral_score LIVE (EBM Stage 3)

Pruned (~25 features):
  in_overlap_zone, in_moderate_zone, overlap_cellular, overlap_good_month,
  overlap_avoid_month, overlap_prior_contact, overlap_default_unknown,
  cpi_low_cellular, good_month_low_cpi, cpi_conf_aligned, macro_sentiment,
  good_month_low_conf, senior_high_edu, senior_low_edu, young_high_edu,
  default_unk_cellular, default_unk_telephone, default_unk_crisis,
  default_unk_prior, moderate_zone_low_conf, moderate_zone_sentiment,
  moderate_zone_good_month, overlap_density, high_stress_density, overlap_decay.

DAG position: LAST module — requires outputs from every prior module:
  crisis.py    → cellular_contact, high_conversion_month, default_clean
  integrals.py → economic_stress_integral (for low_stress_zone)
  prior.py     → has_prior_contact (for behavioral_favorability)

Intermediate dependencies (all confirmed in df at call time):
  cellular_contact       (crisis.py)
  high_conversion_month  (crisis.py)
  default_clean          (crisis.py)
  economic_stress_integral (integrals.py)
  has_prior_contact      (prior.py)
"""

import numpy as np
import pandas as pd

__all__ = ["add_overlap_features"]

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------
_LOW_STRESS_THRESH: float = 0.2   # economic_stress_integral < this → overlap zone
_CPI_HIGH_THRESH: float = 93.5    # cons.price.idx threshold from MI analysis

# behavioral_favorability component weights (from Cell 10F Part F)
_BF_WEIGHTS: dict = {
    'cellular':      3,
    'good_month':    3,
    'prior_contact': 2,
    'default_clean': 1,
    'age_senior':    1,
}
_BF_TOTAL_WEIGHT: float = float(sum(_BF_WEIGHTS.values()))

_AGE_SENIOR_THRESH: int = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_overlap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overlap-zone interaction and behavioral composite features.

    Must be called last in the pipeline — requires crisis.py, integrals.py,
    and prior.py outputs to be present in df.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after all prior add_*_features() calls.
        Required columns: economic_stress_integral, cons.price.idx, contact,
        cellular_contact, high_conversion_month, has_prior_contact,
        default_clean, age.

    Returns
    -------
    pd.DataFrame
        Copy of df with overlap features appended.

    New columns
    -----------
    low_stress_zone          int  {0,1}
        economic_stress_integral < 0.2.
        INTERMEDIATE — ~60-65% of dataset; economic features lose discriminative
        power here (healthy economy regime).  Feeds overlap_* features.

    cpi_high_cellular        int  {0,1}
        (cons.price.idx >= 93.5) AND (contact == 0).
        LIVE (RF Stage 2, EBM Stage 3).
        Top MI interaction: 50.2% lift.  Contact method effect modulated
        by consumer price environment.

    behavioral_favorability  float  [0, 1]
        Weighted sum of five behavioral signals, normalised to [0,1]:
          cellular(w=3) + good_month(w=3) + prior_contact(w=2)
          + default_clean(w=1) + age≥50(w=1)  /  10
        LIVE (RF Stage 2, EBM Stage 3).
        Aggregate prospect quality, independent of economic conditions.

    overlap_default_clean    int  {0,1}
        low_stress_zone × (default == 0).
        LIVE (EBM Stage 3).
        Clean credit history signal, active ONLY in the overlap zone where
        economic features cannot disambiguate.

    overlap_behavioral_score float  [0, 1]
        low_stress_zone × behavioral_favorability.
        LIVE (EBM Stage 3).
        Behavioral quality activated ONLY in the healthy-economy regime.
    """
    _check_required_columns(df)
    df = df.copy()

    # ------------------------------------------------------------------
    # Intermediate: explicit low-stress zone flag
    # ------------------------------------------------------------------
    df['low_stress_zone'] = (
        df['economic_stress_integral'] < _LOW_STRESS_THRESH
    ).astype(int)

    # ------------------------------------------------------------------
    # LIVE: cpi_high_cellular  (RF Stage 2, EBM Stage 3)
    # ------------------------------------------------------------------
    df['cpi_high_cellular'] = (
        (df['cons.price.idx'] >= _CPI_HIGH_THRESH)
        & (df['contact'] == 0)
    ).astype(int)

    # ------------------------------------------------------------------
    # LIVE: behavioral_favorability  (RF Stage 2, EBM Stage 3)
    # Weighted combination of five orthogonal behavioral signals.
    # Uses pre-computed intermediates from crisis.py and prior.py rather
    # than recomputing inline (as original Cell 10F did).
    # ------------------------------------------------------------------
    df['behavioral_favorability'] = (
        _BF_WEIGHTS['cellular']      * df['cellular_contact']
        + _BF_WEIGHTS['good_month']    * df['high_conversion_month']
        + _BF_WEIGHTS['prior_contact'] * df['has_prior_contact']
        + _BF_WEIGHTS['default_clean'] * df['default_clean']
        + _BF_WEIGHTS['age_senior']    * (df['age'] >= _AGE_SENIOR_THRESH).astype(int)
    ) / _BF_TOTAL_WEIGHT

    # ------------------------------------------------------------------
    # LIVE: overlap_default_clean  (EBM Stage 3)
    # ------------------------------------------------------------------
    df['overlap_default_clean'] = (
        df['low_stress_zone'] * df['default_clean']
    )

    # ------------------------------------------------------------------
    # LIVE: overlap_behavioral_score  (EBM Stage 3)
    # ------------------------------------------------------------------
    df['overlap_behavioral_score'] = (
        df['low_stress_zone'] * df['behavioral_favorability']
    )

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS: tuple = (
    'economic_stress_integral',  # integrals.py
    'cons.price.idx',            # raw UCI
    'contact',                   # raw UCI
    'cellular_contact',          # crisis.py
    'high_conversion_month',     # crisis.py
    'default_clean',             # crisis.py
    'has_prior_contact',         # prior.py
    'age',                       # raw UCI
)


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_overlap_features is missing required columns: {missing}\n"
            "Ensure the full pipeline has run: crisis → integrals → derivatives "
            "→ temporal → prior → overlap."
        )