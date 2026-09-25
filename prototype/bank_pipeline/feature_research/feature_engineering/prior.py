"""
feature_research.feature_engineering.prior
===========================================
Previous-contact interaction features extracted from the original Cell 10E.

Survivors after dead-feature audit (8 → 2):
  - has_prior_contact    INTERMEDIATE — feeds prior_x_stress (here) and
                         behavioral_favorability in overlap.py.
                         Originally in crisis.py draft; correct home is here.
  - prior_x_stress       LIVE (EBM Stage 3)

Pruned:
  - previous_bucket        dead
  - prior_x_good_month     dead
  - prior_x_cellular       dead
  - cold_intensity         dead
  - warm_intensity         dead
  - prior_x_euribor_decay  dead

Background: `previous` ranked #4 (Composite=0.2966) but the raw signal is
almost entirely in the binary "has ever been contacted" split.  previous≥5
has 60-70% subscribe rate but tiny N; prior_x_stress captures the meaningful
interaction (prior relationship + economic stress) without tail noise.

DAG position: must run AFTER add_integral_features() (needs economic_stress_integral)
and BEFORE add_overlap_features() (overlap.py needs has_prior_contact for
behavioral_favorability).
"""

import pandas as pd

__all__ = ["add_prior_features"]


def add_prior_features(
    df: pd.DataFrame,
    target_col: str = 'y',
) -> pd.DataFrame:
    """
    Add prior-contact binary flag and its economic stress interaction.

    Requires columns: previous, economic_stress_integral.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after add_integral_features() (needs economic_stress_integral).
    target_col : str
        Retained for API symmetry; not used in this module.

    Returns
    -------
    pd.DataFrame
        Copy of df with prior features appended.

    New columns
    -----------
    has_prior_contact    int  {0, 1}
        Binary: previous >= 1.
        INTERMEDIATE — feeds prior_x_stress (here) and behavioral_favorability
        (overlap.py).

    prior_x_stress       float  [0, ~1+]
        has_prior_contact × economic_stress_integral.
        LIVE (EBM Stage 3).
        Captures: prior relationship value amplified during economic crisis.
        Absorbed cumulative_campaign_pressure signal in EBM Round 10.
    """
    if 'previous' not in df.columns:
        raise ValueError("add_prior_features requires 'previous' column.")
    if 'economic_stress_integral' not in df.columns:
        raise ValueError(
            "add_prior_features requires 'economic_stress_integral'. "
            "Run add_integral_features() first."
        )

    df = df.copy()

    df['has_prior_contact'] = (df['previous'] >= 1).astype(int)

    df['prior_x_stress'] = (
        df['has_prior_contact'] * df['economic_stress_integral']
    )

    return df