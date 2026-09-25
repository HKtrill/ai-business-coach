"""
feature_research.feature_engineering.crisis
============================================
Bare-minimum features extracted from the original Cell 10A exploration.

After the dead-feature audit, only ONE feature survives directly from 10A:
  - cellular_crisis  (LR Stage 1)

Plus four intermediates required by downstream modules:
  - economic_crisis_score  → cellular_crisis (here) + cellular_crisis check
  - high_conversion_month  → behavioral_favorability (Cell 10F)
  - cellular_contact       → behavioral_favorability (Cell 10F)
  - has_prior_contact      → prior_x_stress (Cell 10E; was warm_lead in 10A)
  - default_clean          → behavioral_favorability + overlap_default_clean (Cell 10F)

Everything else from the original engineer_bank_features() (26 features across
Tier 1/2/3/4/5) was pruned as dead code — zero coverage in LR/RF/EBM configs.
"""

import pandas as pd

__all__ = ["add_crisis_features"]

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------
_CRISIS_EURIBOR_THRESH: float = 1.5
_CRISIS_EMP_VAR_THRESH: float = -1.0
_CRISIS_NR_EMP_THRESH: float = 5100.0
_CRISIS_SCORE_CUTOFF: int = 3

_HIGH_CONVERSION_MONTHS: tuple = (3, 9, 10, 12)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_crisis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add economic crisis indicators and behavioural intermediates.

    Produces the single live Cell-10A survivor (cellular_crisis) plus the
    set of intermediate binary flags consumed by downstream modules:
    integrals.py, and the upcoming Cell 10E / 10F modules.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.  Must contain: euribor3m, emp.var.rate,
        nr.employed, contact, month, previous, default.

    Returns
    -------
    pd.DataFrame
        Copy of df with crisis features appended.

    New columns
    -----------
    economic_crisis_score   int   [0-6]   Weighted sum of three crisis flags.
                                          INTERMEDIATE — feeds cellular_crisis.
    cellular_crisis         int   {0,1}   LIVE (LR Stage 1).
                                          cellular contact during economic crisis.
    high_conversion_month   int   {0,1}   INTERMEDIATE — feeds behavioral_favorability (10F).
                                          Month in {Mar, Sep, Oct, Dec}.
    cellular_contact        int   {0,1}   INTERMEDIATE — feeds behavioral_favorability (10F).
                                          contact == 0 (cellular).
    default_clean           int   {0,1}   INTERMEDIATE — feeds behavioral_favorability
                                          and overlap_default_clean (10F).
                                          default == 0 (no default).

    NOTE: has_prior_contact lives in prior.py (Cell 10E), not here.
    overlap.py (behavioral_favorability) runs after prior.py in the DAG.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Intermediate: weighted economic crisis score
    # Weights from Cell 10A feature importance analysis:
    #   euribor3m < 1.5%  → weight 3 (strongest signal)
    #   emp.var.rate < -1 → weight 2
    #   nr.employed < 5100 → weight 1
    # Score range [0, 6]; threshold >= 3 defines "in crisis"
    # ------------------------------------------------------------------
    df['economic_crisis_score'] = (
        (df['euribor3m'] < _CRISIS_EURIBOR_THRESH).astype(int) * 3
        + (df['emp.var.rate'] < _CRISIS_EMP_VAR_THRESH).astype(int) * 2
        + (df['nr.employed'] < _CRISIS_NR_EMP_THRESH).astype(int) * 1
    )

    # ------------------------------------------------------------------
    # LIVE: cellular_crisis  (LR Stage 1)
    # cellular (contact==0) AND economic crisis score >= 3
    # Captures best channel during economic downturns.
    # ------------------------------------------------------------------
    df['cellular_crisis'] = (
        (df['contact'] == 0)
        & (df['economic_crisis_score'] >= _CRISIS_SCORE_CUTOFF)
    ).astype(int)

    # ------------------------------------------------------------------
    # Intermediates — consumed by Cell 10F (behavioral_favorability)
    # ------------------------------------------------------------------
    df['high_conversion_month'] = (
        df['month'].isin(_HIGH_CONVERSION_MONTHS)
    ).astype(int)

    df['cellular_contact'] = (df['contact'] == 0).astype(int)

    # ------------------------------------------------------------------
    # Intermediate — consumed by Cell 10F (overlap.py)
    # (Not in original Cell 10A; added here as logical home)
    # Note: has_prior_contact lives in prior.py, NOT here — Cell 10E
    # is its natural home and overlap.py runs after prior.py.
    # ------------------------------------------------------------------
    df['default_clean'] = (df['default'] == 0).astype(int)

    return df