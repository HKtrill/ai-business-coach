"""
feature_research/config.py
==========================
Centralized configuration for the Glass Cascade exploratory research notebook.

Owns:
- Output / figure directory setup
- Global reproducibility settings (random seed, plot style)
- Third-party logging suppression

Import this module first in any notebook or script that participates in the
feature-research pipeline so that all downstream code shares identical settings.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: feature_research/ package directory, resolved from this file — independent
#: of the notebook or script's working directory.
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

#: Root of all research artefacts written by this pipeline.
OUTPUT_DIR: Path = PACKAGE_ROOT / "research_logs"

#: Subdirectory for all matplotlib / seaborn figures.
FIG_DIR: Path = OUTPUT_DIR / "figures"


def setup_directories() -> None:
    """Create output directories if they do not already exist.

    Paths are anchored to the package directory, so outputs land in the same
    place regardless of where the notebook or script is run from. Idempotent.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory : {OUTPUT_DIR}")
    print(f"📁 Figures directory: {FIG_DIR}")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Global random seed — seeds NumPy here; pass it explicitly as random_state /
#: seed to sklearn and Optuna.
RANDOM_SEED: int = 42


def apply_global_settings() -> None:
    """Apply plot style, random seed, and logging verbosity.

    Only Optuna's logging and experimental warnings are silenced; all other
    warnings (sklearn feature-name mismatches, convergence, pandas) stay
    visible. Idempotent. Call after :func:`setup_directories`.
    """
    np.random.seed(RANDOM_SEED)
    plt.style.use("seaborn-v0_8-darkgrid")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    print(f"🎲 Random seed      : {RANDOM_SEED}")
    print("🎨 Plot style       : seaborn-v0_8-darkgrid")
    print("🔇 Optuna logging / experimental warnings suppressed")