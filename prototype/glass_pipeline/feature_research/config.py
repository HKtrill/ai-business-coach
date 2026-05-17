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

#: Root of all research artefacts written by this pipeline.
OUTPUT_DIR: Path = Path("research_logs")

#: Subdirectory for all matplotlib / seaborn figures.
FIG_DIR: Path = OUTPUT_DIR / "figures"


def setup_directories() -> None:
    """Create output directories if they do not already exist.

    Call once at notebook startup (Cell 3 equivalent).  Idempotent — safe to
    call multiple times.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory : {OUTPUT_DIR.absolute()}")
    print(f"📁 Figures directory: {FIG_DIR.absolute()}")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Global random seed shared by numpy, sklearn, and optuna samplers.
RANDOM_SEED: int = 42


def apply_global_settings() -> None:
    """Apply plot style, random seed, and logging verbosity.

    Idempotent.  Call after :func:`setup_directories` during notebook startup.
    """
    np.random.seed(RANDOM_SEED)
    plt.style.use("seaborn-v0_8-darkgrid")
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"🎲 Random seed      : {RANDOM_SEED}")
    print("🎨 Plot style       : seaborn-v0_8-darkgrid")
    print("🔇 Warnings / Optuna logging suppressed")