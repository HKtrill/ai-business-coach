"""
feature_research
================
Exploratory research modules for the Glass Cascade ML pipeline.

Public API
----------
config        — output paths, global settings, random seed
data          — raw data loading and preprocessing
metrics       — SeparationMetrics container + cramers_v / cohens_d / wilson_ci
validation    — sanity_check_data, classify_features
separation    — compute_*_separation, compute_all_separations, display_feature_rankings
visualization — plot_numeric_feature, plot_categorical_feature, generate_all_plots
interactions  — search_interactions_mi, display_interaction_rankings
"""

from feature_research.config import (
    FIG_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
    apply_global_settings,
    setup_directories,
)
from feature_research.data import load_and_preprocess
from feature_research.metrics import (
    SeparationMetrics,
    cohens_d,
    cramers_v,
    wilson_ci,
)
from feature_research.validation import classify_features, sanity_check_data
from feature_research.separation import (
    compute_all_separations,
    compute_categorical_separation,
    compute_numeric_separation,
    display_feature_rankings,
)
from feature_research.visualization import (
    generate_all_plots,
    plot_categorical_feature,
    plot_numeric_feature,
)
from feature_research.interactions import (
    search_interactions_mi,
    display_interaction_rankings,
)

__all__ = [
    # config
    "OUTPUT_DIR",
    "FIG_DIR",
    "RANDOM_SEED",
    "setup_directories",
    "apply_global_settings",
    # data
    "load_and_preprocess",
    # metrics
    "SeparationMetrics",
    "cramers_v",
    "cohens_d",
    "wilson_ci",
    # validation
    "sanity_check_data",
    "classify_features",
    # separation
    "compute_numeric_separation",
    "compute_categorical_separation",
    "compute_all_separations",
    "display_feature_rankings",
    # visualization
    "plot_numeric_feature",
    "plot_categorical_feature",
    "generate_all_plots",
    # interactions
    "search_interactions_mi",
    "display_interaction_rankings",
]