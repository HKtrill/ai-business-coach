"""Fixtures for the feature-research tests (builders live in tests/support)."""

from __future__ import annotations

from functools import partial

import pytest
from sklearn.model_selection import StratifiedKFold

from feature_research.feature_engineering import FeaturePipeline


@pytest.fixture
def make_fp():
    """Zero-arg factory for the notebook's FeaturePipeline (Cell 11 FE_PARAMS)."""
    return partial(FeaturePipeline, random_state=0, n_bins=20, smoothing_factor=100)


@pytest.fixture
def cv():
    """Small stand-in for the notebook's Cell 5 StratifiedKFold."""
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
