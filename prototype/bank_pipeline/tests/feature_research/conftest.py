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


@pytest.fixture
def stage3_dag_transition():
    """PR 42 accepted identity and the reviewed PR 43 Step 5B successor (11 modules)."""
    return ("3a8d195700ce34627f7d10d59294212ba967bcd69fe14a1c5741e87e25720017",
            "a38638e9499a40d59db3bbe5386e0c0ac3ed65581948680a31e4a5c26ff9a83c")
