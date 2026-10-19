"""
Shared pytest fixtures for the bank pipeline suite.

Builders live in ``tests/support`` as plain functions; fixtures here are thin,
function-scoped wrappers, so each test gets a fresh object it may modify.
"""

from __future__ import annotations

import pytest

from tests.support.recorders import CallLog
from tests.support.synthetic import make_bank_frame, split_frame


@pytest.fixture
def bank_frame():
    """400 bank-like rows, 20% positives, planted signal, irregular index."""
    return make_bank_frame()


@pytest.fixture
def null_signal_frame():
    """Same schema and prevalence, labels independent of every feature."""
    return make_bank_frame(signal=False)


@pytest.fixture
def bank_split(bank_frame):
    """Stratified 75/25 ``GLOBAL_SPLIT``-style dict of ``bank_frame``."""
    return split_frame(bank_frame)


@pytest.fixture
def call_log():
    """Empty, clone-proof log for ``RecordingEstimator``."""
    return CallLog()
