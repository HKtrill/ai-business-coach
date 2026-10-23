"""
Shared pytest fixtures for the bank pipeline suite.

Builders live in ``tests/support`` as plain functions; fixtures here are thin,
function-scoped wrappers, so each test gets a fresh object it may modify.
"""

from __future__ import annotations

import pytest

from tests.support.recorders import CallLog
from tests.support.synthetic import make_bank_frame, split_frame


def pytest_collection_modifyitems(config, items):
    """
    ``realdata`` tests run only when the ``-m`` expression names ``realdata``
    (``pytest -m realdata``). Any other expression — ``-m "not slow"``, which
    replaces the default selection in pytest.ini — leaves them out, so the
    synthetic suites never need saved artifacts.
    """
    if "realdata" in (config.getoption("markexpr") or ""):
        return
    dropped = [item for item in items if item.get_closest_marker("realdata") is not None]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [item for item in items if item.get_closest_marker("realdata") is None]


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
