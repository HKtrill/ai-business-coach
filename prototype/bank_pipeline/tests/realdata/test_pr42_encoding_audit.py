"""
PR 42 Commit 3, Step 3A on the real training rows - opt-in, like every
realdata test (it reads ``data/raw`` and ``data/db``):

    pytest -m realdata tests/realdata/test_pr42_encoding_audit.py

1. Each training row's index label is its line in the source CSV, through the
   database load, preprocessing and the split; the time index the audit derives
   from that order passes its calendar and indicator checks (26 periods, May
   2008 to November 2010, 485 contact days).
2. The Stage 2 outer fold 0 and its inner plan are the ones the accepted Stage 2
   run recorded (PR 38 ``c3_20261112_153333``, fold 0 inner-feature pathway).
3. The Stage 1 out-of-fold attribution (10 outer folds, both arms' shared
   designs) reconstructs the production encodings and gives the counts
   ``research_logs/pr42/encoding_audit.md`` reports.

Only training rows are bound; no artifact is opened.
"""

from __future__ import annotations

import pytest

from feature_research import encoding_audit as EA
from feature_research.training_side import draw_training_side

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def train():
    side = draw_training_side(verbose=False)
    return side.X_train, side.y_train


def test_training_rows_are_their_source_lines_and_the_time_index_holds(train):
    X, y = train
    source = EA.verify_source_order(X, y)
    assert source["passed"].all(), source.to_string()
    T = EA.build_time_index(X)
    checks = EA.time_index_checks(X, T)
    assert checks["passed"].all(), checks.to_string()
    periods = EA.period_table(T, y)
    assert len(periods) == 26 and periods["rows"].sum() == len(X) == 32_950
    assert (periods["period"].iloc[0], periods["period"].iloc[-1]) == ("2008-05", "2010-11")
    assert T["day"].nunique() == 485 and EA.recurring_day_keys(T, X) == 18


def test_the_stage2_fold0_plan_is_the_accepted_runs(train):
    X, y = train
    ref = EA.router_plan_reference(X, y, EA.fit_paths(y))
    assert ref["passed"].all(), ref.to_string()


def test_the_stage1_out_of_fold_attribution_reproduces_the_reported_counts(train):
    X, y = train
    T = EA.build_time_index(X)
    paths = EA.fit_paths(y)
    outer = {"stage1": [p for p in paths["stage1"] if p.path == "outer"]}
    summary = EA.path_summary(EA.audit_paths(X, y, T, outer, verbose=False))
    held = summary[summary["role"] == "held_out"].set_index("feature")
    assert (held["comparisons_passed"] == 10).all() and (held["comparisons_failed"] == 0).all()
    assert (held["max_abs_diff"] <= 1e-12).all() and (held["self_rows"] == 0).all()
    assert held["rows"].tolist() == [32_950, 32_950]
    assert held.loc["subscribe_rate_by_euribor_bin", "later_month_rows"] == 20_680
    assert held.loc["subscribe_rate_by_euribor_bin", "any_later_rows"] == 29_228
    assert held.loc["subscribe_rate_by_euribor_bin", "majority_later_rows"] == 13_311
    assert held.loc["subscribe_rate_by_dow_month", "later_month_rows"] == 31_142
    assert held.loc["subscribe_rate_by_dow_month", "any_later_rows"] == 32_941
    assert held.loc["subscribe_rate_by_dow_month", "majority_later_rows"] == 11_446
