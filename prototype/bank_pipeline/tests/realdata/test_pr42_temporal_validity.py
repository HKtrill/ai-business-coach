"""
PR 42 Commit 3, Step 3B on the real training rows - opt-in, like every
realdata test (it reads ``data/raw`` and ``data/db``):

    pytest -m realdata tests/realdata/test_pr42_temporal_validity.py

1. The registered windows on the real time index, and every past-only state
   each rule uses in the full-history context equals the production encoder
   fitted on that state's rows (482 daily states for lag 0).
2. The availability counts ``research_logs/pr42/temporal_validity.md`` reports.
3. No unavailable label reaches a past-only encoding (40 probe days, the
   full-history context and a random fold's).
4. The LR comparisons (no MLP: the LR is bit-reproducible across platforms):
   the accepted LR reproduces the PR 39 Commit 2 best CV ROC-AUC exactly, and
   the protocol differences, pooled and within windows, are the reported ones.
5. The lag-consistent arm fits only outcomes known at each cutoff, and its
   regression (unavailable pre-cutoff outcomes flipped) holds.

Only training rows are bound; no artifact is opened.
"""

from __future__ import annotations

import numpy as np
import pytest

from feature_research import encoding_audit as EA
from feature_research import temporal_validity as TV
from feature_research.training_side import draw_training_side

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def train():
    side = draw_training_side(verbose=False)
    X, y = side.X_train, side.y_train
    return X, y, EA.build_time_index(X)


@pytest.fixture(scope="module")
def lr_run(train):
    X, y, T = train
    return TV.evaluate(X, y, T, mlp=False, verbose=False)


def test_the_windows_are_the_registered_ones(train):
    _, y, T = train
    w = TV.window_table(T, y).set_index("segment")
    assert list(w.index) == ["burn-in", "W1", "W2", "W3", "W4", "W5"]
    assert list(w["rows"]) == [19_179, 2_941, 2_178, 4_661, 1_543, 2_448]
    assert w.loc["burn-in", "periods"] == "2008-05..2008-08" and w.loc["W5", "periods"] == "2009-10..2010-11"


@pytest.mark.parametrize("rule", list(TV.RULES))
def test_every_state_equals_the_production_encoder(train, rule):
    X, y, T = train
    bank = TV.StateBank(X, y, T["day"].to_numpy(), np.arange(len(y)))
    res = TV.verify_against_production(X, y, T, bank, rule=rule, label="all")
    assert res["passed"] and res["max_abs_diff"] == 0.0, res
    assert res["states_compared"] == {"past_daily_lag0": 482, "past_daily_lag5": 477,
                                      "past_daily_lag20": 462, "past_monthly": 25}[rule]


def test_availability_is_the_reported_one(lr_run):
    A = TV.availability_tables(lr_run).set_index(["scope", "rule"])
    full = A.loc[("training rows, full history", TV.PRIMARY)]
    assert (full["available"], full["cold_start_rows"]) == (32_033, 917)
    assert (full["euribor_fallback_rows"], full["cell_unseen_rows"]) == (12_636, 3_929)
    assert (full["min_history_rows"], full["min_history_positives"]) == (917, 15)
    held = A.loc[("random held-out rows", TV.PRIMARY)]
    assert held["available"] == 32_033 and held["min_history_rows"] >= TV.MIN_STATE_ROWS
    assert TV.availability_checks(A.reset_index())["passed"].all()


def test_no_unavailable_label_reaches_a_past_only_encoding(train, lr_run):
    X, y, T = train
    ctx = {k: v for k, v in TV.invariance_contexts(lr_run).items()}
    checks = TV.invariance_checks(X, y, T, ctx)
    assert checks["passed"].all(), checks.to_string()
    assert lr_run["chrono_checks"]["passed"].all(), lr_run["chrono_checks"].to_string()


def test_the_lr_comparisons_are_the_reported_ones(lr_run):
    rep = TV.reproduction_checks(lr_run)
    assert rep["passed"].all(), rep.to_string()
    C = TV.comparisons(lr_run)
    pooled = C[C["scope"] == "pooled"].set_index("question")
    within = C[C["scope"] == "mean over windows"].set_index("question")
    assert pooled.loc["past-only vs accepted protocol, random split", "rows"] == 32_033
    assert pooled.loc["past-only vs accepted protocol, random split", "diff"] == pytest.approx(-0.0065, abs=5e-4)
    assert pooled.loc["chronological vs random split, accepted protocol", "diff"] == pytest.approx(-0.0886, abs=5e-4)
    eff = TV.factorial(lr_run, "lr")["effects"].set_index("effect")
    assert eff.loc["past-only vs accepted protocol, random split (B - A)", "mean_window_diff"] == \
        pytest.approx(-0.0464, abs=5e-4)
    d_b = eff.loc["chronological vs random split, past-only protocol (D - B)"]
    assert d_b["mean_window_diff"] == pytest.approx(-0.0005, abs=5e-4)
    assert d_b["mean_window_ci_low"] < 0 < d_b["mean_window_ci_high"]          # not resolved
    lag20 = "outcome lag 20 throughout (lag-consistent) vs lag 0, chronological"
    assert pooled.loc[lag20, "diff"] == pytest.approx(-0.1509, abs=5e-4)
    assert within.loc[lag20, "diff"] == pytest.approx(-0.0090, abs=5e-4)
    assert within.loc[lag20, "ci_low"] < 0 < within.loc[lag20, "ci_high"]      # not resolved
    only = "lag-0 model fed lag-20 encodings at prediction, chronological"
    assert pooled.loc[only, "diff"] == pytest.approx(-0.0501, abs=5e-4)


def test_the_lag_consistent_arm_uses_only_outcomes_known_at_each_cutoff(lr_run):
    fits = lr_run["lag_consistent_fits"].set_index(["window", "rule"])
    assert fits.loc[("W1", "past_daily_lag20"), "outcomes_known_at_cutoff"] == 14_644
    assert fits.loc[("W1", "past_daily_lag20"), "fit_rows"] == 7_408
    assert (fits.xs("past_daily_lag0", level="rule")["outcomes_known_at_cutoff"]
            == fits.xs("past_daily_lag0", level="rule")["training_rows_before_cutoff"]).all()
    checks = lr_run["chrono_checks"]
    regression = checks[checks["check"].str.contains("still unavailable|lag-0 model of the prediction")]
    assert len(regression) == 10 and regression["passed"].all(), regression.to_string()
