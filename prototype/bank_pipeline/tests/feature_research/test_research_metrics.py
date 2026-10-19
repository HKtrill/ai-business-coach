"""
Feature-ranking statistics (feature_research.metrics / separation / interactions).

Checked against independent references (scipy, statsmodels) or hand-computed
values, and against data with a planted signal, rather than against the
implementation itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp, pointbiserialr
from scipy.stats.contingency import association
from statsmodels.stats.proportion import proportion_confint

from feature_research import (
    classify_features, compute_all_separations, sanity_check_data, search_interactions_mi,
)
from feature_research.interactions import quantile_bin_codes
from feature_research.metrics import cohens_d, cramers_v, wilson_ci
from feature_research.separation import compute_categorical_separation, compute_numeric_separation
from tests.support.synthetic import TARGET


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def test_cohens_d_matches_hand_computed_values():
    assert cohens_d(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])) == pytest.approx(-3.0)
    assert cohens_d(np.array([4.0, 5.0, 6.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(3.0)
    assert cohens_d(np.ones(5), np.ones(4)) == 0.0                      # zero pooled SD


@pytest.mark.parametrize("successes, trials", [(0, 10), (3, 10), (80, 400), (400, 400)])
def test_wilson_interval_matches_statsmodels(successes, trials):
    lo, hi = wilson_ci(successes, trials)
    ref_lo, ref_hi = proportion_confint(successes, trials, alpha=0.05, method="wilson")
    assert (lo, hi) == pytest.approx((ref_lo, ref_hi), abs=1e-9)
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_cramers_v_matches_scipy_beyond_two_levels():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.integers(0, 4, 300))
    y = pd.Series((rng.random(300) < 0.2 + 0.1 * (x == 3)).astype(int))
    assert cramers_v(x, y) == pytest.approx(association(pd.crosstab(x, y).to_numpy(), method="cramer"))
    assert cramers_v(pd.Series(["a"] * 10), pd.Series([0, 1] * 5)) == 0.0   # degenerate table


def test_cramers_v_binary_features_use_yates_correction_f13():
    """F13: for a 2-level feature scipy's default Yates correction applies, so a
    perfect association scores below the textbook value of 1.0."""
    x = pd.Series([0] * 50 + [1] * 50)
    y = pd.Series([0] * 50 + [1] * 50)
    textbook = association(pd.crosstab(x, y).to_numpy(), method="cramer")
    assert textbook == pytest.approx(1.0)
    assert cramers_v(x, y) == pytest.approx(0.98, abs=1e-6)


# ----------------------------------------------------------------------
# separation metrics
# ----------------------------------------------------------------------
def test_numeric_separation_uses_the_reference_statistics(bank_frame):
    m = compute_numeric_separation(bank_frame, "euribor3m", TARGET)
    c0 = bank_frame.loc[bank_frame[TARGET] == 0, "euribor3m"]
    c1 = bank_frame.loc[bank_frame[TARGET] == 1, "euribor3m"]
    assert m.ks_stat == pytest.approx(ks_2samp(c0, c1).statistic)
    assert m.point_biserial == pytest.approx(pointbiserialr(bank_frame[TARGET], bank_frame["euribor3m"])[0])
    assert m.cohens_d < 0 and m.point_biserial < 0      # positives have lower euribor (planted)


def test_categorical_separation_ranks_the_signal_above_noise(bank_frame):
    signal = compute_categorical_separation(bank_frame, "contact", TARGET)
    noise = compute_categorical_separation(bank_frame, "marital", TARGET)
    assert signal.cramers_v > noise.cramers_v


def test_feature_ranking_is_deterministic_and_puts_the_signal_first(bank_frame):
    num, cat = classify_features(bank_frame, TARGET)
    a = compute_all_separations(bank_frame, num, cat, TARGET)
    b = compute_all_separations(bank_frame, num, cat, TARGET)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["feature"]) == set(num) | set(cat)
    assert a["composite_score"].is_monotonic_decreasing
    assert a["feature"].iloc[0] == "euribor3m"


def test_feature_typing_and_sanity_check(bank_frame):
    assert sanity_check_data(bank_frame) == TARGET
    num, cat = classify_features(bank_frame, TARGET)
    assert TARGET not in num + cat
    assert set(num) | set(cat) == set(bank_frame.columns) - {TARGET}
    assert not set(num) & set(cat)


# ----------------------------------------------------------------------
# interactions
# ----------------------------------------------------------------------
def test_interaction_search_is_deterministic_and_ranked(bank_frame):
    num, cat = classify_features(bank_frame, TARGET)
    kw = dict(numeric_features=num, top_k=8, max_pairs=60)
    a = search_interactions_mi(bank_frame, num + cat, TARGET, **kw)
    pd.testing.assert_frame_equal(a, search_interactions_mi(bank_frame, num + cat, TARGET, **kw))
    assert len(a) == 8
    assert a["lift"].is_monotonic_decreasing
    np.testing.assert_allclose(a["lift"], a["mi_joint"] - a["mi_sum"])
    assert (a["feature_1"] != a["feature_2"]).all()


def test_quantile_bin_codes_are_ordered_and_bounded(bank_frame):
    values = bank_frame["euribor3m"]
    codes = quantile_bin_codes(values, 5)
    assert set(np.unique(codes)) == set(range(5))
    order = np.argsort(values.to_numpy(), kind="stable")
    assert (np.diff(codes[order]) >= 0).all()                           # monotone in the value
