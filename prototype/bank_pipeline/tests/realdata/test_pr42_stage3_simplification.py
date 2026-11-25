"""
PR 42 Commit 5, Step 5A on the real training rows - opt-in, like every realdata
test (it reads ``data/raw`` and ``data/db``):

    pytest -m realdata tests/realdata/test_pr42_stage3_simplification.py

The notebook's Step 5A path up to the baseline fit (Cells 49-52), without
writing a record (the candidate fits run in the notebook, about half an hour):

1. The registration is the pinned 1.0.0; the baseline's features, pairs,
   protocol and transcribed hyperparameters are the accepted ones.
2. The training-side Stage 3 block keeps the rows and the PR 39 fold plan, and
   decay_x_density = joint_economic_decay x the density in every frame.
3. The past-only density reproduces the production engineer on its history,
   never reads a held-out or later label (each with a control), and leaves
   cold-start rows missing.
4. The remaining correlated pair is the registered one, and the candidates'
   pairs are resolved by name.
5. The refitted baseline's fitted terms are the accepted ones, its calibration
   and nested thresholds are isolated by fold, and it matches the PR 39
   records within the stated tolerance (row by row as well where the accepted
   artifact exists locally; with PR42_REQUIRE_ACCEPTED_VIEWS=1 all three views
   must pass, and a present artifact that fails its pin or boundary check
   raises).
6. The past-only candidate's fallback rows carry their fold's fit-side median
   in the EBM frames, their count is the stated one, and flipping later fit
   labels moves that fallback while the first valued day's raw values stay.

Only training rows are bound; accepted artifacts are read only through the
boundary extractors.
"""

from __future__ import annotations

import pytest

from feature_research import encoding_audit as EA
from feature_research import stage3_simplification as SS
from feature_research.training_side import draw_training_side

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def step5a():
    side = draw_training_side(verbose=False)
    X, y = side.X_train, side.y_train
    T = EA.build_time_index(X)
    block = SS.training_block(X, y)
    past = SS.add_past_density(block, X, y, T)
    configs = SS.configurations()
    base = SS.fit_configuration(configs["baseline"], block, SS.frames_for_fits(block, past))
    return dict(side=side, X=X, y=y, T=T, block=block, past=past, configs=configs, base=base)


def test_the_registration_and_the_baseline_sources(step5a):
    checks = SS.registration_checks()
    assert checks["passed"].all(), checks.to_string()


def test_the_block_and_the_past_only_density(step5a):
    s = step5a
    b = SS.block_checks(s["block"], s["X"], s["y"])
    assert b["passed"].all(), b.to_string()
    p = SS.past_density_checks(s["block"], s["X"], s["y"], s["T"], s["past"])
    assert p["passed"].all(), p.to_string()
    cx = s["past"]["contexts"].set_index("context")
    assert cx.loc["all rows", "fit_missing"] == 917                       # Step 3B's lag-0 cold start


def test_the_correlated_pair_and_the_candidates(step5a):
    s = step5a
    c = SS.correlation_checks(SS.correlations(s["block"], s["past"]["X_all"]))
    assert c["passed"].all(), c.to_string()
    assert SS.configuration_checks(s["configs"])["passed"].all()


def test_the_refitted_baseline_matches_the_accepted_records(step5a):
    s = step5a
    base = s["base"]
    assert base["terms_ok"]
    iso = SS.isolation_checks(base, s["block"])
    assert iso["passed"].all(), iso.to_string()
    acc = SS.accepted_figures()
    assert base["threshold_raw"] == acc["threshold_raw"]
    assert abs(base["calibration"]["ece_before"] - acc["ece_before"]) < 0.002
    assert abs(base["calibration"]["ece_after"] - acc["ece_after"]) < 0.002
    assert max(abs(a - float(b)) for a, b in zip(acc["nested_by_fold"], base["nested"]["by_fold"])) <= 0.03
    views = SS.accepted_views(s["side"].train_reference, require=SS.views_required(), y=s["y"].to_numpy())
    if views["available"]:
        par = SS.parity(base, acc, views).set_index("quantity")["refit"]
        assert par["raw out-of-fold score: Pearson correlation (row-aligned)"] > 0.99


def test_the_past_only_fallback_on_the_real_rows(step5a):
    s = step5a
    tr = SS.fallback_trace(s["block"], s["X"], s["y"], s["T"], s["past"])
    assert SS.fallback_checks(tr, s["past"])["passed"].all()
    fb = SS.STATED["fallback"]
    assert tr["totals"]["held_out_fallback"] == fb["held_out_fallback"]
    assert tr["totals"]["held_out_raw"] == fb["held_out_raw"]
    assert tr["sensitivity"]["fallback_moved"].all() and tr["sensitivity"]["control_raw_values_unchanged"].all()
