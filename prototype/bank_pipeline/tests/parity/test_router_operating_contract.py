"""
F17 — the two routers' operating contracts, side by side (PR 36, Commit 3 Step 3C).

Shared by both routers (pinned here)
    three states        "pass1" (NOT_SUBSCRIBE) / "pass2" (SUBSCRIBE) / "uncertain"
                        (abstain, confidence 0); Pass 1 claims first — a row
                        Pass 1 routes is never offered to Pass 2.
    one probability     Pass 1 b(1 - conf), Pass 2 b + (1 - b) conf, abstain b,
    transform           b = the producing router's training base rate. ``conf``
                        is the matched unit's precision in its route: a GLASS
                        rule's (in sample), an RF band's (Pass 1 NPV / Pass 2
                        precision, on training OOF). Reading the unit's
                        positive rate back is therefore exact for both — the
                        basis of the GLASS arbiter's router evidence (F25) in
                        the 2x2 crossover (F29; test_stage4_crossover.py).

Different by design — the F17 question
    Pass 2 population   RF Router: always the OOF Pass 1 remainder
                        (``RFRouter.fit``: remainder = OOF p1 >= t1).
                        GLASS Router: ``config.pass2_population`` — the
                        notebook's active "legacy" profile is "full" (Pass 2
                        rules selected and their precision measured on every
                        training row, applied to what Pass 1 leaves); the
                        "recommended" profile is "oof_remainder", the RF
                        construction. Unchanged here: switching the GLASS
                        profile is a methodology decision (it changes the
                        GLASS rules and needs Stage 2 regenerated).
    Recorded            Stage 4 records each router's Pass 2 population for
                        its cross-fit (train outputs) and full-train router
                        (test outputs) and refuses an artifact whose two
                        disagree; the arbiters' artifacts carry it, so a
                        comparison cannot hide the difference. A missing
                        declaration is refused, with or without F12 label
                        fingerprints (Commit 4; PARTIAL evidence in Commit 3).

Covered elsewhere and not repeated: the GLASS oof_remainder fit
(tests/stage2/test_glass_router_pass2_remainder.py), the RF remainder and
its OOF discipline (tests/stage2/test_rf_router_oof.py), unknown route labels
at Stage 4 (tests/stage4, tests/cascade).
"""

from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from blackbox_pipeline.models.rf_router import RFRouter
from blackbox_pipeline.models.rf_router.router.base import apply_cuts
from glass_pipeline.meta_ebm.loader import load_stage_outputs
from glass_pipeline.meta_ebm.router_evidence import rule_positive_rate
import shared.stage4.adapters as adapters
from shared.stage4 import build_stage4_inputs, router_stream
from shared.validation import ValidationError
from tests.stage2.conftest import legacy_config, new_router, router_with_rules, run_crossfit, selected_rule
from tests.support.notebooks import BANK_ROOT
from tests.support.rf_router_world import bank_world, quiet, router_config, save_world
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3
from tests.support.synthetic import make_bank_frame, split_frame

RF_REMAINDER = "OOF Pass 1 remainder (RF router design)"       # crossfit_rf_router's protocol


# ----------------------------------------------------------------------
# The shared three-state contract
# ----------------------------------------------------------------------
def test_both_routers_route_with_pass1_precedence_into_three_states():
    X = pd.DataFrame({"f": [1, 0, 1, 0], "g": [1, 1, 0, 0]}, index=[11, 12, 13, 14])
    glass = router_with_rules([selected_rule("P1", [("f", 1)], "pass1", 0.95)],
                              [selected_rule("P2", [("g", 1)], "pass2", 0.40)], base_rate=0.2)
    pred, conf, dec = glass.predict(X)
    # row 11 matches a Pass 1 AND a Pass 2 rule: Pass 1 claims it
    assert dec.tolist() == ["pass1", "pass2", "pass1", "uncertain"]
    assert pred.tolist() == [0, 1, 0, -1] and conf.tolist() == [0.95, 0.40, 0.95, 0.0]
    p1 = np.array([0.1, 0.9, 0.1, 0.9])                   # row 0 also clears the Pass 2 cut
    p2 = np.array([0.9, 0.9, np.nan, 0.1])                 # NaN: Pass 2 never scored it
    rf_pred, rf_dec = apply_cuts(p1, p2, t1=0.5, t2=0.5)
    assert rf_dec.tolist() == dec.tolist() and rf_pred.tolist() == pred.tolist()


def test_both_routers_turn_route_and_confidence_into_probability_with_one_transform():
    dec = np.array(["pass1", "pass2", "uncertain", "pass1", "pass2"], dtype=object)
    conf = np.array([0.95, 0.40, 0.0, 0.90, 0.55])
    b = 0.17
    X = pd.DataFrame({"r": [0, 1, 2, 3, 4]})
    glass = router_with_rules([selected_rule("P1a", [("r", 0)], "pass1", 0.95),
                               selected_rule("P1b", [("r", 3)], "pass1", 0.90)],
                              [selected_rule("P2a", [("r", 1)], "pass2", 0.40),
                               selected_rule("P2b", [("r", 4)], "pass2", 0.55)], base_rate=b)
    p_glass = glass.predict_proba(X)
    p_rf = RFRouter(router_config())._proba_from_triple(conf, dec, base_rate=b)
    np.testing.assert_allclose(p_glass, p_rf, rtol=0, atol=1e-15)
    # the matched unit's positive rate reads back exactly from either
    r = rule_positive_rate(p_rf[:, 1], dec == "pass1", dec == "pass2", b)
    np.testing.assert_allclose(r[dec == "pass1"], 1 - conf[dec == "pass1"], atol=1e-12)
    np.testing.assert_allclose(r[dec == "pass2"], conf[dec == "pass2"], atol=1e-12)
    assert p_rf[2, 1] == b                                  # abstain: the base rate, no vote


def test_the_rf_routers_saved_confidence_is_its_band_precision():
    bw = bank_world()
    art = joblib.load(save_world(bw, tempfile.mkdtemp()))
    bands, e = art["bands"], art["extra"]
    dec, conf = np.asarray(e["test_decisions"]).astype(str), np.asarray(e["test_confidence"], float)
    assert set(np.unique(conf[dec == "pass1"])) == {bands["pass1_confidence"]}
    assert set(np.unique(conf[dec == "pass2"])) == {bands["pass2_confidence"]}
    assert (conf[dec == "uncertain"] == 0).all()


# ----------------------------------------------------------------------
# Pass 2 population: declared, recorded, consistent
# ----------------------------------------------------------------------
def _stub_rule_search(router):
    """Rule generation, evaluation and the ILP replaced; the fit's bookkeeping is real."""
    router.rule_generator = SimpleNamespace(generate_candidates=lambda X, y: [])
    router.rule_evaluator = SimpleNamespace(evaluate_candidates=lambda c, X, y: [])
    router.ilp_selector = SimpleNamespace(select_rules=lambda **k: {"pass1_rules": [], "pass2_rules": []})
    return router


def test_each_router_records_the_pass2_population_it_was_fitted_on():
    X = pd.DataFrame({"f": np.r_[np.zeros(30), np.ones(30)].astype(int)})
    y = pd.Series(np.r_[np.zeros(45), np.ones(15)].astype(int))
    glass = quiet(_stub_rule_search(new_router(legacy_config(pass2_population="full"))).fit, X, y)
    assert glass.fit_report_ == {"pass2_population": "full"}
    rf = bank_world()
    assert rf.router.fit_report_.remainder["source"] == "OOF pass1 probabilities >= t1"
    assert rf.crossfit.protocol["pass2_population"] == RF_REMAINDER


@pytest.mark.parametrize("population", ["full", "oof_remainder"])
def test_the_glass_cross_fit_records_the_configured_pass2_population(population):
    split = split_frame(make_bank_frame(n=400))
    run = run_crossfit(split, config=legacy_config(pass2_population=population))
    assert run.oof.protocol["glass_pass2_population"] == population


def _glass_notebook_profiles():
    cells = json.loads((BANK_ROOT / "glass_pipeline" / "glass_cascade.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in cells["cells"] if c["cell_type"] == "code")
    return src


def test_the_glass_notebook_runs_the_full_population_profile_for_both_fits():
    """Pins the current choice (F17): changing it is a methodology decision."""
    src = _glass_notebook_profiles()
    assert 'CONFIG_PROFILE = "legacy"' in src
    legacy = src[src.index('"legacy": dict('):src.index('"recommended": dict(')]
    recommended = src[src.index('"recommended": dict('):]
    assert 'pass2_population="full"' in legacy
    assert 'pass2_population="oof_remainder"' in recommended
    # one config object drives the full-train fit and the cross-fit
    assert "config = GlassRouterConfig(**_SHARED, **_PROFILES[CONFIG_PROFILE])" in src
    assert "GlassRouterPipeline(\n    config=config," in src and "glass_config=config," in src


@pytest.fixture
def world():
    return build_world(n=1500, n_test=350)


def _router_notes(art, w):
    stream, checks = router_stream(art, "router", "Router", w["GS"]["y_train"].to_numpy())
    inp = build_stage4_inputs(w["GS"], [stream], checks, verbose=False, extra_fingerprints=ROUTER_FP)
    return inp["router"].notes


def test_stage4_records_each_arms_pass2_population(world, tmp_path):
    glass = _router_notes(make_router(world, pass2_population="full"), world)
    rf = _router_notes(make_router(world, layout="extra", band_blend=True, pass2_population=RF_REMAINDER),
                       world)
    assert glass["pass2_population"] == {"cross_fit": "full", "full_train": "full"}
    assert rf["pass2_population"] == {"cross_fit": "oof_remainder", "full_train": "oof_remainder"}
    bw = bank_world()                                        # the real RF producers
    art = joblib.load(save_world(bw, tmp_path))
    stream, checks = router_stream(art, "rf_router", "RF", bw.global_split["y_train"].to_numpy())
    real = build_stage4_inputs(bw.global_split, [stream], checks, verbose=False)
    assert real["rf_router"].notes["pass2_population"] == {"cross_fit": "oof_remainder",
                                                           "full_train": "oof_remainder"}
    # the GLASS full-train population falls back to the saved config
    from_config = make_router(world, pass2_population="full")
    del from_config["fit_report"]                       # only the saved config declares it
    from_config["config_full"] = {"pass2_population": "full"}
    assert _router_notes(from_config, world)["pass2_population"] == {"cross_fit": "full",
                                                                     "full_train": "full"}


@pytest.mark.parametrize("case", ["glass cross-fit oof_remainder, full-train full",
                                  "glass cross-fit full, full-train oof_remainder",
                                  "rf full-train single forest"])
def test_train_and_test_outputs_from_different_pass2_populations_are_refused(world, case):
    if case.startswith("glass"):
        art = make_router(world, pass2_population="full")
        cross, full = ("oof_remainder", "full") if "cross-fit oof" in case else ("full", "oof_remainder")
        art["oof_protocol"]["glass_pass2_population"], art["fit_report"]["pass2_population"] = cross, full
    else:
        art = make_router(world, layout="extra", band_blend=True, pass2_population=RF_REMAINDER)
        art["mode"] = "single_rf"
    with pytest.raises(ValidationError, match="cross-fit and full-train router share one Pass 2 population"):
        _router_notes(art, world)


def test_the_arbiters_record_both_routers_pass2_populations(world):
    """The F17 difference travels with every Stage 4 result."""
    glass = load_stage_outputs(world["GS"], make_stage1(world), make_router(world, pass2_population="full"),
                               make_stage3(world), verbose=False, extra_fingerprints=ROUTER_FP)
    assert glass.upstream["stages"]["stage2"]["pass2_population"] == {"cross_fit": "full",
                                                                       "full_train": "full"}
    assert glass.provenance()["streams"]["glass"]["pass2_population"]["cross_fit"] == "full"


# ----------------------------------------------------------------------
# Incomplete declarations: refused, with or without label fingerprints
# ----------------------------------------------------------------------
LAYOUTS = {"glass": "top-level", "rf": "extra"}


def _router(world, kind, current=True):
    """A router artifact declaring both populations; ``current=False``: written
    before the F12 fingerprints (refused for those too, tests/parity/
    test_router_label_fingerprint.py)."""
    kw = {"pass2_population": "full" if kind == "glass" else RF_REMAINDER, "label_fingerprint": current}
    if kind == "rf":
        kw["band_blend"] = True
    return make_router(world, layout=LAYOUTS[kind], **kw)


def _drop(art, kind, side):
    if kind == "glass" and side == "cross-fit":
        del art["oof_protocol"]["glass_pass2_population"]
    elif kind == "glass":
        del art["fit_report"]                       # and no config_full to fall back on
    elif side == "cross-fit":
        del art["extra"]["oof_protocol"]["pass2_population"]
    else:
        del art["mode"]


@pytest.mark.parametrize("kind", ["glass", "rf"])
def test_matching_declarations_are_checked_and_reported_complete(world, kind):
    notes = _router_notes(_router(world, kind), world)
    assert notes["pass2_population_evidence"] == adapters.PASS2_COMPLETE
    want = "full" if kind == "glass" else "oof_remainder"
    assert notes["pass2_population"] == {"cross_fit": want, "full_train": want}


@pytest.mark.parametrize("current", [True, False], ids=["with label fingerprints", "without"])
@pytest.mark.parametrize("missing", [("cross-fit",), ("full-train",), ("cross-fit", "full-train")],
                         ids=["no cross-fit declaration", "no full-train declaration", "neither"])
@pytest.mark.parametrize("kind", ["glass", "rf"])
def test_an_artifact_missing_a_declaration_is_refused(world, kind, missing, current):
    """Commit 3 refused this only when the artifact also recorded label
    fingerprints and accepted it as PARTIAL otherwise; since Commit 4 it is
    refused either way, naming what is missing."""
    art = _router(world, kind, current)
    for side in missing:
        _drop(art, kind, side)
    with pytest.raises(ValidationError) as exc:
        _router_notes(art, world)
    lines = [ln for ln in str(exc.value).splitlines() if "Pass 2 population declared" in ln]
    assert len(lines) == 1 and "regenerate Stage 2" in lines[0]
    assert all(side in lines[0].split("missing:")[1] for side in missing)


@pytest.mark.parametrize("kind", ["glass", "rf"])
def test_an_unrecognised_declaration_is_refused_even_when_both_sides_agree(world, kind):
    art = _router(world, kind)
    if kind == "glass":
        art["oof_protocol"]["glass_pass2_population"] = art["fit_report"]["pass2_population"] = "custom"
    else:
        art["extra"]["oof_protocol"]["pass2_population"] = "custom"
        art["mode"] = "custom"
    with pytest.raises(ValidationError, match="share one Pass 2 population"):
        _router_notes(art, world)


def test_the_arbiter_refuses_a_router_that_declares_no_pass2_population(world):
    """Through the GLASS arbiter's loader: a fingerprinted router artifact that
    declares neither population used to be recorded as PARTIAL evidence."""
    undeclared = _router(world, "glass")
    for side in ("cross-fit", "full-train"):
        _drop(undeclared, "glass", side)
    with pytest.raises(ValidationError, match="glass: Pass 2 population declared for the cross-fit"):
        load_stage_outputs(world["GS"], make_stage1(world), undeclared, make_stage3(world),
                           verbose=False, extra_fingerprints=ROUTER_FP)
