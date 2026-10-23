"""
F12 — both routers record the training labels they were fitted on
(PR 36, Commit 3; mandatory from Commit 4).

    shared.stage_io.label_fingerprint(index, y)    the fitting rows and their 0/1 labels,
                                                   in fitting order, 12 lowercase hex

    written by                                     where it lands         what it vouches for
    GLASS crossfit_stage2, RF crossfit_rf_router   oof_protocol           train outputs (OOF)
      ["train_label_fingerprint"]
    GlassRouterPipeline.fit, RFRouter.fit,         artifact top level     test outputs
      SingleRFRouter.fit -> .train_label_fingerprint_   (ModelSaver,      (full-train router)
                                                   save_rf_router)

    checked by shared.stage4 (both arms): each recorded value must be well
    formed and equal label_fingerprint(GLOBAL_SPLIT X_train.index, y_train).

Why: a router's fold ids and base rates do not determine its labels
(tests/parity/test_split_fingerprint_parity.py, within-fold swap). With
both fingerprints recorded, a router fitted on swapped labels is refused.

Mandatory since Commit 4: an artifact missing either fingerprint — every
router artifact written before Commit 3 — is refused, naming the missing side;
it was accepted with label evidence PARTIAL in Commit 3. Fingerprints are
never backfilled: the artifact is regenerated. The same refusal through both
loaders, on a router fitted on swapped labels: tests/parity/
test_stage4_split_fold_provenance.py.

End to end through the production savers: the RF Router here
(tests.support.rf_router_world), the GLASS Router in tests/cascade.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

import glass_pipeline.glass_router.pipeline.glass_router_pipeline as glass_mod
import shared.stage4.contract as contract
from blackbox_pipeline.models.meta_xgb import load_blackbox_stage4_inputs
from blackbox_pipeline.models.rf_router import RFRouter, SingleRFRouter
from glass_pipeline.meta_ebm.loader import load_stage_outputs
from shared.stage4 import build_stage4_inputs, router_stream
import shared.stage_io as stage_io
from shared.stage_io import fold_assignment
from shared.validation import ValidationError
from tests.stage2.conftest import new_router, run_crossfit
from tests.support import mutations as mut
from tests.support.rf_router_world import bank_world, quiet, router_config, save_world, serial_forests
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3
from tests.support.synthetic import make_bank_frame, split_frame

ARMS = ("glass", "blackbox")
ROUTER = {"glass": "glass", "blackbox": "rf_router"}
SIDES = ("train outputs", "test outputs")              # = contract.LABEL_FINGERPRINT_SIDES


def label_fingerprint(index, y) -> str:
    """``shared.stage_io.label_fingerprint``, resolved at call time."""
    return stage_io.label_fingerprint(index, y)


def swapped(y: pd.Series) -> pd.Series:
    """Opposite labels swapped within folds: same fold ids, same base rates."""
    return mut.swap_labels_within_folds(y, fold_assignment(y, cv_folds=5, random_state=42))


# ----------------------------------------------------------------------
# The fingerprint
# ----------------------------------------------------------------------
def test_label_fingerprint_identifies_the_rows_and_their_labels_in_order():
    y = pd.Series(np.r_[np.zeros(30, int), np.ones(10, int)][np.random.default_rng(0).permutation(40)],
                  index=pd.Index(np.arange(500, 540) * 3))
    fp = label_fingerprint(y.index, y)
    assert len(fp) == 12 and set(fp) <= set("0123456789abcdef")
    for same in (y.astype("int8"), y.astype(bool), y.astype(float), y.to_numpy()):
        assert label_fingerprint(y.index, same) == fp                  # label dtype does not matter
    flipped = y.copy()
    flipped.iloc[0] = 1 - flipped.iloc[0]
    changed = {
        "one label flipped": (y.index, flipped),
        "labels swapped within folds": (y.index, mut.swap_labels_within_folds(y, np.zeros(40, int))),
        "rows reordered": (y.index[::-1], y.to_numpy()[::-1]),
        "index reset": (pd.RangeIndex(40), y),
        "foreign row": (mut.replace_index_label(y).index, y),
        "row dropped": (y.index[1:], y.iloc[1:]),
    }
    for case, (index, labels) in changed.items():
        assert label_fingerprint(index, labels) != fp, case
    with pytest.raises(ValueError, match="39 labels for 40 rows"):
        label_fingerprint(y.index, y.iloc[1:])


# ----------------------------------------------------------------------
# Producers: each fit records the rows and labels it actually used
# ----------------------------------------------------------------------
def test_the_rf_router_records_its_fits_labels_and_the_saved_artifact_passes_stage4(tmp_path):
    bw = bank_world()                                 # real RFRouter + crossfit_rf_router
    gs = bw.global_split
    want = label_fingerprint(gs["X_train"].index, gs["y_train"])
    assert bw.router.train_label_fingerprint_ == want                   # full-train fit
    assert bw.crossfit.protocol["train_label_fingerprint"] == want      # outer cross-fit
    art = joblib.load(save_world(bw, tmp_path))                         # save_rf_router
    assert art["train_label_fingerprint"] == art["extra"]["oof_protocol"]["train_label_fingerprint"] == want
    stream, checks = router_stream(art, "rf_router", "RF Router", gs["y_train"].to_numpy())
    inp = build_stage4_inputs(gs, [stream], checks, verbose=False)      # real fingerprint schemes
    assert inp["rf_router"].notes["label_evidence"] == contract.LABELS_FINGERPRINTED
    assert {r["check"] for r in inp.validation if "rows and labels" in r["check"]} == {
        f"rf_router: {side} fitted on GLOBAL_SPLIT's training rows and labels" for side in SIDES}


@pytest.mark.parametrize("cls, mode", [(RFRouter, "two_pass"), (SingleRFRouter, "single_rf")])
def test_an_rf_router_fit_fingerprints_the_labels_it_was_given(cls, mode):
    s2 = bank_world().stage2
    X, y = s2["X_train"], s2["y_train"]
    assert cls(router_config(mode=mode)).train_label_fingerprint_ is None          # unfitted
    for labels in (y, swapped(y)):
        with serial_forests():
            r = quiet(cls(router_config(mode=mode)).fit, X, labels, s2["split_fingerprint"])
        assert r.train_label_fingerprint_ == label_fingerprint(X.index, labels)
    assert label_fingerprint(X.index, swapped(y)) != label_fingerprint(X.index, y)


@pytest.fixture(scope="module")
def glass_split():
    return split_frame(make_bank_frame(n=400))


def test_the_glass_cross_fit_records_the_labels_its_folds_were_fitted_on(glass_split):
    y = glass_split["y_train"]
    for labels in (y, swapped(y)):
        protocol = run_crossfit(glass_split, y_train=labels).oof.protocol
        assert protocol["train_label_fingerprint"] == label_fingerprint(labels.index, labels)


def test_a_glass_router_fit_fingerprints_the_rows_and_labels_it_was_given(glass_split, monkeypatch):
    for step in ("_fit_full", "_fit_oof_remainder"):                     # no rule search
        monkeypatch.setattr(glass_mod.GlassRouterPipeline, step, lambda self, *a: None)
    X, y = glass_split["X_train"], glass_split["y_train"]
    router = new_router()
    assert router.train_label_fingerprint_ is None
    for labels in (y, swapped(y)):
        assert quiet(new_router().fit, X, labels).train_label_fingerprint_ == \
            label_fingerprint(X.index, labels)
    # labels without an index are fingerprinted against X's rows
    assert quiet(new_router().fit, X, y.to_numpy()).train_label_fingerprint_ == label_fingerprint(X.index, y)


# ----------------------------------------------------------------------
# Stage 4, both arms
# ----------------------------------------------------------------------
@pytest.fixture
def world():
    return build_world(n=1500, n_test=350)


@pytest.fixture(params=ARMS)
def arm(request):
    return request.param


def load(arm, w, router):
    if arm == "glass":
        return load_stage_outputs(w["GS"], make_stage1(w), router, make_stage3(w), verbose=False,
                                  extra_fingerprints=ROUTER_FP)
    return load_blackbox_stage4_inputs(w["GS"], make_stage1(w, arm="blackbox", model="mlp"), router,
                                       make_stage3(w, family="xgboost"), verbose=False,
                                       extra_fingerprints=ROUTER_FP)


def router_art(arm, w, **kw):
    return make_router(w, layout="top-level" if arm == "glass" else "extra", **kw)


def set_label_fp(arm, art, side, value):
    """Put ``value`` where the producers write ``side``'s fingerprint (None: remove it)."""
    block = art if arm == "glass" else art["extra"]
    target = block["oof_protocol"] if side == "train outputs" else art
    if value is None:
        target.pop("train_label_fingerprint", None)
    else:
        target["train_label_fingerprint"] = value


def failures(exc) -> list:
    return [line.strip() for line in str(exc.value).splitlines()[1:] if line.strip().startswith("[")]


def test_a_router_recording_the_splits_labels_is_accepted_with_complete_evidence(world, arm):
    inp = load(arm, world, router_art(arm, world, label_fingerprint=True))
    assert inp[ROUTER[arm]].notes["label_evidence"] == contract.LABELS_FINGERPRINTED
    assert contract.LABEL_FINGERPRINT_SIDES == SIDES


@pytest.mark.parametrize("sides", [SIDES[:1], SIDES[1:], SIDES], ids=["train side", "test side", "both"])
def test_a_router_fitted_on_within_fold_swapped_labels_is_refused(world, arm, sides):
    """The F12 counterexample, now refused wherever the fingerprint is recorded."""
    GS = world["GS"]
    other = swapped(GS["y_train"])
    art = router_art(arm, world, label_fingerprint=True)
    for side in sides:
        set_label_fp(arm, art, side, label_fingerprint(other.index, other))
    with pytest.raises(ValidationError) as exc:
        load(arm, world, art)
    lines = failures(exc)
    assert [line.split(" — ")[0] for line in lines] == [
        f"[stage2] {ROUTER[arm]}: {side} fitted on GLOBAL_SPLIT's training rows and labels"
        for side in sides]                              # nothing else in the cascade notices
    assert all("fitted on other training rows or labels" in line for line in lines)


def test_a_router_built_end_to_end_on_swapped_labels_is_refused(world, arm):
    """The whole stand-in router is produced on the swapped labels, as a real fit would be."""
    GS = world["GS"]
    relabelled = {**world, "GS": {**GS, "y_train": swapped(GS["y_train"])}}
    with pytest.raises(ValidationError) as exc:
        load(arm, world, router_art(arm, relabelled, label_fingerprint=True))
    assert len(failures(exc)) == 2 and all("fitted on other training" in f for f in failures(exc))


@pytest.mark.parametrize("value", [
    "", "abc", "0123456789AB", "0123456789abc", " 123456789ab", "0123456789ag",
    123456789012, b"0123456789ab", ["0123456789ab"], float("nan"),
], ids=["empty", "short", "uppercase", "13 chars", "leading space", "not hex",
        "int", "bytes", "list", "nan"])
@pytest.mark.parametrize("side", SIDES)
def test_a_malformed_label_fingerprint_is_refused_not_ignored(world, arm, side, value):
    art = router_art(arm, world, label_fingerprint=True)
    set_label_fp(arm, art, side, value)
    with pytest.raises(ValidationError) as exc:
        load(arm, world, art)
    assert [f.split(" — ")[0] for f in failures(exc)] == [
        f"[stage2] {ROUTER[arm]}: {side} fitted on GLOBAL_SPLIT's training rows and labels"]
    assert "malformed label fingerprint" in failures(exc)[0]


def test_the_fingerprint_check_reads_the_training_labels_only(world, arm):
    """Stage 4 recomputes the expected fingerprint from GLOBAL_SPLIT's training
    rows and labels: moving the held-out labels changes nothing (no test label
    builds or validates it); moving the training labels is refused."""
    GS = world["GS"]
    art = router_art(arm, world)

    def build(split):
        stream, checks = router_stream(art, ROUTER[arm], "Router", split["y_train"].to_numpy())
        return build_stage4_inputs(split, [stream], checks, verbose=False, extra_fingerprints=ROUTER_FP)

    moved_test = {**GS, "y_test": mut.permute_labels(GS["y_test"], seed=5)}
    assert not moved_test["y_test"].equals(GS["y_test"])
    assert build(moved_test)[ROUTER[arm]].notes["label_evidence"] == contract.LABELS_FINGERPRINTED
    with pytest.raises(ValidationError, match="fitted on other training rows or labels"):
        build({**GS, "y_train": swapped(GS["y_train"])})


@pytest.mark.parametrize("form", ["absent", "None"])
@pytest.mark.parametrize("missing", [SIDES[:1], SIDES[1:], SIDES], ids=["train side", "test side", "both"])
def test_a_router_missing_a_label_fingerprint_is_refused(world, arm, missing, form):
    """Commit 3 accepted these as PARTIAL label evidence (an artifact from before
    the fingerprint), checking any side that was recorded. Each missing side
    is now refused on its own; a recorded side is still checked, and nothing
    else in the cascade fails."""
    art = router_art(arm, world, label_fingerprint=True)
    for side in missing:
        set_label_fp(arm, art, side, None)
        if form == "None":                                  # recorded as None = not recorded
            block = art if arm == "glass" else art["extra"]
            (block["oof_protocol"] if side == "train outputs" else art)["train_label_fingerprint"] = None
    with pytest.raises(ValidationError) as exc:
        load(arm, world, art)
    lines = failures(exc)
    assert [line.split(" — ")[0] for line in lines] == [
        f"[stage2] {ROUTER[arm]}: {side} fitted on GLOBAL_SPLIT's training rows and labels"
        for side in missing]
    assert all("no label fingerprint recorded" in line and "never backfilled" in line for line in lines)
