"""
PR 39 Step 4A: A4.4 across the two registered split-fingerprint schemes
(``upstream_regeneration.checks.stage3_split_identity`` and ``stage3_checks``).

The arms are stamped exactly as the notebooks stamp them, with the real scheme
functions (no stand-ins): the GLASS EBM by GLASS Cell 17,
``shared.split_fingerprint(X_train.index, X_test.index, y_train, y_test)``
(12 hex, rows and labels), the XGBoost by black-box Cell 30,
``blackbox_pipeline.models.rf_router.split_fingerprint(X_train, X_test)``
(16 hex, rows only).

* The raw comparison refuses that pair exactly as the first ``finalize`` of
  run c4_20261115_204125 did; ``shared.stage3.protocol`` is not changed.
* A4.4 passes on it, records both serialized values and their schemes, and
  never rewrites an artifact.
* A real difference fails both the identity check and the protocol check:
  another split, other rows or another row order behind a correct stamp,
  relabelled training or test rows (which the router scheme cannot see),
  missing labels, an unregistered fingerprint, a split that is not the
  frozen one.
* Every other shared-protocol field stays blocking under the exemption.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from blackbox_pipeline.models.rf_router import split_fingerprint as blackbox_cell_30_fp
from glass_pipeline.glass_router.pipeline import fold_fingerprint
from shared import split_fingerprint as glass_cell_17_fp
from shared.stage3.feature_engineering import EBM_FEATURES
from shared.stage3.protocol import assert_matched_protocol
from shared.stage4.contract import split_reference
from tests.support.stage4_world import build_world, make_stage3
from upstream_regeneration import checks as C


def _stamps(GS) -> dict:
    """The two notebooks' Stage 3 stamps, as their cells compute them."""
    return {"ebm": glass_cell_17_fp(GS["X_train"].index, GS["X_test"].index,
                                    GS["y_train"], GS["y_test"]),
            "xgboost": blackbox_cell_30_fp(GS["X_train"], GS["X_test"])}


@pytest.fixture(scope="module")
def world():
    w = build_world(n=1500, n_test=350)
    stamps = _stamps(w["GS"])
    plan = make_stage3(w)
    population = {"stage_io_split_fingerprint": stamps["ebm"],
                  "router_split_fingerprint": stamps["xgboost"],
                  "fold_fingerprint": fold_fingerprint(plan.oof["fold_id"])}
    return w, stamps, population


def _arm(w, stamps, family):
    a = make_stage3(w, family=family)
    a.features = list(EBM_FEATURES)
    a.tuning = {"tuned": True, "n_trials_requested": 3, "random_state": 42,
                "feature_fit_scope": "per_fold"}
    a.split_fingerprint = stamps[family]
    return a


def _arms(world):
    w, stamps, _ = world
    return {"stage3.ebm": _arm(w, stamps, "ebm"), "stage3.xgb": _arm(w, stamps, "xgboost")}


def _a44(world, arts, population=None, split=None):
    w, _, pop = world
    gate = C.Gate("A4.4")
    out = C.stage3_checks(gate, arts, population or pop, split=split or w["GS"])
    return gate, out


# ----------------------------------------------------------------------
# the two registered representations
# ----------------------------------------------------------------------
def test_the_notebooks_stamp_one_split_under_two_registered_schemes(world):
    w, stamps, _ = world
    assert len(stamps["ebm"]) == 12 and len(stamps["xgboost"]) == 16
    ref = split_reference(w["GS"])                       # the real scheme registry, as in production
    assert ref.scheme_of(stamps["ebm"]) == "stage_io"
    assert ref.scheme_of(stamps["xgboost"]) == "router"


def test_the_raw_comparison_refuses_them_as_the_first_finalize_did(world):
    arts = _arms(world)
    _, stamps, _ = world
    with pytest.raises(AssertionError) as err:
        assert_matched_protocol(arts["stage3.ebm"], arts["stage3.xgb"], verbose=False)
    # one blocking difference, the split fingerprint, in the format the failed record showed
    assert str(err.value) == ("Stage 3 arms did not run the same protocol:\n"
                              f"  split_fingerprint: {stamps['ebm']!r} vs {stamps['xgboost']!r}")


def test_a44_passes_on_the_two_schemes_and_keeps_the_serialized_values(world):
    arts = _arms(world)
    _, stamps, pop = world
    before = {r: pickle.dumps(a) for r, a in arts.items()}
    gate, out = _a44(world, arts)
    assert gate.passed, gate.failed()
    assert out["split_identity"] == {
        "stage3.ebm": {"recorded": stamps["ebm"], "scheme": "stage_io",
                       "canonical_stage_io": pop["stage_io_split_fingerprint"]},
        "stage3.xgb": {"recorded": stamps["xgboost"], "scheme": "router",
                       "canonical_stage_io": pop["stage_io_split_fingerprint"]}}
    assert out["protocol_allowed"] == ["search_space", "split_fingerprint"]
    assert ["split_fingerprint", stamps["ebm"], stamps["xgboost"]] in out["protocol_differences"]
    assert out["stage3.ebm"]["split_fingerprint"] == stamps["ebm"]          # reported as stored
    assert {r: pickle.dumps(a) for r, a in arts.items()} == before          # nothing rewritten
    names = [c["check"] for c in gate.checks]
    assert names.index(C.STAGE3_SPLIT_IDENTITY) < names.index(C.STAGE3_PROTOCOL)


@pytest.mark.parametrize("family", ["ebm", "xgboost"])
def test_one_scheme_on_both_sides_needs_no_exemption(world, family):
    w, stamps, _ = world
    arts = _arms(world)
    for a in arts.values():
        a.split_fingerprint = stamps[family]
    gate, out = _a44(world, arts)
    assert gate.passed, gate.failed()
    assert out["protocol_allowed"] == ["search_space"]
    assert not [d for d in out["protocol_differences"] if d[0] == "split_fingerprint"]


# ----------------------------------------------------------------------
# real differences behind the stamps
# ----------------------------------------------------------------------
def _other_split(GS):
    """GLOBAL_SPLIT with one training row and one test row exchanged."""
    tr, te = GS["X_train"].index.tolist(), GS["X_test"].index.tolist()
    tr[0], te[0] = te[0], tr[0]
    y = pd.concat([GS["y_train"], GS["y_test"]])
    X = pd.concat([GS["X_train"], GS["X_test"]])
    return {"X_train": X.loc[tr], "X_test": X.loc[te], "y_train": y.loc[tr], "y_test": y.loc[te]}


def _flip(values, i=0):
    v = list(values)
    v[i] = 1 - int(v[i])
    return v


def _other_stamp(role):
    def edit(a, GS):
        a.split_fingerprint = _stamps(_other_split(GS))["ebm" if role == "stage3.ebm" else "xgboost"]
    return edit


def _records_of(split_fn):
    def edit(a, GS):
        o = split_fn(GS)
        a.index_train, a.index_test = list(o["X_train"].index), list(o["X_test"].index)
        a.y_train = [int(v) for v in o["y_train"]]
        a.y_test = [int(v) for v in o["y_test"]]
    return edit


def _reordered_test(GS):
    te = GS["X_test"].index[::-1]
    return {"X_train": GS["X_train"], "X_test": GS["X_test"].loc[te],
            "y_train": GS["y_train"], "y_test": GS["y_test"].loc[te]}


@pytest.mark.parametrize("role, edit, why", [
    ("stage3.xgb", _other_stamp("stage3.xgb"), "no fingerprint scheme in common"),
    ("stage3.ebm", _other_stamp("stage3.ebm"), "no fingerprint scheme in common"),
    ("stage3.xgb", _records_of(_other_split), "recorded training rows are not GLOBAL_SPLIT's"),
    ("stage3.xgb", _records_of(_reordered_test), "recorded test rows are not GLOBAL_SPLIT's"),
    # the router scheme hashes rows only: relabelled rows keep a correct router stamp (F12)
    ("stage3.xgb", lambda a, GS: setattr(a, "y_train", _flip(a.y_train)),
     "recorded training labels are not GLOBAL_SPLIT's (1 differ)"),
    ("stage3.ebm", lambda a, GS: setattr(a, "y_test", _flip(a.y_test, 3)),
     "recorded test labels are not GLOBAL_SPLIT's (1 differ)"),
    ("stage3.xgb", lambda a, GS: setattr(a, "y_test", []),
     "recorded test labels (0) do not cover its recorded test rows (350)"),
    ("stage3.ebm", lambda a, GS: setattr(a, "split_fingerprint", "0" * 12),
     "no fingerprint scheme in common"),
    ("stage3.xgb", lambda a, GS: setattr(a, "split_fingerprint", None),
     "no fingerprint scheme in common"),
])
def test_a_real_split_difference_fails_identity_and_protocol(world, role, edit, why):
    w = world[0]
    arts = _arms(world)
    edit(arts[role], w["GS"])
    gate, out = _a44(world, arts)
    failed = gate.failed()
    assert C.STAGE3_SPLIT_IDENTITY in failed and C.STAGE3_PROTOCOL in failed, failed
    detail = next(c["detail"] for c in gate.checks if c["check"] == C.STAGE3_SPLIT_IDENTITY)
    assert why in detail, detail
    assert out["split_identity"] == {} and out["protocol_allowed"] == ["search_space"]


def test_a_split_that_is_not_the_frozen_one_fails(world):
    w, stamps, pop = world
    gate, _ = _a44(world, _arms(world), population={**pop, "stage_io_split_fingerprint": "f" * 12})
    assert C.STAGE3_SPLIT_IDENTITY in gate.failed()
    detail = next(c["detail"] for c in gate.checks if c["check"] == C.STAGE3_SPLIT_IDENTITY)
    assert "is not the frozen protocol's split" in detail


def test_both_arms_on_one_other_split_are_still_refused(world):
    """Consistent with each other is not enough: the split must be the frozen one."""
    w, _, pop = world
    other = _other_split(w["GS"])
    arts = _arms(world)
    for role, family in (("stage3.ebm", "ebm"), ("stage3.xgb", "xgboost")):
        _records_of(lambda GS: other)(arts[role], w["GS"])
        arts[role].split_fingerprint = _stamps(other)[family]
    gate, _ = _a44(world, arts, split=w["GS"])
    assert C.STAGE3_SPLIT_IDENTITY in gate.failed() and C.STAGE3_PROTOCOL in gate.failed()


# ----------------------------------------------------------------------
# every other shared-protocol requirement is kept
# ----------------------------------------------------------------------
@pytest.mark.parametrize("field, edit", [
    ("fold_id", lambda a: a.fold_plan.update(fold_id=list(np.roll(a.fold_plan["fold_id"], 1)))),
    ("n_eval_folds", lambda a: a.config.update(n_eval_folds=7)),
    ("class_weights", lambda a: setattr(a, "class_weights", {0: 1.0, 1: 9.0})),
    ("tuning.n_trials_requested", lambda a: a.tuning.update(n_trials_requested=4)),
    ("population", lambda a: setattr(a, "population", "oof_remainder")),
    ("feature_fit_scope", lambda a: setattr(a, "feature_fit_scope", "global")),
    ("index_test", lambda a: setattr(a, "index_test", list(a.index_test)[::-1])),
])
def test_the_exemption_covers_the_split_fingerprint_only(world, field, edit):
    arts = _arms(world)
    edit(arts["stage3.xgb"])
    gate, _ = _a44(world, arts)
    assert C.STAGE3_PROTOCOL in gate.failed(), gate.failed()
    detail = next(c["detail"] for c in gate.checks if c["check"] == C.STAGE3_PROTOCOL)
    assert f"  {field}: " in detail, detail
    if field != "index_test":                       # the split itself is intact otherwise
        assert C.STAGE3_SPLIT_IDENTITY not in gate.failed()
        assert "split_fingerprint" not in detail, detail


def test_the_exemption_is_withheld_when_the_identity_check_cannot_run(world):
    """A split that cannot be read (no labels in GLOBAL_SPLIT) proves nothing: no exemption."""
    w = world[0]
    broken = {k: v for k, v in w["GS"].items() if k != "y_test"}
    gate, out = _a44(world, _arms(world), split=broken)
    assert {C.STAGE3_SPLIT_IDENTITY, C.STAGE3_PROTOCOL} <= set(gate.failed())
    assert out["protocol_allowed"] == ["search_space"]
