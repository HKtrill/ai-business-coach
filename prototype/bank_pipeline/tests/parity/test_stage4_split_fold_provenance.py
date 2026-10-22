"""
F11 / F12 — what Stage 4 accepts as the same split, rows and fold plan, in
both arms (PR 36, Commit 3 Step 3B).

Both arms' Stage 4 loaders (GLASS ``load_stage_outputs``, black-box
``load_blackbox_stage4_inputs``) run the shared contract
(``shared.stage4.build_stage4_inputs``) over the real Stage 1-3 artifact
layouts of ``tests.support.stage4_world``.

Fold plan (F11)
    Every stream's fold ids must be the shared stratified splitter's plan on
    GLOBAL_SPLIT's training labels, in GLOBAL_SPLIT row order, at the
    stream's declared seed; Stages 2 and 3 must hold one partition; Stage 1's
    10 folds stay distinct by design.
    Before: fold ids were only checked pairwise between streams with the same
    fold COUNT, so a Stage 2 cross-fit on Stage 1's 10-fold partition passed
    as "distinct by design", and any fold plan passed when no stream with
    its fold count disagreed.
Label evidence (F12 — PARTIAL)
    A router artifact carries no labels and a row-only split fingerprint, so
    a router cross-fitted on the same rows with other labels passed the split
    check. Its fold ids and a recorded ``training_base_rate`` are now checked:
    a flipped label or a permutation reassigns most fold ids. They are not
    sufficient: opposite labels swapped within one fold change no fold id and
    no base rate. Since Commit 3 both routers record ``train_label_fingerprint``
    for their cross-fit and full-train fits, and Stage 4 checks it when present
    (tests/parity/test_router_label_fingerprint.py). Router artifacts written
    before that carry none and are still accepted, PARTIAL; every stream's
    ``notes["label_evidence"]`` says which case it is. Commit 4 makes the
    fingerprint mandatory and regenerates the router artifacts; the test
    pinning the transitional acceptance is marked ``f12_partial``.
Split-fingerprint consistency (F12)
    A router artifact may record its split fingerprint in several places
    (top level, the cross-fit's ``oof_protocol``, the RF Stage 4 block);
    Stage 4 verified one and ignored the rest. They must now agree.
Row identity (F12)
    A reset index (``0..n-1``) or no index used to pass as a "positional"
    contract, so rows reordered before the reset — here within one fold and
    one label, where no label or fold check can see it — were consumed
    misaligned. Stage 4 now requires the split's row labels.

Each case is built so that it breaks only the contract under test (GLASS
router probabilities are re-blended at the new fold base rates where fold
ids change); the full ValidationError still lists every failure.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.model_selection import KFold

from blackbox_pipeline.models.meta_xgb import load_blackbox_stage4_inputs
from glass_pipeline.meta_ebm.loader import load_stage_outputs
import shared.stage4.contract as contract
from shared.stage4 import build_stage4_inputs, router_stream, stage_output_stream
from shared.stage_io import fold_assignment
from shared.validation import ValidationError
from tests.support import mutations as mut
from tests.support import stage4_world
from tests.support.stage4_world import (
    ROUTER_FP, build_world, fold_fp, make_router, make_stage1, make_stage3,
)

ARMS = ("glass", "blackbox")
NAMES = {"glass": ("lr", "glass", "ebm"), "blackbox": ("mlp", "rf_router", "xgb")}
INDEX_KEY = {"glass": {"train": "train_idx", "test": "index_test"},
             "blackbox": {"train": "index_train", "test": "index_test"}}
ROUTER_ROWS = {"train": ("train_proba", "train_decisions", "train_pred", "train_confidence"),
               "test": ("test_proba", "test_decisions", "test_pred", "test_confidence")}
FOLD_CHECK = "fold ids = shared stratified splitter on GLOBAL_SPLIT labels"
SHARED_CHECK = "stages 2-3 share one partition"


@pytest.fixture
def world():
    return build_world(n=1500, n_test=350)


@pytest.fixture(params=ARMS)
def arm(request):
    return request.param


def artifacts(w, arm) -> dict:
    if arm == "glass":
        return {"s1": make_stage1(w), "s2": make_router(w), "s3": make_stage3(w)}
    return {"s1": make_stage1(w, arm="blackbox", model="mlp"), "s2": make_router(w, layout="extra"),
            "s3": make_stage3(w, family="xgboost")}


def load(arm, w, arts):
    loader = load_stage_outputs if arm == "glass" else load_blackbox_stage4_inputs
    return loader(w["GS"], arts["s1"], arts["s2"], arts["s3"], verbose=False,
                  extra_fingerprints=ROUTER_FP)


def router_block(arm, art) -> dict:
    return art if arm == "glass" else art["extra"]


def reblend(block, y):
    """GLASS bundle: train probabilities = rule blends at the (new) fold base rates."""
    fold, y = np.asarray(block["train_fold_id"], int), np.asarray(y, int)
    base = np.array([y[fold != k].mean() for k in range(fold.max() + 1)])[fold]
    dec, prec = np.asarray(block["train_decisions"]), np.asarray(block["train_confidence"], float)
    p = np.where(dec == "pass1", base * (1 - prec), np.where(dec == "pass2", base + (1 - base) * prec, base))
    block["train_proba"] = np.where(dec == "uncertain", np.nan, p)


def set_router_folds(arm, art, fold, y, seed=42):
    block = router_block(arm, art)
    block["train_fold_id"] = fold = np.asarray(fold, int)
    block["oof_protocol"] = {**block["oof_protocol"], "n_folds": len(np.unique(fold)),
                             "random_state": seed, "fold_fingerprint": fold_fp(fold)}
    if arm == "glass":
        reblend(block, y)


def failed(exc_info) -> str:
    return str(exc_info.value)


# ----------------------------------------------------------------------
# Accepted: one population, one split, Stage 1's 10 folds beside one 5-fold partition
# ----------------------------------------------------------------------
def test_stage1_ten_folds_coexist_with_the_one_stage2_3_partition(world, arm):
    inp = load(arm, world, artifacts(world, arm))
    s1, s2, s3 = NAMES[arm]
    streams = inp.fold_provenance["streams"]
    assert {n: (p["n_folds"], p["seed"], p["reproduced_by_shared_fold_assignment"])
            for n, p in streams.items()} == {s1: (10, 42, True), s2: (5, 42, True), s3: (5, 42, True)}
    assert inp.fold_provenance["shared_partitions"] == [[s2, s3, 5]]
    assert {tuple(p["streams"]) for p in inp.fold_provenance["distinct_partitions"]} == {
        (s1, s2), (s1, s3)}
    passed = {r["check"] for r in inp.validation if r["passed"]}
    assert {f"{n}: {FOLD_CHECK}" for n in NAMES[arm]} | {f"{SHARED_CHECK} ({s2}, {s3})"} <= passed
    assert all(r["passed"] for r in inp.validation)
    for n in NAMES[arm]:                                   # the split's own rows, never realigned
        assert list(inp[n].train_index) == list(world["GS"]["X_train"].index)
        assert list(inp[n].test_index) == list(world["GS"]["X_test"].index)


# ----------------------------------------------------------------------
# Fold plans that are not the declared shared plan
# ----------------------------------------------------------------------
def _swap_two(y, fold):
    """Two same-label rows from different folds exchange folds: same sizes, same strata."""
    f, y = fold.copy(), np.asarray(y)
    i = int(np.flatnonzero((f == 0) & (y == 1))[0])
    j = int(np.flatnonzero((f == 1) & (y == 1))[0])
    f[i], f[j] = f[j], f[i]
    return f


def _kfold(y):
    f = np.full(len(y), -1)
    for k, (_, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)))):
        f[va] = k
    return f


def _other_row_order(y):
    shuffled = mut.shuffle_rows(y, seed=3)
    return fold_assignment(shuffled, cv_folds=5, random_state=42).reindex(y.index).to_numpy()


#: case -> (fold ids from GLOBAL_SPLIT y_train and its shared 5-fold plan, declared seed, message)
FOLD_CASES = {
    "stage 1's 10-fold partition": (lambda y, f: fold_assignment(y, cv_folds=10, random_state=42),
                                    42, SHARED_CHECK),
    "plan of a relabelled split": (lambda y, f: fold_assignment(mut.permute_labels(y, seed=1),
                                                                cv_folds=5, random_state=42), 42, FOLD_CHECK),
    "plan of another row order": (lambda y, f: _other_row_order(y), 42, FOLD_CHECK),
    "unstratified splitter": (lambda y, f: _kfold(y), 42, FOLD_CHECK),
    "two rows exchanged": (_swap_two, 42, FOLD_CHECK),
    "undeclared seed": (lambda y, f: fold_assignment(y, cv_folds=5, random_state=7), 42, FOLD_CHECK),
    "declared other seed": (lambda y, f: fold_assignment(y, cv_folds=5, random_state=7), 7, SHARED_CHECK),
}


@pytest.mark.parametrize("case", list(FOLD_CASES))
def test_a_router_fold_plan_that_is_not_the_shared_plan_is_refused(world, arm, case):
    make, seed, message = FOLD_CASES[case]
    y = world["GS"]["y_train"]
    shared = fold_assignment(y, cv_folds=5, random_state=42).to_numpy()
    fold = np.asarray(make(y, shared), int)
    assert not np.array_equal(fold, shared)
    arts = artifacts(world, arm)
    set_router_folds(arm, arts["s2"], fold, y, seed)
    with pytest.raises(ValidationError) as exc:
        load(arm, world, arts)
    assert message in failed(exc)


def test_fold_ids_must_be_made_at_the_streams_own_declared_seed(world, arm, monkeypatch):
    """A cascade run at another common seed is consistent and accepted; a router
    declaring seed 42 is refused even though its partners did use its real seed."""
    monkeypatch.setattr(stage4_world, "SEED", 7)
    arts = artifacts(world, arm)                                  # every stage at seed 7
    monkeypatch.undo()
    inp = load(arm, world, arts)
    assert {p["seed"] for p in inp.fold_provenance["streams"].values()} == {7}
    router_block(arm, arts["s2"])["oof_protocol"]["random_state"] = 42
    with pytest.raises(ValidationError) as exc:
        load(arm, world, arts)
    assert f"{NAMES[arm][1]}: {FOLD_CHECK}" in failed(exc)


@pytest.mark.parametrize("case, message", [
    ("row in no fold", "not a partition"),
    ("empty fold", "not a partition"),
    ("no fold ids", "fold ids carried"),
])
def test_router_fold_ids_that_are_missing_or_not_a_partition_are_refused(world, arm, case, message):
    arts = artifacts(world, arm)
    block = router_block(arm, arts["s2"])
    fold = np.asarray(block["train_fold_id"]).copy()
    if case == "row in no fold":
        block["train_fold_id"] = np.where(np.arange(len(fold)) == 0, -1, fold)
    elif case == "empty fold":
        block["train_fold_id"] = np.where(fold == 3, 4, fold)
    else:
        del block["train_fold_id"]
    with pytest.raises(ValidationError) as exc:
        load(arm, world, arts)
    assert message in failed(exc)


# ----------------------------------------------------------------------
# A router that never saw these labels
# ----------------------------------------------------------------------
@pytest.mark.parametrize("partners", [(), ("stage1",)], ids=["router alone", "with stage 1"])
@pytest.mark.parametrize("layout", ["top-level", "extra"], ids=["GLASS bundle", "RF artifact"])
def test_a_router_cross_fitted_on_relabelled_rows_is_refused_without_a_stage3_partner(
        world, layout, partners):
    GS = world["GS"]
    relabelled = {**world, "GS": {**GS, "y_train": mut.permute_labels(GS["y_train"], seed=1)}}
    art = make_router(relabelled, layout=layout)                     # same rows, other labels
    assert art["split_fingerprint"] == ROUTER_FP["router"](GS)      # the row-only scheme agrees
    stream, checks = router_stream(art, "router", "Router", GS["y_train"].to_numpy())
    streams = [stream]
    if partners:
        s1, c1 = stage_output_stream(make_stage1(world), "lr", "LR", len(GS["y_train"]))
        streams, checks = [s1, stream], c1 + checks
    with pytest.raises(ValidationError) as exc:
        build_stage4_inputs(GS, streams, checks, verbose=False, extra_fingerprints=ROUTER_FP)
    assert f"router: {FOLD_CHECK}" in failed(exc)


def test_a_router_whose_training_base_rate_is_not_the_splits_is_refused(world, arm):
    arts = artifacts(world, arm)
    rate = float(world["GS"]["y_train"].mean())
    arts["s2"]["training_base_rate"] = rate + 0.01
    with pytest.raises(ValidationError) as exc:
        load(arm, world, arts)
    assert "training_base_rate = GLOBAL_SPLIT training positive rate" in failed(exc)
    arts["s2"]["training_base_rate"] = rate
    load(arm, world, arts)


# ----------------------------------------------------------------------
# What a label-less router artifact can and cannot prove (F12, partial)
# ----------------------------------------------------------------------
def test_every_stream_reports_how_its_training_labels_were_established(world, arm, capsys):
    inp = load(arm, world, artifacts(world, arm))
    s1, s2, s3 = NAMES[arm]
    assert {n: inp[n].notes["label_evidence"] for n in NAMES[arm]} == {
        s1: contract.LABELS_CARRIED, s2: contract.LABELS_PARTIAL, s3: contract.LABELS_CARRIED}
    assert inp[s2].provenance()["label_evidence"].startswith("PARTIAL (PR 36 F12")
    loader = load_stage_outputs if arm == "glass" else load_blackbox_stage4_inputs
    arts = artifacts(world, arm)
    loader(world["GS"], arts["s1"], arts["s2"], arts["s3"], verbose=True, extra_fingerprints=ROUTER_FP)
    assert f"label evidence {s2}: PARTIAL (PR 36 F12" in capsys.readouterr().out


def test_f12_partial_a_router_fitted_on_within_fold_swapped_labels_still_passes(world, arm):
    """Pins the open part of F12: a router artifact written before the routers
    recorded ``train_label_fingerprint`` carries nothing that can tell. Flip to
    a refusal when the fingerprint becomes mandatory (Commit 4)."""
    GS = world["GS"]
    y = GS["y_train"]
    swapped = mut.swap_labels_within_folds(y, fold_assignment(y, cv_folds=5, random_state=42))
    assert int((swapped != y).sum()) > 0.25 * int(y.sum())        # a quarter of positives moved
    arts = artifacts(world, arm)
    arts["s2"] = make_router({**world, "GS": {**GS, "y_train": swapped}},
                             layout="top-level" if arm == "glass" else "extra")
    arts["s2"]["training_base_rate"] = float(swapped.mean())         # = the split's
    inp = load(arm, world, arts)                                      # accepted
    router = NAMES[arm][1]
    assert inp.fold_provenance["streams"][router]["reproduced_by_shared_fold_assignment"]
    assert inp[router].notes["label_evidence"] == contract.LABELS_PARTIAL


@pytest.mark.parametrize("arm, field", [
    ("glass", "oof_protocol"), ("blackbox", "oof_protocol"), ("blackbox", "extra"),
], ids=["GLASS top level vs oof_protocol", "RF top level vs extra.oof_protocol",
        "RF top level vs extra"])
def test_a_router_artifact_whose_split_fingerprints_disagree_is_refused(world, arm, field):
    arts = artifacts(world, arm)
    block = router_block(arm, arts["s2"])
    target = block["oof_protocol"] if field == "oof_protocol" else block
    target["split_fingerprint"] = arts["s2"]["split_fingerprint"]       # agreeing copy: accepted
    load(arm, world, arts)
    target["split_fingerprint"] = "f" * 16
    with pytest.raises(ValidationError) as exc:
        load(arm, world, arts)
    assert "the split fingerprints the artifact records agree" in failed(exc)


# ----------------------------------------------------------------------
# Row identity: Stage 4 never realigns and never trusts positions
# ----------------------------------------------------------------------
def _hidden_permutation(groups) -> np.ndarray:
    """A permutation moving rows only within ``groups`` (fold x label), so no
    label or fold check can see it."""
    perm = np.arange(len(groups))
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        perm[rows] = np.roll(rows, 1)
    return perm


def _groups(w, side, fold=None):
    y = np.asarray(w["GS"][f"y_{side}"], int)
    return y if fold is None else np.asarray(fold, int) * 2 + y


def reorder_router_rows(arm, art, w, side):
    block = router_block(arm, art)
    fold = block["train_fold_id"] if side == "train" else None
    perm = _hidden_permutation(_groups(w, side, fold))
    for key in ROUTER_ROWS[side]:
        block[key] = np.asarray(block[key])[perm]
    return perm


#: case -> (stream, side, mutation, message)
ROW_CASES = {
    "train rows reordered": ("s2", "train", "reorder", "different ORDER"),
    "train index reset": ("s2", "train", "reset", "index was reset"),
    "train rows reordered, index reset": ("s2", "train", "reorder+reset", "index was reset"),
    "train index missing": ("s2", "train", "missing", "train index carried"),
    "train row label duplicated": ("s2", "train", "duplicate", "index does not match"),
    "test index reset": ("s2", "test", "reset", "index was reset"),
    "test rows reordered, index reset": ("s2", "test", "reorder+reset", "index was reset"),
    "test index missing": ("s2", "test", "missing", "test index carried"),
    "foreign test row": ("s2", "test", "foreign", "index does not match"),
    "stage 3 test rows reordered, index reset": ("s3", "test", "reorder+reset", "index was reset"),
}


@pytest.mark.parametrize("case", list(ROW_CASES))
def test_rows_that_cannot_be_identified_as_the_splits_rows_are_refused(world, arm, case):
    stream, side, how, message = ROW_CASES[case]
    arts = artifacts(world, arm)
    reference = list(world["GS"][f"X_{side}"].index)
    if stream == "s3":
        art = arts["s3"]
        perm = _hidden_permutation(_groups(world, side))
        for key in ("test_proba", "test_proba_calibrated"):
            art.refit[key] = np.asarray(art.refit[key])[perm]
        art.index_test = list(range(len(reference)))
    else:
        block, key = router_block(arm, arts["s2"]), INDEX_KEY[arm][side]
        if "reorder" in how:
            perm = reorder_router_rows(arm, arts["s2"], world, side)
            block[key] = [reference[i] for i in perm]
        if "reset" in how:
            block[key] = list(range(len(reference)))
        elif how == "missing":
            del block[key]
        elif how == "duplicate":
            block[key] = [reference[0]] + reference[:1] + reference[2:]
        elif how == "foreign":
            block[key] = [-1] + reference[1:]
    with pytest.raises(ValidationError) as exc:
        load(arm, world, arts)
    assert message in failed(exc)


def test_the_hidden_reorder_changes_rows_but_no_label_or_fold(world, arm):
    """Guard for the fixture above: only the row identity can reveal the reorder."""
    arts = artifacts(world, arm)
    block = copy.deepcopy(router_block(arm, arts["s2"]))
    perm = reorder_router_rows(arm, arts["s2"], world, "train")
    y = np.asarray(world["GS"]["y_train"], int)
    assert (perm != np.arange(len(perm))).mean() > 0.9
    np.testing.assert_array_equal(y[perm], y)
    np.testing.assert_array_equal(np.asarray(block["train_fold_id"])[perm], block["train_fold_id"])
    assert not np.array_equal(router_block(arm, arts["s2"])["train_decisions"], block["train_decisions"])
