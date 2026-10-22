"""
F12 — one split, several legitimate fingerprint schemes (PR 36, Commit 3 Step 3B).

    scheme      function                                         what it hashes
    stage_io    shared.stage_io.split_fingerprint(train_idx,     row identities, row order,
                test_idx, y_train, y_test)            12 hex     sizes AND labels
    router      rf_router.split_fingerprint(X_train, X_test)     row identities, row order,
                                                      16 hex     sizes; NO labels
    fold plan   fold_fingerprint(fold_assignment(y_train, k,     a many-to-one function of the
                seed))                                16 hex     training label sequence; no
                                                                 row identities
    train       shared.stage_io.label_fingerprint(X_train.index, the training rows, their order
    labels      y_train)                              12 hex     AND labels; no test rows

Who records which: Stage 1 StageOutputs, the GLASS Stage 3 block and both
Meta artifacts record ``stage_io``; both routers and the black-box Stage 3
block record ``router``; Stage 2-4 artifacts record their fold ids; since
Commit 3 both routers' fits record ``train labels`` (their training rows).

Contracts pinned here
* exactly what each scheme detects. The router scheme is blind to labels and
  the fold plan to row identities. Together they detect every mutation here
  but one: two opposite labels swapped within one fold change no fold id and
  no base rate (a stratified splitter deals each class's rows to folds in
  row order). Stage 4 reads a label-less router artifact's fold ids as
  PARTIAL label evidence only, and says so. The training-label fingerprint
  each router fit records since Commit 3 (``train labels`` below) closes it
  when present; it becomes mandatory in Commit 4 (PR 36 F12, PARTIAL until then).
* ``check_split_identity``: an artifact recording several schemes must agree
  on EVERY scheme both sides can compute, and at least one. Schemes are
  compared by name — never "some recorded hash equals some recomputed hash",
  which let the label-blind router value vouch for a relabelled split. An
  artifact's primary fingerprint must be one of the values it records.
* the Meta-EBM vs Meta-XGB comparison refuses either side built on another
  split, including the same rows with other labels, or contradicting itself,
  and still accepts artifacts recording a single (primary) fingerprint.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import shared.stage4 as stage4
import shared.stage4.contract as contract
from blackbox_pipeline.models.meta_xgb.artifact import load_meta_xgb, save_meta_xgb
from blackbox_pipeline.models.meta_xgb.comparison import (
    compare_with_meta_ebm, load_meta_ebm_for_comparison,
)
from blackbox_pipeline.models.rf_router import split_fingerprint as router_split_fingerprint
from glass_pipeline.glass_router.pipeline import fold_fingerprint
from shared.stage4 import Stage4InputError, split_reference
import shared.stage_io as stage_io
from shared.stage_io import fold_assignment, split_fingerprint
from tests.support import mutations as mut
from tests.support.meta_xgb_world import blackbox_world, run_meta
from tests.support.stage4_world import ROUTER_FP, build_world

SCHEMES = ("stage_io", "router", "fold_plan", "train_labels")


def label_fingerprint(index, y) -> str:
    """``shared.stage_io.label_fingerprint``, resolved at call time."""
    return stage_io.label_fingerprint(index, y)


def schemes(GS) -> dict:
    ytr, yte = GS["y_train"].astype(int), GS["y_test"].astype(int)
    return {"stage_io": split_fingerprint(GS["X_train"].index, GS["X_test"].index, ytr, yte),
            "router": router_split_fingerprint(GS["X_train"], GS["X_test"]),
            "fold_plan": fold_fingerprint(fold_assignment(ytr, cv_folds=5, random_state=42).to_numpy()),
            "train_labels": label_fingerprint(GS["X_train"].index, ytr)}


def _train(GS, fn):
    return {**GS, "X_train": fn(GS["X_train"]), "y_train": fn(GS["y_train"])}


def _moved(GS):
    row = GS["X_test"].index[:1]
    return {"X_train": pd.concat([GS["X_train"], GS["X_test"].loc[row]]),
            "y_train": pd.concat([GS["y_train"], GS["y_test"].loc[row]]),
            "X_test": GS["X_test"].drop(row), "y_test": GS["y_test"].drop(row)}


def _flipped(GS):
    y = GS["y_train"].copy()
    y.iloc[0] = 1 - y.iloc[0]
    return {**GS, "y_train": y}


def _swapped_within_folds(GS, max_pairs=1):
    y = GS["y_train"].astype(int)
    fold = fold_assignment(y, cv_folds=5, random_state=42).to_numpy()
    return {**GS, "y_train": mut.swap_labels_within_folds(y, fold, max_pairs=max_pairs)}


#: mutation -> (GLOBAL_SPLIT -> mutated split, detected by (stage_io, router, fold plan,
#: train labels))
MUTATIONS = {
    "rows reordered": (lambda GS: _train(GS, lambda o: mut.shuffle_rows(o, seed=3)),
                       (True, True, True, True)),
    "index reset": (lambda GS: _train(GS, mut.reset_index), (True, True, False, True)),
    "row label duplicated": (lambda GS: _train(GS, mut.duplicate_index_label), (True, True, False, True)),
    "foreign row": (lambda GS: _train(GS, mut.replace_index_label), (True, True, False, True)),
    "row moved test -> train": (_moved, (True, True, True, True)),
    "one label flipped": (_flipped, (True, False, True, True)),
    "labels permuted": (lambda GS: {**GS, "y_train": mut.permute_labels(GS["y_train"], seed=1)},
                        (True, False, True, True)),
    "opposite labels swapped within a fold": (_swapped_within_folds, (True, False, False, True)),
}
#: What a label-less router artifact (router scheme + fold ids) cannot see.
UNSEEN_BY_ROUTER_EVIDENCE = {"opposite labels swapped within a fold"}


@pytest.fixture(scope="module")
def GS():
    return build_world(n=900, n_test=200)["GS"]


def test_every_scheme_is_deterministic_and_reads_values_not_objects(GS):
    copy = {k: v.copy() for k, v in GS.items()}
    assert schemes(copy) == schemes(GS)
    assert len(set(schemes(GS).values())) == len(SCHEMES)


@pytest.mark.parametrize("mutation", list(MUTATIONS))
def test_what_each_scheme_detects(GS, mutation):
    change, expected = MUTATIONS[mutation]
    base, moved = schemes(GS), schemes(change(GS))
    detected = tuple(base[s] != moved[s] for s in SCHEMES)
    assert detected == expected
    # A label-less router artifact (router scheme + fold ids) is as identifying
    # as the label-bearing stage_io scheme, if its fold ids are checked — except
    # for the within-fold swap (PR 36 F12, partial) ...
    covered = (detected[1] or detected[2]) is detected[0]
    assert covered is (mutation not in UNSEEN_BY_ROUTER_EVIDENCE)
    # ... which the routers' training-label fingerprint closes, where recorded.
    assert (detected[1] or detected[3]) is detected[0]


def test_a_within_fold_swap_of_opposite_labels_changes_no_fold_id_and_no_base_rate(GS):
    """The F12 counterexample, on the shared fold_assignment (5 folds, seed 42)."""
    y = GS["y_train"].astype(int)
    fold = fold_assignment(y, cv_folds=5, random_state=42).to_numpy()
    for max_pairs in (1, None):                        # one swap; every disjoint one
        z = mut.swap_labels_within_folds(y, fold, max_pairs=max_pairs)
        np.testing.assert_array_equal(fold_assignment(z, cv_folds=5, random_state=42).to_numpy(), fold)
        assert z.sum() == y.sum()                                          # training base rate
        for k in range(5):                                # held-out and fold-model base rates
            assert z[fold == k].sum() == y[fold == k].sum()
    changed = int((z != y).sum())
    assert changed > 0.25 * y.sum()        # a quarter of the positives can move, unseen
    assert split_fingerprint(GS["X_train"].index, GS["X_test"].index, z,
                             GS["y_test"].astype(int)) != schemes(GS)["stage_io"]


# ----------------------------------------------------------------------
# check_split_identity: per-scheme comparison
# ----------------------------------------------------------------------
def _relabelled(GS, side="train", seed=1):
    key = f"y_{side}"
    return {**GS, key: mut.permute_labels(GS[key], seed=seed)}


def test_every_scheme_both_sides_know_is_compared_by_name(GS):
    ref = split_reference(GS, ROUTER_FP)
    fps = dict(ref.fingerprints)
    check = stage4.check_split_identity
    assert check(fps, ref, "artifact") == "stage_io, router"
    assert check({"router": fps["router"]}, ref, "artifact") == "router"     # Stage 2/3 artifacts
    assert check({"stage_io": fps["stage_io"]}, ref, "artifact") == "stage_io"
    # a scheme this environment cannot recompute is not evidence either way
    assert check({**fps, "retired_scheme": "0" * 12}, ref, "artifact") == "stage_io, router"
    assert check({**fps, "router": None}, ref, "artifact") == "stage_io"


@pytest.mark.parametrize("side", ["train", "test"])
def test_one_disagreeing_scheme_refuses_the_artifact_even_when_another_agrees(GS, side):
    ref = split_reference(GS, ROUTER_FP)
    other = split_reference(_relabelled(GS, side), ROUTER_FP).fingerprints
    assert other["router"] == ref.fingerprints["router"]      # the rows are the same rows
    with pytest.raises(Stage4InputError, match=r"artifact was built on another split \(stage_io: "):
        stage4.check_split_identity(other, ref, "artifact")


@pytest.mark.parametrize("recorded", [
    {}, {"retired_scheme": "0" * 12}, {"stage_io": None},
], ids=["nothing", "unknown scheme only", "empty value"])
def test_an_artifact_without_a_verifiable_scheme_is_refused(GS, recorded):
    with pytest.raises(Stage4InputError, match="no fingerprint scheme in common"):
        stage4.check_split_identity(recorded, split_reference(GS, ROUTER_FP), "artifact")


def test_a_hash_recorded_under_another_schemes_name_is_not_accepted(GS):
    ref = split_reference(GS, ROUTER_FP)
    with pytest.raises(Stage4InputError, match="another split"):
        stage4.check_split_identity({"stage_io": ref.fingerprints["router"]}, ref, "artifact")


def test_a_primary_fingerprint_must_be_one_of_the_recorded_values(GS):
    ref = split_reference(GS, ROUTER_FP)
    fps, check = dict(ref.fingerprints), stage4.check_split_identity
    assert check(fps, ref, "artifact", primary=fps["stage_io"]) == "stage_io, router"
    assert check(None, ref, "artifact", primary=fps["router"]) == "router"     # primary only
    relabelled = split_reference(_relabelled(GS), ROUTER_FP).fingerprint
    for primary in ("0" * 12, relabelled):
        with pytest.raises(Stage4InputError, match="contradicts its own split_fingerprints"):
            check(fps, ref, "artifact", primary=primary)
    with pytest.raises(Stage4InputError, match="no fingerprint scheme in common"):
        check(None, ref, "artifact", primary=relabelled)                  # primary only, unknown


# ----------------------------------------------------------------------
# Meta-EBM vs Meta-XGB comparison (both consumers of check_split_identity)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def meta(tmp_path_factory):
    """A real Meta-XGB artifact (small fixed params) on the black-box Stage 4 world."""
    w, arts = blackbox_world()
    art, _ = run_meta(w, arts, tmp_path_factory.mktemp("meta"))
    return w["GS"], art


def meta_ebm(GS, primary_only=False) -> dict:
    """The meta_ebm/2 fields the comparison reads, recorded on ``GS``."""
    ref = split_reference(GS, ROUTER_FP)
    p = np.random.default_rng(0).random(len(GS["y_test"]))
    art = {"schema": "meta_ebm/2", "split_fingerprint": ref.fingerprint,
           "split_fingerprints": dict(ref.fingerprints), "index_test": list(GS["X_test"].index),
           "test_preds": {"no_abstain": (p > .5).astype(int),
                          "with_abstain": np.where(np.abs(p - .5) < .1, -1, (p > .5).astype(int))},
           "test_probs": {"meta": p}, "thresholds": {}, "abstention": {}, "timestamp": "fixture"}
    if primary_only:
        del art["split_fingerprints"]
    return art


@pytest.mark.parametrize("primary_only", [False, True], ids=["all schemes", "primary only"])
def test_both_meta_artifacts_from_the_split_compared_on_are_accepted(meta, primary_only):
    GS, art = meta
    out = compare_with_meta_ebm(GS, meta_ebm(GS, primary_only), art, n_boot=20, verbose=False,
                                extra_fingerprints=ROUTER_FP)
    assert out["protocol"]["n_test"] == len(GS["y_test"])
    assert set(art.split_fingerprints) == {"stage_io", "router"}


@pytest.mark.parametrize("side", ["train", "test"])
def test_a_meta_ebm_built_on_the_same_rows_with_other_labels_is_refused(meta, side):
    GS, _ = meta
    other = meta_ebm(_relabelled(GS, side))
    assert other["split_fingerprints"]["router"] == split_reference(GS, ROUTER_FP).fingerprints["router"]
    with pytest.raises(Stage4InputError, match=r"Meta-EBM artifact was built on another split \(stage_io"):
        load_meta_ebm_for_comparison(GS, other, ROUTER_FP)


@pytest.mark.parametrize("side", ["train", "test"])
def test_a_meta_xgb_built_on_the_same_rows_with_other_labels_is_refused(meta, side):
    GS, art = meta
    split = _relabelled(GS, side)                     # the Meta-EBM side agrees with this split
    assert list(art.index_test) == list(split["X_test"].index)
    with pytest.raises(Stage4InputError, match=r"Meta-XGB artifact was built on another split \(stage_io"):
        compare_with_meta_ebm(split, meta_ebm(split), art, n_boot=20, verbose=False,
                              extra_fingerprints=ROUTER_FP)


def test_a_meta_artifact_whose_primary_fingerprint_contradicts_its_dictionary_is_refused(
        meta, tmp_path):
    GS, art = meta
    ebm = {**meta_ebm(GS), "split_fingerprint": "0" * 12}
    with pytest.raises(Stage4InputError, match="Meta-EBM artifact: its split_fingerprint"):
        load_meta_ebm_for_comparison(GS, ebm, ROUTER_FP)
    bad = dataclasses.replace(art, split_fingerprint="0" * 12)
    with pytest.raises(Stage4InputError, match="Meta-XGB artifact: its split_fingerprint"):
        compare_with_meta_ebm(GS, meta_ebm(GS), bad, n_boot=20, verbose=False,
                              extra_fingerprints=ROUTER_FP)
    with pytest.raises(ValueError, match="contradicts its own split_fingerprints"):
        load_meta_xgb(save_meta_xgb(bad, str(tmp_path / "bad")))
    good = load_meta_xgb(save_meta_xgb(art, str(tmp_path / "good")),
                         expected_split_fingerprint=art.split_fingerprint)
    assert good.split_fingerprints == art.split_fingerprints


def test_the_meta_artifact_records_that_router_label_evidence_is_partial(meta):
    """The open part of F12 is written into the artifact, not only printed."""
    _, art = meta
    evidence = {n: s["label_evidence"] for n, s in art.stage4_inputs["streams"].items()}
    assert evidence["rf_router"] == contract.LABELS_PARTIAL
    assert evidence["mlp"] == evidence["xgb"] == contract.LABELS_CARRIED
