"""
shared.stage4.contract
======================
The Stage 4 input boundary shared by BOTH arms: aligned, provenance-carrying
base-model streams plus every alignment / provenance check.

Arm-agnostic. Artifact-format adapters live in ``shared.stage4.adapters``;
each arm's loader (``glass_pipeline.meta_ebm.loader``,
``blackbox_pipeline.models.meta_xgb.inputs``) only chooses which adapter reads
which file.

Every ``StageStream`` carries

* ``train_proba``   OUT-OF-FOLD P(y=1) over GLOBAL_SPLIT's training rows, in
                    GLOBAL_SPLIT order — never an in-sample refit prediction;
* ``test_proba``    P(y=1) on the test rows from the full-train / refit model;
* the exact artifact field each array came from, its probability space, its
  split fingerprint (and which scheme produced it), its train-side fold ids and
  its standalone threshold (reported only — Stage 4 never operates at it).

``build_stage4_inputs`` refuses to return unless, for every stream:

* its split fingerprint equals a fingerprint RECOMPUTED from ``GLOBAL_SPLIT``
  (see "Fingerprint schemes");
* every train array has ``len(y_train)`` rows and every test array
  ``len(y_test)`` rows;
* any carried index equals ``X_train.index`` / ``X_test.index`` exactly —
  nothing is ever realigned;
* any carried labels equal ``GLOBAL_SPLIT``'s labels;
* probabilities are finite and in [0, 1] where defined;
* fold ids are a valid partition, and streams with the same number of folds
  share the identical partition (one cascade splitter).

Fingerprint schemes
-------------------
The cascade currently writes two split-fingerprint formats:

``stage_io``  ``shared.stage_io.split_fingerprint(idx_tr, idx_te, y_tr, y_te)``
              — 12 hex; ``StageOutput`` (Stage 1, both arms).
``router``    ``split_fingerprint(X_train, X_test)`` from the router packages
              — 16 hex; GLASS Router, RF Router and both Stage 3 artifacts.

Both are deterministic functions of the current split, so a stream passes if
its fingerprint equals the value of ANY available scheme recomputed from
``GLOBAL_SPLIT``; the matching scheme is recorded. A stream whose scheme cannot
be recomputed here fails.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd


class Stage4InputError(ValueError):
    """An upstream artifact cannot be consumed honestly by Stage 4."""


# ======================================================================
# Streams
# ======================================================================
@dataclass
class StageStream:
    """One upstream model's Stage 4 feed, aligned to GLOBAL_SPLIT order."""

    name: str                       # Stage 4 key: "lr", "glass", "mlp", ...
    stage: str                      # "stage1" / "stage2" / "stage3"
    model: str                      # human label, e.g. "GLASS Router"
    train_proba: np.ndarray         # OOF
    test_proba: np.ndarray          # refit / full-train
    probability_space: str
    train_source: str               # exact artifact field consumed
    test_source: str
    split_fingerprint: Optional[str]
    artifact_path: Optional[str] = None

    train_fold_id: Optional[np.ndarray] = None
    fold_source: Optional[str] = None
    fold_seed: Optional[int] = None

    # Partial-coverage models (routers) also hand over decisions, normalised
    # to "pass1" / "pass2" / "abstain".
    train_decisions: Optional[np.ndarray] = None
    test_decisions: Optional[np.ndarray] = None
    decision_source: Optional[str] = None
    train_defined: Optional[np.ndarray] = None     # rows where proba is meaningful
    test_defined: Optional[np.ndarray] = None

    # Upstream operating point — provenance only, never used by Stage 4.
    standalone_threshold: Optional[float] = None
    standalone_threshold_source: Optional[str] = None

    # Alignment evidence carried by the artifact.
    train_index: Optional[object] = None
    test_index: Optional[object] = None
    index_source: Optional[str] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    fingerprint_scheme: Optional[str] = None       # filled by build_stage4_inputs
    # Streams without labels (routers): {"train outputs" | "test outputs":
    # train_label_fingerprint recorded by the fit behind those outputs}.
    label_fingerprints: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def provenance(self) -> dict:
        """Scalars only — safe for an artifact and a JSON sidecar."""
        out = {
            "stage": self.stage,
            "model": self.model,
            "artifact_path": self.artifact_path,
            "split_fingerprint": self.split_fingerprint,
            "fingerprint_scheme": self.fingerprint_scheme,
            "probability_space": self.probability_space,
            "train_source": self.train_source,
            "test_source": self.test_source,
            "train_is_oof": True,
            "fold_source": self.fold_source,
            "n_folds": (int(len(np.unique(self.train_fold_id)))
                        if self.train_fold_id is not None else None),
            "decision_source": self.decision_source,
            "standalone_threshold": self.standalone_threshold,
            "standalone_threshold_source": self.standalone_threshold_source,
            "standalone_threshold_used_by_stage4": False,
            "index_source": self.index_source,
        }
        out.update(self.notes)
        return out


@dataclass
class Stage4Inputs:
    """Everything Stage 4 may see, verified aligned to GLOBAL_SPLIT."""

    index_train: pd.Index
    index_test: pd.Index
    y_train: np.ndarray
    y_test: np.ndarray
    split_fingerprint: str                  # stage_io scheme (primary)
    split_fingerprints: dict                # every scheme recomputed here
    streams: dict
    fold_provenance: dict
    validation: list                        # ValidationReport rows
    upstream: Optional[dict] = None         # which arm filled the slots (PR 36 F29)

    def __getitem__(self, name: str) -> StageStream:
        return self.streams[name]

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    def fold_frame(self) -> pd.DataFrame:
        """Train-side fold ids per stream. Bookkeeping, not features."""
        return pd.DataFrame(
            {f"{n}_fold_id": s.train_fold_id for n, s in self.streams.items()
             if s.train_fold_id is not None},
            index=self.index_train,
        )

    def provenance(self) -> dict:
        out = {
            "split_fingerprint": self.split_fingerprint,
            "split_fingerprints": dict(self.split_fingerprints),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "streams": {n: s.provenance() for n, s in self.streams.items()},
            "folds": self.fold_provenance,
            "validation": [dict(r) for r in self.validation],
        }
        if self.upstream is not None:
            out["upstream"] = self.upstream
        return out


# ======================================================================
# Reference split + fingerprint schemes
# ======================================================================
#: (scheme, module, attribute, call style). Imported lazily; a missing module
#: just means that scheme is unavailable in this environment.
FINGERPRINT_SCHEMES = (
    ("stage_io", "shared.stage_io", "split_fingerprint", "index+labels"),
    ("router", "blackbox_pipeline.models.rf_router", "split_fingerprint", "frames"),
    ("router", "glass_pipeline.glass_router.pipeline", "split_fingerprint", "frames"),
)


@dataclass(frozen=True)
class SplitReference:
    index_train: pd.Index
    index_test: pd.Index
    y_train: np.ndarray
    y_test: np.ndarray
    fingerprints: dict              # scheme -> value

    @property
    def fingerprint(self) -> str:
        return self.fingerprints["stage_io"]

    def scheme_of(self, fp) -> Optional[str]:
        for k, v in self.fingerprints.items():
            if fp is not None and v == fp:
                return k
        return None


def _labels(y, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if not np.isin(arr, (0, 1)).all():
        raise Stage4InputError(f"GLOBAL_SPLIT[{name!r}] is not binary 0/1")
    return arr.astype(int)


def split_reference(GLOBAL_SPLIT: dict, extra_fingerprints: Optional[dict] = None
                    ) -> SplitReference:
    """
    Indices, int labels and every available split fingerprint of the split
    Stage 4 runs on. ``extra_fingerprints``: {scheme: fn(GLOBAL_SPLIT) -> str}.
    """
    missing = [k for k in ("X_train", "X_test", "y_train", "y_test")
               if k not in GLOBAL_SPLIT]
    if missing:
        raise Stage4InputError(f"GLOBAL_SPLIT missing keys: {missing}")
    X_tr, X_te = GLOBAL_SPLIT["X_train"], GLOBAL_SPLIT["X_test"]
    if not hasattr(X_tr, "index") or not hasattr(X_te, "index"):
        raise Stage4InputError("GLOBAL_SPLIT X_train / X_test must be DataFrames")
    y_tr = _labels(GLOBAL_SPLIT["y_train"], "y_train")
    y_te = _labels(GLOBAL_SPLIT["y_test"], "y_test")
    if len(y_tr) != len(X_tr) or len(y_te) != len(X_te):
        raise Stage4InputError("GLOBAL_SPLIT label / feature lengths differ")
    for name, ref in (("y_train", X_tr.index), ("y_test", X_te.index)):
        y = GLOBAL_SPLIT[name]
        if isinstance(y, pd.Series) and not y.index.equals(ref):
            raise Stage4InputError(f"GLOBAL_SPLIT[{name!r}].index != X index")

    fps: dict = {}
    for scheme, mod, attr, style in FINGERPRINT_SCHEMES:
        if scheme in fps:
            continue
        try:
            fn = getattr(importlib.import_module(mod), attr)
        except (ImportError, AttributeError):
            continue
        fps[scheme] = (fn(X_tr.index, X_te.index, y_tr, y_te)
                       if style == "index+labels" else fn(X_tr, X_te))
    for scheme, fn in (extra_fingerprints or {}).items():
        fps[scheme] = fn(GLOBAL_SPLIT)
    if "stage_io" not in fps:
        raise Stage4InputError(
            "shared.stage_io.split_fingerprint is required to verify the split")
    return SplitReference(X_tr.index, X_te.index, y_tr, y_te, fps)


def check_split_identity(recorded: Optional[dict], ref: SplitReference, what: str,
                         primary: Optional[str] = None) -> str:
    """
    ``recorded`` ({scheme: fingerprint}, as a Stage 4 artifact stores it)
    describes the split ``ref`` was recomputed from. ``primary``: the
    artifact's single ``split_fingerprint`` field, if it has one. Returns the
    schemes compared; raises ``Stage4InputError`` otherwise.

    PR 36 (audit F12): artifacts recording several schemes were accepted if
    ANY recorded value equalled ANY recomputed one. The router scheme hashes
    row indices only, so an artifact built on the same rows with other
    training labels (its label-bearing ``stage_io`` fingerprint differing)
    matched on the router value alone. Now every scheme both sides know must
    agree, and at least one must be shared. A primary fingerprint must be one
    of the recorded values (it used to be ignored whenever the dictionary was
    present, so the two could contradict each other); alone, it is compared
    under the scheme it matches.
    """
    recorded = {k: v for k, v in (recorded or {}).items() if v}
    if primary and recorded and primary not in recorded.values():
        raise Stage4InputError(
            f"{what}: its split_fingerprint {primary!r} contradicts its own "
            f"split_fingerprints {recorded}")
    if primary and not recorded:
        recorded = {ref.scheme_of(primary) or "unrecognised": primary}
    common = [k for k in recorded if k in ref.fingerprints]
    if not common:
        raise Stage4InputError(
            f"{what}: no fingerprint scheme in common with GLOBAL_SPLIT (recorded "
            f"{sorted(recorded)}, recomputable {sorted(ref.fingerprints)})")
    bad = [f"{k}: {recorded[k]} != {ref.fingerprints[k]}" for k in common
           if recorded[k] != ref.fingerprints[k]]
    if bad:
        raise Stage4InputError(f"{what} was built on another split ({'; '.join(bad)})")
    return ", ".join(common)


# ======================================================================
# Primitive checks (each raises with a precise message, or returns detail)
# ======================================================================
def index_alignment(candidate, reference: pd.Index, what: str) -> str:
    """
    'label' if ``candidate`` equals ``reference`` element-wise. Anything else
    — the same labels in another order, a reset index, foreign or missing
    rows — raises. Stage 4 never realigns.

    PR 36 (audit F12): a candidate equal to ``range(n)`` used to be accepted
    as "positional" (row positions in GLOBAL_SPLIT order). A reset index
    carries no row identity, so rows reordered before the reset (within a
    fold, among rows with one label) were accepted unnoticed. Every current
    producer records the split's row labels.
    """
    cand = pd.Index(candidate) if not isinstance(candidate, pd.Index) else candidate
    if len(cand) != len(reference):
        raise Stage4InputError(
            f"{what}: {len(cand)} index entries vs {len(reference)} rows")
    if cand.equals(reference) or np.array_equal(
            np.asarray(cand, dtype=object), np.asarray(reference, dtype=object)):
        return "label"
    if np.array_equal(np.asarray(cand), np.arange(len(reference))):
        raise Stage4InputError(
            f"{what}: the index was reset to row positions 0..{len(reference) - 1}; "
            "Stage 4 needs the split's row labels to verify the row order.")
    if set(cand) == set(reference):
        raise Stage4InputError(
            f"{what}: same rows as GLOBAL_SPLIT but in a different ORDER. "
            "Stage 4 does not silently realign; re-export the artifact in "
            "GLOBAL_SPLIT order.")
    raise Stage4InputError(f"{what}: index does not match GLOBAL_SPLIT")


def check_proba(p: np.ndarray, what: str, defined: Optional[np.ndarray] = None) -> str:
    p = np.asarray(p, dtype=float)
    q = p if defined is None else p[np.asarray(defined, dtype=bool)]
    if not np.isfinite(q).all():
        raise Stage4InputError(f"{what}: {int((~np.isfinite(q)).sum())} non-finite values")
    if ((q < 0) | (q > 1)).any():
        raise Stage4InputError(f"{what}: values outside [0, 1]")
    return f"{len(q):,} values in [{q.min():.4f}, {q.max():.4f}]" if len(q) else "0 values"


def fold_partition(fold_id: np.ndarray, what: str) -> int:
    f = np.asarray(fold_id)
    if not np.issubdtype(f.dtype, np.integer):
        if not np.all(np.equal(np.mod(f, 1), 0)):
            raise Stage4InputError(f"{what}: fold ids are not integers")
        f = f.astype(int)
    k = int(f.max()) + 1 if len(f) else 0
    counts = np.bincount(f, minlength=k) if len(f) and f.min() >= 0 else None
    if counts is None or (counts == 0).any() or k < 2:
        raise Stage4InputError(f"{what}: fold ids are not a partition 0..k-1")
    return k


def reproduce_folds(fold_id: np.ndarray, y_train: np.ndarray, seeds) -> Optional[int]:
    """Seed for which shared.stage_io.fold_assignment reproduces these ids."""
    from shared.stage_io import fold_assignment
    k = int(np.max(fold_id)) + 1
    for s in dict.fromkeys(x for x in seeds if x is not None):
        ref = fold_assignment(y_train, cv_folds=k, random_state=int(s)).to_numpy()
        if np.array_equal(ref, np.asarray(fold_id, dtype=int)):
            return int(s)
    return None


def not_identical(a, b) -> bool:
    """True unless ``a`` and ``b`` are the same vector (aliasing guard)."""
    if b is None:
        return True
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return True
    return not np.array_equal(a, b, equal_nan=True)


# ======================================================================
# Assembly
# ======================================================================
SEC_SPLIT, SEC_ALIGN, SEC_FOLD, SEC_OOF, SEC_SEM = (
    "4a. Split identity", "4b. Row alignment", "4c. Fold provenance",
    "4d. OOF honesty", "4e. Router semantics")

#: ``StageStream.notes["label_evidence"]``: how Stage 4 established that a
#: stream was produced on GLOBAL_SPLIT's training labels. Since PR 36 Commit 4
#: (F12) there is no partial case: a stream that carries no labels must record
#: a label fingerprint for both its train and its test outputs, or it is refused.
LABELS_CARRIED = "complete: training labels carried and compared with GLOBAL_SPLIT"
LABELS_FINGERPRINTED = (
    "complete: the label fingerprints of the fits behind the train and the test outputs "
    "match GLOBAL_SPLIT's training rows and labels")
LABEL_FINGERPRINT_SIDES = ("train outputs", "test outputs")
_LABEL_FP = re.compile(r"[0-9a-f]{12}")


def label_identity(recorded, expected: str, what: str) -> str:
    """
    ``recorded``: a ``train_label_fingerprint`` from an artifact; ``expected``:
    ``shared.stage_io.label_fingerprint`` of GLOBAL_SPLIT's training rows and
    labels. Raises ``Stage4InputError`` on a missing or malformed value or
    another set of rows / labels (PR 36 audit F12). A missing value is an
    artifact written before the producers recorded one: it is refused, never
    filled in, because a fingerprint added after the fit certifies nothing.
    """
    if recorded is None:
        raise Stage4InputError(
            f"{what}: no label fingerprint recorded — the artifact was written before "
            "PR 36 F12 made it mandatory; regenerate it (fingerprints are never backfilled)")
    if not isinstance(recorded, str) or not _LABEL_FP.fullmatch(recorded):
        raise Stage4InputError(f"{what}: malformed label fingerprint {recorded!r} "
                               "(expected 12 lowercase hex characters)")
    if recorded != expected:
        raise Stage4InputError(f"{what}: fitted on other training rows or labels "
                               f"(label fingerprint {recorded} != {expected})")
    return recorded


def build_stage4_inputs(
    GLOBAL_SPLIT: dict,
    streams: list,
    extra_checks: Optional[list] = None,
    verbose: bool = True,
    extra_fingerprints: Optional[dict] = None,
    upstream: Optional[dict] = None,
) -> Stage4Inputs:
    """
    Run every alignment / provenance check, then return ``Stage4Inputs``.

    ``extra_checks``: (section, stage, name, fn, detail) tuples from adapters.
    ``upstream``: {"arm", "arbiter", "arbiter_arm"} from an arbiter's loader —
    every stream must then belong to that arm, and the verified combination is
    recorded as ``Stage4Inputs.upstream`` (PR 36 F29; ``shared.stage4.upstream``).
    Raises ``shared.validation.ValidationError`` listing every failure.
    """
    from shared.stage_io import label_fingerprint
    from shared.validation import ValidationReport

    ref = split_reference(GLOBAL_SPLIT, extra_fingerprints)
    n_tr, n_te = len(ref.y_train), len(ref.y_test)
    report = ValidationReport()

    report.check(SEC_SPLIT, "GLOBAL_SPLIT", "fingerprints recomputed",
                 lambda: bool(ref.fingerprints), str(ref.fingerprints))
    for s in streams:
        s.fingerprint_scheme = ref.scheme_of(s.split_fingerprint)
        report.check(SEC_SPLIT, s.stage, f"{s.name}: split fingerprint = GLOBAL_SPLIT",
                     lambda s=s: s.fingerprint_scheme is not None,
                     f"{s.split_fingerprint!r} ({s.fingerprint_scheme or 'no scheme matches'})")
    report.check(SEC_SPLIT, "all stages", "every stage built on this split",
                 lambda: all(s.fingerprint_scheme for s in streams),
                 str({s.name: s.fingerprint_scheme for s in streams}))

    for s in streams:
        st = s.stage
        report.check(SEC_ALIGN, st, f"{s.name}: train proba has len(y_train) rows",
                     lambda s=s: len(s.train_proba) == n_tr,
                     f"{len(s.train_proba):,} vs {n_tr:,}")
        report.check(SEC_ALIGN, st, f"{s.name}: test proba has len(y_test) rows",
                     lambda s=s: len(s.test_proba) == n_te,
                     f"{len(s.test_proba):,} vs {n_te:,}")
        report.check(SEC_ALIGN, st, f"{s.name}: train proba finite, in [0,1]",
                     lambda s=s: check_proba(s.train_proba, f"{s.name} train",
                                             s.train_defined))
        report.check(SEC_ALIGN, st, f"{s.name}: test proba finite, in [0,1]",
                     lambda s=s: check_proba(s.test_proba, f"{s.name} test",
                                             s.test_defined))
        if s.train_decisions is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: train decisions have len(y_train) rows",
                         lambda s=s: len(s.train_decisions) == n_tr)
            report.check(SEC_ALIGN, st, f"{s.name}: test decisions have len(y_test) rows",
                         lambda s=s: len(s.test_decisions) == n_te)
        if s.train_fold_id is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: train fold ids have len(y_train) rows",
                         lambda s=s: len(s.train_fold_id) == n_tr)
        if s.train_index is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: train index = X_train.index",
                         lambda s=s: index_alignment(s.train_index, ref.index_train,
                                                     f"{s.name} train index"),
                         s.index_source or "")
        else:
            # PR 36 (audit F12): used to pass as a "positional contract".
            report.check(SEC_ALIGN, st, f"{s.name}: train index carried",
                         lambda: False, "no row index carried — the row order cannot be verified")
        if s.test_index is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: test index = X_test.index",
                         lambda s=s: index_alignment(s.test_index, ref.index_test,
                                                     f"{s.name} test index"),
                         s.index_source or "")
        else:
            report.check(SEC_ALIGN, st, f"{s.name}: test index carried",
                         lambda: False, "no row index carried — the row order cannot be verified")
        if s.y_train is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: y_train = GLOBAL_SPLIT",
                         lambda s=s: np.array_equal(np.asarray(s.y_train).astype(int),
                                                    ref.y_train))
        if s.y_test is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: y_test = GLOBAL_SPLIT",
                         lambda s=s: np.array_equal(np.asarray(s.y_test).astype(int),
                                                    ref.y_test))
        if s.y_train is None:
            # PR 36 (audit F12): a stream without labels (a router) must record the
            # label fingerprint of the fit behind EACH side, compared with GLOBAL_SPLIT's
            # training rows and labels (never its test labels). Required since Commit 4:
            # its fold ids and base rate cannot see opposite labels swapped within a fold.
            want = label_fingerprint(ref.index_train, ref.y_train)
            for side in LABEL_FINGERPRINT_SIDES:
                fp = s.label_fingerprints.get(side)
                report.check(SEC_SPLIT, st, f"{s.name}: {side} fitted on GLOBAL_SPLIT's training "
                             "rows and labels",
                             lambda fp=fp, want=want, what=f"{s.name} {side}": label_identity(
                                 fp, want, what),
                             "train_label_fingerprint" + ("" if fp is not None else " (missing)"))
        s.notes["label_evidence"] = LABELS_CARRIED if s.y_train is not None else LABELS_FINGERPRINTED

    # ---- folds ---------------------------------------------------------
    fold_prov: dict = {"streams": {}, "shared_partitions": [], "distinct_partitions": []}
    folds = {s.name: s for s in streams if s.train_fold_id is not None}
    seeds = [s.fold_seed for s in streams] + [42]
    ks: dict = {}
    for n, s in folds.items():
        ok = report.check(SEC_FOLD, s.stage, f"{n}: fold ids partition train",
                          lambda s=s: fold_partition(s.train_fold_id, s.name))
        if not ok:
            continue
        k = fold_partition(s.train_fold_id, n)
        ks[n] = k
        # PR 36 (audit F11/F12): the fold plan must be the shared stratified
        # splitter's plan on GLOBAL_SPLIT's training labels, in GLOBAL_SPLIT
        # row order, with the stream's own declared seed. It used to be
        # recorded only, so a fold plan built on other labels (a relabelled
        # split under a label-blind fingerprint), in another row order or by
        # another splitter was accepted whenever no stream with the same fold
        # count disagreed. One relabelled row reassigns most fold ids.
        own = [s.fold_seed] if s.fold_seed is not None else seeds
        seed = reproduce_folds(s.train_fold_id, ref.y_train, own)
        report.check(SEC_FOLD, s.stage,
                     f"{n}: fold ids = shared stratified splitter on GLOBAL_SPLIT labels",
                     lambda seed=seed: seed is not None,
                     f"{k} folds, declared seed {s.fold_seed}")
        fold_prov["streams"][n] = {
            "source": s.fold_source, "n_folds": k,
            "reproduced_by_shared_fold_assignment": seed is not None,
            "seed": seed,
        }
    for a, b in combinations(ks, 2):
        fa, fb = folds[a].train_fold_id, folds[b].train_fold_id
        if ks[a] == ks[b]:
            report.check(SEC_FOLD, "cascade", f"{a} & {b}: identical {ks[a]}-fold partition",
                         lambda fa=fa, fb=fb: np.array_equal(np.asarray(fa, int),
                                                             np.asarray(fb, int)))
            fold_prov["shared_partitions"].append([a, b, ks[a]])
        else:
            fold_prov["distinct_partitions"].append(
                {"streams": [a, b], "n_folds": [ks[a], ks[b]],
                 "why": "different n_splits — distinct OOF partitions by design; "
                        "each stream is still OOF w.r.t. its own folds"})
    # PR 36 (audit F11): Stages 2 and 3 are declared to share ONE partition
    # (5 stratified folds, the cascade seed). Only equal fold COUNTS were
    # compared, so a Stage 2 cross-fit on another fold count passed as
    # "distinct by design". Stage 1's own fold count stays independent.
    mid = [n for n in ks if folds[n].stage in ("stage2", "stage3")]
    if len(mid) > 1:
        report.check(SEC_FOLD, "cascade", f"stages 2-3 share one partition ({', '.join(mid)})",
                     lambda: all(np.array_equal(np.asarray(folds[mid[0]].train_fold_id, int),
                                                np.asarray(folds[n].train_fold_id, int))
                                 for n in mid[1:]),
                     str({n: ks[n] for n in mid}))

    for sec, stage, name, fn, detail in (extra_checks or []):
        report.check(sec or SEC_OOF, stage, name, fn, detail)
    if upstream is not None:
        from .upstream import upstream_checks
        for sec, stage, name, fn, detail in upstream_checks(streams, upstream["arm"]):
            report.check(sec, stage, name, fn, detail)

    if verbose:
        report.show()
    report.raise_if_failed()

    return Stage4Inputs(
        index_train=ref.index_train,
        index_test=ref.index_test,
        y_train=ref.y_train,
        y_test=ref.y_test,
        split_fingerprint=ref.fingerprint,
        split_fingerprints=dict(ref.fingerprints),
        streams={s.name: s for s in streams},
        fold_provenance=fold_prov,
        validation=list(report.rows),
        upstream=None if upstream is None else _upstream_record(streams, upstream),
    )


def _upstream_record(streams, upstream: dict) -> dict:
    from .upstream import upstream_record
    return upstream_record(streams, upstream["arm"], upstream["arbiter"], upstream["arbiter_arm"])
