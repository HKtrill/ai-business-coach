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
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Optional

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
        return {
            "split_fingerprint": self.split_fingerprint,
            "split_fingerprints": dict(self.split_fingerprints),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "streams": {n: s.provenance() for n, s in self.streams.items()},
            "folds": self.fold_provenance,
            "validation": [dict(r) for r in self.validation],
        }


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


# ======================================================================
# Primitive checks (each raises with a precise message, or returns detail)
# ======================================================================
def index_alignment(candidate, reference: pd.Index, what: str) -> str:
    """
    'label' if ``candidate`` equals ``reference`` element-wise, 'positional'
    if it is exactly ``range(n)`` (row positions in GLOBAL_SPLIT order).
    Anything else — including the same labels in another order — raises.
    Stage 4 never realigns.
    """
    cand = pd.Index(candidate) if not isinstance(candidate, pd.Index) else candidate
    if len(cand) != len(reference):
        raise Stage4InputError(
            f"{what}: {len(cand)} index entries vs {len(reference)} rows")
    if cand.equals(reference) or np.array_equal(
            np.asarray(cand, dtype=object), np.asarray(reference, dtype=object)):
        return "label"
    if np.array_equal(np.asarray(cand), np.arange(len(reference))):
        return "positional"
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


def build_stage4_inputs(
    GLOBAL_SPLIT: dict,
    streams: list,
    extra_checks: Optional[list] = None,
    verbose: bool = True,
    extra_fingerprints: Optional[dict] = None,
) -> Stage4Inputs:
    """
    Run every alignment / provenance check, then return ``Stage4Inputs``.

    ``extra_checks``: (section, stage, name, fn, detail) tuples from adapters.
    Raises ``shared.validation.ValidationError`` listing every failure.
    """
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
            report.check(SEC_ALIGN, st, f"{s.name}: train index carried",
                         lambda: True, "no index carried — positional contract only")
        if s.test_index is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: test index = X_test.index",
                         lambda s=s: index_alignment(s.test_index, ref.index_test,
                                                     f"{s.name} test index"),
                         s.index_source or "")
        if s.y_train is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: y_train = GLOBAL_SPLIT",
                         lambda s=s: np.array_equal(np.asarray(s.y_train).astype(int),
                                                    ref.y_train))
        if s.y_test is not None:
            report.check(SEC_ALIGN, st, f"{s.name}: y_test = GLOBAL_SPLIT",
                         lambda s=s: np.array_equal(np.asarray(s.y_test).astype(int),
                                                    ref.y_test))

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
        seed = reproduce_folds(s.train_fold_id, ref.y_train, seeds)
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

    for sec, stage, name, fn, detail in (extra_checks or []):
        report.check(sec or SEC_OOF, stage, name, fn, detail)

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
    )
