"""
Leakage verdicts computed from a ``CallLog`` (see ``tests.support.recorders``).

These answer questions about what actually happened at fit / predict time,
not about what an output's shape or metadata claims:

assert_never_fit_on(log, rows)
    No fit (estimator or transformer) ever saw any of ``rows``. Use it for
    held-out test rows, or validation rows when fitting must stay in-fold.

assert_oof_provenance(log, train_rows)
    The calls look like honest out-of-fold scoring of ``train_rows``: at
    least two fold models, each fitted on a strict subset of the training
    rows, each scoring only rows it never saw, and together scoring every
    training row exactly once. A full-train refit is ignored here, because
    scoring the training rows with it is in-sample by definition; keep its
    output out of anything labelled OOF.

Pass ``methods=("transform",)`` to check per-fold feature fitting instead of
model scoring.

Both raise ``ProvenanceError`` (an ``AssertionError``) with the offending fits.
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable

from .recorders import PREDICT_METHODS, CallLog


class ProvenanceError(AssertionError):
    """A recorded fit / predict pattern violates the leakage contract."""


def _preview(keys, n: int = 5) -> str:
    keys = sorted(keys, key=repr)
    more = f" … (+{len(keys) - n})" if len(keys) > n else ""
    return ", ".join(map(repr, keys[:n])) + more


def assert_never_fit_on(log: CallLog, rows: Iterable[Hashable]) -> None:
    """Raise if any recorded fit included one of ``rows``."""
    forbidden = set(rows)
    bad = {fid: keys & forbidden for fid, keys in log.fits().items() if keys & forbidden}
    if bad:
        detail = "; ".join(f"fit {fid}: {len(k)} rows ({_preview(k)})" for fid, k in bad.items())
        raise ProvenanceError(f"{len(bad)} fit(s) saw forbidden rows: {detail}")


def assert_oof_provenance(
    log: CallLog,
    train_rows: Iterable[Hashable],
    methods: Iterable[str] = PREDICT_METHODS,
) -> None:
    """Raise unless ``train_rows`` were each scored once by a model that never saw them."""
    train = set(train_rows)
    methods = tuple(methods)
    fold_fits = {fid: keys for fid, keys in log.fits().items() if keys < train}
    if len(fold_fits) < 2:
        raise ProvenanceError(
            f"found {len(fold_fits)} fold model(s) fitted on a strict subset of the "
            f"{len(train)} training rows; out-of-fold scoring needs at least 2"
        )

    scored: Counter = Counter()
    for fid, fitted_on in fold_fits.items():
        rows = [k for c in log.calls(fid, methods) for k in c.keys if k in train]
        seen_own = fitted_on.intersection(rows)
        if seen_own:
            raise ProvenanceError(
                f"fit {fid} scored {len(seen_own)} row(s) it was fitted on "
                f"({_preview(seen_own)})"
            )
        scored.update(rows)

    missing = train - set(scored)
    repeated = {k for k, n in scored.items() if n > 1}
    if missing or repeated:
        raise ProvenanceError(
            f"out-of-fold coverage broken: {len(missing)} training row(s) never "
            f"scored ({_preview(missing)}), {len(repeated)} scored more than once "
            f"({_preview(repeated)})"
        )
