"""
blackbox_pipeline.models.rf_router.tuning.provenance
=====================================================
What a cached tuning output was computed FROM, and the rules for reusing it.

A cached result is reusable only when everything that could change it is
identical. The split fingerprint alone is not enough: a feature-engineering
change (new bins, a reordered or recast column, a different feature block)
leaves the train/test row split untouched but makes every tuned
hyperparameter and every OOF score meaningless. So every cache key carries:

* ``TUNING_CONTRACT_VERSION`` — bumped whenever an objective, the OOF protocol,
  the search space or the structure of a cached value changes;
* ``data_fingerprint`` — row count, the ORDERED feature schema with dtypes,
  and content digests of the row index, the feature values and the labels,
  plus caller-supplied provenance (e.g. which feature block produced X);
* the split fingerprint, budget signature, constraints and — for Pass 2 runs —
  the exact remainder and the Pass 1 parameters that produced it.

Reuse rules (``cached``)
------------------------
* Each key gets its own file, ``<stem>-<key digest>.joblib``. Changed inputs map
  to a different file, so they are recomputed — never served a result computed
  from other inputs. Results for other inputs are listed, not reused.
* A file found under the right name is still checked: its stored key must equal
  the current key, and the optional ``validate`` callback must accept the value.
  Any failure — unreadable file, foreign structure, key mismatch, validator
  rejection — raises ``IncompatibleCacheError``. Nothing is overwritten
  silently; delete the file or pass ``refresh=True`` after inspecting it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from ..oof.folds import check_aligned

__all__ = [
    "TUNING_CONTRACT_VERSION",
    "IncompatibleCacheError",
    "data_fingerprint",
    "cache_key",
    "cached",
    "frame_validator",
]

#: Bump on ANY change to objective definitions, the trial OOF protocol, the
#: search space, ``evaluate_params`` outputs or the structure of cached values.
#: Every cache file written under an older version is then unreachable.
TUNING_CONTRACT_VERSION = "pr35-stage2-tuning-v2"


class IncompatibleCacheError(RuntimeError):
    """A cache file exists for this key but cannot be trusted as its result."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serialisable in a cache key: {type(o).__name__}")


def _normalise(obj: Any) -> Any:
    """Canonical JSON form, so tuples/lists and numpy scalars compare equal."""
    return json.loads(json.dumps(obj, sort_keys=True, default=_json_default))


def data_fingerprint(X: pd.DataFrame, y, *, provenance: Optional[Mapping] = None) -> dict:
    """
    Identity of a training feature block and its labels.

    ``content_digest`` covers row count, ordered schema (names and dtypes), the
    row index, every feature value and every label. ``digest`` additionally
    covers ``provenance`` — free-form, caller-supplied facts about where X came
    from that its content cannot reveal.
    """
    X, y = check_aligned(X, y)
    fp = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "schema": [[str(c), str(t)] for c, t in zip(X.columns, X.dtypes)],
        "index": _sha(pd.util.hash_pandas_object(X.index, index=False).to_numpy().tobytes()),
        "values": _sha(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes()),
        "labels": _sha(np.asarray(y).astype(np.int64).tobytes()),
    }
    fp["content_digest"] = _sha(json.dumps(fp, sort_keys=True).encode())
    fp["provenance"] = _normalise(dict(provenance or {}))
    fp["digest"] = _sha(json.dumps([fp["content_digest"], fp["provenance"]],
                                   sort_keys=True).encode())
    return fp


def cache_key(kind: str, *, data_fp: Mapping, split_fp: str, **parts) -> dict:
    """The standard key: contract version + data identity + split + ``parts``."""
    return _normalise({
        "contract": TUNING_CONTRACT_VERSION,
        "kind": kind,
        "split_fp": split_fp,
        "data": dict(data_fp),
        **parts,
    })


def _digest(key: Mapping) -> str:
    return _sha(json.dumps(_normalise(key), sort_keys=True).encode())[:12]


def _differing_fields(a: Mapping, b: Mapping) -> list:
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        if a.get(k) != b.get(k):
            if isinstance(a.get(k), Mapping) and isinstance(b.get(k), Mapping):
                out += [f"{k}.{sub}" for sub in _differing_fields(a[k], b[k])]
            else:
                out.append(k)
    return out


def cached(
    cache_dir: str | Path,
    stem: str,
    key: Mapping,
    compute: Callable[[], Any],
    *,
    validate: Optional[Callable[[Any], None]] = None,
    refresh: bool = False,
    verbose: bool = True,
):
    """
    Return the cached value for ``key``, or compute, validate and store it.

    See the module docstring for the reuse rules. ``validate(value)`` must raise
    ``IncompatibleCacheError`` when ``value`` does not match the current inputs;
    it runs on loaded AND freshly computed values.
    """
    key = _normalise(key)
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{stem}-{_digest(key)}.joblib"

    if path.exists() and not refresh:
        try:
            blob = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 — any unreadable file is incompatible
            raise IncompatibleCacheError(
                f"{path.name} is unreadable ({type(exc).__name__}). Inspect it, then "
                "delete it or pass refresh=True."
            ) from exc
        if not (isinstance(blob, dict) and {"key", "value"} <= set(blob)):
            raise IncompatibleCacheError(f"{path.name} is not a PR 35 cache file")
        stored = _normalise(blob["key"])
        if stored != key:
            raise IncompatibleCacheError(
                f"{path.name} was written for different inputs "
                f"(differs in: {', '.join(_differing_fields(stored, key)) or '?'})."
            )
        if validate is not None:
            validate(blob["value"])
        if verbose:
            print(f"  ↺ cached  {path.name}")
        return blob["value"]

    if verbose and not refresh:
        others = sorted(p for p in cache_dir.glob(f"{stem}-*.joblib") if p != path)
        if others:
            note = ""
            try:
                prev = _normalise(joblib.load(others[-1]).get("key", {}))
                note = f" (latest differs in: {', '.join(_differing_fields(prev, key))})"
            except Exception:  # noqa: BLE001 — the note is best-effort
                pass
            print(f"  ⓘ {len(others)} cached {stem} result(s) exist for other inputs — "
                  f"not reused{note}")
        print(f"  ⧗ computing {path.name}")

    value = compute()
    if validate is not None:
        validate(value)
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"key": key, "value": value,
                 "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                path, compress=3)
    return value


def frame_validator(columns: Iterable[str], n_rows: Optional[int] = None):
    """Validator for cached DataFrames: required columns and, optionally, row count."""
    required = list(columns)

    def validate(value):
        if not isinstance(value, pd.DataFrame):
            raise IncompatibleCacheError(f"expected a DataFrame, found {type(value).__name__}")
        missing = [c for c in required if c not in value.columns]
        if missing:
            raise IncompatibleCacheError(f"cached frame lacks columns {missing}")
        if n_rows is not None and len(value) != n_rows:
            raise IncompatibleCacheError(f"cached frame has {len(value)} rows, expected {n_rows}")

    return validate
