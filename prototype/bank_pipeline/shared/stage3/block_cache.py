"""
shared.stage3.block_cache
=========================
The GLASS notebook's cached ``Stage3Block`` (PR 36 audit F23): reuse a block
only when it is provably the block that would be built now; otherwise rebuild
it and replace the cache.

    block, status = load_or_build_block(GLOBAL_SPLIT, EBMFeaturePipeline, config,
                                        split_fingerprint=fp, cache_path=path)

Before F23 the cache was keyed by the split fingerprint alone and checked
against the config's fold settings, so a block built by older code (a pre-F22
block without inner folds, older feature engineering) was reloaded silently
and failed later, or not at all, and had to be deleted by hand.

Identity (``block_identity``) — everything the block's content depends on that
is known before building it:

    data             the raw X / y frames handed to the feature pipeline
                     (values, dtypes, columns, row index and order)
    split            the split fingerprint the block is stamped with
    build            the Stage3Config settings ``Stage3Block.build`` reads
                     (fold counts, seed, stratification, feature fit scope,
                     weighting, hygiene, expected features)
    pipeline         the feature pipeline factory, by qualified name
    implementation   a SHA-256 of the source of ``Stage3Block``'s module, the
                     pipeline's module and every project module they import,
                     transitively (import statements, so imported constants
                     such as a seed count too)
    libraries        Python, numpy, pandas, scikit-learn and SciPy versions
    schema           this cache format's version

A cache is reused only if it is a payload of this format whose identity equals
the current one AND its block still passes the block's own contracts: inner
folds present (F22), content equal to its recorded fingerprint, settings equal
to the config's, the same split fingerprint, rows and features. Anything else
— no file, an unreadable or foreign file, a bare pickled block (the pre-F23
format), missing, other or malformed metadata, a failed contract, a block too
malformed to check — rebuilds the block and writes it atomically (temporary
file in the same folder, then ``os.replace``), so a crash never leaves a
partial cache behind. Malformed cache contents are a reason to rebuild, not
an error; errors from building the replacement propagate unchanged.

Performance trade-off: the content fingerprint of a block can only be known by
building it, so the key is a description of the inputs and the code, not of
the output. Checking a hit costs hashing the raw split, reading the source
files and re-hashing the cached block (seconds) instead of rebuilding it
(41 pipeline fits at the notebook's fold counts). The implementation hash is
deliberately conservative: any edit to a reachable module, including a
comment, triggers one rebuild.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd

from .data import Stage3Block, _hash_frame

__all__ = ["BLOCK_CACHE_SCHEMA", "block_identity", "implementation_identity", "load_or_build_block"]

#: Version of the payload written by ``load_or_build_block``.
BLOCK_CACHE_SCHEMA = "stage3_block_cache/1"

#: Top-level packages of this repository whose modules can change a block.
PROJECT_PACKAGES = ("shared", "data", "glass_pipeline", "blackbox_pipeline", "feature_research")

#: The Stage3Config fields ``Stage3Block.from_global_split`` passes to ``build``.
BUILD_FIELDS = ("feature_fit_scope", "n_tune_folds", "n_eval_folds", "random_state", "stratify",
                "class_weight", "clean_inf", "impute_missing", "expected_features")


# ======================================================================
# identity
# ======================================================================
def _data_digest(GLOBAL_SPLIT: dict) -> str:
    h = hashlib.sha256()
    for key in ("X_train", "X_test", "y_train", "y_test"):
        obj = GLOBAL_SPLIT[key]
        frame = obj if isinstance(obj, pd.DataFrame) else pd.Series(obj).to_frame()
        h.update(key.encode())
        h.update("|".join(f"{c}:{t}" for c, t in frame.dtypes.items()).encode())
        _hash_frame(h, frame)
    return h.hexdigest()


def _is_project(name: str) -> bool:
    return name.split(".")[0] in PROJECT_PACKAGES


def _module_source(name: str) -> Optional[Path]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
    origin = getattr(spec, "origin", None) if spec is not None else None
    return Path(origin) if origin and origin.endswith(".py") else None


def _imported_modules(name: str, source: Path) -> set:
    """Project modules ``name``'s import statements bring in (any scope)."""
    spec = importlib.util.find_spec(name)
    package = name if spec.submodule_search_locations else name.rpartition(".")[0]
    found = set()
    for node in ast.walk(ast.parse(source.read_bytes(), filename=str(source))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                    if node.level else node.module)
            for alias in node.names:
                sub = f"{base}.{alias.name}"
                found.add(sub if _module_source(sub) is not None else base)
    return {m for m in found if _is_project(m)}


def _module_digest(name: str, source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


def implementation_identity(*roots: Any) -> dict:
    """
    ``{module: sha256 of its source}`` for the modules defining ``roots``
    (classes or functions) and every project module they import, transitively.
    """
    todo = [r.__module__ for r in roots]
    out: dict = {}
    while todo:
        name = todo.pop()
        if name in out:
            continue
        source = _module_source(name)
        if source is None:
            out[name] = None                 # no Python source to hash (e.g. builtin)
            continue
        out[name] = _module_digest(name, source)
        todo.extend(sorted(_imported_modules(name, source) - set(out)))
    return dict(sorted(out.items()))


def _libraries() -> dict:
    import scipy
    import sklearn
    return {"python": ".".join(platform.python_version_tuple()[:2]), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def block_identity(GLOBAL_SPLIT: dict, pipeline_factory: Callable[[], Any], config,
                   split_fingerprint: Optional[str]) -> dict:
    """What a block built from these inputs is, before building it (module docstring)."""
    build = {f: getattr(config, f) for f in BUILD_FIELDS}
    if build["expected_features"] is not None:
        build["expected_features"] = list(build["expected_features"])
    return {
        "schema": BLOCK_CACHE_SCHEMA,
        "data": _data_digest(GLOBAL_SPLIT),
        "split": split_fingerprint,
        "build": build,
        "pipeline": (f"{pipeline_factory.__module__}."
                     f"{getattr(pipeline_factory, '__qualname__', type(pipeline_factory).__qualname__)}"),
        "implementation": implementation_identity(Stage3Block, pipeline_factory),
        "libraries": _libraries(),
    }


# ======================================================================
# validation, rebuild, atomic write
# ======================================================================
def _same(stored, current) -> bool:
    """``stored == current`` for a value read from a cache file: one that cannot be compared differs."""
    try:
        return bool(stored == current)
    except Exception:                        # noqa: BLE001 — e.g. an array where a mapping belongs
        return False


def _malformed_implementation(stored: dict) -> Optional[str]:
    """Why the stored implementation identity is not ``{module name: digest or None}``, if it is not."""
    value = stored.get("implementation")
    if not isinstance(value, dict):          # NoneType when the key is missing
        return f"implementation is {type(value).__name__}, not a mapping"
    if not all(isinstance(m, str) and (d is None or isinstance(d, str)) for m, d in value.items()):
        return "implementation does not map module names to digests"
    return None


def _difference(stored: dict, current: dict) -> str:
    """'' when the stored identity is the current one, else what differs (``stored`` well-formed)."""
    changed = [k for k in current if k not in stored or not _same(stored[k], current[k])]
    if "implementation" in changed:
        old, new = stored["implementation"], current["implementation"]
        modules = sorted(m for m in set(old) | set(new) if old.get(m) != new.get(m))
        changed[changed.index("implementation")] = f"implementation ({', '.join(modules)})"
    extra = sorted(repr(k) for k in stored if k not in current)
    if extra:
        changed.append(f"unexpected keys {', '.join(extra)}")
    return ", ".join(changed)


def _contract_failure(block, GLOBAL_SPLIT: dict, config, split_fingerprint) -> Optional[str]:
    """
    None when a cached block passes the block's own contracts; otherwise which
    failed. Only ever given a block read from the cache, so a block too
    malformed to check (an attribute missing or of the wrong type, inner folds
    included) is a reason to rebuild, never an error. The replacement is built
    after this returns, outside it, so errors from building are not caught.
    """
    try:
        try:
            block.require_inner_folds()
        except ValueError:
            return "missing or incomplete inner folds (pre-F22 block)"
        if not _same(block.compute_fingerprint(), block.block_fingerprint):
            return "block content differs from its fingerprint"
        try:
            block.check_config(config)
        except ValueError:
            return "block settings differ from the config"
        if not _same(block.split_fingerprint, split_fingerprint):
            return "split fingerprint differs"
        if not (block.index_train.equals(GLOBAL_SPLIT["X_train"].index)
                and block.index_test.equals(GLOBAL_SPLIT["X_test"].index)):
            return "rows differ from GLOBAL_SPLIT"
    except Exception as exc:                 # noqa: BLE001 — a malformed cached block is stale
        return f"malformed block ({type(exc).__name__})"
    return None


def _cached_block(path: Path, identity: dict, GLOBAL_SPLIT: dict, config,
                  split_fingerprint: Optional[str]) -> tuple:
    """``(block, None)`` when the cache at ``path`` may be reused, else ``(None, why not)``."""
    if not path.exists():
        return None, "no cache"
    try:
        payload = joblib.load(path)
    except Exception as exc:                 # noqa: BLE001 — any unreadable file is stale
        return None, f"unreadable cache ({type(exc).__name__})"
    if not isinstance(payload, dict) or "identity" not in payload or "block" not in payload:
        return None, f"no cache metadata ({type(payload).__name__}; written before F23)"
    stored = payload["identity"]
    if not isinstance(stored, dict) or not _same(stored.get("schema"), BLOCK_CACHE_SCHEMA):
        schema = stored.get("schema") if isinstance(stored, dict) else stored
        return None, f"cache schema {schema!r} is not {BLOCK_CACHE_SCHEMA!r}"
    malformed = _malformed_implementation(stored)
    if malformed:
        return None, f"malformed cache metadata ({malformed})"
    changed = _difference(stored, identity)
    if changed:
        return None, f"changed: {changed}"
    block = payload["block"]
    if not isinstance(block, Stage3Block):
        return None, f"payload holds {type(block).__name__}, not a Stage3Block"
    failure = _contract_failure(block, GLOBAL_SPLIT, config, split_fingerprint)
    return (None, failure) if failure else (block, None)


def _write_atomically(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def load_or_build_block(GLOBAL_SPLIT: dict, pipeline_factory: Callable[[], Any], config, *,
                        split_fingerprint: Optional[str], cache_path) -> tuple:
    """
    ``(Stage3Block, status)``: the cached block if it is the one that would be
    built now (status ``"reused"``), else a block built with
    ``Stage3Block.from_global_split`` and saved to ``cache_path`` with its
    identity (status ``"built (no cache)"`` or ``"rebuilt (<why>)"``).
    """
    path = Path(cache_path)
    identity = block_identity(GLOBAL_SPLIT, pipeline_factory, config, split_fingerprint)
    block, reason = _cached_block(path, identity, GLOBAL_SPLIT, config, split_fingerprint)
    if block is not None:
        return block, "reused"
    block = Stage3Block.from_global_split(GLOBAL_SPLIT, pipeline_factory, config,
                                          split_fingerprint=split_fingerprint)
    _write_atomically({"identity": identity, "block": block}, path)
    return block, "built (no cache)" if reason == "no cache" else f"rebuilt ({reason})"
