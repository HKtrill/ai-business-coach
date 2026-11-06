"""
shared.artifact_versions
========================
Immutable run versions plus a mutable "latest" alias for artifacts that used to
be saved under one fixed name (PR 37).

Before PR 37, ``save_rf_router`` and ``save_stage1_mlp`` wrote
``<stem>.joblib`` (and its JSON/CSV sidecars) in place, so every notebook run
replaced the previous model and the earlier version was lost. Each save now
writes two kinds of file:

* **Immutable version** - ``<stem>_<YYYYMMDD_HHMMSS_ffffff>.joblib`` (UTC) and
  its sidecars ``<stem>_<id>_metadata.json`` and ``<stem>_<id><table>.csv``.
  Created exclusively (a hard link from a temporary file, or an exclusive
  create where hard links are unavailable), so an existing file is never
  replaced; then marked read-only. The metadata records the version's file
  name, version id and SHA-256 (``artifact_file``).
* **Mutable latest alias** - ``<stem>.joblib`` and its sidecars, byte copies of
  the newest version, replaced atomically on every save so existing readers of
  the fixed name keep working. The alias metadata says so explicitly
  (``latest_alias``: ``mutable``, ``points_to``, ``sha256``). An alias is a
  convenience for "the newest"; it is never evidence of which model a later
  stage consumed. ``alias_target`` maps an alias to the immutable version it
  currently copies.

A fixed-name file written before this module existed has no immutable twin.
Before an alias is replaced, such a file and its sidecars are kept as
``<stem>_preserved_<sha256[:12]>...`` (immutable, read-only), so the first run
after the change loses nothing either.

Model payloads are written by the caller's own code and are not changed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

__all__ = ["ArtifactExistsError", "VersionedArtifact", "save_versioned", "alias_target",
           "file_sha256", "new_version_id"]

MODEL_SUFFIX = ".joblib"
METADATA_SUFFIX = "_metadata.json"
PRESERVED = "preserved"
_READ_ONLY = stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH


class ArtifactExistsError(FileExistsError):
    """An immutable artifact file already exists; it is never overwritten."""


@dataclass(frozen=True)
class VersionedArtifact:
    """Result of ``save_versioned``."""

    path: Path                 #: the immutable version (what consumers should record)
    alias: Path                #: the mutable latest alias (``<stem>.joblib``)
    version_id: str
    sha256: str
    preserved: tuple = ()      #: files kept from a pre-existing unversioned alias


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


# ----------------------------------------------------------------------
def _tmp(folder: Path, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(dir=str(folder), prefix=".tmp-", suffix=suffix)
    os.close(fd)
    return Path(name)


def _commit_immutable(tmp: Path, final: Path, times=None) -> None:
    """Move ``tmp`` to ``final`` only if ``final`` does not exist; mark it read-only."""
    try:
        try:
            os.link(tmp, final)
        except FileExistsError:
            raise ArtifactExistsError(
                f"{final} already exists; immutable artifact versions are never "
                "overwritten") from None
        except OSError:
            # No hard links on this filesystem: exclusive create, never replace.
            try:
                dst = open(final, "xb")
            except FileExistsError:
                raise ArtifactExistsError(
                    f"{final} already exists; immutable artifact versions are never "
                    "overwritten") from None
            try:
                with dst, open(tmp, "rb") as src:
                    shutil.copyfileobj(src, dst)
            except BaseException:
                final.unlink()
                raise
    finally:
        if tmp.exists():
            tmp.unlink()
    if times is not None:
        os.utime(final, times)
    os.chmod(final, _READ_ONLY)


def _write_immutable(final: Path, write: Callable[[Path], None], times=None) -> None:
    tmp = _tmp(final.parent, final.suffix)
    try:
        write(tmp)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    _commit_immutable(tmp, final, times)


def _copy_immutable(src: Path, final: Path) -> None:
    """
    Keep a byte copy of ``src`` at ``final`` with ``src``'s modification time
    (so "newest" stays truthful); an identical existing copy is accepted.
    """
    if final.exists():
        if file_sha256(final) == file_sha256(src):
            return
        raise ArtifactExistsError(f"{final} already exists with different content; it is "
                                  "never overwritten")
    st = src.stat()
    _write_immutable(final, lambda tmp: shutil.copyfile(src, tmp),
                     times=(st.st_atime, st.st_mtime))


def _replace_alias(src: Path, alias: Path) -> None:
    """Atomically make ``alias`` a byte copy of ``src`` (the alias stays writable)."""
    tmp = _tmp(alias.parent, alias.suffix)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, alias)
    finally:
        if tmp.exists():
            tmp.unlink()


def _alias_record(folder: Path, stem: str) -> Optional[dict]:
    try:
        meta = json.loads((folder / f"{stem}{METADATA_SUFFIX}").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    record = meta.get("latest_alias") if isinstance(meta, dict) else None
    return record if isinstance(record, dict) else None


def alias_target(alias) -> Optional[Path]:
    """
    The immutable version the alias ``<stem>.joblib`` currently copies, or None
    when the alias is unversioned (written before PR 37) or no longer matches
    the version its metadata names.
    """
    alias = Path(alias)
    if not alias.is_file() or alias.suffix != MODEL_SUFFIX:
        return None
    record = _alias_record(alias.parent, alias.stem)
    if not record or not record.get("points_to"):
        return None
    target = alias.parent / str(record["points_to"])
    if target.name != Path(str(record["points_to"])).name or not target.is_file():
        return None
    digest = file_sha256(alias)
    if digest != record.get("sha256") or file_sha256(target) != digest:
        return None
    return target


def _preserve_unversioned(folder: Path, stem: str, companions: Sequence[str]) -> tuple:
    """Keep a pre-existing alias that has no immutable twin, with its sidecars."""
    alias = folder / f"{stem}{MODEL_SUFFIX}"
    if not alias.is_file() or alias_target(alias) is not None:
        return ()
    keep = f"{stem}_{PRESERVED}_{file_sha256(alias)[:12]}"
    kept = []
    for suffix in (MODEL_SUFFIX, METADATA_SUFFIX, *companions):
        src = folder / f"{stem}{suffix}"
        if src.is_file():
            dst = folder / f"{keep}{suffix}"
            _copy_immutable(src, dst)
            kept.append(dst)
    return tuple(kept)


# ----------------------------------------------------------------------
def save_versioned(
    output_dir,
    stem: str,
    *,
    write_model: Callable[[Path], None],
    metadata: Mapping,
    dump_metadata: Callable[[dict], str],
    write_metadata: Optional[Callable[[Path, str], None]] = None,
    tables: Optional[Mapping[str, Callable[[Path], None]]] = None,
    companions: Sequence[str] = (),
) -> VersionedArtifact:
    """
    Save one immutable version of ``stem`` and point the latest alias at it.

    Parameters
    ----------
    write_model
        ``write_model(path)`` writes the model file (e.g. ``joblib.dump``).
    metadata, dump_metadata, write_metadata
        The JSON sidecar: ``dump_metadata(dict) -> str`` and
        ``write_metadata(path, text)`` (default ``Path.write_text``), as the
        caller wrote it before.
    tables
        ``{suffix: write(path)}`` for the CSV sidecars written by this save,
        e.g. ``{"_pass1_sweep.csv": lambda p: sweep.to_csv(p, index=False)}``.
    companions
        Every sidecar suffix this artifact can have; used to keep a
        pre-existing unversioned alias complete.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = dict(tables or {})
    write_metadata = write_metadata or (lambda path, text: Path(path).write_text(text))
    preserved = _preserve_unversioned(out, stem, tuple(dict.fromkeys([*companions, *tables])))

    for _ in range(100):
        version_id = new_version_id()
        vstem = f"{stem}_{version_id}"
        if not any((out / f"{vstem}{s}").exists()
                   for s in (MODEL_SUFFIX, METADATA_SUFFIX, *tables)):
            break
    else:  # pragma: no cover - the clock would have to stand still
        raise ArtifactExistsError(f"no free version id for {stem} in {out}")

    version = out / f"{vstem}{MODEL_SUFFIX}"
    _write_immutable(version, write_model)
    digest = file_sha256(version)
    identity = {"name": version.name, "version_id": version_id, "sha256": digest,
                "immutable": True}

    meta = {**dict(metadata), "artifact_file": identity}
    _write_immutable(out / f"{vstem}{METADATA_SUFFIX}",
                     lambda tmp: write_metadata(tmp, dump_metadata(meta)))
    for suffix, write in tables.items():
        _write_immutable(out / f"{vstem}{suffix}", write)

    alias = out / f"{stem}{MODEL_SUFFIX}"
    _replace_alias(version, alias)
    alias_meta = {**meta, "latest_alias": {
        "mutable": True, "name": alias.name, "points_to": version.name, "sha256": digest,
        "note": "a copy of the newest version, replaced by every save; record the "
                "immutable version's path, never this alias"}}
    tmp = _tmp(out, ".json")
    try:
        write_metadata(tmp, dump_metadata(alias_meta))
        os.replace(tmp, out / f"{stem}{METADATA_SUFFIX}")
    finally:
        if tmp.exists():
            tmp.unlink()
    for suffix in tables:
        _replace_alias(out / f"{vstem}{suffix}", out / f"{stem}{suffix}")
    return VersionedArtifact(path=version, alias=alias, version_id=version_id, sha256=digest,
                             preserved=preserved)
