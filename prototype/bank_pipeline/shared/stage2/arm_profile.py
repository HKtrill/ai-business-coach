"""
shared.stage2.arm_profile
=========================
Versioned, fingerprinted arm configuration records (PR 37 Commit 4, Step 4D).

The shared operating contract (``shared.stage2.contract``) fixes the
obligations both routers must meet. Everything else that shapes a router's
selection is the arm's own configuration: GLASS's per-rule gates, rule counts,
novelty and complexity settings and its RF feature ranker; the RF Router's
forest hyperparameters and fold settings. Those were never part of the
contract, so before Step 4D they lived in notebooks, and the RF Router read its
hyperparameters from a GLASS-side artifact at run time.

An ``ArmProfile`` is one such configuration as a version-controlled JSON
record: the settings, where they came from (historical source identities,
derivation, exposure, limitations) and an approval. ``fingerprint()``
identifies it. Each arm turns its own profile into its own config type
(``glass_pipeline.glass_router.profile``, ``blackbox_pipeline.models.rf_router
.profile``); this module only reads and validates the record, and imports
neither arm.

A profile never names a test-derived field, and a frozen profile carries an
approval and no pre-release version label.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

from shared.stage2.contract import (
    STATUS_FROZEN,
    STATUS_WORKING,
    TEST_DERIVED_FIELD_TOKENS,
    ContractViolation,
    approval_confirmed,
)

__all__ = ["SCHEMA_VERSION", "ARMS", "ArmProfile", "load_arm_profile"]

SCHEMA_VERSION = "stage2-arm-profile/1"
ARMS = ("glass", "rf")

_PROVENANCE_KEYS = ("summary", "historical_sources", "derivation", "tuning", "exposure",
                    "limitations", "relation_to_pr38")
_SOURCE_KEYS = ("role", "path", "sha256", "fields")
_APPROVAL_KEYS = ("state", "approved_by", "approved_on", "note")
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SEGMENT_RE = re.compile(r"[a-z0-9]+")


def _text(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _texts(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(_text(x) for x in v)


def _freeze(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _thaw(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_thaw(v) for v in obj]
    return obj


def _exact(where: str, value: Any, keys: tuple, errors: list) -> bool:
    if not isinstance(value, Mapping):
        errors.append(f"{where} must be an object")
        return False
    missing = [k for k in keys if k not in value]
    unknown = [k for k in value if k not in keys]
    if missing:
        errors.append(f"{where} is missing {missing}")
    if unknown:
        errors.append(f"{where} has unknown keys {unknown}")
    return not missing


@dataclass(frozen=True)
class ArmProfile:
    """
    One arm's versioned configuration record.

    profile_id, profile_version, status
        Identity. ``status`` is ``"working"`` or ``"frozen"``; a frozen
        profile's version carries no pre-release label.
    arm
        ``"glass"`` or ``"rf"``.
    settings
        The arm's own settings; their shape is the arm's (validated when the
        arm builds its config from them). Stored read-only.
    provenance
        ``summary``, ``historical_sources`` (role, path, SHA-256 and the fields
        the values were copied from), ``derivation``, ``tuning``, ``exposure``,
        ``limitations`` and ``relation_to_pr38``.
    approval
        ``state`` (equal to ``status``), ``approved_by``, ``approved_on``,
        ``note``; a frozen profile needs the approver and the date.
    """

    profile_id: str
    schema_version: str
    arm: str
    profile_version: str
    status: str
    settings: Mapping[str, Any] = field(hash=False)
    provenance: Mapping[str, Any] = field(hash=False)
    approval: Mapping[str, Any] = field(hash=False)

    def __post_init__(self) -> None:
        errors: list = []
        if not _text(self.profile_id):
            errors.append("profile_id must be a non-empty string")
        if self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
        if self.arm not in ARMS:
            errors.append(f"arm must be one of {list(ARMS)}, got {self.arm!r}")
        if not (isinstance(self.profile_version, str)
                and _VERSION_RE.fullmatch(self.profile_version)):
            errors.append("profile_version must be MAJOR.MINOR.PATCH[-label]")
        if self.status not in (STATUS_WORKING, STATUS_FROZEN):
            errors.append(f"status must be {STATUS_WORKING!r} or {STATUS_FROZEN!r}")
        elif isinstance(self.profile_version, str):
            labelled = "-" in self.profile_version
            if self.status == STATUS_FROZEN and labelled:
                errors.append("a frozen profile's version must not carry a pre-release label")
            if self.status == STATUS_WORKING and not labelled:
                errors.append("a working profile's version must carry a pre-release label")
        if not isinstance(self.settings, Mapping) or not self.settings:
            errors.append("settings must be a non-empty object")
        self._validate_provenance(errors)
        self._validate_approval(errors)
        if errors:
            raise ContractViolation("ArmProfile validation errors:\n"
                                    + "\n".join(f"  - {e}" for e in errors))
        for name in ("settings", "provenance", "approval"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def _validate_provenance(self, errors: list) -> None:
        p = self.provenance
        if not _exact("provenance", p, _PROVENANCE_KEYS, errors):
            return
        for key in ("summary", "derivation", "tuning", "relation_to_pr38"):
            if not _text(p[key]):
                errors.append(f"provenance.{key} must be a non-empty string")
        for key in ("exposure", "limitations"):
            if not _texts(p[key]):
                errors.append(f"provenance.{key} must be a non-empty list of strings")
        sources = p["historical_sources"]
        if not isinstance(sources, (list, tuple)) or not sources:
            errors.append("provenance.historical_sources must be a non-empty list")
            return
        for i, s in enumerate(sources):
            where = f"provenance.historical_sources[{i}]"
            if not _exact(where, s, _SOURCE_KEYS, errors):
                continue
            for key in ("role", "path"):
                if not _text(s[key]):
                    errors.append(f"{where}.{key} must be a non-empty string")
            sha = s["sha256"]
            if sha is not None and not (isinstance(sha, str) and _SHA256_RE.fullmatch(sha)):
                errors.append(f"{where}.sha256 must be null or 64 lowercase hex characters")
            if not _texts(s["fields"]):
                errors.append(f"{where}.fields must be a non-empty list of strings")
                continue
            for name in s["fields"]:
                hit = set(_SEGMENT_RE.findall(name.lower())) & set(TEST_DERIVED_FIELD_TOKENS)
                if hit:
                    errors.append(f"{where}.fields names a test-derived field {name!r}: a "
                                  "profile may not be copied from test-derived values")

    def _validate_approval(self, errors: list) -> None:
        a = self.approval
        if not _exact("approval", a, _APPROVAL_KEYS, errors):
            return
        if a["state"] != self.status:
            errors.append(f"approval.state {a['state']!r} does not match status {self.status!r}")
        if not _text(a["note"]):
            errors.append("approval.note must be a non-empty string")
        if self.status == STATUS_FROZEN:
            if not _text(a["approved_by"]):
                errors.append("a frozen profile needs approval.approved_by")
            try:
                _dt.date.fromisoformat(a["approved_on"])
            except (TypeError, ValueError):
                errors.append("a frozen profile needs approval.approved_on as YYYY-MM-DD")
        elif a["approved_by"] is not None or a["approved_on"] is not None:
            errors.append("a working profile is not approved: approved_by and approved_on "
                          "must be null")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}

    def content_sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        """SHA-256 (16 hex) of the canonical JSON of the whole record."""
        return self.content_sha256()[:16]

    @property
    def approval_confirmed(self) -> bool:
        """A confirmed owner approval (``shared.stage2.contract.approval_confirmed``)."""
        return approval_confirmed(self.approval)

    def identity(self) -> dict:
        """What a protocol records to bind this profile."""
        return {"profile_id": self.profile_id, "profile_version": self.profile_version,
                "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ArmProfile":
        if not isinstance(data, Mapping):
            raise ContractViolation("an arm profile record must be an object")
        names = [f.name for f in fields(cls)]
        missing = [n for n in names if n not in data]
        unknown = [k for k in data if k not in names]
        if missing or unknown:
            raise ContractViolation(f"arm profile fields: missing {missing}, unknown {unknown}")
        return cls(**{n: data[n] for n in names})


def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ContractViolation(f"arm profile record repeats the key {k!r}")
        out[k] = v
    return out


def _no_constant(token):
    raise ContractViolation(f"arm profile record contains the non-finite number {token}")


def load_arm_profile(path, *, arm: str, expected_fingerprint: Optional[str] = None,
                     require_status: Optional[str] = None,
                     require_confirmed_approval: bool = False) -> ArmProfile:
    """
    Load one arm profile record. ``arm`` must match the record; an expected
    fingerprint, a required status and a confirmed owner approval are checked
    when asked for (PR 38 asks for all three).
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys,
                          parse_constant=_no_constant)
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"{path}: not valid JSON ({exc})") from exc
    profile = ArmProfile.from_dict(data)
    if profile.arm != arm:
        raise ContractViolation(f"{path}: a {profile.arm!r} profile, not the {arm!r} arm's")
    if require_status is not None and profile.status != require_status:
        raise ContractViolation(f"{path}: profile status is {profile.status!r}, "
                                f"{require_status!r} is required")
    if expected_fingerprint is not None and profile.fingerprint() != expected_fingerprint:
        raise ContractViolation(f"{path}: profile fingerprint {profile.fingerprint()} does not "
                                f"match the expected {expected_fingerprint}")
    if require_confirmed_approval and not profile.approval_confirmed:
        raise ContractViolation(f"{path}: the {profile.arm} profile is a review candidate: its "
                                "owner approval is unconfirmed, so it cannot govern PR 38")
    return profile
