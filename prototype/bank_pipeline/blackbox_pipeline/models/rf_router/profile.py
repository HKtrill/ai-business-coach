"""
blackbox_pipeline.models.rf_router.profile
==========================================
The RF Router's own frozen configuration for the PR 38 primary comparison
(PR 37 Commit 4, Step 4D; decision M8).

Historically the black-box notebook read the RF hyperparameters from the
newest GLASS-arm RF artifact at run time (``tuning.legacy``,
``RF_PARAM_SOURCE='artifact'``). The values themselves are legitimate to share
(F17 record, section 5); the run-time read was a cross-arm file lookup. The
primary pathway now reads one versioned record owned by this arm,
``blackbox_pipeline/profiles/rf_router_primary_v1.json``, whose values equal the
historical ones
and whose provenance names the artifacts they were copied from. The historical
pathway (``tuning.legacy``) is unchanged, so historical runs still reproduce.

PR 38 calls ``rf_primary_config()``. It takes no record from the caller: it
loads the frozen contract and profile strictly (status, pinned fingerprint,
confirmed approval) and applies the contract's constraints
(``rf_constraints_from_contract_v2``, ``check_rf_config_follows_contract``).
``check_rf_primary_config`` refuses any config that is not exactly the primary
one (run-time options aside), so a configuration built by hand or from another
profile is never treated as the PR 38 primary. ``rf_config_from_profile``
builds a config from given records for reviews and tests only. Nothing here
opens an artifact.
"""

from __future__ import annotations

from pathlib import Path

from shared.stage2.arm_profile import ArmProfile, load_arm_profile
from shared.stage2.contract import (
    STATUS_FROZEN,
    ContractViolation,
    SharedOperatingContractV2,
    load_frozen_contract,
)

from .config import RFRouterConfig, validate_rf_params
from .contract import (
    RF_F17_PRIMARY,
    check_rf_config_follows_contract,
    rf_constraints_from_contract_v2,
)

__all__ = ["PROFILES_DIR", "PRIMARY_PROFILE_FINGERPRINT", "PROFILE_FIELDS", "RUNTIME_OPTIONS",
           "primary_profile_path", "load_rf_primary_profile", "rf_config_from_profile",
           "rf_primary_config", "check_rf_primary_config"]

#: ``blackbox_pipeline/profiles``: version-controlled. Not under ``models/``, whose
#: JSON files are Git-ignored as model artifacts (.gitignore ``**/models/**/*.json``).
PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"
_PRIMARY = "rf_router_primary_v1.json"
#: Pinned identity of the frozen profile (PR 37 Commit 4, Step 4D).
PRIMARY_PROFILE_FINGERPRINT = "4e3f9c07e6ecc8ad"

#: The RFRouterConfig fields a profile fixes. ``constraints`` come only from the
#: shared contract; ``RUNTIME_OPTIONS`` never change a selection.
PROFILE_FIELDS = ("mode", "rf_params", "pass2_rf_params", "n_oof_folds", "random_state",
                  "stratify_oof", "remainder_min_size", "n_inner_folds", "eligibility",
                  "eligibility_folds")
RUNTIME_OPTIONS = ("verbose", "validate_nested")


def primary_profile_path() -> Path:
    return PROFILES_DIR / _PRIMARY


def load_rf_primary_profile(path=None) -> ArmProfile:
    """
    The frozen RF profile, refused unless its status is ``"frozen"``, its
    fingerprint is the pinned one and its owner approval is confirmed.
    """
    return load_arm_profile(path or primary_profile_path(), arm="rf",
                            expected_fingerprint=PRIMARY_PROFILE_FINGERPRINT,
                            require_status=STATUS_FROZEN, require_confirmed_approval=True)


def _router_settings(profile: ArmProfile) -> dict:
    if not isinstance(profile, ArmProfile) or profile.arm != "rf":
        raise ContractViolation("expected the RF arm's ArmProfile")
    if set(profile.settings) != {"router_config"}:
        raise ContractViolation("an RF profile's settings hold exactly 'router_config'")
    cfg = dict(profile.settings["router_config"])
    missing = [k for k in PROFILE_FIELDS if k not in cfg]
    unknown = [k for k in cfg if k not in PROFILE_FIELDS]
    if missing or unknown:
        raise ContractViolation(f"RF profile router_config: missing {missing}, unknown {unknown} "
                                "(constraints come only from the contract)")
    problems = [f"{name}={cfg[name]!r}, the F17 primary requires {want!r}"
                for name, want in RF_F17_PRIMARY if cfg[name] != want]
    if problems:
        raise ContractViolation("RF profile is not the F17 primary:\n"
                                + "\n".join(f"  - {p}" for p in problems))
    cfg["rf_params"] = validate_rf_params(dict(cfg["rf_params"]), "profile rf_params")
    if cfg["pass2_rf_params"] is not None:
        cfg["pass2_rf_params"] = validate_rf_params(dict(cfg["pass2_rf_params"]),
                                                    "profile pass2_rf_params")
    return cfg


def rf_config_from_profile(contract: SharedOperatingContractV2, profile: ArmProfile, *,
                           verbose: bool = False, validate_nested: bool = False) -> RFRouterConfig:
    """
    ``profile`` with ``contract``'s constraints, for reviews and tests. Never
    the PR 38 primary config: that is ``rf_primary_config()``, and
    ``check_rf_primary_config`` refuses anything else.
    """
    cfg = _router_settings(profile)
    config = RFRouterConfig(**cfg, constraints=rf_constraints_from_contract_v2(contract),
                            verbose=bool(verbose), validate_nested=bool(validate_nested))
    check_rf_config_follows_contract(config, contract)
    return config


def rf_primary_config(*, verbose: bool = False, validate_nested: bool = False) -> RFRouterConfig:
    """
    The PR 38 primary RF Router config. Takes no record: the frozen contract
    (``load_frozen_contract``) and the frozen profile
    (``load_rf_primary_profile``) are loaded strictly. ``verbose`` and
    ``validate_nested`` are run-time options; neither changes a selection.
    """
    return rf_config_from_profile(load_frozen_contract(), load_rf_primary_profile(),
                                  verbose=verbose, validate_nested=validate_nested)


def check_rf_primary_config(config: RFRouterConfig) -> None:
    """Refuse a config that is not exactly ``rf_primary_config()`` (run-time options aside)."""
    if not isinstance(config, RFRouterConfig):
        raise ContractViolation(f"expected an RFRouterConfig, got {type(config).__name__}")
    want, got = rf_primary_config().to_dict(), config.to_dict()
    differ = sorted(k for k in want if k not in RUNTIME_OPTIONS and got.get(k) != want[k])
    if differ:
        raise ContractViolation("not the frozen PR 38 primary RF Router config "
                                f"(rf_primary_config()): {differ} differ")
