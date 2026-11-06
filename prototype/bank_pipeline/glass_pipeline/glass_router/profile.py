"""
glass_pipeline.glass_router.profile
===================================
The GLASS Router's own frozen configuration for the PR 38 primary comparison
(PR 37 Commit 4, Step 4D).

The shared contract fixes the union obligations; everything else in a
``GlassRouterConfig`` (per-rule gates, rule counts, depth-2 pruning, overlap,
novelty, complexity, diversity) and the RF feature ranker are the arm's own
settings. They shape GLASS coverage, so Step 4B required them fixed with the
frozen contract before PR 38 results. ``profiles/glass_router_primary_v1.json``
records them: the historical legacy profile, union obligations delegated to the
contract, the F17 primary settings.

PR 38 calls ``glass_primary_config()`` and ``ranker_settings()``. Neither takes
a record from the caller: both load the frozen contract and profile strictly
(status, pinned fingerprint, confirmed approval). ``check_glass_primary_config``
refuses any config that is not exactly the primary one, so a configuration
built by hand or from another profile is never treated as the PR 38 primary;
``check_glass_primary_ranker`` does the same for the ranker settings passed to
the cross-fit and the full-train ranker factory (Commit 5).
``glass_config_from_profile`` builds a config from given records for reviews
and tests only. Nothing here opens an artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import numpy as np

from shared.stage2.arm_profile import ArmProfile, load_arm_profile
from shared.stage2.contract import (
    STATUS_FROZEN,
    ContractViolation,
    SharedOperatingContractV2,
    load_frozen_contract,
)

from .contract import (
    GLASS_F17_PRIMARY,
    UNION_OBLIGATION_FIELDS,
    check_glass_config_follows_contract,
    glass_config_from_contract,
)
from .core.config import GlassRouterConfig

__all__ = ["PROFILES_DIR", "PRIMARY_PROFILE_FINGERPRINT", "PROFILE_FIELDS", "RANKER_PARAM_KEYS",
           "primary_profile_path", "load_glass_primary_profile", "glass_base_config",
           "glass_config_from_profile", "ranker_settings_from_profile", "glass_primary_config",
           "ranker_settings", "check_glass_primary_config", "check_glass_primary_ranker"]

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
_PRIMARY = "glass_router_primary_v1.json"
#: Pinned identity of the frozen profile (PR 37 Commit 4, Step 4D).
PRIMARY_PROFILE_FINGERPRINT = "198731d5cdf263f4"

#: The GlassRouterConfig fields a profile fixes: all but the union obligations
#: and the contract provenance, which only the shared contract supplies.
PROFILE_FIELDS = tuple(f.name for f in fields(GlassRouterConfig)
                       if f.name not in UNION_OBLIGATION_FIELDS + ("contract_provenance",))
#: The RF hyperparameter keys of the ranker (the cross-fit's ``rf_params``).
RANKER_PARAM_KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "max_features",
                     "minority_weight", "class_weight")


def primary_profile_path() -> Path:
    return PROFILES_DIR / _PRIMARY


def load_glass_primary_profile(path=None) -> ArmProfile:
    """
    The frozen GLASS profile, refused unless its status is ``"frozen"``, its
    fingerprint is the pinned one and its owner approval is confirmed.
    """
    return load_arm_profile(path or primary_profile_path(), arm="glass",
                            expected_fingerprint=PRIMARY_PROFILE_FINGERPRINT,
                            require_status=STATUS_FROZEN, require_confirmed_approval=True)


def _settings(profile: ArmProfile) -> dict:
    if not isinstance(profile, ArmProfile) or profile.arm != "glass":
        raise ContractViolation("expected the GLASS arm's ArmProfile")
    if set(profile.settings) != {"router_config", "ranker"}:
        raise ContractViolation("a GLASS profile's settings hold exactly 'router_config' and "
                                "'ranker'")
    return {k: dict(v) for k, v in profile.settings.items()}


def glass_base_config(profile: ArmProfile) -> GlassRouterConfig:
    """``profile``'s ``GlassRouterConfig``: F17 primary settings, no union obligation set."""
    router = _settings(profile)["router_config"]
    missing = [k for k in PROFILE_FIELDS if k not in router]
    unknown = [k for k in router if k not in PROFILE_FIELDS]
    if missing or unknown:
        raise ContractViolation(f"GLASS profile router_config: missing {missing}, unknown "
                                f"{unknown} (union obligations come only from the contract)")
    problems = [f"{name}={router[name]!r}, the F17 primary requires {want!r}"
                for name, want in GLASS_F17_PRIMARY.items() if router[name] != want]
    if problems:
        raise ContractViolation("GLASS profile is not the F17 primary:\n"
                                + "\n".join(f"  - {p}" for p in problems))
    return GlassRouterConfig(**router)


def ranker_settings_from_profile(profile: ArmProfile) -> dict:
    """``{"glass_rf_model": ..., "rf_params": {...}}`` of ``profile`` (reviews and tests)."""
    ranker = _settings(profile)["ranker"]
    if set(ranker) != {"glass_rf_model", "rf_params"}:
        raise ContractViolation("a GLASS profile's ranker holds exactly 'glass_rf_model' and "
                                "'rf_params'")
    if ranker["glass_rf_model"] != "fold_rf":
        raise ContractViolation("the F17 primary's fold_nested_rf inner ranker needs "
                                "glass_rf_model='fold_rf'")
    params = dict(ranker["rf_params"])
    if tuple(params) != RANKER_PARAM_KEYS:
        raise ContractViolation(f"ranker rf_params must hold exactly {list(RANKER_PARAM_KEYS)}")
    return {"glass_rf_model": ranker["glass_rf_model"], "rf_params": params}


def glass_config_from_profile(contract: SharedOperatingContractV2,
                              profile: ArmProfile) -> GlassRouterConfig:
    """
    ``profile`` under ``contract``, for reviews and tests. Never the PR 38
    primary config: that is ``glass_primary_config()``, and
    ``check_glass_primary_config`` refuses anything else.
    """
    config = glass_config_from_contract(glass_base_config(profile), contract)
    check_glass_config_follows_contract(config, contract)
    return config


def glass_primary_config() -> GlassRouterConfig:
    """
    The PR 38 primary GLASS config. Takes no record: the frozen contract
    (``load_frozen_contract``) and the frozen profile
    (``load_glass_primary_profile``) are loaded strictly.
    """
    return glass_config_from_profile(load_frozen_contract(), load_glass_primary_profile())


def ranker_settings() -> dict:
    """The PR 38 primary ranker settings, from the strictly loaded frozen profile."""
    return ranker_settings_from_profile(load_glass_primary_profile())


def check_glass_primary_config(config: GlassRouterConfig) -> None:
    """Refuse a config that is not exactly ``glass_primary_config()``."""
    if not isinstance(config, GlassRouterConfig):
        raise ContractViolation(f"expected a GlassRouterConfig, got {type(config).__name__}")
    want, got = glass_primary_config().to_dict(), config.to_dict()
    differ = sorted(k for k in want if got.get(k) != want[k])
    if differ:
        raise ContractViolation("not the frozen PR 38 primary GLASS config "
                                f"(glass_primary_config()): {differ} differ")


def _kind(value):
    """The value kind a ranker setting may take; ``None`` for anything else."""
    if isinstance(value, (bool, np.bool_)):           # before int: bool is an int subclass
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "none"
    return None


def _same(got, want) -> bool:
    """
    Exact equality of one scalar setting. A NumPy scalar equal to the frozen
    value is accepted (``np.int64(650)`` for 650); an array of any shape, the
    zero-dimensional one included, never is; a boolean (``bool`` or ``np.bool_``)
    never stands for a number, nor an integer for a float or the reverse.
    """
    if isinstance(got, np.ndarray) or isinstance(want, np.ndarray):
        return False
    kind = _kind(want)
    if kind is None or _kind(got) != kind:
        return False
    return bool(got == want)


def check_glass_primary_ranker(*, glass_rf_model, rf_params) -> None:
    """
    Refuse ranker settings that are not exactly ``ranker_settings()`` (PR 37
    Commit 5). ``check_glass_primary_config`` covers the ``GlassRouterConfig``
    only; the ranker's RF hyperparameters reach ``crossfit_stage2(rf_params=...,
    glass_rf_model=...)`` and the full-train ranker factory as separate
    arguments, so a re-tuned or hand-edited value would otherwise go unchecked.
    """
    want = ranker_settings()
    if not _same(glass_rf_model, want["glass_rf_model"]):
        raise ContractViolation(f"glass_rf_model {glass_rf_model!r} is not the frozen PR 38 "
                                f"primary ranker's {want['glass_rf_model']!r}")
    if not isinstance(rf_params, Mapping):
        raise ContractViolation(f"rf_params must be a mapping of hyperparameters, got "
                                f"{type(rf_params).__name__}")
    expected, got = want["rf_params"], dict(rf_params)
    missing = sorted(set(expected) - set(got))
    unknown = sorted(set(got) - set(expected), key=str)
    differ = sorted(k for k in expected if k in got and not _same(got[k], expected[k]))
    if missing or unknown or differ:
        raise ContractViolation("not the frozen PR 38 primary ranker settings "
                                f"(ranker_settings()): missing {missing}, unknown {unknown}, "
                                f"differ {differ}")
