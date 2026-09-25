"""
shared.stage3.protocol
======================
Cross-arm fairness checks: ``protocol_diff`` / ``assert_matched_protocol``
compare two artifacts field by field (only ``FAMILY_SPECIFIC`` fields may
differ), and ``check_param_reuse`` validates a previous artifact as a
hyperparameter source.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from .config import PROTOCOL_FIELDS


FAMILY_SPECIFIC = {"search_space"}


def effective_tuning(artifact) -> dict:
    """
    The tuning run that produced ``artifact.best_params``.

    For a run that reused parameters, that is the provenance record of the
    original search; ``{}`` if no search is on record (explicit params).
    """
    t = artifact.tuning or {}
    if t.get("tuned", "n_trials_requested" in t):
        return t
    prov = t.get("provenance") or {}
    return prov if prov.get("n_trials_requested") is not None else {}


def protocol_diff(a, b) -> list[tuple[str, Any, Any]]:
    """(field, a_value, b_value) for every protocol-relevant difference."""
    diffs: list[tuple[str, Any, Any]] = []
    for k in PROTOCOL_FIELDS:
        va, vb = a.config.get(k), b.config.get(k)
        if va != vb:
            diffs.append((k, va, vb))

    for k in ("split_fingerprint", "block_fingerprint", "feature_fit_scope",
              "population"):
        va, vb = getattr(a, k, None), getattr(b, k, None)
        if va != vb or va is None:
            diffs.append((k, va, vb))

    if list(a.features) != list(b.features):
        diffs.append(("features", a.features, b.features))
    if list(a.index_train) != list(b.index_train):
        diffs.append(("index_train", "…", "…"))
    if list(a.index_test) != list(b.index_test):
        diffs.append(("index_test", "…", "…"))
    fa, fb = a.fold_plan.get("fold_id"), b.fold_plan.get("fold_id")
    if fa is None or fb is None or not np.array_equal(fa, fb):
        diffs.append(("fold_id", "…", "…"))
    if a.class_weights != b.class_weights:
        diffs.append(("class_weights", a.class_weights, b.class_weights))

    ta, tb = effective_tuning(a), effective_tuning(b)
    for k in ("n_trials_requested", "random_state", "feature_fit_scope"):
        va, vb = ta.get(k), tb.get(k)
        if va != vb or va is None:
            diffs.append((f"tuning.{k}", va, vb))

    sa = (a.config.get("search_space") or {})
    sb = (b.config.get("search_space") or {})
    if sa != sb:
        diffs.append(("search_space", "family-specific", "family-specific"))
    return diffs


def assert_matched_protocol(a, b, allow: Iterable[str] = (),
                            verbose: bool = True) -> list[tuple[str, Any, Any]]:
    """Raise unless ``a`` and ``b`` differ only in family-specific fields."""
    allowed = set(allow) | FAMILY_SPECIFIC
    diffs = protocol_diff(a, b)
    blocking = [d for d in diffs if d[0] not in allowed]
    if verbose:
        print(f"  protocol check {a.model_family} vs {b.model_family}: "
              f"{len(diffs)} difference(s), {len(blocking)} blocking")
        for k, va, vb in diffs:
            tag = "  (allowed)" if k in allowed else ""
            print(f"     {k}: {va!r} vs {vb!r}{tag}")
    if blocking:
        raise AssertionError(
            "Stage 3 arms did not run the same protocol:\n"
            + "\n".join(f"  {k}: {va!r} vs {vb!r}" for k, va, vb in blocking)
        )
    return diffs


def check_param_reuse(
    prev,
    config,
    split_fingerprint: str,
    block_fingerprint: Optional[str],
    features: list[str],
) -> dict:
    """
    Validate a previous artifact as a hyperparameter source.

    Returns ``{"params", "source", "provenance"}`` for the runner, or raises.
    """
    problems: list[str] = []
    if prev.split_fingerprint != split_fingerprint:
        problems.append(f"split {prev.split_fingerprint!r} != {split_fingerprint!r}")
    if block_fingerprint is not None and \
            getattr(prev, "block_fingerprint", None) != block_fingerprint:
        problems.append("block fingerprint differs (features or folds changed)")
    if list(prev.features) != list(features):
        problems.append("feature contract differs")
    tuning = effective_tuning(prev)
    if not tuning:
        problems.append("the source artifact has no Optuna search on record")
    else:
        if tuning.get("n_trials_requested") != config.n_trials:
            problems.append(f"n_trials {tuning.get('n_trials_requested')} != "
                            f"{config.n_trials}")
        if tuning.get("feature_fit_scope") != config.feature_fit_scope:
            problems.append(f"feature_fit_scope {tuning.get('feature_fit_scope')!r}"
                            f" != {config.feature_fit_scope!r}")
    prev_cfg = prev.config or {}
    for k in ("n_tune_folds", "random_state", "beta", "tuning_decision_threshold",
              "pruner_startup_trials", "pruner_warmup_steps", "class_weight"):
        if prev_cfg.get(k) != getattr(config, k):
            problems.append(f"{k} {prev_cfg.get(k)!r} != {getattr(config, k)!r}")
    space = config.search_space.to_dict() if hasattr(config.search_space, "to_dict") \
        else config.search_space
    if prev_cfg.get("search_space") != space:
        problems.append("search space differs")
    if problems:
        raise ValueError(
            "Cannot reuse hyperparameters from this artifact:\n  - "
            + "\n  - ".join(problems)
            + "\nRe-tune with PARAM_SOURCE='tune'."
        )
    prov = {k: v for k, v in tuning.items() if k != "top_trials"}
    return {
        "params": dict(prev.best_params),
        "source": f"artifact created {prev.created_at} "
                  f"(tuned CV F2 {tuning.get('best_value', float('nan')):.6f})",
        "provenance": prov,
    }
