"""
blackbox_pipeline.models.rf_router.artifacts
=============================================
Persistence for a fitted router.

A saved artifact carries the models AND the frozen operating point AND the
provenance needed to prove the operating point was not chosen against the test
split:

    * fitted forest(s)
    * t1, t2 and the band statistics that justified them
    * empirical band confidences and their support
    * the full RFRouterConfig and OperatingConstraints
    * the split fingerprint, so a router can never be scored against a
      different split than it was fitted under
    * library versions and a timestamp

The threshold sweeps are saved as CSV alongside the pickle: they are the audit
trail showing which constraint bound at every candidate band, and they are what
you reach for when a reviewer asks why t1 sits where it does.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

__all__ = ["save_rf_router", "load_rf_router", "RouterArtifact"]

_ARTIFACT_VERSION = "1.0"


class RouterArtifact(dict):
    """Loaded artifact. A dict with attribute access for convenience."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - passthrough
            raise AttributeError(item) from exc


# ======================================================================
def save_rf_router(
    router: Any,
    output_dir: str | Path,
    *,
    name: Optional[str] = None,
    test_metrics: Optional[dict] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """
    Write a fitted router to ``output_dir``.

    Parameters
    ----------
    router
        A fitted ``RFRouter`` or ``SingleRFRouter``.
    test_metrics
        Optional final evaluation output from ``evaluate_router``. Saved for the
        record only — nothing in the artifact is derived from it, and reloading
        an artifact never re-reads these numbers into the model. Test metrics
        must never feed back into threshold or configuration choices.
    """
    if not getattr(router, "is_fitted", False):
        raise ValueError("Router is not fitted; nothing to save")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mode = router.config.mode
    stem = name or f"rf_router_{mode}"

    payload = {
        "artifact_version": _ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "config": router.config.to_dict(),
        "constraints": router.config.constraints.to_dict(),
        "feature_names": list(router.feature_names_),
        "training_base_rate": float(router.training_base_rate),
        "thresholds": router.thresholds_.to_dict(),
        "bands": router.bands_.to_dict(),
        "fit_report": _jsonable(router.fit_report_),
        "split_fingerprint": router.split_fingerprint_,
        "test_metrics": test_metrics,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    if extra_metadata:
        payload["extra"] = extra_metadata

    if mode == "two_pass":
        payload["pass1_model"] = router.pass1_model_
        payload["pass2_model"] = router.pass2_model_
        payload["remainder_mask"] = router.remainder_mask_
    else:
        payload["model"] = router.model_

    model_path = out / f"{stem}.joblib"
    joblib.dump(payload, model_path, compress=3)

    # Human-readable sidecar (models stripped).
    meta = {k: v for k, v in payload.items() if not _is_model_key(k)}
    (out / f"{stem}_metadata.json").write_text(
        json.dumps(_jsonable(meta), indent=2, default=str)
    )

    # Threshold sweeps — the audit trail for the operating point.
    for pass_name, res in (
        ("pass1", router.thresholds_.pass1),
        ("pass2", router.thresholds_.pass2),
    ):
        if getattr(res, "sweep", None) is not None:
            res.sweep.to_csv(out / f"{stem}_{pass_name}_sweep.csv", index=False)

    print(f"✅ saved {mode} router → {model_path}")
    print(f"   metadata     : {out / f'{stem}_metadata.json'}")
    print(f"   threshold sweeps written alongside")
    return model_path


# ======================================================================
def load_rf_router(
    path: str | Path,
    *,
    expected_split_fingerprint: Optional[str] = None,
) -> RouterArtifact:
    """
    Load a saved artifact.

    Parameters
    ----------
    expected_split_fingerprint
        When given, the artifact's fingerprint must match. Pass
        ``split_fingerprint(GLOBAL_SPLIT["X_train"], GLOBAL_SPLIT["X_test"])``
        whenever you reload a router for evaluation: it is the guard against
        scoring a router against a split it was not fitted under, which would
        put training rows into the evaluation set without any visible error.
    """
    payload = joblib.load(Path(path))

    if payload.get("artifact_version") != _ARTIFACT_VERSION:
        print(
            f"⚠️  artifact version {payload.get('artifact_version')} "
            f"(expected {_ARTIFACT_VERSION}) — fields may have moved"
        )

    if expected_split_fingerprint is not None:
        found = payload.get("split_fingerprint")
        if found != expected_split_fingerprint:
            raise ValueError(
                "Split fingerprint mismatch.\n"
                f"  artifact : {found}\n"
                f"  expected : {expected_split_fingerprint}\n"
                "This router was fitted under a different train/test split. "
                "Scoring it against the current split risks evaluating on rows "
                "it trained on. Refit rather than overriding this check."
            )

    return RouterArtifact(payload)


# ======================================================================
def _is_model_key(key: str) -> bool:
    return key in {"model", "pass1_model", "pass2_model", "remainder_mask"}


def _jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / numpy scalars for JSON."""
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame {obj.shape[0]}x{obj.shape[1]}>"
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape}>"
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
