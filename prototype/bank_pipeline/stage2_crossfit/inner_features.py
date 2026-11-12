"""
stage2_crossfit.inner_features
==============================
The G13 correction (PR 37 F17, limitation L1), opt-in: inner-fold features
without the inner fold's labels. Specified by the versioned record
``records/inner_feature_isolation_v1.json`` (``pathway_record``).

``FoldLabelExcludedFeatures(X_raw, y)`` is the provider both routers accept
(``GlassRouterPipeline.fit`` / ``RFRouter.fit``, ``inner_features=``). It fits
the same ``RFFeatureEngineer`` the fit's own engineering fits, checks that its
bins ARE the bins the router is fitted on, and for a fold plan returns
``{j: bins}``: every row's bins from ``g13.label_excluded_engineer`` without
fold j's labels (label-free structure fixed, the locked thresholds). Each
inner fold's eligibility steps then use only those bins and the labels outside
the fold, so no label of fold j reaches fold j's eligibility through an
engineered feature, directly or through the training rows of its ranker,
forest, selector or threshold solver. ``provenance(plan)`` is what a router
keeps; the provider (which holds the raw rows) is never stored or saved.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from shared.stage2.feature_engineering import RFFeatureEngineer
from shared.stage_io import label_fingerprint

from . import evidence as E
from .g13 import bin_source, fold_excluded_frames, kde_reference_positions

__all__ = ["PATHWAY_OUTER", "PATHWAY_FOLD", "PATHWAYS", "pathway_record", "pathway_identity",
           "provider_factory", "frame_fingerprint", "plan_fingerprint",
           "FoldLabelExcludedFeatures"]

#: Historical (the frozen PR 37 pathway): the inner folds use the fit engineer's bins.
#: The G13 correction: ``fold_label_excluded``.
PATHWAY_OUTER, PATHWAY_FOLD = E.PATHWAY_OUTER, E.PATHWAY_FOLD
PATHWAYS = (PATHWAY_OUTER, PATHWAY_FOLD)


def pathway_record() -> Dict[str, Any]:
    return E.load_versioned_record(E.FEATURE_PATHWAY_FILE, E.FEATURE_PATHWAY_FINGERPRINT)


def pathway_identity(pathway: str) -> Dict[str, Any]:
    """``{"pathway", "record"}``: the record identity for the correction, None historically."""
    if pathway not in PATHWAYS:
        raise ValueError(f"unknown feature pathway {pathway!r}; expected one of {PATHWAYS}")
    return {"pathway": pathway,
            "record": E.record_identity(pathway_record()) if pathway == PATHWAY_FOLD else None}


def provider_factory(pathway: str):
    """What the cross-fits take as ``inner_features``: None historically."""
    pathway_identity(pathway)
    return FoldLabelExcludedFeatures if pathway == PATHWAY_FOLD else None


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 (16 hex) of a bins frame: index labels, columns and int64 values."""
    h = hashlib.sha256()
    h.update("|".join(map(str, frame.index)).encode("utf-8"))
    h.update("|".join(map(str, frame.columns)).encode("utf-8"))
    h.update(np.ascontiguousarray(frame.to_numpy(dtype=np.int64)).tobytes())
    return h.hexdigest()[:16]


def plan_fingerprint(plan) -> str:
    return hashlib.sha256(np.asarray(plan, dtype=np.int64).tobytes()).hexdigest()[:16]


class FoldLabelExcludedFeatures:
    """The ``fold_label_excluded`` provider for one fit's rows (module docstring)."""

    def __init__(self, X_raw: pd.DataFrame, y):
        self._X = X_raw
        self._y = np.asarray(y).astype(int)
        self.engineer = RFFeatureEngineer().fit(X_raw, pd.Series(self._y, index=X_raw.index))
        self.base_bins = bin_source(self.engineer.transform(X_raw))
        self._ref = kde_reference_positions(self.engineer, X_raw)
        self._frames: Dict[bytes, Dict[int, pd.DataFrame]] = {}
        self._prov: Dict[bytes, Dict[str, Any]] = {}

    def _check(self, X: pd.DataFrame, y) -> None:
        if not (X.index.equals(self._X.index) and set(X.columns) == set(self.base_bins.columns)
                and X.astype("int64").equals(self.base_bins[list(X.columns)])):
            raise AssertionError("inner features: the router's bins are not this provider's "
                                 "engineer's bins")
        if not np.array_equal(np.asarray(y).astype(int), self._y):
            raise AssertionError("inner features: the router's labels are not this provider's")

    def __call__(self, X: pd.DataFrame, y, fold_id) -> Dict[int, pd.DataFrame]:
        self._check(X, y)
        plan = np.asarray(fold_id).astype(int)
        key = plan.tobytes()
        if key not in self._frames:
            dtypes = X.dtypes.to_dict()
            frames = {j: bin_source(src)[list(X.columns)].astype(dtypes)
                      for j, src in fold_excluded_frames(self.engineer, self._X, self._y,
                                                         plan).items() if j >= 0}
            self._frames[key] = frames
            self._prov[key] = {
                **pathway_identity(PATHWAY_FOLD),
                "plan_fingerprint": plan_fingerprint(plan),
                "fit_rows": int(len(self._y)), "fit_positives": int(self._y.sum()),
                "fit_label_fingerprint": label_fingerprint(self._X.index, self._y),
                "base_bins_fingerprint": frame_fingerprint(self.base_bins),
                "kde_reference_rows": int(len(self._ref)),
                "inner_folds": [{"fold": j, "labels_excluded": int((plan == j).sum()),
                                 "kde_reference_rows_kept": int((plan[self._ref] != j).sum()),
                                 "bins_fingerprint": frame_fingerprint(f)}
                                for j, f in sorted(frames.items())],
            }
        return self._frames[key]

    def provenance(self, fold_id) -> Optional[Dict[str, Any]]:
        """What the router keeps: identities and fingerprints, no rows."""
        key = np.asarray(fold_id).astype(int).tobytes()
        if key not in self._prov:
            raise AssertionError("inner features: no frames were built for this fold plan")
        return dict(self._prov[key])
