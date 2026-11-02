"""
shared.stage2.diagnostics.schema
================================
Vocabulary and read-only containers for train-side Stage 2 router diagnostics
(PR 37, Commit 1).

Nothing here reads an artifact or computes a metric. The containers are built
only by ``shared.stage2.diagnostics.boundary`` (the train-only extractors) and
consumed by ``operating_points``, ``metrics`` and ``equivalence``.

Routing states
--------------
Both routers write the same three route tokens, and the diagnostics map them to
one shared vocabulary without changing the stored originals:

    stored token   stored pred   normalised state
    "pass1"         0            PASS_1_NOT_SUBSCRIBE
    "pass2"         1            PASS_2_SUBSCRIBE
    "uncertain"    -1            ABSTAIN

Only these exact tokens are accepted (``GlassRouterPipeline.predict`` and
``rf_router.router.base.apply_cuts`` write exactly these). The Stage 4 adapter's
``normalize_router_decisions`` is deliberately NOT reused: it accepts many
spellings of abstain ("", "nan", "none", "remainder", ...), which suits Stage 4
input compatibility but would let a missing value pass as an abstention here.

Scores, confidences and probabilities
-------------------------------------
Every score stream carries an explicit ``space``. A value in [0, 1] is not
assumed to be a calibrated probability:

    calibrated_probability  per-fold calibrated P(y = 1) (LR train_proba_oof,
                            EBM oof["proba_calibrated"])
    raw_model_score         an uncalibrated model output in [0, 1]
                            (EBM oof["proba"])
    weighted_forest_score   predict_proba of a class-weighted Random Forest
                            (RF Router Pass 1); not a probability
    router_transform        the routers' shared deterministic transform of
                            route + confidence (b(1 - c), b + (1 - b)c, b);
                            not calibrated

Provenance statuses
-------------------
    validated      recomputed or compared on the training side here
    attested       equal to a reference value supplied by the caller (for
                   example the stage_io split fingerprint recorded in the
                   baseline PROVENANCE.json); not recomputed here
    recorded_only  present in the artifact but not re-verifiable here
    not_recorded   absent from the (historical) artifact; optional

Historical test-set exposure
----------------------------
See ``HISTORICAL_TEST_EXPOSURE``. The diagnostics prevent further test-derived
influence; they cannot make the held-out split untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PASS_1_NOT_SUBSCRIBE", "PASS_2_SUBSCRIBE", "ABSTAIN", "ROUTER_STATES",
    "SOURCE_DECISIONS", "SCORE_SPACES",
    "SELECTION_GLOBAL_TRAIN_LABELS", "SELECTION_FOLD_NESTED", "SELECTION_RATE_MATCHED",
    "VALIDATED", "ATTESTED", "RECORDED_ONLY", "NOT_RECORDED", "PROVENANCE_STATUSES",
    "HISTORICAL_TEST_EXPOSURE",
    "RouterDiagnosticError", "UnsupportedDiagnostic",
    "ProvenanceItem", "TrainReference", "ScoreStream", "ThresholdRecord",
    "StoredDecision", "RouterOOFView", "UpstreamOOFView", "Stage4Thresholds",
    "RFPass1ScoreCandidate",
]

# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------
PASS_1_NOT_SUBSCRIBE = "PASS_1_NOT_SUBSCRIBE"
PASS_2_SUBSCRIBE = "PASS_2_SUBSCRIBE"
ABSTAIN = "ABSTAIN"
ROUTER_STATES: Tuple[str, str, str] = (PASS_1_NOT_SUBSCRIBE, PASS_2_SUBSCRIBE, ABSTAIN)

#: Stored route token -> (normalised state, the stored ``*_pred`` value it implies).
SOURCE_DECISIONS: Mapping[str, Tuple[str, int]] = MappingProxyType({
    "pass1": (PASS_1_NOT_SUBSCRIBE, 0),
    "pass2": (PASS_2_SUBSCRIBE, 1),
    "uncertain": (ABSTAIN, -1),
})

SCORE_SPACES: Mapping[str, str] = MappingProxyType({
    "calibrated_probability": "per-fold calibrated P(y = 1) on out-of-fold training rows",
    "raw_model_score": "uncalibrated model output in [0, 1]; not a calibrated probability",
    "weighted_forest_score": "class-weighted Random Forest predict_proba; not a probability",
    "router_transform": "router route + confidence transform (b(1-c) / b+(1-b)c / b); not calibrated",
})

SELECTION_GLOBAL_TRAIN_LABELS = (
    "global: one value selected using the labels of ALL training rows, then applied "
    "back to those rows' out-of-fold scores (selection-dependent; disclose)")
SELECTION_FOLD_NESTED = (
    "fold-nested: each fold's threshold selected on inner out-of-fold scores of that "
    "fold's training rows; the fold's own labels never reach it")
SELECTION_RATE_MATCHED = (
    "rate-matched: a label-free cut matched to a router's realised count")

VALIDATED = "validated"
ATTESTED = "attested"
RECORDED_ONLY = "recorded_only"
NOT_RECORDED = "not_recorded"
PROVENANCE_STATUSES = (VALIDATED, ATTESTED, RECORDED_ONLY, NOT_RECORDED)

HISTORICAL_TEST_EXPOSURE = (
    "Historical test-set exposure: the held-out test split was evaluated in PRs 33-36, "
    "and the October 23 artifacts store test metrics and test-oracle thresholds. PR 37 "
    "diagnostics read only allowlisted training-side fields and prevent further "
    "test-derived influence; they do not make the test split untouched.")


class RouterDiagnosticError(ValueError):
    """A diagnostic input is missing, malformed or used outside its contract."""


class UnsupportedDiagnostic(RouterDiagnosticError):
    """The persisted artifacts cannot support the requested diagnostic."""


def _frozen(values, dtype=None) -> np.ndarray:
    """A private, read-only copy (callers can never mutate a view's arrays)."""
    a = np.array(values, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


def _proxy(mapping: Optional[Mapping]) -> Mapping:
    return MappingProxyType(dict(mapping or {}))


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProvenanceItem:
    """One provenance fact about a view and how it was established."""

    name: str
    status: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in PROVENANCE_STATUSES:
            raise RouterDiagnosticError(
                f"unknown provenance status {self.status!r}; expected one of {PROVENANCE_STATUSES}")


def _provenance_frame(items) -> pd.DataFrame:
    return pd.DataFrame([{"name": p.name, "status": p.status, "detail": p.detail} for p in items],
                        columns=["name", "status", "detail"])


# ----------------------------------------------------------------------
# Train reference
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TrainReference:
    """
    The training rows every view is checked against. Holds no test labels.

    Built by ``boundary.build_train_reference`` from three explicit inputs: the
    training row index, the training labels, and the test row index (row
    identities only, for the 16-hex router split fingerprint, which hashes row
    indices and no labels).

    ``stage_io_split_fingerprint`` is ATTESTED, never recomputed: its hash
    includes the test labels.
    """

    index_train: pd.Index
    y_train: np.ndarray
    router_split_fingerprint: str
    train_label_fingerprint: str
    stage_io_split_fingerprint: Optional[str] = None
    stage_io_attestation: Optional[str] = None

    @property
    def n_train(self) -> int:
        return len(self.index_train)

    def fold_ids(self, n_folds: int, seed: int) -> np.ndarray:
        """The shared stratified plan on these labels (shared.stage_io.fold_assignment)."""
        from shared.stage_io import fold_assignment
        return fold_assignment(self.y_train, cv_folds=int(n_folds),
                               random_state=int(seed)).to_numpy()


# ----------------------------------------------------------------------
# Score / threshold / decision records
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreStream:
    """One per-row score over the training rows, with its space and origin."""

    name: str
    values: np.ndarray
    space: str
    field: str
    provenance: str

    def __post_init__(self) -> None:
        if self.space not in SCORE_SPACES:
            raise RouterDiagnosticError(f"{self.name}: unknown score space {self.space!r}")


@dataclass(frozen=True)
class ThresholdRecord:
    """A stored scalar threshold, the space it applies to and how it was chosen."""

    name: str
    value: float
    space: str
    field: str
    selection: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.space not in SCORE_SPACES:
            raise RouterDiagnosticError(f"{self.name}: unknown threshold space {self.space!r}")


@dataclass(frozen=True)
class StoredDecision:
    """Stored per-row decisions with the per-row thresholds that produced them."""

    name: str
    values: np.ndarray
    per_row_threshold: np.ndarray
    space: str
    field: str
    selection: str
    by_fold: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.space not in SCORE_SPACES:
            raise RouterDiagnosticError(f"{self.name}: unknown decision space {self.space!r}")


# ----------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RouterOOFView:
    """
    One router's outer-cross-fit (train-side, out-of-fold) outputs, one row per
    training row in ``X_train`` order. Built only by ``extract_router_view``.

    ``state`` holds the normalised states; ``source_decision`` / ``source_pred``
    the stored originals, unchanged. ``confidence`` is the stored router
    confidence (0.0 on abstained rows); ``confidence_kind`` says what it is.
    ``router_score`` is the positive column of the stored ``train_proba``
    (space ``router_transform``). ``fold_thresholds`` (RF only): each outer
    fold router's ``t1`` / ``t2``. ``metadata``: allowlisted training-side
    configuration (hyperparameters, constraint source, full-train thresholds).
    """

    arm: str
    model_family: str
    router_mode: Optional[str]
    index: pd.Index
    y: np.ndarray
    fold_id: np.ndarray
    state: np.ndarray
    source_decision: np.ndarray
    source_pred: np.ndarray
    confidence: np.ndarray
    confidence_kind: str
    router_score: ScoreStream
    pass2_population: Mapping[str, str]
    fold_thresholds: Optional[Mapping[int, Mapping[str, float]]]
    metadata: Mapping[str, object]
    provenance: Tuple[ProvenanceItem, ...]
    source: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def pass1(self) -> np.ndarray:
        return self.state == PASS_1_NOT_SUBSCRIBE

    @property
    def pass2(self) -> np.ndarray:
        return self.state == PASS_2_SUBSCRIBE

    @property
    def abstain(self) -> np.ndarray:
        return self.state == ABSTAIN

    def fold_base_rates(self) -> np.ndarray:
        """Per row: the positive rate of the rows its fold model was fitted on."""
        out = np.empty(self.n, dtype=float)
        y = self.y.astype(float)
        for k in np.unique(self.fold_id):
            out[self.fold_id == k] = y[self.fold_id != k].mean()
        return out

    def to_frame(self) -> pd.DataFrame:
        """All per-row fields, indexed by the training row index, in X_train order."""
        return pd.DataFrame(
            {
                "y": self.y,
                "fold_id": self.fold_id,
                "state": pd.Categorical(self.state, categories=list(ROUTER_STATES)),
                "pass1": self.pass1,
                "pass2": self.pass2,
                "abstain": self.abstain,
                "source_decision": self.source_decision,
                "source_pred": self.source_pred,
                "router_confidence": self.confidence,
                "router_score": self.router_score.values,
            },
            index=self.index,
        )

    def provenance_frame(self) -> pd.DataFrame:
        return _provenance_frame(self.provenance)


@dataclass(frozen=True)
class UpstreamOOFView:
    """
    Stage 1 LR or Stage 3 EBM out-of-fold outputs over the training rows.
    ``scores`` is keyed by score space; ``thresholds`` and ``stored_decisions``
    by source name (see ``operating_points.OPERATING_POINTS``).
    """

    model: str
    arm: str
    index: pd.Index
    y: np.ndarray
    fold_id: np.ndarray
    scores: Mapping[str, ScoreStream]
    thresholds: Mapping[str, ThresholdRecord]
    stored_decisions: Mapping[str, StoredDecision]
    provenance: Tuple[ProvenanceItem, ...]
    source: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def n(self) -> int:
        return len(self.index)

    def provenance_frame(self) -> pd.DataFrame:
        return _provenance_frame(self.provenance)


@dataclass(frozen=True)
class Stage4Thresholds:
    """The Stage 4 arbiter's recall-targeted LR / EBM cuts (secondary operating points)."""

    records: Mapping[str, ThresholdRecord]
    target_recall: float
    provenance: Tuple[ProvenanceItem, ...]
    source: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def provenance_frame(self) -> pd.DataFrame:
        return _provenance_frame(self.provenance)


@dataclass(frozen=True)
class RFPass1ScoreCandidate:
    """
    A per-row score that may be the RF Router's outer Pass 1 score (the GLASS-arm
    RF result's ``oof_proba``). Not usable as a score stream until
    ``equivalence.check_rf_pass1_score_equivalence`` passes.
    """

    index: pd.Index
    fold_id: np.ndarray
    values: np.ndarray
    params: Mapping[str, object]
    random_state: int
    split_fingerprint: str
    fold_fingerprint: str
    train_label_fingerprint: str
    protocol: Mapping[str, object]
    provenance: Tuple[ProvenanceItem, ...]
    source: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def digest(self) -> str:
        """SHA-256 over the scores and fold ids: ties an equivalence report to this candidate."""
        import hashlib
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.values, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.fold_id, dtype=np.int64).tobytes())
        return h.hexdigest()
