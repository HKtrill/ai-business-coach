"""
shared.stage_io
===============
Cross-arm output contract for the Stage 4 handoff.

Comparable outputs from the GLASS and black-box arms are normalized into the
same representation before they are consumed by the final arbiter:

    * train-side probabilities that are OUT-OF-FOLD,
    * test-side probabilities from the full-train model,
    * both indexed on GLOBAL_SPLIT's index,
    * the fold each training row was held out in,
    * the operating point and explicit provenance for how it was chosen.

This module is not the internal stage-to-stage contract of the cascade.
Stages remain free to use model-specific artifacts internally. ``StageOutput``
exists at the boundary where outputs from the GLASS and black-box arms must be
aligned, audited and consumed by the final arbiter.

Why this exists
---------------
Before this contract, comparable outputs could be represented differently
across the two arms or rely on positional alignment. That is unsafe for the
final arbiter: a silent row-order mismatch can produce apparently valid
training data whose predictions no longer correspond to the correct labels.

``StageOutput`` therefore carries explicit indices, fold assignments and split
provenance so Stage 4 can verify that paired inputs describe the same rows
under the same experimental split.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

__all__ = [
    "STAGE_OUTPUT_VERSION",
    "THRESHOLD_SOURCES",
    "StageOutput",
    "StageOutputError",
    "ThresholdContaminationError",
    "fold_assignment",
    "split_fingerprint",
    "assert_comparable",
]

#: Contract version. Bumped whenever a field is added, removed or renamed;
#: a mismatch on load prints a warning rather than failing.
STAGE_OUTPUT_VERSION = "1.1"   # 1.1: + split_fingerprint

#: Allowed ``threshold_source`` values, mapped to whether a threshold chosen
#: that way is cross-fitted with respect to the training rows. Only a
#: cross-fitted threshold may produce train-side decisions without an opt-in.
THRESHOLD_SOURCES: Dict[str, bool] = {
    # One scalar, argmax of F2 over 0.05-0.49, chosen on ALL training labels
    # and applied back to those same rows. Matches GLASS and the MLP today.
    "in_stage_f2_cv": False,
    # The wider notebook sweep (0.05-0.95 / 0.005). Same contamination, more
    # selection optimism — 181 candidates instead of 45.
    "notebook_wide_sweep": False,
    # Re-swept inside each outer fold on that fold's TRAINING rows only, then
    # applied to the held-out rows. The only cross-fitted option.
    "nested_per_fold": True,
}


class StageOutputError(ValueError):
    """
    A ``StageOutput`` violates the contract.

    Raised on construction or load for: a non-Series field, mismatched
    indices, NaN or out-of-range probabilities, non-binary labels, an unknown
    ``threshold_source``, a threshold outside (0, 1), or overlapping
    train/test indices. Also raised by :func:`assert_comparable`.
    """


class ThresholdContaminationError(RuntimeError):
    """
    Train-side decisions were requested from a threshold fitted on train labels.

    Raised by :meth:`StageOutput.train_decisions_oof` unless the threshold is
    cross-fitted or the caller passes ``allow_contaminated=True``.
    """


# ----------------------------------------------------------------------
def fold_assignment(
    y,
    *,
    cv_folds: int,
    random_state: int,
    index: Optional[pd.Index] = None,
) -> pd.Series:
    """
    Reproduce the fold each training row was held out in.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Binary training labels, in the same row order the stage used.
    cv_folds, random_state : int
        Must match what the stage passed to ``StratifiedKFold``.
    index : pandas.Index, optional
        Index for the returned Series. Taken from ``y`` if it is a Series.

    Returns
    -------
    pandas.Series of int
        Named ``fold_id``, values ``0 .. cv_folds - 1``.

    Raises
    ------
    StageOutputError
        If any row is left without a fold (cannot happen with a valid
        splitter; guards against a future change).

    Notes
    -----
    ``StratifiedKFold(shuffle=True)`` is a pure function of ``(n_splits,
    random_state, y, row order)``, so this reconstructs the exact partition
    ``cross_val_predict`` used without re-running it. Both arms use
    ``(10, shuffle=True, 42)``, which is what makes their OOF probabilities
    paired row-for-row and fold-for-fold.

    Examples
    --------
    >>> fold_assignment(y_train, cv_folds=10, random_state=42).value_counts()
    """
    if index is None:
        index = y.index if isinstance(y, pd.Series) else pd.RangeIndex(len(y))
    y_arr = np.asarray(y).astype(int)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold = np.full(len(y_arr), -1, dtype=int)
    for k, (_, va) in enumerate(cv.split(np.zeros(len(y_arr)), y_arr)):
        fold[va] = k

    if (fold < 0).any():
        raise StageOutputError("fold_assignment left rows unassigned")
    return pd.Series(fold, index=index, name="fold_id")


# ----------------------------------------------------------------------
def split_fingerprint(train_index, test_index, y_train=None, y_test=None) -> str:
    """
    Short, stable identifier for one train/test split.

    Parameters
    ----------
    train_index, test_index : pandas.Index or array-like
        Row identifiers of each split, in order.
    y_train, y_test : array-like, optional
        Labels. Included when given, so a relabelled split gets a different
        fingerprint even if the rows are the same.

    Returns
    -------
    str
        12 hex characters (SHA-256 prefix).

    Notes
    -----
    Order-sensitive on purpose: the same rows in a different order is a
    different contract, because Stage 4 joins positionally inside a split
    whenever someone forgets to join on the index.

    Hashes the string form of each value, so it is stable across pandas and
    numpy versions — unlike ``pd.util.hash_pandas_object``. Both notebooks must
    compute it with this function for fingerprints to be comparable.

    Examples
    --------
    >>> split_fingerprint(X_train.index, X_test.index, y_train, y_test)
    '3f9a1c0b7e22'
    """
    h = hashlib.sha256()
    for tag, obj in (("train", train_index), ("test", test_index),
                     ("y_train", y_train), ("y_test", y_test)):
        if obj is None:
            continue
        h.update(tag.encode())
        h.update(",".join(map(str, np.asarray(obj).tolist())).encode())
    return h.hexdigest()[:12]


# ----------------------------------------------------------------------
@dataclass
class StageOutput:
    """
    One stage, one arm, everything the next stage is allowed to see.

    Parameters
    ----------
    stage, arm, model : str
        e.g. ``"stage1"``, ``"glass"``, ``"calibrated_lr"``.
    feature_names : list of str
        The stage's input columns, in fitted order.
    train_proba_oof : pandas.Series
        OUT-OF-FOLD calibrated ``P(y=1)`` over the training split. Not
        ``predict_proba(X_train)`` — that is in-sample and every row scores
        itself.
    test_proba : pandas.Series
        Calibrated ``P(y=1)`` on the test split from the full-train model.
    y_train, y_test : pandas.Series
        Labels, indexed like their probabilities.
    train_fold_id : pandas.Series
        From :func:`fold_assignment`.
    threshold : float
        The operating point in force.
    threshold_source : str
        A key of :data:`THRESHOLD_SOURCES`. Mandatory, no default — the
        caller has to state it.
    calibration_method : str
        The family actually selected. Under ``'auto'`` this is the winner,
        not the request; see ``calibration_requested``.
    calibration_requested : str
        What was asked for.
    oof_provenance : str
        From :func:`~shared.calibration_guard.assert_refittable`.
    best_params : dict, optional
        Winning hyperparameters.
    best_cv_score : float, optional
        Tuning objective at the winner (mean CV ROC-AUC for Stage 1).
    cv_f2 : float, optional
        F2 at ``threshold`` on the OOF probabilities.
    threshold_sweep : pandas.DataFrame, optional
        The full sweep the threshold was picked from.
    calibration_metrics : dict, optional
        Diagnostics from ``fit_calibration``.
    metrics_test : dict, optional
        Held-out metrics.
    config : dict, optional
        Everything needed to reproduce the fit.
    split_fingerprint : str, optional
        :func:`split_fingerprint` of the train/test indices and labels.
        Computed automatically; if supplied (e.g. on load) it is recomputed
        and must match, so a file cannot silently carry another split's rows.
    contract_version, created_utc, environment
        Filled in automatically; preserved across save/load.

    Attributes
    ----------
    threshold_is_cross_fitted : bool
        Whether ``threshold`` is honest with respect to the training rows.

    Raises
    ------
    StageOutputError
        On any index mismatch, NaN, non-binary label, out-of-range
        probability, or unknown ``threshold_source``.
    """

    stage: str
    arm: str
    model: str
    feature_names: List[str]

    train_proba_oof: pd.Series
    test_proba: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    train_fold_id: pd.Series

    threshold: float
    threshold_source: str
    calibration_method: str
    calibration_requested: str
    oof_provenance: str

    best_params: Dict = field(default_factory=dict)
    best_cv_score: Optional[float] = None
    cv_f2: Optional[float] = None
    threshold_sweep: Optional[pd.DataFrame] = None
    calibration_metrics: Dict = field(default_factory=dict)
    metrics_test: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)

    split_fingerprint: str = ""
    contract_version: str = STAGE_OUTPUT_VERSION
    created_utc: str = ""
    environment: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Fill provenance defaults, then run every contract check."""
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.environment:
            self.environment = {
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            }

        if self.threshold_source not in THRESHOLD_SOURCES:
            raise StageOutputError(
                f"threshold_source must be one of {sorted(THRESHOLD_SOURCES)}, "
                f"got {self.threshold_source!r}"
            )
        if not 0.0 < float(self.threshold) < 1.0:
            raise StageOutputError(f"threshold out of range: {self.threshold}")

        self._check_block("train", self.train_proba_oof, self.y_train, self.train_fold_id)
        self._check_block("test", self.test_proba, self.y_test, None)

        overlap = self.train_proba_oof.index.intersection(self.test_proba.index)
        if len(overlap):
            raise StageOutputError(
                f"train and test indices overlap on {len(overlap)} rows — "
                "the split is not disjoint"
            )

        fp = split_fingerprint(self.y_train.index, self.y_test.index,
                               self.y_train, self.y_test)
        if self.split_fingerprint and self.split_fingerprint != fp:
            raise StageOutputError(
                f"split_fingerprint {self.split_fingerprint!r} does not match "
                f"the rows and labels carried ({fp!r})"
            )
        self.split_fingerprint = fp

    @staticmethod
    def _check_block(name, proba, y, fold) -> None:
        """
        Validate one split's probabilities, labels and (train only) fold ids.

        Each must be a Series; all must share one index; probabilities must be
        finite and in [0, 1]; labels must be 0/1.
        """
        for label, obj in (("proba", proba), ("y", y), ("fold_id", fold)):
            if obj is None:
                continue
            if not isinstance(obj, pd.Series):
                raise StageOutputError(
                    f"{name} {label} must be a pandas Series (got "
                    f"{type(obj).__name__}) — Stage 4 joins on the index, and "
                    "a bare array has none"
                )
        if not proba.index.equals(y.index):
            raise StageOutputError(f"{name}: proba.index != y.index")
        if fold is not None and not proba.index.equals(fold.index):
            raise StageOutputError(f"{name}: proba.index != fold_id.index")
        if proba.isna().any():
            raise StageOutputError(f"{name}: proba contains NaN")
        if not proba.between(0.0, 1.0).all():
            raise StageOutputError(f"{name}: proba outside [0, 1]")
        if not np.isin(np.asarray(y), (0, 1)).all():
            raise StageOutputError(f"{name}: y is not binary 0/1")

    # ------------------------------------------------------------------
    @property
    def threshold_is_cross_fitted(self) -> bool:
        """
        Whether ``threshold`` is honest with respect to the training rows.

        True only for ``threshold_source='nested_per_fold'``, where each held-out
        row's threshold was chosen without its label.
        """
        return THRESHOLD_SOURCES[self.threshold_source]

    def train_decisions_oof(self, *, allow_contaminated: bool = False) -> pd.Series:
        """
        Train-side hard decisions at ``threshold``.

        Parameters
        ----------
        allow_contaminated : bool, default False
            Required when ``threshold_is_cross_fitted`` is False.

        Returns
        -------
        pandas.Series of int8
            ``train_proba_oof >= threshold``, indexed like the training split,
            named ``"{stage}_{arm}_pred_oof"``.

        Raises
        ------
        ThresholdContaminationError
            If the threshold was fitted on these rows' labels and
            ``allow_contaminated`` is False.

        Notes
        -----
        The probabilities are out-of-fold; the CUT applied to them is not. It
        was chosen by argmax over a grid using every training label, then
        applied back to those same rows. Any Stage 4 feature built from this —
        a decision, a margin, a cross-arm agreement flag — inherits that.
        """
        if not self.threshold_is_cross_fitted and not allow_contaminated:
            raise ThresholdContaminationError(
                f"threshold_source={self.threshold_source!r} means this "
                "threshold was selected by argmax over the labels of the same "
                "training rows you are about to describe.\n"
                "The probabilities are out-of-fold; these decisions are not.\n"
                "\n"
                "Either build Stage 4 features from train_proba_oof directly, "
                "or re-fit the threshold per outer fold "
                "(threshold_source='nested_per_fold').\n"
                "Pass allow_contaminated=True to proceed anyway — and say so "
                "in the write-up, because the Stage 4 train matrix will be "
                "optimistic."
            )
        return (self.train_proba_oof >= self.threshold).astype("int8").rename(
            f"{self.stage}_{self.arm}_pred_oof"
        )

    def test_decisions(self) -> pd.Series:
        """
        Test-side hard decisions at ``threshold``.

        Returns
        -------
        pandas.Series of int8
            ``test_proba >= threshold``, indexed like the test split, named
            ``"{stage}_{arm}_pred"``.

        Notes
        -----
        Always safe: the threshold was chosen on training data only, so it
        carries no information about test labels.
        """
        return (self.test_proba >= self.threshold).astype("int8").rename(
            f"{self.stage}_{self.arm}_pred"
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """
        Plain-dict form. This, not the dataclass, is what gets pickled.

        A pickled dataclass records its module path and fails to load once the
        package moves or is imported under a different name; a dict of pandas
        objects and primitives does not.

        Returns
        -------
        dict
            One key per dataclass field, plus ``"__stage_output__"`` holding
            the contract version as a type marker.
        """
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["__stage_output__"] = self.contract_version
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "StageOutput":
        """
        Rebuild from :meth:`to_dict` output, re-running every contract check.

        Parameters
        ----------
        d : dict
            Must carry the ``"__stage_output__"`` marker. Unknown keys are
            ignored, so a newer file with extra fields still loads.

        Returns
        -------
        StageOutput

        Raises
        ------
        StageOutputError
            If the marker is missing or any contract check fails.

        Notes
        -----
        A version mismatch prints a warning and proceeds.
        """
        if "__stage_output__" not in d:
            raise StageOutputError("dict does not hold a StageOutput")
        version = d["__stage_output__"]
        if version != STAGE_OUTPUT_VERSION:
            print(
                f"⚠️  contract version {version} (expected "
                f"{STAGE_OUTPUT_VERSION}) — fields may have moved"
            )
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, output_dir: str | Path, *, name: Optional[str] = None) -> Path:
        """
        Write the contract to disk as a compressed joblib dict.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Created if missing.
        name : str, optional
            Filename stem. Defaults to ``"{stage}_{arm}"``.

        Returns
        -------
        pathlib.Path
            ``<output_dir>/<name>_output.joblib``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name or f'{self.stage}_{self.arm}'}_output.joblib"
        joblib.dump(self.to_dict(), path, compress=3)
        print(f"✅ {self.stage}/{self.arm} → {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StageOutput":
        """
        Load a file written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path

        Returns
        -------
        StageOutput
            Validated — every contract check re-runs.

        Raises
        ------
        StageOutputError
            If the file does not hold a StageOutput dict or fails a check.
        """
        obj = joblib.load(Path(path))
        if not isinstance(obj, dict):
            raise StageOutputError(f"{path} does not hold a StageOutput dict")
        return cls.from_dict(obj)

    def summary(self) -> Dict:
        """
        Scalar metadata, for printing or a JSON dump.

        Returns
        -------
        dict
            Every field except the Series and the sweep DataFrame, plus
            ``n_train``, ``n_test`` and ``threshold_is_cross_fitted``.
        """
        skip = {"train_proba_oof", "test_proba", "y_train", "y_test",
                "train_fold_id", "threshold_sweep"}
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}
        d["n_train"] = len(self.train_proba_oof)
        d["n_test"] = len(self.test_proba)
        d["threshold_is_cross_fitted"] = self.threshold_is_cross_fitted
        return d

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        xf = "cross-fitted" if self.threshold_is_cross_fitted else "FITTED-ON-TRAIN"
        return (
            f"StageOutput({self.stage}/{self.arm}/{self.model}, "
            f"n_train={len(self.train_proba_oof)}, n_test={len(self.test_proba)}, "
            f"t={self.threshold:.3f} [{xf}], cal={self.calibration_method})"
        )


# ----------------------------------------------------------------------
def assert_comparable(a: StageOutput, b: StageOutput) -> None:
    """
    Check two arms' outputs can legitimately be compared row-for-row.

    Parameters
    ----------
    a, b : StageOutput
        Typically the GLASS and black-box outputs of the same stage.

    Raises
    ------
    StageOutputError
        On differing stage, split fingerprint, feature set, index, labels or
        fold partition.

    Notes
    -----
    The fold check is the one that matters for paired comparison: identical
    fold ids mean each row's two OOF probabilities were produced by models
    trained on the same rows, so per-row and per-fold differences measure the
    model class rather than the split.

    A differing ``threshold_source`` is printed as a warning, not raised: the
    arms can still be compared on probabilities, but not as "the same
    procedure".

    Examples
    --------
    >>> assert_comparable(glass_out, mlp_out)
    """
    if a.stage != b.stage:
        raise StageOutputError(f"different stages: {a.stage} vs {b.stage}")
    if a.split_fingerprint != b.split_fingerprint:
        raise StageOutputError(
            f"different splits: {a.arm}={a.split_fingerprint} vs "
            f"{b.arm}={b.split_fingerprint}"
        )
    if set(a.feature_names) != set(b.feature_names):
        raise StageOutputError(
            f"feature sets differ:\n  {a.arm}: {sorted(a.feature_names)}\n"
            f"  {b.arm}: {sorted(b.feature_names)}"
        )
    for split, sa, sb in (
        ("train", a.train_proba_oof, b.train_proba_oof),
        ("test", a.test_proba, b.test_proba),
    ):
        if not sa.index.equals(sb.index):
            raise StageOutputError(f"{split} indices differ between arms")
    if not a.y_train.equals(b.y_train) or not a.y_test.equals(b.y_test):
        raise StageOutputError("labels differ between arms")
    if not a.train_fold_id.equals(b.train_fold_id):
        raise StageOutputError(
            "fold partitions differ — per-row comparison is not paired. "
            "Both arms must use the same (cv_folds, random_state) for OOF."
        )
    if a.threshold_source != b.threshold_source:
        print(
            f"⚠️  threshold_source differs: {a.arm}={a.threshold_source!r} vs "
            f"{b.arm}={b.threshold_source!r} — the arms did not pick their "
            "operating points by the same procedure"
        )
