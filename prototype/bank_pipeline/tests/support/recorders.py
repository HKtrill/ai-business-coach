"""
Record which rows every fit / transform / predict call saw.

Wrap any sklearn-style estimator or transformer in ``RecordingEstimator`` and
pass a shared ``CallLog``. Each call appends a ``Call`` naming the method, the
fit it belongs to, and the rows involved. ``tests.support.checks`` then turns
the log into leakage verdicts.

The wrapper is transparent: it fits a clone of the wrapped object, delegates
every call and fitted attribute, and keeps the wrapped object's sklearn tags,
so ``cross_val_predict`` still stratifies a wrapped classifier.

Rows are identified by their index label for pandas input. For a plain array
(e.g. the scaled matrix Stage 1 hands to ``cross_val_predict``) each row is
identified by a hash of its bytes; ``array_row_labels`` maps those hashes back
to index labels.

Public API
----------
Call                one recorded call
CallLog             shared log; survives sklearn ``clone``
RecordingEstimator  transparent recording wrapper (sklearn-style objects)
RowTaggedEstimator  the same for arrays whose rows are not unique: rows carry
                    a leading integer tag column, stripped before delegating
tag_rows(X)         prepend that tag column (row positions by default)
RecordingStage3Estimator  the same for the shared Stage 3 estimator interface
row_keys(X)         row identifiers for a DataFrame / Series / 2-D array
array_row_labels    {row hash: index label} for an array and its index
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

FIT_METHODS = ("fit", "fit_transform")
PREDICT_METHODS = ("predict", "predict_proba", "decision_function")


@dataclass(frozen=True)
class Call:
    """One recorded call: ``method`` on the model produced by fit ``fit_id``."""

    method: str
    fit_id: int
    keys: Tuple[Hashable, ...]


class CallLog(list):
    """
    List of ``Call`` shared by a wrapper and all its clones.

    sklearn's ``clone`` deep-copies non-estimator parameters; returning
    ``self`` from ``__deepcopy__`` keeps every clone writing to one log.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self._next_fit_id = 0

    def __deepcopy__(self, memo):
        return self

    def new_fit_id(self) -> int:
        self._next_fit_id += 1
        return self._next_fit_id

    def fits(self) -> Dict[int, frozenset]:
        """``{fit_id: rows the model was fitted on}``."""
        return {c.fit_id: frozenset(c.keys) for c in self if c.method in FIT_METHODS}

    def calls(self, fit_id: int, methods: Iterable[str]) -> list:
        """Calls of ``methods`` made with the model from ``fit_id``."""
        methods = tuple(methods)
        return [c for c in self if c.fit_id == fit_id and c.method in methods]


# ----------------------------------------------------------------------
def _row_hash(row: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=8).hexdigest()


def row_keys(X) -> Tuple[Hashable, ...]:
    """Index labels for pandas input; per-row byte hashes for a 2-D array."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return tuple(X.index)
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise TypeError(f"expected a DataFrame, Series or 2-D array, got ndim={arr.ndim}")
    return tuple(_row_hash(r) for r in arr)


def array_row_labels(X_array, index: Iterable[Hashable]) -> Dict[str, Hashable]:
    """
    ``{row hash: index label}`` for the rows of ``X_array``, in ``index`` order.

    Raises if two rows are identical: their calls could not be told apart.
    """
    keys = row_keys(np.asarray(X_array))
    labels = list(index)
    if len(keys) != len(labels):
        raise ValueError(f"{len(keys)} rows but {len(labels)} index labels")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate rows: array rows cannot identify index labels")
    return dict(zip(keys, labels))


# ----------------------------------------------------------------------
class RecordingEstimator(BaseEstimator):
    """
    Transparent wrapper that logs every fit / transform / predict call.

    Parameters
    ----------
    estimator
        Any object with sklearn-style ``fit`` and the methods you will call.
    log : CallLog
        Shared log. All clones (e.g. one per CV fold) append to it.
    """

    def __init__(self, estimator=None, log=None):
        self.estimator = estimator
        self.log = log

    # -- sklearn plumbing -------------------------------------------------
    def __sklearn_tags__(self):
        return self.estimator.__sklearn_tags__()

    @property
    def _estimator_type(self):
        return getattr(self.estimator, "_estimator_type", None)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails: delegate fitted attributes
        # (classes_, n_features_in_, feature_names_in_, ...) to the clone.
        if name.startswith("__") or name in ("estimator", "log", "estimator_", "fit_id_"):
            raise AttributeError(name)
        inner = self.__dict__.get("estimator_")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    # -- recording --------------------------------------------------------
    def _new_inner(self):
        try:
            return clone(self.estimator)
        except TypeError:          # not an sklearn estimator: copy it instead
            return copy.deepcopy(self.estimator)

    def _record(self, method: str, X) -> None:
        self.log.append(Call(method, self.fit_id_, row_keys(X)))

    def _start_fit(self, method: str, X) -> None:
        if self.log is None:
            raise ValueError("RecordingEstimator needs a CallLog (log=...)")
        self.estimator_ = self._new_inner()
        self.fit_id_ = self.log.new_fit_id()
        self._record(method, X)

    def fit(self, X, y=None, **kwargs):
        self._start_fit("fit", X)
        self.estimator_.fit(X, y, **kwargs)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        self._start_fit("fit_transform", X)
        return self.estimator_.fit_transform(X, y, **kwargs)

    def transform(self, X):
        self._record("transform", X)
        return self.estimator_.transform(X)

    def predict(self, X):
        self._record("predict", X)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        self._record("predict_proba", X)
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        self._record("decision_function", X)
        return self.estimator_.decision_function(X)


# ----------------------------------------------------------------------
def tag_rows(X, tags=None) -> np.ndarray:
    """``X`` as an array with a leading tag column (row positions by default)."""
    arr = np.asarray(X, dtype=float)
    tags = np.arange(len(arr)) if tags is None else np.asarray(tags)
    if len(tags) != len(arr):
        raise ValueError(f"{len(tags)} tags for {len(arr)} rows")
    return np.column_stack([tags, arr])


class RowTaggedEstimator(RecordingEstimator):
    """
    ``RecordingEstimator`` for arrays built with :func:`tag_rows`.

    Byte hashes cannot tell identical rows apart, and binned or encoded
    features repeat often. Here each row carries its own identity in a leading
    tag column: it is logged, then stripped, so the wrapped estimator sees
    exactly the original features and produces exactly the original output.
    """

    @staticmethod
    def _strip(X) -> np.ndarray:
        return np.asarray(X)[:, 1:]

    def _record(self, method: str, X) -> None:
        tags = tuple(int(t) for t in np.asarray(X)[:, 0])
        self.log.append(Call(method, self.fit_id_, tags))

    def fit(self, X, y=None, **kwargs):
        self._start_fit("fit", X)
        self.estimator_.fit(self._strip(X), y, **kwargs)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        self._start_fit("fit_transform", X)
        return self.estimator_.fit_transform(self._strip(X), y, **kwargs)

    def transform(self, X):
        self._record("transform", X)
        return self.estimator_.transform(self._strip(X))

    def predict(self, X):
        self._record("predict", X)
        return self.estimator_.predict(self._strip(X))

    def predict_proba(self, X):
        self._record("predict_proba", X)
        return self.estimator_.predict_proba(self._strip(X))

    def decision_function(self, X):
        self._record("decision_function", X)
        return self.estimator_.decision_function(self._strip(X))


# ----------------------------------------------------------------------
class RecordingStage3Estimator:
    """
    Transparent wrapper for a ``shared.stage3`` estimator.

    The shared Stage 3 runner talks to each arm through
    ``fit(params, X, y, sample_weight)`` (returns a model) and
    ``positive_proba(model, X)``, not the sklearn interface, so it needs its
    own recorder. Every fit and every scoring call is logged; the arm's real
    models are returned unchanged, so artifacts are unaffected. All other
    attributes (``family``, ``as_array``, ``normalize_params``, ``to_dict``,
    ...) are delegated.
    """

    def __init__(self, estimator, log: CallLog):
        self._estimator = estimator
        self.log = log
        self._models: Dict[int, tuple] = {}        # id(model) -> (model, fit_id)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("_estimator", "log", "_models"):
            raise AttributeError(name)
        return getattr(self._estimator, name)

    def fit(self, params, X, y, sample_weight=None):
        fit_id = self.log.new_fit_id()
        self.log.append(Call("fit", fit_id, row_keys(X)))
        model = self._estimator.fit(params, X, y, sample_weight=sample_weight)
        self._models[id(model)] = (model, fit_id)  # keeps the model alive: ids stay unique
        return model

    def positive_proba(self, model, X):
        _, fit_id = self._models.get(id(model), (None, -1))
        self.log.append(Call("predict_proba", fit_id, row_keys(X)))
        return self._estimator.positive_proba(model, X)
