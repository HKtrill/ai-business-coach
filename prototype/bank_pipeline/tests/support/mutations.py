"""
Deliberate corruptions for negative tests.

Every function returns a NEW object and leaves its input untouched, so one
fixture can feed many mutations. The ``*_MUTATIONS`` dicts map a readable name
to a one-argument callable, ready for ``pytest.mark.parametrize``.

Public API
----------
Rows / index   shuffle_rows, reset_index, duplicate_index_label,
               replace_index_label, INDEX_MUTATIONS
Columns        drop_column, add_column, duplicate_column, swap_columns,
               rename_column, cast_column, COLUMN_MUTATIONS
Labels         permute_labels
Probabilities  corrupt_probabilities, INVALID_PROBABILITY_KINDS
Provenance     with_field
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

PandasObj = Union[pd.DataFrame, pd.Series]


# ----------------------------------------------------------------------
# rows / index
# ----------------------------------------------------------------------
def shuffle_rows(obj: PandasObj, seed: int = 0) -> PandasObj:
    """Same rows and labels, different order."""
    order = np.random.default_rng(seed).permutation(len(obj))
    if (order == np.arange(len(obj))).all():          # vanishingly rare; keep it a real change
        order = order[::-1]
    return obj.iloc[order].copy()


def reset_index(obj: PandasObj) -> PandasObj:
    """Labels replaced by positions (``RangeIndex``)."""
    return obj.reset_index(drop=True)


def duplicate_index_label(obj: PandasObj, position: int = 1) -> PandasObj:
    """The label at ``position`` replaced by the label at position 0."""
    out = obj.copy()
    labels = out.index.to_numpy().copy()
    labels[position] = labels[0]
    out.index = pd.Index(labels, name=obj.index.name)
    return out


def replace_index_label(obj: PandasObj, position: int = 0, label: Any = -1) -> PandasObj:
    """One label replaced by a label the original index does not contain."""
    if label in obj.index:
        raise ValueError(f"label {label!r} already in the index")
    out = obj.copy()
    labels = out.index.to_numpy().copy()
    labels[position] = label
    out.index = pd.Index(labels, name=obj.index.name)
    return out


INDEX_MUTATIONS: Dict[str, Callable[[PandasObj], PandasObj]] = {
    "shuffled_rows": shuffle_rows,
    "reset_index": reset_index,
    "duplicated_label": duplicate_index_label,
    "foreign_label": replace_index_label,
}


# ----------------------------------------------------------------------
# columns
# ----------------------------------------------------------------------
def drop_column(df: pd.DataFrame, col: Optional[str] = None) -> pd.DataFrame:
    """Missing column (the last one by default)."""
    return df.drop(columns=[col if col is not None else df.columns[-1]])


def add_column(df: pd.DataFrame, name: str = "unexpected", value: Any = 0.0) -> pd.DataFrame:
    """Extra column the contract does not know."""
    if name in df.columns:
        raise ValueError(f"column {name!r} already exists")
    out = df.copy()
    out[name] = value
    return out


def duplicate_column(df: pd.DataFrame, col: Optional[str] = None) -> pd.DataFrame:
    """A second column with the same name appended (the first by default)."""
    col = col if col is not None else df.columns[0]
    return pd.concat([df, df[[col]]], axis=1)


def swap_columns(df: pd.DataFrame, i: int = 0, j: int = 1) -> pd.DataFrame:
    """Same columns, two positions exchanged."""
    cols = list(df.columns)
    cols[i], cols[j] = cols[j], cols[i]
    return df[cols].copy()


def rename_column(df: pd.DataFrame, col: Optional[str] = None, new: str = "renamed") -> pd.DataFrame:
    """One column under a different name (the first by default)."""
    return df.rename(columns={col if col is not None else df.columns[0]: new})


def cast_column(df: pd.DataFrame, col: Optional[str] = None, dtype: Any = "float64") -> pd.DataFrame:
    """One column with a different dtype (the first non-float by default)."""
    if col is None:
        col = next(c for c in df.columns if df[c].dtype != np.dtype(dtype))
    out = df.copy()
    out[col] = out[col].astype(dtype)
    return out


COLUMN_MUTATIONS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "missing_column": drop_column,
    "extra_column": add_column,
    "duplicated_column": duplicate_column,
    "reordered_columns": swap_columns,
    "renamed_column": rename_column,
    "changed_dtype": cast_column,
}


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------
def permute_labels(y: pd.Series, seed: int = 0, rows: Optional[Iterable] = None) -> pd.Series:
    """
    Shuffle label values among ``rows`` (index labels; all rows by default).

    The index, the class counts and every label outside ``rows`` are
    unchanged. Use it to show that something depends, or does not depend,
    on those rows' labels, e.g. the held-out test labels.
    """
    out = y.copy()
    target = out.index if rows is None else pd.Index(list(rows))
    values = out.loc[target].to_numpy()
    rng = np.random.default_rng(seed)
    if len(np.unique(values)) > 1:                     # guarantee at least one label moves
        for _ in range(100):
            shuffled = rng.permutation(values)
            if not (shuffled == values).all():
                break
        values = shuffled
    out.loc[target] = values
    return out


# ----------------------------------------------------------------------
# probabilities
# ----------------------------------------------------------------------
INVALID_PROBABILITY_KINDS = ("nan", "inf", "above_one", "below_zero")
_BAD_VALUES = {"nan": np.nan, "inf": np.inf, "above_one": 1.0 + 1e-9, "below_zero": -1e-9}


def corrupt_probabilities(p, kind: str, position: int = 0):
    """Copy of ``p`` (Series or array) with one value made invalid."""
    if kind not in _BAD_VALUES:
        raise ValueError(f"unknown kind {kind!r}; expected one of {INVALID_PROBABILITY_KINDS}")
    if isinstance(p, pd.Series):
        out = p.astype(float).copy()
        out.iloc[position] = _BAD_VALUES[kind]
        return out
    out = np.array(p, dtype=float, copy=True)
    out[position] = _BAD_VALUES[kind]
    return out


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------
def with_field(mapping: dict, path: str, value: Any) -> dict:
    """
    Copy of a (nested) dict with one field replaced, e.g.
    ``with_field(stage_output.to_dict(), "split_fingerprint", "0" * 12)``.

    ``path`` uses dots for nesting (``"extra.oof_protocol.fold_fingerprint"``).
    Every dict on the path is copied; the input is not modified.
    """
    keys = path.split(".")
    out = copy.copy(mapping)
    node = out
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            raise KeyError(f"{path!r}: {k!r} is not a nested dict")
        node[k] = copy.copy(node[k])
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"{path!r}: no field {keys[-1]!r}")
    node[keys[-1]] = value
    return out
