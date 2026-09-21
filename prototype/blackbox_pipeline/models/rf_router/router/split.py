"""
Global-split guards.

Stage 2 does not create a split, re-run the preprocessor, or re-engineer
features. The RF arm must consume the same ``GLOBAL_SPLIT`` and the same
``engineer_features`` output the GLASS arm consumes; an RF-specific source of
information would make the comparison meaningless even if no test data were
involved.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["split_fingerprint", "assert_disjoint_split", "unpack_global_split"]


def split_fingerprint(X_train: pd.DataFrame, X_test: pd.DataFrame) -> str:
    """
    Stable hash of the train/test row indices.

    Recorded at fit time and re-checked at evaluation time so a router fitted
    under one split can never be scored against another. A silently regenerated
    split would put training rows into the test set and every downstream metric
    would look excellent and mean nothing.
    """
    h = hashlib.sha256()
    for name, idx in (("train", X_train.index), ("test", X_test.index)):
        h.update(name.encode())
        h.update(np.asarray(idx).astype(str).tobytes())
        h.update(str(len(idx)).encode())
    return h.hexdigest()[:16]


def assert_disjoint_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Hard stop if any row index appears in both splits."""
    overlap = X_train.index.intersection(X_test.index)
    if len(overlap) > 0:
        raise AssertionError(
            f"Train/test indices overlap in {len(overlap)} rows "
            f"(first few: {list(overlap[:5])}). The global split has been "
            "regenerated or the frames were rebuilt from different sources."
        )


def unpack_global_split(
    global_split: dict,
    engineered: Optional[dict] = None,
) -> dict:
    """
    Read the shared split, verify it, and return the frames Stage 2 uses.

    Parameters
    ----------
    global_split
        ``GLOBAL_SPLIT`` from ``GlobalSplitManager.create_split`` — expects
        ``X_train``, ``X_test``, ``y_train``, ``y_test``.
    engineered
        Optional dict carrying the already-binned 29-column frames, e.g.
        ``BRW_DATA`` with ``X_eng_train`` / ``X_eng_test``. When omitted the raw
        split frames are returned and the caller must feed the router the
        engineered ones itself.
    """
    required = ("X_train", "X_test", "y_train", "y_test")
    missing = [k for k in required if k not in global_split]
    if missing:
        raise KeyError(f"GLOBAL_SPLIT missing keys: {missing}")

    X_train = global_split["X_train"]
    X_test = global_split["X_test"]

    assert_disjoint_split(X_train, X_test)

    out = {
        "y_train": global_split["y_train"],
        "y_test": global_split["y_test"],
        "split_fingerprint": split_fingerprint(X_train, X_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    if engineered is None:
        out["X_train"], out["X_test"] = X_train, X_test
        return out

    for src in ("X_eng_train", "X_eng_test"):
        if src not in engineered:
            raise KeyError(f"engineered dict missing key '{src}'")
    Xtr, Xte = engineered["X_eng_train"], engineered["X_eng_test"]

    # The engineered frames must describe the same rows, in the same order, as
    # the split they came from. A reset_index upstream would pass a length check
    # and silently pair features with the wrong labels.
    if len(Xtr) != len(X_train) or len(Xte) != len(X_test):
        raise ValueError(
            "Engineered frames do not match the split row counts: "
            f"train {len(Xtr)} vs {len(X_train)}, "
            f"test {len(Xte)} vs {len(X_test)}"
        )
    if not Xtr.index.equals(X_train.index) or not Xte.index.equals(X_test.index):
        raise ValueError(
            "Engineered frames carry a different index from GLOBAL_SPLIT. "
            "Feature rows and label rows must line up positionally."
        )
    assert_disjoint_split(Xtr, Xte)

    out["X_train"], out["X_test"] = Xtr, Xte
    return out