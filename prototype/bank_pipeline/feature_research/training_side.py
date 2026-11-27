"""
feature_research.training_side
==============================
The cascades' split, training side only (PR 42, Commit 2).

    TRAIN = draw_training_side()

``draw_training_side`` runs the one protocol every notebook runs
(``tests/parity/test_notebook_protocol_parity.py``, ``SPLIT_PROTOCOL``)::

    build_database()
    BankPreprocessor(drop_leaky=True).fit_transform()
    sanity_check_data(df_processed)                       -> target column
    GlobalSplitManager(test_size=0.20, random_state=RANDOM_SEED)
        .create_split(df_processed, target_col=target)

and returns a ``TrainingSide``: the training rows, their labels, the protocol,
train-side identities and the PR 38 ``TrainReference`` the boundary extractors
check artifacts against. Nothing it returns holds a test row, a test label or a
test row identity, so a notebook that only calls this function never binds
any of them:

* the preprocessed population, the split dictionary and the split manager
  (which keeps the split in ``_split_dict``) exist only inside the call and are
  released before it returns;
* everything those steps print (full-population and test-split class counts)
  is captured and discarded; the summary printed instead is training-side only;
* the test row index is used inside the call for exactly one thing, the
  routers' 16-hex split fingerprint (``boundary.build_train_reference``, which
  hashes row identities and no labels); ``TrainReference`` stores the
  fingerprint, not the index;
* the label-bearing ``shared.stage_io`` split fingerprint is never recomputed
  (its hash includes the test labels): the value recorded by the accepted
  PR 39 Commit 4 run is attested, as the PR 38 boundary prescribes.

The returned frames are copies, so no object they reference is the full
population.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from data.global_splits.split_manager import GlobalSplitManager
from data.preprocessing import BankPreprocessor, build_database
from feature_research.config import RANDOM_SEED
from feature_research.validation import sanity_check_data
from shared.stage2.diagnostics.boundary import build_train_reference
from shared.stage2.diagnostics.schema import TrainReference

__all__ = ["TEST_SIZE", "STAGE1_FOLDS", "SHARED_FOLDS", "ATTESTATION", "TrainingSide",
           "draw_training_side", "train_side_identities"]

BANK_ROOT = Path(__file__).resolve().parents[1]

#: The cascades' split and fold plans (notebook map section 2.6).
TEST_SIZE = 0.20
STAGE1_FOLDS = 10
SHARED_FOLDS = 5

#: Where the label-bearing ``shared.stage_io`` split fingerprint is recorded.
ATTESTATION = MappingProxyType({
    "record": "research_logs/pr39/c4_20261115_204125/run_record.json",
    "field": "identities.population.stage_io_split_fingerprint",
})


@dataclass(frozen=True)
class TrainingSide:
    """The cascades' training rows and nothing else (module docstring)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    target_col: str
    protocol: Mapping[str, Any]
    identities: Mapping[str, Any]
    train_reference: TrainReference = field(repr=False)

    @property
    def n_rows(self) -> int:
        return int(len(self.y_train))

    def frame(self) -> pd.DataFrame:
        """Training rows with the target column (a copy)."""
        return self.X_train.assign(**{self.target_col: self.y_train})

    def describe(self) -> str:
        ids = self.identities
        return (f"training rows {ids['train_rows']:,} | positives {ids['train_positives']:,} "
                f"({ids['train_positive_rate']:.4f}) | features {self.X_train.shape[1]}\n"
                f"label fingerprint {ids['train_label_fingerprint']} | "
                f"shared {SHARED_FOLDS}-fold {ids['shared_fold_fingerprint']} | "
                f"Stage 1 {STAGE1_FOLDS}-fold {ids['stage1_fold_fingerprint']}\n"
                f"router split fingerprint {ids['router_split_fingerprint']} (row ids only) | "
                f"stage_io split fingerprint {ids['stage_io_split_fingerprint']} (attested)")


def _attested_stage_io_fingerprint(root: Path = BANK_ROOT) -> str:
    """The accepted run's recorded value, after checking its ``.sha256`` sidecar."""
    path = Path(root) / ATTESTATION["record"]
    data = path.read_bytes()
    side = path.with_name(path.name + ".sha256")
    if side.read_text(encoding="ascii").strip() != hashlib.sha256(data).hexdigest():
        raise RuntimeError(f"{path} does not match its .sha256 sidecar")
    value: Any = json.loads(data.decode("utf-8"))
    for key in ATTESTATION["field"].split("."):
        value = value[key]
    return str(value)


def train_side_identities(X_train: pd.DataFrame, y_train: pd.Series,
                          ref: TrainReference) -> Dict[str, Any]:
    """Identities computable from the training rows (and the reference's fingerprints)."""
    from glass_pipeline.glass_router.pipeline import fold_fingerprint, make_folds
    from shared.stage_io import fold_assignment

    y = y_train.astype(int)
    fold5 = np.full(len(y), -1)
    for k, (_, va) in enumerate(make_folds(y, SHARED_FOLDS, RANDOM_SEED)):
        fold5[va] = k
    fold10 = fold_assignment(y, cv_folds=STAGE1_FOLDS, random_state=RANDOM_SEED).to_numpy()
    return {
        "train_rows": int(len(y)),
        "train_positives": int(y.sum()),
        "train_positive_rate": float(y.mean()),
        "train_label_fingerprint": ref.train_label_fingerprint,
        "router_split_fingerprint": ref.router_split_fingerprint,
        "stage_io_split_fingerprint": ref.stage_io_split_fingerprint,
        "stage_io_attestation": ref.stage_io_attestation,
        "shared_fold_fingerprint": fold_fingerprint(fold5),
        "stage1_fold_fingerprint": fold_fingerprint(fold10),
    }


def draw_training_side(*, verbose: bool = True, indices_only: bool = False) -> TrainingSide:
    """
    The cascades' split, returned without its test side (module docstring).

    ``indices_only=True`` partitions identifiers first and selects only training values.
    This avoids the legacy split manager's descriptive report on both split sides.

    ``verbose`` prints the training-side summary; the protocol's own console
    output is never shown (it reports full-population and test-split counts).
    """
    attested = _attested_stage_io_fingerprint()
    with contextlib.redirect_stdout(io.StringIO()):
        build_database()
        population = BankPreprocessor(drop_leaky=True).fit_transform()
        target_col = sanity_check_data(population)
        if indices_only:
            # Same stratification and ordering as create_split. Only row identifiers
            # cross the split before selecting the training frame.
            from sklearn.model_selection import train_test_split
            train_idx, other_idx = train_test_split(
                population.index, stratify=population[target_col],
                test_size=TEST_SIZE, random_state=RANDOM_SEED)
            rows = population.loc[train_idx]
            X_train = rows.drop(columns=target_col)
            y_train = rows[target_col].copy()
        else:
            split = GlobalSplitManager(test_size=TEST_SIZE, random_state=RANDOM_SEED).create_split(
                population, target_col=target_col)
            X_train = split["X_train"].copy()
            y_train = split["y_train"].copy()
            other_idx = split["X_test"].index
            del split
        ref = build_train_reference(
            index_train=X_train.index, y_train=y_train, index_test=other_idx,
            attested_stage_io_split_fingerprint=attested,
            attestation_source=f"{ATTESTATION['record']} ({ATTESTATION['field']})")
        del other_idx, population
    protocol = MappingProxyType({
        "database": "data.preprocessing.build_database()",
        "population": "data.preprocessing.BankPreprocessor(drop_leaky=True).fit_transform()",
        "target": f"feature_research.validation.sanity_check_data -> {target_col!r}",
        "split": ("stratified row indices; same test_size and random_state as create_split" if indices_only else
                  f"data.global_splits.split_manager.GlobalSplitManager(test_size={TEST_SIZE}, "
                  f"random_state={RANDOM_SEED}).create_split(population, "
                  f"target_col={target_col!r})"),
        "kept": "X_train, y_train (copies); TrainReference (fingerprints only)",
        "not_kept": "the population, X_test, y_test, the test row index, the split manager",
        "fold_plans": {"stage1": [STAGE1_FOLDS, RANDOM_SEED],
                       "stages2_3": [SHARED_FOLDS, RANDOM_SEED]},
    })
    side = TrainingSide(X_train=X_train, y_train=y_train, target_col=target_col,
                        protocol=protocol,
                        identities=MappingProxyType(train_side_identities(X_train, y_train, ref)),
                        train_reference=ref)
    if verbose:
        print(side.describe())
    return side
