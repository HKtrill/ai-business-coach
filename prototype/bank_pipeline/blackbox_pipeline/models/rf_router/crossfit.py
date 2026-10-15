"""
blackbox_pipeline.models.rf_router.crossfit
============================================
Outer cross-fit of the two-pass RF router — Stage 4 train-side inputs.

Formerly the loop in notebook Cell 25B, unchanged in behaviour. Mirrors
``crossfit_stage2`` on the GLASS side. For each shared fold:

    engineer features on the other folds (target-encoded) -> bins
    -> full RFRouter fit on those bins (its own inner OOF + thresholds)
    -> route ONLY the held-out fold.

Every train row is therefore routed by features, forests AND thresholds that
never saw its label.

Hyperparameters are the one input NOT refitted per fold: the config factory
supplies globally resolved RF params. When those were tuned with CV over all of
``X_train`` (Cell 22 or the GLASS RF artifact), the held-out fold's labels took
part in choosing them. The dependence is weak under a ROC-AUC objective; it is
recorded in ``protocol["hyperparameters"]`` so it is never silent.

It becomes stronger if Stage 2 adopts objective-specialized parameters
(``tuning.ablation``), because those objectives ARE the Stage 2 operating
quantities. Before regenerating Stage 4 inputs with specialized parameters, the
tuning has to be repeated inside each outer fold (nested tuning) so no held-out
row's label influences the hyperparameters that route it. This module does not
do that yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import RFRouterConfig
from .router import RFRouter
from .router.decisions import ABSTAIN, DECISION_DTYPE, NOT_SUBSCRIBE, SUBSCRIBE, UNCERTAIN

__all__ = ["CrossfitResult", "crossfit_rf_router"]

Engineer = Callable[[pd.DataFrame, pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.DataFrame]]


@dataclass
class CrossfitResult:
    """
    Attributes
    ----------
    oof : dict
        ``pred``, ``confidence``, ``decisions``, ``proba`` over every training
        row — same keys and dtypes as the former ``RF_OOF`` dict.
    folds : pandas.DataFrame
        One row per outer fold: thresholds, remainder fraction, routing shares,
        held-out leakage, cumulative seconds — the former ``RF_OOF_FOLDS``.
    protocol : dict
        Provenance — the former ``RF_OOF_PROTOCOL``.
    """

    oof: dict
    folds: pd.DataFrame
    protocol: dict = field(default_factory=dict)


def crossfit_rf_router(
    X_raw: pd.DataFrame,
    y: pd.Series,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    engineer: Engineer,
    make_config: Callable[[], RFRouterConfig],
    split_fp: str,
    fold_fp: str,
    random_state: int,
    hyperparameter_note: str = "resolved once on full X_train (not refitted per fold)",
    verbose: bool = True,
) -> CrossfitResult:
    """
    Route every training row with a router that never saw its label.

    Parameters
    ----------
    X_raw, y
        Raw training features and labels, index-aligned.
    folds
        Positional ``(train_idx, val_idx)`` pairs — the shared ``STAGE_FOLDS``.
    engineer
        ``engineer(X_tr_raw, X_va_raw, y_tr) -> (X_tr_eng, X_va_eng)``. Fitted on
        the training folds only.
    make_config
        Zero-argument factory returning a fresh ``RFRouterConfig`` per fold.
    split_fp, fold_fp, random_state
        Provenance recorded in ``protocol``.
    """
    if not X_raw.index.equals(y.index):
        raise AssertionError("X_raw and y are not index-aligned")

    n = len(y)
    oof = {
        "pred": np.full(n, ABSTAIN, dtype=int),
        "confidence": np.zeros(n),
        "decisions": np.array([UNCERTAIN] * n, dtype=DECISION_DTYPE),
        "proba": np.zeros((n, 2)),
    }
    routed_once = np.zeros(n, dtype=int)
    rows = []
    t0 = time.time()

    for k, (tr, va) in enumerate(folds):
        if np.intersect1d(tr, va).size:
            raise AssertionError(f"outer fold {k}: train/validation overlap")

        X_tr_eng, X_va_eng = engineer(X_raw.iloc[tr], X_raw.iloc[va], y.iloc[tr])
        r_k = RFRouter(make_config()).fit(X_tr_eng, y.iloc[tr], split_fp)

        p, c, d, proba = r_k.predict_with_proba(X_va_eng)
        oof["pred"][va] = p
        oof["confidence"][va] = c
        oof["decisions"][va] = d
        oof["proba"][va] = proba
        routed_once[va] += 1

        yv = y.iloc[va].to_numpy()
        rows.append({
            "fold": k,
            "t1": r_k.thresholds_.t1,
            "t2": r_k.thresholds_.t2,
            "remainder_frac": r_k.fit_report_.remainder["fraction"],
            "pass1": (p == NOT_SUBSCRIBE).mean(),
            "pass2": (p == SUBSCRIBE).mean(),
            "abstain": (p == ABSTAIN).mean(),
            "leak": ((p == NOT_SUBSCRIBE) & (yv == 1)).sum() / max((yv == 1).sum(), 1),
            "seconds": round(time.time() - t0, 1),
        })
        if verbose:
            print(
                f"   fold {k}: t1 {r_k.thresholds_.t1:.4f} t2 {r_k.thresholds_.t2:.4f} | "
                f"pass1 {(p == NOT_SUBSCRIBE).mean():.1%} "
                f"pass2 {(p == SUBSCRIBE).mean():.1%} "
                f"abstain {(p == ABSTAIN).mean():.1%}"
            )
        del r_k

    if not (routed_once == 1).all():
        raise AssertionError(
            "Outer folds do not partition the training rows: "
            f"{int((routed_once == 0).sum())} unrouted, "
            f"{int((routed_once > 1).sum())} routed more than once"
        )

    protocol = {
        "train_outputs_oof": True,
        "n_folds": len(folds),
        "fold_fingerprint": fold_fp,
        "random_state": random_state,
        "feature_engineering": "engineer_features refit per fold; locked bins",
        "router": "full RFRouter refit per fold (inner OOF + per-fold t1/t2)",
        "pass2_population": "OOF Pass 1 remainder (RF router design)",
        "hyperparameters": hyperparameter_note,
        "test_outputs": "from the full-train router",
        "split_fingerprint": split_fp,
    }
    if verbose:
        print(f"\n⏱️  outer cross-fit: {time.time() - t0:.0f}s")
    return CrossfitResult(oof=oof, folds=pd.DataFrame(rows), protocol=protocol)
