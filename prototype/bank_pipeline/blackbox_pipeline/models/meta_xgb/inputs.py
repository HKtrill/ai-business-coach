"""
blackbox_pipeline.models.meta_xgb.inputs
========================================
Black-box Stage 1–3 artifacts → verified ``Stage4Inputs`` → the Meta-XGB
feature block.

Reading and checking is shared with the GLASS arm (``shared.stage4``):

    mlp        Stage 1  MLP StageOutput          stage_output_stream
                        (``MLP_S1_PATH`` StageOutput file, or the
                        ``MLP_ARTIFACT_PATH`` dict holding ``stage_output``)
    rf_router  Stage 2  RF Router two-pass       router_stream
                        (``rf_router_two_pass.joblib``; Stage 4 block under
                        ``"extra"``: train_outputs_oof=True, outer cross-fit
                        train_*, full-train test_*)
    xgb        Stage 3  XGBoost Stage3Artifact   stage3_stream

``build_feature_frames`` turns the verified streams into the Meta-XGB matrix
(``config.STAGE4_FEATURES``) and proves, cell by cell, that every value is the
stream value it claims to be.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from shared.stage4 import (
    ABSTAIN, PASS1, PASS2, Stage4InputError, Stage4Inputs,
    build_stage4_inputs, router_stream, split_reference, stage3_stream,
    stage_output_stream,
)

from .config import STAGE4_FEATURES

STREAMS = {"stage1": "mlp", "stage2": "rf_router", "stage3": "xgb"}


def load_blackbox_stage4_inputs(
    GLOBAL_SPLIT: dict,
    mlp_path,
    rf_path,
    xgb_path,
    verbose: bool = True,
    extra_fingerprints: Optional[dict] = None,
) -> Stage4Inputs:
    """Load + verify the black-box Stage 1–3 outputs (paths or loaded objects)."""
    ref = split_reference(GLOBAL_SPLIT, extra_fingerprints)
    s1, c1 = stage_output_stream(mlp_path, "mlp", "MLP", len(ref.y_train))
    s2, c2 = router_stream(rf_path, "rf_router", "RF Router (two-pass)", ref.y_train)
    s3, c3 = stage3_stream(xgb_path, "xgb", "XGBoost")
    return build_stage4_inputs(GLOBAL_SPLIT, [s1, s2, s3], extra_checks=c1 + c2 + c3,
                               verbose=verbose, extra_fingerprints=extra_fingerprints)


def _side(inputs: Stage4Inputs, side: str) -> pd.DataFrame:
    s1, s2, s3 = (inputs[STREAMS[k]] for k in ("stage1", "stage2", "stage3"))
    dec = getattr(s2, f"{side}_decisions")
    p2 = np.asarray(getattr(s2, f"{side}_proba"), dtype=float)
    idx = inputs.index_train if side == "train" else inputs.index_test
    frame = pd.DataFrame({
        "stage1_proba": np.asarray(getattr(s1, f"{side}_proba"), dtype=float),
        "stage2_proba": np.where(dec == ABSTAIN, np.nan, p2),
        "stage2_pass1": (dec == PASS1).astype(np.int8),
        "stage2_pass2": (dec == PASS2).astype(np.int8),
        "stage2_abstain": (dec == ABSTAIN).astype(np.int8),
        "stage3_proba": np.asarray(getattr(s3, f"{side}_proba"), dtype=float),
    }, index=idx)
    return frame[list(STAGE4_FEATURES)]


def build_feature_frames(inputs: Stage4Inputs) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    (X_train, X_test, checks). X_train holds only upstream OOF values; X_test
    only upstream full-train / refit values. Router probability is NaN where
    the router abstains (never 0.5); the one-hot route marks those rows.
    """
    X_tr, X_te = _side(inputs, "train"), _side(inputs, "test")
    s1, s2, s3 = (inputs[STREAMS[k]] for k in ("stage1", "stage2", "stage3"))
    SEC = "4f. Meta-XGB feature block"

    def same(col, arr, frame, mask=None):
        a, b = frame[col].to_numpy(dtype=float), np.asarray(arr, dtype=float)
        if mask is not None:
            a, b = a[mask], b[mask]
        return np.array_equal(a, b, equal_nan=True)

    checks = []
    for side, X, idx in (("train", X_tr, inputs.index_train), ("test", X_te, inputs.index_test)):
        dec = getattr(s2, f"{side}_decisions")
        cov = dec != ABSTAIN
        checks += [
            (SEC, "stage4", f"{side}: columns = feature contract",
             lambda X=X: list(X.columns) == list(STAGE4_FEATURES), ", ".join(STAGE4_FEATURES)),
            (SEC, "stage4", f"{side}: index = GLOBAL_SPLIT X_{side}",
             lambda X=X, idx=idx: X.index.equals(idx), f"{len(idx):,} rows"),
            (SEC, "stage4", f"{side}: stage1_proba = {getattr(s1, f'{side}_source')}",
             lambda X=X, side=side: same("stage1_proba", getattr(s1, f"{side}_proba"), X), ""),
            (SEC, "stage4", f"{side}: stage3_proba = {getattr(s3, f'{side}_source')}",
             lambda X=X, side=side: same("stage3_proba", getattr(s3, f"{side}_proba"), X), ""),
            (SEC, "stage4", f"{side}: stage2_proba = router proba on covered rows",
             lambda X=X, side=side, cov=cov: same("stage2_proba", getattr(s2, f"{side}_proba"),
                                                  X, cov), ""),
            (SEC, "stage4", f"{side}: stage2_proba missing exactly on abstain rows",
             lambda X=X, cov=cov: np.array_equal(X["stage2_proba"].isna().to_numpy(), ~cov),
             f"{int((~cov).sum()):,} abstain rows"),
            (SEC, "stage4", f"{side}: route one-hot sums to 1",
             lambda X=X: bool((X[["stage2_pass1", "stage2_pass2", "stage2_abstain"]]
                               .sum(axis=1) == 1).all()), ""),
            (SEC, "stage4", f"{side}: no other missing values",
             lambda X=X: not X.drop(columns="stage2_proba").isna().any().any(), ""),
        ]
    return X_tr, X_te, checks
