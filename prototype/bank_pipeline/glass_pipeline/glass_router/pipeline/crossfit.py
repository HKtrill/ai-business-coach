"""
glass_pipeline.glass_router.pipeline.crossfit
=============================================
Out-of-fold (cross-fitted) Stage 2 outputs for both arms of the comparison:

    GLASS Router  — two-pass symbolic rule router
    RF router     — black-box Random Forest (tuned params from train_rf_stage)

Every training row is scored by models that never saw it. Both arms share one
fold plan, so their Stage 4 inputs follow the same protocol.

Per fold k (train = other folds, held-out = fold k):
    1. RFFeatureEngineer fit on the training folds only (its KDE, curvature and
       day x month encodings are target-derived), then the locked bins applied
       to both parts.                         -> engineer_features()
    2. RF refit with the already-tuned params (no re-tuning inside folds),
       same sample-weight recipe as train_rf_stage().
    3. GLASS Router fit on the training folds only, with the fold RF as its
       rf_model (mirrors the full-train fit, which uses rf_result.model).
    4. Both arms score the held-out fold.

Test-side outputs are NOT produced here: they come from the models fitted on
the full training set, exactly as before (the "refit" block).

Protocol notes (recorded in Stage2OOF.protocol)
-----------------------------------------------
- Pass 2 population: follows glass_config.pass2_population ("full" or
  "oof_remainder"), exactly as in the full-train fit. With "oof_remainder"
  each outer-fold GLASS fit builds its own inner OOF Pass 1 remainder.
- Folds: pass the Stage 3 fold plan via `folds=` so Stage 2 and Stage 3 OOF
  blocks share folds. Default is StratifiedKFold(5, shuffle, random_state).

Public API
----------
make_folds(y, n_splits, random_state)                   -> list[(train_pos, val_pos)]
fold_fingerprint(fold_id)                               -> str
oof_vs_insample(y, p_oof, p_insample)                   -> dict (optimism check)
crossfit_stage2(X_train_raw, y_train, glass_config, rf_params, ...) -> Stage2OOF
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from glass_pipeline.glass_router.core.rule import ABSTAIN, NOT_SUBSCRIBE, SUBSCRIBE
from glass_pipeline.glass_router.rf.feature_engineering import engineer_features
from glass_pipeline.glass_router.rf.rf_training import _build_pipe, _make_sample_weights
from glass_pipeline.glass_router.rule_generator.rule_logger import RuleLogger
from .glass_router_pipeline import GlassRouterPipeline

Fold = Tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Stage2OOF:
    """
    Out-of-fold Stage 2 outputs, one row per training row, in X_train order.

        index            training-row index (same order as X_train_raw)
        fold_id          fold that held each row out (0..k-1)
        glass_pred       0 / 1 / -1 (abstain)
        glass_confidence rule precision of the matching rule (0.0 on abstain)
        glass_decisions  "pass1" / "pass2" / "uncertain"
        glass_proba      (n, 2) [P(NOT_SUBSCRIBE), P(SUBSCRIBE)]
        rf_proba         RF P(SUBSCRIBE)
        protocol         how the block was produced (stored in artifacts)
        fold_summaries   per-fold diagnostics
    """
    index: pd.Index
    fold_id: np.ndarray
    glass_pred: np.ndarray
    glass_confidence: np.ndarray
    glass_decisions: np.ndarray
    glass_proba: np.ndarray
    rf_proba: np.ndarray
    protocol: Dict[str, Any]
    fold_summaries: List[Dict[str, Any]] = field(default_factory=list)

    def glass_train_out(self) -> Dict[str, Any]:
        """GLASS OOF block in the ModelSaver output contract."""
        return {
            "pred": self.glass_pred,
            "confidence": self.glass_confidence,
            "decisions": self.glass_decisions,
            "covered": self.glass_pred != ABSTAIN,
            "abstained": self.glass_pred == ABSTAIN,
        }

    def to_frame(self) -> pd.DataFrame:
        """All OOF columns as one DataFrame indexed like X_train."""
        return pd.DataFrame(
            {
                "fold_id": self.fold_id,
                "glass_pred": self.glass_pred,
                "glass_confidence": self.glass_confidence,
                "glass_decision": self.glass_decisions,
                "glass_p_subscribe": self.glass_proba[:, 1],
                "rf_p_subscribe": self.rf_proba,
            },
            index=self.index,
        )

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.fold_summaries)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def make_folds(y: pd.Series, n_splits: int = 5, random_state: int = 42) -> List[Fold]:
    """Default fold plan: StratifiedKFold(n_splits, shuffle=True, random_state)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), np.asarray(y))]


def fold_fingerprint(fold_id: np.ndarray) -> str:
    """Short hash of the fold assignment, so Stage 4 can check both arms share folds."""
    return hashlib.sha256(np.asarray(fold_id, dtype=np.int64).tobytes()).hexdigest()[:16]


def oof_vs_insample(y, p_oof: np.ndarray, p_insample: np.ndarray) -> Dict[str, float]:
    """
    Train-side AUC of P(SUBSCRIBE) out-of-fold vs in-sample.

    The gap is the optimism Stage 4 would inherit from in-sample features
    (the Stage 3 XGBoost equivalent was +0.049).
    """
    y = np.asarray(y)
    auc_oof = float(roc_auc_score(y, p_oof))
    auc_in = float(roc_auc_score(y, p_insample))
    return {"auc_oof": auc_oof, "auc_insample": auc_in, "optimism": auc_in - auc_oof}


def _validate_folds(folds: Sequence[Fold], n: int) -> np.ndarray:
    """Every row held out exactly once; train/val disjoint. Returns fold_id."""
    fold_id = np.full(n, -1, dtype=int)
    for k, (tr, va) in enumerate(folds):
        tr, va = np.asarray(tr), np.asarray(va)
        if np.intersect1d(tr, va).size:
            raise ValueError(f"Fold {k}: train and validation positions overlap.")
        if va.min() < 0 or va.max() >= n or tr.min() < 0 or tr.max() >= n:
            raise ValueError(f"Fold {k}: positions out of range for {n} rows.")
        if (fold_id[va] != -1).any():
            raise ValueError(f"Fold {k}: some rows are held out by more than one fold.")
        fold_id[va] = k
    if (fold_id == -1).any():
        raise ValueError(f"{int((fold_id == -1).sum())} rows are never held out.")
    return fold_id


# ---------------------------------------------------------------------------
# Cross-fit
# ---------------------------------------------------------------------------

def crossfit_stage2(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    glass_config,
    rf_params: Dict[str, Any],
    folds: Optional[Sequence[Fold]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    glass_rf_model: str = "fold_rf",
    rule_logger_factory: Optional[Callable[[], RuleLogger]] = None,
    verbose: bool = False,
) -> Stage2OOF:
    """
    Produce honest OOF Stage 2 outputs for GLASS Router and the RF router.

    Parameters
    ----------
    X_train_raw : raw preprocessed training features (the input to
                  engineer_features, i.e. Cell 10's X_train — NOT the bins).
    y_train     : binary target aligned to X_train_raw (same index).
    glass_config: GlassRouterConfig used for the full-train GLASS fit.
    rf_params   : rf_result.params from train_rf_stage (tuned, reused as-is).
    folds       : optional fold plan [(train_pos, val_pos), ...] in positional
                  indices — pass the Stage 3 plan to share folds. If None,
                  make_folds(y_train, n_splits, random_state).
    glass_rf_model : "fold_rf" (default) gives each fold's GLASS fit the fold
                  RF, mirroring GlassRouterPipeline(config, rf_model=rf_result.model).
                  Use "none" if your full-train fit passes rf_model=None.
    rule_logger_factory : returns a fresh RuleLogger per fold
                  (default: logging disabled).
    verbose     : show the per-fold feature-engineering / GLASS logs.
    """
    if not X_train_raw.index.equals(y_train.index):
        raise ValueError("X_train_raw and y_train must share the same index (order included).")
    if glass_rf_model not in ("fold_rf", "none"):
        raise ValueError("glass_rf_model must be 'fold_rf' or 'none'.")
    if rule_logger_factory is None:
        rule_logger_factory = lambda: RuleLogger(enabled=False)  # noqa: E731

    n = len(X_train_raw)
    fold_source = "provided" if folds is not None else (
        f"default StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={random_state})"
    )
    folds = list(folds) if folds is not None else make_folds(y_train, n_splits, random_state)
    fold_id = _validate_folds(folds, n)
    y_arr = np.asarray(y_train)

    glass_pred = np.full(n, ABSTAIN, dtype=int)
    glass_conf = np.zeros(n)
    glass_dec = np.array(["uncertain"] * n, dtype=object)
    glass_proba = np.zeros((n, 2))
    rf_proba = np.full(n, np.nan)
    summaries: List[Dict[str, Any]] = []

    print("=" * 80)
    print(f"🔁 STAGE 2 CROSS-FIT — {len(folds)} folds | {n:,} training rows")
    print("=" * 80)

    for k, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        X_tr, X_va = X_train_raw.iloc[tr], X_train_raw.iloc[va]
        y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]

        quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with quiet:
            # 1. Feature engineering fit on training folds only
            Xtr_eng, Xva_eng = engineer_features(X_tr, X_va, y_tr)

            # 2. RF refit with tuned params (no re-tuning)
            rf_pipe = _build_pipe(rf_params, random_state, n_jobs=-1)
            rf_pipe.fit(
                Xtr_eng, y_tr,
                clf__sample_weight=_make_sample_weights(y_tr, rf_params.get("minority_weight", 1.0)),
            )
            rf_proba[va] = rf_pipe.predict_proba(Xva_eng)[:, 1]

            # 3. GLASS Router fit on training folds only
            glass = GlassRouterPipeline(
                glass_config,
                rf_model=rf_pipe.named_steps["clf"] if glass_rf_model == "fold_rf" else None,
                rule_logger=rule_logger_factory(),
            )
            glass.fit(Xtr_eng, y_tr)

            # 4. Score the held-out fold
            p, c, d = glass.predict(Xva_eng)
            glass_pred[va] = p
            glass_conf[va] = c
            glass_dec[va] = d
            glass_proba[va] = glass.predict_proba(Xva_eng)

        yv = y_arr[va]
        subs = max(int((yv == SUBSCRIBE).sum()), 1)
        summary = {
            "fold": k,
            "n_train": len(tr),
            "n_val": len(va),
            "rf_auc": float(roc_auc_score(yv, rf_proba[va])) if len(np.unique(yv)) > 1 else np.nan,
            "glass_pass1_rules": len(glass.pass1_rules),
            "glass_pass2_rules": len(glass.pass2_rules),
            "pass1_rate": float((p == NOT_SUBSCRIBE).mean()),
            "pass2_rate": float((p == SUBSCRIBE).mean()),
            "abstain_rate": float((p == ABSTAIN).mean()),
            "pass1_leakage": float(((p == NOT_SUBSCRIBE) & (yv == SUBSCRIBE)).sum() / subs),
            "pass2_recall": float(((p == SUBSCRIBE) & (yv == SUBSCRIBE)).sum() / subs),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        summaries.append(summary)
        print(
            f"   fold {k}: RF AUC {summary['rf_auc']:.4f} | GLASS rules "
            f"{summary['glass_pass1_rules']}+{summary['glass_pass2_rules']} | "
            f"pass1 {summary['pass1_rate']:.1%} pass2 {summary['pass2_rate']:.1%} "
            f"abstain {summary['abstain_rate']:.1%} | leak {summary['pass1_leakage']:.1%} "
            f"| {summary['seconds']}s"
        )

    if np.isnan(rf_proba).any():
        raise RuntimeError("Some training rows received no OOF RF score.")

    subs_all = max(int((y_arr == SUBSCRIBE).sum()), 1)
    print("-" * 80)
    print(f"   Fold fingerprint         : {fold_fingerprint(fold_id)}")
    print(f"   OOF RF AUC (all rows)   : {roc_auc_score(y_arr, rf_proba):.4f}")
    print(f"   OOF GLASS pass1/pass2/abstain: "
          f"{(glass_pred == NOT_SUBSCRIBE).mean():.1%} / {(glass_pred == SUBSCRIBE).mean():.1%} / "
          f"{(glass_pred == ABSTAIN).mean():.1%}")
    print(f"   OOF GLASS Pass 1 leakage : "
          f"{((glass_pred == NOT_SUBSCRIBE) & (y_arr == SUBSCRIBE)).sum() / subs_all:.1%}")
    print(f"   OOF GLASS Pass 2 recall  : "
          f"{((glass_pred == SUBSCRIBE) & (y_arr == SUBSCRIBE)).sum() / subs_all:.1%}")
    print("=" * 80)

    protocol = {
        "train_outputs_oof": True,
        "n_folds": len(folds),
        "fold_fingerprint": fold_fingerprint(fold_id),
        "fold_source": fold_source,
        "random_state": random_state,
        "feature_engineering": "RFFeatureEngineer refit per fold; locked BINNING_STRATEGY bins",
        "rf": "tuned params reused, refit per fold with train_rf_stage sample weights (no re-tuning)",
        "glass_rf_model": glass_rf_model,
        "glass_pass2_population": getattr(glass_config, "pass2_population", "full"),
        "glass_abstain_label": "uncertain",
        "test_outputs": "from full-train refit models (not produced here)",
    }

    return Stage2OOF(
        index=X_train_raw.index,
        fold_id=fold_id,
        glass_pred=glass_pred,
        glass_confidence=glass_conf,
        glass_decisions=glass_dec,
        glass_proba=glass_proba,
        rf_proba=rf_proba,
        protocol=protocol,
        fold_summaries=summaries,
    )