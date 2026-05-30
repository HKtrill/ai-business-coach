"""
glass_pipeline.meta_ebm.loader
================================
Artifact loader for Stage 4 Meta-EBM.

Loads LR (calibrated), GLASS-BRW, and EBM Stage 3 artifacts and
assembles the probability arrays and thresholds needed by the arbiter.

y_train / y_test are accepted as explicit parameters (from GLOBAL_SPLIT)
rather than being extracted from artifacts — labels are always in scope
in the cascade and this avoids depending on artifact schemas storing them.

LR artifact: use the calibrated path (lr_calibrated_*.joblib), not baseline.

Artifact key contract (canonical names, newest artifacts)
----------------------------------------------------------
    train_predictions   calibrated train probabilities
    test_predictions    calibrated test probabilities
    optimal_threshold   CV-optimised F2 threshold

Legacy fallbacks are kept in _pick() so that artifacts written before the
key consolidation (which used _calibrated suffix aliases) still load cleanly.
"""

import numpy as np
import joblib


def _pick(d: dict, keys: list, label: str, path: str):
    """
    Try keys in order; return the first hit that is present and non-None.
    Raises a descriptive KeyError if none are found.

    Keys are ordered canonical-first so that new artifacts resolve on the
    first try and legacy artifacts fall through to their suffix variants.
    """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    raise KeyError(
        f"[load_stage_outputs] Missing '{label}' in {path}\n"
        f"Tried: {keys}\n"
        f"Available: {sorted(d.keys())}"
    )


def load_stage_outputs(
    lr_path:    str,
    glass_path: str,
    ebm_path:   str,
    y_train:    np.ndarray,
    y_test:     np.ndarray,
) -> dict:
    """
    Load stage artifacts and assemble arbiter inputs.

    Parameters
    ----------
    lr_path    : path to lr_calibrated_*.joblib  (NOT lr_baseline)
    glass_path : path to glass_brw_*.joblib
    ebm_path   : path to ebm_stage3_*.joblib
    y_train    : training labels from GLOBAL_SPLIT['y_train']
    y_test     : test labels from GLOBAL_SPLIT['y_test']

    Returns
    -------
    dict with keys:
        y_train, y_test,
        lr_prob_train, lr_prob_test, lr_thresh,
        ebm_prob_train, ebm_prob_test, ebm_thresh,
        glass_prob_train, glass_prob_test,
        glass_decisions_train, glass_decisions_test
    """
    lr    = joblib.load(lr_path)
    glass = joblib.load(glass_path)
    ebm   = joblib.load(ebm_path)

    n_train = len(y_train)
    n_test  = len(y_test)

    # ------------------------------------------------------------------
    # LR — canonical names first, legacy _calibrated suffix as fallback
    # ------------------------------------------------------------------
    lr_prob_train = np.array(_pick(lr,
        ["train_predictions", "train_predictions_calibrated", "train_proba"],
        "lr_prob_train", lr_path,
    ))
    lr_prob_test = np.array(_pick(lr,
        ["test_predictions", "test_predictions_calibrated", "test_proba"],
        "lr_prob_test", lr_path,
    ))
    lr_thresh = float(_pick(lr,
        ["optimal_threshold", "optimal_threshold_calibrated", "threshold"],
        "lr_thresh", lr_path,
    ))

    # ------------------------------------------------------------------
    # EBM — canonical names first, legacy _calibrated suffix as fallback
    # ------------------------------------------------------------------
    ebm_prob_train = np.array(_pick(ebm,
        ["train_predictions", "train_predictions_calibrated", "train_proba"],
        "ebm_prob_train", ebm_path,
    ))
    ebm_prob_test = np.array(_pick(ebm,
        ["test_predictions", "test_predictions_calibrated", "test_proba"],
        "ebm_prob_test", ebm_path,
    ))
    ebm_thresh = float(_pick(ebm,
        ["optimal_threshold", "threshold", "optimal_threshold_calibrated"],
        "ebm_thresh", ebm_path,
    ))

    # ------------------------------------------------------------------
    # GLASS — decisions (full array, always)
    # ------------------------------------------------------------------
    glass_decisions_train = np.array(_pick(glass,
        ["train_decisions", "decisions_train", "train_routes"],
        "glass_decisions_train", glass_path,
    ))
    glass_decisions_test = np.array(_pick(glass,
        ["test_decisions", "decisions_test", "test_routes"],
        "glass_decisions_test", glass_path,
    ))

    # ------------------------------------------------------------------
    # GLASS — probs (may be stored as subset of covered samples only)
    # ------------------------------------------------------------------
    gtrain = np.array(_pick(glass,
        ["train_proba", "proba_train", "train_predictions", "train_prob"],
        "glass_prob_train", glass_path,
    ))
    gtest = np.array(_pick(glass,
        ["test_proba", "proba_test", "test_predictions", "test_prob"],
        "glass_prob_test", glass_path,
    ))

    # Extract positive-class column if 2D
    if gtrain.ndim == 2 and gtrain.shape[1] == 2:
        gtrain = gtrain[:, 1]
    if gtest.ndim == 2 and gtest.shape[1] == 2:
        gtest = gtest[:, 1]

    gtrain = gtrain.astype(float)
    gtest  = gtest.astype(float)

    # Expand subset arrays to full length — uncovered samples get 0.5 (neutral)
    covered_train   = (glass_decisions_train == "pass1") | (glass_decisions_train == "pass2")
    covered_test    = (glass_decisions_test  == "pass1") | (glass_decisions_test  == "pass2")
    n_covered_train = covered_train.sum()
    n_covered_test  = covered_test.sum()

    if len(gtrain) == n_covered_train and n_covered_train < n_train:
        print(f"   [loader] Expanding glass_prob_train: {len(gtrain)} → {n_train}")
        glass_prob_train = np.full(n_train, 0.5, dtype=float)
        glass_prob_train[covered_train] = gtrain
    elif len(gtrain) == n_train:
        glass_prob_train = gtrain
    else:
        raise ValueError(
            f"[loader] glass_prob_train shape {len(gtrain)} unexpected. "
            f"Expected {n_train} (full) or {n_covered_train} (covered subset)."
        )

    if len(gtest) == n_covered_test and n_covered_test < n_test:
        print(f"   [loader] Expanding glass_prob_test: {len(gtest)} → {n_test}")
        glass_prob_test = np.full(n_test, 0.5, dtype=float)
        glass_prob_test[covered_test] = gtest
    elif len(gtest) == n_test:
        glass_prob_test = gtest
    else:
        raise ValueError(
            f"[loader] glass_prob_test shape {len(gtest)} unexpected. "
            f"Expected {n_test} (full) or {n_covered_test} (covered subset)."
        )

    return {
        "y_train":               y_train,
        "y_test":                y_test,
        "lr_prob_train":         lr_prob_train,
        "lr_prob_test":          lr_prob_test,
        "lr_thresh":             lr_thresh,
        "ebm_prob_train":        ebm_prob_train,
        "ebm_prob_test":         ebm_prob_test,
        "ebm_thresh":            ebm_thresh,
        "glass_prob_train":      glass_prob_train,
        "glass_prob_test":       glass_prob_test,
        "glass_decisions_train": glass_decisions_train,
        "glass_decisions_test":  glass_decisions_test,
    }