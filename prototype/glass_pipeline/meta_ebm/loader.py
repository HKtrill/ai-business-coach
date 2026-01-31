import joblib
import numpy as np

def _pick(d, keys, label, path):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    raise KeyError(
        f"[load_stage_outputs] Missing '{label}' in {path}\n"
        f"Tried: {keys}\n"
        f"Available: {sorted(d.keys())}"
    )

def load_stage_outputs(lr_path, glass_path, ebm_path):
    lr = joblib.load(lr_path)
    glass = joblib.load(glass_path)
    ebm = joblib.load(ebm_path)

    # --- labels (prefer LR artifact, fallback if needed) ---
    y_train = np.array(_pick(lr, ["y_train", "train_labels", "labels_train"], "y_train", lr_path))
    y_test  = np.array(_pick(lr, ["y_test", "test_labels", "labels_test"], "y_test", lr_path))

    n_train = len(y_train)
    n_test = len(y_test)

    # --- LR probs + threshold ---
    lr_prob_train = np.array(_pick(lr,
        ["train_predictions_calibrated", "train_predictions", "train_proba", "proba_train"],
        "lr_prob_train", lr_path
    ))
    lr_prob_test = np.array(_pick(lr,
        ["test_predictions_calibrated", "test_predictions", "test_proba", "proba_test"],
        "lr_prob_test", lr_path
    ))
    lr_thresh = float(_pick(lr,
        ["optimal_threshold_calibrated", "optimal_threshold", "threshold"],
        "lr_thresh", lr_path
    ))

    # --- EBM probs + threshold ---
    ebm_prob_train = np.array(_pick(ebm,
        ["train_predictions_calibrated", "train_predictions", "train_proba", "proba_train"],
        "ebm_prob_train", ebm_path
    ))
    ebm_prob_test = np.array(_pick(ebm,
        ["test_predictions_calibrated", "test_predictions", "test_proba", "proba_test"],
        "ebm_prob_test", ebm_path
    ))
    ebm_thresh = float(_pick(ebm,
        ["optimal_threshold", "threshold", "optimal_threshold_calibrated"],
        "ebm_thresh", ebm_path
    ))

    # --- GLASS decisions (always full array) ---
    glass_decisions_train = np.array(_pick(glass,
        ["train_decisions", "decisions_train", "train_routes"],
        "glass_decisions_train", glass_path
    ))
    glass_decisions_test = np.array(_pick(glass,
        ["test_decisions", "decisions_test", "test_routes"],
        "glass_decisions_test", glass_path
    ))

    # --- GLASS probs ---
    # May be stored as (N,2) or (N,) or as SUBSET of covered samples only
    gtrain = _pick(glass, ["train_proba", "proba_train", "train_predictions", "train_prob"], "glass_prob_train", glass_path)
    gtest  = _pick(glass, ["test_proba", "proba_test", "test_predictions", "test_prob"], "glass_prob_test", glass_path)

    gtrain = np.array(gtrain)
    gtest  = np.array(gtest)

    # Extract positive class probability if 2D
    if gtrain.ndim == 2 and gtrain.shape[1] == 2:
        gtrain = gtrain[:, 1]
    else:
        gtrain = gtrain.astype(float)

    if gtest.ndim == 2 and gtest.shape[1] == 2:
        gtest = gtest[:, 1]
    else:
        gtest = gtest.astype(float)

    # --- CRITICAL FIX: Expand GLASS probs if they're subset ---
    # GLASS only covers pass1/pass2 samples, not abstain
    # If stored probs are subset, expand to full array with 0.5 for uncovered
    
    covered_train = (glass_decisions_train == "pass1") | (glass_decisions_train == "pass2")
    covered_test = (glass_decisions_test == "pass1") | (glass_decisions_test == "pass2")
    
    n_covered_train = covered_train.sum()
    n_covered_test = covered_test.sum()

    # Train probs
    if len(gtrain) == n_covered_train and n_covered_train < n_train:
        # Stored as subset - expand to full array
        print(f"   [load_stage_outputs] Expanding glass_prob_train: {len(gtrain)} → {n_train}")
        glass_prob_train = np.full(n_train, 0.5, dtype=float)
        glass_prob_train[covered_train] = gtrain
    elif len(gtrain) == n_train:
        # Already full array
        glass_prob_train = gtrain
    else:
        raise ValueError(
            f"[load_stage_outputs] glass_prob_train has unexpected shape: {len(gtrain)}\n"
            f"Expected {n_train} (full) or {n_covered_train} (covered only)"
        )

    # Test probs
    if len(gtest) == n_covered_test and n_covered_test < n_test:
        # Stored as subset - expand to full array
        print(f"   [load_stage_outputs] Expanding glass_prob_test: {len(gtest)} → {n_test}")
        glass_prob_test = np.full(n_test, 0.5, dtype=float)
        glass_prob_test[covered_test] = gtest
    elif len(gtest) == n_test:
        # Already full array
        glass_prob_test = gtest
    else:
        raise ValueError(
            f"[load_stage_outputs] glass_prob_test has unexpected shape: {len(gtest)}\n"
            f"Expected {n_test} (full) or {n_covered_test} (covered only)"
        )

    return {
        "y_train": y_train,
        "y_test": y_test,
        "lr_prob_train": lr_prob_train,
        "lr_prob_test": lr_prob_test,
        "lr_thresh": lr_thresh,
        "ebm_prob_train": ebm_prob_train,
        "ebm_prob_test": ebm_prob_test,
        "ebm_thresh": ebm_thresh,
        "glass_prob_train": glass_prob_train,
        "glass_prob_test": glass_prob_test,
        "glass_decisions_train": glass_decisions_train,
        "glass_decisions_test": glass_decisions_test,
    }