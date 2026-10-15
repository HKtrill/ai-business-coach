"""
Run Stage 2 notebook cells outside Jupyter, against synthetic upstream state.

Shared by tests/test_stage2_notebook.py. Supplies every upstream name the
Stage 2 cells read (GLOBAL_SPLIT, GLASS_ROUTER_DATA, a GLASS Router bundle on
disk, the glass_pipeline fold helpers, engineer_rf_features, quiet, …) and
executes cells in one namespace in order — what a clean kernel does.

Nothing here is used by the pipeline itself.
"""

from __future__ import annotations

import hashlib
import math
import types
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from blackbox_pipeline.models.rf_router import RFRouter, make_folds, split_fingerprint

try:
    from stage2_synthetic import CONSTRAINTS_OK, make_data, router_config
except ImportError:  # tests/ is a package in some layouts
    from .stage2_synthetic import CONSTRAINTS_OK, make_data, router_config  # type: ignore

CELL_SEPARATOR = "\n# %%\n"
REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_CELLS = REPO_ROOT / "notebook" / "stage2_cells.py"
OLD_CELLS = Path(__file__).resolve().parent / "fixtures" / "stage2_cells_pre_pr35.py"


def load_cells(path: Path) -> List[Tuple[str, str]]:
    """``[(title, source), …]`` for every ``# %%`` block (file preamble dropped)."""
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    out = []
    for block in text.split(CELL_SEPARATOR)[1:]:
        title = next((ln.strip("# ").strip() for ln in block.splitlines()
                      if ln.startswith("# CELL") or "STAGE 2" in ln), "?")
        out.append((title, block))
    return out


def stub_glass_modules() -> Dict[str, types.ModuleType]:
    """``glass_pipeline.glass_router.pipeline`` with make_folds / fold_fingerprint."""
    gp = types.ModuleType("glass_pipeline")
    gr = types.ModuleType("glass_pipeline.glass_router")
    gpp = types.ModuleType("glass_pipeline.glass_router.pipeline")
    gpp.make_folds = lambda y, n, rs: make_folds(y, n, rs, True)
    gpp.fold_fingerprint = (
        lambda fid: hashlib.sha256(np.asarray(fid, dtype=np.int64).tobytes()).hexdigest()[:16]
    )
    gp.glass_router, gr.pipeline = gr, gpp
    return {"glass_pipeline": gp, "glass_pipeline.glass_router": gr,
            "glass_pipeline.glass_router.pipeline": gpp}


def upstream_namespace(artifact_dir: Path, *, n_stage_folds: int = 3,
                       n: int = 4000, seed: int = 3) -> dict:
    """
    Every name the Stage 2 cells expect from earlier notebook cells.

    The GLASS bundle is produced by a looser RF router so the "match_glass_oof"
    constraints derived from it are feasible for the RF arm.
    """
    import sys

    X, y = make_data(n=n, seed=seed)
    test_mask = np.random.default_rng(1).random(len(y)) < 0.25
    X_tr, X_te, y_tr, y_te = X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]
    fp = split_fingerprint(X_tr, X_te)

    glass_like = RFRouter(router_config(constraints=CONSTRAINTS_OK, n_oof_folds=n_stage_folds,
                                        remainder_min_size=100)).fit(X_tr, y_tr, fp)
    tp, tc, td = glass_like.predict_train_oof()
    ep, ec, ed = glass_like.predict(X_te)
    fid = np.full(len(y_tr), -1)
    for k, (_, va) in enumerate(make_folds(y_tr, n_stage_folds, 42)):
        fid[va] = k
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "split_fingerprint": fp, "index_test": X_te.index, "train_outputs_oof": True,
        "oof_protocol": {"fold_fingerprint": stub_glass_modules()[
            "glass_pipeline.glass_router.pipeline"].fold_fingerprint(fid)},
        "train_pred": tp, "train_confidence": tc, "train_decisions": td,
        "test_pred": ep, "test_confidence": ec, "test_decisions": ed,
        "training_base_rate": float(y_tr.mean()),
    }, artifact_dir / "glass_router_synthetic.joblib")

    return dict(
        sys=sys, np=np, pd=pd, Path=Path,
        GLOBAL_SPLIT={"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te},
        GLASS_ROUTER_DATA={"X_eng_train": X_tr, "X_eng_test": X_te},
        N_STAGE_FOLDS=n_stage_folds, RANDOM_STATE=42,
        GLASS_ARTIFACT_DIRS=[str(artifact_dir)],
        RF_PARAM_SOURCE="explicit",
        RF_PARAMS_EXPLICIT={"n_estimators": 40, "max_depth": None, "min_samples_leaf": 10,
                            "max_features": "sqrt", "minority_weight": 1.0,
                            "class_weight": "balanced"},
        RF_PASS2_PARAMS=None,
        engineer_rf_features=lambda a, b, c: (a, b),  # synthetic features are already bins
        quiet=lambda f, *a, **k: f(*a, **k),
    )


def run_cells(cells: List[Tuple[str, str]], ns: dict) -> dict:
    """Execute cells in order in ``ns`` (mutated and returned)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i, (title, src) in enumerate(cells):
        try:
            exec(compile(src, f"<cell {i}: {title}>", "exec"), ns)
        except Exception as exc:  # re-raise with the cell named
            raise RuntimeError(f"cell {i} ({title}) failed: {exc!r}") from exc
        finally:
            plt.close("all")
    return ns


# ----------------------------------------------------------------------
# deep comparison
# ----------------------------------------------------------------------
def _num_equal(a, b, atol):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= atol
    return a == b


def diff(a, b, path: str = "", *, atol: float = 1e-12, skip: Tuple[str, ...] = ()) -> List[str]:
    """Human-readable differences between two nested Stage 2 outputs."""
    if path in skip:
        return []
    if hasattr(a, "thresholds_") and hasattr(a, "bands_"):  # a fitted router
        return (diff(a.thresholds_.to_dict(), b.thresholds_.to_dict(), f"{path}.thresholds", atol=atol)
                + diff(a.bands_.to_dict(), b.bands_.to_dict(), f"{path}.bands", atol=atol))
    if isinstance(a, pd.DataFrame):
        if not isinstance(b, pd.DataFrame) or list(a.columns) != list(b.columns) \
                or not a.index.equals(b.index):
            return [f"{path}: frame shape/labels differ"]
        out = []
        for c in a.columns:
            out += diff(a[c].tolist(), b[c].tolist(), f"{path}[{c!r}]", atol=atol)
        return out
    if isinstance(a, pd.Series):
        if not isinstance(b, pd.Series) or not a.index.equals(b.index):
            return [f"{path}: series index differs"]
        return diff(a.to_numpy(), b.to_numpy(), path, atol=atol)
    if isinstance(a, np.ndarray):
        b = np.asarray(b)
        if a.shape != b.shape:
            return [f"{path}: shape {a.shape} vs {b.shape}"]
        if a.dtype.kind in "fc":
            ok = np.allclose(a, b, atol=atol, rtol=0, equal_nan=True)
        else:
            ok = np.array_equal(a, b)
        return [] if ok else [f"{path}: arrays differ"]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ {sorted(set(a) ^ set(b))}"]
        out = []
        for k in a:
            out += diff(a[k], b[k], f"{path}.{k}", atol=atol, skip=skip)
        return out
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]", atol=atol, skip=skip)
        return out
    if isinstance(a, (np.floating, float)) or isinstance(b, (np.floating, float)):
        return [] if _num_equal(float(a), float(b), atol) else [f"{path}: {a!r} vs {b!r}"]
    if isinstance(a, pd.Index):
        return [] if a.equals(b) else [f"{path}: index differs"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
