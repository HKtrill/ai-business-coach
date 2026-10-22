"""
F11 / F12 — both research notebooks run ONE experimental protocol
(PR 36, Commit 3 Step 3B). The notebooks are read as source, never executed.

GLOBAL_SPLIT
    Both arms: ``build_database()`` -> ``BankPreprocessor(drop_leaky=True)
    .fit_transform()`` -> ``GlobalSplitManager(test_size=0.20, random_state=42)
    .create_split(df_processed, target_col="y")``; assigned once, never
    mutated. Two independent builds under that protocol are the same split
    under every fingerprint scheme.
Fold plans
    Stage 1 (GLASS LR, black-box MLP): 10 stratified folds, seed 42.
    Stages 2-3 (both routers' outer cross-fit, GLASS EBM, black-box XGBoost)
    and the black-box meta partition: 5 stratified folds, seed 42.
    The two counts differ BY DESIGN; Stage 4 checks each stream against its
    own (tests/parity/test_stage4_split_fold_provenance.py). Every 5-fold maker
    either arm calls gives the same fold rows for the same labels, so "same
    (k, seed)" means "same partition".

Not repeated here: the split manager's own guarantees
(tests/data/test_split_manager.py) and each arm's single-notebook protocol
checks (tests/stage1, tests/stage2, tests/stage3).
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from blackbox_pipeline.models.meta_xgb import MetaXGBConfig
from blackbox_pipeline.models.mlp.config import Stage1MLPConfig
from blackbox_pipeline.models.rf_router import make_folds as rf_make_folds
from blackbox_pipeline.models.rf_router import split_fingerprint as router_split_fingerprint
from blackbox_pipeline.models.xgb import XGBStage3Config
from data.global_splits.split_manager import GlobalSplitManager
from glass_pipeline.glass_router.pipeline import fold_fingerprint
from glass_pipeline.glass_router.pipeline import make_folds as glass_make_folds
from glass_pipeline.lr import lr_stage_calibrated
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from shared.stage3.config import Stage3Config
from shared.stage3.data import FoldPlan
from shared.stage_io import fold_assignment, split_fingerprint
from tests.support.notebooks import BANK_ROOT
from tests.support.synthetic import make_bank_frame

NOTEBOOKS = {"glass": BANK_ROOT / "glass_pipeline" / "glass_cascade.ipynb",
             "blackbox": BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"}

SPLIT_PROTOCOL = {
    "database": ["build_database()"],
    "population": ("BankPreprocessor", {"drop_leaky": True}, "fit_transform()"),
    "split": ("GlobalSplitManager", {"test_size": 0.20, "random_state": 42}),
    "create_split": {"target_col": "y"},
}
FOLD_PROTOCOL = {"stage1": (10, 42), "stage2": (5, 42), "stage3": (5, 42)}
FOLD_KEYS = {"n_tune_folds", "random_state", "stratify"}
STAGE3_CONFIGS = {"Stage3Config": Stage3Config, "XGBStage3Config": XGBStage3Config}


class Notebook:
    """Code cells as one AST forest (IPython magics and shell lines dropped)."""

    def __init__(self, path: Path):
        cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
        trees = [ast.parse("\n".join(line for line in "".join(c["source"]).splitlines()
                                     if not line.lstrip().startswith(("%", "!"))))
                 for c in cells if c["cell_type"] == "code"]
        self.nodes = [n for tree in trees for n in ast.walk(tree)]
        self.assigned: dict = {}
        for n in self.nodes:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                self.assigned.setdefault(n.targets[0].id, []).append(n.value)

    def one(self, name: str) -> ast.AST:
        values = self.assigned.get(name, [])
        assert len(values) == 1, f"{name} is assigned {len(values)} times"
        return values[0]

    def value(self, node: ast.AST):
        """A literal, following names assigned exactly once."""
        return self.value(self.one(node.id)) if isinstance(node, ast.Name) else ast.literal_eval(node)

    def calls(self, name: str) -> list:
        return [n for n in self.nodes if isinstance(n, ast.Call)
                and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == name]

    def kwargs(self, call: ast.Call, keys) -> dict:
        return {k.arg: self.value(k.value) for k in call.keywords if k.arg in keys}


@pytest.fixture(scope="module", params=sorted(NOTEBOOKS))
def notebook(request):
    return request.param, Notebook(NOTEBOOKS[request.param])


# ----------------------------------------------------------------------
# GLOBAL_SPLIT
# ----------------------------------------------------------------------
def split_protocol(nb: Notebook) -> dict:
    call = nb.one("GLOBAL_SPLIT")
    quiet = getattr(call.func, "id", None) == "quiet"               # quiet(fn, *args, **kwargs)
    fn, args = (call.args[0], call.args[1:]) if quiet else (call.func, call.args)
    assert isinstance(fn, ast.Attribute) and fn.attr == "create_split" and len(args) == 1
    manager, data = nb.one(fn.value.id), nb.one(args[0].id)
    assert isinstance(data.func, ast.Attribute) and isinstance(data.func.value, ast.Name)
    pre = nb.one(data.func.value.id)
    return {
        "database": sorted(ast.unparse(c) for c in nb.calls("build_database")),
        "population": (ast.unparse(pre.func),
                       nb.kwargs(pre, {k.arg for k in pre.keywords} - {"verbose"}),
                       f"{data.func.attr}({', '.join(map(ast.unparse, data.args + data.keywords))})"),
        "split": (ast.unparse(manager.func), nb.kwargs(manager, {k.arg for k in manager.keywords})),
        "create_split": nb.kwargs(call, {k.arg for k in call.keywords}),
    }


ALIASES = {"GLOBAL_SPLIT", "X_train", "X_test", "y_train", "y_test"}
MUTATORS = {"update", "pop", "popitem", "setdefault", "clear", "drop", "fillna", "insert",
            "rename", "reset_index", "set_index", "sort_index", "sort_values"}


def split_mutations(nb: Notebook) -> list:
    """Writes into GLOBAL_SPLIT or the frames it holds (bound under their GLASS names)."""
    root = lambda t: ast.unparse(t).split("[")[0].split(".")[0]             # noqa: E731
    found = []
    for n in nb.nodes:
        targets = (n.targets if isinstance(n, (ast.Assign, ast.Delete))
                   else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else [])
        found += [ast.unparse(n) for t in targets
                  if isinstance(t, (ast.Subscript, ast.Attribute)) and root(t) in ALIASES]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and root(n.func.value) in ALIASES:
            if n.func.attr in MUTATORS or any(k.arg == "inplace" for k in n.keywords):
                found.append(ast.unparse(n))
    return found


def test_both_notebooks_build_global_split_from_one_population_under_one_protocol(notebook):
    name, nb = notebook
    assert split_protocol(nb) == SPLIT_PROTOCOL, name
    assert split_mutations(nb) == [], name


def test_two_independent_builds_under_that_protocol_are_one_split():
    """What each arm's GLOBAL_SPLIT cell does, run twice on one population."""
    frame = make_bank_frame(n=600, seed=4)
    _, kwargs = SPLIT_PROTOCOL["split"]
    a, b = (GlobalSplitManager(**kwargs).create_split(frame.copy(), **SPLIT_PROTOCOL["create_split"])
            for _ in range(2))

    def identity(gs):
        ytr, yte = gs["y_train"].astype(int), gs["y_test"].astype(int)
        return (split_fingerprint(gs["X_train"].index, gs["X_test"].index, ytr, yte),
                router_split_fingerprint(gs["X_train"], gs["X_test"]),
                fold_fingerprint(fold_assignment(ytr, cv_folds=5, random_state=42).to_numpy()))
    assert identity(a) == identity(b)
    assert len(a["X_test"]) == round(0.20 * len(frame))
    assert not set(a["X_train"].index) & set(a["X_test"].index)


# ----------------------------------------------------------------------
# Fold plans
# ----------------------------------------------------------------------
def _lr_factory_protocol() -> tuple:
    """(cv_folds, random_state) the GLASS notebook's train_calibrated_lr_stage builds the LR with."""
    tree = ast.parse(inspect.getsource(lr_stage_calibrated.train_lr_stage))
    (call,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "CalibratedLRStage"]
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    defaults = inspect.signature(CalibratedLRStage).parameters
    return tuple(kw.get(k, defaults[k].default) for k in ("cv_folds", "random_state"))


def _stage3_protocol(nb: Notebook, name: str) -> tuple:
    call = nb.one(name)
    assert not {k.arg for k in call.keywords} & FOLD_KEYS, f"{name} overrides the fold protocol"
    cfg = STAGE3_CONFIGS[ast.unparse(call.func)]()
    assert cfg.stratify is True
    return cfg.n_tune_folds, cfg.random_state


def fold_protocol(name: str, nb: Notebook) -> dict:
    if name == "glass":
        (lr,) = nb.calls("train_calibrated_lr_stage")
        assert not {k.arg for k in lr.keywords} & {"cv_folds", "random_state"}
        (cf,) = nb.calls("crossfit_stage2")
        kw = {k.arg: k.value for k in cf.keywords}
        assert nb.value(kw["folds"]) is None                 # -> make_folds(y_train, n_splits, random_state)
        return {"stage1": _lr_factory_protocol(),
                "stage2": (nb.value(kw["n_splits"]), nb.value(kw["random_state"])),
                "stage3": _stage3_protocol(nb, "EBM_STAGE3_CONFIG")}
    (mlp,) = nb.calls("CalibratedStage1MLP")
    plan = nb.one("STAGE_FOLDS")
    assert ast.unparse(plan.func) == "make_folds"
    (cf,) = nb.calls("crossfit_rf_router")
    assert ast.unparse(cf.args[2]) == "STAGE_FOLDS"
    return {"stage1": tuple(nb.kwargs(mlp, ("cv_folds", "random_state"))[k]
                            for k in ("cv_folds", "random_state")),
            "stage2": (nb.value(plan.args[1]), nb.value(plan.args[2])),
            "stage3": _stage3_protocol(nb, "XGB_STAGE3_CONFIG")}


def test_corresponding_stages_declare_the_same_fold_plan_in_both_notebooks(notebook):
    name, nb = notebook
    assert fold_protocol(name, nb) == FOLD_PROTOCOL, name


def test_the_blackbox_meta_partition_is_the_stage2_3_partition():
    nb = Notebook(NOTEBOOKS["blackbox"])
    for call in nb.calls("MetaXGBConfig"):
        assert not {k.arg for k in call.keywords} & FOLD_KEYS
    cfg = MetaXGBConfig()
    assert (cfg.n_tune_folds, cfg.random_state, cfg.stratify) == (*FOLD_PROTOCOL["stage3"], True)


def test_stage1_defaults_are_one_protocol_across_arms():
    """The Stage 1 classes themselves, for runs that do not pass the notebook's values."""
    mlp = Stage1MLPConfig()
    lr = inspect.signature(CalibratedLRStage).parameters
    assert (mlp.cv_folds, mlp.random_state) == (lr["cv_folds"].default, lr["random_state"].default) \
        == FOLD_PROTOCOL["stage1"]


def _fold_id(folds, n):
    out = np.full(n, -1)
    for k, (_, va) in enumerate(folds):
        out[va] = k
    return out


def test_every_stage2_3_fold_maker_gives_the_same_fold_rows():
    frame = make_bank_frame(n=900, seed=5)
    y = GlobalSplitManager(**SPLIT_PROTOCOL["split"][1]).create_split(frame, target_col="y")["y_train"]
    k, seed = FOLD_PROTOCOL["stage2"]
    plans = {
        "shared.stage_io.fold_assignment": fold_assignment(y, cv_folds=k, random_state=seed).to_numpy(),
        "GLASS crossfit_stage2 default (make_folds)": _fold_id(glass_make_folds(y, k, seed), len(y)),
        "black-box STAGE_FOLDS / rf_router.make_folds": _fold_id(rf_make_folds(y, k, seed), len(y)),
    }
    for cls in (Stage3Config, XGBStage3Config, MetaXGBConfig):
        cfg = cls()
        plans[f"{cls.__name__} FoldPlan"] = FoldPlan.build(
            y, cfg.n_tune_folds, cfg.random_state, cfg.stratify).fold_id
    shared = plans["shared.stage_io.fold_assignment"]
    assert sorted(np.unique(shared)) == list(range(k))
    assert {n: fold_fingerprint(p) for n, p in plans.items()} == {n: fold_fingerprint(shared) for n in plans}
    stage1 = fold_assignment(y, cv_folds=FOLD_PROTOCOL["stage1"][0], random_state=seed).to_numpy()
    assert fold_fingerprint(stage1) != fold_fingerprint(shared)        # distinct partitions by design
