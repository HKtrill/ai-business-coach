"""
upstream_regeneration.inventory
===============================
The SHA-256 inventory of the model folders (execution contract section 7.5,
check 5) and what one notebook execution may change in it.

``snapshot`` lists every file under ``models/`` and ``blackbox_pipeline/models/``
(Python sources and caches excluded: ``blackbox_pipeline/models`` is also a
package) with its size and SHA-256. ``classify`` takes the difference between
two snapshots around one notebook's execution and accepts only:

* additions matching that notebook's registered outputs (``OUTPUTS``): each
  role's immutable version exactly once, and its sidecars;
* its companions (``COMPANIONS``), each exactly once: files its active cells
  always write that nothing downstream reads, so they are never pinned (the
  GLASS Cell 6 baseline LR);
* the documented mutable aliases its producers replace on every save
  (``MUTABLE_ALIASES``: the ``save_versioned`` latest copies of the Stage 1
  MLP and their Git-tracked CSVs, F12 / D7, and the Stage 3 ``*_latest`` copies).

Anything else — a deleted file, any other modified file, an unregistered
addition (for example in a legacy router folder or ``models/pr38``) — is a
problem that stops the run. Nothing here reads a file's content beyond hashing it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

__all__ = ["INVENTORY_ROOTS", "OUTPUTS", "COMPANIONS", "SIDECARS", "MUTABLE_ALIASES",
           "PROTECTED", "RUN", "snapshot", "diff", "classify"]

BANK_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_ROOTS = ("models", "blackbox_pipeline/models")
_TS = r"[0-9]{8}_[0-9]{6}"
_TS_US = r"[0-9]{8}_[0-9]{6}_[0-9]{6}"

#: Per notebook: role -> the immutable version its producer writes (exactly one per run).
OUTPUTS = {
    "glass": {
        "stage1.lr": rf"models/lr/lr_calibrated_{_TS}\.joblib",
        "stage3.ebm": rf"models/ebm/ebm_stage3_{_TS_US}\.joblib",
    },
    "blackbox": {
        "stage1.mlp": rf"models/mlp/stage1_mlp_{_TS_US}\.joblib",
        "stage3.xgb": rf"blackbox_pipeline/models/xgb/xgb_stage3_{_TS_US}\.joblib",
    },
}
#: Per notebook: files an active cell writes on every run that no later cell or stage reads
#: (never pinned), each required exactly once. GLASS Cell 6 (train_baseline_lr_stage,
#: save=True): the uncalibrated reference LR, glass_pipeline.lr.lr_stage.LRStage.save.
COMPANIONS = {
    "glass": {"stage1.lr_baseline": rf"models/lr/lr_baseline_{_TS}\.joblib"},
    "blackbox": {},
}
#: Per notebook: other files a producer adds beside its version (never consumed by role).
SIDECARS = {
    "glass": (rf"models/ebm/ebm_stage3_{_TS_US}\.json",
              # GLASS Cell 17: the Stage 3 block cache, named by its identity digest (PR 39)
              r"models/stage3/stage3_block_[0-9a-f]+_[0-9a-f]{12}\.joblib"),
    "blackbox": (rf"models/mlp/stage1_mlp_{_TS_US}"
                 r"(_metadata\.json|_threshold_sweep\.csv|_tuning_trials\.csv|_training_history\.csv)",
                 # save_versioned keeps a pre-PR 37 unversioned alias before replacing it
                 r"models/mlp/stage1_mlp_preserved_[0-9a-f]{12}.*",
                 rf"blackbox_pipeline/models/xgb/xgb_stage3_{_TS_US}\.json"),
}
#: Per notebook: mutable latest aliases its producers replace (added if absent before).
MUTABLE_ALIASES = {
    "glass": ("models/ebm/ebm_stage3_latest.joblib",),
    "blackbox": ("models/mlp/stage1_mlp.joblib", "models/mlp/stage1_mlp_metadata.json",
                 "models/mlp/stage1_mlp_threshold_sweep.csv",
                 "models/mlp/stage1_mlp_tuning_trials.csv",
                 "models/mlp/stage1_mlp_training_history.csv",
                 "blackbox_pipeline/models/xgb/xgb_stage3_latest.joblib"),
}
#: Never added to, modified or deleted by a PR 39 Commit 4A run (named in the problems).
PROTECTED = (r"models/pr38/.*", r"models/pr39/.*", r"models/glass_router/.*", r"models/rf/.*",
             r"models/rf_router/.*", r"models/meta_ebm/.*", r"models/glass_arbiter/.*",
             r"blackbox_pipeline/models/meta_xgb/.*", r"blackbox_pipeline/models/stage4_crossover/.*")


def _skip(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in (".py", ".pyc", ".pyo")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root: Path = BANK_ROOT) -> Dict[str, Dict[str, Any]]:
    """``{relative path: {"bytes", "sha256"}}`` of every artifact file in the model folders."""
    root = Path(root)
    out = {}
    for top in INVENTORY_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not _skip(p.relative_to(root)):
                out[p.relative_to(root).as_posix()] = {"bytes": p.stat().st_size,
                                                       "sha256": _sha256(p)}
    return out


def diff(before: Mapping[str, Mapping], after: Mapping[str, Mapping]) -> Dict[str, List[str]]:
    return {"added": sorted(set(after) - set(before)),
            "modified": sorted(k for k in set(before) & set(after)
                               if before[k]["sha256"] != after[k]["sha256"]),
            "deleted": sorted(set(before) - set(after))}


def _match(patterns, rel: str) -> bool:
    return any(re.fullmatch(p, rel) for p in patterns)


#: The whole run (both notebooks): what ``finalize`` holds the run's net change to.
RUN = "run"


def classify(before: Mapping[str, Mapping], after: Mapping[str, Mapping],
             notebook: str) -> Dict[str, Any]:
    """
    What one notebook's execution (or, for ``RUN``, the whole run) changed, by
    role; ``problems`` empty means accepted.
    """
    if notebook == RUN:
        outputs = {r: p for nb in OUTPUTS for r, p in OUTPUTS[nb].items()}
        companions = {r: p for nb in COMPANIONS for r, p in COMPANIONS[nb].items()}
        sidecars = tuple(p for nb in SIDECARS for p in SIDECARS[nb])
        aliases = {a for nb in MUTABLE_ALIASES for a in MUTABLE_ALIASES[nb]}
    elif notebook in OUTPUTS:
        outputs, sidecars = OUTPUTS[notebook], SIDECARS[notebook]
        companions = COMPANIONS[notebook]
        aliases = set(MUTABLE_ALIASES[notebook])
    else:
        raise ValueError(f"unknown notebook {notebook!r}: expected one of {sorted(OUTPUTS)} or "
                         f"{RUN!r}")
    d = diff(before, after)
    produced: Dict[str, List[str]] = {role: [] for role in outputs}
    company: Dict[str, List[str]] = {name: [] for name in companions}
    side, alias, problems = [], [], []
    for rel in d["added"]:
        role = next((r for r, pat in outputs.items() if re.fullmatch(pat, rel)), None)
        name = next((c for c, pat in companions.items() if re.fullmatch(pat, rel)), None)
        if role is not None:
            produced[role].append(rel)
        elif name is not None:
            company[name].append(rel)
        elif _match(sidecars, rel):
            side.append(rel)
        elif rel in aliases:
            alias.append(rel)
        else:
            problems.append(f"unregistered addition {rel}"
                            + (" (a protected folder)" if _match(PROTECTED, rel) else ""))
    for rel in d["modified"]:
        if rel in aliases:
            alias.append(rel)
        else:
            problems.append(f"existing file modified: {rel}"
                            + (" (a protected folder)" if _match(PROTECTED, rel) else ""))
    problems += [f"existing file deleted: {rel}" for rel in d["deleted"]]
    for role, found in produced.items():
        if len(found) != 1:
            problems.append(f"{role}: {len(found)} new versions ({found}), expected exactly one")
    for name, found in company.items():
        if len(found) != 1:
            problems.append(f"{name}: {len(found)} new files ({found}), expected exactly one")
    return {"notebook": notebook, "diff": {k: len(v) for k, v in d.items()},
            "produced": {r: (f[0] if len(f) == 1 else f) for r, f in produced.items()},
            "companions": {c: (f[0] if len(f) == 1 else f) for c, f in company.items()},
            "sidecars": sorted(side), "mutable_aliases": sorted(alias),
            "problems": problems, "passed": not problems}
