"""
tests.support.artifact_audit
============================
Audit of the saved cascade artifacts (PR 36 Commit 4, P5). ``tests/realdata``
runs it on the artifacts the notebooks write (``pytest -m realdata``, after
regeneration); ``tests/support/test_artifact_audit.py`` runs it on synthetic
artifacts in the same formats. It only reads: nothing is trained, refitted,
regenerated, backfilled or deleted.

What is saved, and what the audit starts from
---------------------------------------------
The four Stage 4 arbiters of the 2x2 are the roots (``ROOTS``):

    key                   upstream   saved by                     folder (bank_pipeline/)
    meta_ebm              glass      GLASS Cell 19 (native)       models/meta_ebm
    meta_xgb              blackbox   black-box Cell 36 (native)   blackbox_pipeline/models/meta_xgb
    meta_ebm_on_blackbox  blackbox   black-box Cell 37            blackbox_pipeline/models/stage4_crossover/arbiter_on_blackbox
    meta_xgb_on_glass     glass      black-box Cell 37            blackbox_pipeline/models/stage4_crossover/meta_xgb_on_glass

Each records, in ``stage4_inputs["upstream"]`` (PR 36 F29), the path and a
content fingerprint of exactly the Stage 1-3 outputs it consumed — GLASS LR /
GLASS Router bundle / GLASS EBM or MLP / RF Router (cross-fit train outputs
and full-train test outputs in one file each) / XGBoost — so those six files
are taken from the arbiters' own records, never searched for. The Cell 38
crossover comparison and the Cell 34 Stage 3 comparison are computed, not
saved; the Cell 36 cascade comparison is saved inside the native Meta-XGB.

Selection — explicit, or newest, never a fallback
-------------------------------------------------
A JSON manifest (``{"meta_ebm": "<path>", ...}``, all four keys, paths
absolute or relative to bank_pipeline/) named by ``$BANK_ARTIFACT_MANIFEST``
fixes the roots. Without one, each root is the newest file (modification
time) matching its producer's name in its producer's folder. Either way that
file is audited and nothing else: a missing, unreadable, legacy or
inconsistent artifact fails with its path and the field at fault; the audit
never moves on to an older file.

Checks, all through the existing validators
-------------------------------------------
* ``load_arbiter``           schema and split / test-index identity (the
                             comparison module's loaders and checks).
* ``check_arbiter_identity`` the cell it claims to be, and complete recorded
                             evidence: arbiter, upstream arm, crossover flag,
                             per-stage family / arm / slot, label evidence
                             (F12), Pass 2 populations (F17: GLASS ``full``,
                             RF ``oof_remainder``), a ledger with every check
                             passed, Meta-XGB feature contract.
* ``reload_upstream``        the recorded files through the arbiter's own
                             Stage 4 loader (F12 / F17 / F21 / F25 / F29,
                             split, folds, OOF honesty), and the outputs on
                             disk are the ones the arbiter consumed.
* ``check_fresh_reload``     every audited file unpickles in a clean
                             interpreter as the type and identity recorded.
* ``check_stage3_serving`` / ``check_meta_xgb_serving``  saved models
                             reproduce their saved test probabilities.
* ``check_crossover``        both columns compare with the upstream held fixed
                             (``require_same_upstream=True``), rows differ,
                             and the Cell 36 comparison used the audited
                             Meta-EBM.
* ``unloadable_xgb_artifacts``  XGBoost files that ``XGB_PARAM_SOURCE='artifact'``
                             could pick but that need modules removed in P4.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from tests.support.notebooks import BANK_ROOT

STAGES = ("stage1", "stage2", "stage3")
NATIVE_ARM = {"meta_ebm": "glass", "meta_xgb": "blackbox"}
SLOTS = {"meta_ebm": {"stage1": "lr", "stage2": "glass", "stage3": "ebm"},
         "meta_xgb": {"stage1": "mlp", "stage2": "rf_router", "stage3": "xgb"}}
#: Normalised Pass 2 population per upstream router (F17): GLASS fits Pass 2 on the full
#: population, the RF Router on its OOF Pass 1 remainder.
PASS2_POPULATION = {"glass": "full", "blackbox": "oof_remainder"}
MANIFEST_ENV = "BANK_ARTIFACT_MANIFEST"


class AuditError(AssertionError):
    """A saved artifact failed the audit; the message names the file and the field."""


@dataclass(frozen=True)
class Root:
    arbiter: str
    upstream: str
    folder: str           # relative to bank_pipeline/
    pattern: str
    producer: str


_XO = "blackbox_pipeline/models/stage4_crossover"
_META_XGB = "meta_xgb_stage4_*.joblib"     # MetaXGBConfig.artifact_stem (asserted by the harness tests)
ROOTS = {
    "meta_ebm": Root("meta_ebm", "glass", "models/meta_ebm", "meta_ebm_*.joblib",
                     "GLASS notebook Cell 19"),
    "meta_xgb": Root("meta_xgb", "blackbox", "blackbox_pipeline/models/meta_xgb",
                     _META_XGB, "black-box notebook Cell 36"),
    "meta_ebm_on_blackbox": Root("meta_ebm", "blackbox", f"{_XO}/arbiter_on_blackbox",
                                 "meta_ebm_*.joblib", "black-box notebook Cell 37"),
    "meta_xgb_on_glass": Root("meta_xgb", "glass", f"{_XO}/meta_xgb_on_glass",
                              _META_XGB, "black-box notebook Cell 37"),
}
#: Crossover columns (one upstream, both arbiters) and rows (one arbiter, both upstreams).
COLUMNS = {"glass": ("meta_ebm", "meta_xgb_on_glass"),
           "blackbox": ("meta_ebm_on_blackbox", "meta_xgb")}
ROWS = {"meta_ebm": ("meta_ebm", "meta_ebm_on_blackbox"),
        "meta_xgb": ("meta_xgb_on_glass", "meta_xgb")}
XGB_STAGE3_FOLDER = "blackbox_pipeline/models/xgb"     # XGB_ARTIFACT_DIRS, black-box Cell 29


# ======================================================================
# the real split and the roots
# ======================================================================
def real_global_split() -> dict:
    """GLOBAL_SPLIT exactly as both notebooks build it (Cells 2 / 4 and 2 / 6); reads only."""
    from data.bank_database import DEFAULT_DB_PATH, raw_bank_exists
    if not raw_bank_exists(DEFAULT_DB_PATH):
        raise AuditError(f"no raw_bank table in {DEFAULT_DB_PATH}: run a cascade notebook's "
                         "database cell (build_database) before auditing its artifacts")
    from data.global_splits.split_manager import GlobalSplitManager
    from data.preprocessing.bank_preprocessing import BankPreprocessor
    with contextlib.redirect_stdout(io.StringIO()):
        df = BankPreprocessor(drop_leaky=True, verbose=False).fit_transform()
        return GlobalSplitManager(test_size=0.20, random_state=42).create_split(df, target_col="y")


def manifest_from_env() -> Optional[Path]:
    value = os.environ.get(MANIFEST_ENV)
    return Path(value) if value else None


def _resolve(path, root_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root_dir / p


def select_roots(root_dir=BANK_ROOT, manifest=None) -> dict:
    """``{key: path}`` of the four arbiters to audit (module docstring: Selection)."""
    root_dir = Path(root_dir)
    if manifest is not None:
        try:
            entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AuditError(f"manifest {manifest}: unreadable ({exc})") from None
        if not isinstance(entries, dict):
            raise AuditError(f"manifest {manifest}: expected {{key: path}}, got {type(entries).__name__}")
        missing, unknown = sorted(set(ROOTS) - set(entries)), sorted(set(entries) - set(ROOTS))
        if missing or unknown:
            raise AuditError(f"manifest {manifest}: needs exactly {sorted(ROOTS)}; "
                             f"missing {missing}, unknown {unknown}")
        out = {k: _resolve(entries[k], root_dir) for k in ROOTS}
        absent = [f"{k}: {p}" for k, p in out.items() if not p.is_file()]
        if absent:
            raise AuditError(f"manifest {manifest} names files that do not exist: {absent}")
        return out
    out, absent = {}, []
    for key, r in ROOTS.items():
        folder = root_dir / r.folder
        files = sorted(folder.glob(r.pattern), key=lambda p: (p.stat().st_mtime, p.name)) \
            if folder.is_dir() else []
        if files:
            out[key] = files[-1]
        else:
            absent.append(f"{key}: no {r.pattern} in {folder} (written by {r.producer})")
    if absent:
        raise AuditError("arbiter artifacts missing — regenerate them (P6) or name them in "
                         f"${MANIFEST_ENV}:\n  " + "\n  ".join(absent))
    return out


# ======================================================================
# one arbiter: load, identity, upstream
# ======================================================================
def _fields(art) -> dict:
    return art if isinstance(art, dict) else vars(art)


def load_arbiter(key: str, path, GLOBAL_SPLIT: dict, extra_fingerprints: Optional[dict] = None):
    """The arbiter artifact, schema- and split-checked by the existing loaders."""
    from shared.stage4 import check_split_identity, index_alignment, split_reference
    try:
        if ROOTS[key].arbiter == "meta_ebm":
            from blackbox_pipeline.models.meta_xgb.comparison import load_meta_ebm_for_comparison
            return load_meta_ebm_for_comparison(GLOBAL_SPLIT, str(path), extra_fingerprints)
        from blackbox_pipeline.models.meta_xgb.artifact import load_meta_xgb
        art = load_meta_xgb(str(path))
        ref = split_reference(GLOBAL_SPLIT, extra_fingerprints)
        check_split_identity(art.split_fingerprints, ref, f"{key} artifact",
                             primary=art.split_fingerprint)
        index_alignment(art.index_test, ref.index_test, f"{key} index_test")
        return art
    except Exception as exc:                    # noqa: BLE001 — every failure is a finding
        raise AuditError(f"{key} {path}: {type(exc).__name__}: {exc}") from None


def upstream_record(key: str, art) -> dict:
    record = (_fields(art).get("stage4_inputs") or {}).get("upstream")
    if not isinstance(record, dict):
        raise AuditError(f"{key}: stage4_inputs['upstream'] missing — written before PR 36 F29 "
                         "records which Stage 1-3 outputs it consumed; regenerate it")
    return record


def check_arbiter_identity(key: str, art) -> None:
    """The 2x2 cell ``key`` claims to be, with complete recorded evidence."""
    from blackbox_pipeline.models.meta_xgb.artifact import SCHEMA as META_XGB_SCHEMA
    from blackbox_pipeline.models.meta_xgb.config import STAGE4_FEATURES
    from glass_pipeline.meta_ebm.meta_stage import ARTIFACT_SCHEMA as META_EBM_SCHEMA
    from shared.stage4 import UPSTREAM
    from shared.stage4.adapters import PASS2_COMPLETE
    from shared.stage4.contract import LABEL_FINGERPRINT_SIDES, LABELS_CARRIED, LABELS_FINGERPRINTED

    r, d, problems = ROOTS[key], _fields(art), []

    def want(field, got, expected):
        if got != expected:
            problems.append(f"{field} = {got!r}, expected {expected!r}")

    if r.arbiter == "meta_ebm":
        want("schema", d.get("schema"), META_EBM_SCHEMA)
    else:
        for field, expected in (("schema", META_XGB_SCHEMA), ("stage", "stage4"),
                                ("model_family", "meta_xgb"), ("arm", "blackbox")):
            want(field, d.get(field), expected)
        want("feature_names", list(d.get("feature_names") or []), list(STAGE4_FEATURES))
        learner = d.get("learner")
        want("learner (model_family, config arm)",
             (getattr(learner, "model_family", None), (getattr(learner, "config", None) or {}).get("arm")),
             ("meta_xgb", "blackbox"))
    slots = SLOTS[r.arbiter]
    want("stage4_inputs streams", sorted((d.get("stage4_inputs") or {}).get("streams") or {}),
         sorted(slots.values()))
    rec = upstream_record(key, art)
    native = NATIVE_ARM[r.arbiter]
    for field, expected in (("arm", r.upstream), ("arbiter", r.arbiter), ("arbiter_arm", native),
                            ("crossover", r.upstream != native),
                            ("combination", f"{r.upstream} upstream -> {r.arbiter}")):
        want(f"upstream.{field}", rec.get(field), expected)
    stages = rec.get("stages") or {}
    for stage in STAGES:
        s = stages.get(stage) or {}
        for field, expected in UPSTREAM[r.upstream][stage]["want"].items():
            want(f"upstream.stages.{stage}.{field}", s.get(field), expected)
        want(f"upstream.stages.{stage}.slot", s.get("slot"), slots[stage])
        labels = s.get("label_evidence")
        if labels not in ((LABELS_FINGERPRINTED,) if stage == "stage2"
                          else (LABELS_CARRIED, LABELS_FINGERPRINTED)):
            problems.append(f"upstream.stages.{stage}.label_evidence = {labels!r} (F12)")
        for field in ("artifact_path", "content_fingerprint"):
            if not s.get(field):
                problems.append(f"upstream.stages.{stage}.{field} missing")
    pop = PASS2_POPULATION[r.upstream]
    s2 = stages.get("stage2") or {}
    want("upstream.stages.stage2.pass2_population (F17)", s2.get("pass2_population"),
         {"cross_fit": pop, "full_train": pop})
    want("upstream.stages.stage2.pass2_population_evidence (F17)",
         s2.get("pass2_population_evidence"), PASS2_COMPLETE)
    ledger = (d.get("stage4_inputs") or {}).get("validation") or []
    failed = [row.get("check") for row in ledger if not row.get("passed")]
    if failed:
        problems.append(f"stage4_inputs.validation: failed checks {failed}")
    router = slots["stage2"]
    required = [f"{router}: {side} fitted on GLOBAL_SPLIT's training rows and labels"
                for side in LABEL_FINGERPRINT_SIDES]
    required.append(f"{router}: cross-fit and full-train router share one Pass 2 population")
    absent = [name for name in required if name not in {row.get("check") for row in ledger}]
    if absent:
        problems.append(f"stage4_inputs.validation: no record of {absent} (F12 / F17)")
    if problems:
        raise AuditError(f"{key} is not the {r.upstream}-upstream {r.arbiter} artifact it "
                         "should be:\n  " + "\n  ".join(problems))


def upstream_paths(key: str, art, root_dir=BANK_ROOT) -> dict:
    """``{stage: path}`` the arbiter recorded; each must still exist."""
    stages = upstream_record(key, art).get("stages") or {}
    out, problems = {}, []
    for stage in STAGES:
        recorded = (stages.get(stage) or {}).get("artifact_path")
        if not recorded:
            problems.append(f"{stage}: no artifact_path recorded")
            continue
        out[stage] = _resolve(recorded, Path(root_dir))
        if not out[stage].is_file():
            problems.append(f"{stage}: {recorded} not found (moved, deleted or never saved)")
    if problems:
        raise AuditError(f"{key}: its recorded upstream files are not all there:\n  "
                         + "\n  ".join(problems))
    return out


def _plain(x):
    return json.loads(json.dumps(x, default=str))


def reload_upstream(key: str, art, GLOBAL_SPLIT: dict, root_dir=BANK_ROOT,
                    extra_fingerprints: Optional[dict] = None):
    """
    ``(Stage4Inputs, {stage: path})``: the recorded upstream files through the
    arbiter's own Stage 4 loader, which re-runs every contract check; the
    outputs on disk must be the outputs the arbiter consumed.
    """
    from shared.stage4 import same_upstream
    r = ROOTS[key]
    paths = upstream_paths(key, art, root_dir)
    if r.arbiter == "meta_ebm":
        from glass_pipeline.meta_ebm.loader import load_stage_outputs as loader
    else:
        from blackbox_pipeline.models.meta_xgb.inputs import load_blackbox_stage4_inputs as loader
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            inputs = loader(GLOBAL_SPLIT, *(str(paths[s]) for s in STAGES), verbose=False,
                            extra_fingerprints=extra_fingerprints, upstream=r.upstream)
    except Exception as exc:                    # noqa: BLE001 — every failure is a finding
        raise AuditError(f"{key}: its upstream files fail the Stage 4 contract today "
                         f"({', '.join(map(str, paths.values()))}):\n{exc}") from None
    stored, now = upstream_record(key, art), inputs.upstream
    problems = []
    differs = same_upstream(stored, now)
    if differs:
        problems.append(f"{differs} — the files on disk are not the outputs it consumed")
    for stage in STAGES:
        a, b = (stored["stages"].get(stage) or {}), (now["stages"].get(stage) or {})
        for field in ("slot", "model_family", "arm", "probability_space", "label_evidence",
                      "pass2_population", "pass2_population_evidence"):
            if _plain(a.get(field)) != _plain(b.get(field)):
                problems.append(f"{stage}.{field}: recorded {a.get(field)!r}, reloaded {b.get(field)!r}")
    recorded, reloaded = _fields(art)["stage4_inputs"], inputs.provenance()
    for field in ("split_fingerprint", "split_fingerprints", "n_train", "n_test", "folds"):
        if _plain(recorded.get(field)) != _plain(reloaded.get(field)):
            problems.append(f"stage4_inputs.{field}: recorded and reloaded differ")
    if problems:
        raise AuditError(f"{key}: recorded upstream vs the files on disk "
                         f"({', '.join(map(str, paths.values()))}):\n  " + "\n  ".join(problems))
    return inputs, paths


# ======================================================================
# reload in a clean interpreter
# ======================================================================
def _summary(obj) -> dict:
    """Type and identity fields of an unpickled artifact (also run in the subprocess)."""
    get = obj.get if isinstance(obj, dict) else (lambda k: getattr(obj, k, None))
    out = {"type": f"{type(obj).__module__}.{type(obj).__qualname__}"}
    for k in ("schema", "stage", "model_family", "arm", "split_fingerprint", "block_fingerprint"):
        v = get(k)
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
    for k in ("features", "feature_names"):
        v = get(k)
        if isinstance(v, (list, tuple)):
            out[k] = [str(x) for x in v]
    config = get("config")
    if isinstance(config, dict):
        out["config.arm"] = config.get("arm")
    return out


_LOAD_EACH = """
import json, sys
import joblib
out = {}
for path in json.loads(sys.stdin.read()):
    try:
        out[path] = _summary(joblib.load(path))
    except Exception as exc:
        out[path] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out, default=str))
"""


def load_in_fresh_interpreter(paths, root_dir=BANK_ROOT) -> dict:
    """``{path: summary or {"error": ...}}`` from a new Python process (nothing pre-imported)."""
    paths = [str(p) for p in paths]
    script = "from __future__ import annotations\n" + textwrap.dedent(inspect.getsource(_summary)) \
        + _LOAD_EACH
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(BANK_ROOT), os.environ.get("PYTHONPATH")]))}
    run = subprocess.run([sys.executable, "-c", script], input=json.dumps(paths),
                         capture_output=True, text=True, encoding="utf-8", cwd=str(root_dir),
                         env=env, timeout=1800)
    if run.returncode != 0:
        raise AuditError(f"fresh-interpreter reload crashed:\n{run.stderr[-3000:]}")
    return json.loads(run.stdout.strip().splitlines()[-1])


def _hint(error: str) -> str:
    if "No module named 'blackbox_pipeline.models.xgb." in error:
        return (" — written by the standalone XGBoost package removed in PR 36 Commit 4 (P4a); "
                "regenerate it on the shared protocol (black-box Cell 31) and move this file "
                "out of the producer folder")
    if "No module named" in error:
        return " — written by code that no longer exists; regenerate it"
    return ""


def check_fresh_reload(expected: dict, root_dir=BANK_ROOT) -> dict:
    """
    ``expected``: ``{path: {field: value}}`` (``_summary`` fields). Each file
    must unpickle in a clean interpreter with those values. Returns the summaries.
    """
    loaded = load_in_fresh_interpreter(list(expected), root_dir)
    problems = []
    for path, want in expected.items():
        got = loaded[str(path)]
        if "error" in got:
            problems.append(f"{path}: {got['error']}{_hint(got['error'])}")
            continue
        for field, value in want.items():
            if got.get(field) != value:
                problems.append(f"{path}: {field} = {got.get(field)!r}, expected {value!r}")
    if problems:
        raise AuditError("saved artifacts do not reload as recorded in a clean interpreter:\n  "
                         + "\n  ".join(problems))
    return loaded


def expected_reload(key: str, path, art, upstream: dict) -> dict:
    """What a clean reload of the arbiter and its upstream files must show."""
    from shared.stage4 import UPSTREAM
    r = ROOTS[key]
    if r.arbiter == "meta_ebm":
        out = {str(path): {"type": "builtins.dict", "schema": _fields(art)["schema"],
                           "split_fingerprint": _fields(art)["split_fingerprint"]}}
    else:
        out = {str(path): {"type": "blackbox_pipeline.models.meta_xgb.artifact.MetaXGBArtifact",
                           "schema": art.schema, "model_family": "meta_xgb", "arm": "blackbox",
                           "split_fingerprint": art.split_fingerprint,
                           "feature_names": [str(f) for f in art.feature_names]}}
    family, arm = (UPSTREAM[r.upstream]["stage3"]["want"][k] for k in ("model_family", "arm"))
    stage3 = _fields(art)["stage4_inputs"]["streams"][SLOTS[r.arbiter]["stage3"]]
    out[str(upstream["stage3"])] = {"type": "shared.stage3.artifact.Stage3Artifact",
                                    "model_family": family, "config.arm": arm,
                                    "split_fingerprint": stage3.get("split_fingerprint")}
    out[str(upstream["stage1"])] = {"type": "builtins.dict"}
    out[str(upstream["stage2"])] = {"type": "builtins.dict"}
    return out


# ======================================================================
# saved models reproduce their saved outputs
# ======================================================================
def check_stage3_serving(path, GLOBAL_SPLIT: dict, upstream: str, expected_features=None) -> None:
    """
    A Stage 3 artifact: F21 identity, sidecar = payload, and its serving model
    turns GLOBAL_SPLIT's raw test rows into exactly the saved test probabilities.
    """
    from shared.stage4 import UPSTREAM
    art, problems = joblib.load(path), []
    want_id = UPSTREAM[upstream]["stage3"]["want"]
    got_id = {"model_family": art.model_family, "arm": (art.config or {}).get("arm")}
    if got_id != want_id:
        problems.append(f"identity {got_id}, expected {want_id} (F21)")
    if expected_features is not None and list(art.features) != list(expected_features):
        problems.append("features differ from the expected feature contract")
    sidecar = Path(str(path)[:-len(".joblib")] + ".json")
    if not sidecar.is_file():
        problems.append(f"no JSON sidecar {sidecar.name}")
    else:
        side = json.loads(sidecar.read_text(encoding="utf-8"))
        for field in ("split_fingerprint", "block_fingerprint", "features", "model_family"):
            if _plain(side.get(field)) != _plain(getattr(art, field)):
                problems.append(f"sidecar {field} differs from the payload")
    serving = art.serving_model
    if serving is None:
        problems.append("refit['serving_model'] missing")
    else:
        if list(serving.features) != list(art.features):
            problems.append("serving model features differ from the artifact's")
        X_test = GLOBAL_SPLIT["X_test"]
        if list(X_test.index) != list(art.index_test):
            problems.append("index_test differs from GLOBAL_SPLIT")
        else:
            served = serving.predict_proba_raw(X_test)
            saved = np.asarray(art.refit["test_proba"], dtype=float)
            if served.shape != saved.shape or not np.allclose(served, saved, rtol=0, atol=1e-9):
                problems.append("serving model does not reproduce refit['test_proba']")
    if problems:
        raise AuditError(f"Stage 3 ({upstream}) {path}:\n  " + "\n  ".join(problems))


def check_meta_xgb_serving(key: str, art, inputs) -> None:
    """
    The Meta-XGB feature block rebuilt from the reloaded upstream passes its
    cell-by-cell checks, in the saved feature order, and the saved learner
    scores it exactly as saved.
    """
    from blackbox_pipeline.models.meta_xgb.inputs import build_feature_frames
    _, X_test, checks = build_feature_frames(inputs)
    problems = [f"feature block: {name}" for _, _, name, fn, _ in checks if not fn()]
    if list(X_test.columns) != list(art.feature_names):
        problems.append(f"feature order {list(X_test.columns)} differs from {art.feature_names}")
    else:
        served = art.learner.serving_model.predict_proba_raw(X_test)
        if not np.allclose(served, np.asarray(art.test["proba_raw"], dtype=float), rtol=0, atol=1e-9):
            problems.append("learner.serving_model does not reproduce test['proba_raw']")
    if problems:
        raise AuditError(f"{key}:\n  " + "\n  ".join(problems))


# ======================================================================
# the 2x2
# ======================================================================
def check_crossover(arts: dict, paths: dict, GLOBAL_SPLIT: dict,
                    extra_fingerprints: Optional[dict] = None, n_boot: int = 200) -> dict:
    """
    Columns: one upstream, both arbiters, compared with the upstream held fixed
    (Cell 38: ``compare_with_meta_ebm(require_same_upstream=True)``). Rows: one
    arbiter on both upstreams consumed different outputs at every stage — no
    artifact was mixed in from the other arm. All four on one split and test
    index. The native Meta-XGB's saved cascade comparison (Cell 36) used the
    audited native Meta-EBM. Returns the two column comparisons.
    """
    from blackbox_pipeline.models.meta_xgb.comparison import compare_with_meta_ebm
    problems, columns = [], {}
    fps = {k: _plain(_fields(a)["split_fingerprints"]) for k, a in arts.items()}
    tests = {k: list(_fields(a)["index_test"]) for k, a in arts.items()}
    if len({json.dumps(v, sort_keys=True) for v in fps.values()}) != 1:
        problems.append(f"split fingerprints differ across arbiters: {fps}")
    if any(t != tests["meta_ebm"] for t in tests.values()):
        problems.append("index_test differs across arbiters")
    for arm, (ebm_key, xgb_key) in COLUMNS.items():
        try:
            res = compare_with_meta_ebm(GLOBAL_SPLIT, str(paths[ebm_key]), arts[xgb_key],
                                        n_boot=n_boot, verbose=False,
                                        extra_fingerprints=extra_fingerprints,
                                        require_same_upstream=True)
        except Exception as exc:                # noqa: BLE001 — every failure is a finding
            problems.append(f"{arm} column ({paths[ebm_key]} vs {paths[xgb_key]}): "
                            f"{type(exc).__name__}: {exc}")
            continue
        up = res["protocol"]["upstream"]
        expected = {"meta_ebm": f"{arm} upstream -> meta_ebm", "meta_xgb": f"{arm} upstream -> meta_xgb",
                    "held_fixed": True}
        got = {k: up.get(k) for k in expected}
        if got != expected:
            problems.append(f"{arm} column: comparison recorded {got}, expected {expected}")
        columns[arm] = res
    for arbiter, (a, b) in ROWS.items():
        ra, rb = upstream_record(a, arts[a]), upstream_record(b, arts[b])
        for stage in STAGES:
            fa = (ra["stages"].get(stage) or {}).get("content_fingerprint")
            if fa == (rb["stages"].get(stage) or {}).get("content_fingerprint"):
                problems.append(f"{arbiter}: {stage} outputs are identical on both upstreams "
                                f"({a} vs {b}) — an artifact was mixed in from the other arm")
    comparison = getattr(arts["meta_xgb"], "comparison", None)
    if not comparison:
        problems.append("meta_xgb records no cascade comparison (Cell 36 passes meta_ebm_path)")
    else:
        got = (comparison.get("meta_ebm_config") or {}).get("timestamp")
        if got != _fields(arts["meta_ebm"]).get("timestamp"):
            problems.append(f"meta_xgb's cascade comparison used the Meta-EBM of {got!r}, not the "
                            f"audited {paths['meta_ebm']} ({_fields(arts['meta_ebm']).get('timestamp')!r})")
        up = (comparison.get("protocol") or {}).get("upstream") or {}
        if (up.get("meta_ebm"), up.get("meta_xgb"), up.get("held_fixed")) != \
                ("glass upstream -> meta_ebm", "blackbox upstream -> meta_xgb", False):
            problems.append(f"meta_xgb's cascade comparison records {up}")
    if problems:
        raise AuditError("2x2 crossover:\n  " + "\n  ".join(problems))
    return columns


# ======================================================================
# legacy files a notebook could still pick
# ======================================================================
def unloadable_xgb_artifacts(root_dir=BANK_ROOT) -> list:
    """
    ``[(path, why)]`` for files in the XGBoost Stage 3 folder that black-box
    Cell 31 (``XGB_PARAM_SOURCE='artifact'``) could load as the newest but that
    do not unpickle as a ``Stage3Artifact`` in a clean interpreter — above all
    pickles of the standalone package removed in P4a.
    """
    folder = Path(root_dir) / XGB_STAGE3_FOLDER
    files = sorted(folder.glob("xgb_stage3_*.joblib")) if folder.is_dir() else []
    out = []
    for path, got in (load_in_fresh_interpreter(files, root_dir).items() if files else ()):
        if "error" in got:
            out.append((path, got["error"] + _hint(got["error"])))
        elif got["type"] != "shared.stage3.artifact.Stage3Artifact":
            out.append((path, f"holds {got['type']}, not a shared Stage3Artifact"))
    return out
