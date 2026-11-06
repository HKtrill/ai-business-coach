"""
shared.stage2.diagnostics.boundary
==================================
The train-only boundary: the ONLY code in the diagnostics package that reads
saved artifacts. Every other module works on the read-only views built here.

What the extractors read
------------------------
Each extractor reads an explicit allowlist of training-side field paths through
``_take``, which refuses any path containing a test-side token
(``FORBIDDEN_FIELD_TOKENS``: test, oracle, holdout, refit, fold_mean, metrics,
glass_router_data). The October 23 artifacts also hold test-side outputs, test
metrics, test-oracle thresholds and (the GLASS-arm RF result) the engineered
test labels; none of those fields is read, iterated over or copied here.

``build_train_reference`` takes exactly three inputs: the training row index,
the training labels and the test row INDEX (row identities only, for the
16-hex router split fingerprint, which hashes no labels). The label-bearing
``shared.stage_io`` split fingerprint is never recomputed (its hash includes
the test labels); a caller may supply it as an ATTESTED value, for example the
one recorded in the baseline snapshot's PROVENANCE.json.

Mandatory vs optional evidence
------------------------------
Mandatory checks are collected in a ``shared.validation.ValidationReport`` and
raised together (``BoundaryCheckError``) so one run lists every failure. Fields
that historical artifacts never stored (per-fold GLASS rule sets, RF Pass 2
outer scores, feature hashes) are recorded as ``not_recorded`` provenance, never
filled in.

Invariants verified here, not assumed
-------------------------------------
* stored route tokens are exactly "pass1" / "pass2" / "uncertain", and the
  stored ``train_pred`` is 0 / 1 / -1 accordingly;
* router confidence is finite, in [0, 1], and exactly 0.0 on abstained rows;
* the stored router score equals the routers' transform (b(1 - c) / b + (1 - b)c
  / b, b = the fold model's training base rate) within ``TRANSFORM_ATOL``
  (absolute). A larger difference fails; the tolerance is never relaxed here;
* fold ids are a partition, reproduce ``shared.stage_io.fold_assignment`` at the
  declared seed, and match the recorded fold fingerprint;
* both label fingerprints (cross-fit and full-train fits) match the training
  rows and labels; the training base rate matches;
* the Pass 2 population is declared exactly and consistently (PR 36 F17).
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from shared.stage4.adapters import PASS2_POPULATIONS, load as _load_artifact, vec as _vec
from shared.stage4.contract import (
    FINGERPRINT_SCHEMES,
    check_proba,
    fold_partition,
    index_alignment,
    label_identity,
    reproduce_folds,
)
from shared.stage_io import THRESHOLD_SOURCES, label_fingerprint
from shared.validation import ValidationReport

from .schema import (
    ATTESTED,
    HISTORICAL_TEST_EXPOSURE,
    NOT_RECORDED,
    RECORDED_ONLY,
    SELECTION_FOLD_NESTED,
    SELECTION_GLOBAL_TRAIN_LABELS,
    SOURCE_DECISIONS,
    VALIDATED,
    ProvenanceItem,
    RFPass1ScoreCandidate,
    RouterDiagnosticError,
    RouterOOFView,
    ScoreStream,
    Stage4Thresholds,
    StoredDecision,
    ThresholdRecord,
    TrainReference,
    UpstreamOOFView,
    _frozen,
    _proxy,
)

__all__ = [
    "FORBIDDEN_FIELD_TOKENS", "TRANSFORM_ATOL", "ROUTER_N_FOLDS", "LR_N_FOLDS",
    "RF_PASS2_REMAINDER_PROTOCOL", "BoundaryCheckError",
    "router_split_fingerprint", "build_train_reference", "train_reference_from_global_split",
    "extract_router_view", "extract_lr_view", "extract_ebm_view",
    "extract_stage4_recall_thresholds", "extract_rf_pass1_candidate", "file_sha256",
]

#: A field path containing any of these (case-insensitive) is never read.
FORBIDDEN_FIELD_TOKENS = ("test", "oracle", "holdout", "refit", "fold_mean", "metrics",
                          "glass_router_data")
#: Absolute tolerance for the router transform check (Step 1C decision 5).
TRANSFORM_ATOL = 1e-12
#: The shared Stage 2-3 outer fold plan and the Stage 1 plan.
ROUTER_N_FOLDS = 5
LR_N_FOLDS = 10
#: ``crossfit_rf_router``'s exact Pass 2 population declaration.
RF_PASS2_REMAINDER_PROTOCOL = "OOF Pass 1 remainder (RF router design)"
_STAGE_IO_FP = re.compile(r"[0-9a-f]{12}")
_ROUTER_FP = re.compile(r"[0-9a-f]{16}")

_SECTION = "Stage 2 diagnostics boundary"
_MISSING = object()


class BoundaryCheckError(RouterDiagnosticError):
    """One or more mandatory boundary checks failed; ``report`` lists every check."""

    def __init__(self, message: str, report: pd.DataFrame):
        super().__init__(message)
        self.report = report


# ======================================================================
# Reading
# ======================================================================
def _take(obj, path: Sequence[str], *, required: bool = True):
    """
    The value at ``path`` (dict keys or attributes). Refuses a path containing a
    forbidden token BEFORE touching anything; returns ``_MISSING`` for an absent
    optional field.
    """
    dotted = ".".join(map(str, path))
    for key in path:
        hits = [t for t in FORBIDDEN_FIELD_TOKENS if t in str(key).lower()]
        if hits:
            raise RouterDiagnosticError(
                f"field path {dotted!r} is not train-side (contains {hits}); the "
                "diagnostics boundary never reads it")
    cur = obj
    for key in path:
        if isinstance(cur, Mapping):
            present = key in cur
            nxt = cur[key] if present else None
        else:
            present = hasattr(cur, key)
            nxt = getattr(cur, key) if present else None
        if not present or nxt is None:
            if required:
                raise RouterDiagnosticError(f"required field {dotted!r} is missing")
            return _MISSING
        cur = nxt
    return cur


class _Fields:
    """Collects required fields, so every missing one is reported at once."""

    def __init__(self, obj, what: str):
        self.obj, self.what = obj, what
        self.missing: list = []
        self.paths: dict = {}

    def get(self, *path: str, required: bool = True):
        full = tuple(path)
        value = _take(self.obj, full, required=False)
        self.paths[".".join(full)] = value is not _MISSING
        if value is _MISSING and required:
            self.missing.append(".".join(full))
        return None if value is _MISSING else value

    def raise_if_missing(self) -> None:
        if self.missing:
            raise RouterDiagnosticError(
                f"{self.what}: missing required train-side field(s) {self.missing}")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open(obj_or_path, expected_sha256: Optional[str], what: str):
    """(artifact, path or None, sha256 ProvenanceItem)."""
    path = None
    if isinstance(obj_or_path, (str, bytes)) or hasattr(obj_or_path, "__fspath__"):
        path = str(obj_or_path)
        digest = file_sha256(path)
        if expected_sha256 is not None:
            if digest != str(expected_sha256).strip().lower():
                raise RouterDiagnosticError(
                    f"{what}: {path} SHA-256 {digest} != expected {expected_sha256}; "
                    "this is not the preserved October 23 artifact")
            item = ProvenanceItem("file_sha256", ATTESTED,
                                  f"{digest} equals the supplied baseline value")
        else:
            item = ProvenanceItem("file_sha256", NOT_RECORDED,
                                  f"{digest}; no expected value supplied")
    else:
        if expected_sha256 is not None:
            raise RouterDiagnosticError(
                f"{what}: expected_sha256 needs a file path, not a loaded object")
        item = ProvenanceItem("file_sha256", NOT_RECORDED, "loaded object; no file")
    art, _ = _load_artifact(obj_or_path)
    return art, path, item


def _report_failures(rep: ValidationReport, what: str) -> pd.DataFrame:
    df = rep.frame()
    bad = df[~df["passed"]]
    if len(bad):
        lines = [f"  - {r.check}: {r.error or r.detail}" for r in bad.itertuples()]
        raise BoundaryCheckError(
            f"{what}: {len(bad)} mandatory check(s) failed:\n" + "\n".join(lines), df)
    return df


def _safe(fn, default="") -> str:
    try:
        return str(fn())
    except Exception:  # noqa: BLE001 - a detail string must never mask the check itself
        return default


def _hex(value, pattern) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


# ======================================================================
# Train reference
# ======================================================================
def _resolve_router_fp_fn() -> Callable:
    for scheme, mod, attr, style in FINGERPRINT_SCHEMES:
        if scheme != "router" or style != "frames":
            continue
        try:
            return getattr(importlib.import_module(mod), attr)
        except (ImportError, AttributeError):
            continue
    raise RouterDiagnosticError(
        "no router split_fingerprint implementation is importable "
        "(tried the 'router' entries of shared.stage4.FINGERPRINT_SCHEMES)")


def router_split_fingerprint(index_train, index_test) -> str:
    """The routers' 16-hex split fingerprint. Hashes row indices only, no labels."""
    fn = _resolve_router_fp_fn()
    return fn(pd.DataFrame(index=pd.Index(index_train)), pd.DataFrame(index=pd.Index(index_test)))


def build_train_reference(*, index_train, y_train, index_test,
                          attested_stage_io_split_fingerprint: Optional[str] = None,
                          attestation_source: Optional[str] = None,
                          router_fingerprint: Optional[Callable] = None) -> TrainReference:
    """
    The training rows the views are checked against.

    ``index_test`` is used for the router split fingerprint only (row ids).
    ``attested_stage_io_split_fingerprint``: the 12-hex ``shared.stage_io``
    fingerprint as recorded elsewhere (never recomputed here; its hash includes
    the test labels); needs ``attestation_source`` naming that record.
    ``router_fingerprint(index_train, index_test)`` overrides the default.
    """
    idx_tr, idx_te = pd.Index(index_train), pd.Index(index_test)
    if not idx_tr.is_unique or not idx_te.is_unique:
        raise RouterDiagnosticError("training / test row indices must be unique")
    if len(idx_tr.intersection(idx_te)):
        raise RouterDiagnosticError("training and test row indices overlap")
    if isinstance(y_train, pd.Series) and not y_train.index.equals(idx_tr):
        raise RouterDiagnosticError("y_train.index != index_train")
    y = np.asarray(y_train)
    if len(y) != len(idx_tr):
        raise RouterDiagnosticError(f"{len(y)} labels for {len(idx_tr)} training rows")
    if not np.isin(y, (0, 1)).all():
        raise RouterDiagnosticError("y_train is not binary 0/1")
    y = y.astype(int)
    if attested_stage_io_split_fingerprint is not None:
        if not _hex(attested_stage_io_split_fingerprint, _STAGE_IO_FP):
            raise RouterDiagnosticError(
                f"attested stage_io fingerprint {attested_stage_io_split_fingerprint!r} is not "
                "12 lowercase hex characters")
        if not attestation_source:
            raise RouterDiagnosticError(
                "an attested stage_io fingerprint needs attestation_source (where it is recorded)")
    fp = (router_fingerprint or router_split_fingerprint)(idx_tr, idx_te)
    if not _hex(fp, _ROUTER_FP):
        raise RouterDiagnosticError(f"router split fingerprint {fp!r} is not 16 hex characters")
    return TrainReference(
        index_train=idx_tr,
        y_train=_frozen(y),
        router_split_fingerprint=fp,
        train_label_fingerprint=label_fingerprint(idx_tr, y),
        stage_io_split_fingerprint=attested_stage_io_split_fingerprint,
        stage_io_attestation=attestation_source if attested_stage_io_split_fingerprint else None,
    )


def train_reference_from_global_split(GLOBAL_SPLIT, **kwargs) -> TrainReference:
    """Reads ``X_train.index``, ``X_test.index`` and ``y_train`` only (never ``y_test``)."""
    X_tr, X_te = GLOBAL_SPLIT["X_train"], GLOBAL_SPLIT["X_test"]
    return build_train_reference(index_train=X_tr.index, y_train=GLOBAL_SPLIT["y_train"],
                                 index_test=X_te.index, **kwargs)


def _stage_io_item(ref: TrainReference) -> ProvenanceItem:
    if ref.stage_io_split_fingerprint:
        return ProvenanceItem("stage_io_split_fingerprint", ATTESTED,
                              f"{ref.stage_io_split_fingerprint} (from {ref.stage_io_attestation}; "
                              "not recomputed: its hash includes the test labels)")
    return ProvenanceItem("stage_io_split_fingerprint", NOT_RECORDED,
                          "no attestation supplied; never recomputed (hash includes test labels)")


# ======================================================================
# Shared checks
# ======================================================================
def _fold_checks(chk, fold_raw, ref: TrainReference, *, n_folds_expected: int,
                 declared_n_folds, seed, fingerprint=None, fold_fp_fn=None, field="fold ids"):
    fold = np.asarray(fold_raw)
    chk(f"{field}: one per training row", lambda: len(fold) == ref.n_train,
        f"{len(fold)} vs {ref.n_train}")
    chk(f"{field}: partition 0..k-1",
        lambda: fold_partition(fold, field) == n_folds_expected,
        f"expected {n_folds_expected} folds")
    if declared_n_folds is not None:
        chk(f"{field}: declared n_folds = fold ids",
            lambda: int(declared_n_folds) == len(np.unique(fold)) == n_folds_expected,
            f"declared {declared_n_folds}")
    chk(f"{field}: reproduce shared.stage_io.fold_assignment at the declared seed",
        lambda: seed is not None and reproduce_folds(fold, ref.y_train, [seed]) == int(seed),
        f"seed {seed}")
    if fingerprint is not None:
        chk(f"{field}: fold fingerprint matches", lambda: fold_fp_fn(fold) == fingerprint,
            str(fingerprint))
    return fold


def _resolve_fold_fp_fn(fold_fingerprint_fn):
    if fold_fingerprint_fn is not None:
        return fold_fingerprint_fn
    mod = importlib.import_module("glass_pipeline.glass_router.pipeline")
    return getattr(mod, "fold_fingerprint")


def _same_list(a, b) -> bool:
    return a is not None and [str(x) for x in a] == [str(x) for x in b]


def _is_partition(fold: np.ndarray, n: int, k: int) -> bool:
    try:
        return fold.shape == (n,) and fold_partition(fold, "folds") == k
    except Exception:  # noqa: BLE001 - reported by the fold checks
        return False


# ======================================================================
# Stage 2 routers
# ======================================================================
def _router_layout(art, what: str) -> str:
    if not isinstance(art, Mapping):
        raise RouterDiagnosticError(f"{what}: router artifact is {type(art).__name__}, expected a dict")
    extra = art.get("extra")
    if isinstance(extra, Mapping) and "train_outputs_oof" in extra:
        return "extra"
    if "train_outputs_oof" in art:
        return "top-level"
    raise RouterDiagnosticError(f"{what}: no train_outputs_oof at the top level or under 'extra'")


_GLASS_NOT_RECORDED = (
    ("fold_rule_sets", "per-fold GLASS rule sets and candidate rules are not persisted"),
    ("fold_summaries", "GLASS cross-fit fold summaries are not persisted"),
)
_RF_NOT_RECORDED = (
    ("pass2_outer_scores", "RF Pass 2 outer out-of-fold scores are not persisted"),
    ("pass1_outer_scores", "RF Pass 1 outer scores are not stored in the router artifact; "
                           "see check_rf_pass1_score_equivalence"),
)
_GLASS_CONFIG_KEYS = ("mode", "pass2_population", "min_support_pass1", "min_support_pass2",
                      "min_precision_not_subscribe", "max_subscriber_leakage_rate",
                      "max_union_leakage_rate_pass1", "min_precision_subscribe",
                      "max_recall_subscribe")
_PROTOCOL_KEYS = ("n_folds", "random_state", "fold_fingerprint", "train_label_fingerprint",
                  "split_fingerprint", "pass2_population", "glass_pass2_population",
                  "glass_abstain_label", "fold_source")


def _router_fields(art, layout: str, what: str) -> dict:
    """Every field a router view reads, by explicit path; all missing ones reported at once."""
    pre = () if layout == "top-level" else ("extra",)
    F = _Fields(art, what)
    f = {
        "flag": F.get(*pre, "train_outputs_oof"),
        "decisions": F.get(*pre, "train_decisions"),
        "pred": F.get(*pre, "train_pred"),
        "confidence": F.get(*pre, "train_confidence"),
        "proba": F.get(*pre, "train_proba"),
        "fold": F.get(*pre, "train_fold_id"),
        "index": F.get(*pre, "train_idx" if layout == "top-level" else "index_train"),
        "n_folds": F.get(*pre, "oof_protocol", "n_folds"),
        "seed": F.get(*pre, "oof_protocol", "random_state"),
        "fold_fp": F.get(*pre, "oof_protocol", "fold_fingerprint"),
        "cross_fit_label_fp": F.get(*pre, "oof_protocol", "train_label_fingerprint"),
        "full_train_label_fp": F.get("train_label_fingerprint"),
        "split_fp": F.get("split_fingerprint"),
        "base_rate": F.get("training_base_rate"),
    }
    if layout == "top-level":
        f.update(features=F.get("features"),
                 pop_cross=F.get("oof_protocol", "glass_pass2_population"),
                 pop_full=F.get("fit_report", "pass2_population"),
                 pop_config=F.get("config_full", "pass2_population", required=False),
                 mode=None, protocol_split_fp=None)
    else:
        f.update(features=F.get("feature_names"),
                 pop_cross=F.get("extra", "oof_protocol", "pass2_population"),
                 mode=F.get("mode"),
                 protocol_split_fp=F.get("extra", "oof_protocol", "split_fingerprint",
                                         required=False),
                 fold_summary=F.get("extra", "oof_fold_summary"),
                 rf_params=F.get("config", "rf_params"),
                 rf_seed=F.get("config", "random_state"))
    F.raise_if_missing()
    f["fields_read"] = tuple(sorted(F.paths))
    f["pre"] = pre
    return f


def _check_routes(chk, f: dict, ref: TrainReference, what: str, fold_fp_fn):
    """Index, tokens, pred, confidence, score, folds and the route transform.
    Returns (decisions, pred, confidence, score, fold, max transform diff)."""
    n = ref.n_train
    chk("train_outputs_oof is True", lambda: f["flag"] is True, repr(f["flag"]))
    chk("row index = GLOBAL_SPLIT X_train index (same order)",
        lambda: index_alignment(f["index"], ref.index_train, f"{what} index") == "label")
    raw = f["decisions"]
    dec = np.asarray(raw.to_numpy() if isinstance(raw, pd.Series) else raw, dtype=object)
    bad = sorted({repr(v) for v in dec.ravel() if not (isinstance(v, str) and v in SOURCE_DECISIONS)})
    tokens_ok = dec.ndim == 1 and len(dec) == n and not bad
    chk("decisions are exactly 'pass1' / 'pass2' / 'uncertain'", lambda: tokens_ok,
        f"unexpected {bad[:5]}" if bad else f"{len(dec):,} rows")
    pred = np.asarray(f["pred"])
    chk("train_pred = 0 / 1 / -1 for pass1 / pass2 / uncertain",
        lambda: tokens_ok and pred.dtype.kind in "biuf" and pred.shape == (n,) and np.array_equal(
            pred.astype(float), np.array([SOURCE_DECISIONS[v][1] for v in dec], dtype=float)))
    conf = np.asarray(f["confidence"], dtype=float)
    conf_ok = conf.shape == (n,)
    abst = dec == "uncertain"
    chk("confidence finite in [0, 1]", lambda: conf_ok and check_proba(conf, "confidence"))
    chk("confidence exactly 0.0 on abstained rows",
        lambda: tokens_ok and conf_ok and bool(np.all(conf[abst] == 0.0)),
        _safe(lambda: f"max on abstain {conf[abst].max() if abst.any() else 0.0!r}"))
    score = _vec(f["proba"], f"{what} train_proba")
    score_ok = score.shape == (n,)
    chk("router score finite in [0, 1]", lambda: score_ok and check_proba(score, "train_proba"))
    fold = _fold_checks(chk, f["fold"], ref, n_folds_expected=ROUTER_N_FOLDS,
                        declared_n_folds=f["n_folds"], seed=f["seed"], fingerprint=f["fold_fp"],
                        fold_fp_fn=fold_fp_fn)
    diff = math.inf
    if tokens_ok and conf_ok and score_ok and _is_partition(fold, n, ROUTER_N_FOLDS):
        y = ref.y_train.astype(float)
        b = np.empty(n)
        for k in range(ROUTER_N_FOLDS):
            b[fold == k] = y[fold != k].mean()
        expected = np.where(dec == "pass1", b * (1.0 - conf),
                            np.where(dec == "pass2", b + (1.0 - b) * conf, b))
        diff = float(np.max(np.abs(score - expected)))
    chk(f"router score = route transform with fold base rates (|diff| <= {TRANSFORM_ATOL:g})",
        lambda: diff <= TRANSFORM_ATOL, f"max |diff| {diff:.3g}")
    return dec, pred, conf, score, fold, diff


def _check_identity(chk, f: dict, ref: TrainReference, what: str) -> None:
    """Split and label fingerprints, base rate and feature set."""
    split_fp = f["split_fp"]
    chk("router split fingerprint = recomputed (row indices only)",
        lambda: split_fp == ref.router_split_fingerprint,
        f"{split_fp} vs {ref.router_split_fingerprint}")
    if f["protocol_split_fp"] is not None:
        chk("oof_protocol split fingerprint = artifact split fingerprint",
            lambda: f["protocol_split_fp"] == split_fp, str(f["protocol_split_fp"]))
    chk("cross-fit label fingerprint = training rows and labels",
        lambda: label_identity(f["cross_fit_label_fp"], ref.train_label_fingerprint,
                               f"{what} cross-fit"))
    chk("full-train label fingerprint = training rows and labels",
        lambda: label_identity(f["full_train_label_fp"], ref.train_label_fingerprint,
                               f"{what} full-train"))
    rate = float(ref.y_train.mean())
    chk("training_base_rate = training positive rate",
        lambda: abs(float(f["base_rate"]) - rate) <= 1e-12,
        _safe(lambda: f"{float(f['base_rate']):.6f} vs {rate:.6f}"))
    from shared.stage2.binning import RF_FEATURES_BINARY
    chk("features = shared.stage2 RF_FEATURES_BINARY",
        lambda: _same_list(f["features"], RF_FEATURES_BINARY),
        f"{len(RF_FEATURES_BINARY)} binary bins")


def _check_glass_population(chk, f: dict) -> dict:
    cross, full, cfg = f["pop_cross"], f["pop_full"], f["pop_config"]
    chk("Pass 2 population declared for the cross-fit and the full-train fit, equal and known",
        lambda: cross == full and cross in PASS2_POPULATIONS,
        f"cross-fit {cross!r}, full-train {full!r}")
    if cfg is not None:
        chk("config_full Pass 2 population = fit_report", lambda: cfg == full, repr(cfg))
    return {"cross_fit": cross, "full_train": full, "normalised": cross}


def _check_rf_population_and_folds(chk, f: dict):
    mode, cross = f["mode"], f["pop_cross"]
    chk("RF Router mode is 'two_pass'", lambda: mode == "two_pass", repr(mode))
    chk("Pass 2 population is the OOF Pass 1 remainder (exact declaration)",
        lambda: cross == RF_PASS2_REMAINDER_PROTOCOL, repr(cross))

    def fold_thresholds():
        out = {}
        for row in f["fold_summary"]:
            k = int(row["fold"])
            t1, t2 = float(row["t1"]), float(row["t2"])
            if k in out or not (math.isfinite(t1) and math.isfinite(t2)):
                raise RouterDiagnosticError(f"oof_fold_summary fold {k}: duplicate or non-finite")
            out[k] = MappingProxyType({"t1": t1, "t2": t2})
        if sorted(out) != list(range(ROUTER_N_FOLDS)):
            raise RouterDiagnosticError(f"oof_fold_summary folds {sorted(out)}")
        return out

    chk("oof_fold_summary: one finite t1 / t2 per outer fold", lambda: fold_thresholds() is not None)
    population = {"cross_fit": cross, "full_train": f"mode {mode!r}", "normalised": "oof_remainder"}
    return population, _safe_call(fold_thresholds)


def _optional(art, path):
    value = _take(art, path, required=False)
    return None if value is _MISSING else value


def _router_metadata(art, f: dict, layout: str) -> dict:
    """Allowlisted training-side configuration, for the record (never used to decide)."""
    pre = f["pre"]
    protocol = {k: _optional(art, (*pre, "oof_protocol", k)) for k in _PROTOCOL_KEYS}
    meta = {"training_base_rate": float(f["base_rate"]), "split_fingerprint": f["split_fp"],
            "oof_protocol": {k: v for k, v in protocol.items() if v is not None}}
    if layout == "top-level":
        cfg = {k: _optional(art, ("config_full", k)) for k in _GLASS_CONFIG_KEYS}
        meta["config_full"] = {k: v for k, v in cfg.items() if v is not None}
        rules = _optional(art, ("rules",))
        if rules is not None:
            meta["n_rules"] = {p: sum(1 for r in rules if r.get("pass") == p)
                               for p in ("Pass 1", "Pass 2")}
        return meta
    meta.update({
        "mode": f["mode"],
        "rf_params": dict(f["rf_params"]),
        "random_state": f["rf_seed"],
        "n_oof_folds": _optional(art, ("config", "n_oof_folds")),
        "constraints": _optional(art, ("constraints",)),
        "constraint_source": _optional(art, ("extra", "constraint_source")),
        "rf_params_origin": _optional(art, ("extra", "rf_params_origin")),
        "full_train_thresholds": {k: _optional(art, ("thresholds", k)) for k in ("t1", "t2")},
        "bands": {k: _optional(art, ("bands", k))
                  for k in ("kind", "pass1_confidence", "pass2_confidence")},
    })
    return meta


def extract_router_view(artifact, ref: TrainReference, *, expected_arm: str, path=None,
                        expected_sha256: Optional[str] = None,
                        fold_fingerprint_fn: Optional[Callable] = None) -> RouterOOFView:
    """
    The outer cross-fit (train-side) outputs of the GLASS Router bundle
    (``expected_arm="glass"``) or the two-pass RF Router (``"blackbox"``).
    ``artifact``: a loaded artifact or a path; ``path`` names a loaded one.
    """
    if expected_arm not in ("glass", "blackbox"):
        raise RouterDiagnosticError(f"expected_arm must be 'glass' or 'blackbox', got {expected_arm!r}")
    what = f"{expected_arm} router"
    art, opened_path, sha_item = _open(artifact, expected_sha256, what)
    path = opened_path or path
    layout = _router_layout(art, what)
    family = "glass_router" if layout == "top-level" else "rf_router"
    if {"top-level": "glass", "extra": "blackbox"}[layout] != expected_arm:
        raise RouterDiagnosticError(f"{what}: artifact layout {layout!r} is the {family} format, "
                                    f"not the {expected_arm} router")
    f = _router_fields(art, layout, what)

    rep = ValidationReport()
    chk = lambda name, fn, detail="": rep.check(_SECTION, what, name, fn, detail)  # noqa: E731
    dec, pred, conf, score, fold, diff = _check_routes(chk, f, ref, what,
                                                       _resolve_fold_fp_fn(fold_fingerprint_fn))
    _check_identity(chk, f, ref, what)
    if layout == "top-level":
        population, fold_thresholds = _check_glass_population(chk, f), None
    else:
        population, fold_thresholds = _check_rf_population_and_folds(chk, f)
    checks = _report_failures(rep, what)

    provenance = [
        ProvenanceItem("train_outputs_oof", VALIDATED, "outer cross-fit outputs"),
        ProvenanceItem("row_index", VALIDATED, "equals X_train index, same order"),
        ProvenanceItem("decision_tokens", VALIDATED, "exact tokens; train_pred consistent"),
        ProvenanceItem("confidence", VALIDATED, "finite, in [0, 1], 0.0 on abstain"),
        ProvenanceItem("router_transform", VALIDATED,
                       f"max |diff| {diff:.3g} <= {TRANSFORM_ATOL:g} (fold base rates)"),
        ProvenanceItem("fold_plan", VALIDATED,
                       f"{ROUTER_N_FOLDS} folds, seed {f['seed']}, fingerprint {f['fold_fp']}"),
        ProvenanceItem("router_split_fingerprint", VALIDATED, ref.router_split_fingerprint),
        _stage_io_item(ref),
        ProvenanceItem("train_label_fingerprint.cross_fit", VALIDATED, str(f["cross_fit_label_fp"])),
        ProvenanceItem("train_label_fingerprint.full_train", VALIDATED, str(f["full_train_label_fp"])),
        ProvenanceItem("training_base_rate", VALIDATED, f"{float(f['base_rate']):.6f}"),
        ProvenanceItem("pass2_population", VALIDATED,
                       f"declared: cross-fit {population['cross_fit']!r}, "
                       f"full-train {population['full_train']!r}"),
        ProvenanceItem("features", VALIDATED, "RF_FEATURES_BINARY"),
        ProvenanceItem("feature_hash", NOT_RECORDED, "no feature-matrix hash is stored"),
        sha_item,
    ]
    if fold_thresholds is not None:
        provenance.append(ProvenanceItem("fold_thresholds", VALIDATED,
                                         "oof_fold_summary: one finite t1 / t2 per fold"))
    provenance += [ProvenanceItem(name, NOT_RECORDED, detail) for name, detail in
                   (_GLASS_NOT_RECORDED if layout == "top-level" else _RF_NOT_RECORDED)]
    confidence_kind = (
        "in-sample precision of the first matching rule of the outer fold's GLASS fit "
        "(0.0 on abstain)" if layout == "top-level" else
        "band precision measured on the outer fold router's inner OOF (0.0 on abstain)")
    states = np.array([SOURCE_DECISIONS[v][0] for v in dec], dtype=object)
    return RouterOOFView(
        arm=expected_arm, model_family=family, router_mode=f["mode"],
        index=ref.index_train, y=ref.y_train, fold_id=_frozen(fold, int),
        state=_frozen(states, object), source_decision=_frozen(dec, object),
        source_pred=_frozen(pred.astype(int)), confidence=_frozen(conf, float),
        confidence_kind=confidence_kind,
        router_score=ScoreStream(f"{expected_arm}.router_score", _frozen(score, float),
                                 "router_transform", ".".join(f["pre"] + ("train_proba",)) + "[:, 1]",
                                 "router-native transform of route and confidence; not calibrated"),
        pass2_population=_proxy(population),
        fold_thresholds=None if fold_thresholds is None else MappingProxyType(dict(fold_thresholds)),
        metadata=_proxy(_router_metadata(art, f, layout)),
        provenance=tuple(provenance),
        source=_proxy({"path": path, "layout": layout,
                       "checks": tuple(checks.to_dict("records")),
                       "fields_read": f["fields_read"],
                       "historical_test_exposure": HISTORICAL_TEST_EXPOSURE}),
    )


def _safe_call(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - already reported by the check
        return None


# ======================================================================
# Stage 1 LR
# ======================================================================
def extract_lr_view(artifact, ref: TrainReference, *, expected_arm: str = "glass", path=None,
                    expected_sha256: Optional[str] = None) -> UpstreamOOFView:
    """The GLASS Stage 1 calibrated LR's out-of-fold outputs (StageOutput contract)."""
    what = f"{expected_arm} Stage 1 LR"
    art, opened_path, sha_item = _open(artifact, expected_sha256, what)
    path = opened_path or path
    if isinstance(art, Mapping) and "__stage_output__" in art:
        so, src = art, ""
    elif isinstance(art, Mapping) and "stage_output" in art:
        so, src = art["stage_output"], "stage_output."
    elif type(art).__name__ == "StageOutput":
        so, src = art, ""
    else:
        raise RouterDiagnosticError(f"{what}: no StageOutput contract in the artifact")
    F = _Fields(so, what)
    stage, arm, model = F.get("stage"), F.get("arm"), F.get("model")
    features = F.get("feature_names")
    proba = F.get("train_proba_oof")
    y = F.get("y_train")
    fold_raw = F.get("train_fold_id")
    thr, thr_src = F.get("threshold"), F.get("threshold_source")
    cal = F.get("calibration_method")
    prov = F.get("oof_provenance")
    seed = F.get("config", "random_state")
    split_fp = F.get("split_fingerprint")
    F.raise_if_missing()

    rep = ValidationReport()
    chk = lambda name, fn, detail="": rep.check(_SECTION, what, name, fn, detail)  # noqa: E731
    from shared.stage1.feature_engineering import LR_FEATURES
    chk("stage is 'stage1'", lambda: stage == "stage1", repr(stage))
    chk(f"arm is {expected_arm!r}", lambda: arm == expected_arm, repr(arm))
    chk("feature_names = shared.stage1 LR_FEATURES", lambda: _same_list(features, LR_FEATURES),
        _safe(lambda: list(features)))
    for name, s in (("train_proba_oof", proba), ("y_train", y), ("train_fold_id", fold_raw)):
        chk(f"{name} index = X_train index (same order)",
            lambda s=s, name=name: isinstance(s, pd.Series)
            and index_alignment(s.index, ref.index_train, f"{what} {name}") == "label")
    yv = np.asarray(y)
    chk("carried training labels = GLOBAL_SPLIT y_train",
        lambda: yv.shape == (ref.n_train,) and np.array_equal(yv.astype(int), ref.y_train))
    p = np.asarray(proba, dtype=float)
    chk("train_proba_oof finite in [0, 1]", lambda: p.shape == (ref.n_train,) and check_proba(p, "lr"))
    _fold_checks(chk, np.asarray(fold_raw), ref, n_folds_expected=LR_N_FOLDS,
                 declared_n_folds=None, seed=seed)
    chk("threshold_source is a shared.stage_io THRESHOLD_SOURCES key",
        lambda: thr_src in THRESHOLD_SOURCES, repr(thr_src))
    chk("threshold finite in (0, 1)", lambda: 0.0 < float(thr) < 1.0, repr(thr))
    chk("OOF calibration nested inside each fold (oof_provenance)",
        lambda: "nested cross-fit" in str(prov), str(prov)[:80])
    if ref.stage_io_split_fingerprint:
        chk("stage_io split fingerprint = attested value",
            lambda: split_fp == ref.stage_io_split_fingerprint,
            f"{split_fp} vs attested {ref.stage_io_split_fingerprint}")
    checks = _report_failures(rep, what)

    space = "calibrated_probability" if cal not in (None, "none") else "raw_model_score"
    stream = ScoreStream("lr.train_proba_oof", _frozen(p, float), space, f"{src}train_proba_oof",
                         f"out-of-fold over {LR_N_FOLDS} folds; calibration {cal}, nested per fold")
    thresholds = {}
    if thr_src == "in_stage_f2_cv":
        thresholds["lr.in_stage_f2_cv"] = ThresholdRecord(
            "lr.in_stage_f2_cv", float(thr), space, f"{src}threshold", SELECTION_GLOBAL_TRAIN_LABELS,
            "F2 argmax over the OOF probabilities of all training rows (threshold_source "
            "in_stage_f2_cv; not cross-fitted)")
    split_item = (ProvenanceItem("stage_io_split_fingerprint", ATTESTED,
                                 f"{split_fp} = attested ({ref.stage_io_attestation})")
                  if ref.stage_io_split_fingerprint else
                  ProvenanceItem("stage_io_split_fingerprint", RECORDED_ONLY,
                                 f"{split_fp}; no attestation supplied"))
    provenance = (
        ProvenanceItem("row_index", VALIDATED, "train_proba_oof / y_train / train_fold_id"),
        ProvenanceItem("training_labels", VALIDATED, "carried and equal to GLOBAL_SPLIT y_train"),
        ProvenanceItem("fold_plan", VALIDATED, f"{LR_N_FOLDS} folds, seed {seed}"),
        ProvenanceItem("oof_calibration", VALIDATED, str(prov)[:120]),
        ProvenanceItem("threshold_source", VALIDATED,
                       f"{thr_src} (registered: {sorted(thresholds) or 'none'})"),
        split_item,
        ProvenanceItem("feature_hash", NOT_RECORDED, "no feature-matrix hash is stored"),
        sha_item,
    )
    return UpstreamOOFView(
        model="lr", arm=expected_arm, index=ref.index_train, y=ref.y_train,
        fold_id=_frozen(np.asarray(fold_raw), int),
        scores=_proxy({space: stream}), thresholds=_proxy(thresholds), stored_decisions=_proxy({}),
        provenance=provenance,
        source=_proxy({"path": path, "model_name": model, "calibration_method": cal,
                       "checks": tuple(checks.to_dict("records")),
                       "fields_read": tuple(sorted(F.paths)),
                       "historical_test_exposure": HISTORICAL_TEST_EXPOSURE}),
    )


# ======================================================================
# Stage 3 EBM
# ======================================================================
def extract_ebm_view(artifact, ref: TrainReference, *, expected_arm: str = "glass", path=None,
                     expected_sha256: Optional[str] = None,
                     fold_fingerprint_fn: Optional[Callable] = None) -> UpstreamOOFView:
    """The GLASS Stage 3 EBM's out-of-fold outputs (``Stage3Artifact``)."""
    what = f"{expected_arm} Stage 3 EBM"
    art, opened_path, sha_item = _open(artifact, expected_sha256, what)
    path = opened_path or path
    if isinstance(art, Mapping) or not any(c.__name__ == "Stage3Artifact"
                                           for c in type(art).__mro__):
        raise RouterDiagnosticError(f"{what}: expected a Stage3Artifact, got {type(art).__name__}")
    F = _Fields(art, what)
    family, arm = F.get("model_family"), F.get("config", "arm")
    features, scope = F.get("features"), F.get("feature_fit_scope")
    idx, y = F.get("index_train"), F.get("y_train")
    plan_fold, plan_k = F.get("fold_plan", "fold_id"), F.get("fold_plan", "n_splits")
    plan_seed = F.get("fold_plan", "random_state")
    raw, calp = F.get("oof", "proba"), F.get("oof", "proba_calibrated")
    fold_raw, decision = F.get("oof", "fold_id"), F.get("oof", "decision")
    per_row = F.get("oof", "threshold_per_row")
    t_raw = F.get("threshold", "oof", "threshold")
    t_cal = F.get("threshold", "oof_calibrated_space", "threshold")
    by_fold = F.get("threshold", "oof_nested", "by_fold")
    drives = F.get("threshold", "oof_nested", "drives_train_decisions")
    fold_applied = F.get("calibration", "fold_applied")
    split_fp = F.get("split_fingerprint")
    block_fp = F.get("block_fingerprint", required=False)
    F.raise_if_missing()

    rep = ValidationReport()
    chk = lambda name, fn, detail="": rep.check(_SECTION, what, name, fn, detail)  # noqa: E731
    from shared.stage3.feature_engineering import EBM_FEATURES
    chk("model_family is 'ebm'", lambda: family == "ebm", repr(family))
    chk(f"config arm is {expected_arm!r}", lambda: arm == expected_arm, repr(arm))
    chk("features = shared.stage3 EBM_FEATURES", lambda: _same_list(features, EBM_FEATURES),
        f"{len(EBM_FEATURES)} features")
    chk("feature_fit_scope is 'per_fold'", lambda: scope == "per_fold", repr(scope))
    chk("index_train = X_train index (same order)",
        lambda: index_alignment(idx, ref.index_train, f"{what} index_train") == "label")
    yv = np.asarray(y)
    chk("carried training labels = GLOBAL_SPLIT y_train",
        lambda: yv.shape == (ref.n_train,) and np.array_equal(yv.astype(int), ref.y_train))
    fold = np.asarray(fold_raw)
    chk("oof fold_id = fold_plan fold_id", lambda: np.array_equal(fold, np.asarray(plan_fold)))
    fold_fp_fn = _resolve_fold_fp_fn(fold_fingerprint_fn)
    _fold_checks(chk, fold, ref, n_folds_expected=ROUTER_N_FOLDS, declared_n_folds=plan_k,
                 seed=plan_seed)
    pr, pc = np.asarray(raw, dtype=float), np.asarray(calp, dtype=float)
    chk("raw OOF score finite in [0, 1]", lambda: pr.shape == (ref.n_train,) and check_proba(pr, "raw"))
    chk("calibrated OOF probability finite in [0, 1]",
        lambda: pc.shape == (ref.n_train,) and check_proba(pc, "calibrated"))
    chk("per-fold calibration gates recorded (PR 36 F22)",
        lambda: len(fold_applied) == ROUTER_N_FOLDS, _safe(lambda: list(fold_applied)))
    for name, t in (("global raw threshold", t_raw), ("global calibrated threshold", t_cal)):
        chk(f"{name} finite in (0, 1)", lambda t=t: 0.0 < float(t) < 1.0, repr(t))
    chk("drives_train_decisions is a bool", lambda: isinstance(drives, (bool, np.bool_)), repr(drives))
    thr_row = np.asarray(per_row, dtype=float)
    dec = np.asarray(decision)
    nested = bool(drives) if isinstance(drives, (bool, np.bool_)) else False
    if nested:
        def _constant_per_fold():
            bf = [float(v) for v in by_fold]
            return len(bf) == ROUTER_N_FOLDS and all(
                np.all(thr_row[fold == k] == bf[k]) for k in range(ROUTER_N_FOLDS))
        chk("nested thresholds: per-row threshold constant per fold and = by_fold[k]",
            _constant_per_fold, _safe(lambda: [round(float(v), 6) for v in by_fold]))
        chk("stored decision = (raw OOF score >= per-row nested threshold)",
            lambda: dec.shape == (ref.n_train,) and np.isin(dec, (0, 1)).all()
            and np.array_equal(dec.astype(int), (pr >= thr_row).astype(int)))
    if _hex(split_fp, _ROUTER_FP):
        split_status = VALIDATED
        chk("split fingerprint (router scheme) = recomputed", lambda: split_fp == ref.router_split_fingerprint,
            f"{split_fp} vs {ref.router_split_fingerprint}")
    elif ref.stage_io_split_fingerprint:
        split_status = ATTESTED
        chk("split fingerprint (stage_io scheme) = attested value",
            lambda: split_fp == ref.stage_io_split_fingerprint,
            f"{split_fp} vs attested {ref.stage_io_split_fingerprint}")
    else:
        split_status = RECORDED_ONLY
        chk("split fingerprint is a stage_io (12-hex) value", lambda: _hex(split_fp, _STAGE_IO_FP),
            repr(split_fp))
    checks = _report_failures(rep, what)

    scores = {
        "raw_model_score": ScoreStream("ebm.oof_proba", _frozen(pr, float), "raw_model_score",
                                       "oof.proba", "out-of-fold EBM score; not calibrated"),
        "calibrated_probability": ScoreStream(
            "ebm.oof_proba_calibrated", _frozen(pc, float), "calibrated_probability",
            "oof.proba_calibrated", "per-fold calibration from inner OOF of each fold's training rows"),
    }
    thresholds = {
        "ebm.global_f2_raw": ThresholdRecord(
            "ebm.global_f2_raw", float(t_raw), "raw_model_score", "threshold.oof.threshold",
            SELECTION_GLOBAL_TRAIN_LABELS, "F2 selection on all training rows' raw OOF scores"),
        "ebm.global_f2_calibrated": ThresholdRecord(
            "ebm.global_f2_calibrated", float(t_cal), "calibrated_probability",
            "threshold.oof_calibrated_space.threshold", SELECTION_GLOBAL_TRAIN_LABELS,
            "F2 selection on all training rows' calibrated OOF probabilities"),
    }
    stored = {}
    if nested:
        stored["ebm.nested_f2"] = StoredDecision(
            "ebm.nested_f2", _frozen(dec.astype(int)), _frozen(thr_row, float), "raw_model_score",
            "oof.decision / oof.threshold_per_row", SELECTION_FOLD_NESTED,
            tuple(float(v) for v in by_fold))
    split_detail = {VALIDATED: "router scheme, recomputed",
                    ATTESTED: f"stage_io scheme = attested ({ref.stage_io_attestation})",
                    RECORDED_ONLY: "stage_io scheme; no attestation supplied"}[split_status]
    provenance = (
        ProvenanceItem("row_index", VALIDATED, "index_train equals X_train index"),
        ProvenanceItem("training_labels", VALIDATED, "carried and equal to GLOBAL_SPLIT y_train"),
        ProvenanceItem("fold_plan", VALIDATED,
                       f"{ROUTER_N_FOLDS} folds, seed {plan_seed}, fingerprint {fold_fp_fn(fold)}"),
        ProvenanceItem("calibration_fold_gates", VALIDATED, _safe(lambda: list(fold_applied))),
        ProvenanceItem("nested_decisions", VALIDATED if nested else NOT_RECORDED,
                       "stored decisions reproduce from per-fold nested thresholds" if nested
                       else "drives_train_decisions is False: no stored nested decisions"),
        ProvenanceItem("split_fingerprint", split_status, f"{split_fp} ({split_detail})"),
        ProvenanceItem("block_fingerprint", RECORDED_ONLY if block_fp else NOT_RECORDED,
                       str(block_fp)),
        ProvenanceItem("feature_hash", NOT_RECORDED, "no feature-matrix hash is stored"),
        sha_item,
    )
    return UpstreamOOFView(
        model="ebm", arm=expected_arm, index=ref.index_train, y=ref.y_train,
        fold_id=_frozen(fold, int), scores=_proxy(scores), thresholds=_proxy(thresholds),
        stored_decisions=_proxy(stored), provenance=provenance,
        source=_proxy({"path": path, "checks": tuple(checks.to_dict("records")),
                       "fields_read": tuple(sorted(F.paths)),
                       "historical_test_exposure": HISTORICAL_TEST_EXPOSURE}),
    )


# ======================================================================
# Stage 4 recall-targeted cuts (secondary operating points)
# ======================================================================
def _default_recall_fn():
    try:
        return getattr(importlib.import_module("glass_pipeline.meta_ebm.meta_stage"),
                       "find_recall_threshold")
    except (ImportError, AttributeError):
        return None


def extract_stage4_recall_thresholds(meta, ref: TrainReference, *, lr_view: UpstreamOOFView,
                                     ebm_view: UpstreamOOFView, path=None,
                                     expected_sha256: Optional[str] = None,
                                     recall_threshold_fn: Optional[Callable] = None
                                     ) -> Stage4Thresholds:
    """
    The Meta-EBM's recall-targeted LR / EBM cuts (``thresholds.lr`` / ``.ebm``),
    as secondary operating points in the calibrated space. Each is recomputed
    with the Meta-EBM's own ``find_recall_threshold`` from the views' training
    OOF probabilities and labels; equal -> validated (the artifact was built on
    these inputs). If that function is not importable the values are
    ``recorded_only``.
    """
    what = "Stage 4 Meta-EBM thresholds"
    art, opened_path, sha_item = _open(meta, expected_sha256, what)
    path = opened_path or path
    F = _Fields(art, what)
    t_lr, t_ebm = F.get("thresholds", "lr"), F.get("thresholds", "ebm")
    target = F.get("thresholds", "target_recall")
    src_lr, src_ebm = F.get("thresholds", "sources", "lr"), F.get("thresholds", "sources", "ebm")
    split_fp = F.get("split_fingerprint", required=False)
    F.raise_if_missing()
    for v, m in ((lr_view, "lr"), (ebm_view, "ebm")):
        if v.model != m or not v.index.equals(ref.index_train):
            raise RouterDiagnosticError(f"{what}: {m}_view is not the {m} view on these training rows")

    rep = ValidationReport()
    chk = lambda name, fn, detail="": rep.check(_SECTION, what, name, fn, detail)  # noqa: E731
    chk("target_recall is 0.70", lambda: float(target) == 0.70, repr(target))
    for m, t, s in (("lr", t_lr, src_lr), ("ebm", t_ebm, src_ebm)):
        chk(f"{m} cut finite in (0, 1)", lambda t=t: 0.0 < float(t) < 1.0, repr(t))
        chk(f"{m} cut source is find_recall_threshold on train OOF",
            lambda s=s: "find_recall_threshold(y_train" in str(s), str(s))
    if ref.stage_io_split_fingerprint and split_fp is not None:
        chk("Stage 4 split fingerprint = attested value",
            lambda: split_fp == ref.stage_io_split_fingerprint, str(split_fp))
    fn = recall_threshold_fn or _default_recall_fn()
    lr_stream = lr_view.scores.get("calibrated_probability")
    ebm_stream = ebm_view.scores.get("calibrated_probability")
    if fn is not None:
        for m, t, st in (("lr", t_lr, lr_stream), ("ebm", t_ebm, ebm_stream)):
            chk(f"{m} cut recomputes from the view's calibrated OOF and training labels",
                lambda t=t, st=st: st is not None
                and float(fn(ref.y_train, np.asarray(st.values), float(target))) == float(t),
                _safe(lambda st=st: f"recomputed {float(fn(ref.y_train, np.asarray(st.values), float(target))):.6f}"))
    checks = _report_failures(rep, what)
    status = VALIDATED if fn is not None else RECORDED_ONLY
    records = {
        f"{m}.stage4_recall70": ThresholdRecord(
            f"{m}.stage4_recall70", float(t), "calibrated_probability", f"thresholds.{m}",
            SELECTION_GLOBAL_TRAIN_LABELS,
            "highest observed cut with training OOF recall >= 0.70 (Stage 4's own; all "
            "training labels)")
        for m, t in (("lr", t_lr), ("ebm", t_ebm))}
    provenance = (
        ProvenanceItem("recall_cuts", status,
                       "recomputed with find_recall_threshold" if fn is not None
                       else "find_recall_threshold not importable"),
        ProvenanceItem("stage_io_split_fingerprint",
                       ATTESTED if ref.stage_io_split_fingerprint and split_fp is not None
                       else RECORDED_ONLY, str(split_fp)),
        sha_item,
    )
    return Stage4Thresholds(records=_proxy(records), target_recall=float(target),
                            provenance=provenance,
                            source=_proxy({"path": path, "checks": tuple(checks.to_dict("records")),
                                           "fields_read": tuple(sorted(F.paths))}))


# ======================================================================
# RF Pass 1 outer score candidate (GLASS-arm RF result)
# ======================================================================
def extract_rf_pass1_candidate(rf_result, ref: TrainReference, *, path=None,
                               expected_sha256: Optional[str] = None,
                               fold_fingerprint_fn: Optional[Callable] = None
                               ) -> RFPass1ScoreCandidate:
    """
    ``oof_proba`` of the GLASS-arm RF result (GLASS notebook Cell 16): the
    per-fold RF refit of ``crossfit_stage2``, a CANDIDATE for the RF Router's
    outer Pass 1 score. Usable only after ``check_rf_pass1_score_equivalence``
    passes. This artifact also stores ``glass_router_data`` (with the engineered
    test labels) and ``metrics_test``; neither is read.
    """
    what = "GLASS-arm RF result"
    art, opened_path, sha_item = _open(rf_result, expected_sha256, what)
    path = opened_path or path
    if not isinstance(art, Mapping):
        raise RouterDiagnosticError(f"{what}: expected a dict, got {type(art).__name__}")
    F = _Fields(art, what)
    flag = F.get("train_outputs_oof")
    values, idx, fold_raw = F.get("oof_proba"), F.get("oof_index"), F.get("oof_fold_id")
    params, split_fp = F.get("params"), F.get("split_fingerprint")
    proto = {k: F.get("oof_protocol", k, required=k in ("n_folds", "random_state",
                                                          "fold_fingerprint",
                                                          "train_label_fingerprint"))
             for k in ("n_folds", "random_state", "fold_fingerprint", "train_label_fingerprint",
                       "rf", "feature_engineering", "glass_rf_model", "fold_source")}
    F.raise_if_missing()

    rep = ValidationReport()
    chk = lambda name, fn, detail="": rep.check(_SECTION, what, name, fn, detail)  # noqa: E731
    chk("train_outputs_oof is True", lambda: flag is True, repr(flag))
    chk("oof_index = X_train index (same order)",
        lambda: index_alignment(idx, ref.index_train, f"{what} oof_index") == "label")
    v = np.asarray(values, dtype=float)
    chk("oof_proba finite in [0, 1]", lambda: v.shape == (ref.n_train,) and check_proba(v, "oof_proba"))
    fold_fp_fn = _resolve_fold_fp_fn(fold_fingerprint_fn)
    fold = _fold_checks(chk, fold_raw, ref, n_folds_expected=ROUTER_N_FOLDS,
                        declared_n_folds=proto["n_folds"], seed=proto["random_state"],
                        fingerprint=proto["fold_fingerprint"], fold_fp_fn=fold_fp_fn)
    chk("cross-fit label fingerprint = training rows and labels",
        lambda: label_identity(proto["train_label_fingerprint"], ref.train_label_fingerprint, what))
    chk("router split fingerprint = recomputed", lambda: split_fp == ref.router_split_fingerprint,
        f"{split_fp} vs {ref.router_split_fingerprint}")
    chk("params is a dict", lambda: isinstance(params, Mapping), type(params).__name__)
    checks = _report_failures(rep, what)
    return RFPass1ScoreCandidate(
        index=ref.index_train, fold_id=_frozen(fold, int), values=_frozen(v, float),
        params=_proxy(params), random_state=int(proto["random_state"]),
        split_fingerprint=split_fp, fold_fingerprint=proto["fold_fingerprint"],
        train_label_fingerprint=proto["train_label_fingerprint"],
        protocol=_proxy({k: val for k, val in proto.items() if val is not None}),
        provenance=(ProvenanceItem("row_index", VALIDATED, "oof_index equals X_train index"),
                    ProvenanceItem("fold_plan", VALIDATED, f"fingerprint {proto['fold_fingerprint']}"),
                    ProvenanceItem("train_label_fingerprint", VALIDATED,
                                   str(proto["train_label_fingerprint"])),
                    ProvenanceItem("router_split_fingerprint", VALIDATED, str(split_fp)),
                    ProvenanceItem("equivalence", NOT_RECORDED,
                                   "a candidate only: see check_rf_pass1_score_equivalence"),
                    sha_item),
        source=_proxy({"path": path, "checks": tuple(checks.to_dict("records")),
                       "fields_read": tuple(sorted(F.paths))}),
    )
