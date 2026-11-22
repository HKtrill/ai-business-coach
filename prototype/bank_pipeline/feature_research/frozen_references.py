"""
feature_research.frozen_references
==================================
PR 42, Commit 2: the frozen references later PR 42 commits compare against,
recomputed from this branch's code and compared with the values the accepted
PR 39 records hold.

    static_references()                 code and records only (no data)
    frozen_references(TRAIN)            + the train-side identities, from training rows
    accepted_artifact_views(TRAIN.train_reference)
                                        the accepted Stage 1-3 artifacts, read ONLY
                                        through the PR 38 boundary extractors

References
----------
Stage 1     feature contract 2.0.0 (and the read-only historical 1.0.0), the
            ordered-name SHA-256, the 10-fold plan
Stage 2     binning identity (``shared.stage2.binning.binning_identity``), the
            ordered 29-bin SHA-256, the 8 sources, the inner-fold feature pathway
Stage 3     ordered 13-feature SHA-256, the feature-DAG implementation identity
            (``shared.stage3.block_cache.implementation_identity``), the recorded
            block content fingerprint and block-cache file, the three EBM
            interaction pairs, the shared 5-fold plan
Stage 4     Meta-XGB input channels

Statuses
--------
``verified``             recomputed here and equal to the recorded value
``MISMATCH``             recomputed here and different (the run must stop)
``code only``            no version-controlled record holds it; the value is the code's
``recorded only``        cannot be recomputed without the test split; the record's value
``local verification``   held by a local, Git-ignored file (an artifact or the block
                         cache), identified by size and SHA-256 and, for an artifact with a
                         PR 38 extractor, read through it; checked by
                         ``tests/realdata/test_pr42_frozen_references.py``

Records are read only after their ``.sha256`` sidecar matches (the PR 39 Step 3A
record has none; its SHA-256 is reported instead). No artifact is opened
except by ``accepted_artifact_views``, and that only through
``shared.stage2.diagnostics.boundary`` extractors, which read allowlisted
training-side fields and refuse test-side ones. Nothing here, in the notebook or
in the PR 42 tests unpickles an accepted model or the Stage 3 block cache outside
those extractors: what the extractors do not expose (the EBM's stored
interaction pairs, the block cache's stored implementation identity) is taken
from the production code and stated as such.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

__all__ = ["RECORDS", "PINS_FILE", "EXTRACTED_ROLES", "UNEXTRACTED_ROLES", "read_record",
           "dag_implementation", "static_references", "frozen_references",
           "accepted_artifact_views", "names_sha256"]

BANK_ROOT = Path(__file__).resolve().parents[1]

#: The accepted records the references are compared with.
RECORDS = MappingProxyType({
    "c3_binning": "research_logs/pr39/c3_20261114_231604/stage2_binning_evidence.json",
    "c4_run": "research_logs/pr39/c4_20261115_204125/run_record.json",
    "c4_manifest": "research_logs/pr39/c4_20261115_204125/artifact_manifest.json",
    "c5_run": "research_logs/pr39/c5_20261116_153459/run_record.json",
})
#: The frozen Phase 1 Stage 1-3 artifacts: PR 39 Commit 4's manifest, a pins file
#: (``shared.pinned_artifacts.pinned_path(role, pins_file=PINS_FILE)``).
PINS_FILE = RECORDS["c4_manifest"]

#: Role -> (boundary extractor, expected_arm).
EXTRACTED_ROLES = MappingProxyType({
    "stage1.lr": ("extract_lr_view", "glass"),
    "stage2.glass_router": ("extract_router_view", "glass"),
    "stage2.rf_router": ("extract_router_view", "blackbox"),
    "stage3.ebm": ("extract_ebm_view", "glass"),
})
#: Accepted roles with no PR 38 boundary extractor: never opened here.
UNEXTRACTED_ROLES = ("stage1.mlp", "stage3.xgb")

#: The block cache the accepted PR 39 Commit 4 run wrote (artifact_manifest inventory).
BLOCK_CACHE = "models/stage3/stage3_block_d6cddadc3439_c950583f1d15.joblib"


# ----------------------------------------------------------------------
# records
# ----------------------------------------------------------------------
def read_record(key: str, root: Path = BANK_ROOT) -> Tuple[Any, Dict[str, Any]]:
    """(parsed record, provenance): the file's SHA-256 and its sidecar status."""
    rel = RECORDS[key]
    path = Path(root) / rel
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    side = path.with_name(path.name + ".sha256")
    if side.is_file():
        if side.read_text(encoding="ascii").strip() != digest:
            raise RuntimeError(f"{rel} does not match its .sha256 sidecar")
        status = "sidecar verified"
    else:
        status = "no sidecar"
    return json.loads(data.decode("utf-8")), {"path": rel, "sha256": digest, "sidecar": status}


def _get(obj: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        obj = obj[key]
    return obj


def names_sha256(names) -> str:
    """The PR 39 ordered-name identity (``upstream_regeneration.checks.identities``)."""
    return hashlib.sha256(json.dumps([str(x) for x in names]).encode("utf-8")).hexdigest()


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def dag_implementation() -> Dict[str, Any]:
    """
    The Stage 3 feature DAG as the block cache identifies it: every module
    reachable from ``Stage3Block`` and ``EBMFeaturePipeline``.

    ``block_cache`` hashes raw bytes, so its digests depend on the checkout's line
    endings; the LF-normalised digests (and their combined SHA-256) do not.
    """
    import importlib.util
    from shared.stage3.block_cache import implementation_identity
    from shared.stage3.data import Stage3Block
    from shared.stage3.feature_engineering import EBMFeaturePipeline

    raw = implementation_identity(Stage3Block, EBMFeaturePipeline)
    lf, files = {}, {}
    for module in raw:
        origin = Path(importlib.util.find_spec(module).origin)
        files[module] = origin.relative_to(BANK_ROOT).as_posix()
        lf[module] = _lf_sha256(origin)
    combined = hashlib.sha256("".join(f"{m}\0{lf[m]}\n" for m in sorted(lf)).encode()).hexdigest()
    return {"modules": sorted(raw), "files": files, "sha256_lf": lf,
            "sha256_lf_combined": combined, "block_cache_scheme_sha256": raw,
            "order": ["drop leaky", "crisis", "integrals", "derivatives", "temporal", "prior",
                      "overlap", "select EBM_FEATURES"]}


def _ref(name, computed, recorded, recorded_in, status, note="") -> Dict[str, Any]:
    return {"name": name, "computed": computed, "recorded": recorded,
            "recorded_in": recorded_in, "status": status, "note": note}


def _match(name, computed, recorded, recorded_in, note="") -> Dict[str, Any]:
    return _ref(name, computed, recorded, recorded_in,
                "verified" if computed == recorded else "MISMATCH", note)


def static_references(root: Path = BANK_ROOT) -> List[Dict[str, Any]]:
    """Every reference computable from code and records alone (no data, deterministic)."""
    from blackbox_pipeline.models.meta_xgb.config import STAGE4_FEATURES
    from glass_pipeline.ebm.interactions import define_ebm_interactions
    from shared.stage1.feature_engineering import (
        HISTORICAL_STAGE1_FEATURE_CONTRACT, LR_FEATURES, STAGE1_FEATURE_CONTRACT,
        STAGE1_FEATURE_RENAMES, feature_contract_record)
    from shared.stage2.binning import RF_FEATURES_BINARY, binning_identity
    from shared.stage2.feature_engineering import RF_SOURCE_FEATURES
    from shared.stage3.feature_engineering import EBM_FEATURES
    from stage2_crossfit import inner_features as IF
    import pandas as pd

    c3, _ = read_record("c3_binning", root)
    c4, _ = read_record("c4_run", root)
    c5, _ = read_record("c5_run", root)
    manifest, _ = read_record("c4_manifest", root)
    c4p, c5p, c3p = RECORDS["c4_run"], RECORDS["c5_run"], RECORDS["c3_binning"]
    ids4 = c4["identities"]
    out = []

    s1 = feature_contract_record(STAGE1_FEATURE_CONTRACT)
    out.append(_match("Stage 1 feature contract 2.0.0",
                      f"{s1['id']} {s1['version']} {s1['features']} fingerprint {s1['fingerprint']}",
                      "{id} {version} {features} fingerprint {fingerprint}".format(
                          **_get(ids4, "feature_contracts.stage1.contract")),
                      f"{c4p} identities.feature_contracts.stage1.contract"))
    out.append(_match("Stage 1 ordered names SHA-256", names_sha256(LR_FEATURES),
                      _get(ids4, "feature_contracts.stage1.sha256"),
                      f"{c4p} identities.feature_contracts.stage1.sha256"))
    h = feature_contract_record(HISTORICAL_STAGE1_FEATURE_CONTRACT)
    out.append(_ref("Stage 1 historical contract 1.0.0 (read-only)",
                    f"{h['features']} fingerprint {h['fingerprint']}; renames "
                    f"{dict(STAGE1_FEATURE_RENAMES)}", "-",
                    "shared/stage1/feature_engineering.py", "code only",
                    "names every pre-PR 39 artifact carries; read as saved, never written"))
    b = binning_identity()
    out.append(_match("Stage 2 binning identity (SHA-256)", b["sha256"],
                      _get(ids4, "stage2_binning.sha256"), f"{c4p} identities.stage2_binning"))
    out.append(_match("Stage 2 binning identity, PR 39 Step 3A decision", b["sha256"],
                      _get(c3, "decision.binning_identity.sha256"),
                      f"{c3p} decision.binning_identity (no sidecar)",
                      f"outcome {_get(c3, 'decision.outcome')!r} for NSD and ECI"))
    out.append(_match("Stage 2 ordered 29 bins SHA-256", names_sha256(RF_FEATURES_BINARY),
                      _get(ids4, "feature_contracts.stage2.sha256"),
                      f"{c4p} identities.feature_contracts.stage2.sha256"))
    out.append(_ref("Stage 2 sources (8, ordered) SHA-256", names_sha256(RF_SOURCE_FEATURES), "-",
                    "shared/stage2/feature_engineering.py RF_SOURCE_FEATURES", "code only"))
    out.append(_match("Stage 2 inner-fold feature pathway",
                      json.loads(json.dumps(IF.pathway_identity(IF.PATHWAY_FOLD))),
                      _get(ids4, "configuration.feature_pathway"),
                      f"{c4p} identities.configuration.feature_pathway"))
    out.append(_match("Stage 3 ordered 13 features SHA-256", names_sha256(EBM_FEATURES),
                      _get(ids4, "feature_contracts.stage3.sha256"),
                      f"{c4p} identities.feature_contracts.stage3.sha256"))
    ev = c4["evidence"]["stage3"]
    out.append(_match("Stage 3 features of both accepted Stage 3 artifacts",
                      list(EBM_FEATURES),
                      ev["stage3.ebm"]["features"] if ev["stage3.ebm"]["features"] ==
                      ev["stage3.xgb"]["features"] else "EBM and XGBoost records differ",
                      f"{c4p} evidence.stage3.stage3.ebm / stage3.xgb .features"))
    dag = dag_implementation()
    out.append(_ref("Stage 3 feature DAG implementation (LF SHA-256 over its modules)",
                    f"{dag['sha256_lf_combined']} ({len(dag['modules'])} modules: "
                    + ", ".join(dag["files"][m] for m in dag["modules"]) + ")", "-",
                    "no PR 39 record holds per-module digests", "code only",
                    "the accepted block cache stores the block_cache identity but is not opened "
                    "(it holds the test split); these LF digests, written by every PR 42 run, "
                    "are the reference later commits compare with"))
    fp = {ev["stage3.ebm"]["block_fingerprint"], ev["stage3.xgb"]["block_fingerprint"]}
    out.append(_ref("Stage 3 block content fingerprint",
                    "not recomputed: the block holds the test split", " / ".join(sorted(fp)),
                    f"{c4p} evidence.stage3 (stage3.ebm and stage3.xgb)", "recorded only",
                    "the accepted EBM's own block_fingerprint is compared through "
                    "extract_ebm_view when the artifact is local"))
    cache = next((f for f in manifest["inventory"]["files"] if f["path"] == BLOCK_CACHE), None)
    out.append(_ref("Stage 3 block-cache file", BLOCK_CACHE,
                    f"{cache['bytes']} bytes, SHA-256 {cache['sha256']}" if cache else "absent",
                    f"{RECORDS['c4_manifest']} inventory.files", "local verification",
                    "identified by size and SHA-256 only, never unpickled (it holds the test "
                    "split); name = stage3_block_<stage_io split fingerprint>_<block_identity "
                    "digest[:12]>"))
    pairs = define_ebm_interactions(pd.DataFrame(columns=list(EBM_FEATURES)))
    out.append(_ref("EBM interaction pairs (3)", "; ".join(f"{a} x {b}" for a, b in pairs), "-",
                    "glass_pipeline/ebm/interactions.py define_ebm_interactions", "code only",
                    "verified: the production definition, which train_ebm_stage passes to the "
                    "EBM factory and stores as extras['interactions'] (tests/stage3/"
                    "test_glass_ebm_stage.py), yields these pairs for the 13 EBM_FEATURES. Not "
                    "verified: the pairs stored in the accepted EBM; the PR 38 EBM extractor "
                    "does not expose extras, and the model is not opened outside it"))
    out.append(_match("Stage 4 Meta-XGB inputs SHA-256", names_sha256(STAGE4_FEATURES),
                      _get(c5, "identities.feature_contracts.stage4_meta_xgb.sha256"),
                      f"{c5p} identities.feature_contracts.stage4_meta_xgb.sha256"))
    pop = ids4["population"]
    out.append(_ref("Train-side population",
                    "verified on training rows by the notebook",
                    f"rows {pop['train_rows']}, positives {pop['train_positives']}, label "
                    f"{pop['train_label_fingerprint']}, shared 5-fold "
                    f"{pop['shared_fold_fingerprint']}, Stage 1 10-fold "
                    f"{pop['stage1_fold_plan']['fingerprint']}, router split "
                    f"{pop['router_split_fingerprint']}, stage_io split "
                    f"{pop['stage_io_split_fingerprint']} (attested)",
                    f"{c4p} identities.population", "recorded only"))
    return out


def frozen_references(train_side, root: Path = BANK_ROOT) -> List[Dict[str, Any]]:
    """``static_references`` plus the train-side identities recomputed from ``train_side``."""
    c4, _ = read_record("c4_run", root)
    pop = c4["identities"]["population"]
    ids = train_side.identities
    out = [r for r in static_references(root) if r["name"] != "Train-side population"]
    p = f"{RECORDS['c4_run']} identities.population"
    for name, computed, recorded, field in (
            ("training rows", ids["train_rows"], pop["train_rows"], "train_rows"),
            ("training positives", ids["train_positives"], pop["train_positives"],
             "train_positives"),
            ("training label fingerprint", ids["train_label_fingerprint"],
             pop["train_label_fingerprint"], "train_label_fingerprint"),
            ("shared 5-fold plan fingerprint (Stages 2-3)", ids["shared_fold_fingerprint"],
             pop["shared_fold_fingerprint"], "shared_fold_fingerprint"),
            ("Stage 1 10-fold plan fingerprint", ids["stage1_fold_fingerprint"],
             pop["stage1_fold_plan"]["fingerprint"], "stage1_fold_plan.fingerprint"),
            ("router split fingerprint (row ids only)", ids["router_split_fingerprint"],
             pop["router_split_fingerprint"], "router_split_fingerprint")):
        out.append(_match(f"Train side: {name}", computed, recorded, f"{p}.{field}"))
    out.append(_ref("Train side: stage_io split fingerprint", "not recomputed (hashes test labels)",
                    pop["stage_io_split_fingerprint"], f"{p}.stage_io_split_fingerprint",
                    "recorded only", f"attested to the boundary extractors: "
                                     f"{ids['stage_io_split_fingerprint']}"))
    return out


# ----------------------------------------------------------------------
# accepted artifacts, through the PR 38 boundary extractors only
# ----------------------------------------------------------------------
def _provenance(view) -> List[Dict[str, str]]:
    return [{"name": p.name, "status": p.status, "detail": p.detail} for p in view.provenance]


def accepted_artifact_views(ref, root: Path = BANK_ROOT, *,
                            extractors: Optional[Mapping[str, Callable]] = None
                            ) -> List[Dict[str, Any]]:
    """
    One summary per accepted Stage 1-3 role: read through its PR 38 boundary
    extractor at the SHA-256 the PR 39 Commit 4 manifest pins, or not read.

    ``not available locally``  the pinned file is absent (models/ is Git-ignored)
    ``passed``                 pin and every mandatory extractor check passed
    ``FAILED``                 the file differs from its pin, or a check failed
    ``not read``               no boundary extractor exists for the role

    Only summaries leave this function (checks, provenance, fields read); the
    views, which carry training-side OOF vectors, are dropped.
    """
    from shared.pinned_artifacts import PinError, pinned_path, read_pins
    from shared.stage2.diagnostics import boundary

    root = Path(root)
    pins_file = root / PINS_FILE
    pins = read_pins(pins_file)
    c4, _ = read_record("c4_run", root)
    recorded_block = c4["evidence"]["stage3"]["stage3.ebm"]["block_fingerprint"]
    fns = dict(extractors or {})
    out = []
    for role, (fn_name, arm) in EXTRACTED_ROLES.items():
        entry = pins["files"][role]
        row = {"role": role, "extractor": f"shared.stage2.diagnostics.boundary.{fn_name}",
               "expected_arm": arm, "path": entry["path"], "sha256": entry["sha256"]}
        path = root / entry["path"]
        if not path.is_file():
            out.append({**row, "status": "not available locally"})
            continue
        try:
            verified = pinned_path(role, pins_file=pins_file, root=root)
            fn = fns.get(fn_name) or getattr(boundary, fn_name)
            view = fn(str(verified), ref, expected_arm=arm, expected_sha256=entry["sha256"])
        except (PinError, boundary.RouterDiagnosticError, OSError) as exc:
            out.append({**row, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:2000]})
            continue
        checks = list(view.source.get("checks", ()))
        row.update(status="passed", checks=len(checks),
                   checks_passed=sum(bool(c.get("passed")) for c in checks),
                   check_names=[c.get("check") for c in checks],
                   fields_read=list(view.source.get("fields_read", ())),
                   provenance=_provenance(view))
        if role == "stage3.ebm":
            got = next((p.detail for p in view.provenance if p.name == "block_fingerprint"), None)
            row["block_fingerprint"] = {"artifact": got, "recorded": recorded_block,
                                        "match": got == recorded_block}
            if got != recorded_block:
                row["status"] = "FAILED"
        out.append(row)
    for role in UNEXTRACTED_ROLES:
        entry = pins["files"][role]
        out.append({"role": role, "extractor": None, "path": entry["path"],
                    "sha256": entry["sha256"], "status": "not read",
                    "note": ("no PR 38 boundary extractor exists for this role; its feature "
                             "contract is the code's and the PR 39 Commit 4 record's")})
    return out
