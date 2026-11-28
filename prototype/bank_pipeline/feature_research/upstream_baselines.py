"""PR 44 Step 2A. Verify frozen training streams without fitting any estimator."""
from time import perf_counter

import numpy as np
from sklearn.metrics import log_loss

from feature_research import frozen_references, research_records as records
from feature_research import stage4_holdout, stage4_nested as nested, training_inputs as inputs
from feature_research.linear_arbiter import DIAGNOSTIC_RUN, DIAGNOSTIC_SHA
from feature_research.stage4_stability import RUN as COMPARATOR_RUN, SHA as COMPARATOR_SHA
from shared.metrics import binary_metrics

ROOT = inputs.BANK_ROOT / "research_logs/pr44"
NAME = "upstream_baselines"
COLUMNS = ("y", "stage1_fold", "stage2_fold", "stage3_fold", "stage1_proba", "route",
           "router_confidence", "router_score", "router_base_rate", "stage3_raw", "stage3_proba")
HOLDOUTS = {
    "glass": ("c5_20261126_184203", "971f8a374ad2ae4b4ddbb595c5fc5e63690e17b8c03991d78eec662e9f72fed3"),
    "blackbox": ("c5_20261126_104038", "416ab0a324d07fc08d120c62b697810b69ca5a404c667625bafaa7c788e816c7"),
}
PR42 = {
    "feature_inventory": "c2_20261122_124809", "encoding_audit": "c3_20261122_201830",
    "temporal_validity": "c3_20261122_232809", "binning_drift": "c4_20261123_081749",
    "binning_candidates": "c4_20261123_151515", "stage3_simplification": "c5_20261124_002645",
    "interaction_selection": "c5_20261124_122029",
}


def _quality(table, labels, expected_digest):
    # pandas' frame hash binds values and index, but does not bind column names.
    if (tuple(table.columns) != COLUMNS or not table.index.is_unique
            or not table.y.equals(labels) or nested._digest(table) != expected_digest):
        raise ValueError("Incompatible frozen OOF columns, rows, labels or predictions")
    result = []
    for scope, rows in (("all", table), ("deferred", table.loc[table.route.eq("uncertain")])):
        for voter in ("stage1", "stage3"):
            p = rows[voter + "_proba"]
            m = binary_metrics(rows.y, p)
            positive = int(rows.y.sum())
            result.append(dict(scope=scope, voter=voter, n=len(rows), positive=positive,
                               negative=len(rows)-positive, prevalence=rows.y.mean(),
                               roc_auc=m.roc_auc, average_precision=m.pr_auc, log_loss=log_loss(rows.y, p),
                               brier=m.brier, ece=m.ece))
    return result


def _check_glass(rows, accepted):
    for expected in accepted:
        row, = [r for r in rows if r["voter"] == expected["voter"] and r["scope"] == "deferred"]
        for field, value in expected.items():
            if field in ("voter", "n", "positive", "negative"):
                if row[field] != value:
                    raise ValueError(f"GLASS reference differs: {field}")
            else:
                np.testing.assert_allclose(row[field], value, atol=1e-12, rtol=0, err_msg=field)


def _inventory(index, reference):
    """A missing optional archive limits later paired work. A corrupt archive fails verification."""
    summaries = {name: {"run_id": rid, "content_sha256": records.read_run_record(rid, name)["content_sha256"]}
                 for name, rid in PR42.items()}
    archives = {}
    comparator = records.read_run_record(COMPARATOR_RUN, "stage4_comparators", root=nested.LOGS)
    if comparator["content_sha256"] != COMPARATOR_SHA:
        raise ValueError("Comparator inventory differs from the accepted record")
    for name, record in (("nested_references", reference), ("stage4_comparators", comparator)):
        rid = record["run"]["run_id"]
        path = nested.LOGS / rid / record["predictions"]["file"]
        if path.is_file():
            nested.load_predictions(rid, index=index, record_name=name)
        archives[name] = dict(run_id=rid, content_sha256=record["content_sha256"],
                              path=str(path.relative_to(inputs.BANK_ROOT).as_posix()),
                              row_archive="verified" if path.is_file() else "missing")
    for arm, (rid, digest) in HOLDOUTS.items():
        folder = nested.LOGS / "holdout_cache" / arm
        r = records.read_run_record(rid, "holdout_" + arm, root=nested.LOGS)
        if r["content_sha256"] != digest:
            raise ValueError(f"{arm}: simulated split record differs")
        required = [folder / f"{stage}_{e['producer']}" / f for stage, entries in r["ledger"].items()
                    for e in entries for f in ("receipt.json", "scores.npz")]
        missing = [str(p.relative_to(folder).as_posix()) for p in required if not p.is_file()]
        if not missing:
            stage4_holdout.verify(rid, arm, archives=True)
        archives["holdout_" + arm] = dict(run_id=rid, content_sha256=digest,
                                          path=folder.relative_to(inputs.BANK_ROOT).as_posix(),
                                          row_archive="incomplete" if missing else "verified",
                                          missing_files=len(missing))
    return dict(pr42_summary_records=summaries, pr43_archives=archives,
                use="PR 42 records support feature comparisons. PR 43 records support historical "
                    "summaries. Restore the matching numerical archive for new paired Stage 4 or "
                    "same-row refit contrasts. Published notebooks contain presentation outputs.")


def run(train):
    start = perf_counter()
    run_id = records.start_run(commit=2, seeds={"split_and_fold_seed": 42}, notebook=inputs.NOTEBOOK,
                               root=ROOT, code_files=(*nested.CODE_FILES, "feature_research/upstream_baselines.py",
                                                       "feature_research/frozen_references.py", "shared/metrics.py"))
    diag = records.read_run_record(DIAGNOSTIC_RUN, "voter_diagnostics", root=nested.LOGS)
    if diag["content_sha256"] != DIAGNOSTIC_SHA:
        raise ValueError("Voter diagnostics differ from the accepted record")
    frozen = records.read_run_record(diag["inputs"]["frozen_inputs"]["run_id"], "frozen_inputs", root=nested.LOGS)
    _, identity = nested.verified_inputs(frozen)
    reference = records.read_run_record(diag["nested_reference"]["run_id"], "nested_references", root=nested.LOGS)
    if identity != diag["inputs"] or reference["content_sha256"] != diag["nested_reference"]["content_sha256"]:
        raise ValueError("Frozen input or table reference differs from the diagnostic parent")
    if dict(train.identities) != frozen["population"] or not train.y_train.index.is_unique:
        raise ValueError("Training population differs from the accepted frozen inputs")
    checks = frozen_references.frozen_references(train)
    if any(c["status"] == "MISMATCH" for c in checks):
        raise ValueError("Frozen feature or population reference differs")
    for path, sha in frozen["source_records"].items():
        if inputs.boundary.file_sha256(inputs.BANK_ROOT / path) != sha:
            raise ValueError(f"PR 39 parent differs: {path}")
    pins = inputs.read_pins(inputs.BANK_ROOT / inputs.PINS_FILE)
    roles = {role for stages in inputs.ARMS.values() for role, _ in stages}
    if roles != set(pins["files"]) or roles != set(frozen["upstreams"]):
        raise ValueError("Frozen six-role manifest differs")
    available = {}
    for role, pin in pins["files"].items():
        receipt = frozen["upstreams"][role]
        if any(pin[k] != receipt[k] for k in ("sha256", "path")):
            raise ValueError(f"{role}: manifest differs from PR 43 receipt")
        path = inputs.BANK_ROOT / pin["path"]
        available[role] = path.is_file()
        if available[role] and (path.stat().st_size != pin["bytes"] or inputs.boundary.file_sha256(path) != pin["sha256"]):
            raise ValueError(f"{role}: local artifact differs from accepted pin")

    parent = inputs.baseline_records.require_passed(pins["run_id"], "run_record")
    hashes = reference["inputs"]["tables_sha256"]
    results = {}
    for arm, stages in inputs.ARMS.items():
        if all(available[role] for role, _ in stages):
            table, sources = inputs._arm_table(train, arm, pins["files"], parent["identities"]["population"],
                                               parent["evidence"])
            expected = {role: frozen["upstreams"][role] for role, _ in stages}
            if records.content_digest(sources) != records.content_digest(expected):
                raise ValueError(f"{arm}: producer contracts differ from receipt")
            results[arm] = _quality(table, train.y_train, hashes[arm])
    if "glass" in results:
        _check_glass(results["glass"], diag["deferred_quality"])
    evidence = dict(status="reproduced" if len(results) == 2 else "blocked_missing_artifacts",
                    population=dict(train.identities), pr39=frozen["source_records"], pr43=identity,
                    upstream_run=pins["run_id"], local_artifacts=available, tables_sha256=hashes,
                    diagnostic=dict(run_id=DIAGNOSTIC_RUN, content_sha256=diag["content_sha256"]),
                    historical_glass_deferred=diag["deferred_quality"],
                    historical_glass_sole_correct=diag["handoff"]["sole_correct_by_active_voter"],
                    calibrated_oof=results, evidence=_inventory(train.y_train.index, reference),
                    elapsed_seconds=perf_counter()-start)
    path = records.write_run_record(run_id, NAME, evidence)
    print(path)
    if len(results) != 2:
        missing = ", ".join(role for role, present in available.items() if not present)
        raise FileNotFoundError(f"Step 2A incomplete. Restore accepted pins for {missing}. Inventory: {path}")
    return path, results


if __name__ == "__main__":
    import pandas as pd

    _, results = run(inputs.draw_training_side(indices_only=True))
    print(pd.concat({arm: pd.DataFrame(rows) for arm, rows in results.items()}))
