"""PR 44 Step 2B: register a fixed policy against the accepted local Step 2A receipt."""
import json
from math import isfinite
from pathlib import Path

from feature_research import research_records as records
from feature_research.stage4_policy import NOTEBOOK, POLICY_SHA256, load_policy
from shared.pinned_artifacts import sha256_file

ROOT = records.BANK_ROOT / "research_logs/pr44"
SOURCE = ROOT / "upstream_policy_v1.json"
DIGEST = "dceb16cb495ea8323dbb5668188a35af68fa5795f3b387ad1a991edce58f84d0"


def policy():
    value = json.loads(SOURCE.read_text(encoding="utf-8"))
    if records.content_digest({"policy": value}) != DIGEST:
        raise ValueError("Changed upstream policy; register a new version before use")
    return value


def verdict(interval, margin, *, safeguards=True, supported=True):
    """Positive effects favor the candidate. None denotes an unavailable safeguard."""
    if safeguards is False:
        return "rejected"
    if not supported or safeguards is None or interval is None:
        return "inconclusive"
    low, high = interval
    if not all(map(isfinite, (low, high, margin))) or low > high or margin <= 0:
        raise ValueError("Invalid paired interval or practical margin")
    if low > margin:
        return "improved"
    if low >= -margin and high <= margin:
        return "comparable"
    if high < -margin:
        return "rejected"
    return "inconclusive"


def check_partition(outer_training, outer_evaluation, fitting, validation):
    """Check global row IDs before any inner transform, stopping split or calibration fit."""
    groups = [set(g) for g in (outer_training, outer_evaluation, fitting, validation)]
    train, evaluation, fit, valid = groups
    if (any(not g or len(g) != len(ids) for g, ids in
            zip(groups, (outer_training, outer_evaluation, fitting, validation)))
            or train & evaluation or fit & valid or not (fit | valid) <= train):
        raise ValueError("Invalid or overlapping fitting/evaluation row identities")


def within_budget(budget, *, trials, fits, hours, conditional=False, triggered=False):
    """Projected cumulative cost including the proposed job; callers meter actual usage."""
    if not all(isfinite(v) and v >= 0 for v in (trials, fits, hours)):
        raise ValueError("Invalid projected cost")
    return ((not conditional or triggered) and trials <= budget["trials"]
            and fits <= budget["cap"] and hours <= budget["hours_max"])


def _baseline(root, design):
    rid = design["baseline_run"]
    record = records.read_run_record(rid, "upstream_baselines", root=root)
    stage4 = load_policy()
    frozen = record["pr43"]["frozen_inputs"]
    if (record["run"]["run_id"] != rid or record["run"]["commit"] != 2
            or record["run"]["notebook"] != NOTEBOOK or record["status"] != "reproduced"
            or record["pr43"]["policy"]["canonical_sha256"] != POLICY_SHA256
            or frozen["run_id"] != stage4["input_run"]
            or any(record["population"].get(k) != v for k, v in stage4["population"].items())):
        raise ValueError("Incompatible Step 2A run, policy or training population")
    # Read only the direct accepted summaries that identify the two row-ordered tables.
    diag = records.read_run_record(record["diagnostic"]["run_id"], "voter_diagnostics", root=records.BANK_ROOT/"research_logs/pr43")
    reference = diag["nested_reference"]
    nested = records.read_run_record(reference["run_id"], "nested_references", root=records.BANK_ROOT/"research_logs/pr43")
    if (diag["content_sha256"] != "b5ab3779cb5f74557ce7cb7a81e37a8c4be3a469d883a33e0f9b116e875ce8ba"
            or diag["content_sha256"] != record["diagnostic"]["content_sha256"]
            or diag["inputs"] != record["pr43"] or nested["content_sha256"] != reference["content_sha256"]
            or record["tables_sha256"] != nested["inputs"]["tables_sha256"]):
        raise ValueError("Incompatible frozen upstream identities")
    roles = {"stage1.lr", "stage1.mlp", "stage2.glass_router", "stage2.rf_router", "stage3.ebm", "stage3.xgb"}
    if record["local_artifacts"] != dict.fromkeys(roles, True) or set(record["calibrated_oof"]) != {"glass", "blackbox"}:
        raise ValueError("Incomplete upstream evidence")
    for arm, counts in (("glass", (14628, 1120)), ("blackbox", (10839, 686))):
        rows = record["calibrated_oof"][arm]
        if len(rows) != 4 or {(r["scope"], r["voter"]) for r in rows} != {
                (s, v) for s in ("all", "deferred") for v in ("stage1", "stage3")}:
            raise ValueError("Incomplete Step 2A baseline rows")
        for row in rows:
            n, pos = (32950, 3712) if row["scope"] == "all" else counts
            if ((row["n"], row["positive"], row["negative"]) != (n, pos, n-pos)
                    or not abs(row["prevalence"] - pos/n) <= 1e-12
                    or any(not 0 <= row[k] <= 1 for k in ("average_precision", "roc_auc", "brier", "ece"))
                    or not isfinite(row["log_loss"]) or row["log_loss"] < 0):
                raise ValueError("Invalid baseline measurement or support")
    path = Path(root) / rid / "upstream_baselines.json"
    return {"run_id": rid, "content_sha256": record["content_sha256"], "file_sha256": sha256_file(path),
            "tables_sha256": record["tables_sha256"]}


def _receipt(root):
    design = policy()
    return {"policy": {"id": design["id"], "version": design["version"], "digest": DIGEST},
                "baseline": _baseline(root, design), "stage4_policy_sha256": POLICY_SHA256}


def register(root=ROOT):
    evidence = _receipt(root)
    run = records.start_run(commit=2, seeds={"bootstrap": policy()["uncertainty"]["seed"]}, root=root,
                            notebook=NOTEBOOK, code_files=(NOTEBOOK, SOURCE.relative_to(records.BANK_ROOT).as_posix(),
                            "feature_research/upstream_registration.py", "feature_research/research_records.py"))
    return records.write_run_record(run, "upstream_registration", evidence)


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify", metavar="RUN_ID", help="recheck the receipt against current policy and parent")
    args = cli.parse_args()
    if args.verify:
        receipt = records.read_run_record(args.verify, "upstream_registration", root=ROOT)
        if (receipt["run"]["run_id"] != args.verify
                or receipt["content_sha256"] != records.content_digest(_receipt(ROOT))):
            raise ValueError("Registration differs from policy or accepted Step 2A parent")
        print(receipt["content_sha256"])
    else:
        print(register())
