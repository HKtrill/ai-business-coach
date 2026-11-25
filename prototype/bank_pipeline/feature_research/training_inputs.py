"""PR 43 Step 2A. Read the frozen OOF channels and replay the accepted operating rule."""
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from feature_research import research_records
from feature_research.frozen_references import BANK_ROOT, PINS_FILE
from feature_research.training_side import draw_training_side
from glass_pipeline.glass_arbiter.arbiter import meta_arbiter
from glass_pipeline.glass_arbiter.router_evidence import fold_base_rates
from glass_pipeline.glass_arbiter.tuning import tune_arbiter_threshold
from glass_pipeline.glass_router.pipeline import fold_fingerprint
from shared.pinned_artifacts import PinnedFile, read_pins, read_verified
from shared.stage2.diagnostics import boundary
from shared.stage4.protocol import ABSTENTION_MIN_COVERAGE, ABSTENTION_SEARCH, tune_min_confidence
from stage4_regeneration import runner as stage4_records
from upstream_regeneration import records as baseline_records

NOTEBOOK = "feature_research/training_research.ipynb"
RECORD_ROOT = BANK_ROOT / "research_logs/pr43"
CROSSOVER = "c5_20261116_153459"
ARMS = {
    "glass": (("stage1.lr", boundary.extract_lr_view),
              ("stage2.glass_router", boundary.extract_router_view),
              ("stage3.ebm", boundary.extract_ebm_view)),
    "blackbox": (("stage1.mlp", boundary.extract_mlp_view),
                 ("stage2.rf_router", boundary.extract_router_view),
                 ("stage3.xgb", boundary.extract_xgb_view)),
}


def _equal(actual, expected, name):
    if not np.array_equal(actual, expected):
        raise ValueError(f"Frozen reference mismatch: {name}")


def _arm_table(train, arm, pins, population, evidence):
    views, sources = [], {}
    for role, extract in ARMS[arm]:
        pin = pins[role]
        path = BANK_ROOT / pin["path"]
        if path.stat().st_size != pin["bytes"]:
            raise ValueError(f"{role}: pinned size mismatch")
        view = extract(path, train.train_reference, expected_arm=arm, expected_sha256=pin["sha256"])
        _equal(view.index, train.y_train.index, role + " row order")
        _equal(view.y, train.y_train, role + " labels")
        stage = role.split(".")[0]
        fp = fold_fingerprint(view.fold_id)
        expected = (population["stage1_fold_plan"]["fingerprint"] if stage == "stage1"
                    else population["shared_fold_fingerprint"])
        _equal(fp, expected, role + " producer folds")
        sources[role] = {"path": pin["path"], "sha256": pin["sha256"], "fold_fingerprint": fp,
                         "fields_read": list(view.source["fields_read"])}
        if stage != "stage2":
            sources[role]["streams"] = {key: stream.field for key, stream in view.scores.items()}
            config = evidence[stage][role]
            _equal(fp, config["fold_fingerprint"], role + " recorded folds")
            if stage == "stage1":
                _equal(view.source["model_name"], "calibrated_" + view.model, role)
                _equal(view.source["calibration_method"], config["calibration_method"], role)
                _equal(view.thresholds[view.model + ".in_stage_f2_cv"].value, config["threshold"], role)
            else:
                block = next(p.detail for p in view.provenance if p.name == "block_fingerprint")
                _equal(block, config["block_fingerprint"], role + " feature block")
                _equal(view.thresholds[view.model + ".global_f2_raw"].value, config["threshold"], role)
            sources[role]["accepted_configuration"] = config
        else:
            sources[role]["confidence"] = view.confidence_kind
            sources[role]["transform"] = "pass1: b*(1-c); pass2: b+(1-b)*c; uncertain: b, c=0; no vote"
            sources[role]["configuration"] = dict(view.metadata)
            sources[role]["route_counts"] = pd.Series(view.source_decision).value_counts().to_dict()
            sources[role]["population"] = dict(view.pass2_population)
        views.append(view)

    first, router, third = views
    _equal(router.fold_id, third.fold_id, arm + " Stage 2/3 folds")
    table = pd.DataFrame({
        "y": train.y_train.to_numpy(), "stage1_fold": first.fold_id,
        "stage2_fold": router.fold_id, "stage3_fold": third.fold_id,
        "stage1_proba": first.scores["calibrated_probability"].values,
        "route": router.source_decision, "router_confidence": router.confidence,
        "router_score": router.router_score.values,
        "router_base_rate": fold_base_rates(router.y, router.fold_id),
        "stage3_raw": third.scores["raw_model_score"].values,
        "stage3_proba": third.scores["calibrated_probability"].values,
    }, index=train.y_train.index)
    return table, sources


def _historical_search(table, name, config, artifact_pin):
    """Replay only the train-selected cut, with the recorded score and voter settings."""
    expected = config["abstention"]
    _equal(expected["min_coverage"], ABSTENTION_MIN_COVERAGE, name + " coverage floor")
    _equal(expected["search"], ABSTENTION_SEARCH, name + " search")
    if "weights" in config:
        cuts = config["thresholds"]
        result = tune_arbiter_threshold(
            meta_arbiter, tuple(table[k].to_numpy() for k in ("stage1_proba", "stage3_proba", "router_score")),
            tuple(table.route.eq(state).to_numpy() for state in ("pass1", "pass2")),
            table.y.to_numpy(), cuts["lr"], cuts["ebm"], config["weights"],
            glass_base_rate=table.router_base_rate.to_numpy())
        cut_key = "min_weighted_confidence"
    else:
        pin = PinnedFile(name, artifact_pin["path"], artifact_pin["sha256"], artifact_pin["bytes"])
        meta = joblib.load(BytesIO(read_verified(pin, BANK_ROOT)))
        _equal(meta.index_train, table.index, name + " row order")
        _equal(meta.learner.y_train, table.y, name + " labels")
        _equal(meta.operating["space"], "raw", name + " operating space")
        _equal(meta.operating["threshold"], config["operating"]["threshold"], name + " operating cut")
        _equal(meta.operating["train_side_thresholds"], "fold-nested per row", name)
        oof = meta.learner.oof
        _equal(oof["fold_id"], table.stage3_fold, name + " meta folds")
        p, t = np.asarray(oof["proba"]), np.asarray(oof["threshold_per_row"])
        _equal(p.shape, table.y.shape, name + " probability shape")
        _equal(t.shape, p.shape, name + " threshold shape")
        result = tune_min_confidence(table.y, p >= t, np.abs(p - t))
        cut_key = "min_confidence"
    for key in (cut_key, "train_coverage", "train_f2"):
        if result is None or not np.isclose(result[key], expected[key], rtol=0, atol=1e-12):
            raise ValueError(f"{name}: historical search did not reproduce {key}")
    return {key: result[key] for key in (cut_key, "train_coverage", "train_f2", "n_candidates")}


def frozen_inputs(train):
    """Both upstream tables and compact evidence. Fail before recording on any mismatch."""
    pins = read_pins(BANK_ROOT / PINS_FILE)
    run_id = pins["run_id"]
    run = baseline_records.require_passed(run_id, "run_record")
    accepted = baseline_records.require_passed(run_id, "upstream_acceptance")
    cross = stage4_records.require_passed(CROSSOVER, "crossover_baseline")
    links = cross["references"]["commit4"]
    _equal(links["run_id"], run_id, "crossover upstream")
    _equal(accepted["decision"]["accepted"], True, "upstream acceptance")
    digests = {}
    for rel, expected in (
        (PINS_FILE, accepted["artifact_manifest"]["sha256"]),
        (f"research_logs/pr39/{run_id}/upstream_acceptance.json", links["acceptance_sha256"]),
        (f"research_logs/pr39/{run_id}/run_record.json", accepted["records"]["run_record"]),
        (f"research_logs/pr39/{CROSSOVER}/crossover_baseline.json",
         "f05e2532dedeeb30f9da257b889f27b943c471a5d69a3f714e6e840dcced1b3c"),
    ):
        digest = boundary.file_sha256(BANK_ROOT / rel)
        _equal(digest, expected, rel)
        digests[rel] = digest

    pop = run["identities"]["population"]
    for key in ("train_rows", "train_positives", "train_label_fingerprint", "shared_fold_fingerprint",
                "router_split_fingerprint", "stage_io_split_fingerprint"):
        _equal(train.identities[key], pop[key], key)
    tables, sources, reference = {}, {}, {}
    for arm in ARMS:
        tables[arm], evidence = _arm_table(train, arm, pins["files"], pop, run["evidence"])
        sources.update(evidence)
    if not tables["glass"].index.equals(tables["blackbox"].index):
        raise ValueError("Mismatched cascade row order")
    _equal(tables["glass"].y, tables["blackbox"].y, "cross-cascade training labels")
    for name, artifact in cross["arbiters"].items():
        arm = artifact["upstream"]
        for stage, (role, _) in enumerate(ARMS[arm], start=1):
            upstream = artifact["upstream_files"][f"stage{stage}"]
            _equal(upstream["role"], role, name + " upstream role")
            _equal(upstream["sha256"], sources[role]["sha256"], name + " upstream pin")
        config = cross["results"]["cells"][name]["configuration"]
        replay = _historical_search(tables[arm], name, config, artifact)
        reference[name] = {"configuration": config, "replay": replay, "arbiter_sha256": artifact["sha256"]}
    return tables, {"source_records": digests, "population": dict(train.identities),
                    "upstreams": sources, "historical_reference": reference,
                    "reproduce": "python -m feature_research.training_inputs",
                    "boundary": "Stratified row identifiers; pinned train-side OOF fields only. "
                                "Stage I/O split fingerprint attested from the accepted record.",
                    "historical_search": "shared.stage4.protocol.tune_min_confidence: distinct "
                                         "training confidences; retain >= cut; coverage >= 0.50; "
                                         "maximize retained F2; ties choose the smaller cut.",
                    "claim": "Frozen random-split OOF benchmark; temporal validation is pending."}


def record_inputs(evidence):
    run = research_records.start_run(commit=2, seeds={"split_and_fold_seed": 42},
                                     notebook=NOTEBOOK, root=RECORD_ROOT,
                                     code_files=(NOTEBOOK, "feature_research/training_inputs.py",
                                                 "feature_research/training_side.py",
                                                 "shared/stage2/diagnostics/boundary.py",
                                                 "shared/stage4/protocol.py"))
    return research_records.write_run_record(run, "frozen_inputs", evidence)


if __name__ == "__main__":
    _, evidence = frozen_inputs(draw_training_side(indices_only=True))
    print(record_inputs(evidence))
