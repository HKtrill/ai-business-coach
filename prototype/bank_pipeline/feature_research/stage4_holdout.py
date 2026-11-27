"""PR 43 Step 5B. Fit blackbox, freeze its results, decide, then conditionally fit glass."""

import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_info, threadpool_limits

from feature_research import holdout_evaluation as ev, holdout_producers as prod, stage4_stability as a
from feature_research import holdout_recovery as recovery
from shared.pinned_artifacts import sha256_file
from upstream_regeneration import records as provenance

n, inputs = a.n, a.inputs
FIVE = "c5_20261125_224854"
SHA = "0969cbc371b1edb20ec4188c9ebf0921a73091a6790d490fccd8747a43314ad4"
FILES = (
    *n.CODE_FILES,
    "feature_research/stage4_holdout.py",
    "feature_research/holdout_producers.py",
    "feature_research/holdout_evaluation.py",
    "stage1_calibration/evidence.py",
)


def split_rows(y):
    if not y.index.is_unique:
        raise ValueError("Duplicate training rows")
    fit, held = train_test_split(np.arange(len(y)), test_size=0.2, random_state=4305, stratify=y)
    np.testing.assert_array_equal(np.sort(np.r_[fit, held]), np.arange(len(y)))
    return np.sort(fit), np.sort(held)


def sources():
    if a.verify(FIVE) != SHA:
        raise ValueError("Step 5A content mismatch")
    five = n.records.read_run_record(FIVE, a.NAME, root=n.LOGS)
    policy = n.load_policy()
    frozen = n.records.read_run_record(policy["input_run"], "frozen_inputs", root=n.LOGS)
    _, ids = n.verified_inputs(frozen)
    linked = {}
    for name, record in [
        ("nested", "nested_references"),
        ("linear", "linear_arbiter"),
        ("comparator", "stage4_comparators"),
    ]:
        item = five[name]
        linked[name] = n.records.read_run_record(item["run_id"], record, root=n.LOGS)
        if linked[name]["content_sha256"] != item["content_sha256"]:
            raise ValueError(f"{name} record mismatch")
    diag = n.records.read_run_record(a.comp.linear.DIAGNOSTIC_RUN, "voter_diagnostics", root=n.LOGS)
    a.comp.linear.eligibility(diag, linked["nested"], ids)
    if (
        diag["content_sha256"] != five["diagnostic"]
        or five["inputs"] != ids
        or five["budget"]["holdout"] != policy["search"]["holdout"]
    ):
        raise ValueError("Accepted evidence chain differs")
    return five, policy, frozen, linked


def digest(value):
    return n.records.content_digest({"value": value})


def checkpoint(folder, binding, stage, k, make):
    """Publish a complete producer atomically. Keep failed attempts, reject incompatible completed work."""
    path = folder / f"{stage}_{k}"
    if not path.exists():
        if make is None:
            raise ValueError(f"Missing required checkpoint: {path}")
        folder.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(dir=folder, prefix=f"{stage}_{k}_attempt_"))
        start = time.perf_counter()
        try:
            arrays, info = make()
            if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
                raise ValueError("Object arrays cannot be checkpointed")
            np.savez_compressed(temp / "scores.npz", **arrays)
            body = provenance.canonical(
                dict(
                    binding=binding,
                    stage=stage,
                    fold=k,
                    archive=sha256_file(temp / "scores.npz"),
                    info=dict(info, seconds=time.perf_counter() - start),
                )
            )
            (temp / "receipt.json").write_text(json.dumps(dict(body, digest=digest(body))), encoding="utf8")
            temp.rename(path)
        except Exception:
            (temp / "failure.txt").write_text(traceback.format_exc(), encoding="utf8")
            raise
    body = json.loads((path / "receipt.json").read_text(encoding="utf8"))
    sha = body.pop("digest")
    if (
        digest(body) != sha
        or body["binding"] != binding
        or (body["stage"], body["fold"]) != (stage, k)
        or sha256_file(path / "scores.npz") != body["archive"]
    ):
        raise ValueError(f"Stale or damaged checkpoint: {path}")
    if "audit_sha256" in body["info"]:
        audit = folder / "solver_audits" / f"router_{k}"
        if any(sha256_file(audit / (name + ".json")) != expected
               for name, expected in body["info"]["audit_sha256"].items()):
            raise ValueError(f"Solver audit differs: {audit}")
    with np.load(path / "scores.npz", allow_pickle=False) as z:
        return {key: z[key] for key in z.files}, dict(body["info"], checkpoint=sha, producer=k)


def assemble(p, cache):
    train, full, avg = pd.DataFrame(index=p.x.index), pd.DataFrame(index=p.h.index), pd.DataFrame(index=p.h.index)
    rules, ledger = [], {}
    for stage, fold in p.folds.items():
        vals, parts, records = [], [], []
        raw = np.full(len(train), np.nan)
        count = int(fold.max()) + 1
        for k in [*range(count), -1]:
            tr = np.arange(len(train)) if k == -1 else np.flatnonzero(fold != k)
            va = np.flatnonzero(fold == k)
            target = pd.concat([p.x.iloc[va], p.h], verify_integrity=True)

            def produce():
                arr, info = getattr(p, stage)(k, tr, va, target, raw)
                return arr, dict(info, fit_rows=len(tr), validation_rows=len(va), holdout_rows=len(p.h))

            arrays, info = cache(stage, k, produce)
            if k == 0:
                print(
                    f"{stage}: first {len(tr)} fitting rows, {info['seconds']:.1f}s. "
                    f"Other outer producers at similar size: ~{(count - 1) * info['seconds']:.0f}s.",
                    flush=True,
                )
            if "matches" in arrays:
                rules.append((k, info.pop("rules"), arrays.pop("matches"), p.x.index[va]))
            if "raw" in arrays:
                raw[va] = arrays.pop("raw")[: len(va)]
            df = pd.DataFrame(arrays, index=target.index)
            if df.isna().any().any():
                raise ValueError(f"Missing {stage} scores")
            if k == -1:
                full = full.join(df, validate="one_to_one")
            else:
                vals.append(df.iloc[: len(va)])
                parts.append(df.iloc[len(va) :])
            records.append(info)
        out = pd.concat(vals, verify_integrity=True).loc[train.index]
        train = train.join(out, validate="one_to_one")
        train[{"stage1": "stage1_fold", "router": "stage2_fold", "stage3": "stage3_fold"}[stage]] = fold
        avg = avg.join(
            prod.average_routes(parts) if stage == "router" else sum(parts) / count, validate="one_to_one"
        )
        ledger[stage] = records
    return train.assign(y=p.y), dict(full=full, average=avg), ledger, rules


def trigger(arms, retained, missing):
    hits = []
    for name, r in arms.items():
        for key, limit in [
            ("coverage", 0.05),
            ("positive_retention", 0.05),
            ("negative_retention", 0.05),
            ("loss", 0.005),
        ]:
            delta = r["paired"]["upstream"]["delta"][key]
            if abs(delta) >= limit or np.isclose(abs(delta), limit, rtol=0, atol=1e-14):
                hits.append([name, key, delta])
    return dict(
        expand=bool(hits or retained and missing),
        numeric=hits,
        necessary_missing_rule_evidence=bool(retained and missing),
    )


def verify(run_id, arm, archives=False):
    r = n.records.read_run_record(run_id, "holdout_" + arm, root=n.LOGS)
    if (
        r["arm"] != arm
        or r["step5a"] != SHA
        or r["policy"] != n.POLICY_SHA256
        or (r["split"]["fit_rows"], r["split"]["holdout_rows"]) != (26360, 6590)
        or r["split"]["fit_positive"] + r["split"]["holdout_positive"] != 3712
    ):
        raise ValueError("Invalid phase provenance or population")
    for stage, count in [("stage1", 111), ("router", 6), ("stage3", 31), ("meta", 6)]:
        field = "router_objects" if stage == "router" else "base_fits"
        if sum(v[field] for v in r["ledger"][stage]) != count:
            raise ValueError("Producer budget differs")
    if arm == "blackbox" and sum(v["forests"] for v in r["ledger"]["router"]) != 222:
        raise ValueError("RF forest budget differs")
    if set(r["arms"]) != (
        {"meta_xgb", "glass_formula", "logit_margin"} if arm == "glass" else {"meta_xgb", "glass_formula"}
    ):
        raise ValueError("Missing reference or unexpected candidate")
    for report in r["arms"].values():
        for mode, cell in report["conditions"].items():
            v = cell["issued"]
            prefix = "fit" if mode == "fitting_oof" else "holdout"
            if (
                v["kept"] + v["deferred_positive"] + v["deferred_negative"] != r["split"][prefix + "_rows"]
                or v["tp"] + v["fn"] + v["deferred_positive"] != r["split"][prefix + "_positive"]
            ):
                raise ValueError("Invalid issued-action accounting")
    if archives:
        folder = n.LOGS / "holdout_cache" / arm
        for stage, entries in r["ledger"].items():
            for entry in entries:
                k = entry["producer"]
                if not (folder / f"{stage}_{k}").is_dir():
                    raise ValueError(f"Missing producer archive: {stage} {k}")
                original = r.get("ilp_recovery") if stage == "stage1" else None
                binding = original["stage1_binding"] if original else r["binding"]
                _, info = checkpoint(folder, binding, stage, k, None)
                if original and info["checkpoint"] != original["receipts"][str(k)]:
                    raise ValueError("Original Stage 1 receipt differs")
                if info["checkpoint"] != entry["checkpoint"]:
                    raise ValueError("Producer receipt differs from frozen phase")
    return r


def write(name, data):
    run = n.records.start_run(
        commit=5,
        seeds={"split": 4305, "producers": 42, "bootstrap": 4302},
        notebook=n.NOTEBOOK,
        root=n.LOGS,
        code_files=FILES,
    )
    return n.records.write_run_record(run, name, data)


def decision(run_id, linked):
    black = verify(run_id, "blackbox")
    comp = linked["comparator"]
    return dict(
        blackbox=run_id,
        blackbox_sha=black["content_sha256"],
        policy=n.POLICY_SHA256,
        step5a=SHA,
        split=black["split"],
        **trigger(
            black["arms"], a.eligible(comp["results"]["glass"]), comp["rule_evidence"].startswith("Unresolved.")
        ),
    )


def stage1_receipts(folder, binding):
    return {str(k): checkpoint(folder, binding, "stage1", k, None)[1]["checkpoint"]
            for k in (*range(10), -1)}


def run(phase, parent=None, *, recover=False, check_only=False):
    if (recover and phase != "glass") or (check_only and not recover):
        raise ValueError("Recovery is scoped to the failed GLASS phase")
    started = time.perf_counter()
    five, policy, frozen, linked = sources()
    if phase == "decide":
        return write("simulated_decision", decision(parent, linked))
    authorization = None
    if phase == "glass":
        authorization = n.records.read_run_record(parent, "simulated_decision", root=n.LOGS)
        expected = decision(authorization["blackbox"], linked)
        if not expected["expand"] or any(authorization[k] != v for k, v in expected.items()):
            raise ValueError("GLASS expansion was not authorized by the frozen black-box results")
    side = inputs.draw_training_side(indices_only=True)
    if dict(side.identities) != frozen["population"] or not side.X_train.index.equals(side.y_train.index):
        raise ValueError("Training population differs from Step 2A")
    fit, held = split_rows(side.y_train)
    x, h, y = side.X_train.iloc[fit], side.X_train.iloc[held], side.y_train.iloc[fit]
    split = dict(
        fit_rows=len(fit),
        holdout_rows=len(held),
        fit_positive=int(y.sum()),
        holdout_positive=int(side.identities["train_positives"] - y.sum()),
        fit_ids=n._array_digest(pd.util.hash_pandas_object(x.index).to_numpy()),
        holdout_ids=n._array_digest(pd.util.hash_pandas_object(h.index).to_numpy()),
    )
    if (len(fit), len(held)) != (26360, 6590) or (authorization and split != authorization["split"]):
        raise ValueError("Registered holdout differs")
    path = inputs.BANK_ROOT / inputs.PINS_FILE
    inputs._equal(sha256_file(path), frozen["source_records"][inputs.PINS_FILE], "PR39 manifest")
    pins = inputs.read_pins(path)["files"]

    def pinned(role):
        v = pins[role]
        inputs._equal(v["sha256"], frozen["upstreams"][role]["sha256"], role)
        pin = inputs.PinnedFile(role, v["path"], v["sha256"], v["bytes"])
        return inputs.joblib.load(inputs.BytesIO(inputs.read_verified(pin, inputs.BANK_ROOT)))

    first = pinned("stage1." + ("mlp" if phase == "blackbox" else "lr"))
    inputs._equal(first["feature_names"], prod.LR_FEATURES, "Stage 1 features")
    inputs._equal(first["calibration_method"], "sigmoid", "Stage 1 calibration")
    base = clone(first["model" if phase == "blackbox" else "base_model"])
    del first
    role = "stage3." + ("xgb" if phase == "blackbox" else "ebm")
    artifact = pinned(role)
    params, features = artifact.best_params, artifact.features
    cfg = artifact.config
    calibration = dict(
        method=cfg["calibration_method"],
        ece_threshold=cfg["ece_threshold"],
        ece_bins=cfg["ece_bins"],
        force=cfg["force_calibration"],
        random_state=cfg["random_state"],
    )
    del artifact
    inputs._equal(features, frozen["upstreams"][role]["accepted_configuration"]["features"], "Stage 3 features")
    (meta,) = linked["nested"]["procedures"][phase]["meta_xgb_accepted_params"]["candidates"]
    p = prod.Producers(phase, x, y, h, base, params, features)
    p.calibration = calibration
    settings = dict(
        calibration=calibration,
        stage1=base.get_params(),
        stage2=prod.router.settings_record(p.settings),
        stage3=params,
        features=features,
        meta=meta,
    )
    code = n.records.code_identity(code_files=FILES)
    environment = provenance.environment()
    base_inputs = dict(split=split, X=n._digest(side.X_train), y=n._digest(side.y_train),
                       inputs=SHA, policy=n.POLICY_SHA256, env=environment)

    def bindings(arm, params, files, tree):
        return (
            digest(dict(base_inputs, phase=arm, settings=params, files=files, tree=tree)),
            digest(dict(split=split, code=tree, files=files, environment=environment)),
        )

    binding, common = bindings(phase, settings, code["files_sha256_lf"], code["tree"])
    cache_root = n.LOGS / "holdout_cache"
    folder = cache_root / phase
    reuse = None
    if recover:
        black = verify(authorization["blackbox"], "blackbox", archives=True)
        files, tree = recovery.original_code(inputs.BANK_ROOT, code, black)
        old_black, old_common = bindings("blackbox", black["settings"], files, tree)
        if ((old_black, old_common) != (black["binding"], black["common_binding"])
                or authorization["code"] != black["code"] or authorization["environment"] != environment):
            raise ValueError("Original execution or decision identity differs")
        old_glass, _ = bindings("glass", settings, files, tree)
        reuse = dict(blackbox=authorization["blackbox"], blackbox_sha=black["content_sha256"],
                     decision=parent, decision_sha=authorization["content_sha256"],
                     original_common=old_common, corrected_common=common,
                     failure=recovery.failure_identity(cache_root), stage1_binding=old_glass,
                     receipts=stage1_receipts(folder, old_glass))
        binding = digest(dict(corrected=binding, authorized_reuse=reuse))
        receipt = dict(reuse, corrected_binding=binding, corrected_code=code, environment=environment)
        if check_only:
            return json.dumps(receipt, indent=2)
        # Preserve the bridge even if a later solver stops. Never edit historical receipts.
        path = folder / "ilp_recovery.json"
        if not path.exists():
            with path.open("x", encoding="utf8") as stream:
                json.dump(receipt, stream, indent=2)
        elif json.loads(path.read_text(encoding="utf8")) != receipt:
            raise ValueError("Recorded recovery authorization differs")
    elif authorization and verify(authorization["blackbox"], "blackbox")["common_binding"] != common:
        raise ValueError("Black-box and GLASS execution identities differ")
    p.audit_root, p.audit_binding = folder / "solver_audits", binding

    def cache(stage, k, fn):
        if reuse and stage == "stage1":
            arrays, info = checkpoint(folder, reuse["stage1_binding"], stage, k, None)
            if info["checkpoint"] != reuse["receipts"][str(k)]:
                raise ValueError("Stage 1 receipt changed after preflight")
        else:
            arrays, info = checkpoint(folder, binding, stage, k, fn)
        if stage == "meta" and k == 0:
            print(
                f"Meta first producer: {info['fit_rows']} rows, {info['seconds']:.2f}s; remaining four outer fits ~{4 * info['seconds']:.2f}s.",
                flush=True,
            )
        return arrays, info

    print(f"{phase} approved fit budget: {five['budget']['base_fits']}; {five['budget']['estimate']}", flush=True)
    with threadpool_limits(limits=1):
        train, holds, ledger, rules = assemble(p, cache)
        p.__dict__.pop("block", None)  # release inner feature matrices before Stage 4
        model, streams, locks, ledger["meta"] = ev.lock_scores(
            train, holds, meta, policy, phase == "glass", cache
        )
        rule_report = None
        if phase == "glass":
            rule_report, conf = prod.independent_evidence(rules, y, h.index)
            if conf is not None:
                f = holds["full"].copy()
                f.update(prod.routes(f.index, f.route, conf, f.router_base_rate))
                ablation = ev.formula(f, model, True)
            rule_report["ablation"] = (
                "locked full-refit diagnostic" if conf is not None else "inactive or unsupported"
            )
        labels = side.y_train.iloc[held]  # first consumption of simulated holdout outcomes after locking
        if rule_report:
            row = iter(rule_report["producers"])
            for _, selected, masks, validation in rules:
                for i, _ in enumerate(selected):
                    entry = next(row)
                    for label, mask in zip(("matched", "issued"), masks[2 * i : 2 * i + 2, len(validation) :]):
                        entry["holdout_" + label + "_positive"] = int(labels.to_numpy()[mask].sum())
        arms = ev.evaluate(streams, locks, y, labels, policy)
        if rule_report and conf is not None:
            d, c = ev.c.FORMULA.operate(ablation, None)
            cut = locks["logit_margin"]["cut"]
            rule_report["ablation_metrics"] = n._outcome(labels.to_numpy(), d, c >= cut, policy["cost"])
        profiles = {
            "fitting_oof": ev.profile(train, y, model, policy),
            **{m: ev.profile(f, labels, model, policy) for m, f in holds.items()},
        }
        threads = [{k: i[k] for k in ("internal_api", "num_threads")} for i in threadpool_info()]
    full, avg = holds["full"], holds["average"]
    time_ledger = {
        stage: {
            "first_producer_seconds": entries[0]["seconds"],
            "first_fit_rows": entries[0]["fit_rows"],
            "full_refit_seconds": entries[-1]["seconds"],
            "full_fit_rows": entries[-1]["fit_rows"],
        }
        for stage, entries in ledger.items()
    }
    timing = {stage: sum(v["seconds"] for v in entries) for stage, entries in ledger.items()}
    extrapolated = {
        stage: sum(v["seconds"] * 32950 / 26360 for v in entries) for stage, entries in ledger.items()
    }
    path = write(
        "holdout_" + phase,
        dict(
            **({"ilp_recovery": reuse} if reuse else {}),
            **({"router_fit_accounting": dict(
                registered_primary=6, original_failed_primary=int(bool(reuse)),
                replacement_primary=int(bool(reuse)),  # included in the six successful primary fits
                additional_m3=sum(i["m3_router_fits"] for i in ledger["router"]),
                total_attempted=6 + int(bool(reuse)) + sum(i["m3_router_fits"] for i in ledger["router"]),
            )} if phase == "glass" else {}),
            arm=phase,
            split=split,
            policy=n.POLICY_SHA256,
            step5a=SHA,
            sources={k: five[k] for k in ("nested", "linear", "diagnostic", "comparator")},
            binding=binding,
            common_binding=common,
            settings=settings,
            folds={k: n._array_digest(v) for k, v in p.folds.items()},
            formula=model,
            arms=arms,
            transfer_flags=trigger(arms, False, False)["numeric"],
            locked_prediction_digests={
                arm: {mode: n._digest(f) for mode, f in conditions.items()} for arm, conditions in streams.items()
            },
            profiles=profiles,
            route_transitions=pd.crosstab(full.route, avg.route).to_dict(),
            rule_evidence=rule_report,
            stage_deltas={
                col: ev.summary(full[col] - avg[col])
                for col in ("stage1_proba", "router_score", "stage3_raw", "stage3_proba")
            },
            voter_changes={
                col: int(((full[col] >= t) != (avg[col] >= t)).sum())
                for col, t in [("stage1_proba", model["lr"]), ("stage3_proba", model["ebm"])]
            },
            decision_run=parent,
            decision_sha=authorization["content_sha256"] if authorization else None,
            designs=dict(
                stage1_encoder_scaler_pairs=sum(v["base_fits"] for v in ledger["stage1"]),
                stage1_sigmoid_maps=sum(v["maps"] for v in ledger["stage1"]),
                stage3_feature_cleaner_pairs=31,
                stage3_gated_maps=sum(v["maps"] for v in ledger["stage3"]),
                stage2_outer_full_designs=6,
                formula_inner_fits=3,
                formula_full_fits=1,
                formula_weights_shared=True,
            ),
            ledger=ledger,
            producer_seconds=timing,
            timing_by_role=time_ledger,
            planning=dict(
                pr45_full_size_linear_seconds=extrapolated,
                pr49_seconds_per_same_size_split=sum(timing.values()),
                basis="Serial producer work; PR45 linear row-count extrapolation only. Solver and optimization costs may scale nonlinearly. Excludes search, archive IO, reporting and PR49 temporal redesign.",
            ),
            bottleneck=max(timing, key=timing.get),
            stage4_frame_bytes={
                key: int(frame.memory_usage(index=True, deep=True).sum())
                for key, frame in {"fitting": train, **holds}.items()
            },
            wall_seconds=time.perf_counter() - started,
            thread_pools=threads,
            interpretation="Raw Meta, paired differences first minus second. On one simulated holdout the paired loss "
            "bootstrap preserves true class sizes. Stage 3 balances fitting-partition classes once. Stage 1 MLP early "
            "stopping shares encoder fitting. Historical features and parameter choices precede this split. "
            "Neither causal attribution beyond these locked interventions nor temporal validation follows.",
            handoff="PR45 must validate retained arms on regenerated inputs and resolve unsupported rule transfer. "
            "PR49 repeated splits multiply producer work. Single-split timings and transfer results are conditional.",
            reproduce="python -m feature_research.stage4_holdout " + phase + (" " + parent if parent else ""),
        ),
    )
    verify(path.parent.name, phase)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["blackbox", "decide", "glass", "verify"])
    parser.add_argument("parent", nargs="?")
    parser.add_argument("--arm", choices=["blackbox", "glass"])
    parser.add_argument("--archives", action="store_true")
    parser.add_argument("--recover-router0", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if (args.recover_router0 and args.phase != "glass") or (args.check_only and not args.recover_router0):
        parser.error("use glass DECISION --recover-router0 [--check-only]")
    if args.phase == "verify":
        if not args.arm or not args.parent:
            parser.error("verify requires a run ID and --arm")
        print(verify(args.parent, args.arm, args.archives)["content_sha256"])
    else:
        if args.phase in ("glass", "decide") and not args.parent:
            parser.error("supply the frozen parent run ID")
        try:
            print(run(args.phase, args.parent, recover=args.recover_router0, check_only=args.check_only))
        except Exception:
            failure = n.LOGS / "holdout_cache" / "failures"
            failure.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", prefix=args.phase + "_", suffix=".txt", dir=failure, encoding="utf8", delete=False
            ) as log:
                log.write(traceback.format_exc())
            print(f"Failure evidence: {log.name}", flush=True)
            raise
