"""PR 43 Step 4B. Fold-local secondary comparisons on fixed upstream evidence."""
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from feature_research import linear_arbiter as linear, stage4_nested as n, voter_diagnostics as diag
from shared.stage3.calibration import Stage3Calibrator

LINEAR_RUN = "c4_20261125_181030"
LINEAR_SHA = "e0a464b259da2aace61471b6ab004be733016e35524d3adb7df25d035f959815"
BASE = "formula_reference"
MANDATORY = {BASE: (.7, .5), "logit_margin": (.7, .5), "recall60": (.6, .5), "recall80": (.8, .5),
             "blend25": (.7, .25), "blend75": (.7, .75)}
CORNERS = {f"joint_{r}_{a}": (r / 100, a / 100) for r, a in product((60, 80), (25, 75))}
FORMULA = n.Formula()
META = linear.OPERATING


def corners_enabled(loss):
    return all(any(loss[k] <= loss[BASE] - .005 for k in group)
               for group in (("recall60", "recall80"), ("blend25", "blend75")))


def normalized(rows, model, raw):
    """Signed terms in LR, EBM, router order; confidence ties vote negative."""
    x, route = linear.evidence(rows, [model["lr"], model["ebm"]])
    x[:, 2] = np.where(route == 0, np.minimum(x[:, 2], 0), np.maximum(x[:, 2], 0))
    w = np.array([model["weights"][k] for k in ("lr", "ebm", "glass")])
    terms = x * w / (w[:2].sum() + (route != 2) * w[2])[:, None]
    pos, neg = np.maximum(terms, 0).sum(axis=1), -np.minimum(terms, 0).sum(axis=1)
    return raw.assign(decision=(pos > neg).astype(int), confidence=np.maximum(pos, neg)), terms


def formula_variants(train, evaluation, ids, policy, accepted_model):
    configs = MANDATORY | CORNERS
    cache = {}

    def score(name, fit, target, j):
        r, alpha = configs[name]
        key = j, r, alpha
        if key not in cache:
            model = accepted_model if j == 3 and (r, alpha) == configs[BASE] else FORMULA.fit(
                {"target_recall": r, "alpha": alpha}, fit)
            cache[key] = model, FORMULA.score(model, target)
        model, raw = cache[key]
        return normalized(target, model, raw)[0] if name == "logit_margin" else raw

    locks = {}

    def screen(options):
        for name in options:
            parts = [score(name, train[ids != j], train[ids == j], j) for j in range(3)]
            scores = pd.concat(parts, verify_integrity=True).loc[train.index]
            locks[name] = n.lock("registered", FORMULA, scores, train.y.to_numpy(), policy)
            if locks[name] is None:
                raise ValueError(f"Infeasible registered cut: {name}")
    screen(MANDATORY)
    enabled = corners_enabled({k: v["inner"]["loss"] for k, v in locks.items()})
    if enabled:
        screen(CORNERS)
    winner = min(locks, key=lambda k: locks[k]["inner"]["loss"])
    arms = {k: (FORMULA, score(k, train, evaluation, 3), lock) for k, lock in locks.items()}
    arms["formula_selected"] = arms[winner]
    return arms, dict(corners=enabled, selected=winner, configurations={
        k: cache[(3, *configs[k])][0] for k in locks if k != "logit_margin"})


def winners(train, k, outer, upstream, ref, arrays):
    raw = arrays[f"{upstream}.meta_xgb.inner.ranking"][:, k, outer != k].astype(float)
    candidates = ref["meta_xgb"]["candidates"]
    if len(candidates) != 12 or len({tuple(sorted(p.items())) for p in candidates}) != 12:
        raise ValueError("Expected twelve distinct registered proposals")
    selection = ref["meta_xgb"]["selection"][k]
    losses = [None if str(i) in selection["failures"] else log_loss(train.y, p, labels=[0, 1])
              for i, p in enumerate(raw)]
    f2 = selection["chosen"]
    ll = min((i for i, loss in enumerate(losses) if loss is not None), key=losses.__getitem__)
    fixed = ref["meta_xgb_accepted_params"]["candidates"][0]
    sources = {i: ("meta_xgb" if i == f2 else "meta_xgb_accepted_params" if candidates[i] == fixed else None)
               for i in dict.fromkeys((f2, ll))}
    return raw, losses, f2, ll, sources


def calibrated(train, ids, raw, params, arm):
    plan = n.FoldPlan(ids, 3, 42, True)
    sub = np.full((3, len(train)), np.nan)
    fits = 0
    for j, (fit, _) in enumerate(plan):
        for a, b in n.FoldPlan.build(train.y.iloc[fit], 3, 43):
            model = arm.fit(params, train.iloc[fit[a]])
            fits += 1
            sub[j, fit[b]] = arm.score(model, train.iloc[fit[b]]).ranking
    mapper = Stage3Calibrator(method="isotonic").fit(raw, train.y, plan, sub)
    return mapper, dict(mapper.report_.to_dict(), subfits=fits)


def meta_variants(train, evaluation, ids, k, outer, upstream, ref, arrays, policy, plan, cache):
    raw, losses, f2, ll, sources = plan
    candidates = ref["meta_xgb"]["candidates"]
    mappers, details, locks = {}, {}, {}
    for winner, source in sources.items():
        mapper, info = calibrated(train, ids, raw[winner], candidates[winner], META)
        mappers[winner] = mapper
        if any(info["fold_applied"]):
            cache[f"{upstream}.{k}.{winner}.inner_calibrated"] = n._compact(mapper.oof_calibrated_)
        details[str(winner)] = dict(source=source or "outer refit", calibration=info)
        for space, p in (("raw", raw[winner]), ("calibrated", mapper.oof_calibrated_)):
            lock = n.lock("registered", META, pd.DataFrame({"ranking": p}), train.y.to_numpy(), policy)
            if lock is None:
                raise ValueError(f"Infeasible registered cut for proposal {winner} in {space}")
            locks[winner, space] = lock
    variants = {f"meta_{objective}_{space}": (i, space)
                for objective, i in (("f2", f2), ("logloss", ll)) for space in ("raw", "calibrated")}
    chosen = min(variants, key=lambda key: (locks[variants[key]]["inner"]["loss"], key.endswith("calibrated")))
    streams, refits = {}, 0
    for winner, source in sources.items():
        key = f"{upstream}.{k}.{winner}"
        if source:
            outside = arrays[f"{upstream}.{source}.outer.ranking"][outer == k].astype(float)
        else:
            model = META.fit(candidates[winner], train)
            outside = META.score(model, evaluation).ranking.to_numpy()
            refits += 1
            cache[key + ".outer_raw"] = n._compact(outside)
        corrected = mappers[winner].transform(outside)
        if mappers[winner].applied_:
            cache[key + ".outer_calibrated"] = n._compact(corrected)
        for space, p in (("raw", outside), ("calibrated", corrected)):
            streams[winner, space] = pd.DataFrame({"ranking": p}, index=evaluation.index)
    arms = {name: (META, streams[pair], locks[pair]) for name, pair in variants.items()}
    arms["meta_selected"] = arms[chosen]
    return arms, dict(selected=chosen, logloss=losses, f2_winner=f2, logloss_winner=ll, winners=details,
                      outer_refits=refits, calibration_subfits=sum(detail["calibration"]["subfits"] for detail in details.values()))


def replay_linear(rows, fold):
    cfg = fold["configurations"][fold["selected"]]
    z = linear.logits(np.asarray(cfg["beta"]), linear.evidence(rows, cfg["thresholds"]))
    p = expit(cfg["correction_slope"] * z + cfg["correction_intercept"]) if cfg["sigmoid"] else expit(z)
    return META, pd.DataFrame({"ranking": p}, index=rows.index), cfg["locked"]


def evaluate(table, u, outer, inner, ref, arrays, accepted, policy, plans, cache):
    stored = ref["procedures"][u]
    bases = {k: linear.reference_frame(k, table, outer, ref, arrays, policy, u)
             for k in ("glass_formula", "meta_xgb", "meta_xgb_accepted_params")}
    pieces, measurements, design, matched = {}, {}, [], []
    curve_samples = {}
    coverage_axis = np.linspace(.5, 1, 21)
    for k, ids in inner.items():
        train, rows = table[outer != k], table[outer == k]
        features = rows.drop(columns="y")
        arms, formula = formula_variants(train, features, ids, policy, stored["glass_formula"]["configuration"][k])
        meta, tuning = meta_variants(train, features, ids, k, outer, u, stored, arrays, policy, plans[k], cache)
        arms.update(meta)
        fold = accepted["selection"][k]
        if fold["fold"] != k:
            raise ValueError("Step 4A fold mismatch")
        arms["linear"] = replay_linear(features, fold)
        curves = {}
        y = rows.y.to_numpy()
        for name, (arm, scores, lock) in arms.items():
            if not rows.index.equals(scores.index):
                raise ValueError("Comparator row alignment mismatch")
            row, d, c, keep = n._fold_row(arm, y, scores, lock, policy)
            frame = scores.assign(outer_fold=k, decision=d, confidence=c, kept=keep,
                                  row_loss=linear.realized(y, d, keep, policy["cost"]))
            pieces.setdefault(name, []).append(frame)
            measurements.setdefault(name, []).append(dict(outer_fold=k, rows=len(rows), **row))
            curves[name] = n.risk_curve(y, d, c, policy["cost"])
        fixed = bases["meta_xgb_accepted_params"][0].loc[rows.index]
        curves["meta_accepted_raw"] = n.risk_curve(y, fixed.decision, fixed.confidence, policy["cost"])
        # Display minimum attainable coverage risk at a compact grid; exact AURC uses every tie group.
        for name, curve in curves.items():
            values = curve.risk.to_numpy()[curve.coverage.to_numpy().searchsorted(coverage_axis)]
            curve_samples.setdefault(name, []).append((len(rows), values))
        for g in policy["coverage"]["grid"]:
            count, risks, brackets = n.matched_risks(curves, g)
            matched.append(dict(fold=k, coverage=g, count=count, risks=risks, brackets=brackets))
        design.append(dict(fold=k, formula=formula, meta=tuning))
    reports, contrasts, selected = {}, {}, {}
    for name, parts in pieces.items():
        folds = pd.DataFrame(measurements[name])
        # Pooled reports retain the full confusion and sensitivity accounting; keep fold evidence compact.
        columns = ["outer_fold", "rows", "threshold", "cut", "inner_loss", "outer_loss", "loss_optimism",
                   "outer_coverage", "outer_positive_retention", "outer_negative_retention", "partial_aurc"]
        fold_summary = folds[columns].to_dict("list")
        if len(parts) != len(inner):
            folds_base = pd.DataFrame(measurements[BASE]).set_index("outer_fold")
            fold_summary["delta_loss_vs_reference"] = (folds.set_index("outer_fold").outer_loss -
                folds_base.loc[folds.outer_fold, "outer_loss"]).tolist()
            reports[name] = {"eligible_folds_only": fold_summary}
            continue
        frame = pd.concat(parts, verify_integrity=True).loc[table.index]
        probability = name.startswith("meta") or name == "linear"
        pooled = n._pooled(folds, table.y.to_numpy(), frame.decision.to_numpy(), frame.ranking.to_numpy(),
                           frame.kept.to_numpy(), probability, policy)
        reports[name] = dict(pooled=pooled, folds=fold_summary)
        if name.endswith("_selected"):
            selected[name] = frame, pooled
        if name == "linear":
            expected = accepted["procedures"]["selected"]["pooled"]
            for section in ("outer", "all_rows"):
                for key, v in expected[section].items():
                    np.testing.assert_allclose(pooled[section][key], v, atol=1e-12, rtol=0)
            continue
        base = "meta_xgb" if probability else "glass_formula"
        if name in (BASE, "meta_f2_raw"):
            accepted_cuts = stored[base]["rules"]["registered"]["folds"]["cut"]
            np.testing.assert_allclose(folds.cut, accepted_cuts, atol=0, rtol=0)
            for col in ("decision", "kept", "ranking", "confidence"):
                np.testing.assert_array_equal(frame[col], bases[base][0][col])
            continue
        contrasts[name] = linear.compare(table, frame, pooled, folds, bases[base], policy,
                                         primary=False, candidate_name=name)
    frame, pooled = selected["formula_selected"]
    contrasts["selected_formula_vs_selected_meta"] = linear.compare(
        table, frame, pooled, pd.DataFrame(measurements["formula_selected"]), selected["meta_selected"],
        policy, primary=False, candidate_name="formula_selected")
    for comparison in contrasts.values():
        comparison["secondary_result"] = (comparison["loss_verdict"] if all(comparison["safeguards"].values())
                                          else "reject: safeguard failure")
    reports["meta_accepted_raw"] = dict(pooled=bases["meta_xgb_accepted_params"][1])
    display_curves = {name: np.average([v for _, v in parts], axis=0, weights=[nrows for nrows, _ in parts]).tolist()
                      for name, parts in curve_samples.items() if len(parts) == len(inner)}
    return dict(arms=reports, secondary=contrasts, design=design, matched=matched,
                coverage_axis=coverage_axis.tolist(), minimum_coverage_risk=display_curves,
                linear_disposition=accepted["disposition"])


def run():
    start = perf_counter()
    accepted = n.records.read_run_record(LINEAR_RUN, "linear_arbiter", root=n.LOGS)
    if accepted["content_sha256"] != LINEAR_SHA:
        raise ValueError("Accepted Step 4A content mismatch")
    tables, _, policy, identity, ref, arrays, outer, inner = diag.verified_tables()
    diagnostics = n.records.read_run_record(linear.DIAGNOSTIC_RUN, "voter_diagnostics", root=n.LOGS)
    linear.eligibility(diagnostics, ref, identity)
    if (accepted["inputs"] != identity or accepted["diagnostic"] != linear.DIAGNOSTIC_SHA or
            accepted["nested_reference"] != diagnostics["nested_reference"]):
        raise ValueError("Accepted records refer to different evidence")
    plans = {u: {k: winners(t[outer != k], k, outer, u, ref["procedures"][u], arrays) for k in inner}
             for u, t in tables.items()}
    sources = [plan[-1] for arm in plans.values() for plan in arm.values()]
    budget = dict(outer_refits=sum(v is None for src in sources for v in src.values()),
                  calibration_subfits=9 * sum(map(len, sources)), upstream_fits=0, linear_fits=0)
    print("Additional fit budget", budget, flush=True)
    cache = {}
    with threadpool_limits(limits=1):
        results = {u: evaluate(t, u, outer, inner, ref, arrays, accepted["results"][u], policy, plans[u], cache)
                   for u, t in tables.items()}
    run = n.records.start_run(commit=4, seeds={"inner": 42, "subfold": 43, "bootstrap": 4302}, root=n.LOGS,
        notebook=n.NOTEBOOK, code_files=(*n.CODE_FILES, "feature_research/stage4_comparators.py",
        "feature_research/linear_arbiter.py", "shared/stage3/calibration.py", "feature_research/voter_diagnostics.py"))
    actual = {key: sum(f["meta"][key] for result in results.values() for f in result["design"])
              for key in ("outer_refits", "calibration_subfits")}
    if any(actual[k] != budget[k] for k in actual):
        raise AssertionError("Observed Meta fits differ from verified winner budget")
    index = next(iter(tables.values())).index
    manifest = dict(file=n.PREDICTIONS, rows_sha256=n._array_digest(pd.util.hash_pandas_object(index).to_numpy()),
                    arrays={k: dict(shape=list(a.shape), dtype=a.dtype.str, sha256=n._array_digest(a)) for k, a in cache.items()},
                    layout="upstream.outerFold.proposal.stream. Inner rows in Step 2C outer-training order, outer rows "
                           "in outer-evaluation order. Identity gates reuse raw streams from the linked store.")
    evidence = dict(predictions=manifest, actual_fits=actual, inputs=identity, nested=accepted["nested_reference"],
        linear=dict(run_id=LINEAR_RUN, content_sha256=LINEAR_SHA), diagnostic=linear.DIAGNOSTIC_SHA,
        results=results, planned_fits=budget, rule_evidence=diagnostics["rule_evidence"],
        rule_handoff="Route-class support and outcomes are in Commit 3. Per-rule membership, independent "
                     "counts and rates cannot be reconstructed. Leave unseen rules unsupported. If required "
                     "in Commit 5, retain ordered rule IDs, producing folds, matching rows and independent "
                     "counts. A fresh five-fold router cross-fit requires at least five producer fits; estimate "
                     "runtime from current router fit times. PR 38 handoff records 1192.6 s for a historical five-fold GLASS run; corrected inner work and solver behavior can change this substantially.",
        seconds=perf_counter() - start, interpretation="Secondary descriptive 95% paired intervals; "
        "the four primary Step 4A contrasts and their registered decisions retain their original identities.",
        limitations=n.LIMITATIONS[0] + " " + n.LIMITATIONS[-1], reproduce="python -m feature_research.stage4_comparators")
    support = tables["glass"].groupby(["stage2_fold", "route"]).y.agg(rows="size", positive="sum")
    evidence["glass_router_support"] = support.assign(negative=support.rows - support.positive).reset_index().to_dict("list")
    return n.write_predictions(run, "stage4_comparators", evidence, cache)[0]


def verify(run_id):
    """Verify the compact comparator archive and registered population/budget accounting without fitting."""
    record, _ = n.load_predictions(run_id, record_name="stage4_comparators")
    policy = n.load_policy()
    pop = policy["population"]
    if (record["linear"] != {"run_id": LINEAR_RUN, "content_sha256": LINEAR_SHA}
            or record["diagnostic"] != linear.DIAGNOSTIC_SHA
            or record["inputs"]["policy"]["canonical_sha256"] != n.POLICY_SHA256):
        raise ValueError("Comparator sources differ from the accepted records")
    totals = {"outer_refits": 0, "calibration_subfits": 0}
    if set(record["results"]) != set(n.UPSTREAMS):
        raise ValueError("Missing upstream")
    for upstream, result in record["results"].items():
        if result["linear_disposition"] != "reject" or len(result["matched"]) != 20:
            raise ValueError(f"{upstream}: incomplete comparisons or altered linear disposition")
        for arm in result["arms"].values():
            if "pooled" in arm:
                m = arm["pooled"]["outer"]
                retained = sum(m[k] for k in ("tp", "fp", "tn", "fn"))
                if (retained != m["kept"] or retained + m["deferred_positive"] + m["deferred_negative"] != pop["train_rows"]
                        or m["tp"] + m["fn"] + m["deferred_positive"] != pop["train_positives"]):
                    raise ValueError(f"{upstream}: pooled population mismatch")
        for comp in result["secondary"].values():
            ci = comp["interval"]
            if (ci["level"], ci["replicates"], ci["seed"]) != (.95, 4000, 4302):
                raise ValueError("Secondary interval convention mismatch")
        if sorted(d["fold"] for d in result["design"]) != list(range(5)):
            raise ValueError("Missing fold")
        for d in result["design"]:
            for k in totals:
                totals[k] += d["meta"][k]
            config = d["formula"]["configurations"]
            expected = (set(MANDATORY) - {"logit_margin"}) | (set(CORNERS) if d["formula"]["corners"] else set())
            if set(config) != expected:
                raise ValueError("Unregistered formula budget")
    if totals != record["actual_fits"] or any(record["planned_fits"][k] != v for k, v in totals.items()):
        raise ValueError("Comparator fit ledger mismatch")
    print(run_id, record["content_sha256"], totals)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--verify", metavar="RUN_ID")
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        print(run())
