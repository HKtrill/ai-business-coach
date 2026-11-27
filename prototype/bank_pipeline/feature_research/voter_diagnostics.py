"""PR 43 Commit 3. Conditional diagnostics on accepted training-side OOF streams."""
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score, roc_auc_score

from feature_research.metrics import wilson_ci
from glass_pipeline.glass_arbiter.arbiter import meta_arbiter
from glass_pipeline.glass_arbiter.router_evidence import rule_positive_rate
from shared.metrics import calculate_ece
from shared.stage2.diagnostics.operating_points import matched_rate_cut

NESTED = "c2_20261125_142638"
STATES = ("pass1", "pass2", "uncertain")
VOTERS = ("lr", "ebm", "router")


def vote_counts(frame):
    """Pair denominators exclude absent votes. A lone dissent needs three cast votes."""
    joint = frame.groupby(["route", "lr", "ebm"]).y.agg(n="size", positive="sum").reset_index()
    joint["negative"] = joint.n - joint.positive
    joint["positive_rate"] = joint.positive / joint.n
    joint["positive_ci95"] = [wilson_ci(p, n) for p, n in zip(joint.positive, joint.n)]
    joint["lr_wrong"] = np.where(joint.lr, joint.negative, joint.positive)
    joint["ebm_wrong"] = np.where(joint.ebm, joint.negative, joint.positive)
    pairs, unique = [], []
    for route, group in [("all", frame), *frame.groupby("route")]:
        for label, rows in [("all", group), *group.groupby("y")]:
            ids = dict(route=route, label=label)
            votes = rows[list(VOTERS)]
            active = votes.notna()
            correct = votes.eq(rows.y, axis=0) & active
            for a, b in combinations(VOTERS, 2):
                eligible = active[a] & active[b]
                aa, bb = correct.loc[eligible, a], correct.loc[eligible, b]
                n = len(aa)
                corr = aa.astype(float).corr(bb.astype(float)) if aa.nunique()*bb.nunique() == 4 else None
                pairs.append(dict(**ids, pair=f"{a}/{b}", n=n, agree=int((aa == bb).sum()),
                                  disagree=int((aa != bb).sum()), double_fault=int((~aa & ~bb).sum()),
                                  only_first=int((aa & ~bb).sum()), only_second=int((bb & ~aa).sum()),
                                  error_corr=corr))
            for v in VOTERS:
                sole = correct[v] & correct.sum(axis=1).eq(1)
                three = active.all(axis=1)
                unique.append(dict(**ids, voter=v, active=int(active[v].sum()), sole_correct=int(sole.sum()),
                                   three_votes=int(three.sum()), correct_lone_dissent=int((sole & three).sum())))
    return dict(joint=joint.to_dict("records"), pairs=pairs, unique=unique)


def flag_budgets(rows, frame):
    """Descriptive label-free top-k. Two all-row voters; three voters on routed rows."""
    budgets = {}
    for key, population in (("all_at_lr_count", np.ones(len(rows), bool)),
                            ("routed_at_pass2_count", rows.route.ne("uncertain").to_numpy())):
        target = int(frame.lr.sum() if key.startswith("all") else rows.route.eq("pass2").sum())
        matched = frame.loc[population].copy()
        brackets = {}
        for v, col in (("lr", "stage1_proba"), ("ebm", "stage3_proba")):
            cut = matched_rate_cut(rows[col], target_count=target, population_mask=population)
            brackets[v] = dict(cut=cut.cut_value, below=cut.strict_count, inclusive=cut.inclusive_count,
                               tie_size=cut.tie_block, exact=cut.exact)
            matched[v] = cut.inclusive[population]
        if key.startswith("all"):
            matched["router"] = np.nan  # router has no full-population threshold at this budget
        budgets[key] = dict(target=target, eligible=len(matched), brackets=brackets,
                            counts=vote_counts(matched) if all(c["exact"] for c in brackets.values()) else None)
    return budgets


def router_tradeoffs(f, cost):
    leaks = f[f.route.eq("pass1") & f.y.eq(1)]
    flags = dict(lr=leaks.lr, ebm=leaks.ebm, both=leaks.lr & leaks.ebm,
                 either=leaks.lr | leaks.ebm, neither=~(leaks.lr | leaks.ebm))
    rescued = {k: dict(n=int(v.sum()), rate=v.mean() if len(v) else None,
                       ci95=wilson_ci(v.sum(), len(v)) if len(v) else None) for k, v in flags.items()}
    veto = f[f.route.eq("pass2") & ~f.lr & ~f.ebm]
    tp, fp = int(veto.y.sum()), int((veto.y == 0).sum())
    return dict(pass1_leaked_positive=len(leaks), flags=rescued, pass2_rows=int(f.route.eq("pass2").sum()),
                veto_rows=len(veto), removed_fp=fp, lost_tp=tp,
                veto_positive_rate=tp/len(veto) if len(veto) else None,
                signed_cost_change_per_population=(cost["fn"]*tp-cost["fp"]*fp)/len(f))


def calibration(y, p):
    """Unpenalized joint logistic calibration; conditional inverse-information Wald intervals."""
    y, p = np.asarray(y), np.asarray(p)
    pos = int(y.sum())
    result = dict(n=len(y), positive=pos, negative=len(y)-pos, slope=None)
    if min(pos, len(y)-pos) < 20:
        return dict(result, status="insufficient class support")
    x = logit(np.clip(p, 1e-6, 1-1e-6))
    if np.ptp(x) == 0:
        return dict(result, status="constant score")
    if max(x[y == 0]) <= min(x[y == 1]) or max(x[y == 1]) <= min(x[y == 0]):
        return dict(result, status="separation")
    design = np.column_stack([np.ones(len(y)), x])
    def objective(beta):
        z = design @ beta
        return np.mean(np.logaddexp(0, z)-y*z), design.T @ (expit(z)-y) / len(y)
    fit = minimize(objective, [0., 1.], method="BFGS", jac=True, options={"gtol": 1e-8})
    q = expit(design @ fit.x)
    info = design.T @ ((q*(1-q))[:, None]*design)
    if not fit.success or np.linalg.cond(info) > 1e10:
        return dict(result, status="unstable fit")
    half = 1.95996398454*np.sqrt(np.diag(np.linalg.inv(info)))
    return dict(result, status="estimated", intercept=fit.x[0], slope=fit.x[1],
                intercept_ci95=[fit.x[0]-half[0], fit.x[0]+half[0]],
                slope_ci95=[fit.x[1]-half[1], fit.x[1]+half[1]])


def state_calibration(rows):
    route = rows.route
    rates = rule_positive_rate(rows.router_score, route.eq("pass1"), route.eq("pass2"), rows.router_base_rate)
    estimates = {}
    for state in STATES:
        mask = route.eq(state).to_numpy()
        scores = {"stage1": rows.stage1_proba, "stage3": rows.stage3_proba}
        if state != "uncertain":
            scores["router_rate"] = rates
        for name, values in scores.items():
            estimates[name, state] = calibration(rows.y[mask], np.asarray(values)[mask])
    return estimates


def slope_gate(pooled, folds):
    """Magnitude on pooled outer-training OOF rows; same sign in >=2 supported inner folds."""
    rows = []
    for voter in ("stage1", "stage3", "router_rate"):
        for a, b in combinations(STATES, 2):
            if (voter, a) not in pooled or (voter, b) not in pooled:
                continue
            def delta(estimates):
                x, y = (estimates[voter, state]["slope"] for state in (a, b))
                return None if x is None or y is None else x-y
            d, inner = delta(pooled), [delta(f) for f in folds]
            same = sum(v is not None and d is not None and d*v > 0 for v in inner)
            status = "not met"
            if d is None or (abs(d) >= .5 and same < 2 <= same+inner.count(None)):
                status = "unresolved"
            elif abs(d) >= .5 and same >= 2:
                status = "met"
            rows.append(dict(voter=voter, first=a, second=b, delta=d, inner_delta=inner, status=status))
    statuses = {r["status"] for r in rows}
    return dict(status=next(s for s in ("met", "unresolved", "not met") if s in statuses), contrasts=rows)


def route_trigger(tables, outer, inner):
    out = []
    for upstream, table in tables.items():
        for k, ids in inner.items():
            training = table.iloc[np.flatnonzero(outer != k)]
            pooled = state_calibration(training)
            parts = [state_calibration(training.iloc[np.flatnonzero(ids == j)]) for j in range(3)]
            # Store sufficient subgroup evidence once. Each estimate includes both class counts.
            estimates = {f"{v}/{s}": [p[v,s] for p in [pooled, *parts]] for v,s in pooled}
            out.append(dict(upstream=upstream, outer_fold=k, estimates=estimates, **slope_gate(pooled, parts)))
    return out


def decision_structure(rows, config, cost):
    cuts, weights = config["thresholds"], config["weights"]
    cut = config["abstention"]["min_weighted_confidence"]
    trace = np.empty((len(rows), 3, 2))
    pass1, pass2 = (rows.route.eq(s).to_numpy() for s in STATES[:2])
    lr, ebm = (rows[c].to_numpy() for c in ("stage1_proba", "stage3_proba"))
    d, rank, _, conf = meta_arbiter(lr, ebm, rows.router_score.to_numpy(), pass1, pass2,
                                    cuts["lr"], cuts["ebm"], weights, allow_abstain=False,
                                    return_confidence=True, glass_base_rate=rows.router_base_rate.to_numpy(),
                                    contribution_trace=trace)
    signed = trace[:, :, 0]
    pos, neg = np.maximum(signed, 0).sum(axis=1), -np.minimum(signed, 0).sum(axis=1)
    np.testing.assert_allclose(np.maximum(pos, neg), conf, atol=1e-12, rtol=0)
    np.testing.assert_allclose(trace[:, :, 1].sum(axis=1), rank, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(pos > neg, d)
    kept = conf >= cut
    f = rows[["route", "y"]].assign(lr=lr >= cuts["lr"], ebm=ebm >= cuts["ebm"],
                                     router=np.where(pass1 | pass2, pass2.astype(float), np.nan))
    counts = vote_counts(f)
    annotated = f.assign(retained=kept, correct=d == f.y, positive_side=pos, negative_side=neg,
                         confidence=conf, ranking=rank)
    for j, voter in enumerate(VOTERS):
        annotated[voter+"_signed"] = signed[:, j]
    groups = ["route", "y", "lr", "ebm", "retained", "correct"]
    summary = annotated.groupby(groups)[["positive_side","negative_side","confidence","ranking",
                                          "lr_signed","ebm_signed","router_signed"]]
    composition = summary.mean().assign(n=summary.size()).reset_index()

    # Probability support implies t-p <= t. Holding votes and all positive evidence fixed,
    # push each negative LR/EBM probability to its mathematical endpoint p=0.
    denom = weights["lr"] + weights["ebm"] + (pass1 | pass2)*weights["glass"]
    cap = np.where(~f.lr, weights["lr"]*cuts["lr"]/denom, 0)
    cap += np.where(~f.ebm, weights["ebm"]*cuts["ebm"]/denom, 0)
    ceiling = cap + np.maximum(-signed[:, 2], 0)
    has_negative = ~f.lr | ~f.ebm
    margin = annotated.assign(negative_vote=has_negative, negative_ceiling=ceiling,
        remaining_headroom=ceiling-neg, forced_positive=has_negative & (pos > ceiling+1e-12),
        max_confidence_gain=np.maximum(pos, ceiling)-conf,
        cannot_reach_cut=has_negative & (np.maximum(pos, ceiling) < cut),
        lr_negative=np.maximum(-signed[:, 0], 0), ebm_negative=np.maximum(-signed[:, 1], 0))
    bound = margin.groupby(["route","y","correct"]).agg(n=("y","size"), affected=("negative_vote","sum"),
        fraction_negative=("negative_vote","mean"), forced_positive=("forced_positive","sum"),
        cannot_reach_cut=("cannot_reach_cut","sum"), mean_neg=("negative_side","mean"),
        mean_ceiling=("negative_ceiling","mean"), mean_headroom=("remaining_headroom","mean"),
        mean_max_confidence_gain=("max_confidence_gain","mean"), mean_lr_negative=("lr_negative","mean"),
        mean_ebm_negative=("ebm_negative","mean")).reset_index()
    return dict(accepted=config, contingency=counts, matched=flag_budgets(rows, f),
                router_tradeoffs=router_tradeoffs(f, cost), composition=composition.to_dict("records"),
                margin_bound=bound.to_dict("records"), retained=int(kept.sum()), rows=len(f),
                retained_f2=fbeta_score(f.y[kept], d[kept], beta=2, zero_division=0) if kept.any() else None)


def verified_tables():
    """Load the accepted frozen tables, reference store and common fold identities."""
    from feature_research import research_records as rr, stage4_nested as nested, training_inputs as inputs

    root = nested.LOGS
    frozen = rr.read_run_record(nested.load_policy()["input_run"], "frozen_inputs", root=root)
    policy, identity = nested.verified_inputs(frozen, root)
    train = inputs.draw_training_side(indices_only=True)
    reference, arrays = nested.load_predictions(NESTED, root=root, index=train.y_train.index)
    if any(reference["inputs"][k] != v for k, v in identity.items()):
        raise ValueError("Step 2C belongs to different input or policy records")
    pins_path = inputs.BANK_ROOT / inputs.PINS_FILE
    inputs._equal(inputs.boundary.file_sha256(pins_path), frozen["source_records"][inputs.PINS_FILE], "pins")
    pins = inputs.read_pins(pins_path)
    baseline = inputs.baseline_records.require_passed(pins["run_id"], "run_record")
    tables = {}
    for arm in inputs.ARMS:
        tables[arm], _ = inputs._arm_table(train, arm, pins["files"], baseline["identities"]["population"],
                                            baseline["evidence"])
        if nested._digest(tables[arm]) != reference["inputs"]["tables_sha256"][arm]:
            raise ValueError(f"{arm}: frozen table differs from Step 2C")
    outer, inner = nested.nested_folds(tables)
    np.testing.assert_array_equal(outer, arrays["outer_fold"])
    for k, ids in inner.items():
        np.testing.assert_array_equal(ids, arrays["inner_fold"][k, outer != k])
        np.testing.assert_array_equal(arrays["inner_fold"][k, outer == k], -1)
    return tables, frozen, policy, identity, reference, arrays, outer, inner


def run():
    from threadpoolctl import threadpool_limits
    from feature_research import research_records as rr, stage4_nested as nested, training_inputs as inputs

    start = perf_counter()
    root = nested.LOGS
    tables, frozen, policy, identity, reference, arrays, outer, inner = verified_tables()
    del arrays  # verified store; full-precision accepted summaries are already in its record
    glass = tables["glass"]
    config = frozen["historical_reference"]["glass_arbiter"]["configuration"]
    with threadpool_limits(limits=1):
        structure = decision_structure(glass, config, policy["cost"])
        whole = state_calibration(glass)
        gates = route_trigger(tables, outer, inner)
    operating = config["abstention"]
    inputs._equal(structure["retained"], round(operating["train_coverage"]*len(glass)), "historical coverage")
    np.testing.assert_allclose(structure["retained_f2"], operating["train_f2"], atol=1e-12, rtol=0)
    deferred = glass.loc[glass.route.eq("uncertain")]
    quality = []
    for voter, p in (("stage1", deferred.stage1_proba), ("stage3", deferred.stage3_proba)):
        y = deferred.y
        two = y.nunique() == 2
        quality.append(dict(voter=voter, n=len(y), positive=int(y.sum()), negative=int(len(y)-y.sum()),
                            prevalence=y.mean() if len(y) else None, roc_auc=roc_auc_score(y,p) if two else None,
                            average_precision=average_precision_score(y,p) if two else None,
                            brier=brier_score_loss(y,p) if len(y) else None, ece=calculate_ece(y,p) if len(y) else None))
    comparison = {}
    for rule, ref in reference["procedures"]["glass"]["glass_formula"]["rules"].items():
        result = ref["pooled"]
        comparison[rule] = {k: result["outer"][k] for k in ("coverage","loss","kept")} if result else None
    total_unique = {r["voter"]: r["sole_correct"] for r in structure["contingency"]["unique"]
                    if r["route"] == "all" and r["label"] == "all"}
    enabled = [(r["upstream"], r["outer_fold"]) for r in gates if r["status"] == "met"]
    evidence = dict(handoff=dict(sole_correct_by_active_voter=total_unique,
        forced_positive_decisions_with_negative_votes=sum(r["forced_positive"] for r in structure["margin_bound"]),
        eligible_route_variants=enabled,
        next="Evaluate the registered global Linear Arbiter first. Activate route-specific fits only in "
             "eligible partitions. Assess normalization against the unchanged formula at matched coverage. "
             "Agreement and unique wins describe complementarity; neither measures the loss after removing a voter."),
        inputs=identity, nested_reference=dict(run_id=NESTED, content_sha256=reference["content_sha256"]),
        historical_glass=structure, nested_glass_summary=comparison, deferred_quality=quality,
        calibration=[dict(voter=v, route=s, **r) for (v,s),r in whole.items()],
        registered_conditions=policy["search"]["conditional"], route_coefficients=gates,
        rule_evidence="Unresolved. Fold-specific rule sets and independent cross-fitted per-rule precision "
                      "are absent from the verified streams. Defer the ablation.",
        joint_corners="Pending the two registered one-factor inner-loss gains in Commit 4.",
        commit5_expansion="Pending the black-box experiment in Commit 5.",
        interpretation="Sole correct compares all active voters; a lone dissent requires three votes. "
            "Router veto cost compares reversing the router alone on double-negative Pass 2 rows. "
            "Negative ceiling holds votes, weights and positive evidence fixed and takes negative LR/EBM "
            "probabilities to zero. A positive side above the ceiling cannot be overturned by these voters. "
            "No explicit clipping operation is implied. Bound means use every row in the named stratum. "
            "The ranking score is separate from signed decision evidence. Matched flag counts exclude whole "
            "boundary ties when an exact contrast is unavailable. Global summaries describe frozen rows only.",
        calibration_method="Joint logistic intercept and slope on logits clipped at 1e-6 and 1-1e-6. "
            "Router uses recovered routed positive rate only. 95% Wald intervals condition on these scores. "
            "One-dimensional quasi-separation, low class counts, nonconvergence and unstable information "
            "disable the estimate. Gate estimates are ordered pooled outer training then inner subsets 0,1,2. "
            "Magnitude on pooled rows and matching sign in two supported subsets activate that fold only.",
        limitations=[nested.LIMITATIONS[0], nested.LIMITATIONS[-1],
                     "Conditional row uncertainty excludes upstream refitting; exploratory counts do not identify causal effects."],
        reproduce="python -m feature_research.voter_diagnostics")
    record = rr.start_run(commit=3, seeds={"inner":nested.INNER_SEED}, root=root, notebook=nested.NOTEBOOK,
                          code_files=(*nested.CODE_FILES, "feature_research/voter_diagnostics.py",
                                      "shared/stage2/diagnostics/operating_points.py"))
    evidence["elapsed_seconds"] = perf_counter()-start
    return rr.write_run_record(record, "voter_diagnostics", evidence)


if __name__ == "__main__":
    print(run())
