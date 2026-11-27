"""PR 43 Step 4A. Nested nonnegative linear voting on fixed training-side OOF inputs."""
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

from feature_research import research_records as rr, stage4_nested as n, voter_diagnostics as diagnostics
from feature_research.stage4_policy import loss_verdict, matched_risks, risk_curve
from glass_pipeline.glass_arbiter.router_evidence import rule_positive_rate
from shared.stage1.oof_calibration import PlattCalibrator
from stage1_calibration.evidence import stratified_bootstrap_delta

DIAGNOSTIC_RUN = "c3_20261125_163120"
DIAGNOSTIC_SHA = "b5ab3779cb5f74557ce7cb7a81e37a8c4be3a469d883a33e0f9b116e875ce8ba"
PENALTIES = (.001, .01, .1, 1.)
# Probability threshold enumeration and distance-to-threshold are the accepted Meta operations.
OPERATING = n.MetaXGB([])
STATES = ("pass1", "pass2", "uncertain")


class FitFailure(RuntimeError):
    """An unavailable registered fit cannot yield a scientific verdict."""


def eligibility(record, reference, identity):
    if (record["content_sha256"] != DIAGNOSTIC_SHA or record["inputs"] != identity or
            record["nested_reference"] != {"run_id": diagnostics.NESTED, "content_sha256": reference["content_sha256"]}):
        raise ValueError("Commit 3 identity mismatch")
    gates = {(r["upstream"], r["outer_fold"]): r["status"] for r in record["route_coefficients"]}
    expected = {(u, k): ("met" if u == "blackbox" else "not met") for u, k in product(n.UPSTREAMS, range(5))}
    if gates != expected or len(record["route_coefficients"]) != len(expected):
        raise ValueError("Unexpected Commit 3 trigger dispositions")
    return gates


def cuts(rows):
    return [n.find_recall_threshold(rows.y, rows[c], .7) for c in ("stage1_proba", "stage3_proba")]


def evidence(rows, thresholds):
    """Both producers blend routed positive rate with fold base b. Never sign-flip that rate."""
    route = pd.Categorical(rows.route, categories=STATES).codes
    if (route < 0).any():
        raise ValueError("Unknown routing state")
    b = rows.router_base_rate.to_numpy()
    r = rule_positive_rate(rows.router_score, route == 0, route == 1, b)

    def finite_logit(p):
        return logit(np.clip(p, 1e-6, 1 - 1e-6))
    x = np.column_stack([finite_logit(rows.stage1_proba) - finite_logit(thresholds[0]),
                         finite_logit(rows.stage3_proba) - finite_logit(thresholds[1]),
                         np.where(route == 2, 0., finite_logit(r) - finite_logit(b))])
    return x, route


def ridge(x, state, y, penalty, anchor=None):
    """Mean unweighted logistic loss. Route slopes have unit shrinkage to a fixed global anchor."""
    size = 3 if anchor is None else 9

    def objective(beta):
        slopes = beta[1:].reshape(-1, 3)
        active = slopes[0] if anchor is None else slopes[state]
        z = beta[0] + (x * active).sum(axis=1)
        residual = (expit(z) - y) / len(y)
        g = x * residual[:, None]
        gradient = g.sum(axis=0) if anchor is None else np.stack([g[state == i].sum(axis=0) for i in range(3)])
        loss = np.mean(np.logaddexp(0., z) - y * z) + penalty / 2 * (slopes**2).sum()
        grad = gradient.ravel() + penalty * beta[1:]
        if anchor is not None:
            delta = slopes - anchor[1:]
            loss += (delta**2).sum() / 2
            grad += delta.ravel()
        return loss, np.r_[residual.sum(), grad]
    result = minimize(objective, np.zeros(size + 1), jac=True, method="L-BFGS-B",
                      bounds=[(None, None)] + [(0, None)] * size,
                      options=dict(maxiter=500, gtol=1e-6, ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all():
        raise FitFailure(f"Linear optimization failed: {result.message}")
    return result.x, int(result.nit)


def logits(beta, data):
    x, state = data
    slopes = beta[1:] if len(beta) == 4 else beta[1:].reshape(3, 3)[state]
    return beta[0] + (x * slopes).sum(axis=1)


def select(rows, ids, route_allowed, policy):
    """Cross-fit every supervised operation. No evaluation rows enter this function."""
    keys = list(product(("global", "route") if route_allowed else ("global",), PENALTIES))
    y = rows.y.to_numpy()
    oof = {k: np.full((2, len(y)), np.nan) for k in keys}
    failures = {}
    counts = {"ridge": 0, "sigmoid": 0}

    def fit_scores(fit, val):
        threshold = cuts(rows.iloc[fit])
        x, state = evidence(rows.iloc[fit], threshold)
        target = evidence(rows.iloc[val], threshold)
        scores = {}
        for penalty in PENALTIES:
            anchor = None
            for form in (("global", "route") if route_allowed else ("global",)):
                key = form, penalty
                needs_anchor = form == "global" and route_allowed and ("route", penalty) not in failures
                if key in failures and not needs_anchor:
                    continue
                if form == "route" and anchor is None:
                    failures[key] = "global anchor unavailable"
                    continue
                counts["ridge"] += 1
                try:
                    beta, _ = ridge(x, state, y[fit], penalty, anchor)
                except RuntimeError as exc:
                    failures[key] = str(exc)
                    continue
                scores[key] = logits(beta, target)
                if form == "global":
                    anchor = beta
        return scores
    for j in range(3):
        fit, val = np.flatnonzero(ids != j), np.flatnonzero(ids == j)
        local = n.FoldPlan.build(y[fit], 3, 43).fold_id
        sub = {k: np.full(len(fit), np.nan) for k in keys}
        for h in range(3):
            for key, score in fit_scores(fit[local != h], fit[local == h]).items():
                sub[key][local == h] = score
        for key, z in fit_scores(fit, val).items():
            if key in failures:
                continue
            oof[key][0, val] = z
            counts["sigmoid"] += 1
            try:
                cal = PlattCalibrator().fit(sub[key], y[fit], nondecreasing=True)
                oof[key][1, val] = cal.predict(z)
            except RuntimeError as exc:
                failures[key] = str(exc)
    candidates, viable = [], []
    for order, key in enumerate(keys):
        row = dict(form=key[0], penalty=key[1])
        candidates.append(row)
        if key in failures:
            row["failure"] = failures[key]
            continue
        z, calibrated = oof[key]
        raw = expit(z)
        a, b = log_loss(y, raw), log_loss(y, calibrated)
        correction = b < a
        prob = calibrated if correction else raw
        locked = n.lock("registered", OPERATING, pd.DataFrame({"ranking": prob}), y, policy)
        row.update(raw_log_loss=a, sigmoid_log_loss=b, sigmoid=bool(correction), locked=locked)
        if locked is not None:
            viable.append((locked["inner"]["loss"], key[0] == "route", correction, order))
    if not viable:
        raise FitFailure(f"No feasible linear specification: {candidates}")
    # Rank eligible procedures by loss, then number of coefficients, then registration order.
    ranked = [r[-1] for r in sorted(viable)]
    return candidates, ranked, oof, counts


def fit_outer(train, evaluation, ids, eligible, policy):
    start = perf_counter()
    candidates, ranked, oof, counts = select(train, ids, eligible, policy)
    threshold = cuts(train)
    x, state = evidence(train, threshold)
    target = evidence(evaluation, threshold)
    outputs, anchors = {}, {}
    for i in ranked:
        spec = candidates[i]
        form, penalty = spec["form"], spec["penalty"]
        if form in outputs:
            continue
        if penalty not in anchors:
            anchors[penalty] = ridge(x, state, train.y.to_numpy(), penalty)
            counts["ridge"] += 1
        beta, nit = anchors[penalty]
        if form == "route":
            beta, nit = ridge(x, state, train.y.to_numpy(), penalty, beta)
            counts["ridge"] += 1
        z = logits(beta, target)
        try:
            cal = PlattCalibrator().fit(oof[form, penalty][0], train.y, nondecreasing=True)
        except RuntimeError as exc:
            raise FitFailure(str(exc)) from exc
        counts["sigmoid"] += 1
        raw, corrected = expit(z), cal.predict(z)
        p = corrected if spec["sigmoid"] else raw
        scores = pd.DataFrame(dict(ranking=p, raw=raw, sigmoid=corrected), index=evaluation.index)
        config = dict(form=form, penalty=penalty, thresholds=threshold, beta=beta.tolist(), iterations=nit,
                      global_anchor=anchors[penalty][0].tolist(), sigmoid=spec["sigmoid"],
                      correction_intercept=-cal.b_, correction_slope=-cal.a_, locked=spec["locked"])
        outputs[form] = scores, config
    # Full inner metrics are needed only for locked winners; keep the trial ledger compact.
    trials = [{**{k: v for k, v in row.items() if k != "locked"},
               "inner_loss": row["locked"]["inner"]["loss"] if row.get("locked") else None}
              for row in candidates]
    info = dict(trials=trials, counts=counts, seconds=perf_counter() - start, selected=candidates[ranked[0]]["form"])
    return outputs, info


def realized(y, d, keep, cost):
    return np.where(keep, np.where(d > y, cost["fp"], np.where(d < y, cost["fn"], 0)), cost["defer"])


def reference_frame(name, table, outer, ref, arrays, policy, upstream):
    """Restore the exact accepted operating decisions. Require its full frame fingerprint."""
    info = ref["procedures"][upstream][name]["rules"]["registered"]
    folds = pd.DataFrame(info["folds"]).set_index("outer_fold")
    prefix = f"{upstream}.{name}.outer."
    p = arrays[prefix + "ranking"].astype(float)
    if name == "glass_formula":
        d = arrays[prefix + "decision"].astype(int)
        conf = arrays[prefix + "confidence"].astype(float)
    else:
        t = folds.threshold.loc[outer].to_numpy(float)
        d, conf = (p >= t).astype(int), np.abs(p - t)
    cut = folds.cut.loc[outer].to_numpy(float)
    frame = pd.DataFrame(dict(outer_fold=outer.astype(int), ranking=p, decision=d, confidence=conf, cut=cut,
                              kept=conf >= cut), index=table.index)
    frame["row_loss"] = realized(table.y.to_numpy(), d, frame.kept, policy["cost"])
    if n._digest(frame) != info["outer_predictions"]["sha256"]:
        raise ValueError(f"{upstream}/{name}: restored reference differs from accepted OOF rows")
    return frame, info["pooled"]


def compare(table, frame, pooled, folds, ref, policy, *, formula=False, primary=True, candidate_name="linear"):
    other, summary = ref
    if not frame.index.equals(other.index) or not frame.index.equals(table.index) or not np.array_equal(
            frame.outer_fold, other.outer_fold):
        raise ValueError("Unpaired reference")
    u, limits = policy["uncertainty"], policy["decision"]
    delta = (frame.row_loss - other.row_loss).to_numpy()
    ids, y = frame.outer_fold.to_numpy(), table.y.to_numpy()
    ci = stratified_bootstrap_delta(delta, y, ids, replicates=u["replicates"], seed=u["seed"],
                                    level=1 - u["family_alpha"] / u["family_size"] if primary else .95)
    diffs = [float(delta[ids == k].mean()) for k in range(5)]
    sd = float(np.std(diffs, ddof=1))
    verdict = loss_verdict(ci["lower"], ci["upper"], ci["point"], sd, limits)
    risks = []
    for k in range(5):
        at = ids == k
        curves = {s: risk_curve(y[at], v.decision[at], v.confidence[at], policy["cost"])
                  for s, v in ((candidate_name, frame), ("reference", other))}
        for g in policy["coverage"]["grid"]:
            count, pair, brackets = matched_risks(curves, g)
            risks.append(dict(fold=k, coverage=g, count=count, risks=pair, brackets=brackets))
    m, r, p = pooled["outer"], summary["outer"], pooled["all_rows"]
    coverage = policy["coverage"]
    checks = {"absolute_floors": bool(((folds.outer_coverage >= coverage["min_total"]) &
        (folds.outer_positive_kept >= coverage["min_retained_each_class"]) &
        (folds.outer_negative_kept >= coverage["min_retained_each_class"]) &
        (folds.outer_positive_retention >= coverage["min_each_class"]) &
        (folds.outer_negative_retention >= coverage["min_each_class"])).all()),
        "aurc": pooled["partial_aurc_row_weighted"] <= summary["partial_aurc_row_weighted"] + limits["max_aurc_increase"]}
    for c in ("positive_retention", "negative_retention"):
        checks[c] = m[c] >= r[c] - (0 if formula else limits["max_retention_drop"])
    if "brier" in p:
        checks["ece"] = p["ece"] <= limits["max_ece"]
        checks["stage1_brier"] = p["brier"] <= brier_score_loss(y, table.stage1_proba) + limits["max_brier_increase"]
    if "brier" in p and "brier" in summary["all_rows"]:
        checks["reference_brier"] = p["brier"] <= summary["all_rows"]["brier"] + limits["max_brier_increase"]
    return dict(interval=ci, fold_differences=diffs, fold_sd=sd, loss_verdict=verdict, safeguards=checks,
                matched=risks, sensitivity={str(v): m[f"loss_defer_{v:g}"] - r[f"loss_defer_{v:g}"]
                                           for v in policy["cost"]["sensitivity_defer"]})


def disposition(comparisons):
    verdicts = [c["loss_verdict"] for c in comparisons.values()]
    if "worse" in verdicts or any(not all(c["safeguards"].values()) for c in comparisons.values()):
        return "reject"
    if comparisons["glass_formula"]["loss_verdict"] == "improved" and verdicts.count("inconclusive") == 0:
        return "provisional advance"
    if comparisons["glass_formula"]["loss_verdict"] == "comparable":
        return "retain formula"
    return "inconclusive"


def coefficients(table, outer, eligible, policy):
    """Select once on pooled training rows. Resample paired rows within original fold x class."""
    y = table.y.to_numpy()
    ids = n.FoldPlan.build(y, 3, 42).fold_id
    candidates, ranked, _, counts = select(table, ids, eligible, policy)
    choice = candidates[ranked[0]]
    form, penalty = choice["form"], choice["penalty"]
    threshold = cuts(table)
    x, state = evidence(table, threshold)
    strata = [np.flatnonzero((outer == k) & (y == cls)) for k, cls in product(range(5), (0, 1))]
    rng = np.random.default_rng(4303)
    point, _ = ridge(x, state, y, penalty)
    if form == "route":
        point, _ = ridge(x, state, y, penalty, point)
    estimates = []
    for _ in range(500):
        ix = np.concatenate([rng.choice(v, len(v), replace=True) for v in strata])
        beta, _ = ridge(x[ix], state[ix], y[ix], penalty)
        if form == "route":
            beta, _ = ridge(x[ix], state[ix], y[ix], penalty, beta)
        estimates.append(beta)
    lo, hi = np.quantile(estimates, [.025, .975], axis=0)
    return dict(selected=choice, voter_cuts=threshold, selection_fits=counts, replicates=500, seed=4303,
                bootstrap_ridge_fits=500 * (1 + (form == "route")), point=point.tolist(), lower=lo.tolist(), upper=hi.tolist(),
                estimand="Uncalibrated additive coefficients conditional on fixed channels, selected form, "
                        "pooled voter cuts and ridge penalty. Route anchors refitted in each replicate.")


def evaluate(table, upstream, outer, inner, gates, policy, ref, arrays):
    refs = {name: reference_frame(name, table, outer, ref, arrays, policy, upstream)
            for name in ("glass_formula", "meta_xgb")}
    pieces, logs, details = {}, {}, []
    for k, ids in inner.items():
        tr = outer != k
        allowed = gates[upstream, k] == "met"
        outputs, info = fit_outer(table.loc[tr], table.loc[~tr].drop(columns="y"), ids, allowed, policy)
        info["fold"] = k
        info["configurations"] = {}
        for form, (scores, config) in outputs.items():
            if not scores.index.equals(table.index[~tr]):
                raise ValueError("Outer row order mismatch")
            y = table.y.to_numpy()[~tr]
            result, d, c, keep = n._fold_row(OPERATING, y, scores, config["locked"], policy)
            result.update(rows=len(y), outer_fold=k)
            frame = scores.assign(outer_fold=k, decision=d, confidence=c, kept=keep,
                                  row_loss=realized(y, d, keep, policy["cost"]))
            names = [form, "selected"] if form == info["selected"] else [form]
            for name in names:
                pieces.setdefault(name, []).append(frame)
                logs.setdefault(name, []).append(result)
            info["configurations"][form] = config
        details.append(info)
    assembled, reports = {}, {}
    for name, parts in pieces.items():
        if len(parts) != 5:
            continue
        f = pd.concat(parts, verify_integrity=True).loc[table.index]
        folds = pd.DataFrame(logs[name])
        y = table.y.to_numpy()
        pooled = n._pooled(folds, y, f.decision.to_numpy(), f.ranking.to_numpy(), f.kept.to_numpy(), True, policy)
        quality = {}
        for col in ("raw", "sigmoid"):
            prob = f[col].to_numpy()
            quality[col] = dict(log_loss=log_loss(y, prob), brier=brier_score_loss(y, prob),
                                ece=n.calculate_ece(y, prob), calibration=diagnostics.calibration(y, prob))
        assembled[name] = (f, pooled)
        reports[name] = dict(pooled=pooled, probability=quality, folds=logs[name])
    frame, pooled = assembled["selected"]
    folds = pd.DataFrame(logs["selected"])
    comparisons = {name: compare(table, frame, pooled, folds, other, policy, formula=name == "glass_formula")
                   for name, other in refs.items()}
    verdict = disposition(comparisons)
    if any("failure" in trial for fold in details for trial in fold["trials"]):
        verdict = "inconclusive: optimization failure"
    route_contrast = None
    if "route" in assembled:
        r, p = assembled["route"]
        route_contrast = compare(table, r, p, pd.DataFrame(logs["route"]), assembled["global"], policy, primary=False)
    boot = None
    if verdict == "provisional advance":
        try:
            boot = coefficients(table, outer, all(gates[upstream, k] == "met" for k in inner), policy)
        except FitFailure as exc:
            boot = dict(failure=str(exc))
            verdict = "inconclusive: coefficient refit failed"
    stability = {}
    for form in ("global", "route"):
        values = [f["configurations"][form]["beta"] for f in details if form in f["configurations"]]
        if values:
            betas = np.asarray(values)
            stability[form] = dict(folds=len(values), mean=betas.mean(axis=0).tolist(),
                                   sd=betas.std(axis=0, ddof=1).tolist() if len(values) > 1 else None,
                                   at_zero=(betas[:, 1:] == 0).sum(axis=0).tolist())
    aliases = {}
    for form in ("global", "route"):
        if form in reports and all(f["selected"] == form for f in details):
            del reports[form]
            aliases[form] = "selected"
    route_evidence = None
    if route_contrast:
        route_evidence = ("descriptive support for additional route coefficients"
                          if route_contrast["loss_verdict"] == "improved" and all(route_contrast["safeguards"].values())
                          else "independent advantage of route-specific slopes unestablished")
    return dict(selection=details, procedures=reports, identical_procedures=aliases, coefficient_stability=stability,
                route_evidence=route_evidence, reference_comparisons=comparisons,
                route_minus_global_descriptive=route_contrast, disposition=verdict, coefficient_intervals=boot)


def run():
    start = perf_counter()
    record = rr.read_run_record(DIAGNOSTIC_RUN, "voter_diagnostics", root=n.LOGS)
    if record["content_sha256"] != DIAGNOSTIC_SHA:
        raise ValueError("Unexpected voter diagnostic record")
    tables, _, policy, identity, ref, arrays, outer, inner = diagnostics.verified_tables()
    gates = eligibility(record, ref, identity)
    with threadpool_limits(limits=1):
        results = {}
        for upstream, table in tables.items():
            try:
                results[upstream] = evaluate(table, upstream, outer, inner, gates, policy, ref, arrays)
            except FitFailure as exc:
                results[upstream] = dict(disposition="inconclusive: registered fit unavailable", failure=str(exc))
    run = rr.start_run(commit=4, seeds={"inner": 42, "subfold": 43, "paired": 4302, "coefficients": 4303},
                       root=n.LOGS, notebook=n.NOTEBOOK, code_files=(*n.CODE_FILES,
                       "feature_research/linear_arbiter.py", "feature_research/voter_diagnostics.py",
                       "shared/stage1/oof_calibration.py", "stage1_calibration/evidence.py"))
    return rr.write_run_record(run, "linear_arbiter", dict(inputs=identity, diagnostic=DIAGNOSTIC_SHA,
        nested_reference=record["nested_reference"], results=results, elapsed_seconds=perf_counter() - start,
        reconstruction="Reapply the recorded fold voter cuts, beta, optional sigmoid and locked cuts to the verified tables.",
        layout="Uncalibrated beta: free intercept then Stage 1, Stage 3, router. Route form repeats "
               "the triple for pass1, pass2, uncertain. Deferred router channel is identically zero; "
               "its nominal slope is penalty-determined. Sigmoid is applied to the additive logit.",
        uncertainty="Four paired 98.75% primary intervals. Route/global contrast is secondary. "
                    "Coefficient intervals only for a provisionally retained specification.",
        claim=n.LIMITATIONS[0] + " " + n.LIMITATIONS[-1],
        reproduce="python -m feature_research.linear_arbiter"))


if __name__ == "__main__":
    print(run())
