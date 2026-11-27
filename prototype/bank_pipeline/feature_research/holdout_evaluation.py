"""Fit and lock Stage 4 on fitting-partition OOF evidence, then compare paired holdout predictions."""

import numpy as np
import pandas as pd

from feature_research import stage4_comparators as c
from stage1_calibration.evidence import stratified_bootstrap_delta

n = c.n


def summary(v):
    a = np.asarray(v, float)
    return (
        dict(mean=float(a.mean()), q05_median_q95=np.quantile(a, [0.05, 0.5, 0.95]).tolist()) if len(a) else None
    )


def formula(f, model, margin):
    scores = c.FORMULA.score(model, f)
    return c.normalized(f, model, scores)[0] if margin else scores


def lock_scores(train, holds, params, policy, glass, cache):
    """hold frames contain features only. Each outer Meta fit scores both paths before release."""
    if any("y" in f or not f.index.equals(holds["full"].index) for f in holds.values()):
        raise ValueError("Holdout features are misaligned or contain labels")
    y = train.y.to_numpy()
    inner = n.FoldPlan.build(y, 3, 42).fold_id
    names = ["glass_formula"] + (["logit_margin"] if glass else [])
    pieces = {name: [] for name in names}
    for j in range(3):
        m = c.FORMULA.fit(n.FORMULA, train[inner != j])
        for name in names:
            pieces[name].append(formula(train[inner == j], m, name == "logit_margin"))
    model = c.FORMULA.fit(n.FORMULA, train)
    streams = {
        name: dict(
            fitting_oof=pd.concat(parts, verify_integrity=True).loc[train.index],
            **{mode: formula(f, model, name == "logit_margin") for mode, f in holds.items()},
        )
        for name, parts in pieces.items()
    }
    meta = n.MetaXGBEstimator(random_state=42, n_jobs=1)
    X = n.meta_features(train)
    hold_x = {mode: n.meta_features(f) for mode, f in holds.items()}
    p, parts, ledger = np.full(len(y), np.nan), {mode: [] for mode in holds}, []
    for k in [0, 1, 2, 3, 4, -1]:
        fit = np.ones(len(y), bool) if k == -1 else train.stage3_fold.to_numpy() != k

        def produce():
            m = meta.fit(params, X[fit], y[fit], sample_weight=n.BalancedWeights.balanced(y[fit]).vector)
            a = {mode: meta.positive_proba(m, v) for mode, v in hold_x.items()}
            if k != -1:
                a["oof"] = meta.positive_proba(m, X[~fit])
            return a, dict(base_fits=1, fit_rows=int(fit.sum()))

        a, info = cache("meta", k, produce)
        ledger.append(info)
        if k != -1:
            p[~fit] = a["oof"]
        for mode in holds:
            parts[mode].append(a[mode])
    if not np.isfinite(p).all():
        raise ValueError("Incomplete Meta OOF")
    streams["meta_xgb"] = {"fitting_oof": pd.DataFrame({"ranking": p}, index=train.index)}
    for u, values in parts.items():
        for key, pred in [("full", values[-1]), ("average", np.mean(values[:-1], axis=0))]:
            streams["meta_xgb"][u + "__" + key] = pd.DataFrame({"ranking": pred}, index=holds[u].index)
    locks = {
        name: n.lock("registered", c.META if name == "meta_xgb" else c.FORMULA, v["fitting_oof"], y, policy)
        for name, v in streams.items()
    }
    if any(v is None for v in locks.values()):
        raise ValueError("No feasible fitting-side operating rule")
    return model, streams, locks, ledger


def profile(frame, labels, model, policy):
    if not frame.index.equals(labels.index):
        raise ValueError("Unpaired profile labels")
    x = n.meta_features(frame)
    a, b = frame.stage1_proba >= model["lr"], frame.stage3_proba >= model["ebm"]
    values = x.assign(
        router_base=frame.router_base_rate,
        router_confidence=frame.router_confidence,
        stage3_raw=frame.stage3_raw,
        margin1=frame.stage1_proba - model["lr"],
        margin3=frame.stage3_proba - model["ebm"],
    )
    out = {}
    for name, mask in [("all", np.ones(len(labels), bool)), ("positive", labels == 1), ("negative", labels == 0)]:
        out[name] = dict(
            rows=int(mask.sum()),
            features={k: summary(v[mask].dropna()) for k, v in values.items()},
            route_counts=frame.route[mask].value_counts().to_dict(),
            voter_agreement=float((a[mask] == b[mask]).mean()),
        )
    out["quality"] = {
        col: n._all_rows(
            labels.to_numpy(), (frame[col] >= 0.5).astype(int), frame[col].to_numpy(), True, policy["cost"]
        )
        for col in ("stage1_proba", "stage3_proba")
    }
    return out


def evaluate(streams, locks, train_y, hold_y, policy):
    reports = {}
    for name, cells in streams.items():
        arm = c.META if name == "meta_xgb" else c.FORMULA
        lock = locks[name]
        conditions, actions, curves = {}, {}, {}
        for mode, f in cells.items():
            labels = train_y if mode == "fitting_oof" else hold_y
            if not labels.index.equals(f.index):
                raise ValueError("Unpaired evaluation labels")
            y = labels.to_numpy()
            d, conf = arm.operate(f, lock["threshold"])
            keep = conf >= lock["cut"]
            actions[mode] = d, conf, keep, c.linear.realized(y, d, keep, policy["cost"])
            curve = curves[mode] = n.risk_curve(y, d, conf, policy["cost"])
            conditions[mode] = dict(
                issued=n._outcome(y, d, keep, policy["cost"]),
                all_rows=n._all_rows(y, d, f.ranking.to_numpy(), arm.probability, policy["cost"]),
                partial_aurc=n.partial_aurc(curve, policy["coverage"]["min_total"]),
                ranking=summary(f.ranking),
                confidence=summary(conf),
            )
        pairs = (
            {"upstream": ("full", "average")}
            if name != "meta_xgb"
            else {
                "upstream": ("full__full", "average__full"),
                "upstream_at_mean_meta": ("full__average", "average__average"),
                "meta_at_full_upstream": ("full__full", "full__average"),
                "meta_at_mean_upstream": ("average__full", "average__average"),
                "combined": ("full__full", "average__average"),
            }
        )
        paired = {}
        for contrast, (first, second) in pairs.items():
            d, a, keep, loss = actions[first]
            e, b, other, ref_loss = actions[second]
            u = policy["uncertainty"]
            ci = stratified_bootstrap_delta(
                loss - ref_loss,
                hold_y,
                np.zeros(len(hold_y), int),
                replicates=u["replicates"],
                seed=u["seed"],
                level=0.95,
            )
            left, right = [conditions[s]["issued"] for s in (first, second)]
            paired[contrast] = dict(
                first=first,
                second=second,
                delta={
                    k: left[k] - right[k] if None not in (left[k], right[k]) else None
                    for k in (
                        "coverage",
                        "positive_retention",
                        "negative_retention",
                        "effective_recall",
                        "f2",
                        "loss",
                        "risk",
                    )
                },
                changed_classes=int(sum(d != e)),
                changed_retention=int(sum(keep != other)),
                gained=int(sum(keep & ~other)),
                lost=int(sum(other & ~keep)),
                loss_interval=ci,
                confidence_difference=summary(a - b),
                score_difference=summary(cells[first].ranking - cells[second].ranking),
                matched={
                    str(g): n.matched_risks({s: curves[s] for s in (first, second)}, g)
                    for g in policy["coverage"]["grid"]
                },
            )
        reports[name] = dict(locked=lock, conditions=conditions, paired=paired)
    return reports
