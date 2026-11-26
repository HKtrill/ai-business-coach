"""
PR 43 Step 2C. Nested Stage 4 references on the frozen Step 2A OOF tables.

Registered design (``stage4_policy_v1`` "nesting"): the five outer folds are the accepted shared
producer folds; three stratified inner folds (seed 42) inside each outer training partition give
the inner predictions that select every learned Stage 4 quantity. The selected procedure is
refitted on the outer training partition, its class threshold and deferral cut are locked, and it
scores the untouched outer rows once.

Procedures per upstream: the accepted GLASS formula (recall-0.70 voter cuts, hybrid weights,
weighted vote); Meta-XGBoost over the registered 12 RandomSampler proposals, ranked by the
historical F2-at-0.5 objective, raw operating space; Meta-XGBoost at the accepted PR 39
hyperparameters (descriptive: they were selected on all training rows). Each runs under the
historical rule (retained F2, coverage >= 0.50) and the registered whole-system policy.

Upstream inputs stay fixed: the fits that produced a row's Stage 1-3 inputs can include other
outer rows' labels. This is nested Stage 4 evaluation, not nested cascade evaluation.
"""
import hashlib
from fractions import Fraction
from io import BytesIO
from math import ceil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score, log_loss

from blackbox_pipeline.models.meta_xgb.config import STAGE4_FEATURES
from blackbox_pipeline.models.meta_xgb.estimator import MetaXGBEstimator
from feature_research import research_records as records
from feature_research.stage4_policy import (LOGS, POLICY_SHA256, load_policy, matched_risks,
                                            operating_cut, partial_aurc, risk_curve)
from feature_research.training_inputs import BANK_ROOT, CROSSOVER
from glass_pipeline.glass_arbiter.arbiter import meta_arbiter
from glass_pipeline.glass_arbiter.calibration import compute_calibration
from glass_pipeline.glass_arbiter.meta_stage import find_recall_threshold
from glass_pipeline.glass_arbiter.weighting import compute_hybrid_weights
from glass_pipeline.glass_router.pipeline import fold_fingerprint
from shared.metrics import calculate_ece
from shared.pinned_artifacts import PinnedFile, read_verified
from shared.stage3.data import BalancedWeights, FoldPlan
from shared.stage3.thresholding import F2ThresholdSelector
from shared.stage4.protocol import ABSTENTION_MIN_COVERAGE, tune_min_confidence
from stage4_regeneration import runner as stage4_records
from upstream_regeneration.records import file_sha256_lf

NOTEBOOK = "feature_research/training_research.ipynb"
REGISTRATION = "c2_20261125_112854"                   # accepted Step 2B receipt
UPSTREAMS = ("glass", "blackbox")
RULES = ("historical", "registered")
INNER_FOLDS, INNER_SEED = 3, 42                       # policy "nesting"
CLASS_THRESHOLDS = (0.10, 0.20, 1 / 3, 0.50)          # policy "search.common", listed order
PROPOSALS, ATTEMPTS, SAMPLER_SEED = 12, 24, 42        # policy "search.meta"
FORMULA = {"target_recall": 0.70, "alpha": 0.50}      # accepted PR 39 configuration
PREDICTIONS = "nested_predictions.npz"                # local, git-ignored (research_logs/pr43/.gitignore)
STEP_4B_REUSE = ("Both Meta objectives rank the stored inner predictions of every proposal (no refit). A "
                 "log-loss winner that differs from the F2 winner is refitted once per outer training "
                 "partition. Calibrated-space inner predictions need three seed-43 subfold fits per inner "
                 "fold per distinct winner; outer isotonic maps fit on the stored inner predictions.")
PRODUCER_CODE = ("feature_research/training_inputs.py", "feature_research/training_side.py",
                 "shared/stage2/diagnostics/boundary.py", "shared/stage4/protocol.py")
CODE_FILES = (NOTEBOOK, "feature_research/stage4_nested.py", "feature_research/stage4_policy.py",
              *PRODUCER_CODE, "glass_pipeline/glass_arbiter/arbiter.py",
              "glass_pipeline/glass_arbiter/weighting.py", "glass_pipeline/glass_arbiter/meta_stage.py",
              "blackbox_pipeline/models/meta_xgb/estimator.py")
LIMITATIONS = (
    "Nested Stage 4 only: upstream OOF inputs are fixed. Stage 2/3 inputs of outer-training rows "
    "come from producers fitted with outer-evaluation labels; Stage 1 (ten folds) producers of "
    "evaluation rows saw other evaluation rows. Upstream nested selection is PR 44.",
    "Inner-to-outer differences mix selection bias with sampling variation and fit size: inner "
    "models use two thirds of the outer training partition, the outer model all of it.",
    "Meta-XGBoost selects its proposal, class threshold and deferral cut on the same pooled inner "
    "predictions; the inner loss is optimistic for all three and outer evaluation measures the "
    "total without attributing it. The formula's inner optimism comes from its deferral cut and "
    "from pooling three inner fits with their own cuts and weights.",
    "Historical-rule references select on pooled inner predictions with one class threshold; PR 39 "
    "selected the GLASS cut on its fitting rows and the Meta cut with fold-nested thresholds. "
    "Frozen PR 39 operating points (Step 2A replay) are a different evaluation condition.",
    "Meta-XGBoost uses raw balanced-weight probabilities (accepted operating space); they are not "
    "calibrated. Log-loss ranking and isotonic space are Step 4B; every proposal's inner predictions "
    "are stored for that reuse. Balanced weights come from each fit's own rows.",
    "Random-split, training-only evidence: target-derived features can use later contacts' labels "
    "and MLP early stopping shared encoder fitting (PR 42). Forward-in-time claims need PR 49.",
)


# ----------------------------------------------------------------------
# identity and reference reproduction
# ----------------------------------------------------------------------
def verified_inputs(evidence, root=LOGS):
    """Policy v1, its Step 2B receipt and the accepted Step 2A record; ``evidence`` must reproduce it."""
    policy = load_policy()
    receipt = records.read_run_record(REGISTRATION, "stage4_registration", root=root)
    frozen = receipt["frozen_inputs"]
    accepted = records.read_run_record(frozen["run_id"], "frozen_inputs", root=root)
    code = accepted["code"]["files_sha256_lf"]
    checks = {
        "registered policy digest": receipt["policy"]["canonical_sha256"] == POLICY_SHA256,
        "registered input run": frozen["run_id"] == policy["input_run"],
        "accepted Step 2A content": accepted["content_sha256"] == frozen["content_sha256"],
        "replayed Step 2A content": records.content_digest(evidence) == frozen["content_sha256"],
        "Step 2A producer code": all(code.get(f) == file_sha256_lf(BANK_ROOT / f) for f in PRODUCER_CODE),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Step 2C inputs not verified: {failed}")
    return policy, {"policy": {"id": policy["id"], "version": policy["version"],
                               "canonical_sha256": POLICY_SHA256},
                    "registration": {"run_id": REGISTRATION, "content_sha256": receipt["content_sha256"]},
                    "frozen_inputs": dict(frozen), "checks": sorted(checks)}


def meta_features(rows):
    """The production Meta-XGBoost block (``meta_xgb.inputs._side``) from Step 2A columns."""
    route = rows.route.to_numpy()
    return pd.DataFrame({
        "stage1_proba": rows.stage1_proba.to_numpy(dtype=float),
        "stage2_proba": np.where(route == "uncertain", np.nan, rows.router_score.to_numpy(dtype=float)),
        "stage2_pass1": (route == "pass1").astype(np.int8),
        "stage2_pass2": (route == "pass2").astype(np.int8),
        "stage2_abstain": (route == "uncertain").astype(np.int8),
        "stage3_proba": rows.stage3_proba.to_numpy(dtype=float),
    }, index=rows.index)[list(STAGE4_FEATURES)]


def meta_oof(params, table, fold, estimator=None):
    """Production meta OOF (``Stage3Runner``): one fit per fold, balanced weights of all rows."""
    est = estimator or MetaXGBEstimator(random_state=42, n_jobs=1)
    X, y = meta_features(table), table.y.to_numpy()
    w, p = BalancedWeights.balanced(y).vector, np.full(len(y), np.nan)
    for k in np.unique(fold):
        tr = fold != k
        p[~tr] = est.positive_proba(est.fit(params, X[tr], y[tr], sample_weight=w[tr]), X[~tr])
    return p


def reproduce_references(tables, evidence):
    """The Step 2C fitting paths reproduce the accepted PR 39 Stage 4 fits on all training rows."""
    cross = stage4_records.require_passed(CROSSOVER, "crossover_baseline")
    accepted, out = {}, {}
    for name, ref in evidence["historical_reference"].items():
        art, cfg = cross["arbiters"][name], ref["configuration"]
        if art["sha256"] != ref["arbiter_sha256"]:
            raise ValueError(f"{name}: crossover pin differs from Step 2A")
        table, upstream = tables[art["upstream"]], art["upstream"]
        if "weights" in cfg:                       # GLASS formula: cuts and weights, tolerance 1e-12
            fit, tol = Formula().fit(FORMULA, table), 1e-12
            got = [fit["lr"], fit["ebm"], *fit["weights"].values()]
            want = [cfg["thresholds"]["lr"], cfg["thresholds"]["ebm"],
                    *(cfg["weights"][k] for k in fit["weights"])]
        else:                                      # Meta-XGBoost: meta OOF, exact
            pin = PinnedFile(name, art["path"], art["sha256"], art["bytes"])
            learner = joblib.load(BytesIO(read_verified(pin, BANK_ROOT))).learner
            accepted[upstream] = MetaXGBEstimator().normalize_params(learner.best_params)
            got = meta_oof(accepted[upstream], table, np.asarray(learner.oof["fold_id"]))
            want, tol = np.asarray(learner.oof["proba"], dtype=float), 0.0
        diff = float(np.max(np.abs(np.subtract(got, want))))
        if not diff <= tol:
            raise ValueError(f"{name}: accepted fit not reproduced (max |difference| {diff:.3g})")
        out[name] = {"upstream": upstream, "max_abs_difference": diff, "tolerance": tol,
                     "reproduced": "cuts and weights" if "weights" in cfg else "meta OOF probabilities"}
    if set(accepted) != set(UPSTREAMS):
        raise ValueError("Accepted Meta-XGBoost parameters missing for an upstream")
    return accepted, out


# ----------------------------------------------------------------------
# folds and arms
# ----------------------------------------------------------------------
def nested_folds(tables):
    """Outer: shared producer folds (identical for every upstream). Inner: 3 stratified folds, seed 42."""
    first, *rest = tables.values()
    outer = first.stage3_fold.to_numpy()
    for t in rest:
        if not (t.index.equals(first.index) and np.array_equal(t.stage3_fold, outer)
                and np.array_equal(t.y, first.y)):
            raise ValueError("Upstream tables differ in rows, labels or outer folds")
    y = first.y.to_numpy()
    inner = {int(k): FoldPlan.build(y[outer != k], INNER_FOLDS, INNER_SEED,
                                    purpose=f"Stage 4 inner folds of outer fold {k}").fold_id
             for k in np.unique(outer)}
    return outer, inner


def draw_proposals(estimator, unique=PROPOSALS, attempts=ATTEMPTS, seed=SAMPLER_SEED):
    """Sequential RandomSampler draws, duplicates skipped, until ``unique`` proposals or ``attempts``."""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed))
    out = []
    for n in range(1, attempts + 1):
        params = estimator.normalize_params(estimator.suggest(study.ask()))
        if params not in out:
            out.append(params)
        if len(out) == unique:
            break
    return out, n


class Formula:
    """Accepted GLASS Arbiter procedure (``meta_stage.train_glass_arbiter`` steps 2-3, then the vote)."""
    probability, keep_inner = False, False         # no search: inner scores refit in seconds
    columns = ("ranking", "decision", "confidence")

    def __init__(self):
        self.candidates = (dict(FORMULA),)

    @staticmethod
    def _voters(rows):
        route = rows.route.to_numpy()
        return (rows.stage1_proba.to_numpy(), rows.stage3_proba.to_numpy(),
                rows.router_score.to_numpy(), route == "pass1", route == "pass2")

    def fit(self, cand, rows):
        y = rows.y.to_numpy()
        lr, ebm, r, p1, p2 = self._voters(rows)
        cov = p1 | p2
        cuts = {m: find_recall_threshold(y, p, cand["target_recall"]) for m, p in (("lr", lr), ("ebm", ebm))}
        w = compute_hybrid_weights(compute_calibration(y, lr), compute_calibration(y, ebm),
                                   compute_calibration(y[cov], r[cov]), y, lr, ebm, p2.astype(int),
                                   cuts["lr"], cuts["ebm"], alpha=cand["alpha"], glass_covered_mask=cov)
        return {**cuts, "weights": {k: w[k] for k in ("lr", "ebm", "glass")}}

    def score(self, fitted, rows):
        d, s, _, c = meta_arbiter(*self._voters(rows), fitted["lr"], fitted["ebm"], fitted["weights"],
                                  allow_abstain=False, return_confidence=True,
                                  glass_base_rate=rows.router_base_rate.to_numpy())
        return pd.DataFrame({"ranking": s, "decision": d, "confidence": c}, index=rows.index)

    def select(self, scores, y, inner):
        return 0, {}

    def thresholds(self, rule, y, score):
        return (None,)                             # weighted vote; no class-threshold search

    def operate(self, score, threshold):
        return score.decision.to_numpy(), score.confidence.to_numpy()

    def configuration(self, fitted, chosen):
        return fitted


class MetaXGB:
    """Meta-XGBoost on the six Stage 4 channels; balanced weights from the fitted rows."""
    probability, keep_inner = True, True           # registered reuse across objectives and spaces
    columns = ("ranking",)

    def __init__(self, candidates, estimator=None):
        self.candidates = tuple(candidates)
        self.estimator = estimator or MetaXGBEstimator(random_state=42, n_jobs=1)

    def fit(self, params, rows):
        y = rows.y.to_numpy()
        return self.estimator.fit(params, meta_features(rows), y,
                                  sample_weight=BalancedWeights.balanced(y).vector)

    def score(self, model, rows):
        return pd.DataFrame({"ranking": self.estimator.positive_proba(model, meta_features(rows))},
                            index=rows.index)

    def select(self, scores, y, inner):
        """Historical tuning objective: mean inner-fold F2 at the 0.5 cut; ties keep proposal order."""
        f2, folds = [], [inner == j for j in range(INNER_FOLDS)]
        for s in scores:
            p = None if s is None else s.ranking.to_numpy()
            f2.append(None if p is None else float(np.mean(
                [fbeta_score(y[m], p[m] >= 0.5, beta=2, zero_division=0) for m in folds])))
        valid = [i for i, v in enumerate(f2) if v is not None]
        return max(valid, key=lambda i: f2[i]), {"f2_at_0_5": f2}

    def thresholds(self, rule, y, score):
        if rule == "historical":                   # PR 39: F2 grid 0.10-0.90 x 81
            return (F2ThresholdSelector().select(y, score.ranking.to_numpy()).threshold,)
        return CLASS_THRESHOLDS

    def operate(self, score, threshold):
        p = score.ranking.to_numpy()
        return (p >= threshold).astype(int), np.abs(p - threshold)

    def configuration(self, fitted, chosen):
        return {"candidate": chosen}


# ----------------------------------------------------------------------
# operating rules and registered accounting
# ----------------------------------------------------------------------
def _outcome(y, d, kept, cost):
    """Registered whole-system accounting (``risk_curve``) for one retained set."""
    k, n, p = int(np.count_nonzero(kept)), len(y), int(np.count_nonzero(y))
    if k == 0:                                     # all deferred: retained-only ratios undefined
        wrong = int(np.count_nonzero(np.asarray(d) != np.asarray(y)))
        none = {"kept": 0, "coverage": 0.0, "fp": 0, "fn": 0, "tp": 0, "tn": 0, "positive_kept": 0,
                "negative_kept": 0, "positive_retention": 0.0 if p else None,
                "negative_retention": 0.0 if n - p else None, "deferred_positive": p,
                "deferred_negative": n - p, "correct_deferred": n - wrong, "wrong_deferred": wrong,
                "effective_recall": 0.0 if p else None, "retained_prevalence": None, "risk": None,
                "loss": cost["defer"], "f2": None}
        return {**none, **{f"loss_defer_{x:g}": x for x in cost["sensitivity_defer"]}}
    rank = np.asarray(kept, dtype=float)                 # retained rows rank first
    row = risk_curve(y, d, rank, cost).iloc[0].drop(["cut", "deferred"]).to_dict()  # deferred = N - kept
    den = 5 * row["tp"] + 4 * row["fn"] + row["fp"]
    row["f2"] = 5 * row["tp"] / den if den else None
    for x in cost["sensitivity_defer"]:
        row[f"loss_defer_{x:g}"] = float(risk_curve(y, d, rank, dict(cost, defer=x)).loss.iloc[0])
    return row


def _all_rows(y, d, ranking, probability, cost):
    """Before deferral: confusion counts, PR-AUC on the ranking score, probability quality."""
    every = risk_curve(y, d, np.zeros(len(y)), cost).iloc[-1]
    out = {k: int(every[k]) for k in ("tp", "fp", "tn", "fn")}
    both = 0 < y.sum() < len(y)
    out["pr_auc"] = float(average_precision_score(y, ranking)) if both else None
    if probability:                                # GLASS weighted score has no probability claim
        out.update(log_loss=float(log_loss(y, ranking, labels=[0, 1])),
                   brier=float(brier_score_loss(y, ranking)), ece=calculate_ece(y, ranking))
    return out


def lock(rule, arm, score, y, policy):
    """Inner predictions only: the rule's class threshold and deferral cut, or None if infeasible."""
    best = None
    for order, t in enumerate(arm.thresholds(rule, y, score)):
        d, c = arm.operate(score, t)
        if rule == "historical":
            hit = tune_min_confidence(y, d, c, None, ABSTENTION_MIN_COVERAGE)
            cut = None if hit is None else hit["min_confidence"]
        else:
            cut = operating_cut(risk_curve(y, d, c, policy["cost"]), policy["coverage"])
        if cut is None:
            continue
        inner = _outcome(y, d, c >= cut, policy["cost"])
        key = (inner["loss"], -inner["kept"], order)
        if best is None or key < best[0]:
            best = (key, {"threshold": t, "cut": cut, "inner": inner})
    return None if best is None else best[1]


INNER_KEYS = ("coverage", "loss", "risk", "f2", "positive_retention", "negative_retention")


def _fold_row(arm, y_E, score_E, locked, policy):
    """Apply the locked rule to one outer fold's rows; report its registered outcomes."""
    d, c = arm.operate(score_E, locked["threshold"])
    kept, cost, inner = c >= locked["cut"], policy["cost"], locked["inner"]
    outer, curve = _outcome(y_E, d, kept, cost), risk_curve(y_E, d, c, cost)
    every = _all_rows(y_E, d, score_E.ranking.to_numpy(), arm.probability, cost)
    row = {"threshold": locked["threshold"], "cut": locked["cut"],
           **{f"inner_{k}": inner[k] for k in INNER_KEYS}, **{f"outer_{k}": v for k, v in outer.items()},
           **{f"all_{k}": v for k, v in every.items()},
           "partial_aurc": partial_aurc(curve, policy["coverage"]["grid"][0]),
           "loss_optimism": outer["loss"] - inner["loss"],
           "coverage_shift": outer["coverage"] - inner["coverage"],
           "f2_optimism": None if None in (inner["f2"], outer.get("f2")) else inner["f2"] - outer["f2"]}
    for g in policy["coverage"]["grid"]:                 # exact counts: design.matched_counts
        _, risks, brackets = matched_risks({"arm": curve}, g)
        row.update({f"matched_{g:g}_risk": None if risks is None else risks["arm"],
                    f"matched_{g:g}_bracket": brackets.get("arm")})
    return row, d, c, kept


def _pooled(folds, y, d, ranking, kept, probability, policy):
    """Pooled outer OOF results, and fold values aggregated with outer row weights."""
    w, grid = (folds["rows"] / folds["rows"].sum()).to_numpy(), policy["coverage"]["grid"]
    mean = lambda col: None if folds[col].isna().any() else float(w @ folds[col].astype(float))  # noqa: E731
    return {"outer": _outcome(y, d, kept, policy["cost"]),
            "all_rows": _all_rows(y, d, ranking, probability, policy["cost"]),
            "inner_loss_row_weighted": mean("inner_loss"),
            "loss_optimism_row_weighted": mean("loss_optimism"),
            "coverage_shift_row_weighted": mean("coverage_shift"),
            "partial_aurc_row_weighted": mean("partial_aurc"),
            "matched_risk_row_weighted": {f"{g:g}": mean(f"matched_{g:g}_risk") for g in grid},
            "matched_available_folds": {f"{g:g}": int(folds[f"matched_{g:g}_risk"].notna().sum())
                                        for g in grid},
            "outer_loss_folds_unweighted_mean": float(folds.outer_loss.mean()),
            "outer_loss_folds_sd": float(folds.outer_loss.std(ddof=1)),
            "loss_optimism_folds_sd": float(folds.loss_optimism.std(ddof=1))}


def _digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def _array_digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()


def _compact(a):
    """float32 where exact (XGBoost scores are float32), int8 fold ids, otherwise unchanged."""
    a = np.asarray(a)
    if a.dtype.kind == "f" and np.array_equal(a.astype(np.float32), a, equal_nan=True):
        return a.astype(np.float32)
    return a.astype(np.int8) if a.dtype.kind == "i" else a


# ----------------------------------------------------------------------
# the common nested procedure
# ----------------------------------------------------------------------
def run_nested(arm, table, outer, inner, policy):
    """
    One arm on one upstream: (outer predictions per rule, compact evidence, fold timings, scores).
    ``scores`` holds each outer row's score once (rules only re-read it) and, for arms that keep
    them, every candidate's inner predictions as (candidate, outer fold, row), NaN off-partition.
    """
    y, n, cost = table.y.to_numpy(), len(table), policy["cost"]
    pooled = {r: {"outer_fold": outer.copy(), "ranking": np.full(n, np.nan), "decision": np.full(n, -1),
                  "confidence": np.full(n, np.nan), "cut": np.full(n, np.nan), "kept": np.zeros(n, bool)}
              for r in RULES}
    stored = {f"outer.{c}": np.full(n, np.nan) for c in arm.columns}
    if arm.keep_inner:
        shape = (len(arm.candidates), len(inner), n)
        stored.update({f"inner.{c}": np.full(shape, np.nan) for c in arm.columns})
    scored, fits, fold_rows = np.zeros(n, int), 0, {r: [] for r in RULES}
    selection, configs, seconds = [], [], []
    for k in sorted(inner):
        start, at = perf_counter(), outer == k
        T, E, folds = table[~at], table[at], inner[k]
        y_T, scores, failures = T.y.to_numpy(), [], {}
        for i, cand in enumerate(arm.candidates):
            try:
                parts = []
                for j in range(INNER_FOLDS):
                    fits += 1
                    parts.append(arm.score(arm.fit(cand, T[folds != j]), T[folds == j]))
                scores.append(pd.concat(parts).loc[T.index])
                for c in arm.columns if arm.keep_inner else ():
                    stored[f"inner.{c}"][i, k, ~at] = scores[-1][c].to_numpy()
            except Exception as exc:               # registered: failed trials count; log the cause
                scores.append(None)
                failures[str(i)] = f"{type(exc).__name__}: {exc}"
        if all(sc is None for sc in scores):
            raise RuntimeError(f"Outer fold {k}: every candidate failed {failures}")
        chosen, chosen_by = arm.select(scores, y_T, folds)
        locks = {rule: lock(rule, arm, scores[chosen], y_T, policy) for rule in RULES}   # inner only
        fits += 1
        fitted = arm.fit(arm.candidates[chosen], T)
        score_E = arm.score(fitted, E)             # the one outer scoring, after every rule is locked
        scored[at] += 1
        for c in arm.columns:
            stored[f"outer.{c}"][at] = score_E[c].to_numpy()
        selection.append({"chosen": chosen, "failures": failures, **chosen_by})
        configs.append(arm.configuration(fitted, chosen))
        for rule, locked in locks.items():
            base = {"outer_fold": int(k), "rows": int(at.sum()), "feasible": locked is not None}
            if locked is None:                     # registered: record failure, never relax floors
                fold_rows[rule].append(base)
                continue
            row, d, c, kept = _fold_row(arm, E.y.to_numpy(), score_E, locked, policy)
            fold_rows[rule].append({**base, **row})
            p = pooled[rule]
            p["ranking"][at], p["decision"][at], p["confidence"][at] = score_E.ranking.to_numpy(), d, c
            p["cut"][at], p["kept"][at] = locked["cut"], kept
        seconds.append(perf_counter() - start)
    if not (scored == 1).all():
        raise AssertionError("An outer row was not scored exactly once")
    budget = len(inner) * (len(arm.candidates) * INNER_FOLDS + 1)
    if fits > budget:                              # a failed proposal stops its remaining inner fits
        raise AssertionError(f"{fits} fits exceed the registered budget {budget}")
    frames, rules = {}, {}
    for rule in RULES:
        folds_df = pd.DataFrame(fold_rows[rule])
        if not folds_df.feasible.all():
            rules[rule] = {"folds": folds_df.to_dict("list"), "pooled": None,
                           "infeasible_folds": folds_df.outer_fold[~folds_df.feasible].tolist()}
            continue
        f = pd.DataFrame(pooled[rule], index=table.index)
        d, kept = f.decision.to_numpy(), f.kept.to_numpy()
        f["row_loss"] = np.where(~kept, cost["defer"],
                                 np.where(d > y, cost["fp"], np.where(d < y, cost["fn"], 0.0)))
        frames[rule] = f
        rules[rule] = {"folds": folds_df.to_dict("list"),
                       "pooled": _pooled(folds_df, y, d, f.ranking.to_numpy(), kept, arm.probability, policy),
                       "outer_predictions": {"rows": n, "columns": list(f.columns), "sha256": _digest(f)}}
    evidence = {"candidates": list(arm.candidates), "fits": {"attempted": fits, "budget": budget},
                "selection": selection, "configuration": configs, "rules": rules}
    return frames, evidence, seconds, stored


# ----------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------
def nested_references(tables, evidence, *, root=LOGS):
    """Both upstreams' nested references. Verifies identities and PR 39 reproduction before any search."""
    t0 = perf_counter()
    policy, identity = verified_inputs(evidence, root)
    outer, inner = nested_folds(tables)
    if fold_fingerprint(outer) != evidence["population"]["shared_fold_fingerprint"]:
        raise ValueError("Outer folds are not the accepted shared fold partition")
    before = {u: _digest(t) for u, t in tables.items()}
    accepted, reproduced = reproduce_references(tables, evidence)
    proposals, attempts = draw_proposals(MetaXGBEstimator(random_state=42, n_jobs=1))
    y, rows = tables[UPSTREAMS[0]].y.to_numpy(), tables[UPSTREAMS[0]].index
    inner_ids = np.full((len(inner), len(y)), -1)
    for k, ids in inner.items():
        inner_ids[k, outer != k] = ids
    out, frames, runtime = {}, {}, {"references_s": perf_counter() - t0}
    arrays = {"outer_fold": outer, "inner_fold": inner_ids}
    for u in UPSTREAMS:
        arms = {"glass_formula": Formula(), "meta_xgb": MetaXGB(proposals),
                "meta_xgb_accepted_params": MetaXGB([accepted[u]])}
        out[u], frames[u], runtime[u] = {}, {}, {}
        for name, arm in arms.items():
            frames[u][name], out[u][name], runtime[u][name], scores = run_nested(
                arm, tables[u], outer, inner, policy)
            arrays.update({f"{u}.{name}.{key}": v for key, v in scores.items()})
    arrays = {key: _compact(v) for key, v in arrays.items()}
    if {u: _digest(t) for u, t in tables.items()} != before:
        raise AssertionError("Step 2A tables changed during Step 2C")
    indexes = [f.index for p in frames.values() for a in p.values() for f in a.values()]
    if not all(i.equals(tables[UPSTREAMS[0]].index) for i in indexes):
        raise AssertionError("Procedures evaluated different rows")
    runtime["total_s"] = perf_counter() - t0
    return frames, {
        "inputs": {**identity, "tables_sha256": before},
        "design": {
            "outer": {"source": "accepted shared Stage 2/3 producer folds (policy nesting)",
                      "fingerprint": fold_fingerprint(outer),
                      "rows": np.bincount(outer).tolist(),
                      "positives": np.bincount(outer, weights=y).tolist()},
            "matched_counts": {f"{g:g}": [ceil(Fraction(str(g)) * int(m)) for m in np.bincount(outer)]
                               for g in policy["coverage"]["grid"]},   # stage4_policy.matched_risks
            "inner": {"n_splits": INNER_FOLDS, "seed": INNER_SEED,
                      "construction": "shared.stage3.data.FoldPlan.build on each outer training partition",
                      "fingerprints": {str(k): fold_fingerprint(v) for k, v in inner.items()}},
            "producer_folds_unchanged": {u: {c: fold_fingerprint(t[c]) for c in ("stage1_fold", "stage2_fold",
                                                                                 "stage3_fold")}
                                         for u, t in tables.items()},
            "rules": {"historical": "class decision as PR 39 (vote / F2 grid); cut by shared.stage4."
                                    "protocol.tune_min_confidence on pooled inner predictions",
                      "registered": "class thresholds 0.10, 0.20, 1/3, 0.50 (probability outputs); cut by "
                                    "stage4_policy.operating_cut on pooled inner predictions"},
            "optimism": "loss: outer achieved minus inner selected (positive = inner optimistic); "
                        "f2: inner minus outer; aggregated with outer row weights",
        },
        "reference_reproduction": reproduced,
        "meta_search": {"sampler": f"optuna RandomSampler(seed={SAMPLER_SEED}), sequential, no pruning",
                        "unique_proposals": len(proposals), "attempts": attempts,
                        "budget": f"{PROPOSALS} unique or {ATTEMPTS} attempts; same proposals in every "
                                  "partition (procedures.*.meta_xgb.candidates)",
                        "objective": "mean inner-fold F2 at 0.5 (historical); inner log-loss recorded "
                                     "for Step 4B, not used"},
        "procedures": out,
        "predictions": {
            "file": PREDICTIONS, "storage": "numpy savez_compressed beside this record, git-ignored",
            "rows_sha256": _array_digest(pd.util.hash_pandas_object(rows).to_numpy()),
            "layout": "rows in Step 2A table order; <upstream>.<procedure>.outer.<score> (row,), one per "
                      "procedure for every rule; <upstream>.<procedure>.inner.<score> (candidate, outer fold, "
                      "row), NaN outside the outer training partition; inner_fold -1 on outer rows",
            "arrays": {key: {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": _array_digest(a)}
                       for key, a in arrays.items()},
            "step_4b": STEP_4B_REUSE},
        "claim": "Nested Stage 4 evaluation on fixed upstream OOF inputs (random split, training rows).",
        "limitations": list(LIMITATIONS),
        "reproduce": "python -m feature_research.stage4_nested",
    }, {"seconds": runtime, "arrays": arrays}


def summarize(evidence):
    """
    One row per upstream, procedure and rule: pooled outer results and selection optimism.
    ``risk_at_g`` needs the exact count in every fold; ``exact_count_folds`` gives, per grid
    level, the folds where no confidence tie spans it (brackets are in the record).
    """
    out = []
    for u, procs in evidence["procedures"].items():
        for name, proc in procs.items():
            for rule, res in proc["rules"].items():
                p = res["pooled"] or {}
                o = p.get("outer", {})
                out.append({"upstream": u, "procedure": name, "rule": rule,
                            "inner_loss": p.get("inner_loss_row_weighted"), "outer_loss": o.get("loss"),
                            "loss_optimism": p.get("loss_optimism_row_weighted"),
                            "outer_loss_fold_sd": p.get("outer_loss_folds_sd"),
                            "coverage": o.get("coverage"),
                            "coverage_shift": p.get("coverage_shift_row_weighted"),
                            "effective_recall": o.get("effective_recall"),
                            "positive_retention": o.get("positive_retention"),
                            "negative_retention": o.get("negative_retention"), "retained_f2": o.get("f2"),
                            "partial_aurc": p.get("partial_aurc_row_weighted"),
                            **{f"risk_at_{g}": v for g, v in p.get("matched_risk_row_weighted", {}).items()},
                            "exact_count_folds": "/".join(
                                map(str, p.get("matched_available_folds", {}).values()))})
    return pd.DataFrame(out).set_index(["upstream", "procedure", "rule"])


def load_predictions(run_id, root=LOGS, index=None, *, record_name="nested_references"):
    """A run's stored scores, each checked against its digest in the tracked record."""
    rec = records.read_run_record(run_id, record_name, root=root)
    man = rec["predictions"]
    with np.load(Path(root) / run_id / man["file"], allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    bad = sorted(set(arrays) ^ set(man["arrays"])) or sorted(
        key for key, a in arrays.items() if _array_digest(a) != man["arrays"][key]["sha256"])
    if index is not None and _array_digest(pd.util.hash_pandas_object(index).to_numpy()) != man["rows_sha256"]:
        bad.append("row order")
    if bad:
        raise ValueError(f"Stored predictions do not match the record: {bad[:5]}")
    return rec, arrays


def write_predictions(run, name, evidence, arrays, root=LOGS):
    paths = [records.write_run_record(run, name, evidence), run.folder / PREDICTIONS]
    with open(paths[1], "xb") as fh:
        np.savez_compressed(fh, **arrays)
    load_predictions(run.run_id, root, record_name=name)
    return paths


def record_nested(evidence, local, root=LOGS):
    """The tracked record, then the local score store read back against it, then runtime."""
    seeds = {"inner_folds": INNER_SEED, "random_sampler": SAMPLER_SEED, "meta_xgb": 42}
    run = records.start_run(commit=2, seeds=seeds, notebook=NOTEBOOK, root=root, code_files=CODE_FILES)
    paths = write_predictions(run, "nested_references", evidence, local["arrays"], root)
    stored = {"bytes": paths[1].stat().st_size, "sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest()}
    paths.append(records.write_run_record(run, "nested_runtime", {
        "seconds": local["seconds"], "threads": 1, "predictions_file": stored}))
    return tuple(paths)


if __name__ == "__main__":
    from feature_research.training_inputs import frozen_inputs
    from feature_research.training_side import draw_training_side

    oof, step2a = frozen_inputs(draw_training_side(indices_only=True))
    _, nested, local = nested_references(oof, step2a)
    print(summarize(nested).round(4).to_string())
    print(*record_nested(nested, local), sep="\n")
