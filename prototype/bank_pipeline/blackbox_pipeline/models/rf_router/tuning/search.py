"""
blackbox_pipeline.models.rf_router.tuning.search
=================================================
The tuning harness: one search routine, one budget, every arm.

``tune``                   one Optuna study for one (objective, pass, population)
``headroom_probe``         random configurations scored on BOTH passes — a
                           SCREEN for whether a specialized objective might find
                           something AUC-selection misses
``summarize_headroom``     correlation and regret of AUC-selection per pass,
                           classified as insufficient / limited / promising
``noise_floor``            one configuration re-scored on fresh fold/forest seeds
``fresh_fold_evaluation``  same, for a single pass — estimates the fold-assignment
                           part of a tuned ``best_value``'s selection optimism
``assert_same_budget``     refuse to compare arms tuned under different budgets
``assert_matched_control`` the AUC control and the recall arm differ in objective
                           and nothing else
``tuning_result_validator`` reject a cached result that does not match the inputs

All of it reads training data only. None of it accepts a test split.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..constraints import OperatingConstraints
from .budget import TuningBudget
from .diagnostics import DegenerateObjectiveError, diagnose_pass2_degeneracy
from .objectives import OBJECTIVES, evaluate_params
from .provenance import TUNING_CONTRACT_VERSION, IncompatibleCacheError, data_fingerprint
from .space import REQUIRED_PARAM_KEYS, suggest_rf_params
from .stage import Pass1Stage

__all__ = [
    "TuningResult",
    "tune",
    "headroom_probe",
    "summarize_headroom",
    "noise_floor",
    "fresh_fold_evaluation",
    "assert_same_budget",
    "assert_matched_control",
    "tuning_result_validator",
    "HEADROOM_INSUFFICIENT",
    "HEADROOM_LIMITED",
    "HEADROOM_PROMISING",
    "classify_headroom",
]

#: Seed offset for fresh-fold re-evaluation; keeps those folds disjoint in seed
#: space from the tuning folds and the router's folds.
FRESH_SEED_OFFSET = 1000


@dataclass
class TuningResult:
    """
    One tuning run.

    ``best_value`` is the maximum over trials of an OOF estimate on the training
    labels, so it is optimistic by construction (winner's curse). Fresh-fold
    re-evaluation (``SpecializationPlan.tuning_table``) shows the part of that
    optimism due to fold assignment; the part due to reusing the same labels is
    not removed by any training-side re-scoring. Never report ``best_value`` as
    the arm's performance.

    ``contract_version`` and ``data_fingerprint`` record what produced the
    result, so a cached copy can be checked against the current inputs.
    """

    name: str
    objective: str
    pass_name: str
    params: dict
    best_value: float
    best_trial: int
    trials: pd.DataFrame
    budget: dict
    budget_signature: str
    constraints: dict
    population: dict
    n_infeasible: int
    elapsed_s: float = 0.0
    notes: list = field(default_factory=list)
    contract_version: str = TUNING_CONTRACT_VERSION
    data_fingerprint: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "objective": self.objective,
            "pass": self.pass_name,
            "best_value": self.best_value,
            "best_trial": self.best_trial,
            "n_trials": len(self.trials),
            "n_infeasible": self.n_infeasible,
            "population_rows": self.population.get("n_rows"),
            "remainder_fingerprint": self.population.get("remainder_fingerprint"),
            "budget_signature": self.budget_signature,
            "data_digest": self.data_fingerprint.get("content_digest"),
            **{f"param:{k}": v for k, v in self.params.items()},
        }


def _make_sampler(kind: str, seed: int):
    import optuna

    if kind == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"sampler must be 'tpe' or 'random', got {kind!r}")


def _population(pass_name, X, y, stage: Optional[Pass1Stage], constraints):
    """(mask, n_positives_global, population-info) for one pass."""
    y_arr = np.asarray(y).astype(int)
    if pass_name == "pass1":
        if stage is not None:
            raise ValueError("Pass 1 tunes on the full training split; do not pass a stage")
        return None, None, {"n_rows": int(len(y_arr)), "n_positives": int(y_arr.sum()),
                            "base_rate": float(y_arr.mean())}
    if stage is None:
        raise ValueError("Pass 2 tuning needs the Pass1Stage whose remainder it runs on")
    mask = stage.remainder_mask
    if len(mask) != len(y_arr):
        raise ValueError("stage remainder mask does not match the training population")
    deg = diagnose_pass2_degeneracy(y_arr[mask], constraints, int(y_arr.sum()))
    info = {
        "n_rows": int(mask.sum()),
        "n_positives": int(y_arr[mask].sum()),
        "base_rate": float(y_arr[mask].mean()),
        "remainder_fingerprint": stage.remainder_fingerprint,
        "pass1_params": dict(stage.params),
        "router_folds": asdict(stage.folds),
        "degeneracy": deg.to_dict(),
    }
    return mask, int(y_arr.sum()), info


def tune(
    objective: str,
    pass_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    constraints: OperatingConstraints,
    budget: TuningBudget,
    stage: Optional[Pass1Stage] = None,
    name: Optional[str] = None,
    sampler: str = "tpe",
    allow_degenerate: bool = False,
    verbose: bool = True,
) -> TuningResult:
    """
    Run one Optuna study maximising ``objective`` for ``pass_name``.

    Pass 1 searches on the full training split. Pass 2 searches on
    ``stage.remainder_mask`` — the remainder the deployed router will train Pass 2
    on for ``stage.params`` — and refuses to start when that remainder is
    degenerate (``DegenerateObjectiveError``), unless ``allow_degenerate``.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}")
    spec = OBJECTIVES[objective]
    if pass_name not in spec.passes:
        raise ValueError(f"objective {objective!r} does not apply to {pass_name}")

    mask, npos, population = _population(pass_name, X, y, stage, constraints)
    if pass_name == "pass2" and population["degeneracy"]["degenerate"] and not allow_degenerate:
        raise DegenerateObjectiveError(population["degeneracy"]["verdict"])

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=_make_sampler(sampler, budget.sampler_seed))

    def _objective(trial):
        params = suggest_rf_params(trial, fixed=budget.fixed)
        m = evaluate_params(
            params, X, y, pass_name=pass_name, constraints=constraints,
            n_folds=budget.n_folds, fold_seed=budget.fold_seed,
            forest_seed=budget.forest_seed, stratify=budget.stratify,
            population_mask=mask, n_positives_global=npos,
            forest_n_jobs=budget.forest_n_jobs,
        )
        trial.set_user_attr("params", params)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return float(m[spec.key])

    label = name or f"{pass_name}:{objective}"
    if verbose:
        print(f"  tuning {label}: {budget.n_trials} trials on "
              f"{population['n_rows']:,} rows (budget {budget.signature()})")
    t0 = time.time()
    study.optimize(_objective, n_trials=budget.n_trials, show_progress_bar=verbose)
    elapsed = time.time() - t0

    rows = []
    for t in study.trials:
        attrs = dict(t.user_attrs)
        params = attrs.pop("params", {})
        rows.append({"trial": t.number, "value": t.value,
                     **{f"param:{k}": v for k, v in params.items()}, **attrs})
    trials = pd.DataFrame(rows)

    feasible_col = "p1_feasible" if pass_name == "pass1" else "p2_feasible"
    n_infeasible = int((~trials[feasible_col].astype(bool)).sum())
    best = study.best_trial
    result = TuningResult(
        name=label, objective=objective, pass_name=pass_name,
        params=dict(best.user_attrs["params"]), best_value=float(best.value),
        best_trial=int(best.number), trials=trials, budget=budget.to_dict(),
        budget_signature=budget.signature(), constraints=constraints.to_dict(),
        population=population, n_infeasible=n_infeasible, elapsed_s=elapsed,
        contract_version=TUNING_CONTRACT_VERSION,
        data_fingerprint=data_fingerprint(X, y),
    )
    if n_infeasible:
        result.notes.append(f"{n_infeasible}/{len(trials)} trials infeasible under the constraints")
    if verbose:
        print(f"    best {spec.key} = {result.best_value:.4f} (trial {result.best_trial}, "
              f"{n_infeasible} infeasible, {elapsed:.0f}s)")
    return result


def fresh_fold_evaluation(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    pass_name: str,
    constraints: OperatingConstraints,
    budget: TuningBudget,
    stage: Optional[Pass1Stage] = None,
    seeds: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    """
    Re-score one configuration on fresh fold AND forest seeds.

    The seeds are offset by ``FRESH_SEED_OFFSET`` so they never coincide with the
    tuning folds or the router's folds. The spread across seeds is the
    training-side noise floor. For a tuned configuration, ``best_value`` minus
    the mean here estimates only the FOLD-ASSIGNMENT part of its selection
    optimism: the rows and labels are the same ones the search selected on, so
    this is a lower bound on the total optimism, not a correction for it.
    """
    mask, npos, _ = _population(pass_name, X, y, stage, constraints)
    rows = []
    for s in seeds:
        m = evaluate_params(
            params, X, y, pass_name=pass_name, constraints=constraints,
            n_folds=budget.n_folds, fold_seed=budget.fold_seed + FRESH_SEED_OFFSET + s,
            forest_seed=budget.forest_seed + FRESH_SEED_OFFSET + s,
            stratify=budget.stratify, population_mask=mask, n_positives_global=npos,
            forest_n_jobs=budget.forest_n_jobs,
        )
        rows.append({"seed": s, **m})
    return pd.DataFrame(rows)


def noise_floor(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    constraints: OperatingConstraints,
    budget: TuningBudget,
    stage: Pass1Stage,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """
    Training-side noise of both pass objectives for one fixed configuration.

    One row per seed: ``p1_coverage``/``p1_auc`` on the full split and
    ``p2_recall``/``rem_auc`` on ``stage``'s remainder (NaN when that remainder
    is degenerate). The per-column standard deviation is what an arm-to-arm
    difference must clear before it counts.
    """
    p1 = fresh_fold_evaluation(params, X, y, pass_name="pass1", constraints=constraints,
                               budget=budget, seeds=seeds)
    out = pd.DataFrame({"seed": p1["seed"], "p1_coverage": p1["p1_coverage"],
                        "p1_auc": p1["auc_fold_mean"]})
    y_arr = np.asarray(y).astype(int)
    deg = diagnose_pass2_degeneracy(y_arr[stage.remainder_mask], constraints, int(y_arr.sum()))
    if deg.degenerate:
        out["p2_recall"], out["rem_auc"] = np.nan, np.nan
    else:
        p2 = fresh_fold_evaluation(params, X, y, pass_name="pass2", constraints=constraints,
                                   budget=budget, stage=stage, seeds=seeds)
        out["p2_recall"] = p2["p2_recall"].to_numpy()
        out["rem_auc"] = p2["auc_fold_mean"].to_numpy()
    return out


def headroom_probe(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    constraints: OperatingConstraints,
    budget: TuningBudget,
    stage: Pass1Stage,
    n_configs: int = 40,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score ``n_configs`` RANDOM configurations on both passes.

    Random, not TPE, so the sample describes the search landscape rather than
    concentrating where one objective happens to point. Pass 2 is scored on
    ``stage``'s remainder (normally the current baseline's, via
    ``Pass1Stage.from_router``) and skipped when that remainder is degenerate.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=_make_sampler("random", budget.sampler_seed + 10_000))
    y_arr = np.asarray(y).astype(int)
    deg = diagnose_pass2_degeneracy(y_arr[stage.remainder_mask], constraints, int(y_arr.sum()))
    common = dict(constraints=constraints, n_folds=budget.n_folds,
                  fold_seed=budget.fold_seed, forest_seed=budget.forest_seed,
                  stratify=budget.stratify, forest_n_jobs=budget.forest_n_jobs)
    rows = []
    t0 = time.time()
    for i in range(n_configs):
        trial = study.ask()
        params = suggest_rf_params(trial, fixed=budget.fixed)
        m1 = evaluate_params(params, X, y, pass_name="pass1", **common)
        row = {"config": i, **{f"param:{k}": v for k, v in params.items()},
               "p1_auc": m1["auc_fold_mean"], "p1_coverage": m1["p1_coverage"],
               "p1_feasible": m1["p1_feasible"], "p1_pauc_leak": m1["p1_pauc_leak"]}
        if deg.degenerate:
            row.update(rem_auc=np.nan, p2_recall=np.nan, p2_feasible=False)
        else:
            m2 = evaluate_params(params, X, y, pass_name="pass2",
                                 population_mask=stage.remainder_mask,
                                 n_positives_global=int(y_arr.sum()), **common)
            row.update(rem_auc=m2["auc_fold_mean"], p2_recall=m2["p2_recall"],
                       p2_feasible=m2["p2_feasible"])
        rows.append(row)
        study.tell(trial, 0.0)
        if verbose:
            print(f"   probe {i + 1}/{n_configs}  p1_auc {row['p1_auc']:.4f}  "
                  f"cov {row['p1_coverage']:.4f}  p2_recall {row['p2_recall']:.4f}  "
                  f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


HEADROOM_INSUFFICIENT = "INSUFFICIENT EVIDENCE"
HEADROOM_LIMITED = "LIMITED OBSERVED HEADROOM"
HEADROOM_PROMISING = "PROMISING HEADROOM"

#: Below these the screen does not classify headroom at all.
MIN_PROBE_CONFIGS = 10
MIN_NOISE_SEEDS = 3


def classify_headroom(
    regret: float,
    noise_sd: float,
    *,
    n_configs: int,
    n_noise: int,
    multiplier: float = 2.0,
    min_configs: int = MIN_PROBE_CONFIGS,
    min_noise_seeds: int = MIN_NOISE_SEEDS,
) -> Tuple[str, str]:
    """
    Screening label and the evidence behind it, for one probe question.

    This is a SCREEN, not a test. A finite random probe samples a small part of
    the search space, and both ``regret`` (a maximum over noisy estimates) and
    the best-AUC pick are noisy, so:

    * ``INSUFFICIENT EVIDENCE`` — too few configurations or noise seeds to scale
      the regret at all;
    * ``LIMITED OBSERVED HEADROOM`` — in the configurations sampled, AUC-selection
      came within ``multiplier`` × noise SD of the best observed objective. The
      probe found no gain distinguishable from training noise. That does NOT
      show specialization cannot help — the full equal-budget tuning, or a
      larger probe, could still find one;
    * ``PROMISING HEADROOM`` — some sampled configuration beat AUC-selection by
      at least ``multiplier`` × noise SD. Regret is biased upward, so this
      justifies running the equal-budget tuning; it is not an effect estimate.
    """
    if n_configs < min_configs:
        return HEADROOM_INSUFFICIENT, (
            f"only {n_configs} configurations scored (screen needs >= {min_configs})")
    if not np.isfinite(noise_sd) or n_noise < min_noise_seeds:
        return HEADROOM_INSUFFICIENT, (
            f"no usable noise floor ({n_noise} seeds; screen needs >= {min_noise_seeds})")
    if noise_sd <= 0:
        return HEADROOM_INSUFFICIENT, "noise SD is zero, so regret cannot be scaled"
    ratio = regret / noise_sd
    if ratio < multiplier:
        return HEADROOM_LIMITED, (
            f"in {n_configs} random configurations, AUC-selection came within "
            f"{ratio:.1f}× noise SD of the best observed objective — no gain "
            "distinguishable from noise in this sample; not evidence that "
            "specialization cannot help")
    return HEADROOM_PROMISING, (
        f"a sampled configuration beat AUC-selection by {ratio:.1f}× noise SD; "
        "regret is an upward-biased maximum, so treat this as a reason to run "
        "the equal-budget tuning, not as an effect size")


def summarize_headroom(probe: pd.DataFrame, noise: Optional[pd.DataFrame] = None,
                       noise_multiplier: float = 2.0, *,
                       min_configs: int = MIN_PROBE_CONFIGS,
                       min_noise_seeds: int = MIN_NOISE_SEEDS) -> pd.DataFrame:
    """
    Screen: would a specialized objective plausibly pick a better configuration
    than AUC-selection does?

    For each question: Spearman correlation between the AUC the baseline selects
    on and the pass objective, the REGRET of AUC-selection (best objective in the
    probe minus the objective of the probe's best-AUC configuration), regret in
    units of the training noise SD, and ``classify_headroom``'s label and
    evidence text. Labels describe what this probe observed; none of them is a
    conclusion about specialization in general.
    """
    questions = [
        ("Pass 1 coverage vs full-split AUC", "p1_auc", "p1_coverage"),
        ("Pass 2 recall vs full-split AUC (inherit)", "p1_auc", "p2_recall"),
        ("Pass 2 recall vs remainder AUC (control)", "rem_auc", "p2_recall"),
    ]
    rows = []
    for label, sel, obj in questions:
        d = probe[[sel, obj]].dropna() if {sel, obj} <= set(probe) else probe.iloc[0:0]
        n_noise = int(noise[obj].notna().sum()) if noise is not None and obj in noise else 0
        sd = float(noise[obj].std(ddof=1)) if n_noise > 1 else float("nan")
        if len(d) < 2:
            screening, evidence = HEADROOM_INSUFFICIENT, (
                f"{len(d)} usable configurations (Pass 2 degenerate or not scored)")
            rows.append({"question": label, "n": len(d), "spearman": np.nan,
                         "best objective": np.nan, "objective at best AUC": np.nan,
                         "regret": np.nan, "noise SD": sd, "regret / noise SD": np.nan,
                         "screening": screening, "evidence": evidence})
            continue
        best = float(d[obj].max())
        at_sel = float(d.loc[d[sel].idxmax(), obj])
        regret = best - at_sel
        screening, evidence = classify_headroom(
            regret, sd, n_configs=len(d), n_noise=n_noise, multiplier=noise_multiplier,
            min_configs=min_configs, min_noise_seeds=min_noise_seeds)
        rows.append({
            "question": label, "n": len(d),
            "spearman": float(d[sel].corr(d[obj], method="spearman")),
            "best objective": best, "objective at best AUC": at_sel,
            "regret": regret, "noise SD": sd,
            "regret / noise SD": regret / sd if np.isfinite(sd) and sd > 0 else np.nan,
            "screening": screening, "evidence": evidence,
        })
    return pd.DataFrame(rows).set_index("question")


def assert_same_budget(*results: TuningResult) -> None:
    """Raise unless every result shares one budget signature and one constraint set."""
    results = [r for r in results if r is not None]
    if not results:
        return
    sigs = {r.budget_signature for r in results}
    cons = {repr(sorted(r.constraints.items())) for r in results}
    if len(sigs) > 1 or len(cons) > 1:
        detail = "\n".join(f"  {r.name}: budget {r.budget_signature}" for r in results)
        raise ValueError(
            "Arms were tuned under different budgets or constraints — their "
            f"comparison would confound the objective with the budget:\n{detail}"
        )


def assert_matched_control(control: TuningResult, treatment: TuningResult) -> None:
    """
    Raise unless ``control`` and ``treatment`` differ in their objective ONLY.

    For the Pass 2 control (AUC on B's remainder) against the Pass 2 recall arm,
    that means: same pass, same frozen remainder and the Pass 1 parameters that
    produced it, same budget signature, the same number of trials actually run,
    same constraints, same training data. Anything else would let a population
    or budget difference masquerade as an objective effect.
    """
    problems = []
    if control.objective == treatment.objective:
        problems.append("objectives are identical — nothing is being controlled for")
    for label, a, b in (
        ("pass", control.pass_name, treatment.pass_name),
        ("budget signature", control.budget_signature, treatment.budget_signature),
        ("trials run", len(control.trials), len(treatment.trials)),
        ("constraints", control.constraints, treatment.constraints),
        ("remainder", control.population.get("remainder_fingerprint"),
         treatment.population.get("remainder_fingerprint")),
        ("Pass 1 params behind the remainder", control.population.get("pass1_params"),
         treatment.population.get("pass1_params")),
        ("population rows", control.population.get("n_rows"),
         treatment.population.get("n_rows")),
        ("training data", control.data_fingerprint.get("content_digest"),
         treatment.data_fingerprint.get("content_digest")),
    ):
        if a != b:
            problems.append(f"{label}: {a!r} vs {b!r}")
    if problems:
        raise ValueError(f"{control.name} is not a matched control for {treatment.name}: "
                         + "; ".join(problems))


def tuning_result_validator(
    *,
    objective: str,
    pass_name: str,
    budget: TuningBudget,
    constraints: OperatingConstraints,
    data_fp: dict,
    stage: Optional[Pass1Stage] = None,
):
    """
    Validator for a cached ``TuningResult`` (see ``provenance.cached``).

    Checks the result's own record of what produced it against the current
    inputs, independently of the cache key: contract version, training-data
    content digest, objective and pass, budget signature and trial count,
    constraints, parameter schema and fixed parameters, and — for Pass 2 — the
    remainder and the Pass 1 parameters behind it.
    """
    def validate(r) -> None:
        if not isinstance(r, TuningResult):
            raise IncompatibleCacheError(f"expected a TuningResult, found {type(r).__name__}")
        problems = []
        checks = [
            ("contract version", r.contract_version, TUNING_CONTRACT_VERSION),
            ("training data", r.data_fingerprint.get("content_digest"),
             data_fp.get("content_digest")),
            ("objective", r.objective, objective),
            ("pass", r.pass_name, pass_name),
            ("budget signature", r.budget_signature, budget.signature()),
            ("trials run", len(r.trials), budget.n_trials),
            ("constraints", r.constraints, constraints.to_dict()),
            ("parameter keys", sorted(r.params), sorted(REQUIRED_PARAM_KEYS)),
        ]
        checks += [(f"fixed {k}", r.params.get(k), v) for k, v in budget.fixed.items()]
        if pass_name == "pass2":
            if stage is None:
                raise ValueError("a Pass 2 validator needs the stage it was tuned on")
            checks += [
                ("remainder", r.population.get("remainder_fingerprint"),
                 stage.remainder_fingerprint),
                ("Pass 1 params behind the remainder", r.population.get("pass1_params"),
                 dict(stage.params)),
            ]
        for label, got, want in checks:
            if got != want:
                problems.append(f"{label}: cached {got!r} vs current {want!r}")
        if problems:
            raise IncompatibleCacheError(
                f"cached tuning result {r.name!r} does not match the current inputs — "
                + "; ".join(problems))

    return validate
