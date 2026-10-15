"""
blackbox_pipeline.models.rf_router.tuning.ablation
===================================================
The baseline-versus-specialized ablation: arms, tuning order, coupling checks,
the effect decomposition and the pre-registered decision rule.

Arms (shared constraints, split, router folds, metrics, tuning budget)
----------------------------------------------------------------------
=====  ===========================  =====================================
A0     current baseline             ``RF_PARAMS`` both passes (no re-tune)
B      AUC on full split            inherit (both passes share B's params)
C      B's Pass 1                   AUC re-tuned on B's remainder (control)
D      B's Pass 1                   recall@precision on B's remainder
E      coverage@constraints         recall@precision on E's remainder
=====  ===========================  =====================================

Five tuning runs. Pass 1 decisions depend on Pass 1 parameters alone, so the
Pass 1 objective effect is read from E vs B without a sixth arm.

Decomposition
-------------
``Re-tuning harness``    B − A0   what re-tuning in the new harness alone does
``Pass 1 objective``     E − B    pass1_share at matched constraints
``Pass 2 population``    C − B    tuning Pass 2 on its own remainder at all
``Pass 2 objective``     D − C    THE research question for Pass 2
``Cascade interaction``  E − D    Pass 2 recall moved by E's different remainder

C and D are tuned on the SAME frozen remainder with the same budget, trial count
and data; ``search.assert_matched_control`` enforces it, so D − C isolates the
objective. C − B isolates the population. E − D carries Pass 1's influence on
Pass 2 through the remainder, separate from either objective effect.

Tuning order (remainder coupling)
---------------------------------
Pass 1 first; its remainder is then rebuilt with the ROUTER'S fold plan
(``stage.pass1_stage``) and frozen; Pass 2 is tuned on that frozen remainder and
the result is stamped with its fingerprint. ``fit_arm`` re-checks the fitted
router's remainder against that stamp, so Pass 2 parameters can never be
deployed behind a Pass 1 they were not tuned behind.

Degeneracy
----------
A remainder whose base rate already clears the precision floor makes every
Pass 2 objective flat (``diagnostics``). Those Pass 2 runs are skipped and
recorded in ``plan.skipped`` — the arms that needed them are dropped (C, D) or
fall back to inherit (E's Pass 2), and the write-up reports the degeneracy as the
finding.

What the evidence can and cannot support
----------------------------------------
* Training-OOF comparisons between arms are DIAGNOSTIC. Tuned arms were selected
  on those labels; a separate tuning fold seed and fresh-fold re-evaluation
  reduce and measure the fold-assignment part of that optimism, but the reuse of
  the same labels is not removed by anything on the training side.
* The test split was examined in earlier research. A comparison on it under
  criteria frozen beforehand (``register_test_evaluation``) is a held-out
  comparison, not untouched confirmatory evidence — and it stops being even that
  if its outcome is used to revise configurations, criteria or arms.
* "No effect detected" means no effect larger than the comparison could resolve,
  given its confidence interval; it is not evidence that no effect exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..constraints import OperatingConstraints
from ..router import RFRouter
from .budget import RouterFolds, TuningBudget
from .diagnostics import Pass2Degeneracy, diagnose_pass2_degeneracy
from .provenance import (
    TUNING_CONTRACT_VERSION,
    cache_key,
    cached,
    data_fingerprint,
    frame_validator,
)
from .search import (
    TuningResult,
    assert_matched_control,
    assert_same_budget,
    fresh_fold_evaluation,
    tune,
    tuning_result_validator,
)
from .stage import Pass1Stage, mask_fingerprint, pass1_stage

__all__ = [
    "ARM_A0", "ARM_B", "ARM_C", "ARM_D", "ARM_E",
    "ArmSpec",
    "SpecializationPlan",
    "DecisionRule",
    "run_specialization_tuning",
    "fit_arm",
    "assert_arm_coupling",
    "decomposition_pairs",
    "specialization_verdict",
    "register_test_evaluation",
]

ARM_A0 = "A0 current baseline"
ARM_B = "B auc | inherit"
ARM_C = "C auc | auc@remainder"
ARM_D = "D auc | recall@remainder"
ARM_E = "E coverage | recall@remainder"

#: (label, new arm, reference arm, metrics, decides?) — order is report order.
DECOMPOSITION: Tuple[Tuple[str, str, str, Tuple[str, ...], bool], ...] = (
    ("Re-tuning harness", ARM_B, ARM_A0, ("pass1_share", "pass2_recall"), False),
    ("Pass 1 objective", ARM_E, ARM_B, ("pass1_share",), True),
    ("Pass 2 population", ARM_C, ARM_B, ("pass2_recall",), True),
    ("Pass 2 objective", ARM_D, ARM_C, ("pass2_recall",), True),
    ("Cascade interaction", ARM_E, ARM_D, ("pass2_recall",), False),
)

#: Training-OOF noise column that measures each test metric's training noise.
_NOISE_COLUMN = {"pass1_share": "p1_coverage", "pass2_recall": "p2_recall"}
_OBJECTIVE_COLUMN = {"auc": "auc_fold_mean", "p1_coverage": "p1_coverage",
                     "p2_recall": "p2_recall"}


@dataclass
class ArmSpec:
    name: str
    pass1_params: dict
    pass2_params: Optional[dict]          # None = Pass 2 inherits Pass 1's params
    pass1_source: str
    pass2_source: str
    pass2_tuned_on: Optional[str] = None  # remainder fingerprint Pass 2 was tuned on


@dataclass
class SpecializationPlan:
    """Everything the tuning phase produced. Training data only."""

    results: Dict[str, TuningResult]
    stages: Dict[str, Pass1Stage]
    degeneracy: Dict[str, Pass2Degeneracy]
    arms: Dict[str, ArmSpec]
    skipped: Dict[str, str] = field(default_factory=dict)
    reevaluation: Dict[str, pd.DataFrame] = field(default_factory=dict)
    controls: Dict[str, str] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def tuning_table(self) -> pd.DataFrame:
        """
        One row per tuning run (training labels only — diagnostic).

        ``fold_optimism`` = ``best_value`` minus the mean of the same configuration
        re-scored on fresh fold/forest seeds. It measures only the part of the
        tuned number owed to the folds it was selected on, so it is a LOWER BOUND
        on selection optimism: the same labels were reused throughout.
        """
        rows = []
        for key, r in self.results.items():
            row = r.summary()
            fresh = self.reevaluation.get(key)
            if fresh is not None:
                col = _OBJECTIVE_COLUMN[r.objective]
                row["fresh_mean"] = float(fresh[col].mean())
                row["fresh_sd"] = float(fresh[col].std(ddof=1)) if len(fresh) > 1 else np.nan
                row["fold_optimism"] = row["best_value"] - row["fresh_mean"]
            rows.append(row)
        return pd.DataFrame(rows).set_index("name")

    def arm_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "arm": a.name, "pass1": a.pass1_source, "pass2": a.pass2_source,
            "pass2 tuned on remainder": a.pass2_tuned_on or "—",
        } for a in self.arms.values()]).set_index("arm")


def run_specialization_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    constraints: OperatingConstraints,
    budget: TuningBudget,
    router_folds: RouterFolds,
    split_fp: str,
    baseline_params: dict,
    baseline_pass2_params: Optional[dict] = None,
    data_provenance: Optional[dict] = None,
    cache_dir: Optional[str | Path] = None,
    reevaluate_seeds: Sequence[int] = (0, 1, 2),
    refresh: bool = False,
    verbose: bool = True,
) -> SpecializationPlan:
    """
    The five tuning runs, in coupling-safe order, with degeneracy skips.

    Training data only. Under ``cache_dir`` each run is cached under a key made of
    the tuning contract version, the training data's fingerprint (ordered schema,
    dtypes, index/value/label digests, ``data_provenance``), the split
    fingerprint, budget signature, constraints, router folds, objective and — for
    Pass 2 — the exact remainder and Pass 1 params. A changed input computes a
    new result; a cached file that fails validation raises
    ``IncompatibleCacheError`` instead of being reused or overwritten.
    """
    cache = Path(cache_dir) if cache_dir is not None else None
    data_fp = data_fingerprint(X, y, provenance=data_provenance)
    common_key = dict(budget=budget.signature(), constraints=constraints.to_dict(),
                      router_folds=asdict(router_folds))

    def run(key, objective, pass_name, stage=None):
        def compute():
            return tune(objective, pass_name, X, y, constraints=constraints,
                        budget=budget, stage=stage, name=key, verbose=verbose)
        if cache is None:
            return compute()
        k = cache_key("tune", data_fp=data_fp, split_fp=split_fp, **common_key,
                      objective=objective, pass_name=pass_name,
                      remainder=None if stage is None else stage.remainder_fingerprint,
                      pass1_params=None if stage is None else stage.params)
        validate = tuning_result_validator(objective=objective, pass_name=pass_name,
                                           budget=budget, constraints=constraints,
                                           data_fp=data_fp, stage=stage)
        return cached(cache, f"tune_{key}", k, compute, validate=validate,
                      refresh=refresh, verbose=verbose)

    y_arr = np.asarray(y).astype(int)
    results: Dict[str, TuningResult] = {}
    skipped: Dict[str, str] = {}
    controls: Dict[str, str] = {}

    # ---- Pass 1: AUC (B, C, D) and coverage@constraints (E) --------------
    results["p1_auc"] = run("p1_auc", "auc", "pass1")
    results["p1_coverage"] = run("p1_coverage", "p1_coverage", "pass1")
    assert_matched_control(results["p1_auc"], results["p1_coverage"])
    controls["Pass 1: auc vs coverage"] = "matched (same data, budget, trials)"

    # ---- remainders, rebuilt with the ROUTER's folds, then frozen ---------
    stages = {
        "B": pass1_stage(X, y, results["p1_auc"].params, constraints, router_folds,
                         forest_n_jobs=budget.forest_n_jobs),
        "E": pass1_stage(X, y, results["p1_coverage"].params, constraints, router_folds,
                         forest_n_jobs=budget.forest_n_jobs),
    }
    degeneracy = {k: diagnose_pass2_degeneracy(y_arr[s.remainder_mask], constraints,
                                               int(y_arr.sum()))
                  for k, s in stages.items()}
    if verbose:
        for k, d in degeneracy.items():
            print(f"  remainder {k}: {stages[k].n_remainder:,} rows, base rate "
                  f"{d.base_rate:.4f} vs floor {d.precision_floor:.4f} → {d.verdict.split(' —')[0]}")

    # ---- Pass 2 on B's remainder: AUC control (C) and recall (D) ----------
    if degeneracy["B"].degenerate:
        skipped[ARM_C] = skipped[ARM_D] = "B's remainder is degenerate: " + degeneracy["B"].verdict
    else:
        results["p2_auc_B"] = run("p2_auc_B", "auc", "pass2", stages["B"])
        results["p2_recall_B"] = run("p2_recall_B", "p2_recall", "pass2", stages["B"])
        assert_matched_control(results["p2_auc_B"], results["p2_recall_B"])
        controls["Pass 2: auc vs recall on B's remainder"] = (
            f"matched (remainder {stages['B'].remainder_fingerprint}, same budget, "
            f"{len(results['p2_auc_B'].trials)} trials each)")

    # ---- Pass 2 on E's remainder: recall (E) --------------------------------
    if degeneracy["E"].degenerate:
        skipped["E pass 2"] = ("E's remainder is degenerate; E's Pass 2 inherits: "
                               + degeneracy["E"].verdict)
    else:
        results["p2_recall_E"] = run("p2_recall_E", "p2_recall", "pass2", stages["E"])

    assert_same_budget(*results.values())

    # ---- arms ----------------------------------------------------------------
    p1a, p1c = results["p1_auc"].params, results["p1_coverage"].params
    arms = {
        ARM_A0: ArmSpec(ARM_A0, dict(baseline_params),
                        None if baseline_pass2_params is None else dict(baseline_pass2_params),
                        "current RF_PARAMS",
                        "inherit" if baseline_pass2_params is None else "RF_PASS2_PARAMS"),
        ARM_B: ArmSpec(ARM_B, p1a, None, "AUC (full split)", "inherit"),
    }
    if "p2_auc_B" in results:
        arms[ARM_C] = ArmSpec(ARM_C, p1a, results["p2_auc_B"].params, "AUC (full split)",
                              "AUC (B remainder)", stages["B"].remainder_fingerprint)
        arms[ARM_D] = ArmSpec(ARM_D, p1a, results["p2_recall_B"].params, "AUC (full split)",
                              "recall@precision (B remainder)",
                              stages["B"].remainder_fingerprint)
    if "p2_recall_E" in results:
        arms[ARM_E] = ArmSpec(ARM_E, p1c, results["p2_recall_E"].params,
                              "coverage@constraints", "recall@precision (E remainder)",
                              stages["E"].remainder_fingerprint)
    else:
        arms[ARM_E] = ArmSpec(ARM_E, p1c, None, "coverage@constraints",
                              "inherit (E remainder degenerate)")

    # ---- fold-assignment optimism: fresh folds for every tuned config -------
    reevaluation: Dict[str, pd.DataFrame] = {}
    if reevaluate_seeds:
        for key, r in results.items():
            stage = None
            if r.pass_name == "pass2":
                stage = stages["E"] if key.endswith("_E") else stages["B"]

            def compute(r=r, stage=stage):
                return fresh_fold_evaluation(r.params, X, y, pass_name=r.pass_name,
                                             constraints=constraints, budget=budget,
                                             stage=stage, seeds=reevaluate_seeds)
            if cache is None:
                reevaluation[key] = compute()
            else:
                k = cache_key("fresh", data_fp=data_fp, split_fp=split_fp, **common_key,
                              objective=r.objective, pass_name=r.pass_name,
                              remainder=None if stage is None else stage.remainder_fingerprint,
                              params=r.params, seeds=list(reevaluate_seeds))
                validate = frame_validator(["seed", _OBJECTIVE_COLUMN[r.objective]],
                                           n_rows=len(reevaluate_seeds))
                reevaluation[key] = cached(cache, f"fresh_{key}", k, compute,
                                           validate=validate, refresh=refresh,
                                           verbose=verbose)

    provenance = {
        "contract_version": TUNING_CONTRACT_VERSION,
        "data_digest": data_fp["digest"],
        "data_content_digest": data_fp["content_digest"],
        "data_provenance": data_fp["provenance"],
        "split_fingerprint": split_fp,
        "budget_signature": budget.signature(),
    }
    return SpecializationPlan(results=results, stages=stages, degeneracy=degeneracy,
                              arms=arms, skipped=skipped, reevaluation=reevaluation,
                              controls=controls, provenance=provenance)


def assert_arm_coupling(router: RFRouter, arm: ArmSpec) -> None:
    """
    The fitted router's remainder must be the one ``arm``'s Pass 2 was tuned on.

    Fails when the router config's folds, seed or constraints differ from the
    tuning run's, or when Pass 2 parameters are paired with a Pass 1 they were
    not tuned behind.
    """
    if arm.pass2_tuned_on is None:
        return
    got = mask_fingerprint(router.remainder_mask_)
    if got != arm.pass2_tuned_on:
        raise ValueError(
            f"Remainder coupling violated for {arm.name!r}: Pass 2 was tuned on "
            f"remainder {arm.pass2_tuned_on} but this router's Pass 1 leaves {got}. "
            "Check that the router uses the same Pass 1 params, n_oof_folds, "
            "random_state and constraints as the tuning run."
        )


def fit_arm(arm: ArmSpec, X: pd.DataFrame, y: pd.Series, split_fp: str,
            base_config) -> RFRouter:
    """
    Fit one arm's router: ``base_config`` with only the RF parameters swapped.

    Constraints, folds, seed and remainder floor all come from ``base_config``
    (normally ``make_router_config(validate_nested=False, verbose=False)``), so
    arms differ in hyperparameters and nothing else.
    """
    cfg = replace(
        base_config,
        rf_params=dict(arm.pass1_params),
        pass2_rf_params=None if arm.pass2_params is None else dict(arm.pass2_params),
        validate_nested=False,
    )
    r = RFRouter(cfg).fit(X, y, split_fp)
    assert_arm_coupling(r, arm)
    return r


def decomposition_pairs(available: Sequence[str]) -> List[Tuple[str, str, str, str]]:
    """``(label, new, ref, metric)`` rows whose arms are all present."""
    have = set(available)
    return [(label, new, ref, m)
            for label, new, ref, metrics, _ in DECOMPOSITION
            if new in have and ref in have for m in metrics]


@dataclass(frozen=True)
class DecisionRule:
    """
    The pre-registered rule. Freeze it with ``register_test_evaluation`` BEFORE
    the test split is evaluated; that call refuses a changed rule afterwards.

    Two separate questions, both required for a recommendation:

    * Statistical improvement — (a) the paired-bootstrap CI excludes zero,
      (b) the effect is at least ``max(min_effect, noise_multiplier × training
      noise SD)``, (c) at least ``min_finite_boot`` of the bootstrap replicates
      were defined, and (d) both arms' constraint evaluations are valid.
    * Eligibility for adoption — the new arm satisfies EVERY applicable hard
      constraint on the evaluation split (``evaluation.paired.constraint_slippage``
      ``feasible``). This is absolute: it does not depend on the reference arm,
      so an arm that violates a constraint is never recommended, however the
      control fares.

    Ties go to the simpler shared-objective design.
    """

    min_effect: float = 0.01
    noise_multiplier: float = 2.0
    min_finite_boot: float = 0.95


def register_test_evaluation(spec_dir: str | Path, rule: DecisionRule, *,
                             arms: Sequence[str], split_fp: str,
                             note: str = "") -> int:
    """
    Freeze the decision rule on disk and log this test evaluation.

    The first call writes ``decision_rule.json``; every later call must pass the
    identical rule or it raises — criteria cannot be revised after the numbers
    have been seen. Each call appends to ``test_evaluations.jsonl`` and returns
    how many evaluations were logged before it, so repeated looks at the test
    split are visible in the write-up.
    """
    spec_dir = Path(spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    rule_path = spec_dir / "decision_rule.json"
    current = {"rule": asdict(rule), "contract_version": TUNING_CONTRACT_VERSION}
    if rule_path.exists():
        frozen = json.loads(rule_path.read_text())
        if frozen.get("rule") != current["rule"]:
            raise ValueError(
                f"Decision rule changed after it was frozen in {rule_path}:\n"
                f"  frozen : {frozen.get('rule')}\n  current: {current['rule']}\n"
                "Pre-registered criteria must not be revised after a test evaluation. "
                "Restore the frozen rule, or start a new, separately reported "
                "experiment directory."
            )
    else:
        rule_path.write_text(json.dumps(
            {**current, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            indent=2))

    log_path = spec_dir / "test_evaluations.jsonl"
    prior = len(log_path.read_text().splitlines()) if log_path.exists() else 0
    with log_path.open("a") as fh:
        fh.write(json.dumps({
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "arms": list(arms), "split_fingerprint": split_fp,
            "rule": asdict(rule), "note": note,
        }) + "\n")
    return prior


def _is_true(v) -> bool:
    """Strictly True (Python or numpy bool); NaN, missing or anything else is False."""
    return isinstance(v, (bool, np.bool_)) and bool(v)


def _slippage_value(slippage: pd.DataFrame, row: str, arm: str):
    try:
        return slippage.loc[row, arm]
    except KeyError:
        return np.nan


def _arm_constraints(slippage: pd.DataFrame, arm: str) -> Tuple[bool, bool, str]:
    """
    (valid, feasible, detail) for one arm's ``slippage_table`` column.

    An arm missing from the table, or whose column lacks a boolean ``feasible``
    or ``evaluation_valid``, counts as an invalid evaluation — eligibility that
    was never evaluated is never assumed.
    """
    valid = _is_true(_slippage_value(slippage, "evaluation_valid", arm))
    feas = _slippage_value(slippage, "feasible", arm)
    if not valid or not isinstance(feas, (bool, np.bool_)):
        undefined = _slippage_value(slippage, "undefined", arm)
        detail = undefined if isinstance(undefined, str) and undefined else "not evaluated"
        return False, False, detail
    violated = _slippage_value(slippage, "violated", arm)
    return True, bool(feas), violated if isinstance(violated, str) else ""


def specialization_verdict(
    boot: pd.DataFrame,
    slippage: pd.DataFrame,
    *,
    noise: Optional[pd.DataFrame] = None,
    rule: DecisionRule = DecisionRule(),
) -> pd.DataFrame:
    """
    Apply ``rule`` to every decomposition row present in ``boot``.

    Parameters
    ----------
    boot
        ``evaluation.paired.paired_bootstrap`` output over ``decomposition_pairs``.
    slippage
        ``evaluation.paired.slippage_table`` over the same arms (test metrics).
    noise
        ``search.noise_floor`` output (training OOF); without it the threshold is
        ``rule.min_effect`` alone, which the output says.

    Returns
    -------
    pandas.DataFrame
        Per comparison: the effect and its CI, ``statistical`` (improvement /
        no effect detected / worse / insufficient bootstrap / not assessed),
        ``eligibility`` of the NEW arm (eligible / ineligible: <violated> /
        invalid: <undefined>), ``reference feasible`` for context, and the
        combined ``verdict``. Only "EFFECT OBSERVED — …" recommends the new arm,
        and it requires both a statistical improvement and an eligible arm.

    Verdicts describe this held-out comparison under the frozen rule. They are
    not confirmatory: the test split was examined in earlier research.
    """
    decides = {label: d for label, _, _, _, d in DECOMPOSITION}
    rows = []
    for _, b in boot.iterrows():
        col = _NOISE_COLUMN.get(b["metric"])
        sd = (float(noise[col].std(ddof=1))
              if noise is not None and col in noise and noise[col].notna().sum() > 1
              else float("nan"))
        threshold = rule.min_effect if np.isnan(sd) else max(rule.min_effect,
                                                             rule.noise_multiplier * sd)
        valid_new, feasible_new, detail_new = _arm_constraints(slippage, b["new"])
        valid_ref, feasible_ref, _ = _arm_constraints(slippage, b["reference"])
        requested = b.get("n_boot_requested", b.get("n_boot", np.nan))
        finite_share = (float(b["n_boot"]) / float(requested)
                        if requested and np.isfinite(requested) else np.nan)
        ci_low, ci_high, delta = b["ci_low"], b["ci_high"], b["delta"]

        # ---- statistical improvement (relative to the reference) ------------
        if not (valid_new and valid_ref):
            statistical = "not assessed (invalid evaluation)"
        elif not np.isfinite(finite_share) or finite_share < rule.min_finite_boot:
            statistical = "insufficient bootstrap support"
        elif np.isfinite(ci_low) and ci_low > 0 and delta >= threshold:
            statistical = "improvement"
        elif np.isfinite(ci_high) and ci_high < 0:
            statistical = "worse"
        else:
            statistical = "no effect detected"

        # ---- eligibility for adoption (absolute, new arm only) --------------
        if not valid_new:
            eligibility = f"invalid: {detail_new} undefined"
        elif feasible_new:
            eligibility = "eligible"
        else:
            eligibility = f"ineligible: violates {detail_new}"

        # ---- combined verdict -----------------------------------------------
        if not (valid_new and valid_ref):
            bad = [a for a, ok in ((b["new"], valid_new), (b["reference"], valid_ref)) if not ok]
            verdict = ("INVALID EVALUATION — a critical constraint is undefined for "
                       f"{', '.join(bad)}; cannot be accepted")
        elif statistical == "insufficient bootstrap support":
            verdict = "INSUFFICIENT BOOTSTRAP SUPPORT — metric undefined in too many replicates"
        elif not decides.get(b["comparison"], False):
            verdict = "descriptive"
        elif statistical == "improvement" and feasible_new:
            verdict = ("EFFECT OBSERVED — meets the pre-registered rule and the arm satisfies "
                       "every hard constraint (held-out, not confirmatory)")
        elif statistical == "improvement":
            verdict = (f"EFFECT OBSERVED BUT NOT ELIGIBLE — new arm violates {detail_new}; "
                       "cannot be recommended")
        elif statistical == "worse":
            verdict = "WORSE than reference on this comparison"
        else:
            verdict = ("NO EFFECT DETECTED at this resolution — retain the simpler design "
                       "(not evidence of no effect)")
        rows.append({
            "comparison": b["comparison"], "metric": b["metric"],
            "new": b["new"], "reference": b["reference"],
            "delta": delta, "ci_low": ci_low, "ci_high": ci_high,
            "ci_halfwidth": (ci_high - ci_low) / 2 if np.isfinite(ci_high - ci_low) else np.nan,
            "threshold": threshold,
            "threshold basis": "min_effect" if np.isnan(sd) else
            f"max(min_effect, {rule.noise_multiplier:g}×train SD {sd:.4f})",
            "statistical": statistical,
            "eligibility": eligibility,
            "reference feasible": feasible_ref if valid_ref else None,
            "slippage new": float(_slippage_value(slippage, "max_violation", b["new"])),
            "slippage ref": float(_slippage_value(slippage, "max_violation", b["reference"])),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)
