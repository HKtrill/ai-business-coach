"""
PR 35 commit B — objective-specialization correctness tests.

The guarantees under test:

1. The tuning objective IS the deployed decision rule: for fixed params and the
   router's folds, ``p1_coverage``/``p2_recall`` equal the router's own operating
   point exactly.
2. The remainder Pass 2 is tuned on IS the remainder the router trains Pass 2 on,
   and a mismatch between the two is caught at fit time (remainder coupling).
3. Pass 2 trials fit only remainder rows, each held out exactly once (OOF).
4. Degeneracy is diagnosed exactly and blocks Pass 2 tuning before any trial.
5. Arms share one budget; a mixed-budget comparison is refused.
6. Re-solving a two-pass router cannot silently orphan Pass 2.
7. The paired bootstrap reproduces ``evaluate_router`` point estimates.
8. Cached tuning outputs are reused only for identical inputs (data schema,
   dtypes, values, labels, provenance, contract version, configuration) and an
   incompatible artifact raises instead of being reused.
9. Gate 1 reports what a probe observed — insufficient / limited / promising —
   never a conclusive null.
10. Every configured constraint is evaluated (disabled, inapplicable and
    undefined handled explicitly); an arm is recommended only if it shows a
    statistical improvement AND satisfies every applicable hard constraint,
    independently of the control.
11. Pass agreement follows the router's cut semantics exactly, with
    deterministic, tie-aware rate matching.
12. The AUC control is matched to the recall arm; the decision rule is frozen
    before the test evaluation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from blackbox_pipeline.models.rf_router import OperatingConstraints, RFRouter
from blackbox_pipeline.models.rf_router.evaluation import (
    BOOT_METRICS,
    CONSTRAINT_FIELDS,
    NON_OUTCOME_FIELDS,
    compare_routers,
    constraint_slippage,
    evaluate_router,
    paired_bootstrap,
    slippage_table,
)
from blackbox_pipeline.models.rf_router.threshhold import ThresholdSolver
from blackbox_pipeline.models.rf_router.reporting import pass_agreement
from blackbox_pipeline.models.rf_router.tuning import (
    ARM_A0, ARM_B, ARM_C, ARM_D, ARM_E,
    HEADROOM_INSUFFICIENT,
    HEADROOM_LIMITED,
    HEADROOM_PROMISING,
    ArmSpec,
    DecisionRule,
    DegenerateObjectiveError,
    IncompatibleCacheError,
    assert_matched_control,
    cache_key,
    cached,
    classify_headroom,
    data_fingerprint,
    frame_validator,
    register_test_evaluation,
    tuning_result_validator,
    Pass1Stage,
    RouterFolds,
    TuningBudget,
    TuningResult,
    assert_same_budget,
    cv_auc_legacy,
    decomposition_pairs,
    diagnose_pass2_degeneracy,
    diagnose_router,
    evaluate_params,
    fit_arm,
    headroom_probe,
    noise_floor,
    pass1_stage,
    run_specialization_tuning,
    specialization_verdict,
    summarize_headroom,
    tune,
)
from blackbox_pipeline.models.rf_router.tuning import ablation as ablation_mod
from blackbox_pipeline.models.rf_router.tuning import objectives as objectives_mod
from blackbox_pipeline.models.rf_router.tuning import provenance as provenance_mod

from tests.support.stage2_synthetic import (
    ALT_PARAMS, CONSTRAINTS_DEGENERATE, CONSTRAINTS_OK, N_FOLDS, ROUTER_SEED,
    SMALL_PARAMS, make_data, router_config,
)

TINY_BUDGET = TuningBudget(n_trials=2, sampler_seed=0, n_folds=N_FOLDS,
                           fold_seed=ROUTER_SEED + 1, forest_seed=ROUTER_SEED,
                           fixed_params={"n_estimators": 12})


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def router(data):
    X, y = data
    return RFRouter(router_config()).fit(X, y, "fp-test")


@pytest.fixture(scope="module")
def stage(router):
    return Pass1Stage.from_router(router)


# ----------------------------------------------------------------------
# 1. objective == deployed rule
# ----------------------------------------------------------------------
def test_objective_equals_router_operating_point(router, stage, data):
    X, y = data
    common = dict(constraints=CONSTRAINTS_OK, n_folds=N_FOLDS,
                  fold_seed=ROUTER_SEED, forest_seed=ROUTER_SEED)
    m1 = evaluate_params(SMALL_PARAMS, X, y, pass_name="pass1", **common)
    assert m1["p1_coverage"] == router.thresholds_.pass1.band_fraction
    assert m1["p1_npv"] == pytest.approx(router.thresholds_.pass1.band_precision)

    m2 = evaluate_params(SMALL_PARAMS, X, y, pass_name="pass2",
                         population_mask=stage.remainder_mask, **common)
    assert m2["p2_recall"] == router.thresholds_.pass2.recall
    assert m2["p2_precision"] == pytest.approx(router.thresholds_.pass2.band_precision)
    assert m2["n_rows"] == int(router.remainder_mask_.sum())


def test_cv_auc_legacy_matches_harness_auc(data):
    """The baseline objective (former Cell 22) and the harness's 'auc' agree."""
    X, y = data
    legacy = cv_auc_legacy(X, y, SMALL_PARAMS, random_state=11, n_splits=5)
    harness = evaluate_params(SMALL_PARAMS, X, y, pass_name="pass1",
                              constraints=CONSTRAINTS_OK, n_folds=5, fold_seed=11,
                              forest_seed=11, forest_n_jobs=1)["auc_fold_mean"]
    assert harness == pytest.approx(legacy, abs=1e-12)


# ----------------------------------------------------------------------
# 2. remainder construction and coupling
# ----------------------------------------------------------------------
def test_pass1_stage_reproduces_router_remainder(router, stage, data):
    X, y = data
    rebuilt = pass1_stage(X, y, SMALL_PARAMS, CONSTRAINTS_OK,
                          RouterFolds.from_config(router.config))
    np.testing.assert_array_equal(rebuilt.remainder_mask, router.remainder_mask_)
    np.testing.assert_array_equal(rebuilt.fold_id, router.oof_pass1_fold_id_)
    assert rebuilt.remainder_fingerprint == stage.remainder_fingerprint
    assert rebuilt.t1.threshold == pytest.approx(router.thresholds_.t1, abs=1e-12)


def test_fit_arm_enforces_remainder_coupling(router, stage, data):
    X, y = data
    base = router_config()
    ok = ArmSpec("ok", SMALL_PARAMS, ALT_PARAMS, "p1", "p2", stage.remainder_fingerprint)
    fitted = fit_arm(ok, X, y, "fp-test", base)
    assert fitted.config.effective_pass2_params == ALT_PARAMS

    other = pass1_stage(X, y, ALT_PARAMS, CONSTRAINTS_OK, RouterFolds.from_config(base))
    assert other.remainder_fingerprint != stage.remainder_fingerprint, "fixture too weak"
    mismatched = ArmSpec("bad", ALT_PARAMS, SMALL_PARAMS, "p1", "p2",
                         stage.remainder_fingerprint)
    with pytest.raises(ValueError, match="Remainder coupling violated"):
        fit_arm(mismatched, X, y, "fp-test", base)


# ----------------------------------------------------------------------
# 3. Pass 2 trials are OOF within the remainder
# ----------------------------------------------------------------------
def test_pass2_trials_fit_only_remainder_rows_each_held_out_once(monkeypatch, stage, data):
    X, y = data
    fitted_on = []

    class SpyTrainer(objectives_mod.ForestTrainer):
        def fit(self, X_, y_, random_state=None):
            fitted_on.append(set(X_.index))
            return super().fit(X_, y_, random_state)

    monkeypatch.setattr(objectives_mod, "ForestTrainer", SpyTrainer)
    evaluate_params(SMALL_PARAMS, X, y, pass_name="pass2", constraints=CONSTRAINTS_OK,
                    n_folds=N_FOLDS, fold_seed=99, forest_seed=99,
                    population_mask=stage.remainder_mask)

    rem_labels = set(X.index[stage.remainder_mask])
    assert len(fitted_on) == N_FOLDS
    assert all(s <= rem_labels for s in fitted_on)
    counts = pd.Series(0, index=sorted(rem_labels))
    for s in fitted_on:
        counts[list(s)] += 1
    assert (counts == N_FOLDS - 1).all(), "every remainder row must be held out exactly once"


def test_evaluate_params_rejects_wrong_population(stage, data):
    X, y = data
    with pytest.raises(ValueError):
        evaluate_params(SMALL_PARAMS, X, y, pass_name="pass2", constraints=CONSTRAINTS_OK,
                        n_folds=N_FOLDS, fold_seed=1, forest_seed=1)
    with pytest.raises(ValueError):
        evaluate_params(SMALL_PARAMS, X, y, pass_name="pass1", constraints=CONSTRAINTS_OK,
                        n_folds=N_FOLDS, fold_seed=1, forest_seed=1,
                        population_mask=stage.remainder_mask)


# ----------------------------------------------------------------------
# 4. degeneracy
# ----------------------------------------------------------------------
def test_degeneracy_diagnosis_is_exact_against_the_solver(router):
    y = router._y_train_
    rem = router.remainder_mask_
    p2, y2 = router.oof_pass2_proba_[rem], y[rem]
    base = y2.mean()
    for floor in np.linspace(base - 0.1, base + 0.1, 21):
        c = OperatingConstraints(**{**CONSTRAINTS_OK.to_dict(),
                                    "min_precision_subscribe": float(np.clip(floor, 0, 1))})
        r = ThresholdSolver(c).solve_pass2(p2, y2, n_positives_global=int(y.sum()))
        whole = r.feasible and r.band_size == len(p2)
        assert diagnose_pass2_degeneracy(y2, c, int(y.sum())).degenerate == whole


def test_degenerate_remainder_blocks_pass2_tuning(data):
    X, y = data
    r = RFRouter(router_config(constraints=CONSTRAINTS_DEGENERATE)).fit(X, y, "fp")
    gate = diagnose_router(r)
    assert gate["pass2 degenerate"] and gate["pass2 band fraction (of remainder)"] == 1.0
    assert gate["verdict"].startswith("DEGENERATE")
    with pytest.raises(DegenerateObjectiveError):
        tune("p2_recall", "pass2", X, y, constraints=CONSTRAINTS_DEGENERATE,
             budget=TINY_BUDGET, stage=Pass1Stage.from_router(r), verbose=False)


def test_gate0_on_a_healthy_router(router):
    gate = diagnose_router(router)
    assert not gate["pass2 degenerate"]
    assert gate["pass2 band fraction (of remainder)"] < 1.0
    assert gate["verdict"].startswith(("OK", "NEAR-DEGENERATE"))


# ----------------------------------------------------------------------
# 5. tuning harness, budgets, infeasibility
# ----------------------------------------------------------------------
def test_tune_respects_budget_and_records_every_diagnostic(data):
    X, y = data
    res = tune("p1_coverage", "pass1", X, y, constraints=CONSTRAINTS_OK,
               budget=TINY_BUDGET, verbose=False)
    assert len(res.trials) == TINY_BUDGET.n_trials
    assert res.params["n_estimators"] == 12
    assert res.best_value == pytest.approx(res.trials["value"].max())
    assert res.budget_signature == TINY_BUDGET.signature()
    for col in ("auc_fold_mean", "auc_pooled", "p1_coverage", "p1_npv",
                "p1_leakage_rate", "p1_binding", "p1_pauc_leak"):
        assert col in res.trials


def test_infeasible_trials_score_zero_and_are_counted(data):
    X, y = data
    impossible = OperatingConstraints(**{**CONSTRAINTS_OK.to_dict(),
                                         "min_precision_not_subscribe": 0.9999,
                                         "max_subscriber_leakage_rate": 0.0001})
    res = tune("p1_coverage", "pass1", X, y, constraints=impossible,
               budget=TINY_BUDGET, verbose=False)
    assert res.n_infeasible == TINY_BUDGET.n_trials
    assert res.best_value == 0.0


def test_budget_signature_and_parity():
    a = TuningBudget(n_trials=5, forest_n_jobs=1)
    assert a.signature() == TuningBudget(n_trials=5, forest_n_jobs=-1).signature()
    assert a.signature() != TuningBudget(n_trials=6).signature()
    assert TuningBudget(fixed_params={"n_estimators": 9}).fixed == {"n_estimators": 9}
    with pytest.raises(ValueError):
        TuningBudget(fixed_params={"nope": 1})

    def result(budget, constraints=CONSTRAINTS_OK):
        return TuningResult("r", "auc", "pass1", {}, 0.0, 0, pd.DataFrame(),
                            budget.to_dict(), budget.signature(),
                            constraints.to_dict(), {}, 0)

    assert_same_budget(result(a), result(TuningBudget(n_trials=5)))
    with pytest.raises(ValueError):
        assert_same_budget(result(a), result(TuningBudget(n_trials=6)))
    with pytest.raises(ValueError):
        assert_same_budget(result(a), result(a, CONSTRAINTS_DEGENERATE))


# ----------------------------------------------------------------------
# 6. resolve_at cannot orphan Pass 2
# ----------------------------------------------------------------------
def test_resolve_at_refuses_to_move_the_remainder(data):
    X, y = data
    r = RFRouter(router_config()).fit(X, y, "fp")
    tighter_leak = OperatingConstraints(**{**CONSTRAINTS_OK.to_dict(),
                                           "max_subscriber_leakage_rate": 0.05})
    with pytest.raises(ValueError, match="remainder boundary"):
        r.resolve_at(tighter_leak)

    pass2_only = OperatingConstraints(**{**CONSTRAINTS_OK.to_dict(),
                                         "min_precision_subscribe": 0.55})
    pair = r.resolve_at(pass2_only, apply=True)
    assert pair.t1 == pytest.approx(r.fit_report_.thresholds["t1"])
    assert r.fit_report_.thresholds["t2"] == pytest.approx(pair.t2)
    assert r.config.constraints == pass2_only and r.solver.constraints == pass2_only


# ----------------------------------------------------------------------
# 7. paired bootstrap
# ----------------------------------------------------------------------
def test_paired_bootstrap_point_estimates_match_evaluate_router(router, data):
    X, y = data
    d_a = router.predict(X)[2]
    d_b = RFRouter(router_config(constraints=CONSTRAINTS_DEGENERATE)).fit(X, y, "fp").predict(X)[2]
    m_a = evaluate_router(*router.predict(X), y, label="a")
    pairs = [("self", "a", "a", m) for m in BOOT_METRICS] + [("ab", "b", "a", "pass2_recall")]
    boot = paired_bootstrap({"a": d_a, "b": d_b}, y, pairs=pairs, n_boot=300, seed=1)

    for _, row in boot[boot["comparison"] == "self"].iterrows():
        assert row["new_value"] == pytest.approx(m_a[row["metric"]], nan_ok=True)
        assert row["delta"] == 0.0 and row["ci_low"] == 0.0 and row["ci_high"] == 0.0
        assert row["n_boot_requested"] == 300
    ab = boot[boot["comparison"] == "ab"].iloc[0]
    assert ab["ci_low"] <= ab["delta"] <= ab["ci_high"]


def test_paired_bootstrap_reports_undefined_replicates():
    """An empty Pass 2 band makes precision undefined; n_boot shows it."""
    y = np.array([0, 1] * 50)
    dec_none = np.array(["pass1"] * 100, dtype=object)          # Pass 2 never flags
    boot = paired_bootstrap({"a": dec_none, "b": dec_none}, y,
                            pairs=[("p", "a", "b", "pass2_precision")], n_boot=50, seed=0)
    row = boot.iloc[0]
    assert row["n_boot"] == 0 and row["n_boot_requested"] == 50
    assert np.isnan(row["ci_low"]) and np.isnan(row["new_value"])


# ----------------------------------------------------------------------
# 10. hard-constraint evaluation: every configured constraint
# ----------------------------------------------------------------------
# A consistent evaluate_router-shaped result on a 1,000-row split, evaluated
# against constraints solved on a 4,000-row reference population (800 positives).
REF = dict(reference_rows=4000, reference_positives=800)


def _metrics(**kw):
    base = dict(n_samples=1000, n_positives=200,
                pass1_n=500, pass1_npv=0.96, pass1_subscribers_lost=20, pass1_leakage_rate=0.10,
                pass2_n=200, pass2_precision=0.60, pass2_positives_found=120, pass2_recall=0.60)
    base.update(kw)
    return base


def _cons(**kw):
    return OperatingConstraints(**{**CONSTRAINTS_OK.to_dict(), **kw})


def _slip_of(metrics=None, constraints=CONSTRAINTS_OK, **ref):
    return constraint_slippage(_metrics() if metrics is None else metrics, constraints,
                               **{**REF, **ref})


def test_every_operating_constraint_field_is_accounted_for():
    from dataclasses import fields as dc_fields
    configured = {f.name for f in dc_fields(OperatingConstraints)}
    assert configured == set(CONSTRAINT_FIELDS.values()) | set(NON_OUTCOME_FIELDS)
    s = _slip_of()
    assert {f"{k}_status" for k in CONSTRAINT_FIELDS} <= set(s)
    assert {f"{k}_margin" for k in CONSTRAINT_FIELDS} <= set(s)


def test_all_satisfied_is_feasible_and_disabled_constraints_are_listed():
    s = _slip_of()
    assert s["feasible"] and s["evaluation_valid"] and s["max_violation"] == 0.0
    assert s["n_violated"] == 0 and s["violated"] == "" and s["undefined"] == ""
    assert set(s["disabled"].split(", ")) == {"pass1_coverage", "pass2_recall"}
    assert s["pass1_coverage_status"] == s["pass2_recall_status"] == "disabled"
    assert np.isnan(s["pass1_coverage_margin"])


@pytest.mark.parametrize("name, metrics, constraints, margin", [
    ("npv", _metrics(pass1_npv=0.80), CONSTRAINTS_OK, 0.06),
    ("leakage", _metrics(pass1_leakage_rate=0.40), CONSTRAINTS_OK, 0.10),
    # cap 100 of 800 reference positives -> 12.5% allowed; 30 / 200 = 15% lost
    ("leakage_absolute", _metrics(pass1_subscribers_lost=30), _cons(max_subscriber_leakage_absolute=100), 0.025),
    ("pass1_coverage", _metrics(), _cons(min_pass1_coverage=0.6), 0.10),
    # 2,500 of 4,000 reference rows -> 62.5% required; Pass 1 holds 50%
    ("pass1_support", _metrics(), _cons(min_band_support_pass1=2500), 0.125),
    ("precision", _metrics(pass2_precision=0.45), CONSTRAINTS_OK, 0.05),
    ("pass2_recall", _metrics(), _cons(min_recall_subscribe=0.70), 0.10),
    ("pass2_support", _metrics(), _cons(min_band_support_pass2=1000), 0.05),
])
def test_each_constraint_is_enforced(name, metrics, constraints, margin):
    s = _slip_of(metrics, constraints)
    assert s[f"{name}_status"] == "violated" and s["violated"] == name
    assert s["evaluation_valid"] and not s["feasible"] and s["n_violated"] == 1
    assert s[f"{name}_margin"] == pytest.approx(margin)
    assert s["max_violation"] == pytest.approx(margin)


def test_recall_floor_uses_the_configured_denominator():
    # global recall 120 / 200 = 0.60; remainder recall 120 / (200 - 20) = 0.667
    glob = _slip_of(constraints=_cons(min_recall_subscribe=0.65, recall_denominator="global"))
    rem = _slip_of(constraints=_cons(min_recall_subscribe=0.65, recall_denominator="remainder"))
    assert glob["pass2_recall_status"] == "violated"
    assert rem["pass2_recall_status"] == "ok"
    assert rem["pass2_recall_margin"] == pytest.approx(0.65 - 120 / 180)


@pytest.mark.parametrize("name, constraints, metrics, ref", [
    # exactly on each boundary is ok (solver convention), one step past is not
    ("pass1_support", _cons(min_band_support_pass1=2000), _metrics(), {}),       # 2000/4000 == 500/1000
    ("pass2_support", _cons(min_band_support_pass2=800), _metrics(), {}),        # 800/4000 == 200/1000
    ("leakage_absolute", _cons(max_subscriber_leakage_absolute=80), _metrics(), {}),  # 80/800 == 20/200
    ("pass1_coverage", _cons(min_pass1_coverage=0.5), _metrics(), {}),
    ("npv", _cons(min_precision_not_subscribe=0.96), _metrics(), {}),
    ("precision", _cons(min_precision_subscribe=0.60), _metrics(), {}),
    ("leakage", _cons(max_subscriber_leakage_rate=0.10), _metrics(), {}),
    ("pass2_recall", _cons(min_recall_subscribe=0.60), _metrics(), {}),
])
def test_boundaries_are_satisfied_exactly(name, constraints, metrics, ref):
    s = _slip_of(metrics, constraints, **ref)
    assert s[f"{name}_status"] == "ok", s
    assert s["feasible"]


@pytest.mark.parametrize("name, constraints", [
    ("pass1_support", _cons(min_band_support_pass1=2001)),
    ("pass2_support", _cons(min_band_support_pass2=801)),
    ("leakage_absolute", _cons(max_subscriber_leakage_absolute=79)),
])
def test_one_count_past_a_boundary_is_violated(name, constraints):
    s = _slip_of(constraints=constraints)
    assert s[f"{name}_status"] == "violated" and not s["feasible"]


def test_count_constraints_on_the_reference_population_match_the_solver():
    # evaluated on the population it was solved on, support is the raw count check
    m = _metrics(n_samples=4000, n_positives=800, pass1_n=125, pass2_n=75,
                 pass1_subscribers_lost=5, pass2_positives_found=50)
    c = _cons(min_band_support_pass1=125, min_band_support_pass2=75,
              max_subscriber_leakage_absolute=5)
    assert _slip_of(m, c)["feasible"]
    for field_, key, n in (("min_band_support_pass1", "pass1_support", 126),
                           ("min_band_support_pass2", "pass2_support", 76),
                           ("max_subscriber_leakage_absolute", "leakage_absolute", 4)):
        assert _slip_of(m, _cons(**{**c.to_dict(), field_: n}))[f"{key}_status"] == "violated"


def test_empty_band_is_inapplicable_for_precision_but_violates_support():
    s = _slip_of(_metrics(pass2_n=0, pass2_precision=float("nan"),
                          pass2_positives_found=0, pass2_recall=0.0))
    assert s["precision_status"] == "empty_band" and np.isnan(s["precision_margin"])
    assert s["pass2_support_status"] == "violated"
    assert s["evaluation_valid"] and not s["feasible"]

    s1 = _slip_of(_metrics(pass1_n=0, pass1_npv=float("nan"), pass1_subscribers_lost=0,
                           pass1_leakage_rate=0.0))
    assert s1["npv_status"] == "empty_band" and s1["leakage_status"] == "ok"
    assert s1["pass1_support_status"] == "violated" and not s1["feasible"]


@pytest.mark.parametrize("override, constraints, field", [
    (dict(pass1_npv=float("nan")), CONSTRAINTS_OK, "npv"),            # band non-empty, NaN
    (dict(pass2_precision=float("nan")), CONSTRAINTS_OK, "precision"),
    (dict(n_positives=0, pass1_leakage_rate=0.0), CONSTRAINTS_OK, "leakage"),  # 0.0 is not a pass
    (dict(pass1_leakage_rate=float("inf")), CONSTRAINTS_OK, "leakage"),
    (dict(pass2_n=None), CONSTRAINTS_OK, "precision"),                 # band size unknown
    (dict(pass2_n=None), CONSTRAINTS_OK, "pass2_support"),
    (dict(pass1_subscribers_lost=None), CONSTRAINTS_OK, "leakage_absolute"),
    (dict(pass1_n=12.5), CONSTRAINTS_OK, "pass1_support"),             # not a count
    (dict(n_samples=None), _cons(min_pass1_coverage=0.1), "pass1_coverage"),
    (dict(pass2_recall=float("nan")), _cons(min_recall_subscribe=0.1), "pass2_recall"),
    (dict(n_positives=0), _cons(min_recall_subscribe=0.1), "pass2_recall"),
    # remainder denominator with every positive routed by Pass 1: nothing to recall
    (dict(pass1_subscribers_lost=200), _cons(min_recall_subscribe=0.1,
                                             recall_denominator="remainder"), "pass2_recall"),
])
def test_undefined_metrics_invalidate(override, constraints, field):
    s = _slip_of(_metrics(**override), constraints)
    assert s[f"{field}_status"] == "undefined" and field in s["undefined"].split(", ")
    assert not s["evaluation_valid"] and not s["feasible"] and np.isnan(s["max_violation"])


def test_disabled_constraints_never_become_undefined():
    """A disabled constraint stays disabled even when its metric is missing."""
    s = _slip_of(_metrics(pass2_recall=float("nan")))
    assert s["pass2_recall_status"] == "disabled" and s["feasible"]


def test_missing_keys_are_undefined_not_feasible():
    """Regression: a NaN metric used to drop out of max_violation and look feasible."""
    s = constraint_slippage({"pass1_npv": float("nan"), "pass1_leakage_rate": 0.1,
                             "pass2_precision": 0.6}, CONSTRAINTS_OK, **REF)
    assert not s["evaluation_valid"] and not s["feasible"]
    assert set(s["undefined"].split(", ")) == {
        "npv", "leakage", "leakage_absolute", "pass1_support", "precision", "pass2_support"}


def test_reference_population_is_required_and_validated():
    with pytest.raises(TypeError):
        constraint_slippage(_metrics(), CONSTRAINTS_OK)
    with pytest.raises(ValueError):
        constraint_slippage(_metrics(), CONSTRAINTS_OK, reference_rows=0, reference_positives=1)


def test_slippage_on_the_solve_population_agrees_with_the_solver(router):
    """The router's own training-OOF routing satisfies every constraint it was solved under."""
    y = router._y_train_
    m = evaluate_router(*router.predict_train_oof(), y, label="oof")
    s = constraint_slippage(m, router.config.constraints, reference_rows=len(y),
                            reference_positives=int(y.sum()))
    assert s["evaluation_valid"] and s["feasible"], s
    tighter = _cons(min_precision_subscribe=min(1.0, m["pass2_precision"] + 0.01))
    assert constraint_slippage(m, tighter, reference_rows=len(y),
                               reference_positives=int(y.sum()))["precision_status"] == "violated"


def test_slippage_table_over_compare_routers_frame():
    frame = compare_routers({"label": "ok", **_metrics()},
                            {"label": "bad", **_metrics(pass1_npv=float("nan"))})
    table = slippage_table(frame, CONSTRAINTS_OK, **REF)
    assert list(table.columns) == ["ok", "bad"]
    assert bool(table.loc["feasible", "ok"]) and bool(table.loc["evaluation_valid", "ok"])
    assert not table.loc["evaluation_valid", "bad"] and not table.loc["feasible", "bad"]


# ----------------------------------------------------------------------
# 10b. verdict: statistical improvement vs eligibility for adoption
# ----------------------------------------------------------------------
ARMS = (ARM_A0, ARM_B, ARM_C, ARM_D, ARM_E)


def _slip(invalid=(), violations=None):
    """A slippage table; ``violations`` maps arm -> (constraint names, max margin)."""
    violations = violations or {}
    cols = {}
    for a in ARMS:
        names, margin = violations.get(a, ((), 0.0))
        valid = a not in invalid
        cols[a] = {"feasible": valid and not names, "evaluation_valid": valid,
                   "max_violation": margin if valid else float("nan"),
                   "violated": ", ".join(names), "undefined": "" if valid else "npv"}
    return pd.DataFrame(cols)


BOOT = pd.DataFrame([
    dict(comparison="Pass 2 objective", new=ARM_D, reference=ARM_C, metric="pass2_recall",
         delta=0.04, ci_low=0.01, ci_high=0.07, n_boot=2000, n_boot_requested=2000),
    dict(comparison="Pass 1 objective", new=ARM_E, reference=ARM_B, metric="pass1_share",
         delta=0.004, ci_low=-0.01, ci_high=0.02, n_boot=2000, n_boot_requested=2000),
    dict(comparison="Cascade interaction", new=ARM_E, reference=ARM_D, metric="pass2_recall",
         delta=0.02, ci_low=0.0, ci_high=0.04, n_boot=2000, n_boot_requested=2000),
])
NOISE = pd.DataFrame({"p1_coverage": [0.30, 0.31, 0.30], "p2_recall": [0.50, 0.505, 0.50]})


def _verdict_frame(boot=BOOT, slip=None, rule=DecisionRule()):
    return specialization_verdict(boot, _slip() if slip is None else slip,
                                  noise=NOISE, rule=rule).set_index("comparison")


def _verdicts(boot=BOOT, slip=None, rule=DecisionRule()):
    return _verdict_frame(boot, slip, rule)["verdict"]


def test_fully_feasible_improvement_is_recommended():
    f = _verdict_frame()
    row = f.loc["Pass 2 objective"]
    assert row["statistical"] == "improvement" and row["eligibility"] == "eligible"
    assert row["verdict"].startswith("EFFECT OBSERVED —")
    assert "every hard constraint" in row["verdict"] and "not confirmatory" in row["verdict"]
    assert f.loc["Pass 1 objective", "verdict"].startswith("NO EFFECT DETECTED")
    assert "not evidence of no effect" in f.loc["Pass 1 objective", "verdict"]
    assert f.loc["Cascade interaction", "verdict"] == "descriptive"


def test_equal_violations_do_not_permit_adoption():
    """Regression: equal control/treatment violations used to pass the relative check."""
    slip = _slip(violations={ARM_D: (("npv",), 0.02), ARM_C: (("npv",), 0.02)})
    row = _verdict_frame(slip=slip).loc["Pass 2 objective"]
    assert row["statistical"] == "improvement"
    assert row["eligibility"] == "ineligible: violates npv"
    assert row["verdict"].startswith("EFFECT OBSERVED BUT NOT ELIGIBLE")
    assert row["reference feasible"] is False or row["reference feasible"] == False  # noqa: E712


@pytest.mark.parametrize("new_v, ref_v", [
    ((("leakage",), 0.05), (("leakage",), 0.01)),     # new violates more
    ((("leakage",), 0.01), (("leakage",), 0.05)),     # new violates LESS — still ineligible
    ((("pass2_support",), 0.01), ((), 0.0)),          # only a newly evaluated constraint
])
def test_unequal_violations_never_make_a_violating_arm_eligible(new_v, ref_v):
    slip = _slip(violations={ARM_D: new_v, ARM_C: ref_v})
    row = _verdict_frame(slip=slip).loc["Pass 2 objective"]
    assert row["statistical"] == "improvement"
    assert row["eligibility"].startswith("ineligible")
    assert row["verdict"].startswith("EFFECT OBSERVED BUT NOT ELIGIBLE")


def test_feasible_arm_against_an_infeasible_control_is_judged_on_its_own():
    slip = _slip(violations={ARM_C: (("npv", "leakage"), 0.03)})
    row = _verdict_frame(slip=slip).loc["Pass 2 objective"]
    assert row["eligibility"] == "eligible" and not row["reference feasible"]
    assert row["verdict"].startswith("EFFECT OBSERVED —")


def test_eligibility_is_reported_even_without_an_effect():
    row = _verdict_frame(slip=_slip(violations={ARM_E: (("npv",), 0.01)})).loc["Pass 1 objective"]
    assert row["statistical"] == "no effect detected"
    assert row["eligibility"] == "ineligible: violates npv"
    assert row["verdict"].startswith("NO EFFECT DETECTED")


def test_stricter_rule_raises_the_statistical_bar():
    row = _verdict_frame(rule=DecisionRule(min_effect=0.05)).loc["Pass 2 objective"]
    assert row["statistical"] == "no effect detected" and row["eligibility"] == "eligible"


def test_verdict_never_accepts_an_invalid_evaluation():
    v = _verdicts(slip=_slip(invalid=(ARM_D,)))
    assert v["Pass 2 objective"].startswith("INVALID EVALUATION") and ARM_D in v["Pass 2 objective"]
    assert _verdicts(slip=_slip(invalid=(ARM_C,)))["Pass 2 objective"].startswith("INVALID")
    missing_arm = _slip().drop(columns=ARM_D)
    assert _verdicts(slip=missing_arm)["Pass 2 objective"].startswith("INVALID EVALUATION")
    no_feasible_row = _slip().drop(index="feasible")   # eligibility never evaluated
    row = _verdict_frame(slip=no_feasible_row).loc["Pass 2 objective"]
    assert row["verdict"].startswith("INVALID EVALUATION")
    assert row["eligibility"].startswith("invalid")


def test_verdict_requires_bootstrap_support():
    thin = BOOT.copy()
    thin.loc[0, "n_boot"] = 1500                   # 75% of replicates defined
    assert _verdicts(boot=thin)["Pass 2 objective"].startswith("INSUFFICIENT BOOTSTRAP")


def test_verdict_end_to_end_from_router_metrics():
    """slippage_table -> verdict: a violation in a newly evaluated constraint blocks adoption."""
    feasible, short_band = _metrics(), _metrics(pass2_n=20, pass2_positives_found=12,
                                                pass2_recall=0.06)
    frame = compare_routers({"label": ARM_C, **feasible}, {"label": ARM_D, **short_band})
    slip = slippage_table(frame, _cons(min_band_support_pass2=200), **REF)  # 5% share required
    assert slip.loc["pass2_support_status", ARM_D] == "violated"
    assert slip.loc["violated", ARM_D] == "pass2_support"
    row = specialization_verdict(BOOT.iloc[[0]], slip, noise=NOISE).iloc[0]
    assert row["eligibility"] == "ineligible: violates pass2_support"
    assert row["verdict"].startswith("EFFECT OBSERVED BUT NOT ELIGIBLE")


def test_decomposition_pairs_drop_missing_arms():
    labels = {p[0] for p in decomposition_pairs([ARM_A0, ARM_B, ARM_E])}
    assert labels == {"Re-tuning harness", "Pass 1 objective"}


# ----------------------------------------------------------------------
# 11. pass agreement at the boundaries
# ----------------------------------------------------------------------
def fake_router(p1, p2, t1, t2, remainder=None):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    rem = ~(p1 < t1) if remainder is None else np.asarray(remainder, bool)
    p2_full = np.where(rem, p2, np.nan)
    return SimpleNamespace(thresholds_=SimpleNamespace(t1=t1, t2=t2),
                           oof_pass1_proba_=p1, oof_pass2_proba_=p2_full,
                           remainder_mask_=rem)


def test_pass_agreement_threshold_equality_follows_apply_cuts():
    # p1 == t1 stays in the remainder (Pass 1 routes p1 < t1);
    # p2 == t2 is NOT flagged (Pass 2 flags p2 > t2).
    r = fake_router(p1=[0.1, 0.5, 0.5, 0.7, 0.9], p2=[0.0, 0.4, 0.6, 0.6, 0.8],
                    t1=0.5, t2=0.6)
    out = pass_agreement(r)
    assert out["remainder rows"] == 4
    assert out["pass2 flagged (p2 > t2)"] == 1
    assert out["pass2 scores equal to t2 (not flagged)"] == 2


def test_pass_agreement_exact_count_and_perfect_alignment():
    p = np.linspace(0.3, 0.9, 20)
    out = pass_agreement(fake_router(p1=p, p2=p, t1=0.3, t2=0.75))
    assert out["rate-matched p1 rows"] == out["pass2 flagged (p2 > t2)"]
    assert out["agreement"] == 1.0 and out["overlap with p1 top-k"] == 1.0
    assert out["status"] == "ok"


def test_pass_agreement_ties_are_bracketed_and_deterministic():
    p1 = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
    p2 = np.array([0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1])   # flags rows 0, 1, 3, 6
    r = fake_router(p1, p2, t1=0.0, t2=0.5)
    a, b = pass_agreement(r), pass_agreement(r)
    assert a == b, "same input must give the same answer"
    assert a["p1 ties at the matched cut"] == 4
    lo, hi = a["agreement min over p1 ties"], a["agreement max over p1 ties"]
    assert lo < hi and lo <= a["agreement"] <= hi
    assert a["overlap min over p1 ties"] <= a["overlap with p1 top-k"] <= a["overlap max over p1 ties"]
    # Any row order gives the same bracket (only the tie-break pick moves).
    perm = np.random.default_rng(0).permutation(len(p1))
    c = pass_agreement(fake_router(p1[perm], p2[perm], t1=0.0, t2=0.5))
    assert (c["agreement min over p1 ties"], c["agreement max over p1 ties"]) == (lo, hi)


@pytest.mark.parametrize("t2, status", [(1.0, "flags none"), (-1.0, "flags all")])
def test_pass_agreement_zero_or_full_flagging_is_undefined(t2, status):
    out = pass_agreement(fake_router(p1=[0.6, 0.7, 0.8], p2=[0.2, 0.5, 0.9], t1=0.5, t2=t2))
    assert np.isnan(out["agreement"]) and np.isnan(out["overlap with p1 top-k"])
    assert status in out["status"]


def test_pass_agreement_empty_remainder_and_stale_mask():
    out = pass_agreement(fake_router(p1=[0.1, 0.2], p2=[0.5, 0.5], t1=0.9, t2=0.5))
    assert out["remainder rows"] == 0 and out["status"].startswith("empty remainder")
    assert np.isnan(out["agreement"])
    with pytest.raises(ValueError, match="inconsistent"):
        pass_agreement(fake_router(p1=[0.6, 0.7], p2=[0.5, 0.5], t1=0.5, t2=0.4,
                                   remainder=[True, False]))


def test_pass_agreement_on_a_fitted_router(router):
    out = pass_agreement(router)
    assert out["remainder rows"] == int(router.remainder_mask_.sum())
    if np.isfinite(out["agreement"]):
        assert out["agreement min over p1 ties"] <= out["agreement"] <= out["agreement max over p1 ties"]


# ----------------------------------------------------------------------
# 8. cache provenance
# ----------------------------------------------------------------------
def test_data_fingerprint_detects_feature_block_changes(data):
    X, y = data
    base = data_fingerprint(X, y)
    assert data_fingerprint(X.copy(), y.copy())["digest"] == base["digest"]
    flipped = X.copy()
    flipped["b5"] = 1 - flipped["b5"]
    changes = {
        "values": data_fingerprint(flipped, y),
        "column order": data_fingerprint(X[list(X.columns[::-1])], y),
        "dtype": data_fingerprint(X.astype({"b0": "int8"}), y),
        "labels": data_fingerprint(X, 1 - y),
        "index": data_fingerprint(X.set_axis(X.index + 1), y.set_axis(y.index + 1)),
        "provenance": data_fingerprint(X, y, provenance={"bins": "v2"}),
    }
    for what, fp in changes.items():
        assert fp["digest"] != base["digest"], what
    assert changes["provenance"]["content_digest"] == base["content_digest"]
    assert changes["dtype"]["schema"][0] == ["b0", "int8"]


def test_cached_reuses_identical_inputs_and_recomputes_changed_ones(tmp_path, data, capsys):
    X, y = data
    calls = []
    compute = lambda: calls.append(1) or pd.DataFrame({"seed": [0], "v": [1.0]})  # noqa: E731
    key = cache_key("probe", data_fp=data_fingerprint(X, y), split_fp="s", n=1)
    cached(tmp_path, "probe", key, compute)
    cached(tmp_path, "probe", key, compute)
    assert len(calls) == 1, "identical inputs must reuse"

    flipped = X.copy()
    flipped["b0"] = 1 - flipped["b0"]
    key2 = cache_key("probe", data_fp=data_fingerprint(flipped, y), split_fp="s", n=1)
    cached(tmp_path, "probe", key2, compute)
    assert len(calls) == 2, "a feature change must recompute, not reuse"
    assert "not reused" in capsys.readouterr().out
    assert len(list(tmp_path.glob("probe-*.joblib"))) == 2


def test_contract_version_bump_invalidates(tmp_path, monkeypatch, data):
    X, y = data
    calls = []
    compute = lambda: calls.append(1) or 1.0  # noqa: E731
    fp = data_fingerprint(X, y)
    cached(tmp_path, "r", cache_key("r", data_fp=fp, split_fp="s"), compute, verbose=False)
    monkeypatch.setattr(provenance_mod, "TUNING_CONTRACT_VERSION", "pr35-stage2-tuning-v999")
    cached(tmp_path, "r", cache_key("r", data_fp=fp, split_fp="s"), compute, verbose=False)
    assert len(calls) == 2


def _cache_path(tmp_path, stem, key):
    return tmp_path / f"{stem}-{provenance_mod._digest(key)}.joblib"


def test_incompatible_artifacts_raise_instead_of_being_reused(tmp_path, data):
    X, y = data
    key = cache_key("probe", data_fp=data_fingerprint(X, y), split_fp="s")
    path = _cache_path(tmp_path, "probe", key)

    path.write_bytes(b"not a joblib file")
    with pytest.raises(IncompatibleCacheError, match="unreadable"):
        cached(tmp_path, "probe", key, lambda: 1, verbose=False)

    joblib.dump(["a list, not a cache blob"], path)
    with pytest.raises(IncompatibleCacheError, match="not a PR 35 cache file"):
        cached(tmp_path, "probe", key, lambda: 1, verbose=False)

    joblib.dump({"key": {**key, "split_fp": "other"}, "value": 1}, path)
    with pytest.raises(IncompatibleCacheError, match="different inputs"):
        cached(tmp_path, "probe", key, lambda: 1, verbose=False)

    joblib.dump({"key": key, "value": pd.DataFrame({"x": [1]})}, path)
    with pytest.raises(IncompatibleCacheError, match="lacks columns"):
        cached(tmp_path, "probe", key, lambda: 1, verbose=False,
               validate=frame_validator(["seed"]))

    # refresh=True is the explicit way past an incompatible file
    assert cached(tmp_path, "probe", key, lambda: 7, refresh=True, verbose=False) == 7


def test_tuning_result_validator_rejects_mismatched_results(data):
    X, y = data
    res = tune("p1_coverage", "pass1", X, y, constraints=CONSTRAINTS_OK,
               budget=TINY_BUDGET, verbose=False)
    fp = data_fingerprint(X, y)
    ok = tuning_result_validator(objective="p1_coverage", pass_name="pass1",
                                 budget=TINY_BUDGET, constraints=CONSTRAINTS_OK, data_fp=fp)
    ok(res)
    assert res.data_fingerprint["content_digest"] == fp["content_digest"]

    flipped = X.copy()
    flipped["b1"] = 1 - flipped["b1"]
    for validator, match in (
        (tuning_result_validator(objective="p1_coverage", pass_name="pass1",
                                 budget=TINY_BUDGET, constraints=CONSTRAINTS_OK,
                                 data_fp=data_fingerprint(flipped, y)), "training data"),
        (tuning_result_validator(objective="p1_coverage", pass_name="pass1",
                                 budget=TuningBudget(n_trials=3, n_folds=N_FOLDS,
                                                     fixed_params={"n_estimators": 12}),
                                 constraints=CONSTRAINTS_OK, data_fp=fp), "budget signature"),
        (tuning_result_validator(objective="auc", pass_name="pass1", budget=TINY_BUDGET,
                                 constraints=CONSTRAINTS_OK, data_fp=fp), "objective"),
    ):
        with pytest.raises(IncompatibleCacheError, match=match):
            validator(res)
    with pytest.raises(IncompatibleCacheError, match="TuningResult"):
        ok({"not": "a result"})


# ----------------------------------------------------------------------
# end-to-end (tiny budget): plan, arms, cache, controls, degeneracy skip
# ----------------------------------------------------------------------
def _plan_kwargs(tmp_path, constraints=CONSTRAINTS_OK, **extra):
    return dict(constraints=constraints, budget=TINY_BUDGET,
                router_folds=RouterFolds.from_config(router_config()), split_fp="fp-test",
                baseline_params=SMALL_PARAMS, cache_dir=tmp_path,
                reevaluate_seeds=(0,), verbose=False, **extra)


def test_run_specialization_tuning_end_to_end(tmp_path, monkeypatch, data):
    X, y = data
    plan = run_specialization_tuning(X, y, **_plan_kwargs(tmp_path))

    assert set(plan.arms) == {ARM_A0, ARM_B, ARM_C, ARM_D, ARM_E} and not plan.skipped
    assert plan.arms[ARM_C].pass2_tuned_on == plan.stages["B"].remainder_fingerprint
    assert plan.arms[ARM_E].pass2_tuned_on == plan.stages["E"].remainder_fingerprint
    table = plan.tuning_table()
    assert {"fresh_mean", "fold_optimism", "data_digest"} <= set(table.columns) and len(table) == 5
    assert plan.provenance["data_content_digest"] == data_fingerprint(X, y)["content_digest"]
    assert len(plan.controls) == 2
    for name in (ARM_B, ARM_C, ARM_D, ARM_E):
        fit_arm(plan.arms[name], X, y, "fp-test", router_config())  # coupling holds

    def boom(*a, **k):
        raise AssertionError("tune() called despite a valid cache")
    monkeypatch.setattr(ablation_mod, "tune", boom)
    again = run_specialization_tuning(X, y, **_plan_kwargs(tmp_path))
    assert again.results["p2_recall_B"].params == plan.results["p2_recall_B"].params


def test_feature_change_with_same_split_never_reuses_tuning(tmp_path, monkeypatch, data):
    X, y = data
    run_specialization_tuning(X, y, **_plan_kwargs(tmp_path))
    real_tune, calls = ablation_mod.tune, []

    def counting(*a, **k):
        calls.append(k.get("name"))
        return real_tune(*a, **k)
    monkeypatch.setattr(ablation_mod, "tune", counting)

    rebinned = X.copy()
    rebinned["b2"] = 1 - rebinned["b2"]               # same rows, same split fingerprint
    run_specialization_tuning(rebinned, y, **_plan_kwargs(tmp_path))
    assert set(calls) >= {"p1_auc", "p1_coverage"}, "changed features must retune"

    calls.clear()
    run_specialization_tuning(X, y, **_plan_kwargs(tmp_path, data_provenance={"bins": "v2"}))
    assert set(calls) >= {"p1_auc", "p1_coverage"}, "changed provenance must retune"


def test_tampered_cached_result_raises(tmp_path, data):
    X, y = data
    run_specialization_tuning(X, y, **_plan_kwargs(tmp_path))
    path = next(tmp_path.glob("tune_p1_auc-*.joblib"))
    blob = joblib.load(path)
    blob["value"].budget_signature = "tampered"
    joblib.dump(blob, path)
    with pytest.raises(IncompatibleCacheError, match="budget signature"):
        run_specialization_tuning(X, y, **_plan_kwargs(tmp_path))


def test_auc_control_is_matched_to_the_recall_arm(tmp_path, data):
    X, y = data
    plan = run_specialization_tuning(X, y, **_plan_kwargs(tmp_path))
    c, d = plan.results["p2_auc_B"], plan.results["p2_recall_B"]
    assert c.population["remainder_fingerprint"] == d.population["remainder_fingerprint"] \
        == plan.stages["B"].remainder_fingerprint
    assert c.budget_signature == d.budget_signature and len(c.trials) == len(d.trials)
    assert c.data_fingerprint["content_digest"] == d.data_fingerprint["content_digest"]

    other = plan.results["p2_recall_E"]
    if other.population["remainder_fingerprint"] != c.population["remainder_fingerprint"]:
        with pytest.raises(ValueError, match="remainder"):
            assert_matched_control(c, other)
    short = TuningResult(**{**vars(c), "trials": c.trials.iloc[:1]})
    with pytest.raises(ValueError, match="trials run"):
        assert_matched_control(short, d)
    with pytest.raises(ValueError, match="identical"):
        assert_matched_control(d, d)


def test_run_specialization_tuning_skips_degenerate_pass2(data):
    X, y = data
    plan = run_specialization_tuning(
        X, y, constraints=CONSTRAINTS_DEGENERATE, budget=TINY_BUDGET,
        router_folds=RouterFolds.from_config(router_config()), split_fp="fp",
        baseline_params=SMALL_PARAMS, reevaluate_seeds=(), verbose=False,
    )
    assert ARM_C in plan.skipped and ARM_D in plan.skipped
    assert ARM_C not in plan.arms and ARM_D not in plan.arms
    assert plan.arms[ARM_E].pass2_params is None  # inherits on a degenerate remainder


def test_decision_rule_is_frozen_before_test_evaluation(tmp_path):
    rule = DecisionRule()
    assert register_test_evaluation(tmp_path, rule, arms=["a"], split_fp="s") == 0
    assert register_test_evaluation(tmp_path, rule, arms=["a"], split_fp="s") == 1
    frozen = json.loads((tmp_path / "decision_rule.json").read_text())
    assert frozen["rule"]["min_effect"] == rule.min_effect
    with pytest.raises(ValueError, match="changed after it was frozen"):
        register_test_evaluation(tmp_path, DecisionRule(min_effect=0.005), arms=["a"],
                                 split_fp="s")
    assert len((tmp_path / "test_evaluations.jsonl").read_text().splitlines()) == 2


# ----------------------------------------------------------------------
# 9. Gate 1 headroom screen
# ----------------------------------------------------------------------
@pytest.mark.parametrize("regret, sd, n, n_noise, expected", [
    (0.010, 0.010, 40, 5, HEADROOM_LIMITED),
    (0.050, 0.010, 40, 5, HEADROOM_PROMISING),
    (0.050, 0.010, 4, 5, HEADROOM_INSUFFICIENT),      # too few configurations
    (0.050, 0.010, 40, 2, HEADROOM_INSUFFICIENT),     # too few noise seeds
    (0.050, float("nan"), 40, 5, HEADROOM_INSUFFICIENT),
    (0.050, 0.0, 40, 5, HEADROOM_INSUFFICIENT),
])
def test_classify_headroom(regret, sd, n, n_noise, expected):
    label, evidence = classify_headroom(regret, sd, n_configs=n, n_noise=n_noise)
    assert label == expected and evidence
    if label == HEADROOM_LIMITED:
        assert "not evidence that specialization cannot help" in evidence


def test_summarize_headroom_never_claims_a_conclusive_null():
    rng = np.random.default_rng(0)
    auc = np.linspace(0.70, 0.80, 20)
    probe = pd.DataFrame({"p1_auc": auc, "p1_coverage": 0.4 + rng.normal(0, 0.001, 20),
                          "rem_auc": auc, "p2_recall": 0.7 + 2 * (auc - 0.70)})
    noise = pd.DataFrame({"p1_coverage": [0.40, 0.41, 0.39, 0.40],
                          "p2_recall": [0.70, 0.701, 0.699, 0.70]})
    summary = summarize_headroom(probe, noise)
    assert set(summary["screening"]) <= {HEADROOM_INSUFFICIENT, HEADROOM_LIMITED,
                                         HEADROOM_PROMISING}
    assert summary.loc["Pass 1 coverage vs full-split AUC", "screening"] == HEADROOM_LIMITED
    assert not summary["evidence"].str.contains("NO HEADROOM").any()
    assert "verdict" not in summary.columns
    # Without a noise floor nothing is classified.
    assert set(summarize_headroom(probe, None)["screening"]) == {HEADROOM_INSUFFICIENT}


def test_headroom_probe_and_summary(stage, data):
    X, y = data
    probe = headroom_probe(X, y, constraints=CONSTRAINTS_OK, budget=TINY_BUDGET,
                           stage=stage, n_configs=4, verbose=False)
    assert len(probe) == 4 and {"p1_auc", "p1_coverage", "rem_auc", "p2_recall"} <= set(probe)
    noise = noise_floor(SMALL_PARAMS, X, y, constraints=CONSTRAINTS_OK,
                        budget=TINY_BUDGET, stage=stage, seeds=(0, 1))
    summary = summarize_headroom(probe, noise)
    assert len(summary) == 3
    assert (summary["screening"] == HEADROOM_INSUFFICIENT).all()  # 4 configs, 2 seeds
