"""
stage2_crossfit.runner
======================
The PR 38 corrected Stage 2 pathway for both arms, under the frozen PR 37
specification only. The historical notebooks are not used or changed.

Configuration (G1, G2, G9)
    ``primary_settings()`` loads the frozen protocol, contract and both arm
    profiles strictly, builds ``glass_primary_config()``, ``ranker_settings()``
    and one ``rf_primary_config`` factory, and runs every check the frozen
    records name. ``s1_settings()`` is the same with ``glass_s1_full_config()``
    (S1, GLASS only). Neither takes a record from the caller. Both take the
    inner-fold feature pathway (G13): ``outer_engineer`` (historical, the
    default) or ``fold_label_excluded`` (the correction, ``inner_features``).
Train-side data
    ``load_train_side()`` rebuilds GLOBAL_SPLIT as the notebooks do and keeps
    the training rows, their labels and the test FEATURES; the test labels are
    dropped unread. ``check_population`` compares the rows, labels and the
    shared fold plan with the frozen protocol's pinned identities.
Fit-only fold (G3 preflight)
    ``fit_fold`` fits one outer fold of each arm on its training rows and
    records the fit-time evidence (``stage2_crossfit.evidence``). No held-out row is
    scored and no outer-OOF metric is computed. ``preflight_checks`` keeps
    feasibility, G3 (feasible and every fit status accepted) and the mandatory
    reviews apart; ``require_authorization`` refuses a G3 record that did not
    pass and any required review without a recorded review.
Cross-fit and full-train fits (Commit 3; refused without an authorization)
    ``run_crossfit`` runs both arms' five-fold outer cross-fit with the
    evidence taken inside each fold before it is scored. A fold whose fit
    status is not accepted stops the run there (M4), before its held-out rows
    are scored; no outer-OOF metric is computed. ``fit_full_train`` fits each
    arm once on every training row for the test-side outputs; ``save_run``
    refuses unless every attempted fold is accepted in the ``FoldLedger`` and
    each full-train fit is accepted, then writes new, versioned CANDIDATE
    artifacts (``accepted: False``) in the Stage 4 layouts plus the evidence
    and a SHA-256 manifest. Structural acceptance (V1-V8) is Commit 3A.

A failure is recorded, then raised (G8): nothing falls back to ``global_t1``,
``outer_rf`` or ``full``, and no constraint is relaxed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import shared.stage2.evaluation_protocol as ep
from blackbox_pipeline.models.rf_router import RFRouter, save_rf_router
from blackbox_pipeline.models.rf_router import profile as rf_profile
from blackbox_pipeline.models.rf_router.contract import (
    RF_F17_PRIMARY, check_rf_config_follows_contract, check_rf_constraints_follow_contract,
)
from blackbox_pipeline.models.rf_router.router.split import assert_disjoint_split, split_fingerprint
from glass_pipeline.glass_router import profile as glass_profile
from glass_pipeline.glass_router.contract import (
    GLASS_F17_PRIMARY, UNION_OBLIGATION_FIELDS, check_glass_config_follows_contract,
)
from glass_pipeline.glass_router.core.rule import ABSTAIN
from glass_pipeline.glass_router.model_outputs import build_glass_output
from glass_pipeline.glass_router.model_saver import ModelSaver
from glass_pipeline.glass_router.pipeline import (
    GlassRouterPipeline, crossfit_stage2, fit_glass_fold, fold_fingerprint, make_folds,
)
from glass_pipeline.glass_router.pipeline.crossfit import _nested_ranker
from glass_pipeline.glass_router.rf.feature_ranker import make_rf_feature_ranker_factory
from glass_pipeline.glass_router.rule_generator.rule_logger import RuleLogger
from shared.stage2.contract import constraint_provenance
from shared.stage2.feature_engineering import engineer_features
from shared.stage_io import label_fingerprint

from . import evidence as E
from . import inner_features as IF

__all__ = ["RANDOM_STATE", "N_FOLDS", "PREFLIGHT_FOLD", "ROLE_PRIMARY", "ROLE_S1",
           "RunnerError", "TrainSide", "ArmSettings", "make_train_side", "load_train_side",
           "check_population", "primary_settings", "s1_settings", "settings_record",
           "fit_fold", "preflight_checks", "run_crossfit", "fit_full_train", "fold_ledger",
           "save_run", "require_authorization", "write_record", "sha256_file"]

RANDOM_STATE = 42
N_FOLDS = 5
#: Pre-declared (protocol section 10): the G3 preflight and the G13 audit use outer fold 0.
PREFLIGHT_FOLD = 0
ROLE_PRIMARY, ROLE_S1 = "primary", "s1_full"
RUN_FORMAT = "pr38.stage2.run/1"
#: Gates Step 2B must record as passed before the five-fold primary may run.
AUTHORIZATION_GATES = ("G1", "G2", "G3", "G5", "G8", "G9", "G13")
#: The records an authorization must name (written by ``python -m stage2_crossfit preflight / g13``).
G3_RECORD_FORMAT = "pr38.stage2.g3_preflight/3"
G13_RECORD_FORMAT = "pr38.stage2.g13_fold_fits/3"
#: The original (historical-pathway) G13 audit may be the Step 2A record (/2).
G13_ORIGINAL_FORMATS = ("pr38.stage2.g13_fold_fits/2", G13_RECORD_FORMAT)
CONFIGS_RECORD_FORMAT = "pr38.stage2.configs/2"
#: The test modules whose passing results the authorization's test report must hold
#: (G8 explicit rejection; the G2 / G5 / G13 behaviour behind the records).
AUTHORIZATION_TEST_MODULES = (
    "tests.stage2_crossfit.test_pr38_evidence", "tests.stage2_crossfit.test_pr38_runner",
    "tests.stage2_crossfit.test_pr38_g13", "tests.stage2_crossfit.test_pr38_hooks",
    "tests.stage2_crossfit.test_pr38_phase0", "tests.stage2_crossfit.test_step2b_isolation",
    "tests.stage2_crossfit.test_step2b_authorization")


class RunnerError(RuntimeError):
    """A PR 38 run cannot proceed; the reason is on the record."""


class FoldNotAccepted(RunnerError):
    """A cross-fit fold's fit status is not accepted: the run stops before it is scored (M4)."""


def _quiet(verbose: bool):
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


# ----------------------------------------------------------------------
# Train-side data and identity
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TrainSide:
    """
    The training rows (raw preprocessed features), their labels, the test
    FEATURES (no labels), the shared outer fold plan and the identities.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    folds: tuple
    fold_id: np.ndarray
    fold_fingerprint: str
    router_split_fingerprint: str
    train_label_fingerprint: str
    random_state: int = RANDOM_STATE

    def fold_rows(self, k: int):
        tr, va = self.folds[k]
        return self.X_train.iloc[tr], self.y_train.iloc[tr], self.X_train.iloc[va]


def make_train_side(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, *,
                    n_folds: int = N_FOLDS, random_state: int = RANDOM_STATE) -> TrainSide:
    if not X_train.index.equals(y_train.index):
        raise RunnerError("X_train and y_train are not index-aligned")
    assert_disjoint_split(X_train, X_test)
    folds = tuple((np.asarray(tr), np.asarray(va))
                  for tr, va in make_folds(y_train, n_folds, random_state))
    fold_id = np.full(len(y_train), -1, dtype=int)
    for k, (_, va) in enumerate(folds):
        fold_id[va] = k
    return TrainSide(X_train, y_train.astype(int), X_test, folds, fold_id,
                     fold_fingerprint(fold_id), split_fingerprint(X_train, X_test),
                     label_fingerprint(X_train.index, y_train), random_state)


def load_train_side(*, verbose: bool = False) -> TrainSide:
    """GLOBAL_SPLIT as the notebooks build it (Cells 2-4 / 6); test labels dropped unread."""
    from data.global_splits.split_manager import GlobalSplitManager
    from data.preprocessing import BankPreprocessor
    with _quiet(verbose):
        df = BankPreprocessor(drop_leaky=True, verbose=False).fit_transform()
        split = GlobalSplitManager(test_size=0.20, random_state=RANDOM_STATE).create_split(
            df, target_col="y")
    X_train, y_train, X_test = split["X_train"], split["y_train"], split["X_test"]
    del split                                       # y_test is never read
    return make_train_side(X_train, y_train, X_test)


def check_population(ts: TrainSide, protocol) -> Dict[str, Any]:
    """The train side against the frozen protocol's pinned population (V1 inputs)."""
    p = protocol["population"]
    y = ts.y_train.to_numpy()
    got = {
        "rows": len(y), "positives": int(y.sum()),
        "fold_rows": [int((ts.fold_id == k).sum()) for k in range(len(ts.folds))],
        "fold_positives": [int(y[ts.fold_id == k].sum()) for k in range(len(ts.folds))],
        "router_split_fingerprint": ts.router_split_fingerprint,
        "train_label_fingerprint": ts.train_label_fingerprint,
        "fold_fingerprint": ts.fold_fingerprint,
    }
    want = {k: (list(p[k]) if isinstance(p[k], (list, tuple)) else p[k]) for k in got}
    checks = {k: {"got": got[k], "pinned": want[k], "passed": got[k] == want[k]} for k in got}
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


# ----------------------------------------------------------------------
# Strict configurations (G1, G2, G9)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSettings:
    """What a run fits with. Build it with ``primary_settings`` or ``s1_settings``."""

    role: str
    glass_config: Any
    ranker: Mapping[str, Any]
    rf_config_factory: Optional[Callable[[], Any]]
    contract: Any
    protocol: Any
    checks: Mapping[str, Any] = field(default_factory=dict)
    feature_pathway: str = IF.PATHWAY_OUTER

    @property
    def rf_params(self) -> dict:
        return dict(self.ranker["rf_params"])

    @property
    def inner_features(self):
        """The provider factory for the RF arm (and GLASS ``oof_remainder``), or None."""
        return IF.provider_factory(self.feature_pathway)

    @property
    def glass_inner_features(self):
        return (self.inner_features
                if self.glass_config.pass2_population == "oof_remainder" else None)


def _strict_spec():
    spec = ep.load_frozen_specification()
    glass_prof = glass_profile.load_glass_primary_profile()
    rf_prof = rf_profile.load_rf_primary_profile()
    ep.check_arm_profile_binding(spec.protocol, glass_prof)
    ep.check_arm_profile_binding(spec.protocol, rf_prof)
    if not ep.is_pinned_frozen(spec.protocol):
        raise RunnerError("the loaded protocol is not the pinned frozen record")
    return spec, glass_prof, rf_prof


def _rf_factory() -> Callable[[], Any]:
    def make():
        config = rf_profile.rf_primary_config(verbose=False, validate_nested=False)
        rf_profile.check_rf_primary_config(config)
        return config
    return make


def _f17_check(config, expected: Mapping[str, Any]) -> Dict[str, Any]:
    got = {k: getattr(config, k) for k in expected}
    return {"expected": dict(expected), "got": got, "passed": got == dict(expected)}


def primary_settings(feature_pathway: str = IF.PATHWAY_OUTER) -> ArmSettings:
    """
    The PR 38 primary for both arms; every check the frozen records name,
    recorded. ``feature_pathway`` (G13) changes no config or frozen record.
    """
    pathway = IF.pathway_identity(feature_pathway)
    spec, glass_prof, rf_prof = _strict_spec()
    glass = glass_profile.glass_primary_config()
    glass_profile.check_glass_primary_config(glass)
    check_glass_config_follows_contract(glass, spec.contract)
    ranker = glass_profile.ranker_settings()
    glass_profile.check_glass_primary_ranker(**ranker)
    make_rf = _rf_factory()
    rf_cfg = make_rf()
    check_rf_config_follows_contract(rf_cfg, spec.contract)
    rf_provenance = constraint_provenance(spec.contract)
    check_rf_constraints_follow_contract(rf_cfg.constraints, spec.contract, extra=rf_provenance)
    if rf_cfg.constraints.to_dict() != make_rf().constraints.to_dict():
        raise RunnerError("the RF config factory is not deterministic")
    base = glass_profile.glass_base_config(glass_prof).to_dict()
    from_contract = sorted(k for k, v in glass.to_dict().items() if base.get(k) != v)
    contract_fields = sorted(set(UNION_OBLIGATION_FIELDS) | {"contract_provenance"})
    checks = {
        "strict_loaders": "load_frozen_specification, load_glass_primary_profile, "
                          "load_rf_primary_profile, check_arm_profile_binding (both), "
                          "is_pinned_frozen",
        "check_glass_primary_config": "passed", "check_glass_config_follows_contract": "passed",
        "check_glass_primary_ranker": "passed", "check_rf_primary_config": "passed",
        "check_rf_config_follows_contract": "passed",
        "G1_glass_f17": _f17_check(glass, GLASS_F17_PRIMARY),
        "G1_rf_f17": _f17_check(rf_cfg, dict(RF_F17_PRIMARY)),
        # G1 diff: the primary differs from the frozen profile only in what the contract supplies.
        "G1_glass_fields_set_by_contract": {"got": from_contract,
                                            "contract_fields": contract_fields,
                                            "passed": set(from_contract) <= set(contract_fields)},
        # G9: what the RF artifact records; no GLASS artifact is read for any value.
        "G9_rf_constraint_provenance": rf_provenance,
        "G9_glass_contract_provenance": dict(glass.contract_provenance),
        "identities": {
            "contract": [spec.contract.contract_id, spec.contract.contract_version,
                         spec.contract.fingerprint()],
            "protocol": [spec.protocol.protocol_id, spec.protocol.version,
                         spec.protocol.fingerprint()],
            "glass_profile": glass_prof.identity(), "rf_profile": rf_prof.identity(),
        },
        "feature_pathway": pathway,
    }
    if not (checks["G1_glass_f17"]["passed"] and checks["G1_rf_f17"]["passed"]
            and checks["G1_glass_fields_set_by_contract"]["passed"]):
        raise RunnerError(f"G1 configuration check failed: {checks}")
    return ArmSettings(ROLE_PRIMARY, glass, ranker, make_rf, spec.contract, spec.protocol,
                       checks, feature_pathway)


def s1_settings(feature_pathway: str = IF.PATHWAY_OUTER) -> ArmSettings:
    """S1 (GLASS ``full``): the primary with only ``pass2_population`` changed; GLASS only."""
    primary = primary_settings(feature_pathway)
    s1 = glass_profile.glass_s1_full_config()
    glass_profile.check_glass_s1_config(s1)
    checks = dict(primary.checks, check_glass_s1_config="passed",
                  s1_factor=list(glass_profile.S1_FACTOR))
    return ArmSettings(ROLE_S1, s1, primary.ranker, None, primary.contract, primary.protocol,
                       checks, feature_pathway)


def settings_record(settings: ArmSettings) -> Dict[str, Any]:
    """The configs and check results, for the record (G1, G2, G9)."""
    out = {"role": settings.role, "checks": settings.checks,
           "glass_config": settings.glass_config.to_dict(), "ranker": dict(settings.ranker),
           "feature_pathway": IF.pathway_identity(settings.feature_pathway)}
    if settings.rf_config_factory is not None:
        out["rf_config"] = settings.rf_config_factory().to_dict()
    return E.to_json_safe(out)


# ----------------------------------------------------------------------
# One outer fold, fit-only (G3 preflight; also the G13 audit's fits)
# ----------------------------------------------------------------------
def _glass_fold(ts: TrainSide, settings: ArmSettings, k: int, verbose: bool):
    X_tr, y_tr, X_va = ts.fold_rows(k)
    with _quiet(verbose):
        return fit_glass_fold(X_tr, y_tr, X_va, settings.glass_config, settings.rf_params,
                              random_state=ts.random_state,
                              glass_rf_model=settings.ranker["glass_rf_model"],
                              rule_logger=RuleLogger(enabled=False),
                              inner_features=settings.glass_inner_features)


def _rf_fit(router, X_eng, X_raw, y, split_fp, settings: ArmSettings):
    """Fit ``router``; return the inner-features provider it used (None historically)."""
    factory = settings.inner_features
    if factory is None:
        router.fit(X_eng, y, split_fp)
        return None
    provider = factory(X_raw, y)
    router.fit(X_eng, y, split_fp, inner_features=provider)
    return provider


def _rf_fold(ts: TrainSide, settings: ArmSettings, k: int, verbose: bool):
    X_tr, y_tr, X_va = ts.fold_rows(k)
    with _quiet(verbose):
        X_tr_eng, _ = engineer_features(X_tr, X_va, y_tr)
        router = RFRouter(settings.rf_config_factory())
        provider = _rf_fit(router, X_tr_eng, X_tr, y_tr, ts.router_split_fingerprint, settings)
    return router, X_tr_eng, y_tr, provider


def fit_fold(ts: TrainSide, settings: ArmSettings, k: int = PREFLIGHT_FOLD, *,
             arms: Sequence[str] = (E.ARM_GLASS, E.ARM_RF), verbose: bool = False,
             keep_fits: bool = False) -> Dict[str, Any]:
    """
    Fit outer fold ``k`` of each arm on its training rows and record the
    evidence. Nothing is scored. A fit that raises is recorded as a stop and
    the other arm still runs; ``keep_fits`` returns the fitted objects too.
    """
    if E.ARM_RF in arms and settings.rf_config_factory is None:
        raise RunnerError(f"the {settings.role} settings have no RF arm")
    out: Dict[str, Any] = {"fold": k, "arms": {}, "fits": {}}
    held = ts.X_train.index[ts.folds[k][1]]
    for arm in arms:
        t0 = time.perf_counter()
        try:
            if arm == E.ARM_GLASS:
                fit = _glass_fold(ts, settings, k, verbose)
                ev = E.glass_fold_evidence(fit.glass, fit.X_tr_eng, fit.y_tr, fold=k,
                                           heldout_index=held, inner_features=fit.inner_features)
                out["fits"][arm] = fit
            else:
                router, X_tr_eng, y_tr, provider = _rf_fold(ts, settings, k, verbose)
                ev = E.rf_fold_evidence(router, X_tr_eng, y_tr, fold=k, heldout_index=held,
                                        inner_features=provider)
                out["fits"][arm] = (router, X_tr_eng, y_tr)
        except Exception as exc:                    # recorded, never a fallback (G8)
            partial = getattr(exc, "partial_fold_fit", None)
            reports = partial.glass.selection_reports() if partial is not None else None
            ev = E.stop_evidence(arm, k, exc, reports)
        ev["wall_seconds"] = round(time.perf_counter() - t0, 1)
        ev["fit_status"] = E.fold_status(ev, settings.contract, settings.protocol)
        out["arms"][arm] = ev
    if not keep_fits:
        out.pop("fits")
    return out


def _isolation_clean(iso: Mapping[str, Any]) -> bool:
    counts = ("held_out_rows_in_fit", "remainder_held_out_rows")
    flags = ("inner_rankers_fitted_on_their_inner_train_rows",
             "eligibility_solves_exclude_their_fold", "inner_features_match_the_fit")
    return (all(int(iso.get(k, 0)) == 0 for k in counts)
            and all(all(iso[k]) if isinstance(iso[k], (list, tuple)) else bool(iso[k])
                    for k in flags if k in iso))


def preflight_checks(result: Mapping[str, Any], remainder_min_size: int = 500, *,
                     protocol) -> Dict[str, Any]:
    """
    G3 on the fit-only fold, per arm and overall, three things kept apart:

    ``feasible``  no stop; remainder >= ``remainder_min_size`` rows with both
                  classes; every selection recorded; isolation clean; RF: R1
                  supports at or above their floors and the R2 strips recorded.
    ``passed``    G3: feasible AND every fit status accepted. An unproven
                  incumbent (M3 reproduction) or a greedy fallback (D-C7
                  review) must be resolved in the fit evidence; a review in the
                  authorization cannot turn a status that is not accepted into
                  a pass.
    ``reviews_required``  mandatory reviews the fit raises: the F31 strip
                  review (``ep.f31_strip_review``; an empty band is reviewed
                  too). They do not fail G3, but ``require_authorization``
                  refuses until each has a recorded review.
    ``authorizable``  passed and no review required.
    """
    out = {}
    for arm, ev in result["arms"].items():
        if "stop" in ev:
            out[arm] = {"feasible": False, "fit_status_accepted": False, "passed": False,
                        "stop": ev["stop"], "reviews_required": []}
            continue
        rem = ev.get("remainder") or {}
        rows, pos = rem.get("rows", 0), rem.get("positives", 0)
        status = ev["fit_status"]
        checks = {
            "no_stop": True,
            "remainder_at_least_min_with_both_classes": bool(
                rows >= remainder_min_size and 0 < pos < rows),
            "every_selection_recorded": not status["problems"],
            "isolation_clean": _isolation_clean(ev.get("isolation") or {}),
            "fit_status": status["status"],
            "fit_status_accepted": bool(status["accepted"]),
        }
        reviews = []
        if arm == E.ARM_RF:
            checks["r1_supports_at_or_above_floors"] = ev["r1_bands"][
                "supports_at_or_above_floors"]
            nested = (ev.get("eligibility") or {}).get("rule") == "nested_t1"
            checks["r2_strips_recorded"] = (not nested) or "r2_strips" in ev
            if nested and "r2_strips" in ev:
                s2 = ev["r2_strips"]
                f31 = ep.f31_strip_review({ev["fold"]: (s2["strip_a_in_band"]["rows"],
                                                        s2["band_rows"])}, protocol).iloc[0]
                checks["f31_strip_review"] = {"status": f31["status"], "share": f31["share"],
                                              "basis": E.R2_BASIS}
                if f31["review_required"]:
                    reviews.append(f"F31:{arm}:fold{ev['fold']}:{f31['status']}")
        feasible = all(v for k, v in checks.items()
                       if k not in ("fit_status", "fit_status_accepted", "f31_strip_review"))
        out[arm] = {"feasible": feasible, "fit_status_accepted": checks["fit_status_accepted"],
                    "passed": feasible and checks["fit_status_accepted"], "checks": checks,
                    "feature_pathway": (ev.get("inner_feature_pathway") or {}).get("pathway"),
                    "reviews_required": reviews,
                    "note": None if status["accepted"] else
                    "status not accepted: an unproven incumbent needs its M3 reproduction, a "
                    "greedy fallback its D-C7 review, recorded in the fit evidence; then rerun"}
    reviews = [r for v in out.values() for r in v["reviews_required"]]
    passed = all(v["passed"] for v in out.values())
    return {"feasible": all(v["feasible"] for v in out.values()), "passed": passed,
            "reviews_required": reviews, "authorizable": passed and not reviews, "arms": out}


# ----------------------------------------------------------------------
# Cross-fit and full-train fits (Commit 3)
# ----------------------------------------------------------------------
@dataclass
class RunResult:
    settings_role: str
    glass_oof: Any = None
    rf_crossfit: Any = None
    glass_full: Any = None
    rf_full: Any = None
    evidence: Dict[str, List[dict]] = field(default_factory=dict)
    full_train_evidence: Dict[str, dict] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)


def run_crossfit(ts: TrainSide, settings: ArmSettings, *,
                 arms: Sequence[str] = (E.ARM_GLASS, E.ARM_RF), verbose: bool = False,
                 result: Optional[RunResult] = None) -> RunResult:
    """
    Both arms' five-fold outer cross-fit on the shared plan. Each fold's
    evidence is recorded before its held-out rows are scored; no outer-OOF
    metric is computed. A fold that raises, or whose fit status is not
    accepted, is recorded and stops the run before it is scored (M4).
    """
    res = result or RunResult(settings.role)

    def stop_unless_accepted(arm, ev):
        if not ev["fit_status"]["accepted"]:
            raise FoldNotAccepted(f"{arm} fold {ev['fold']}: fit status "
                                  f"{ev['fit_status']['status']!r} is not accepted; the run stops "
                                  "before this fold is scored (M4)")
    if E.ARM_GLASS in arms:
        res.evidence[E.ARM_GLASS] = evs = []

        def glass_cb(k, fit, tr, va):
            ev = E.glass_fold_evidence(fit.glass, fit.X_tr_eng, fit.y_tr, fold=k,
                                       heldout_index=ts.X_train.index[va],
                                       inner_features=fit.inner_features)
            ev["fit_status"] = E.fold_status(ev, settings.contract, settings.protocol)
            evs.append(ev)
            stop_unless_accepted(E.ARM_GLASS, ev)
            return ev

        t0 = time.perf_counter()
        try:
            with _quiet(verbose):
                res.glass_oof = crossfit_stage2(
                    ts.X_train, ts.y_train, settings.glass_config, settings.rf_params,
                    folds=list(ts.folds), random_state=ts.random_state,
                    glass_rf_model=settings.ranker["glass_rf_model"], verbose=False,
                    fold_callback=glass_cb, report_outer=False,
                    inner_features=settings.glass_inner_features)
        except FoldNotAccepted:
            raise                                   # its evidence is already recorded
        except Exception as exc:
            partial = getattr(exc, "partial_fold_fit", None)
            evs.append(E.stop_evidence(E.ARM_GLASS, len(evs), exc, None if partial is None
                                       else partial.glass.selection_reports()))
            raise RunnerError(f"GLASS fold {len(evs) - 1} stopped: {exc}") from exc
        res.seconds["glass_crossfit"] = round(time.perf_counter() - t0, 1)
    if E.ARM_RF in arms:
        if settings.rf_config_factory is None:
            raise RunnerError(f"the {settings.role} settings have no RF arm")
        res.evidence[E.ARM_RF] = evs_rf = []
        providers = []                              # the fold's provider, for its evidence

        def rf_factory(X_raw, y):
            providers.append(settings.inner_features(X_raw, y))
            return providers[-1]

        def rf_cb(k, router, X_tr_eng, y_tr):
            ev = E.rf_fold_evidence(router, X_tr_eng, y_tr, fold=k,
                                    heldout_index=ts.X_train.index[ts.folds[k][1]],
                                    inner_features=providers[-1] if providers else None)
            ev["fit_status"] = E.fold_status(ev, settings.contract, settings.protocol)
            evs_rf.append(ev)
            stop_unless_accepted(E.ARM_RF, ev)
            return ev

        def engineer(X_tr, X_va, y_tr):
            with _quiet(verbose):
                return engineer_features(X_tr, X_va, y_tr)

        t0 = time.perf_counter()
        try:
            from blackbox_pipeline.models.rf_router import crossfit_rf_router
            res.rf_crossfit = crossfit_rf_router(
                ts.X_train, ts.y_train, list(ts.folds), engineer=engineer,
                make_config=settings.rf_config_factory, split_fp=ts.router_split_fingerprint,
                fold_fp=ts.fold_fingerprint, random_state=ts.random_state,
                hyperparameter_note=("rf_router_primary_v1 profile (frozen; tuned once on full "
                                     "X_train, disclosed in the profile provenance)"),
                verbose=False, fold_callback=rf_cb, report_outer=False,
                inner_features=None if settings.inner_features is None else rf_factory)
        except FoldNotAccepted:
            raise
        except Exception as exc:
            evs_rf.append(E.stop_evidence(E.ARM_RF, len(evs_rf), exc))
            raise RunnerError(f"RF fold {len(evs_rf) - 1} stopped: {exc}") from exc
        res.seconds["rf_crossfit"] = round(time.perf_counter() - t0, 1)
    return res


def fit_full_train(ts: TrainSide, settings: ArmSettings, result: RunResult, *,
                   arms: Sequence[str] = (E.ARM_GLASS, E.ARM_RF), verbose: bool = False
                   ) -> RunResult:
    """Each arm fitted once on every training row; test-side outputs from features only."""
    with _quiet(verbose):
        X_eng_train, X_eng_test = engineer_features(ts.X_train, ts.X_test, ts.y_train)
    if E.ARM_GLASS in arms:
        t0 = time.perf_counter()
        rf_params, rs = settings.rf_params, ts.random_state
        glass_rf_model = settings.ranker["glass_rf_model"]
        nested = _nested_ranker(settings.glass_config, glass_rf_model)
        with _quiet(verbose):
            # The full-train ordering RF: the frozen ranker settings, the fold-RF recipe.
            ordering_rf = (make_rf_feature_ranker_factory(rf_params, rs)(X_eng_train, ts.y_train)
                           if glass_rf_model == "fold_rf" else None)
            kwargs = ({"feature_ranker_factory": make_rf_feature_ranker_factory(rf_params, rs)}
                      if nested else {})
            glass = GlassRouterPipeline(settings.glass_config, rf_model=ordering_rf,
                                        rule_logger=RuleLogger(enabled=False), **kwargs)
            g_factory = settings.glass_inner_features
            provider = None if g_factory is None else g_factory(ts.X_train, ts.y_train)
            try:
                if provider is None:
                    glass.fit(X_eng_train, ts.y_train)
                else:
                    glass.fit(X_eng_train, ts.y_train, inner_features=provider)
            except Exception as exc:
                result.full_train_evidence[E.ARM_GLASS] = E.stop_evidence(
                    E.ARM_GLASS, "full_train", exc, glass.selection_reports())
                raise RunnerError(f"GLASS full-train fit stopped: {exc}") from exc
            test_out = build_glass_output(glass, X_eng_test, abstain_value=ABSTAIN)
            test_proba = glass.predict_proba(X_eng_test)
        ev = E.glass_fold_evidence(glass, X_eng_train, ts.y_train, fold="full_train",
                                   inner_features=provider)
        ev["fit_status"] = E.fold_status(ev, settings.contract, settings.protocol)
        result.full_train_evidence[E.ARM_GLASS] = ev
        result.glass_full = {"glass": glass, "test_out": test_out, "test_proba": test_proba}
        result.seconds["glass_full_train"] = round(time.perf_counter() - t0, 1)
    if E.ARM_RF in arms:
        t0 = time.perf_counter()
        with _quiet(verbose):
            router = RFRouter(settings.rf_config_factory())
            try:
                rf_provider = _rf_fit(router, X_eng_train, ts.X_train, ts.y_train,
                                      ts.router_split_fingerprint, settings)
            except Exception as exc:
                result.full_train_evidence[E.ARM_RF] = E.stop_evidence(E.ARM_RF, "full_train", exc)
                raise RunnerError(f"RF full-train fit stopped: {exc}") from exc
        ev = E.rf_fold_evidence(router, X_eng_train, ts.y_train, fold="full_train",
                                inner_features=rf_provider)
        ev["fit_status"] = E.fold_status(ev, settings.contract, settings.protocol)
        result.full_train_evidence[E.ARM_RF] = ev
        result.rf_full = {"router": router, "test": router.predict_with_proba(X_eng_test)}
        result.seconds["rf_full_train"] = round(time.perf_counter() - t0, 1)
    return result


# ----------------------------------------------------------------------
# Writing (G5, G10): new versioned names, evidence inside the artifacts, a manifest
# ----------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_record(path, record: Mapping[str, Any]) -> Path:
    """Write a JSON record once: an existing file is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(E.to_json_safe(record), indent=2, sort_keys=True)
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return path


def _fold_summary_with_r1(folds_frame: pd.DataFrame, evs: List[dict]) -> List[dict]:
    """``oof_fold_summary`` records with each fold router's R1 bands and supports."""
    rows = folds_frame.to_dict("records")
    by_fold = {ev["fold"]: ev["r1_bands"] for ev in evs if "r1_bands" in ev}
    for row in rows:
        r1 = by_fold.get(int(row["fold"]))
        if r1 is None:
            raise RunnerError(f"RF fold {row['fold']} has no R1 band record")
        row.update({k: r1[k] for k in ("pass1_confidence", "pass2_confidence",
                                       "pass1_support", "pass2_support")})
    return rows


def fold_ledger(result: RunResult, settings: ArmSettings) -> ep.FoldLedger:
    """The run's ``FoldLedger``: each fold re-classified from its evidence (a stop by its status)."""
    status = {arm: {int(ev["fold"]): (E.classify_fold(ev, settings.contract, settings.protocol)
                                      or ev["fit_status"]["status"]) for ev in evs}
              for arm, evs in result.evidence.items()}
    return ep.FoldLedger.build(status, settings.protocol)


def _require_accepted(result: RunResult, settings: ArmSettings) -> ep.FoldLedger:
    """Every attempted fold accepted in each arm run, and each full-train fit accepted."""
    arms = [a for a, cf in ((E.ARM_GLASS, result.glass_oof), (E.ARM_RF, result.rf_crossfit))
            if cf is not None]
    try:
        ledger = fold_ledger(result, settings).require_complete(arms)
    except ep.ProtocolError as exc:
        raise RunnerError(f"not saved: {exc}") from exc
    full = {a: (result.full_train_evidence.get(a) or {}).get("fit_status") or {} for a in arms}
    bad = {a: f.get("status") for a, f in full.items() if not f.get("accepted")}
    if bad:
        raise RunnerError(f"not saved: full-train fit status not accepted {bad}")
    return ledger


def save_run(result: RunResult, ts: TrainSide, settings: ArmSettings, run_root, run_id: str,
             *, extra_identity: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Write the run under ``run_root`` (a new directory; never an existing one):
    the GLASS bundle (``ModelSaver``), the RF Router (``save_rf_router``), both
    in their Stage 4 layouts with the fit-time evidence inside, then
    ``evidence.json`` and ``MANIFEST.json`` with every file's SHA-256.

    Refused, with nothing written, unless every attempted fold is accepted in
    the ``FoldLedger`` and each full-train fit is accepted. The artifacts are
    candidates (``accepted: False``): structural acceptance (V1-V8,
    ``ep.structural_acceptance``) is Commit 3A and nothing here sets it.
    """
    ledger = _require_accepted(result, settings)
    root = Path(run_root)
    if root.exists():
        raise RunnerError(f"{root} exists: a PR 38 run never writes into an existing directory")
    root.mkdir(parents=True)
    acceptance = {"fold_ledger_complete": True, "structural_acceptance": None,
                  "accepted": False, "pending": "structural acceptance V1-V8 (Commit 3A)"}
    pr38 = {"format": RUN_FORMAT, "run_id": run_id, "role": settings.role,
            "settings": settings_record(settings), "acceptance": acceptance}
    written = {}
    if result.glass_oof is not None:
        g = result.glass_full or {}
        if not g:
            raise RunnerError("the GLASS cross-fit has no full-train fit to save with")
        path = root / "glass_router" / f"glass_router_pr38_{settings.role}_{run_id}.joblib"
        with _quiet(False):
            ModelSaver(glass_model=g["glass"]).save_model(
                train_out=None, train_proba=None, oof=result.glass_oof,
                test_out=g["test_out"], test_proba=g["test_proba"],
                global_split={"X_train": ts.X_train, "X_test": ts.X_test},
                output_path=str(path), y_test=None,
                split_fingerprint=ts.router_split_fingerprint,
                extra_fields={"pr38_evidence": E.to_json_safe({
                    **pr38, "folds": result.evidence.get(E.ARM_GLASS, []),
                    "full_train": result.full_train_evidence.get(E.ARM_GLASS)})})
        written["glass_router"] = path
    if result.rf_crossfit is not None:
        r = result.rf_full or {}
        if not r:
            raise RunnerError("the RF cross-fit has no full-train router to save with")
        cf = result.rf_crossfit
        p_te, c_te, d_te, proba_te = r["test"]
        extra = {
            "index_train": ts.X_train.index, "index_test": ts.X_test.index,
            **constraint_provenance(settings.contract),
            "rf_params_origin": {"profile": settings.checks.get("identities", {}).get("rf_profile")},
            "train_outputs_oof": True,
            "train_pred": cf.oof["pred"], "train_confidence": cf.oof["confidence"],
            "train_decisions": cf.oof["decisions"], "train_proba": cf.oof["proba"],
            "train_fold_id": ts.fold_id, "oof_protocol": cf.protocol,
            "oof_fold_summary": _fold_summary_with_r1(cf.folds, result.evidence[E.ARM_RF]),
            "test_pred": p_te, "test_confidence": c_te, "test_decisions": d_te,
            "test_proba": proba_te,
            "pr38_evidence": E.to_json_safe({**pr38, "folds": result.evidence[E.ARM_RF],
                                             "full_train": result.full_train_evidence.get(
                                                 E.ARM_RF)}),
        }
        with _quiet(False):
            written["rf_router"] = save_rf_router(r["router"], root / "rf_router",
                                                  name=f"rf_router_pr38_{settings.role}_{run_id}",
                                                  test_metrics=None, extra_metadata=extra)
    evidence = {"format": RUN_FORMAT, "run_id": run_id, "role": settings.role,
                "acceptance": acceptance, "fold_ledger": ledger.summary().to_dict("records"),
                "folds": result.evidence, "full_train": result.full_train_evidence,
                "seconds": result.seconds}
    write_record(root / "evidence.json", evidence)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    manifest = {
        "format": RUN_FORMAT, "run_id": run_id, "role": settings.role,
        "identities": settings.checks.get("identities"),
        "feature_pathway": IF.pathway_identity(settings.feature_pathway),
        "population": {"router_split_fingerprint": ts.router_split_fingerprint,
                       "train_label_fingerprint": ts.train_label_fingerprint,
                       "fold_fingerprint": ts.fold_fingerprint},
        "extra_identity": dict(extra_identity or {}),
        "acceptance": acceptance,
        "files": {str(p.relative_to(root)).replace("\\", "/"): {"sha256": sha256_file(p),
                                                               "bytes": p.stat().st_size}
                  for p in files},
    }
    write_record(root / "MANIFEST.json", manifest)
    (root / "MANIFEST.json.sha256").write_text(sha256_file(root / "MANIFEST.json") + "\n")
    return {"root": root, "written": written, "manifest": manifest}


def _referenced_file(rec: Mapping[str, Any], key: str) -> Path:
    """The file the authorization names under ``key``, unchanged since (``key_sha256``)."""
    ref = Path(str(rec.get(key) or ""))
    if not rec.get(key) or not ref.is_file():
        raise RunnerError(f"authorization: {key} {rec.get(key)!r} is not a file")
    if sha256_file(ref) != str(rec.get(f"{key}_sha256", "")).lower():
        raise RunnerError(f"authorization: {key} has changed since it was authorized (SHA-256)")
    return ref


def _referenced_record(rec: Mapping[str, Any], key: str, fmt) -> Dict[str, Any]:
    """The JSON record the authorization names under ``key``, unchanged, of format ``fmt``."""
    fmts = (fmt,) if isinstance(fmt, str) else tuple(fmt)
    body = json.loads(_referenced_file(rec, key).read_text(encoding="utf-8"))
    if body.get("format") not in fmts:
        raise RunnerError(f"authorization: {key} is {body.get('format')!r}, not "
                          + " or ".join(repr(f) for f in fmts))
    return body


def _recorded(entry) -> bool:
    return isinstance(entry, Mapping) and all(
        isinstance(entry.get(k), str) and entry[k].strip() for k in ("reviewer", "rationale", "date"))


_REPRODUCTION_TOL = 1e-9


def _g13_problems(body: Mapping[str, Any], *, label: str, pathway: str,
                  dispositions: Mapping[str, Any]) -> List[str]:
    """A G13 record's audit evidence, both arms, on ``pathway`` (the record's fit)."""
    out, g = [], body.get("g13") or {}
    if g.get("uses_test_rows") is not False:
        out.append(f"G13 {label}: the audit does not record uses_test_rows False")
    if (body.get("population") or {}).get("passed") is not True:
        out.append(f"G13 {label}: the audited population did not pass")
    pathways = g.get("pathways") or {}                # absent from Step 2A (/2) records
    for arm in ("glass", "rf"):
        r = (g.get("eligibility_retrained") or {}).get(arm)
        if not isinstance(r, Mapping):
            out.append(f"G13 {label}: no {arm} audit evidence")
            continue
        pw = pathways.get(arm) or {"pathway": IF.PATHWAY_OUTER, "record": None}
        if pw.get("pathway") != pathway or pw.get("record") != IF.pathway_identity(
                pathway)["record"]:
            out.append(f"G13 {label}: {arm} was audited on {pw.get('pathway')!r} "
                       f"{pw.get('record')}, not {pathway!r} {IF.pathway_identity(pathway)['record']}")
        if r.get("complete") is not True:
            out.append(f"G13 {label}: the {arm} retrained eligibility comparison is incomplete")
        if arm == "rf":
            t = r.get("t1_reproduction_max_difference")
            if t is None or float(t) > _REPRODUCTION_TOL:
                out.append(f"G13 {label}: the rf t1_j reproduction is not recorded within "
                           f"{_REPRODUCTION_TOL}")
        elif r.get("reproduced_all") is not True and not _recorded(
                dispositions.get(f"G13:{label}:glass:reproduction")):
            out.append(f"G13 {label}: the glass inner fits did not all reproduce and no "
                       f"disposition 'G13:{label}:glass:reproduction' is recorded")
        if pathway == IF.PATHWAY_FOLD and ((g.get("isolation") or {}).get(arm) or {}).get(
                "pathway_match") is not True:
            out.append(f"G13 {label}: the {arm} fit's inner bins are not the audit's own "
                       "label-excluded recomputation")
    return out


def _g3_problems(body: Mapping[str, Any], pathway: str) -> List[str]:
    """G2 wiring and G5 persistence, from the G3 preflight record's fit evidence."""
    out = []
    if body.get("feature_pathway") != IF.pathway_identity(pathway):
        out.append(f"G3: the preflight ran on {(body.get('feature_pathway') or {}).get('pathway')!r},"
                   f" not the adopted {pathway!r}")
    if (body.get("population") or {}).get("passed") is not True:
        out.append("G3: the preflight population did not pass")
    arms = body.get("arms") or {}
    if set(arms) != {E.ARM_GLASS, E.ARM_RF}:
        return out + [f"G3: the preflight must fit both arms, got {sorted(arms)}"]
    g, r = arms[E.ARM_GLASS], arms[E.ARM_RF]
    gi, ri = g.get("isolation") or {}, r.get("isolation") or {}
    prov = g.get("inner_ranker_provenance") or []
    wiring = {
        "GLASS pass2_population oof_remainder": g.get("pass2_population") == "oof_remainder",
        "GLASS fold_nested_rf rankers, each on its inner-train rows": bool(prov) and all(
            gi.get("inner_rankers_fitted_on_their_inner_train_rows") or [False]),
        "RF nested_t1, each t1_j solved without its fold": (r.get("eligibility") or {}).get(
            "rule") == "nested_t1" and all(ri.get("eligibility_solves_exclude_their_fold")
                                           or [False]),
        **{f"{a} on the adopted feature pathway": (ev.get("inner_feature_pathway") or {}).get(
            "pathway") == pathway and (ev.get("inner_feature_pathway") or {}).get("record")
            == IF.pathway_identity(pathway)["record"] for a, ev in arms.items()},
        **({f"{a} inner bins match the fit": iso.get("inner_features_match_the_fit") is True
            for a, iso in (("glass", gi), ("blackbox", ri))} if pathway == IF.PATHWAY_FOLD
           else {}),
    }
    out += [f"G2: {k}" for k, ok in wiring.items() if not ok]
    need = {E.ARM_GLASS: {"selection_reports", "counts", "remainder", "m6_inner_oof_pass1",
                          "inner_ranker_provenance", "isolation", "fit_status"},
            E.ARM_RF: {"counts", "remainder", "r1_bands", "r2_strips", "eligibility",
                       "isolation", "fit_status"}}
    protocol = ep.load_frozen_protocol()
    for arm, ev in arms.items():
        missing = sorted(need[arm] - set(ev))
        if missing:
            out.append(f"G5: {arm} evidence lacks {missing}")
        if set(ev.get("counts") or {}) != set(ep.required_selection_keys(arm, protocol)):
            out.append(f"G5: {arm} union counts are not recorded for every required selection")
        if (ev.get("fit_status") or {}).get("problems"):
            out.append(f"G5: {arm} selection records are incomplete")
    return out


def _test_report_problems(path: Path) -> List[str]:
    """G8 (and the behaviour behind G2, G5, G13): a JUnit report with no failure."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [f"G8: the test report is not JUnit XML ({exc})"]
    cases = list(root.iter("testcase"))
    bad = [c.get("classname", "") + "::" + c.get("name", "") for c in cases
           if c.find("failure") is not None or c.find("error") is not None]
    passed = {c.get("classname", "") for c in cases
              if c.find("skipped") is None and c.find("failure") is None
              and c.find("error") is None}
    out = [f"G8: {len(bad)} failing tests in the report, e.g. {bad[:3]}"] if bad else []
    out += [f"G8: no passing test from {m} in the report" for m in AUTHORIZATION_TEST_MODULES
            if not any(cn == m or cn.startswith(m + ".") for cn in passed)]
    return out


def _passed_phase0(run_dir: Path) -> bool:
    try:
        return json.loads((run_dir / "phase0.json").read_text(encoding="utf-8")).get(
            "passed") is True
    except (OSError, ValueError):
        return False


def _population_identity(body: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    checks = (body.get("population") or {}).get("checks")
    return ({k: (v or {}).get("got") for k, v in checks.items()}
            if isinstance(checks, Mapping) else None)


def _run_binding_problems(path: Path, rec: Mapping[str, Any], run_id: Optional[str],
                          current: Mapping[str, Optional[Mapping[str, Any]]],
                          original: Mapping[str, Any]) -> List[str]:
    """
    The authorization, its run directory, and every record it names belong to
    one run: the current-run records (``current``: G3, configs and, when
    adopted, the corrected G13) and the test report sit in the authorization's
    own run directory, carry its ``run_id``, and the fold records are outer
    fold ``PREFLIGHT_FOLD``; that run's Phase 0 passed. The original G13 audit
    may belong to its own run (``g13_record_run_id``): it must sit in that
    run's directory beside this one, that run's Phase 0 passed, and its
    ``run_id`` (if recorded) match. All fold records share one training
    population and one fold-0 training-row identity per arm.
    """
    out = []
    run_dir = path.resolve().parent
    rid = rec.get("run_id")
    if not rid or rid != run_dir.name:
        out.append(f"RUN: the authorization's run_id {rid!r} is not its run directory "
                   f"{run_dir.name!r}")
    if run_id is not None and rid != run_id:
        out.append(f"RUN: the authorization belongs to run {rid!r}, not {run_id!r}")
    if not _passed_phase0(run_dir):
        out.append(f"RUN: run {run_dir.name!r} has no passed Phase 0 record")
    keys = [k for k in ("g3_record", "configs_record", "g13_corrected_record", "test_report")
            if rec.get(k)]
    for key in keys:
        if Path(str(rec[key])).resolve().parent != run_dir:
            out.append(f"RUN: {key} is not in run {run_dir.name!r}'s directory")
    for key, body in current.items():
        if body is not None and body.get("run_id") != rid:
            out.append(f"RUN: {key} belongs to run {body.get('run_id')!r}, not {rid!r}")
    orid = rec.get("g13_record_run_id")
    odir = Path(str(rec.get("g13_record") or "")).resolve().parent
    if not orid or odir.name != orid or odir.parent != run_dir.parent:
        out.append(f"RUN: g13_record is not in the directory of its run g13_record_run_id "
                   f"{orid!r} beside this run")
    elif not _passed_phase0(odir):
        out.append(f"RUN: the original G13 run {orid!r} has no passed Phase 0 record")
    if "run_id" in original and original.get("run_id") != orid:
        out.append(f"RUN: g13_record belongs to run {original.get('run_id')!r}, not {orid!r}")
    folds = {k: b for k, b in current.items() if b is not None and k != "configs_record"}
    folds["g13_record"] = original
    for key, body in folds.items():
        if body.get("fold") != PREFLIGHT_FOLD:
            out.append(f"RUN: {key} is fold {body.get('fold')!r}, not the pre-declared fold "
                       f"{PREFLIGHT_FOLD}")
    pops = {k: _population_identity(b) for k, b in folds.items()}
    if None in pops.values() or len({json.dumps(v, sort_keys=True) for v in pops.values()}) != 1:
        out.append(f"RUN: the training-population identities of {sorted(pops)} differ or are "
                   "missing")
    for arm in (E.ARM_GLASS, E.ARM_RF):
        fps = {k: ((b.get("arms") or {}).get(arm) or {}).get("train_label_fingerprint")
               for k, b in folds.items()}
        if None in fps.values() or len(set(fps.values())) != 1:
            out.append(f"RUN: the {arm} fold-{PREFLIGHT_FOLD} training rows differ between "
                       f"{sorted(fps)}: {fps}")
    return out


def require_authorization(path, settings: Optional[ArmSettings] = None, *,
                          run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    The Step 2B authorization record. The five-fold primary is refused without:

    * every gate in ``AUTHORIZATION_GATES`` recorded ``"pass"``, an approver and a date;
    * ``g3_record`` (+ ``_sha256``): the preflight record, which must itself
      have passed G3 (feasible and every fit status accepted); no entry here
      can pass a G3 the record failed;
    * ``reviews``: ``{id: {"reviewer", "rationale", "date"}}`` for every review
      that record requires (``reviews_required``, e.g. the F31 strip review);
    * ``g13_record`` (+ ``_sha256``): the original G13 audit (historical
      pathway), and ``g13_decision`` (``{"decision", "materiality",
      "adopted_pathway", "adopted_record", "reviewer", "rationale", "date"}``);

    and then the evidence behind every gate, all problems reported together:

    * G13: both arms' audit evidence complete, reproduced (a GLASS
      non-reproduction only with a recorded disposition
      ``G13:<original|corrected>:glass:reproduction``), no test row; a material
      decision must adopt ``fold_label_excluded``, and adopting it needs
      ``g13_corrected_record`` (+ ``_sha256``): the audit of a corrected fit
      whose inner bins match the audit's own recomputation;
    * G2 / G3 / G5: the G3 record ran on the adopted pathway, both arms wired
      as the primary (``oof_remainder`` + ``fold_nested_rf``, ``nested_t1``),
      every evidence field and union count recorded;
    * G1 / G9: ``configs_record`` (+ ``_sha256``) equals the primary settings
      built now for the adopted pathway (frozen identities, checks, provenance);
    * G8: ``test_report`` (+ ``_sha256``), a JUnit report with no failure and a
      passing test from every module in ``AUTHORIZATION_TEST_MODULES``;
    * F31: ``f31_r2_clarification`` names the clarification record's identity,
      an approver and a date (the R2 reading is active only then);
    * ``settings``, when given: the run's pathway is the adopted one;
    * RUN (``_run_binding_problems``): ``run_id`` (the authorization's own, its
      directory's, and ``run_id`` when given) binds every record it names;
      ``g13_record_run_id`` names the original audit's run.
    """
    path = Path(path)
    if not path.is_file():
        raise RunnerError(f"no authorization record at {path}: the five-fold primary runs only "
                          "after Step 2B records every required gate as passed")
    rec = json.loads(path.read_text(encoding="utf-8"))
    gates = rec.get("gates") or {}
    failing = [g for g in AUTHORIZATION_GATES if gates.get(g) != "pass"]
    if failing or not rec.get("approved_by") or not rec.get("approved_on"):
        raise RunnerError(f"authorization incomplete: gates not passed {failing}; approver "
                          f"{rec.get('approved_by')!r} on {rec.get('approved_on')!r}")
    g3_body = _referenced_record(rec, "g3_record", G3_RECORD_FORMAT)
    g3 = g3_body.get("g3") or {}
    if g3.get("passed") is not True:
        raise RunnerError("authorization refused: the G3 record did not pass (feasible and every "
                          "fit status accepted); a review cannot replace that")
    reviews = rec.get("reviews") or {}
    missing = [r for r in g3.get("reviews_required") or [] if not _recorded(reviews.get(r))]
    if missing:
        raise RunnerError(f"authorization refused: mandatory reviews not recorded {missing}")
    original = _referenced_record(rec, "g13_record", G13_ORIGINAL_FORMATS)
    decision = rec.get("g13_decision")
    if not (_recorded(decision) and str(decision.get("decision", "")).strip()):
        raise RunnerError("authorization refused: no recorded G13 materiality decision")

    problems: List[str] = []
    corrected = configs = None
    dispositions = rec.get("dispositions") or {}
    adopted = decision.get("adopted_pathway")
    if decision.get("materiality") not in ("material", "not_material"):
        problems.append("G13: g13_decision.materiality must be 'material' or 'not_material'")
    if adopted not in IF.PATHWAYS:
        problems.append(f"G13: g13_decision.adopted_pathway must be one of {list(IF.PATHWAYS)}")
        adopted = None
    elif decision.get("adopted_record") != IF.pathway_identity(adopted)["record"]:
        problems.append(f"G13: g13_decision.adopted_record is not "
                        f"{IF.pathway_identity(adopted)['record']}")
    if decision.get("materiality") == "material" and adopted == IF.PATHWAY_OUTER:
        problems.append("G13: a material dependency needs the correction adopted; primary runs "
                        "without it are not primary evidence")
    problems += _g13_problems(original, label="original", pathway=IF.PATHWAY_OUTER,
                              dispositions=dispositions)
    if adopted == IF.PATHWAY_FOLD:
        try:
            corrected = _referenced_record(rec, "g13_corrected_record", G13_RECORD_FORMAT)
        except RunnerError as exc:
            problems.append(f"G13: the adopted correction has no valid corrected audit ({exc})")
        else:
            problems += _g13_problems(corrected, label="corrected", pathway=IF.PATHWAY_FOLD,
                                      dispositions=dispositions)
    if adopted is not None:
        problems += _g3_problems(g3_body, adopted)
        try:
            configs = _referenced_record(rec, "configs_record", CONFIGS_RECORD_FORMAT)
        except RunnerError as exc:
            problems.append(f"G1: {exc}")
        else:
            now = json.loads(json.dumps(E.to_json_safe(settings_record(primary_settings(adopted))),
                                        sort_keys=True))
            if configs.get("primary") != now:
                problems.append("G1: the configs record is not the primary settings built now "
                                "for the adopted pathway (frozen identities, checks, G9 provenance)")
            checks = (configs.get("primary") or {}).get("checks") or {}
            if not (checks.get("G9_rf_constraint_provenance")
                    and checks.get("G9_glass_contract_provenance")):
                problems.append("G9: the configs record has no independent contract provenance")
    try:
        problems += _test_report_problems(_referenced_file(rec, "test_report"))
    except RunnerError as exc:
        problems.append(f"G8: {exc}")
    clar = rec.get("f31_r2_clarification") or {}
    if clar.get("record") != E.record_identity(E.f31_r2_clarification()) or not (
            str(clar.get("approved_by", "")).strip() and str(clar.get("approved_on", "")).strip()):
        problems.append(f"F31: the R2 clarification {E.record_identity(E.f31_r2_clarification())} "
                        "is not approved (record identity, approved_by, approved_on)")
    problems += _run_binding_problems(path, rec, run_id, {
        "g3_record": g3_body, "configs_record": configs, "g13_corrected_record": corrected},
        original)
    if settings is not None and settings.feature_pathway != adopted:
        problems.append(f"the run's feature pathway {settings.feature_pathway!r} is not the "
                        f"adopted {adopted!r}")
    if problems:
        raise RunnerError("authorization refused: the evidence does not support the gates:\n  - "
                          + "\n  - ".join(problems))
    return rec
