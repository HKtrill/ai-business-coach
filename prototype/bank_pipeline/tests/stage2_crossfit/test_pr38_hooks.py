"""
PR 38 Step 2A: the opt-in hooks the runner uses in existing modules.

1. ``crossfit_stage2(fold_callback=, report_outer=)``: the callback sees each
   fitted fold before any held-out row is scored; ``report_outer=False``
   prints no outer-OOF metric; the defaults leave the historical run unchanged.
   ``fit_glass_fold`` attaches the unfinished fit to an exception it re-raises.
2. ``crossfit_rf_router(fold_callback=)``: the same for the RF Router.
3. ``ModelSaver.save_model(extra_fields=)`` adds fields, never replaces one.
4. Stage 4 (F31 R5): the RF Router's eligibility rule is recorded beside its
   Pass 2 population; a cross-fit / full-train disagreement or a half
   declaration is refused; an artifact that declares neither (written before
   either side recorded it) is recorded as undeclared and still accepted.
"""

from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
import pytest

import glass_pipeline.glass_router.pipeline.crossfit as crossfit_mod
from blackbox_pipeline.models.rf_router import RFRouter, crossfit_rf_router
from glass_pipeline.glass_router.model_saver import ModelSaver
from glass_pipeline.glass_router.rule_generator.rule_logger import RuleLogger
from shared.stage4 import router_stream
from shared.stage4.upstream import upstream_record
from tests.stage2.conftest import TINY_RF, legacy_config, run_crossfit
from tests.support import stage2_crossfit_world as W
from tests.support.rf_router_world import (
    bank_world, engineer, router_config, serial_forests, stage4_block,
)
from tests.support.synthetic_bank import make_bank_frame, split_frame


# ======================================================================
# 1. GLASS cross-fit hooks
# ======================================================================
@pytest.fixture(scope="module")
def split():
    return split_frame(make_bank_frame(n=600, seed=5), test_size=0.25)


def test_the_glass_callback_sees_each_fold_before_it_is_scored(split, monkeypatch):
    seen = []

    def cb(k, fit, tr, va):
        seen.append((k, len(fit.glass.seen), len(tr), len(va)))       # predict calls so far
        return {"fold": k}

    run = run_crossfit(split)                                        # defaults: no hook
    assert run.oof.fold_evidence == []
    with monkeypatch.context() as mp:
        orig = crossfit_mod.crossfit_stage2
        mp.setattr(crossfit_mod, "crossfit_stage2",
                   lambda *a, **k: orig(*a, **{**k, "fold_callback": cb}))
        hooked = run_crossfit(split)
    assert [s[0] for s in seen] == [0, 1, 2, 3, 4] and all(s[1] == 0 for s in seen)
    assert hooked.oof.fold_evidence == [{"fold": k} for k in range(5)]
    np.testing.assert_array_equal(hooked.oof.glass_pred, run.oof.glass_pred)


def test_report_outer_false_prints_no_outer_oof_metric(split, monkeypatch, capsys):
    printed = {}
    for flag in (True, False):
        out = io.StringIO()
        with monkeypatch.context() as mp:
            orig = crossfit_mod.crossfit_stage2

            def call(*a, **k):
                with contextlib.redirect_stdout(out):
                    return orig(*a, **{**k, "report_outer": flag})
            mp.setattr(crossfit_mod, "crossfit_stage2", call)
            run_crossfit(split)
        printed[flag] = out.getvalue()
    assert "OOF GLASS Pass 1 leakage" in printed[True] and "RF AUC" in printed[True]
    for token in ("OOF GLASS", "RF AUC", "leak"):
        assert token not in printed[False], token
    assert "Fold fingerprint" in printed[False]


def test_an_exception_carries_the_unfinished_fold_fit(monkeypatch):
    W.wired(monkeypatch, W.wired_glass_class(script={1: "infeasible"}))
    ts = W.synthetic_train_side()
    X_tr, y_tr, X_va = ts.fold_rows(0)
    with pytest.raises(Exception) as info, contextlib.redirect_stdout(io.StringIO()):
        crossfit_mod.fit_glass_fold(X_tr, y_tr, X_va, W.make_settings().glass_config,
                                    TINY_RF, glass_rf_model="fold_rf")
    partial = info.value.partial_fold_fit
    assert partial.glass.selection_reports()["pass1"]["source"] == "infeasible"


# ======================================================================
# 2. RF cross-fit hook
# ======================================================================
def test_the_rf_callback_sees_each_fold_router_before_it_routes(monkeypatch):
    world = bank_world()
    routed, seen = [], []
    real = RFRouter.predict_with_proba

    def spy(self, X, *a, **k):
        routed.append(len(X))
        return real(self, X, *a, **k)

    monkeypatch.setattr(RFRouter, "predict_with_proba", spy)

    def cb(k, router, X_tr_eng, y_tr):
        seen.append((k, len(routed), router.is_fitted, len(X_tr_eng) == len(y_tr)))
        return {"fold": k, "t1": router.thresholds_.t1}

    with serial_forests():
        cf = crossfit_rf_router(world.global_split["X_train"], world.stage2["y_train"], world.folds,
                                engineer=engineer, make_config=router_config,
                                split_fp=world.stage2["split_fingerprint"],
                                fold_fp=world.fold_fp, random_state=42, verbose=False,
                                fold_callback=cb)
    assert [s[1] for s in seen] == [0, 1, 2, 3, 4]                     # before each routing
    assert all(s[2] and s[3] for s in seen)
    assert [e["t1"] for e in cf.fold_evidence] == list(cf.folds["t1"])
    np.testing.assert_array_equal(cf.oof["decisions"], world.crossfit.oof["decisions"])


# ======================================================================
# 3. ModelSaver
# ======================================================================
def test_extra_fields_are_added_but_never_replace_a_bundle_field(tmp_path, split):
    run = run_crossfit(split)
    with contextlib.redirect_stdout(io.StringIO()):
        glass = W.wired_glass_class()(legacy_config(), rule_logger=RuleLogger(enabled=False))
    glass.fit_report_ = {"pass2_population": "full"}
    glass.training_base_rate = 0.2
    n_te = len(split["X_test"])
    test_out = {"pred": np.zeros(n_te, int), "confidence": np.zeros(n_te),
                "decisions": np.array(["uncertain"] * n_te, dtype=object),
                "covered": np.zeros(n_te, bool), "abstained": np.ones(n_te, bool)}
    kw = dict(train_out=None, train_proba=None, oof=run.oof, test_out=test_out,
              test_proba=np.full((n_te, 2), 0.5), split_fingerprint="fp",
              global_split={"X_train": split["X_train"], "X_test": split["X_test"]})
    with contextlib.redirect_stdout(io.StringIO()):
        saver = ModelSaver(glass_model=glass)
        with pytest.raises(ValueError, match="overwrite"):
            saver.save_model(**kw, output_path=str(tmp_path / "a.joblib"),
                             extra_fields={"oof_protocol": {}})
        path = saver.save_model(**kw, output_path=str(tmp_path / "b.joblib"),
                                extra_fields={"pr38_evidence": {"x": 1}})
    import joblib
    assert joblib.load(path)["pr38_evidence"] == {"x": 1}


# ======================================================================
# 4. Stage 4 R5
# ======================================================================
def _rf_artifact(cross, full):
    world = bank_world()
    art = {"mode": "two_pass", "config": {"eligibility": full} if full else {},
           "training_base_rate": float(world.router.training_base_rate),
           "split_fingerprint": world.stage2["split_fingerprint"],
           "train_label_fingerprint": world.router.train_label_fingerprint_,
           "extra": copy.deepcopy(stage4_block(world.stage2, world.crossfit, world.fold_id,
                                               world.test_out))}
    proto = art["extra"]["oof_protocol"]
    if cross is None:
        proto.pop("pass2_eligibility", None)
    else:
        proto["pass2_eligibility"] = cross
    return art, world.stage2["y_train"].to_numpy()


def _r5(cross, full):
    art, y = _rf_artifact(cross, full)
    stream, checks = router_stream(art, "rf_router", "RF Router", y)
    r5 = [c for c in checks if "eligibility rule" in c[2]]
    return stream, r5


@pytest.mark.parametrize("cross, full, ok", [
    ("nested_t1", "nested_t1", True), ("global_t1", "global_t1", True),
    ("nested_t1", "global_t1", False), ("global_t1", "nested_t1", False),
    ("nested_t1", None, False), (None, "nested_t1", False), ("other", "other", False)])
def test_stage4_records_the_rf_eligibility_rule_and_refuses_a_mismatch(cross, full, ok):
    stream, r5 = _r5(cross, full)
    assert stream.notes["pass2_eligibility"] == {"cross_fit": cross, "full_train": full}
    assert len(r5) == 1 and r5[0][3]() is ok


def test_an_artifact_declaring_no_eligibility_rule_is_recorded_as_undeclared():
    stream, r5 = _r5(None, None)
    assert r5 == [] and stream.notes["pass2_eligibility"] == {"cross_fit": None,
                                                              "full_train": None}
    rec = upstream_record([stream], "blackbox", "meta_xgb", "blackbox")
    assert rec["stages"]["stage2"]["pass2_eligibility"] == {"cross_fit": None, "full_train": None}


def test_the_glass_router_records_no_eligibility_rule(split):
    run = run_crossfit(split)
    assert "pass2_eligibility" not in (run.oof.protocol or {})
