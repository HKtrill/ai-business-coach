"""Fixed-parameter producers for the training-only PR 43 simulated holdout."""

import json
from functools import cached_property, partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from blackbox_pipeline.models.xgb.shared_protocol import XGBEstimator
from glass_pipeline.ebm.estimator import EBMFactory
from glass_pipeline.ebm.interactions import define_ebm_interactions
from shared.pinned_artifacts import sha256_file
from shared.stage1 import crossfit, oof_calibration as s1
from shared.stage1.feature_engineering import LR_FEATURES
from shared.stage3.calibration import Stage3Calibrator, calculate_ece
from shared.stage3.data import FoldPlan, Stage3Block
from shared.stage3.feature_engineering import EBMFeaturePipeline
from stage2_crossfit import runner as router

STATES = np.array(["pass1", "pass2", "uncertain"])


def routes(index, state, confidence, base):
    state, c, b = np.asarray(state), np.asarray(confidence), np.broadcast_to(base, len(index))
    if (
        not np.isin(state, STATES).all()
        or not ((0 <= c) & (c <= 1)).all()
        or not ((0 < b) & (b < 1)).all()
        or c[state == "uncertain"].any()
    ):
        raise ValueError("Invalid router evidence")
    p = np.select([state == "pass1", state == "pass2"], [b * (1 - c), b + (1 - b) * c], default=b)
    return pd.DataFrame(dict(route=state, router_score=p, router_confidence=c, router_base_rate=b), index=index)


def average_routes(frames):
    """Strict majority. A split vote defers. Combine only the supporting producers' routed evidence."""
    index = frames[0].index
    if any(not f.index.equals(index) for f in frames):
        raise ValueError("Unpaired router inputs")
    states = np.stack([f.route for f in frames])
    votes = np.stack([(states == s).sum(axis=0) for s in STATES])
    selected = np.where(votes.max(axis=0) > len(frames) / 2, STATES[votes.argmax(axis=0)], "uncertain")
    support = (states == selected) | (selected == "uncertain")

    def mean(col):
        return (support * np.stack([f[col] for f in frames])).sum(axis=0) / support.sum(axis=0)

    return routes(
        index, selected, np.where(selected == "uncertain", 0, mean("router_confidence")), mean("router_base_rate")
    )


def rules(model, fit, y, target):
    info, matches = [], []
    issued = np.zeros(len(target), bool)
    for direction, selected in [("pass1", model.pass1_rules), ("pass2", model.pass2_rules)]:
        for order, rule in enumerate(selected):
            hit = model._rules_mask([rule], fit)
            if direction == "pass2":
                hit &= model.fit_report_["remainder_mask"]
            independent = model._rules_mask([rule], target)
            matches.extend([independent, independent & ~issued])
            issued |= independent
            info.append(
                dict(
                    route=direction,
                    order=order,
                    id=rule.rule_id,
                    segment=list(rule.segment),
                    fit_rows=int(hit.sum()),
                    fit_positive=int(y[hit].sum()),
                    precision=float(rule.precision),
                    base=float(model.training_base_rate),
                )
            )
    return info, np.asarray(matches, dtype=bool).reshape(-1, len(target))


class Producers:
    def __init__(self, arm, x, y, h, stage1, params, features):
        self.arm, self.x, self.y, self.h = arm, x, y, h
        self.base, self.params, self.features = stage1, params, features
        self.calibration = {}
        self.folds = {
            key: FoldPlan.build(y, count, 42).fold_id
            for key, count in [("stage1", 10), ("router", 5), ("stage3", 5)]
        }

    @cached_property
    def settings(self):
        return router.primary_settings("fold_label_excluded")

    @cached_property
    def block(self):
        # The production builder requires label metadata on the scoring side. Real holdout labels are unavailable here.
        return Stage3Block.build(
            self.x,
            self.y,
            self.h,
            pd.Series(0, index=self.h.index),
            lambda: EBMFeaturePipeline(features=self.features),
            expected_features=self.features,
            n_eval_folds=2,
            verbose=False,
            build_reporting_frames=False,
        )

    def stage1(self, k, tr, va, target, raw_oof):
        x, y = self.x.iloc[tr], self.y.iloc[tr]
        eng = crossfit._quiet_engineer(None, x, y)
        design = crossfit._quiet_transform(eng, x, LR_FEATURES)
        scaler = StandardScaler().fit(design)
        fit = scaler.transform(design)
        score = scaler.transform(crossfit._quiet_transform(eng, target, LR_FEATURES))
        if k == -1:
            model = s1.clone(self.base).fit(fit, y)
            raw = s1.base_scores(model, score)
            mapped = s1._calibrated(s1.make_calibrator("sigmoid").fit(raw_oof, y), raw)
        else:
            # _outer_fold fits the outer model, ten inner models and its sigmoid once.
            d = crossfit.FoldDesign(k, tr, va, fit, score, y.to_numpy(), self.y.iloc[va].to_numpy())
            names = pd.DataFrame(index=self.x.index, columns=LR_FEATURES)
            _, _, raw, _, cal, _, _ = s1._outer_fold(
                self.base, d, names, self.y, self.x, 10, 42, ("sigmoid",), None
            )
            mapped = cal["sigmoid"]
        return dict(raw=raw, stage1_proba=mapped), dict(base_fits=1 if k == -1 else 11, maps=1)

    def router(self, k, tr, va, target, unused):
        x, y, cfg = self.x.iloc[tr], self.y.iloc[tr], self.settings
        if self.arm == "glass":
            # A failed or interrupted solver attempt requires review before another fit.
            folder = self.audit_root / f"router_{k}"
            folder.mkdir(parents=True, exist_ok=False)
            hashes = {}

            def save(name, value):
                path = folder / (name + ".json")
                with path.open("x", encoding="utf8") as stream:
                    json.dump(router.E.to_json_safe(value), stream, allow_nan=False)
                hashes[name] = sha256_file(path)

            fit = partial(
                router.fit_glass_fold,
                x,
                y,
                target,
                cfg.glass_config,
                cfg.rf_params,
                random_state=42,
                glass_rf_model=cfg.ranker["glass_rf_model"],
                rule_logger=router.RuleLogger(enabled=False),
                inner_features=cfg.glass_inner_features,
            )
            save("attempt", {"binding": self.audit_binding, "producer": k})
            f = fit()
            m, xt = f.glass, f.X_va_eng
            audit = router.E.glass_fold_evidence(
                m, f.X_tr_eng, y, fold=k, heldout_index=target.index, inner_features=f.inner_features
            )
            save("primary", audit)

            def reproduce():
                save("m3_started", {"producer": k})
                repeated = fit()
                save("m3_reports", repeated.glass.selection_reports())
                return repeated.glass

            router.document_unproven(audit, reproduce)
        else:
            xx, xt = router.engineer_features(x, target, y)
            m = router.RFRouter(cfg.rf_config_factory())
            provider = router._rf_fit(m, xx, x, y, router.split_fingerprint(x, target), cfg)
            audit = router.E.rf_fold_evidence(
                m, xx, y, fold=k, heldout_index=target.index, inner_features=provider
            )
        status = router.fit_status(audit, cfg)
        if self.arm == "glass":
            save("decision", {"status": status, "m3": audit.get(router.E.M3_KEY)})
        if not status["accepted"]:
            raise ValueError(f"Router {k} requires review: {status}")
        _, c, state = m.predict(xt)
        output = routes(target.index, state, c, m.training_base_rate)
        arrays = {col: output[col].to_numpy(dtype=str if col == "route" else float) for col in output}
        info = dict(router_objects=1, status=status)
        if self.arm == "glass":
            info.update(audit_sha256=hashes, m3_router_fits=int("m3_started" in hashes))
            info["rules"], arrays["matches"] = rules(m, f.X_tr_eng, y, xt)
            info["ordering_forests"] = 1 + (
                len(m.inner_pass1_rules_) if m.inner_feature_ranker == "fold_nested_rf" else 0
            )
        else:
            info["forests"] = 2 + m.config.n_oof_folds * (2 + m.config.eligibility_folds)
        return arrays, info

    def stage3(self, k, tr, va, target, raw_oof):
        b = self.block
        est = (
            XGBEstimator(n_jobs=1)
            if self.arm == "blackbox"
            else EBMFactory(define_ebm_interactions(b.X_train), n_jobs=1)
        )
        if k == -1:
            model = est.fit(self.params, b.X_train, self.y, sample_weight=b.weights.vector)
            raw, inner = est.positive_proba(model, b.X_test), raw_oof
        else:
            f = b.tune_folds[k]
            np.testing.assert_array_equal(f.val_idx, va)
            model = est.fit(self.params, f.X_tr, f.y_tr, sample_weight=f.w_tr)
            raw = np.r_[est.positive_proba(model, f.X_val), est.positive_proba(model, f.X_test)]
            inner = pd.Series(np.nan, index=self.y.index[tr])
            for g in b.inner_folds[k]:
                m = est.fit(self.params, g.X_tr, g.y_tr, sample_weight=g.w_tr)
                inner.loc[g.X_val.index] = est.positive_proba(m, g.X_val)
        if not np.isfinite(inner).all():
            raise ValueError("Missing cross-fitted calibration scores")
        cal = Stage3Calibrator(**self.calibration)
        labels = self.y.iloc[tr].to_numpy()
        gate = cal.force or calculate_ece(labels, inner, cal.ece_bins) > cal.ece_threshold
        mapped = cal._apply(cal._fit_mapper(np.asarray(inner), labels), raw) if gate else raw
        return dict(raw=raw, stage3_raw=raw, stage3_proba=mapped), dict(
            base_fits=1 if k == -1 else 6, maps=int(gate)
        )


def independent_evidence(data, y, held_index):
    """Train-side gate and transfer. No simulated holdout labels. Use first-issued matches for precision."""
    from collections import defaultdict

    groups = defaultdict(lambda: np.zeros((3, 3)))
    inner = pd.Series(FoldPlan.build(y, 3, 42).fold_id, index=y.index)
    records, full = [], []
    for k, rows, masks, va in data:
        if len(masks) != 2 * len(rows):
            raise ValueError("Rule mask ledger differs from selection order")
        for i, r in enumerate(rows):
            segment = (r["route"], tuple(sorted(tuple(v) for v in r["segment"])))
            match, issued = masks[2 * i : 2 * i + 2]
            if (issued & ~match).any() or len(match) != len(va) + len(held_index):
                raise ValueError("Invalid ordered router masks")
            entry = dict(r, producer=k)
            for key, mask in [("matched", match), ("issued", issued)]:
                index = va[mask[: len(va)]]
                entry[key] = dict(count=len(index), positive=int(y.loc[index].sum()))
                entry["holdout_" + key] = int(mask[len(va) :].sum())
            if k == -1:
                full.append((segment, issued))
            else:
                issued_index = va[issued[: len(va)]]
                for j in range(3):
                    ii = issued_index[inner.loc[issued_index].to_numpy() == j]
                    correct = int((y.loc[ii] == int(r["route"] == "pass2")).sum())
                    groups[segment][j] += [len(ii), correct, len(ii) * r["precision"]]
            records.append(entry)
    rates, triggers, gate = {}, [], []
    for key, g in groups.items():
        count, correct, expected = g.T
        gap = np.divide(expected - correct, count, out=np.zeros(3), where=count > 0)
        passed = (count >= 100) & (gap >= 0.05 - 1e-14)
        if passed.sum() >= 2:
            triggers.append(key)
        supported = np.count_nonzero(count >= 100) >= 2
        if supported:
            rates[key] = correct.sum() / count.sum()
        gate.append(
            dict(
                rule=key,
                counts=count,
                independent_correct=correct,
                transfer_supported=bool(supported),
                gaps=[v if num else None for v, num in zip(gap, count)],
                eligible=bool(passed.sum() >= 2),
            )
        )
    confidence = np.zeros(len(held_index))
    unsupported = []
    for key, issued in full:
        if issued.any() and key not in rates:
            unsupported.append(dict(rule=key, rows=int(issued.sum())))
        elif key in rates:
            confidence[issued] = rates[key]
    return dict(
        producers=records,
        inner_gate=gate,
        triggered_rules=triggers,
        unsupported=unsupported,
        transfer="Match pass and canonical segment. Pool only independent first-issued fitting OOF matches. "
        "Transfer requires >=100 independent issues in each of at least two fitting inner folds; "
        "the discrepancy threshold activates the ablation separately. Selected-rule fitting precision uses "
        "matching rows on the applicable pass population (Pass 2 remainder); independent precision uses "
        "only first-issued rows after earlier rules. The gap can include overlap/order effects.",
    ), (confidence if triggers and not unsupported else None)
