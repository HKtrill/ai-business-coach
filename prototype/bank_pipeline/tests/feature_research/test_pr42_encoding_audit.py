"""
PR 42 Commit 3, Step 3A: the encoding construction and leakage audit
(``feature_research.encoding_audit``), on synthetic training rows.

The rows carry a designed calendar (periods, contact days, monthly indicators),
so the time index and every relation count can be checked exactly; the
attribution is checked against the production encoders it explains; the fold
isolation checks must pass on the production builders and fail on a leaky one.
The real training rows are covered by ``tests/realdata/test_pr42_encoding_audit.py``.
"""

from __future__ import annotations

import ast
import shutil

import numpy as np
import pandas as pd
import pytest

from feature_research import encoding_audit as EA
from tests.support.notebooks import BANK_ROOT
from tests.support.synthetic_bank import TARGET, make_bank_frame

#: (year, month, contact days) in source order: the documented first and last
#: periods, with the year wrapping where the month decreases.
CALENDAR = ((2008, 5, 6), (2008, 11, 5), (2009, 3, 5), (2009, 12, 4), (2010, 3, 5), (2010, 11, 6))


def _dated(n_per_day: int = 24, seed: int = 3) -> pd.DataFrame:
    """Bank-like rows in a designed date order; index labels are gapped source lines, rows shuffled."""
    rows = []
    day = 0
    for p, (y, m, k) in enumerate(CALENDAR):
        for d in range(k):
            rows += [{"month": m, "day_of_week": d % 5, "euribor3m": round(4.9 - 0.013 * day, 3),
                      "cons.price.idx": 93.0 + 0.11 * p, "cons.conf.idx": -40.0 + m,
                      "emp.var.rate": 1.1 - 0.4 * (y - 2008),
                      "nr.employed": 5200.0 - 30 * (y - 2008)}] * n_per_day
            day += 1
    cal = pd.DataFrame(rows)
    frame = make_bank_frame(n=len(cal), seed=seed).reset_index(drop=True)
    for c in cal:
        frame[c] = cal[c].astype(frame[c].dtype)
    frame.index = pd.Index(3 * np.arange(len(cal)) + 1, dtype="int64")   # source lines, gapped
    return frame.sample(frac=1.0, random_state=seed)                     # split order != source order


@pytest.fixture(scope="module")
def dated():
    frame = _dated()
    return frame.drop(columns=[TARGET]), frame[TARGET].astype(int)


# ----------------------------------------------------------------------
# time index
# ----------------------------------------------------------------------
def test_the_time_index_recovers_the_designed_calendar(dated):
    X, _ = dated
    T = EA.build_time_index(X)
    assert T.index.equals(X.index)                                   # X's row order
    s = T.sort_values("position")
    assert list(s["period"].unique()) == [f"{y}-{m:02d}" for y, m, _ in CALENDAR]
    assert s["day"].nunique() == sum(k for *_, k in CALENDAR)
    assert (np.diff(s["day"].to_numpy()) >= 0).all()                 # days follow source order
    first = s.groupby("period_index")["day"].min().to_numpy()
    nxt = np.r_[first[1:], s["day"].max() + 1]
    assert np.array_equal(s["next_period_day"].to_numpy(), nxt[s["period_index"].to_numpy()])
    checks = EA.time_index_checks(X, T)
    assert checks["passed"].all(), checks.to_string()


def test_the_time_index_checks_catch_a_misplaced_period(dated):
    X, _ = dated
    bad = X.copy()
    late = bad.index[bad.index > bad.index.max() - 30]
    bad.loc[late, "month"] = 6                       # 2009-05 rows that read as June: a new period
    checks = EA.time_index_checks(bad, EA.build_time_index(bad)).set_index("check")["passed"]
    assert not checks.all()


def test_the_resolution_states_what_the_index_cannot_order():
    for key in ("unit", "within_a_day", "outcome_timing", "training_rows_only"):
        assert EA.RESOLUTION[key]
    assert EA.RELATIONS == ("earlier_day", "same_day", "later_day_same_month", "later_month")
    assert EA.LATER == ("later_day_same_month", "later_month")


# ----------------------------------------------------------------------
# source order (a small database built the production way)
# ----------------------------------------------------------------------
RAW_HEADER = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
              "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
              "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]


def _raw_csv(path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    jobs = ["admin.", "blue-collar", "technician", "services"]
    lines = [";".join(f'"{h}"' for h in RAW_HEADER)]
    for i in range(n):
        lines.append(";".join(map(str, [
            30 + i % 40, f'"{jobs[i % 4]}"', '"married"', '"high.school"', '"no"', '"yes"', '"no"',
            '"cellular"', '"may"' if i < n // 2 else '"jun"',
            f'"{["mon", "tue", "wed", "thu", "fri"][i // 6 % 5]}"', 100 + i, 1 + i % 3, 999, 0,
            '"nonexistent"', 1.1, 93.994 if i < n // 2 else 94.465, -36.4, round(4.85 - 0.001 * (i // 6), 3),
            5191, '"yes"' if rng.random() < 0.3 else '"no"'])))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_source_check_passes_on_the_production_load_and_fails_on_a_misaligned_row(tmp_path):
    from data.bank_database import build_database
    from data.preprocessing.bank_preprocessing import BankPreprocessor

    csv_path, db_path = tmp_path / "bank.csv", tmp_path / "bank.db"
    _raw_csv(csv_path)
    build_database(csv_path, db_path)
    df = BankPreprocessor(drop_leaky=True, verbose=False, db_path=db_path).fit_transform()
    train = df.sample(frac=0.8, random_state=1)                      # a shuffled training subset
    X, y = train.drop(columns=["y"]), train["y"].astype(int)
    checks = EA.verify_source_order(X, y, csv_path=csv_path, db_path=db_path)
    assert checks["passed"].all(), checks.to_string()
    a, b = X.index[0], next(i for i in X.index[1:] if X.loc[i, "age"] != X.loc[X.index[0], "age"])
    swapped = X.rename(index={a: b, b: a})                           # two rows under each other's line
    checks = EA.verify_source_order(swapped, y.rename(index={a: b, b: a}), csv_path=csv_path,
                                    db_path=db_path).set_index("check")["passed"]
    assert not checks.all()


# ----------------------------------------------------------------------
# attribution
# ----------------------------------------------------------------------
def test_attribute_splits_the_label_mass_by_contact_timing_exactly():
    # one group of five fit rows on days 0, 0, 1, 2, 3; day 3 opens the next period
    member = np.zeros(5, dtype=np.int64)
    block = EA.Block("bin", "direct", member, np.zeros(2, dtype=np.int64), np.full((1, 1), 0.2))
    fit_day = np.array([0, 0, 1, 2, 3])
    next_day = np.array([3, 3, 3, 4, 4])                     # first day of the period after each day
    y_fit = np.array([1, 0, 1, 0, 1])
    res = EA.attribute([block], y_fit, fit_day, np.array([1, 1]), next_day, np.array([2, -1]))
    assert np.allclose(res["value"], 0.6)
    m = res["mass"]["direct"]
    assert np.allclose(m["earlier_day"], 0.4) and np.allclose(m["later_day_same_month"], 0.2)
    assert np.allclose(m["later_month"], 0.2)
    assert np.allclose(res["self"], [0.2, 0.0])                      # row 0 is fit row 2 (self)
    assert np.allclose(m["same_day"], [0.0, 0.2])                    # row 1 is a new contact that day
    assert res["rows"]["direct"]["later_month"].tolist() == [1, 1]


def test_the_curvature_operator_is_numpys_double_gradient():
    r = np.random.default_rng(0).random(7)
    assert np.allclose(EA._gradient2(7) @ r, np.gradient(np.gradient(r)))


def _subset(paths, keep):
    return {stage: [p for p in ps if keep(p)] for stage, ps in paths.items()}


@pytest.fixture(scope="module")
def audited(dated):
    """Fold 0 of every path and the final fits; the KDE samples its reference rows (threshold lowered)."""
    import shared.stage2.feature_engineering as s2
    import stage2_crossfit.g13 as g13
    import feature_research.feature_engineering.integrals as s3

    X, y = dated
    T = EA.build_time_index(X)
    paths = EA.fit_paths(y)
    small = _subset(paths, lambda p: p.path == "final" or p.fold == 0 and p.inner in (None, 0))
    with pytest.MonkeyPatch.context() as mp:
        for module in (s2, g13, s3):
            mp.setattr(module, "_KDE_LARGE_DS_THRESH", 300)
        return X, y, T, small, EA.audit_paths(X, y, T, small, verbose=False)


def test_every_expected_comparison_ran_and_reconstructs_the_production_encoders(audited):
    *_, small, audit = audited
    coverage = EA.comparison_coverage(audit, small)
    assert coverage["passed"].all(), coverage[~coverage["passed"]].to_string()
    summary = EA.path_summary(audit)
    assert set(summary["feature"]) == {*EA.STAGE1_FEATURES, *EA.STAGE23_FEATURES}
    assert set(summary["role"]) == {"held_out", "fit_rows", "final_mapping"}
    assert summary["comparisons_failed"].sum() == 0 and summary["nonfinite_values"].sum() == 0
    skipped = summary[summary["comparisons_skipped"] > 0]
    assert set(zip(skipped["stage"], skipped["role"])) == {("stage2", "fit_rows")}   # the policy's skips
    assert (summary["decay_x_density_failed"].dropna() == 0).all()
    assert summary["decay_x_density_passed"].sum() == sum(1 if p.path == "final" else 2
                                                          for p in small["stage3"])


def test_compare_fails_on_any_nonfinite_value_and_keeps_skips_apart():
    good = np.array([0.1, 0.2, 0.3])
    assert EA._compare(good, good.copy())["status"] == "passed"
    assert EA._compare(good, None)["status"] == "skipped"
    for rec, prod in ((np.array([0.1, np.nan, 0.3]), good), (good, np.array([0.1, 0.2, np.inf])),
                      (good, good[:2]), (np.array([np.nan]), np.array([np.nan])),
                      (good, good + 1e-9)):
        assert EA._compare(rec, prod)["status"] == "failed"
    out = EA._compare(np.array([0.1, np.nan]), np.array([0.1, 0.5]))
    assert out["nonfinite_reconstructed"] == 1 and out["max_abs_diff"] == 0.0    # the finite pair agrees


@pytest.fixture(scope="module")
def two_outer_folds(dated):
    X, y = dated
    paths = EA.fit_paths(y)
    return X, y, EA.build_time_index(X), {"stage1": [p for p in paths["stage1"]
                                                    if p.path == "outer" and p.fold in (0, 1)]}


def _corrupt(monkeypatch, fold, how):
    real = EA._stage1_fit

    def fit(X, y, path):
        fitted = real(X, y, path)
        if path.fold == fold:
            key = ("subscribe_rate_by_dow_month", "held_out")
            if how == "nan":
                fitted.values[key] = fitted.values[key].copy()
                fitted.values[key][0] = np.nan
            else:
                fitted.values[key] = None                    # a comparison that was never made
        return fitted

    monkeypatch.setattr(EA, "_stage1_fit", fit)


@pytest.mark.parametrize("how", ["nan", "missing"])
def test_one_bad_comparison_fails_the_coverage_although_another_fold_passes(two_outer_folds, monkeypatch, how):
    X, y, T, paths = two_outer_folds
    _corrupt(monkeypatch, 1, how)
    audit = EA.audit_paths(X, y, T, paths, verbose=False)
    fits = pd.DataFrame(audit["fits"]).set_index(["fold", "feature", "role"])
    assert fits.loc[(0, "subscribe_rate_by_dow_month", "held_out"), "comparison"] == "passed"
    assert fits.loc[(1, "subscribe_rate_by_dow_month", "held_out"), "comparison"] == (
        "failed" if how == "nan" else "skipped")
    coverage = EA.comparison_coverage(audit, paths).set_index("check")["passed"]
    bad = coverage[~coverage]
    assert len(bad) == 2 and any("held_out subscribe_rate_by_dow_month" in c for c in bad.index)
    with pytest.raises(AssertionError, match="reconstruction"):
        EA.require(EA.comparison_coverage(audit, paths), "reconstruction")


def test_a_nonfinite_product_fails_the_decay_x_density_check(dated, monkeypatch):
    from shared.stage3.feature_engineering import EBMFeaturePipeline

    X, y = dated
    paths = EA.fit_paths(y)
    small = {"stage3": [p for p in paths["stage3"] if p.path == "tuning" and p.fold in (0, 1)]}
    real = EBMFeaturePipeline.transform

    def transform(self, X_):
        out = real(self, X_)
        if len(X_) < 200:                                    # the held-out rows of these small folds
            out = out.copy()
            out.iloc[0, out.columns.get_loc("decay_x_density")] = np.nan
        return out

    monkeypatch.setattr(EBMFeaturePipeline, "transform", transform)
    audit = EA.audit_paths(X, y, EA.build_time_index(X), small, verbose=False)
    coverage = EA.comparison_coverage(audit, small).set_index("check")["passed"]
    product = [c for c in coverage.index if "decay_x_density" in c]
    assert product and not coverage[product].any()


def test_held_out_rows_never_carry_their_own_label_and_fit_rows_do(audited):
    *_, audit = audited
    s = EA.path_summary(audit)
    assert (s.loc[s["role"] != "fit_rows", "self_rows"] == 0).all()
    assert EA.held_out_own_labels(s)["passed"].all()
    fit = s[(s["role"] == "fit_rows") & (s["feature"] != "neighborhood_subscription_density")]
    assert (fit["self_rows"] >= 0.99 * fit["rows"]).all()       # rounded edges may map a row next door
    nsd = s[(s["role"] == "fit_rows") & (s["feature"] == "neighborhood_subscription_density")]
    assert (nsd["self_rows"] == nsd["nsd_label_rows_min"] * nsd["fits"]).all()   # reference rows only


def test_shares_add_up_and_the_final_mapping_counts_its_own_row_as_same_day(audited):
    *_, audit = audited
    fits = pd.DataFrame(audit["fits"])
    total = sum(fits[f"{k}_{x}_share_sum"] for k in ("direct", "prior") for x in EA.RELATIONS)
    whole = fits["feature"] != "economic_curvature_intensity"   # a row in no curvature bin has no labels
    assert np.allclose((total + fits["self_share_sum"])[whole], fits.loc[whole, "rows"])
    assert ((total + fits["self_share_sum"]) <= fits["rows"] + 1e-9).all()
    final = fits[fits["path"] == "final"].set_index(["feature", "stage", "role"])
    for feat, stage in {(f, s) for f, s, _ in final.index}:
        a, b = final.loc[(feat, stage, "fit_rows")], final.loc[(feat, stage, "final_mapping")]
        assert np.isclose(a["later_share_sum"], b["later_share_sum"])
        assert b["self_rows"] == 0
    table = EA.mapping_by_period(audit)
    assert set(table["period"]) == {f"{y}-{m:02d}" for y, m, _ in CALENDAR}


def test_the_label_excluded_inner_fold_uses_no_label_of_its_fold(dated):
    X, y = dated
    T = EA.build_time_index(X)
    paths = EA.fit_paths(y)
    inner = [p for p in paths["stage2"] if p.path == "inner_label_excluded" and p.fold == 0][:1]
    outer = [p for p in paths["stage2"] if p.path == "outer" and p.fold == 0]
    audit = EA.audit_paths(X, y, T, {"stage2": outer + inner}, verbose=False)
    s = EA.path_summary(audit).set_index(["feature", "path", "role"])
    nsd = "neighborhood_subscription_density"
    kept = s.loc[(nsd, "inner_label_excluded", "held_out"), "nsd_label_rows_max"]
    assert kept < s.loc[(nsd, "outer", "held_out"), "nsd_label_rows_max"]   # its fold's references dropped
    assert s.loc[(nsd, "inner_label_excluded", "held_out"), "max_abs_diff"] <= 1e-12
    # the held-out rows are in the structure (located), yet carry no label of their own
    held = s.xs(("inner_label_excluded", "held_out"), level=("path", "role"))
    assert (held["self_rows"] == 0).all() and np.isin(inner[0].query, inner[0].structure).all()


def _leaky(path, rows):
    """The same fit with some of its held-out rows' labels deliberately included."""
    return EA.FitPath(path.stage, path.path, path.fold, path.inner, np.sort(np.r_[path.fit, rows]),
                      path.query, structure=path.structure)


def _cell_self_weight(X, fit, row, k=100.0):
    """Own-label weight of a fit row in a Laplace day x month rate: 1/(n+k) + k/((n+k)N)."""
    key = X["day_of_week"].to_numpy() * 100 + X["month"].to_numpy()
    n = float((key[fit] == key[row]).sum())
    return 1 / (n + k) + k / ((n + k) * len(fit))


def test_a_deliberately_included_held_out_label_is_measured_on_stage1(dated):
    from shared.stage1.feature_engineering import LRFeatureEngineer

    X, y = dated
    T = EA.build_time_index(X)
    outer = [p for p in EA.fit_paths(y)["stage1"] if p.path == "outer" and p.fold == 0][0]
    leaked = outer.query[0]
    leaky = _leaky(outer, [leaked])
    audit = EA.audit_paths(X, y, T, {"stage1": [leaky]}, verbose=False)
    rows = {r["feature"]: r for r in audit["fits"] if r["role"] == "held_out"}
    cell = rows["subscribe_rate_by_dow_month"]
    weight = _cell_self_weight(X, leaky.fit, leaked)
    assert cell["self_rows"] == 1 and cell["comparison"] == "passed"
    assert np.isclose(cell["self_share_max"], weight, rtol=0, atol=1e-15)
    # the production encoder moves the leaked row by exactly that weight when its label flips
    yy = y.copy()
    yy.iloc[leaked] = 1 - yy.iloc[leaked]
    before, after = (EA._quiet(EA._quiet(LRFeatureEngineer().fit, X.iloc[leaky.fit], lab.iloc[leaky.fit])
                               .transform, X.iloc[[leaked]])["subscribe_rate_by_dow_month"].iloc[0]
                     for lab in (y, yy))
    assert np.isclose(abs(after - before), weight, rtol=0, atol=1e-15)
    assert not EA.held_out_own_labels(EA.path_summary(audit))["passed"].all()


def test_a_deliberately_kept_label_is_measured_on_a_stage2_label_excluded_fold(dated):
    X, y = dated
    T = EA.build_time_index(X)
    paths = EA.fit_paths(y)
    outer = [p for p in paths["stage2"] if p.path == "outer" and p.fold == 0]
    inner = [p for p in paths["stage2"] if p.path == "inner_label_excluded" and p.fold == 0][0]
    leaked = inner.query[0]                                       # in the structure, label now kept
    leaky = _leaky(inner, [leaked])
    audit = EA.audit_paths(X, y, T, {"stage2": outer + [leaky]}, verbose=False)
    held = {r["feature"]: r for r in audit["fits"]
            if r["role"] == "held_out" and r["path"] == "inner_label_excluded"}
    cell = held["dow_month_encoded"]
    assert cell["self_rows"] == 1 and cell["comparison"] == "passed"
    assert np.isclose(cell["self_share_max"], _cell_self_weight(X, leaky.fit, leaked), rtol=0, atol=1e-15)
    assert held["economic_curvature_intensity"]["self_rows"] == 1


def test_the_findings_cover_every_family_stage_and_path(audited):
    *_, audit = audited
    summary = EA.path_summary(audit)
    by_family = EA.family_findings(summary)
    assert len(by_family) == 2 + 3 * 2
    assert by_family["c_out_of_fold"].str.contains("rows draw labels from a later month").all()
    by_path = EA.findings_table(summary)
    assert len(by_path) == summary.groupby(["feature", "stage", "path"]).ngroups
    folds = by_path[by_path["path"] != "final"]
    assert folds["b_held_out"].str.startswith("0/").all() and (folds["c_rows"] == "held_out").all()
    assert (by_path.loc[by_path["path"] == "final", "c_rows"] == "final_mapping").all()
    compact = EA.compact_fits(audit)
    assert compact and set(compact[0]) == set(EA.COMPACT_COLUMNS)
    assert {r["role"] for r in compact} == {"held_out", "final_mapping"}


# ----------------------------------------------------------------------
# fitting paths and isolation
# ----------------------------------------------------------------------
def test_the_fit_paths_partition_their_rows_as_the_production_plans_do(dated):
    _, y = dated
    paths = EA.fit_paths(y)
    assert {k: len(v) for k, v in paths.items()} == {"stage1": 111, "stage2": 36, "stage3": 41}
    checks = EA.plan_checks(y, paths)
    assert checks["passed"].all(), checks.to_string()
    table = EA.path_table(paths)
    assert len(table) == len(EA.PATH_USE) and table["fits"].tolist() == [1, 10, 100, 1, 5, 5, 25,
                                                                          1, 5, 10, 25]


@pytest.fixture(scope="module")
def small():
    frame = make_bank_frame(n=700, seed=5)
    return frame.drop(columns=[TARGET]), frame[TARGET].astype(int)


def test_the_isolation_checks_pass_on_the_production_builders(small):
    X, y = small
    checks = EA.isolation_checks(X, y, EA.fit_paths(y), self_sample=5)
    assert checks["passed"].all(), checks.to_string()


def test_the_isolation_checks_fail_when_an_encoder_sees_held_out_labels(small, monkeypatch):
    from shared.stage1 import crossfit

    X, y = small
    real = crossfit._designs

    def leaky(X_, y_, X_raw, columns, cv_folds, random_state, make_engineer):
        designs = real(X_, y_, X_raw, columns, cv_folds, random_state, make_engineer)
        if X_raw is None:
            return designs
        engineer = crossfit._quiet_engineer(make_engineer, X_raw, y_)        # every row's label
        for d in designs:
            d.X_val = crossfit._quiet_transform(engineer, X_raw.iloc[d.val], columns).to_numpy()
        return designs

    monkeypatch.setattr(crossfit, "_designs", leaky)
    checks = EA.isolation_checks(X, y, EA.fit_paths(y), self_sample=2).set_index("check")["passed"]
    held = [c for c in checks.index if "held-out labels" in c and c.startswith("stage1")]
    assert held and not checks[held].all()


def test_the_label_argument_checks_fail_on_a_fold_fit_given_every_label(tmp_path):
    assert all(r["passed"] for r in EA.label_argument_checks())
    for rel, *_ in EA.LABEL_ARGUMENTS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(BANK_ROOT / rel, tmp_path / rel)
    target = tmp_path / "shared/stage3/data.py"
    text = target.read_text(encoding="utf-8")
    assert "p.fit_transform(Xa_raw.copy(), y_tr.iloc[tr])" in text
    target.write_text(text.replace("p.fit_transform(Xa_raw.copy(), y_tr.iloc[tr])",
                                   "p.fit_transform(Xa_raw.copy(), y_tr)"), encoding="utf-8")
    failed = [r["check"] for r in EA.label_argument_checks(tmp_path) if not r["passed"]]
    assert failed == ["shared/stage3/data.py: p.fit_transform(...) is passed the fold's training labels"]


def test_the_provenance_reads_the_frozen_thresholds_from_the_pr39_record():
    from shared.stage2.binning import BINNING_STRATEGY

    table = EA.provenance_table().set_index("item")
    for fam in ("NSD", "ECI", "DOW"):
        row = table.loc[f"Stage 2 {fam} thresholds on {BINNING_STRATEGY[fam]['source']}"]
        assert "chosen with all training labels" in row["provenance"]
        assert row["record"].startswith("research_logs/pr39/c3_20261114_231604/")
    assert "without revalidation" in table.loc["Stage 2 DOW thresholds on dow_month_encoded", "provenance"]


def test_the_audit_module_never_unpickles_or_reads_a_split():
    tree = ast.parse((BANK_ROOT / "feature_research" / "encoding_audit.py").read_bytes())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"joblib", "pickle", "dill", "cloudpickle", "torch"}
    froms = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not froms & {"data.global_splits.split_manager", "shared.pinned_artifacts",
                        "shared.stage4.adapters"}
    calls = {getattr(n.func, "attr", getattr(n.func, "id", None)) for n in ast.walk(tree)
             if isinstance(n, ast.Call)}
    assert not calls & {"load", "read_pickle", "_load_bytes", "create_split", "load_split",
                        "read_csv", "read_sql_query"}


def test_the_accepted_fits_used_fold_fitted_encoders_as_recorded():
    from feature_research import research_records as RR

    commit2 = RR.read_run_record(*RR.COMMIT2_INVENTORY)
    table = EA.accepted_fit_scopes([], commit2)
    assert table["passed"].all(), table.to_string()
    roles = table[table["item"].str.startswith("accepted stage")]
    assert set(roles["commit2_record"]) == {"passed", "not read"} and len(roles) == 6
    ebm = roles[roles["item"].str.contains("stage3.ebm")]["evidence"].iloc[0]
    assert "feature_fit_scope is 'per_fold'" in ebm


def test_the_mlp_early_stopping_split_is_reproduced_and_sees_encoder_fitted_labels(dated):
    X, y = dated
    paths = EA.fit_paths(y)
    small = {"stage1": [p for p in paths["stage1"] if p.path == "final" or p.fold == 0 and p.inner in (None, 0)]}
    out = EA.mlp_early_stopping(X, y, small)
    assert out["checks"]["passed"].all() and len(out["checks"]) == 2, out["checks"].to_string()
    s = out["summary"]
    assert (s["validation_rows_with_own_label"] >= 0.99 * s["validation_rows"]).all()
    assert np.allclose(s["validation_own_share_mean"] + s["validation_from_other_validation_share_mean"],
                       s["validation_from_subset_share_mean"])
    assert np.allclose(s["validation_from_subset_share_mean"] + s["validation_from_optimization_share_mean"], 1.0)
    # the subset's labels reach the optimization rows' encodings as well as the validation rows'
    assert (s["optimization_from_subset_share_mean"] > 0.1).all()
    assert (s["optimization_rows_with_subset_labels"] >= 0.9 * s["optimization_rows"]).all()
    cells = s[s["feature"] == "subscribe_rate_by_dow_month"]          # the global rate reaches every row
    assert (cells["optimization_rows_with_subset_labels"] == cells["optimization_rows"]).all()
    assert np.allclose(s["validation_rows"] / (s["validation_rows"] + s["optimization_rows"]), 0.15, atol=0.01)
    assert "excludes the entire early-stopping subset from encoder fitting" in out["finding"]
    assert "outer held-out rows are isolated" in out["finding"]
