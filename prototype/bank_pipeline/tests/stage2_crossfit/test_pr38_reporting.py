"""
PR 38 Commit 4: the consolidated research report (``stage2_crossfit.reporting``).

On real v1 packages written by the Step 3B and 3C generators (the synthetic
Step 2A-2D world), with the run's records and the frozen baseline:

1. Migration: every package table has one disposition and is reproduced byte
   for byte from the report's rows or by its rebuild; each row keeps its
   source; tables are rows, not cells; the summaries carry the run's
   identities and the packages' own values; historical and report checksums
   are kept apart. Packages may be read from a temporary folder.
2. Written once and deterministic; the packages, records and baseline are
   untouched; nothing is deleted.
3. Verification without any package: identical after the packages are
   removed; an edited report, record or manifest copy is refused.
4. Migration refusals before anything is written.
5. The command line, and Git keeping research-log records byte-exact.

Every input is synthetic: no fit, no model artifact, no real record.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import shared.stage2.evaluation_protocol as ep
import stage2_crossfit.acceptance as A
import stage2_crossfit.complementarity as C
import stage2_crossfit.reporting as RP
import stage2_crossfit.runner as R
import stage2_crossfit.sensitivity as S
from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import baseline as b
from tests.stage2_crossfit.test_pr38_complementarity import _evidence, _historical
from tests.stage2_crossfit.test_pr38_sensitivity import (
    _full_evidence, _s1_view, _with_pass1_rules,
)
from tests.support import diagnostics_world as dw
from tests.support import stage2_crossfit_world as W

RUN = "c3_synthetic"
BANK = Path(__file__).resolve().parents[2]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot(directory: Path):
    return {p.relative_to(directory).as_posix(): _sha(p.read_bytes())
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _resign(pkg: Path) -> None:
    """Re-pin a package's manifest to the tables now on disk (a consistent, edited package)."""
    man = json.loads((pkg / C.MANIFEST).read_bytes())
    for p in list(pkg.rglob("*.csv")):
        rel = p.relative_to(pkg).as_posix()
        data = p.read_bytes()
        frame = C._read_text_csv(data)
        man["outputs"][rel] = {"sha256": _sha(data), "bytes": len(data), "rows": len(frame),
                               "columns": list(frame.columns)}
    man_b = b._json_bytes(man)
    (pkg / C.MANIFEST).write_bytes(man_b)
    (pkg / C.SIDECAR).write_bytes((_sha(man_b) + "\n").encode())


# ======================================================================
# a run with both v1 packages, its records and the frozen baseline
# ======================================================================
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("report")
    w = dw.world()
    ref, glass, rf, lr, ebm = dw.views(w)
    trace = d.build_router_trace(glass, rf, ref)
    settings = W.make_settings()
    hist = {"glass": _historical(glass, 1), "blackbox": _historical(rf, 2)}
    ht = C.diagnostics_tables(hist["glass"], hist["blackbox"], ref, lr, ebm)
    frozen = {k: C._read_text_csv(b._csv_bytes(ht[k])) for k in ("states", "ratios")}

    base = root / "baseline"
    tables = {f"tables/{k}.csv": b._csv_bytes(f) for k, f in frozen.items()}
    man = b._json_bytes({"format": "synthetic baseline",
                         "outputs": {p: {"sha256": _sha(x)} for p, x in tables.items()}})
    C.write_package(base, {**tables, C.MANIFEST: man,
                           C.SIDECAR: f"{_sha(man)}  MANIFEST.json\n".encode()})

    record = root / "logs" / RUN
    record.mkdir(parents=True)
    R.write_record(record / A.ACCEPTANCE_FILE, {
        "format": A.ACCEPTANCE_FORMAT, "run_id": RUN, "accepted": True,
        "structural_acceptance": {"accepted": True, "official": True}})
    R.write_record(record / S.S1_ACCEPTANCE_FILE, {
        "format": "pr38.stage2.s1_acceptance/1", "run_id": RUN, "accepted": True,
        "g6_passed": True, "statement": "synthetic S1",
        "gates": {"G6": {"checks": [{"check": "pass1 identical", "passed": True,
                                     "detail": {"folds": [0, 1]}}]},
                  "S1": {"checks": [{"check": "single factor", "passed": True, "detail": ""}]}}})
    R.write_record(record / "authorization.json", {"run_id": RUN, "gates": {"G3": "pass"},
                                                   "approved_on": "2026-11-12"})
    R.write_record(record / "structural_acceptance_failed_20261112T000000Z.json",
                   {"accepted": False})
    acc_sha = R.sha256_file(record / A.ACCEPTANCE_FILE)
    s1_sha = R.sha256_file(record / S.S1_ACCEPTANCE_FILE)

    evidence = _evidence(trace)
    ledger = C.fold_ledger(evidence, settings)
    c_inp = C.Inputs(
        RUN, "primary", settings.protocol, settings.contract, ref, trace, lr, ebm,
        ep.ReferenceFlags.from_upstream(d.primary_reference_flags(lr, ebm)["union"]), hist,
        frozen, ledger, evidence, C.check_alignment(trace, evidence, ledger.attempted),
        {"run": {"run_id": RUN}, "structural_acceptance": {"sha256": acc_sha},
         "baseline": {"manifest_sha256": _sha(man)}})
    C.write_package(record / C.PACKAGE_DIR, C.build_package(c_inp, C.analyse(c_inp)))

    s1_trace = d.build_router_trace(_s1_view(glass, 3), rf, ref)
    rec = R.settings_record(settings)
    rec_p = A._canon(dict(rec, role=R.ROLE_PRIMARY))
    rec_s = A._canon(dict(rec, role=R.ROLE_S1,
                          glass_config=dict(rec_p["glass_config"], pass2_population="full")))
    rec_s.pop("rf_config", None)
    ev_p = _with_pass1_rules(_evidence(trace))
    ev_s = _with_pass1_rules(_full_evidence(ev_p, trace))
    s_inp = S.S1Inputs(
        RUN, settings.protocol, ref, trace, s1_trace, lr, ebm,
        ep.ReferenceFlags.from_upstream(d.primary_reference_flags(lr, ebm)["union"]),
        {"oof_remainder": ev_p, "full": ev_s}, {"oof_remainder": rec_p, "full": rec_s},
        json.loads((record / S.S1_ACCEPTANCE_FILE).read_text(encoding="utf-8")),
        {"ratios": frozen["ratios"]},
        {"run": {"run_id": RUN}, "primary": {"structural_acceptance_sha256": acc_sha},
         "s1": {"acceptance_sha256": s1_sha}})
    C.write_package(record / S.PACKAGE_DIR, S.build_package(s_inp, S.analyse_s1(s_inp)))
    return SimpleNamespace(root=root, record=record, base=base)


def _copy(world, tmp_path) -> Path:
    record = tmp_path / "logs" / RUN
    shutil.copytree(world.record, record)
    shutil.rmtree(record / RP.REPORT_DIR, ignore_errors=True)
    return record


def _drop_packages(record: Path) -> None:
    for n in (RP.CP, RP.SP):
        shutil.rmtree(record / n)


@pytest.fixture(scope="module")
def built(world):
    files, stats = RP.build(RUN, record_dir=world.record, baseline_dir=world.base)
    return SimpleNamespace(files=files, stats=stats)


def _v1(world):
    return {n: json.loads((world.record / n / C.MANIFEST).read_bytes())["outputs"]
            for n in (RP.CP, RP.SP)}


# ======================================================================
# 1. migration
# ======================================================================
def test_every_package_table_has_one_disposition_and_is_reproduced(world, built):
    v1 = _v1(world)
    total = sum(len(o) for o in v1.values())
    s = built.stats
    assert s["package_tables"] == total
    assert s["consolidated"] + s["derived"] + s["duplicate"] + s["restated"] == total
    assert (s["derived"], s["duplicate"], s["restated"]) == (5, 1, 1)
    assert s["report_tables"] == len(RP.GROUPS) == 11

    columns = {(n, rel[7:-4]): o[rel]["columns"] for n, o in v1.items() for rel in o}
    kept = RP.reconstruct(built.files, columns)
    assert len(kept) == s["consolidated"]
    for (n, t), data in kept.items():
        assert data == (world.record / n / "tables" / f"{t}.csv").read_bytes(), (n, t)

    man = json.loads(built.files[RP.MANIFEST])
    hist = man["historical_v1_sources"]
    assert "--check" in hist["note"] and "only while they exist" in hist["note"]
    for n, outputs in v1.items():
        copy = built.files[RP.source_manifest(n)]
        assert copy == (world.record / n / C.MANIFEST).read_bytes()
        assert hist[n]["manifest_sha256"] == _sha(copy)
        assert set(hist[n]["tables"]) == {rel[7:-4] for rel in outputs}
        for t, entry in hist[n]["tables"].items():
            assert entry["sha256"] == outputs[f"tables/{t}.csv"]["sha256"]
            assert entry["disposition"] in {g.file for g in RP.GROUPS} | {
                "derived", "duplicate", "restated"}
    assert set(man["outputs"]) == set(built.files) - {RP.MANIFEST, RP.SIDECAR}
    assert all(man["outputs"][f]["sha256"] == _sha(built.files[f]) for f in man["outputs"])


def test_tables_are_rows_that_name_their_source(world, built):
    csvs = sorted(n for n in built.files if n.endswith(".csv"))
    assert csvs == sorted(g.file for g in RP.GROUPS)
    assert set(built.files) == set(csvs) | set(RP.SUMMARIES) | {
        RP.README, RP.MANIFEST, RP.SIDECAR, RP.source_manifest(RP.CP),
        RP.source_manifest(RP.SP)}
    v1 = _v1(world)
    man = json.loads(built.files[RP.MANIFEST])
    for g in RP.GROUPS:
        f = C._read_text_csv(built.files[g.file])
        assert list(f.columns[:2]) == ["package", "table"]
        assert not {"measure", "value", "source_row"} <= set(f.columns)
        for p, t in g.sources:
            rows = f[(f["package"] == p) & (f["table"] == t)]
            assert len(rows) == v1[p][f"tables/{t}.csv"]["rows"], (p, t)   # one row per row
            used = man["outputs"][g.file]["sources"][f"{p}/{t}"]
            assert len(used) == len(v1[p][f"tables/{t}.csv"]["columns"])
            assert set(used) <= set(f.columns)
    metrics = C._read_text_csv(built.files["02_router_metrics.csv"])
    assert "routing" not in metrics.columns
    pp = metrics[metrics["table"] == "pre_post_metrics"]
    assert set(pp["correction"]) == {"pre", "post"} and set(pp["arm"]) == {"glass", "blackbox"}
    assert set(metrics[metrics["table"] == "metric_table"]["arm"]) == {"glass", "blackbox"}
    states = C._read_text_csv(built.files["03_routing_states.csv"])
    assert set(states[states["table"] == "class_conditional"]["arm"]) == {"glass", "blackbox"}
    text = b"".join(built.files.values()).decode("utf-8")
    assert "/tmp/" not in text and "/home/" not in text


def test_summaries_carry_identities_and_the_packages_own_values(world, built):
    run = json.loads(built.files["run_summary.json"])
    assert run["run_id"] == RUN
    acc = run["acceptance"]["structural_acceptance"]
    assert acc["sha256"] == R.sha256_file(world.record / A.ACCEPTANCE_FILE)
    assert acc["accepted"] is True and acc["verdict"]["official"] is True
    assert run["acceptance"]["s1"]["g6_passed"] is True
    assert set(run["records"]) == {"authorization.json", A.ACCEPTANCE_FILE,
                                   S.S1_ACCEPTANCE_FILE}
    assert run["historical_v1_packages"][RP.CP]["manifest_copy"] == RP.source_manifest(RP.CP)

    comp = json.loads(built.files["complementarity_summary.json"])
    po = C._read_text_csv((world.record / RP.CP / "tables" / "primary_outcomes.csv")
                          .read_bytes())
    assert [r["metric"] for r in comp["primary_outcomes"]] == list(po["metric"])
    assert [r["statement"] for r in comp["primary_outcomes"]] == list(po["statement"])
    assert [r["difference"] for r in comp["primary_outcomes"]] == [
        None if v == "" else (int(v) if v.lstrip("-").isdigit() else float(v))
        for v in po["difference"]]
    assert comp["layers"]["1_implementation_correctness"]["tables"]["trace_checks"] == \
        "10_solver_and_validation.csv"

    s1 = json.loads(built.files["s1_summary.json"])
    assert [r["setting"] for r in s1["configuration_differences"]] == [
        "glass_config.pass2_population"]
    assert {r["fold"] for r in s1["glass_states_pooled"]} == {"all"}
    assert s1["glass_routing_rates_pooled"] and s1["limitations"]

    readme = built.files[RP.README].decode("utf-8")
    assert RUN in readme and "| outcome | metric |" in readme
    assert all(g.file in readme for g in RP.GROUPS)
    assert "pre_post_ratios | derived" in readme and "--check" in readme


def test_packages_can_be_read_from_a_temporary_folder(world, tmp_path):
    record = _copy(world, tmp_path)
    staging = tmp_path / "staging"
    for n in (RP.CP, RP.SP):
        shutil.move(str(record / n), str(staging / n))
    out = RP.run(RUN, record_dir=record, packages_dir=staging, baseline_dir=world.base)
    assert out["status"] == "written" and out["mode"] == "migration"
    built = RP.build(RUN, record_dir=world.record, baseline_dir=world.base)[0]
    assert {p: (record / RP.REPORT_DIR / p).read_bytes() for p in built} == built
    assert (staging / RP.CP / C.MANIFEST).is_file()             # nothing is deleted


# ======================================================================
# 2. written once; sources untouched
# ======================================================================
def test_written_once_deterministic_and_sources_untouched(world, tmp_path):
    record = _copy(world, tmp_path)
    before, base_before = _snapshot(record), _snapshot(world.base)
    assert RP.run(RUN, record_dir=record, baseline_dir=world.base)["status"] == "written"
    assert RP.run(RUN, record_dir=record, baseline_dir=world.base)["status"] == "identical"
    again, _ = RP.build(RUN, record_dir=record, baseline_dir=world.base)
    assert again == {p: (record / RP.REPORT_DIR / p).read_bytes() for p in again}
    after = {k: v for k, v in _snapshot(record).items() if not k.startswith(RP.REPORT_DIR)}
    assert after == before and _snapshot(world.base) == base_before
    out = record / RP.REPORT_DIR / "08_diagnostic_ratios.csv"
    out.write_bytes(out.read_bytes() + b"x\n")
    with pytest.raises(C.ComplementarityConflict):
        RP.run(RUN, record_dir=record, baseline_dir=world.base)


# ======================================================================
# 3. verification without any package
# ======================================================================
@pytest.fixture
def migrated(world, tmp_path):
    record = _copy(world, tmp_path)
    RP.run(RUN, record_dir=record, baseline_dir=world.base)
    _drop_packages(record)
    return record


def test_the_report_verifies_without_its_packages(world, migrated):
    out = RP.run(RUN, record_dir=migrated, baseline_dir=world.base, check=True)
    assert (out["status"], out["mode"], out["problems"]) == ("identical", "verification", [])
    assert out["stats"]["package_tables"] == 64
    assert not (migrated / RP.CP).exists() and not (migrated / RP.SP).exists()
    with pytest.raises(RP.ReportError, match="--check"):                 # migration needs them
        RP.run(RUN, record_dir=migrated, baseline_dir=world.base)


def _resign_report(report: Path) -> None:
    """Re-pin the report's manifest to its files (an edited, self-consistent report)."""
    man = json.loads((report / RP.MANIFEST).read_bytes())
    for name, entry in man["outputs"].items():
        data = (report / name).read_bytes()
        entry.update(sha256=_sha(data), bytes=len(data))
        if name.endswith(".csv"):
            entry["rows"] = len(C._read_text_csv(data))
    man_b = b._json_bytes(man)
    (report / RP.MANIFEST).write_bytes(man_b)
    (report / RP.SIDECAR).write_bytes((_sha(man_b) + "\n").encode())


def _edit_value(record):
    p = record / RP.REPORT_DIR / "02_router_metrics.csv"
    f = C._read_text_csv(p.read_bytes())
    i = f.index[f["table"] == "metric_table"][0]
    f.loc[i, "numerator"] = "999999"
    p.write_bytes(b._csv_bytes(f))


def _edited_and_resigned(record):
    _edit_value(record)
    _resign_report(record / RP.REPORT_DIR)


def _dropped_row(record):
    p = record / RP.REPORT_DIR / "07_score_strata.csv"
    p.write_bytes(b._csv_bytes(C._read_text_csv(p.read_bytes()).iloc[1:]))
    _resign_report(record / RP.REPORT_DIR)


def _edited_manifest_copy(record):
    p = record / RP.REPORT_DIR / RP.source_manifest(RP.SP)
    p.write_bytes(p.read_bytes().replace(b"S1", b"s1", 1))
    _resign_report(record / RP.REPORT_DIR)


def _changed_record(record):
    p = record / "authorization.json"
    p.write_bytes(p.read_bytes() + b" ")


def _removed_record(record):
    (record / "authorization.json").unlink()


def _extra_file(record):
    (record / RP.REPORT_DIR / "notes.txt").write_text("x", encoding="utf-8")


@pytest.mark.parametrize("edit, match", [
    (_edit_value, "02_router_metrics.csv: SHA-256 differs"),
    (_edited_and_resigned, "complementarity_v1/metric_table: reproduced from the report, differs"),
    (_dropped_row, "score_conditional: reproduced from the report, differs"),
    (_edited_manifest_copy, "not the historical s1_sensitivity_v1/MANIFEST.json"),
    (_changed_record, "record authorization.json: changed"),
    (_removed_record, "record authorization.json: missing"),
    (_extra_file, "notes.txt: not in the report's MANIFEST"),
])
def test_verification_refuses_an_edited_report_or_record(world, migrated, edit, match):
    edit(migrated)
    with pytest.raises(RP.ReportError, match=match):
        RP.run(RUN, record_dir=migrated, baseline_dir=world.base, check=True)


def test_verification_refuses_a_changed_acceptance(world, migrated):
    (migrated / A.ACCEPTANCE_FILE).write_text('{"accepted": true}\n', encoding="utf-8")
    with pytest.raises(RP.ReportError, match="structural_acceptance.json: changed"):
        RP.run(RUN, record_dir=migrated, baseline_dir=world.base, check=True)


# ======================================================================
# 4. migration refusals
# ======================================================================
def _extra_table(record):
    (record / RP.SP / "tables" / "new_table.csv").write_bytes(b"a,b\n1,2\n")
    _resign(record / RP.SP)


def _edited_table(record):
    p = record / RP.CP / "tables" / "metric_table.csv"
    p.write_bytes(p.read_bytes().replace(b"glass", b"GLASS", 1))


def _other_acceptance(record):
    (record / A.ACCEPTANCE_FILE).write_text('{"accepted": true}\n', encoding="utf-8")


def _stale_derived(record):
    p = record / RP.CP / "tables" / "pre_post_states.csv"
    f = C._read_text_csv(p.read_bytes())
    f.loc[0, "n_pre"] = "999999"
    p.write_bytes(b._csv_bytes(f))
    _resign(record / RP.CP)


@pytest.mark.parametrize("edit, match", [
    (_extra_table, "new_table: no disposition"),
    (_edited_table, "metric_table.csv: SHA-256 differs"),
    (_other_acceptance, "not this run's accepted ones"),
    (_stale_derived, "pre_post_states .derived. does not rebuild"),
])
def test_migration_refused_before_anything_is_written(world, tmp_path, edit, match):
    record = _copy(world, tmp_path)
    edit(record)
    with pytest.raises(RP.ReportError, match=match):
        RP.run(RUN, record_dir=record, baseline_dir=world.base)
    assert not (record / RP.REPORT_DIR).exists()


def test_a_baseline_that_is_not_the_pinned_one_is_refused(world, tmp_path):
    record = _copy(world, tmp_path)
    base = tmp_path / "baseline"
    shutil.copytree(world.base, base)
    (base / C.MANIFEST).write_bytes((base / C.MANIFEST).read_bytes() + b" ")
    with pytest.raises(RP.ReportError, match="not the package"):
        RP.run(RUN, record_dir=record, baseline_dir=base)


def test_a_run_without_s1_reports_the_primary_alone(world, tmp_path):
    record = _copy(world, tmp_path)
    shutil.rmtree(record / RP.SP)
    out = RP.run(RUN, record_dir=record, baseline_dir=world.base)
    assert "11_s1_sensitivity.csv" not in out["files"] and "s1_summary.json" not in out["files"]
    assert (out["stats"]["derived"], out["stats"]["duplicate"], out["stats"]["restated"]) == \
        (2, 0, 0)
    shutil.rmtree(record / RP.CP)
    assert RP.run(RUN, record_dir=record, baseline_dir=world.base,
                  check=True)["status"] == "identical"


# ======================================================================
# 5. the command line and Git
# ======================================================================
def test_the_command_migrates_then_verifies_without_packages(world, tmp_path, monkeypatch):
    import stage2_crossfit.__main__ as cli
    import stage2_crossfit.phase0 as P
    record = _copy(world, tmp_path)
    monkeypatch.setattr(P, "RECORD_ROOT", tmp_path / "logs")
    monkeypatch.setattr(P, "BASELINE_DIR", world.base)
    monkeypatch.setattr(R, "load_train_side", lambda **k: pytest.fail("data loaded"))

    def main(*argv):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main(["report", "--run-id", RUN, *argv])
        return code, out.getvalue()

    code, out = main()
    assert code == 0 and "research report WRITTEN" in out and "64 v1 package tables" in out
    _drop_packages(record)
    code, out = main("--check")
    assert code == 0 and "research report IDENTICAL" in out and "without v1 packages" in out
    code, out = main()
    assert code == 1 and "REFUSED" in out and "--check" in out


def test_the_generators_can_write_their_packages_elsewhere(monkeypatch, tmp_path):
    import stage2_crossfit.__main__ as cli
    seen = {}

    def capture(name):
        def fake(run_id, sums, **kw):
            seen[name] = kw
            raise _Stop
        return fake
    monkeypatch.setattr(C, "run", capture("c"))
    monkeypatch.setattr(S, "report_s1", capture("s"))
    for cmd in ("complementarity", "s1-report"):
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_Stop):
            cli.main([cmd, "--run-id", RUN, "--sums", "s", "--packages-dir", str(tmp_path)])
    assert seen["c"]["out_dir"] == tmp_path / C.PACKAGE_DIR
    assert seen["s"]["out_dir"] == tmp_path / S.PACKAGE_DIR


class _Stop(Exception):
    pass


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=BANK, capture_output=True, text=True)
    except OSError:
        return None


@pytest.mark.skipif(_git("rev-parse", "--is-inside-work-tree") is None
                    or _git("rev-parse", "--is-inside-work-tree").returncode != 0,
                    reason="not a Git work tree")
def test_git_keeps_records_byte_exact_and_v1_packages_local():
    run = "research_logs/pr38/c3_00000000_000000"
    for path in (f"{run}/structural_acceptance.json", f"{run}/authorization.json",
                 f"{run}/{RP.REPORT_DIR}/run_summary.json",
                 f"{run}/{RP.REPORT_DIR}/01_registered_outcomes.csv",
                 f"{run}/{RP.REPORT_DIR}/{RP.source_manifest(RP.CP)}"):
        assert _git("check-attr", "text", "--", path).stdout.strip().endswith("text: unset"), path
    assert not _git("check-attr", "text", "--", "research_logs/pr38/README.md").stdout \
        .strip().endswith("unset")
    for path, ignored in ((f"{run}/{RP.CP}/tables/metric_table.csv", True),
                          (f"{run}/{RP.CP}/{C.MANIFEST}", True),
                          (f"{run}/{RP.SP}/tables/diagnostics/ratios.csv", True),
                          (f"{run}/structural_acceptance_failed_20261113T000000Z.json", True),
                          (f"{run}/{RP.REPORT_DIR}/08_diagnostic_ratios.csv", False),
                          (f"{run}/{RP.REPORT_DIR}/{RP.source_manifest(RP.SP)}", False),
                          (f"{run}/structural_acceptance.json", False)):
        assert (_git("check-ignore", "-q", "--no-index", path).returncode == 0) is ignored, path
