"""
PR 37 (Commit 5 follow-up): tests.support.archived_artifacts and the archive
baseline CLI, on a synthetic archive.

The October 23 regression tests read only the archived snapshot; the current-
artifact tests read only the live folders and never the October hashes. Every
archived file is checked against SHA256SUMS before use; a path that cannot be
mapped into the archive, an unlisted, missing or changed file is refused. The
pinned loader, audit and baseline CLI are not modified (their identities are
checked by test_the_baseline_pinned_code_is_unchanged).

Synthetic files only; nothing is fitted and no real artifact is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from shared.stage2.diagnostics import baseline as b
from tests.support import archived_artifacts as aa
from tests.support import router_artifacts as ra
from tests.support.notebooks import BANK_ROOT

ARCHIVE_CLI = Path(BANK_ROOT) / "research_logs/baselines/stage2_baseline_archive.py"
REALDATA = Path(BANK_ROOT) / "tests/realdata"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arbiter(stage1, stage2, stage3) -> dict:
    return {"stage4_inputs": {"upstream": {"stages": {
        "stage1": {"artifact_path": stage1}, "stage2": {"artifact_path": stage2},
        "stage3": {"artifact_path": stage3}}}}}


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """``<tmp>/snap/SHA256SUMS.txt`` + ``<tmp>/snap/bank_pipeline/...``, as the real snapshot."""
    monkeypatch.delenv(ra.SUMS_ENV, raising=False)
    snap = tmp_path / "snap"
    root = snap / "bank_pipeline"
    files = {
        "models/lr/lr_A.joblib": b"lr",
        "models/glass_router/glass_router_A.joblib": b"glass",
        "models/ebm/ebm_A.joblib": b"ebm",
        "models/rf_router/rf_router_two_pass.joblib": b"rf router (October)",
        "models/rf/rf_result_A.joblib": b"rf result",
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    arbiters = {
        # Windows separators and "./", as the notebooks recorded them
        "models/meta_ebm/meta_ebm_A.joblib": _arbiter(
            "./models/lr\\lr_A.joblib", "./models/glass_router/glass_router_A.joblib",
            str(Path(BANK_ROOT) / "models" / "ebm" / "ebm_A.joblib")),
        "blackbox_pipeline/models/stage4_crossover/arbiter_on_blackbox/meta_ebm_B.joblib":
            _arbiter("models/lr/lr_A.joblib", "models/rf_router/rf_router_two_pass.joblib",
                     "models/ebm/ebm_A.joblib"),
    }
    for rel, art in arbiters.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(art, root / rel)
    lines = [f"{_sha((root / rel).read_bytes())}  bank_pipeline/{rel}"
             for rel in sorted([*files, *arbiters])]
    (snap / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    keys = {"meta_ebm": "models/meta_ebm/meta_ebm_A.joblib",
            "meta_ebm_on_blackbox": "blackbox_pipeline/models/stage4_crossover/"
                                    "arbiter_on_blackbox/meta_ebm_B.joblib"}
    return SimpleNamespace(snap=snap, root=root, sums=snap / "SHA256SUMS.txt", arbiters=keys)


def _historical(archive):
    return aa.HistoricalReal(sums_path=archive.sums, arbiters=archive.arbiters)


# ======================================================================
# historical: the archive only
# ======================================================================
def test_roots_and_recorded_upstream_files_resolve_into_the_archive_only(archive):
    real = _historical(archive)
    roots = real.roots()
    native, blackbox = real.paths("meta_ebm"), real.paths("meta_ebm_on_blackbox")
    expected = {"stage1": "models/lr/lr_A.joblib",
                "stage2": "models/glass_router/glass_router_A.joblib",
                "stage3": "models/ebm/ebm_A.joblib"}
    assert {k: p.relative_to(archive.root).as_posix() for k, p in native.items()} == expected
    assert blackbox["stage2"] == archive.root / "models/rf_router/rf_router_two_pass.joblib"
    for p in [*roots.values(), *native.values(), *blackbox.values()]:
        assert p.resolve().is_relative_to(archive.root.resolve())
        assert not p.resolve().is_relative_to(Path(BANK_ROOT).resolve())
    sums = ra.read_sums(archive.sums)
    key = "bank_pipeline/models/rf_router/rf_router_two_pass.joblib"
    assert real.sha(blackbox["stage2"]) == sums[key]


def test_the_rf_result_is_the_archived_copy_of_the_recorded_origin(archive):
    class Fake(aa.HistoricalReal):
        def view(self, which):
            return SimpleNamespace(metadata={"rf_params_origin": self.origin})

    real = Fake(sums_path=archive.sums, arbiters=archive.arbiters)
    real.origin = "./models/rf\\rf_result_A.joblib"
    assert real.rf_result_path() == archive.root / "models/rf/rf_result_A.joblib"
    real.origin = None
    with pytest.raises(AssertionError, match="rf_params_origin"):
        real.rf_result_path()


@pytest.mark.parametrize("damage, match", [
    ("changed", "corrupt or was changed"),
    ("unlisted", "is not in the archive"),
    ("missing", "missing from the archive"),
])
def test_a_changed_unlisted_or_missing_archive_file_is_refused(archive, damage, match):
    target = archive.root / "models/glass_router/glass_router_A.joblib"
    if damage == "changed":
        target.write_bytes(b"substituted")
    elif damage == "unlisted":
        text = archive.sums.read_text(encoding="ascii")
        archive.sums.write_text("\n".join(l for l in text.splitlines()
                                          if "glass_router_A" not in l) + "\n", encoding="ascii")
    else:
        target.unlink()
    real = _historical(archive)
    with pytest.raises(AssertionError, match=match):
        real.paths("meta_ebm")


@pytest.mark.parametrize("recorded", ["../outside.joblib", "models/../../x.joblib", "",
                                      "/somewhere/else/entirely.joblib",
                                      "/other/prototype/bank_pipeline",
                                      "/other/prototype/bank_pipeline/../x.joblib"])
def test_paths_that_cannot_be_mapped_into_the_archive_are_refused(archive, recorded):
    with pytest.raises(AssertionError):
        aa.Archive(archive.sums).path(recorded)


@pytest.mark.parametrize("recorded", [
    "C:\\Users\\someone\\other-checkout\\prototype\\bank_pipeline\\models\\lr\\lr_A.joblib",
    "/home/someone/other-checkout/prototype/bank_pipeline/./models/lr/lr_A.joblib",
])
def test_an_absolute_path_into_another_checkout_names_the_archived_file(archive, recorded):
    # The notebook may have run in another checkout; the archived copy is used,
    # and it must still match SHA256SUMS.
    arc = aa.Archive(archive.sums)
    assert arc.path(recorded) == archive.root / "models/lr/lr_A.joblib"
    (archive.root / "models/lr/lr_A.joblib").write_bytes(b"substituted")
    with pytest.raises(AssertionError, match="corrupt or was changed"):
        aa.Archive(archive.sums).path(recorded)


def test_sha_refuses_a_file_outside_the_archive(archive):
    real = _historical(archive)
    with pytest.raises(AssertionError, match="not an archived file"):
        real.sha(Path(BANK_ROOT) / "models/rf_router/rf_router_two_pass.joblib")


def test_the_archive_must_be_named_and_complete(archive, tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match=ra.SUMS_ENV):
        aa.HistoricalReal(arbiters=archive.arbiters)
    with pytest.raises(AssertionError, match="not found"):
        aa.Archive(tmp_path / "nowhere" / "SHA256SUMS.txt")
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "SHA256SUMS.txt").write_text("", encoding="ascii")
    with pytest.raises(AssertionError, match="no bank_pipeline"):
        aa.Archive(bare / "SHA256SUMS.txt")
    monkeypatch.setenv(ra.SUMS_ENV, str(archive.sums))
    assert aa.HistoricalReal(arbiters=archive.arbiters).archive.root == archive.root


def test_the_default_arbiters_are_the_ones_the_frozen_baseline_records():
    assert aa.baseline_arbiters() == {
        "meta_ebm": "models/meta_ebm/meta_ebm_20261023_070055_573687.joblib",
        "meta_ebm_on_blackbox": "blackbox_pipeline/models/stage4_crossover/arbiter_on_blackbox/"
                                "meta_ebm_20261023_095719_949475.joblib"}
    with pytest.raises(AssertionError, match="records no source"):
        aa.baseline_arbiters({"sources": [{"role": b.ROLE_ARBITER_GLASS, "path": "x"}]})


def test_archived_sources_are_recorded_relative_to_the_archive_with_the_repository_dataset(
        archive, monkeypatch):
    seen = {}

    def fake(real, *, bank_root, sums_path):
        seen.update(bank_root=bank_root, sums_path=sums_path)
        return b.BaselineSources(ref=None, glass=None, rf=None, lr=None, ebm=None,
                                 rf_candidate=None, artifacts=())

    monkeypatch.setattr(aa, "collect_baseline_sources", fake)
    sources = aa.collect_archived_baseline_sources(_historical(archive))
    assert seen == {"bank_root": archive.root, "sums_path": archive.sums}
    raw = Path(BANK_ROOT) / ra.RAW_DATA_FILE
    assert sources.data_files == (b.DataFileRecord(
        ra.RAW_DATA_FILE, b.lf_sha256(raw.read_bytes()), aa.RAW_DATA_DESCRIPTION),)


def test_the_dataset_record_matches_the_pinned_collector():
    assert f'"{aa.RAW_DATA_DESCRIPTION}"' in inspect.getsource(ra.collect_baseline_sources)


def test_the_archived_rehash_reports_changed_and_missing_sources(archive):
    rel = "models/lr/lr_A.joblib"
    art = b.SourceArtifact("stage1.lr", rel, _sha(b"lr"), 2, True)
    rehash = aa.archived_rehash(aa.Archive(archive.sums))
    assert rehash(SimpleNamespace(artifacts=(art,))) == []
    (archive.root / rel).write_bytes(b"other")
    assert "changed" in rehash(SimpleNamespace(artifacts=(art,)))[0]
    (archive.root / rel).unlink()
    assert "gone" in rehash(SimpleNamespace(artifacts=(art,)))[0]


# ======================================================================
# current: the live folders, never the October hashes
# ======================================================================
def test_current_artifacts_are_never_checked_against_the_october_hashes(archive, monkeypatch):
    monkeypatch.setenv(ra.SUMS_ENV, str(archive.sums))
    current = aa.CurrentReal()
    assert current.sums == {} and current.sha(Path(BANK_ROOT) / "anything.joblib") is None


def test_historical_realdata_modules_read_only_the_archive():
    for path in sorted(REALDATA.glob("test_*.py")):
        if path.name in ("test_current_artifacts.py", "test_saved_artifacts.py"):
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(?:Pinned)?Real\(", text), path.name
        assert "HistoricalReal" in text, path.name
    text = (REALDATA / "test_current_artifacts.py").read_text(encoding="utf-8")
    assert "CurrentReal()" in text and "HistoricalReal" not in text


# ======================================================================
# the archive baseline CLI
# ======================================================================
def _archive_cli():
    spec = importlib.util.spec_from_file_location("pr37_stage2_baseline_archive", ARCHIVE_CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_archive_cli_refuses_generate_and_runs_the_pinned_cli_otherwise(archive, capsys):
    cli = _archive_cli()
    calls = []
    fake = SimpleNamespace(main=lambda argv, **kw: calls.append((argv, kw)) or 0)
    assert cli.main(["--generate", "--sums", str(archive.sums)], cli=fake) == 2
    assert calls == [] and "REFUSED" in capsys.readouterr().out
    assert cli.main(["--check", "--sums", str(archive.sums)], cli=fake) == 0
    argv, kw = calls[-1]
    assert argv == ["--check", "--sums", str(archive.sums)]
    assert kw["collector"] is cli.archive_collector
    art = b.SourceArtifact("stage1.lr", "models/lr/lr_A.joblib", _sha(b"lr"), 2, True)
    assert kw["rehash"](SimpleNamespace(artifacts=(art,))) == []
    assert cli.main(["--verify"], cli=fake) == 0 and calls[-1][0] == ["--verify"]


def test_the_archive_collector_uses_the_frozen_packages_arbiters(archive, monkeypatch):
    cli = _archive_cli()
    seen = {}
    monkeypatch.setattr(aa, "collect_archived_baseline_sources",
                        lambda real: seen.setdefault("real", real))
    frozen = {"sources": [{"role": b.ROLE_ARBITER_GLASS, "path": archive.arbiters["meta_ebm"]},
                          {"role": b.ROLE_ARBITER_RF,
                           "path": archive.arbiters["meta_ebm_on_blackbox"]}]}
    real = cli.archive_collector("check", frozen, str(archive.sums))
    assert isinstance(real, aa.HistoricalReal) and real is seen["real"]
    assert real.roots()["meta_ebm"] == archive.root / archive.arbiters["meta_ebm"]
    with pytest.raises(ValueError, match="generate"):
        cli.archive_collector("generate", None, str(archive.sums))
