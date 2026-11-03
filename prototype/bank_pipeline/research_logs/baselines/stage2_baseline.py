#!/usr/bin/env python3
"""
PR 37, Commit 2, Step 2E: generate, check or verify the Pre-Correction Stage 2
Baseline (``shared.stage2.diagnostics.baseline``).

    --verify     integrity of the frozen package only: MANIFEST.json, its SHA-256
                 sidecar, every listed file's SHA-256 / size / rows / header, no
                 extra files. Opens no artifact. Writes nothing.
    --check      rebuild the package in memory from the historical artifacts the
                 frozen MANIFEST.json names (pinned arbiter roots, snapshot-attested
                 hashes) and compare it with the frozen files. Writes nothing.
    --generate   build from the local historical artifacts and write the package
                 once, atomically. An identical existing package: nothing is
                 written. A differing (or partial) one: refused. Never overwrites.

--check and --generate need the preserved snapshot's SHA256SUMS
(``PR37_BASELINE_SHA256SUMS`` or ``--sums``): every source file must equal it.
Artifacts are loaded only through tests.support.router_artifacts (the validated
Step 2A-2D loader; no test label is read), hashed before their first load and
again at the end. Nothing is fitted, tuned, regenerated or committed.

Run from prototype/bank_pipeline/ (any directory works: bank_pipeline/ is found
from this file's location).

Exit codes: 0 verified / identical / reproduced / written; 1 integrity failure,
drift, a missing package or a refused overwrite; 2 sources missing, unattested,
invalid or changed during the run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BANK_ROOT = Path(__file__).resolve().parents[2]
SUMS_ENV = "PR37_BASELINE_SHA256SUMS"


def _ensure_path() -> None:
    if str(BANK_ROOT) not in sys.path:
        sys.path.insert(0, str(BANK_ROOT))


def default_collector(mode: str, frozen, sums_path):
    """The real sources: ``Real`` for --generate, ``PinnedReal`` (frozen roots) for --check."""
    from shared.stage2.diagnostics import baseline as b
    from tests.support import router_artifacts as ra
    if frozen is None:
        real = ra.Real()
    else:
        paths = {s["role"]: s["path"] for s in frozen.get("sources") or ()}
        missing = [r for r in (b.ROLE_ARBITER_GLASS, b.ROLE_ARBITER_RF) if r not in paths]
        if missing:
            raise ValueError(f"the frozen MANIFEST.json records no source for {missing}")
        real = ra.PinnedReal({"meta_ebm": BANK_ROOT / paths[b.ROLE_ARBITER_GLASS],
                              "meta_ebm_on_blackbox": BANK_ROOT / paths[b.ROLE_ARBITER_RF]})
    real.sums = ra.read_sums(sums_path)
    return ra.collect_baseline_sources(real, bank_root=BANK_ROOT, sums_path=sums_path)


def default_rehash(sources):
    """Problems if any source file changed or vanished since it was hashed."""
    from shared.stage2.diagnostics import file_sha256
    out = []
    for a in sources.artifacts:
        p = BANK_ROOT / a.path
        if not p.is_file():
            out.append(f"{a.role}: {a.path} is gone")
        elif file_sha256(p) != a.sha256:
            out.append(f"{a.role}: {a.path} changed during the run")
    return out


def _say(tag: str, lines) -> None:
    for line in lines:
        print(f"[2E {tag}] {line}")


def _sources(args, mode, frozen, collector):
    sums = args.sums or os.environ.get(SUMS_ENV)
    if not sums:
        _say(mode, [f"refused: set {SUMS_ENV} (or --sums) to the preserved snapshot's "
                    "SHA256SUMS.txt; the baseline is built only from attested sources"])
        return None
    if not Path(sums).is_file():
        _say(mode, [f"refused: SHA256SUMS file not found: {sums}"])
        return None
    try:
        return collector(mode, frozen, sums)
    except Exception as exc:  # noqa: BLE001 - every failure is reported, exit 2
        _say(mode, [f"sources could not be loaded: {type(exc).__name__}: {exc}"])
        return None


def main(argv=None, *, collector=None, rehash=None) -> int:
    _ensure_path()
    from shared.stage2.diagnostics import baseline as b

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true", help="write the package once")
    mode.add_argument("--check", action="store_true", help="rebuild and compare; writes nothing")
    mode.add_argument("--verify", action="store_true", help="integrity only; writes nothing")
    ap.add_argument("--baseline-dir", help="frozen package directory (default: "
                    f"{b.BASELINES_DIR}/{b.BASELINE_ID})")
    ap.add_argument("--out-parent", help=f"--generate: parent directory (default: {b.BASELINES_DIR})")
    ap.add_argument("--sums", help=f"snapshot SHA256SUMS.txt (default: ${SUMS_ENV})")
    args = ap.parse_args(argv)
    collector = collector or default_collector
    rehash = rehash or default_rehash
    target = (Path(args.baseline_dir) if args.baseline_dir
              else BANK_ROOT / b.BASELINES_DIR / b.BASELINE_ID)

    if args.verify:
        rep = b.verify_baseline(target)
        if rep.ok:
            _say("verify", [f"OK: {target.name}: {rep.files_checked} files match MANIFEST.json "
                            f"(manifest sha256 {rep.manifest_sha256})"])
            return 0
        _say("verify", [f"FAILED: {target}"] + list(rep.problems))
        return 1

    if args.check:
        if not (target / b.MANIFEST).is_file():
            _say("check", [f"no frozen package at {target}"])
            return 1
        try:
            frozen = b.read_manifest(target)
        except b.BaselineError as exc:
            _say("check", [f"FAILED: {exc}"])
            return 1
        sources = _sources(args, "check", frozen, collector)
        if sources is None:
            return 2
        try:
            package = b.build_baseline(sources, b.code_identity(BANK_ROOT),
                                       baseline_id=str(frozen.get("baseline_id")))
        except b.BaselineError as exc:
            _say("check", [f"FAILED: the package could not be rebuilt: {exc}"])
            return 1
        changed = rehash(sources)
        if changed:
            _say("check", ["FAILED: source artifacts changed during the run"] + changed)
            return 2
        cmp = b.compare_baseline(package, target)
        if cmp.reproduced:
            note = ([] if cmp.status == "identical" else
                    ["note: the diagnostic code differs from the code recorded at generation, "
                     "but every frozen output is reproduced exactly"] + list(cmp.code_changes))
            _say("check", [f"OK: {target.name} reproduced from the recorded sources "
                           f"({cmp.status}; {cmp.integrity.files_checked} files)"] + note)
            return 0
        _say("check", [f"FAILED: {target.name} is not reproduced"] + cmp.lines())
        return 1

    # --generate
    sources = _sources(args, "generate", None, collector)
    if sources is None:
        return 2
    try:
        package = b.build_baseline(sources, b.code_identity(BANK_ROOT))
    except b.BaselineError as exc:
        _say("generate", [f"FAILED: nothing written: {exc}"])
        return 1
    changed = rehash(sources)
    if changed:
        _say("generate", ["FAILED: source artifacts changed during the run; nothing written"]
             + changed)
        return 2
    parent = Path(args.out_parent) if args.out_parent else BANK_ROOT / b.BASELINES_DIR
    try:
        result = b.write_baseline(package, parent)
    except b.BaselineConflict as exc:
        _say("generate", [f"REFUSED: {exc}"])
        return 1
    except b.BaselineError as exc:
        _say("generate", [f"FAILED: {exc}"])
        return 1
    if result.status == "identical":
        _say("generate", [f"identical: {result.directory} already holds this package; nothing "
                          "was written"])
    else:
        _say("generate", [f"written: {result.directory}",
                          f"manifest sha256 {result.manifest_sha256}"]
             + [f"  {p}" for p in package.files])
    return 0


if __name__ == "__main__":
    _ensure_path()
    os.chdir(BANK_ROOT)
    sys.exit(main())
