#!/usr/bin/env python3
"""
PR 37 (Commit 5 follow-up): ``stage2_baseline.py --verify`` / ``--check``
against the preserved archive instead of the live ``models/`` folders.

    --verify     unchanged: integrity of the frozen package only (no artifact).
    --check      rebuild the package from the archived October 23 artifacts
                 (tests.support.archived_artifacts) and compare it with the
                 frozen files. Writes nothing.
    --generate   refused: the archive is read-only evidence; a new baseline is
                 a new, versioned package built by its own step.

This runs ``stage2_baseline.main`` unchanged, through the ``collector`` and
``rehash`` hooks it already provides. That CLI, the loader and every diagnostic
module are byte-identical to the code the frozen baseline pins, so a reproduced
package still reports ``identical``. Only where the source files are read from
differs: ``<snapshot>/bank_pipeline/<path>`` for the snapshot whose
SHA256SUMS.txt ``$PR37_BASELINE_SHA256SUMS`` (or ``--sums``) names, every file
checked against it. Regenerating models in the working checkout therefore
cannot make this check fail; a changed archive, package or diagnostic code
still does.

Exit codes as ``stage2_baseline.py``; ``--generate``: 2.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

BANK_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = Path(__file__).resolve().with_name("stage2_baseline.py")
SUMS_ENV = "PR37_BASELINE_SHA256SUMS"


def _ensure_path() -> None:
    if str(BANK_ROOT) not in sys.path:
        sys.path.insert(0, str(BANK_ROOT))


def _pinned_cli():
    spec = importlib.util.spec_from_file_location("pr37_stage2_baseline_cli", CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def archive_collector(mode, frozen, sums_path):
    """Sources from the archive; the arbiters are the ones the frozen package records."""
    from tests.support import archived_artifacts as aa
    if frozen is None:
        raise ValueError("--generate is not supported from the archive")
    real = aa.HistoricalReal(sums_path=sums_path, arbiters=aa.baseline_arbiters(frozen))
    return aa.collect_archived_baseline_sources(real)


def main(argv=None, *, cli=None) -> int:
    _ensure_path()
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--generate" in argv:
        print("[archive] REFUSED: --generate is not supported; the archive is read-only "
              "evidence (use stage2_baseline.py for a new, versioned package)")
        return 2
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sums")
    known, _ = pre.parse_known_args(argv)
    sums = known.sums or os.environ.get(SUMS_ENV)

    def rehash(sources):
        from tests.support import archived_artifacts as aa
        return aa.archived_rehash(aa.Archive(sums))(sources)

    if "--check" in argv and sums:
        print(f"[archive] sources: {Path(sums).parent / 'bank_pipeline'}")
    cli = cli or _pinned_cli()
    return cli.main(argv, collector=archive_collector, rehash=rehash)


if __name__ == "__main__":
    _ensure_path()
    os.chdir(BANK_ROOT)
    sys.exit(main())
