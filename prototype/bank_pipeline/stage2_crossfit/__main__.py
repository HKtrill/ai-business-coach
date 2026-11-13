"""
PR 38 command line, run from ``prototype/bank_pipeline``:

    python -m stage2_crossfit phase0    --sums <SHA256SUMS.txt> --artifact-manifest <json> --run-id <id>
    python -m stage2_crossfit configs   --run-id <id> [--feature-pathway P]
    python -m stage2_crossfit g13       --run-id <id> [--fold 0] [--feature-pathway P]
    python -m stage2_crossfit preflight --run-id <id> [--fold 0] [--feature-pathway P]   (Step 2B)
    python -m stage2_crossfit authorize --run-id <id> --authorization <json> [--feature-pathway P]
    python -m stage2_crossfit primary   --run-id <id> --authorization <json> [--feature-pathway P]
    python -m stage2_crossfit reproduce --run-id <id> --authorization <json> [--feature-pathway P]
    python -m stage2_crossfit accept    --run-id <id> --authorization <json> --sums <SHA256SUMS.txt>
                                        --artifact-manifest <json> [--feature-pathway P]
                                                                                 (Commit 3A)
    python -m stage2_crossfit complementarity --run-id <id> --sums <SHA256SUMS.txt> [--check]
                                                                                 (Step 3B)
    python -m stage2_crossfit s1        --run-id <id>
    python -m stage2_crossfit s1-accept --run-id <id>
    python -m stage2_crossfit s1-report --run-id <id> --sums <SHA256SUMS.txt> [--check]
                                                                                 (Step 3C)

``--feature-pathway`` is the inner-fold feature pathway (G13): ``outer_engineer``
(historical, the default) or ``fold_label_excluded`` (the correction).

Records go to ``research_logs/pr38/<run_id>/`` (version-controlled) and are
written once: an existing record is never replaced. ``g13`` and ``preflight``
refuse to run unless that run's Phase 0 record passed. Nothing here reads a
test label or a held-out fold's labels, or computes an outer-OOF metric.
``primary`` writes candidate artifacts only, once per run (a stop is recorded in
``primary_stopped.json``). ``reproduce`` refits the pre-declared fold (G7) and compares it
with the candidate; ``accept`` runs structural acceptance V1-V8
(``stage2_crossfit.acceptance``): neither computes an outer-OOF metric.
``complementarity`` is the first command that does, and only for a run whose
official structural acceptance re-verifies in full (``acceptance.require_accepted``,
before any data is read); its package is written once to
``research_logs/pr38/<run_id>/complementarity_v1/`` (``--check`` compares only).
``s1`` (G12) fits GLASS ``full`` once per run after the primary's acceptance
re-verifies; ``s1-accept`` records G6 and the S1 checks (S1 is accepted only if
they all pass); ``s1-report`` writes ``s1_sensitivity_v1/`` for an accepted S1
only (``stage2_crossfit.sensitivity``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _record_dir(run_id: str) -> Path:
    from .phase0 import RECORD_ROOT
    return RECORD_ROOT / run_id


def _require_phase0(run_id: str) -> dict:
    path = _record_dir(run_id) / "phase0.json"
    if not path.is_file():
        raise SystemExit(f"no Phase 0 record at {path}: run `python -m stage2_crossfit phase0` first")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("passed") is not True:
        raise SystemExit(f"Phase 0 did not pass ({path}); no real-data fit may run")
    return rec


def _authorization_for(args, settings) -> dict:
    """This run's authorization only: it must sit in this run's record directory."""
    from . import runner as R
    if Path(args.authorization).resolve().parent != _record_dir(args.run_id).resolve():
        raise R.RunnerError(f"authorization {args.authorization} is not in run "
                            f"{args.run_id!r}'s record directory {_record_dir(args.run_id)}")
    return R.require_authorization(args.authorization, settings, run_id=args.run_id)


def _slim(ev: dict) -> dict:
    """An evidence dict without the row-identity lists (kept in the run's artifacts)."""
    out = dict(ev)
    if isinstance(out.get("remainder"), dict):
        out["remainder"] = {k: v for k, v in out["remainder"].items() if k != "index"}
    return out


def main(argv=None) -> int:
    from . import evidence as E
    from . import inner_features as IF
    from . import phase0, runner as R

    ap = argparse.ArgumentParser(prog="python -m stage2_crossfit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("phase0", help="historical integrity checks and the output root")
    p0.add_argument("--sums", required=True, help="the preserved snapshot's SHA256SUMS.txt")
    p0.add_argument("--artifact-manifest", required=True, help="the $BANK_ARTIFACT_MANIFEST file")
    for name in ("phase0", "configs", "g13", "preflight", "authorize", "primary", "reproduce",
                 "accept"):
        sp = p0 if name == "phase0" else sub.add_parser(name)
        sp.add_argument("--run-id", required=True)
        if name != "phase0":
            sp.add_argument("--feature-pathway", choices=IF.PATHWAYS, default=IF.PATHWAY_OUTER)
        if name in ("g13", "preflight"):
            sp.add_argument("--fold", type=int, default=R.PREFLIGHT_FOLD)
        if name in ("g13", "preflight", "reproduce"):
            sp.add_argument("--verbose", action="store_true")
        if name in ("authorize", "primary", "reproduce", "accept"):
            sp.add_argument("--authorization", required=True)
        if name == "primary":
            sp.add_argument("--verbose", action="store_true")
        if name == "accept":
            sp.add_argument("--sums", required=True, help="the preserved snapshot's SHA256SUMS.txt")
            sp.add_argument("--artifact-manifest", required=True,
                            help="the $BANK_ARTIFACT_MANIFEST file")
    cp = sub.add_parser("complementarity", help="Step 3B: the comparative analysis of an "
                                                "accepted run (writes its package once)")
    cp.add_argument("--run-id", required=True)
    cp.add_argument("--sums", required=True, help="the preserved snapshot's SHA256SUMS.txt")
    cp.add_argument("--check", action="store_true",
                    help="rebuild and compare with the written package; write nothing")
    for name, text in (("s1", "Step 3C: run S1 (GLASS full) for an accepted primary, once"),
                       ("s1-accept", "Step 3C: G6 and the S1 checks; record the S1 acceptance"),
                       ("s1-report", "Step 3C: the S1 package, for an accepted S1 only")):
        sp = sub.add_parser(name, help=text)
        sp.add_argument("--run-id", required=True)
        if name == "s1-report":
            sp.add_argument("--sums", required=True,
                            help="the preserved snapshot's SHA256SUMS.txt")
            sp.add_argument("--check", action="store_true",
                            help="rebuild and compare with the written package; write nothing")
    args = ap.parse_args(argv)
    out_dir = _record_dir(args.run_id)
    if args.cmd == "complementarity":
        return _commit3b(args)
    if args.cmd in ("s1", "s1-accept", "s1-report"):
        return _commit3c(args)

    if args.cmd == "phase0":
        rec = phase0.run_phase0(args.sums, args.artifact_manifest, args.run_id)
        path = R.write_record(out_dir / "phase0.json", rec)
        print(f"Phase 0 {'PASSED' if rec['passed'] else 'FAILED'} -> {path}")
        for k in ("snapshot", "baseline", "artifact_manifest", "inventory"):
            print(f"  {k:<18} {'pass' if rec[k]['passed'] else 'FAIL'}")
        return 0 if rec["passed"] else 1

    if args.cmd == "configs":
        settings = R.primary_settings(args.feature_pathway)
        s1 = R.s1_settings(args.feature_pathway)
        rec = {"format": R.CONFIGS_RECORD_FORMAT, "run_id": args.run_id,
               "primary": R.settings_record(settings), "s1": R.settings_record(s1)}
        path = R.write_record(out_dir / "configs.json", rec)
        print(f"G1 / G2 / G9 configuration record ({args.feature_pathway}) -> {path}")
        return 0

    if args.cmd == "authorize":                     # final review; fits nothing
        settings = R.primary_settings(args.feature_pathway)
        try:
            auth = _authorization_for(args, settings)
        except R.RunnerError as exc:
            print(f"AUTHORIZATION REFUSED\n{exc}")
            return 1
        refs = {k: {"path": auth[k], "sha256": auth[f"{k}_sha256"]}
                for k in ("g3_record", "g13_record", "g13_corrected_record", "configs_record",
                          "test_report") if auth.get(k)}
        review = {"format": "pr38.stage2.authorization_review/1",
                  "authorization": {"path": str(args.authorization),
                                    "sha256": R.sha256_file(args.authorization)},
                  "gates": auth["gates"], "approved_by": auth["approved_by"],
                  "approved_on": auth["approved_on"], "referenced": refs,
                  "feature_pathway": IF.pathway_identity(args.feature_pathway),
                  "g13_decision": auth["g13_decision"],
                  "f31_r2_clarification": auth["f31_r2_clarification"],
                  "frozen": settings.checks["identities"]}
        path = R.write_record(out_dir / "authorization_review.json", review)
        print(f"AUTHORIZED for the five-fold primary on {args.feature_pathway} -> {path}")
        return 0

    _require_phase0(args.run_id)
    settings = R.primary_settings(args.feature_pathway)
    auth = None
    if args.cmd in ("primary", "reproduce", "accept"):   # before any data is loaded
        auth = _authorization_for(args, settings)
    if args.cmd in ("reproduce", "accept"):
        return _commit3a(args, settings, auth, out_dir)
    ts = R.load_train_side()
    pop = R.check_population(ts, settings.protocol)
    if not pop["passed"]:
        R.write_record(out_dir / f"{args.cmd}_population_failure.json", pop)
        print("the train side does not match the frozen protocol's population; stopped")
        return 1

    if args.cmd in ("g13", "preflight"):
        res = R.fit_fold(ts, settings, args.fold, verbose=args.verbose,
                         keep_fits=args.cmd == "g13")
        fits = res.pop("fits", {})
        rec = {"run_id": args.run_id, "fold": args.fold, "population": pop,
               "feature_pathway": IF.pathway_identity(args.feature_pathway),
               "arms": {a: _slim(ev) for a, ev in res["arms"].items()}}
        if args.cmd == "preflight":
            rec.update(format=R.G3_RECORD_FORMAT,
                       g3=R.preflight_checks(res, protocol=settings.protocol))
            path = R.write_record(out_dir / f"g3_preflight_fold{args.fold}.json", rec)
            g3 = rec["g3"]
            print(f"G3 preflight {'PASSED' if g3['passed'] else 'FAILED'} -> {path}")
            print(f"  feasible {g3['feasible']}; fit statuses accepted "
                  f"{ {a: v['fit_status_accepted'] for a, v in g3['arms'].items()} }")
            print(f"  mandatory reviews before authorization: {g3['reviews_required'] or 'none'}")
            return 0 if g3["passed"] else 1
        from . import g13
        stops = [a for a, ev in res["arms"].items() if "stop" in ev]
        if stops:
            R.write_record(out_dir / f"g13_fold{args.fold}_fit_stopped.json", rec)
            print(f"G13: the fold-{args.fold} fit stopped for {stops}; nothing to audit")
            return 1
        X_tr, y_tr, _ = ts.fold_rows(args.fold)
        glass_fit = fits[E.ARM_GLASS]
        router, rf_bins, _ = fits[E.ARM_RF]
        rec["format"] = R.G13_RECORD_FORMAT
        rec["g13"] = g13.audit_fold(X_tr, y_tr, fold=args.fold, glass=glass_fit.glass,
                                    glass_bins=glass_fit.X_tr_eng, rf_router=router,
                                    rf_bins=rf_bins)
        name = (f"g13_fold{args.fold}.json" if args.feature_pathway == IF.PATHWAY_OUTER
                else f"g13_fold{args.fold}_{args.feature_pathway}.json")
        path = R.write_record(out_dir / name, rec)
        m = rec["g13"]["materiality_inputs"]
        print(f"G13 audit, fold {args.fold}, {args.feature_pathway} -> {path}")
        for arm, iso in rec["g13"]["isolation"].items():
            r = rec["g13"]["eligibility_retrained"][arm]
            print(f"  {arm:<5} corrected pathway: inner bins match the audit "
                  f"{iso['pathway_match']}, inner fits reproduce {iso['reproduced_all']}, "
                  f"complete {iso['complete']}; effect vs outer_engineer: "
                  f"{r.get('rows_changed')} eligibility rows")
        for arm in (a for a in ("glass", "rf") if a not in rec["g13"]["isolation"]):
            print(f"  {arm:<5} rows with any bin changed (label-excluded): "
                  f"{m.get(f'{arm}_rows_any_bin_changed')}; inner-OOF eligibility rows "
                  f"changed: direct {m.get(f'{arm}_eligibility_rows_changed')}, retrained "
                  f"{m.get(f'{arm}_retrained_eligibility_rows_changed')} (complete "
                  f"{m.get(f'{arm}_retrained_complete')})")
        if not rec["g13"]["inner_folds_shared_by_both_arms"]:
            print("  note: the arms' inner fold plans differ; see per_arm_inner_plan")
        print("  materiality: owner decision, documented before any primary result")
        return 0

    # primary (Commit 3): only with this run's passed Step 2B authorization (checked above)
    root = phase0.OUTPUT_ROOT / args.run_id / R.ROLE_PRIMARY
    try:
        out = R.run_primary(ts, settings, root, args.run_id, out_dir, population=pop,
                            attempt={"authorization": {
                                "name": Path(args.authorization).name,
                                "sha256": R.sha256_file(args.authorization)}},
                            verbose=args.verbose)
    except R.RunnerError as exc:
        print(f"primary STOPPED (not a result; nothing accepted)\n{exc}")
        return 1
    status = {arm: [ev["fit_status"]["status"] for ev in evs]
              for arm, evs in out["result"].evidence.items()}
    print(f"primary CANDIDATE artifacts written -> {root}; fold fit statuses {status}")
    print("not accepted: run `reproduce` (G7) and `accept` (V1-V8); no outer-OOF metric has "
          "been computed")
    return 0


def _commit3a(args, settings, auth, out_dir) -> int:
    """``reproduce`` (G7, the pre-declared fold) and ``accept`` (V1-V8) on this run's candidate."""
    from . import acceptance as A
    from . import runner as R
    if args.cmd == "accept" and (out_dir / A.ACCEPTANCE_FILE).exists():
        print(f"{out_dir / A.ACCEPTANCE_FILE} exists: this run is already accepted")
        return 1
    candidate = A.load_candidate(args.run_id)
    if not candidate.intact:
        print("the candidate FAILED its integrity checks; no artifact is loaded:")
        for c in candidate.integrity:
            print(f"  {'pass' if c['passed'] else 'FAIL'}  {c['check']}: {c['detail']}")
        if args.cmd == "reproduce":
            return 1

    if args.cmd == "reproduce":
        path = out_dir / A.G7_FILE
        if path.exists():
            print(f"{path} exists: the G7 record is written once")
            return 1
        ts = R.load_train_side()
        if not R.check_population(ts, settings.protocol)["passed"]:
            print("the train side does not match the frozen protocol's population; stopped")
            return 1
        rec = A.reproduce_fold(ts, settings, candidate, A.G7_FOLD, verbose=args.verbose)
        R.write_record(path, rec)
        print(f"G7 reproduction, fold {A.G7_FOLD}: {'PASSED' if rec['passed'] else 'FAILED'} "
              f"(bit-identical {rec['bit_identical']}) -> {path}")
        for arm, a in rec["arms"].items():
            print(f"  {arm:<9} evidence differences {a['evidence']['n_differences']} (max float "
                  f"{a['evidence']['max_float_difference']:.3g}); held-out decisions identical "
                  f"{a['held_out']['decisions_identical']}")
        return 0 if rec["passed"] else 1

    inputs = A.acceptance_inputs(out_dir, args.authorization, args.sums, args.artifact_manifest)
    g7, other_g7 = A.load_g7(out_dir)
    dispositions, _ = A.load_dispositions(out_dir, args.run_id)
    kwargs = dict(authorization=auth, inputs=inputs, g7=g7, other_g7=other_g7,
                  dispositions=dispositions, upstream=None, g11=None)
    if candidate.intact:                            # fail closed: no data, no artifact otherwise
        kwargs.update(
            upstream=lambda trace, ref: A.upstream_identity(trace, ref, settings.protocol,
                                                            args.sums),
            g11=A.g11_checks(args.sums, args.artifact_manifest))
    rec = A.acceptance(candidate, R.load_train_side() if candidate.intact else None, settings,
                       **kwargs)
    path = A.write_acceptance(out_dir, rec)
    sa = rec["structural_acceptance"]
    print(f"structural acceptance {'ACCEPTED' if rec['accepted'] else 'NOT ACCEPTED'} "
          f"(official {sa['official']}) -> {path}")
    for gate, g in rec["gates"].items():
        failed = [c for c in g["checks"] if not c["passed"]]
        print(f"  {gate}  {g['status']:<7} {len(g['checks']) - len(failed)}/{len(g['checks'])} "
              f"checks{'' if not g['not_run'] else ' (' + g['not_run'] + ')'}")
        for c in failed:
            print(f"        - {c['check']}: {c['detail'][:300]}")
    if rec.get("required_dispositions"):
        print(f"  owner dispositions required: {rec['required_dispositions']}")
    print("no outer-OOF metric has been computed; Step 3B reads only an accepted candidate "
          "(acceptance.require_accepted)")
    return 0 if rec["accepted"] else 1


def _commit3b(args) -> int:
    """``complementarity``: refused unless the run's structural acceptance re-verifies."""
    from shared.stage2 import diagnostics as d
    import shared.stage2.evaluation_protocol as ep
    from . import complementarity as C
    from . import runner as R
    try:
        out = C.run(args.run_id, args.sums, check=args.check)
    except (R.RunnerError, ep.ProtocolError, d.RouterDiagnosticError) as exc:
        print(f"complementarity REFUSED; nothing was written\n{exc}")
        return 1
    print(f"complementarity package {out['status'].upper()} -> {out['directory']}")
    for p in out["problems"]:
        print(f"  - {p}")
    print("  primary outcomes, pooled (GLASS - RF; 95% stratified paired bootstrap; D-C1):")
    for r in out["statements"]:
        lo, hi, diff = r["difference_low"], r["difference_high"], r["difference"]
        ci = "undefined" if lo is None or hi is None else f"[{lo:.4g}, {hi:.4g}]"
        print(f"    {r['outcome']:<17} {r['metric']:<21} difference "
              f"{'undefined' if diff is None else format(diff, '.4g')} {ci}: {r['statement']}")
    for g in out["gaps"]:
        print(f"  comparative section not yet available: {g}")
    ok = out["status"] in ("written", "identical", "reproduced_with_code_drift")
    return 0 if ok else 1



def _commit3c(args) -> int:
    """``s1``, ``s1-accept``, ``s1-report``: each refused unless the primary is accepted."""
    from shared.stage2 import diagnostics as d
    import shared.stage2.evaluation_protocol as ep
    from . import runner as R
    from . import sensitivity as S
    try:
        if args.cmd == "s1":
            out = S.run_s1(args.run_id)
            print(f"S1 candidate written -> {out['saved']['root']}\n"
                  "no comparison has been computed; next: s1-accept (G6)")
            return 0
        if args.cmd == "s1-accept":
            rec, path = S.s1_acceptance(args.run_id)
            print(f"S1 {'ACCEPTED' if rec['accepted'] else 'NOT ACCEPTED'} -> {path}\n"
                  f"  {rec['statement']}")
            for name, gate in rec["gates"].items():
                failed = [c["check"] for c in gate["checks"] if not c["passed"]]
                print(f"  {name}: {gate['status']}" + (f"; failed: {failed}" if failed else "")
                      + (f" ({gate['not_run']})" if gate.get("not_run") else ""))
            return 0 if rec["accepted"] else 1
        out = S.report_s1(args.run_id, args.sums, check=args.check)
    except (R.RunnerError, ep.ProtocolError, d.RouterDiagnosticError) as exc:
        print(f"{args.cmd} REFUSED\n{exc}")
        return 1
    print(f"S1 package {out['status'].upper()} -> {out['directory']}")
    for p in out["problems"]:
        print(f"  - {p}")
    report = out["report"]
    print(f"  comparative report across {report['packages']}: "
          + ("complete" if not report["gaps"] else f"sections missing {report['gaps']}"))
    ok = out["status"] in ("written", "identical", "reproduced_with_code_drift")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
