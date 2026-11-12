"""
PR 38 command line, run from ``prototype/bank_pipeline``:

    python -m stage2_crossfit phase0    --sums <SHA256SUMS.txt> --artifact-manifest <json> --run-id <id>
    python -m stage2_crossfit configs   --run-id <id> [--feature-pathway P]
    python -m stage2_crossfit g13       --run-id <id> [--fold 0] [--feature-pathway P]
    python -m stage2_crossfit preflight --run-id <id> [--fold 0] [--feature-pathway P]   (Step 2B)
    python -m stage2_crossfit authorize --run-id <id> --authorization <json> [--feature-pathway P]
    python -m stage2_crossfit primary   --run-id <id> --authorization <json> [--feature-pathway P]
                                                                                 (Commit 3)

``--feature-pathway`` is the inner-fold feature pathway (G13): ``outer_engineer``
(historical, the default) or ``fold_label_excluded`` (the correction).

Records go to ``research_logs/pr38/<run_id>/`` (version-controlled) and are
written once: an existing record is never replaced. ``g13`` and ``preflight``
refuse to run unless that run's Phase 0 record passed. Nothing here reads a
test label or a held-out fold's labels, or computes an outer-OOF metric.
``primary`` writes candidate artifacts only (structural acceptance: Commit 3A).
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
    for name in ("phase0", "configs", "g13", "preflight", "authorize", "primary"):
        sp = p0 if name == "phase0" else sub.add_parser(name)
        sp.add_argument("--run-id", required=True)
        if name != "phase0":
            sp.add_argument("--feature-pathway", choices=IF.PATHWAYS, default=IF.PATHWAY_OUTER)
        if name in ("g13", "preflight"):
            sp.add_argument("--fold", type=int, default=R.PREFLIGHT_FOLD)
            sp.add_argument("--verbose", action="store_true")
        if name in ("authorize", "primary"):
            sp.add_argument("--authorization", required=True)
        if name == "primary":
            sp.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    out_dir = _record_dir(args.run_id)

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
    if args.cmd == "primary":                       # before any data is loaded
        _authorization_for(args, settings)
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
    result = R.run_crossfit(ts, settings, verbose=args.verbose)
    R.fit_full_train(ts, settings, result, verbose=args.verbose)
    saved = R.save_run(result, ts, settings, root, args.run_id)
    R.write_record(out_dir / "primary_MANIFEST.json", saved["manifest"])
    status = {arm: [ev["fit_status"]["status"] for ev in evs]
              for arm, evs in result.evidence.items()}
    print(f"primary CANDIDATE artifacts written -> {root}; fold fit statuses {status}")
    print("not accepted: structural acceptance (V1-V8) is Commit 3A; no outer-OOF metric has "
          "been computed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
