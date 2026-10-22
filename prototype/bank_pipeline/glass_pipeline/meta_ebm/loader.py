"""
glass_pipeline.meta_ebm.loader
==============================
GLASS Stage 1–3 artifacts → verified ``Stage4Inputs``.

All reading and checking is shared with the black-box arm
(``shared.stage4``); this module only says which adapter reads which file:

    lr     Stage 1  GLASS LR        shared.stage4.stage_output_stream
    glass  Stage 2  GLASS Router    shared.stage4.router_stream
    ebm    Stage 3  GLASS EBM       shared.stage4.stage3_stream

See ``shared.stage4.adapters`` for the canonical fields consumed. The GLASS
Router's ``train_outputs_oof`` is a flag (True) certifying that its top-level
``train_*`` arrays are the outer cross-fit outputs; abstention is spelled
"uncertain" and normalised to "abstain".

Meta-EBM convention: router probabilities on abstain rows (possibly NaN) are
set to 0.5. The arbiter never reads them — it only uses the router where it
votes — so the value is inert; it only keeps the stored arrays finite.

Router evidence (PR 36 F25): the arbiter reads each routed row's rule
positive rate back out of the router's probability (``router_evidence``).
That is exact only for a GLASS Router bundle, so the loader checks that the
bundle's ``training_base_rate`` is the training split's positive rate and
that every routed probability is the GLASS blend of its stored rule
precision (``train_confidence`` / ``test_confidence``) and base rate (per
fold model for out-of-fold rows). Other routers are refused.

Arm and family (PR 36 F32, F29): every Stage 1-3 artifact must belong to the
declared upstream arm (``shared.stage4.upstream``). By default that is the
GLASS arm — a GLASS-arm StageOutput, the GLASS Router bundle and a GLASS EBM
artifact (``model_family`` 'ebm', ``config['arm']`` 'glass'); before F32 a
black-box MLP or XGBoost artifact, or one labelled 'unset' (PR 36 F21),
loaded as "GLASS LR" / "GLASS EBM".

The 2x2 crossover (``upstream="blackbox"``): the arbiter reads the MLP, the
RF Router and XGBoost in its LR / router / EBM slots. Its LR and EBM rules
(recall-targeted cut, |p - t|) take any calibrated scorer. Its router rule
(F25) reads a matched unit's positive rate back out of the router's
probability, and the RF Router's probabilities are the same blend by design
(``_BaseRouter.predict_proba``: Pass 1 ``b(1 - conf)``, Pass 2
``b + (1 - b) conf``) with ``conf`` its Pass 1 band NPV / Pass 2 band
precision, measured on its training OOF. So the evidence is the routed BAND's
positive rate against the RF Router's training base rate (per fold router on
out-of-fold rows), verified per row exactly as for a GLASS bundle — never
an RF probability read as GLASS rule precision.
"""

from __future__ import annotations

import numpy as np

from shared.stage4 import (
    ABSTAIN,
    PASS1,
    PASS2,
    Stage4Inputs,
    build_stage4_inputs,
    router_stream,
    split_reference,
    stage3_stream,
    stage_output_stream,
)
from shared.stage4.adapters import load
from shared.stage4.contract import SEC_SEM
from shared.stage4.upstream import UPSTREAM, resolve_upstream

from .router_evidence import fold_base_rates, glass_blend_mismatch

__all__ = ["load_stage_outputs", "ROUTER_EVIDENCE", "PASS1", "PASS2", "ABSTAIN"]

#: Upstream arm -> (router artifact, its Stage 4 layout, the unit whose
#: positive rate the arbiter reads as router evidence — PR 36 F25 / F29).
ROUTER_EVIDENCE = {
    "glass": ("GLASS Router bundle", "top-level", "rule"),
    "blackbox": ("RF Router artifact", "extra", "band"),
}


def load_stage_outputs(
    GLOBAL_SPLIT: dict,
    lr_path,
    router_path,
    ebm_path,
    verbose: bool = True,
    extra_fingerprints: dict = None,
    upstream: str = None,
) -> Stage4Inputs:
    """
    Load Stage 1–3 artifacts (paths or already-loaded objects) and return
    verified ``Stage4Inputs``. Raises ``Stage4InputError`` on a structurally
    invalid artifact and ``shared.validation.ValidationError`` (listing every
    failure) on any split / alignment / fold / OOF-honesty / router-semantics
    / upstream-identity violation.

    ``upstream``: the arm the artifacts come from — ``"glass"`` (default) or,
    for the 2x2 crossover, ``"blackbox"`` (PR 36 F29).
    """
    arm = resolve_upstream(upstream, "glass")
    labels = [UPSTREAM[arm][s]["label"] for s in ("stage1", "stage2", "stage3")]
    ref = split_reference(GLOBAL_SPLIT, extra_fingerprints)
    lr, c1 = stage_output_stream(lr_path, "lr", labels[0], len(ref.y_train))
    router_art, router_file = load(router_path)
    router, c2 = router_stream(router_art, "glass", labels[1], ref.y_train)
    router.artifact_path = router_file
    ebm, c3 = stage3_stream(ebm_path, "ebm", labels[2])
    c2 = c2 + _router_evidence_checks(router_art, router, ref.y_train, arm)

    for side in ("train", "test"):
        p = getattr(router, f"{side}_proba")
        d = getattr(router, f"{side}_decisions")
        if len(p) == len(d):
            n = int((np.isnan(p) & (d == ABSTAIN)).sum())
            router.notes[f"{side}_abstain_nan_set_to_0.5"] = n
            setattr(router, f"{side}_proba", np.where((d == ABSTAIN) & np.isnan(p), 0.5, p))

    return build_stage4_inputs(GLOBAL_SPLIT, [lr, router, ebm],
                               extra_checks=c1 + c2 + c3, verbose=verbose,
                               extra_fingerprints=extra_fingerprints,
                               upstream={"arm": arm, "arbiter": "meta_ebm", "arbiter_arm": "glass"})


def _router_evidence_checks(art, router, y_train, arm: str = "glass") -> list:
    """
    PR 36 F25: the arbiter's router evidence needs a router whose probabilities
    are blends of its stored per-row confidence at its training base rate —
    the GLASS Router bundle (rule precision) or, in the declared crossover
    (F29), the RF Router artifact (band precision). Each is checked per row.
    """
    what, layout, unit = ROUTER_EVIDENCE[arm]
    art = art if isinstance(art, dict) else {}
    fields = art.get("extra") if layout == "extra" and isinstance(art.get("extra"), dict) else art
    b = art.get("training_base_rate")
    rate = float(np.mean(y_train))
    stage = "stage2"

    def blend(side):
        base = (fold_base_rates(y_train, router.train_fold_id) if side == "train" else rate)
        dec = getattr(router, f"{side}_decisions")
        why = glass_blend_mismatch(getattr(router, f"{side}_proba"), dec == PASS1, dec == PASS2,
                                   base, np.asarray(fields[f"{side}_confidence"], dtype=float))
        if why:
            raise ValueError(why)
        return True

    return [
        (SEC_SEM, stage, f"glass: {what} with training_base_rate (F25 router evidence)",
         lambda: b is not None and router.notes.get("artifact_layout") == layout,
         f"the arbiter's router confidence reads the {unit} blend of a {what}"),
        (SEC_SEM, stage, "glass: training_base_rate = GLOBAL_SPLIT training positive rate",
         lambda: abs(float(b) - rate) < 1e-12, f"{b} vs {rate:.6f}"),
        (SEC_SEM, stage, f"glass: train probabilities = {unit} blends at fold base rates",
         lambda: blend("train"), "train_confidence, train_fold_id"),
        (SEC_SEM, stage, f"glass: test probabilities = {unit} blends at the training base rate",
         lambda: blend("test"), "test_confidence"),
    ]
