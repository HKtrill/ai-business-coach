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

Arm and family (PR 36 F32): Stage 1 must be a GLASS-arm StageOutput and
Stage 3 a GLASS EBM artifact (``model_family`` 'ebm', ``config['arm']``
'glass'). Before F32 a black-box MLP or XGBoost artifact, or one labelled
'unset' (PR 36 F21), loaded as "GLASS LR" / "GLASS EBM".
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

from .router_evidence import fold_base_rates, glass_blend_mismatch

__all__ = ["load_stage_outputs", "PASS1", "PASS2", "ABSTAIN"]


def load_stage_outputs(
    GLOBAL_SPLIT: dict,
    lr_path,
    router_path,
    ebm_path,
    verbose: bool = True,
    extra_fingerprints: dict = None,
) -> Stage4Inputs:
    """
    Load GLASS Stage 1–3 artifacts (paths or already-loaded objects) and
    return verified ``Stage4Inputs``. Raises ``Stage4InputError`` on a
    structurally invalid artifact and ``shared.validation.ValidationError``
    (listing every failure) on any split / alignment / fold / OOF-honesty /
    router-semantics violation.
    """
    ref = split_reference(GLOBAL_SPLIT, extra_fingerprints)
    lr, c1 = stage_output_stream(lr_path, "lr", "GLASS LR", len(ref.y_train))
    router_art, router_file = load(router_path)
    router, c2 = router_stream(router_art, "glass", "GLASS Router", ref.y_train)
    router.artifact_path = router_file
    ebm, c3 = stage3_stream(ebm_path, "ebm", "GLASS EBM")
    c2 = c2 + _glass_blend_checks(router_art, router, ref.y_train)
    c1 = c1 + _glass_arm_checks(lr, ebm)

    for side in ("train", "test"):
        p = getattr(router, f"{side}_proba")
        d = getattr(router, f"{side}_decisions")
        if len(p) == len(d):
            n = int((np.isnan(p) & (d == ABSTAIN)).sum())
            router.notes[f"{side}_abstain_nan_set_to_0.5"] = n
            setattr(router, f"{side}_proba", np.where((d == ABSTAIN) & np.isnan(p), 0.5, p))

    return build_stage4_inputs(GLOBAL_SPLIT, [lr, router, ebm],
                               extra_checks=c1 + c2 + c3, verbose=verbose,
                               extra_fingerprints=extra_fingerprints)


def _glass_blend_checks(art, router, y_train) -> list:
    """PR 36 F25: the arbiter's router evidence needs a GLASS rule-blend router."""
    fields = art if isinstance(art, dict) else {}
    b = fields.get("training_base_rate")
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
        (SEC_SEM, stage, "glass: GLASS Router bundle with training_base_rate (F25 router evidence)",
         lambda: b is not None and router.notes.get("artifact_layout") == "top-level",
         "the arbiter's router confidence is defined for the GLASS rule blend"),
        (SEC_SEM, stage, "glass: training_base_rate = GLOBAL_SPLIT training positive rate",
         lambda: abs(float(b) - rate) < 1e-12, f"{b} vs {rate:.6f}"),
        (SEC_SEM, stage, "glass: train probabilities = rule blends at fold base rates",
         lambda: blend("train"), "train_confidence, train_fold_id"),
        (SEC_SEM, stage, "glass: test probabilities = rule blends at the training base rate",
         lambda: blend("test"), "test_confidence"),
    ]


def _glass_arm_checks(lr, ebm) -> list:
    """PR 36 F32: the GLASS arbiter reads GLASS-arm Stage 1 and Stage 3 artifacts only."""
    arm1 = lr.notes.get("arm")
    fam3, arm3 = ebm.notes.get("model_family"), ebm.notes.get("arm")
    return [
        (SEC_SEM, "stage1", "lr: GLASS-arm Stage 1 output (StageOutput.arm = 'glass')",
         lambda: arm1 == "glass", f"arm={arm1!r}"),
        (SEC_SEM, "stage3", "ebm: GLASS EBM artifact (model_family 'ebm', arm 'glass')",
         lambda: fam3 == "ebm" and arm3 == "glass",
         f"model_family={fam3!r}, arm={arm3!r}"
         + (" (an 'unset' label is PR 36 F21: regenerate Stage 3)" if "unset" in (fam3, arm3)
            else "")),
    ]
