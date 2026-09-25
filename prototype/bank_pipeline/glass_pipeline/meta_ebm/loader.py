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
    router, c2 = router_stream(router_path, "glass", "GLASS Router", ref.y_train)
    ebm, c3 = stage3_stream(ebm_path, "ebm", "GLASS EBM")

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
