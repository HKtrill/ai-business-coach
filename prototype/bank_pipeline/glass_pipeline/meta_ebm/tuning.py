"""
glass_pipeline.meta_ebm.tuning
================================
Arbiter threshold tuning for the Meta-EBM stage.

1D search over min_weighted_confidence — the sole abstention parameter.
Objective: best F2 on covered samples, subject to a minimum coverage floor.
The rule is the shared Stage 4 one (``shared.stage4.tune_min_confidence``),
applied to the arbiter's own weighted confidence, so the GLASS and
black-box arms select their abstention cut identically.

Train-side only: ``probs`` must be the OOF training streams and ``y_true``
the training labels. Test data never reaches this module.

History
-------
feature_research swept ``np.arange(0.03, 0.55, 0.02)``, abstaining when
max(conf1, conf0) < mc, scoring F1 on covered samples (raw count ≥ 50).
Ported with F2 and a coverage fraction ≥ 0.50. PR 36 F30: the fixed grid
is replaced by an exact search over every distinct weighted confidence on
the training rows (the grid left one eligible value once F25 put the
router's confidence on the LR / EBM scale). The arbiter is called once,
without abstention, and its decisions and confidences are searched
directly; abstaining never changes a retained row's decision.
"""

import numpy as np

from shared.stage4.protocol import tune_min_confidence


def tune_arbiter_threshold(
    arbiter_fn,
    probs:        tuple,
    masks:        tuple,
    y_true:       np.ndarray,
    lr_thresh:    float,
    ebm_thresh:   float,
    weights:      dict,
    min_confs:    np.ndarray = None,
    min_coverage: float = 0.50,
    **arbiter_kwargs,
) -> dict | None:
    """
    Choose min_weighted_confidence by F2 on retained training rows.

    Parameters
    ----------
    arbiter_fn   : meta_arbiter callable (must accept ``return_confidence``)
    probs        : (lr_prob_train, ebm_prob_train, glass_prob_train)
    masks        : (pass1_train, pass2_train)
    y_true       : training labels
    lr_thresh    : LR arbiter threshold (recall-targeted)
    ebm_thresh   : EBM arbiter threshold (recall-targeted)
    weights      : MODEL_WEIGHTS dict
    min_confs    : None (default) searches every distinct weighted confidence
                   (exact, PR 36 F30); an explicit array is searched with the
                   same rule
    min_coverage : minimum coverage fraction to be eligible (default 0.50)
    **arbiter_kwargs : passed to ``arbiter_fn`` unchanged (the router's
                   ``glass_base_rate`` for the training rows, PR 36 F25)

    Returns
    -------
    dict with keys: min_weighted_confidence, train_f2, train_coverage,
    search, n_candidates, n_eligible; or None if no cut met the coverage floor.
    """
    pred, _, _, conf = arbiter_fn(
        *probs, *masks,
        lr_thresh, ebm_thresh,
        weights,
        allow_abstain=False,
        return_confidence=True,
        **arbiter_kwargs,
    )
    best = tune_min_confidence(y_true, pred, conf, grid=min_confs, min_coverage=min_coverage)
    if best is None:
        return None
    return {
        'min_weighted_confidence': best['min_confidence'],
        'train_f2':                best['train_f2'],
        'train_coverage':          best['train_coverage'],
        'search':                  best['search'],
        'n_candidates':            best['n_candidates'],
        'n_eligible':              best['n_eligible'],
    }
