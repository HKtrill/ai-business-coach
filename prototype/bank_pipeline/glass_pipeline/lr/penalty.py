"""
lr.penalty
==========
``LogisticRegression`` penalty arguments that specify the same model on every
supported scikit-learn (requirements.txt: ``>=1.4,<2``).

scikit-learn 1.8 deprecated ``penalty=`` (removal in 1.10) in favour of
``l1_ratio`` alone: ``l2`` -> ``l1_ratio=0``, ``l1`` -> ``l1_ratio=1``,
``elasticnet`` -> its own ``l1_ratio``. On 1.8+ an explicit ``penalty``
still wins, so the old spelling fits the same model but warns on every fit
(and ``penalty='l1'`` with the default ``l1_ratio=0.0`` adds an
"Inconsistent values" warning). Before 1.8, ``l1_ratio`` is ignored unless
``penalty='elasticnet'``, so the old spelling is the only correct one there.

Equivalence on the installed version is checked fit-for-fit (identical
coefficients, intercepts and iteration counts) in
``tests/stage1/test_lr_penalty_compat.py``.

Edge case: on 1.8+ an elasticnet ``l1_ratio`` of exactly 0.0 or 1.0 resolves
to the pure l2 / l1 penalty — the same objective, solved without the
elasticnet bookkeeping. Optuna's continuous draw makes this practically
unreachable.
"""

from __future__ import annotations

from typing import Optional

import sklearn

__all__ = ["SKLEARN_PENALTY_DEPRECATED", "penalty_kwargs"]

SKLEARN_PENALTY_DEPRECATED: bool = (
    tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 8)
)

_RATIO = {"l2": 0.0, "l1": 1.0}


def penalty_kwargs(penalty: Optional[str], l1_ratio: Optional[float] = None) -> dict:
    """
    Keyword arguments selecting ``penalty`` for ``LogisticRegression``.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet'} or other
        The penalty as the stages and Optuna name it. Anything else (e.g.
        ``None``) is passed through unchanged: its 1.8+ spelling also changes
        ``C``, so it is left for the caller to migrate deliberately.
    l1_ratio : float, optional
        Required for ``'elasticnet'``; ignored otherwise.

    Returns
    -------
    dict
        ``{'l1_ratio': ...}`` on scikit-learn 1.8+, else ``{'penalty': ...}``
        (plus ``l1_ratio`` for elasticnet).
    """
    if penalty == "elasticnet" and l1_ratio is None:
        raise ValueError("l1_ratio is required for penalty='elasticnet'")
    if penalty not in ("l1", "l2", "elasticnet"):
        return {"penalty": penalty}
    if SKLEARN_PENALTY_DEPRECATED:
        return {"l1_ratio": _RATIO[penalty] if penalty in _RATIO else float(l1_ratio)}
    kw = {"penalty": penalty}
    if penalty == "elasticnet":
        kw["l1_ratio"] = float(l1_ratio)
    return kw
