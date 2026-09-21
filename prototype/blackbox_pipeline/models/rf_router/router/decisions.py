"""
Decision codes and labels for the Stage 2 output contract.

Mirrors ``glass_brw.core.rule`` — duplicated rather than imported so the
black-box package carries no dependency on the GLASS package.

Not to be confused with ``rf_router.constraints``, which carries the operating
constraints the thresholds are solved against. Nothing here is configurable.
"""

from __future__ import annotations

__all__ = ["SUBSCRIBE", "NOT_SUBSCRIBE", "ABSTAIN", "DECISION_DTYPE",
           "PASS1", "PASS2", "UNCERTAIN"]

SUBSCRIBE = 1
NOT_SUBSCRIBE = 0
ABSTAIN = -1

PASS1 = "pass1"
PASS2 = "pass2"
UNCERTAIN = "uncertain"

DECISION_DTYPE = object