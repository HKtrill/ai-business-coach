"""
shared.validation
=================
The audit ledger both notebooks record their validation checks in.

Each check is a named, zero-argument callable that returns truthy on pass.
Checks are grouped by *section* (e.g. ``"2. Split and alignment"``) and tagged
with the *stage* they audit. A check that raises is recorded as a failure with
the exception text, rather than stopping the notebook half-way through a
section, so one run shows every problem at once. ``raise_if_failed`` then stops
the notebook at the end of the section.

Why it is shared
----------------
The GLASS and black-box notebooks are audited against the same contract. If
each defines its own checker, "all checks passed" means different things in
each. One class, one output format, one failure rule.

Public API
----------
ValidationReport
    ``check`` / ``show`` / ``raise_if_failed`` / ``summary``; ``rows`` holds
    every recorded result.
ValidationError
    Raised by ``raise_if_failed``.

Examples
--------
>>> VALIDATION = ValidationReport()
>>> _SEC = "2. Split and alignment"
>>> VALIDATION.check(_SEC, "Stage 1", "OOF index = X_train",
...                  lambda: out.train_proba_oof.index.equals(X_train.index),
...                  f"{len(X_train):,} rows")
>>> VALIDATION.show(_SEC)
>>> VALIDATION.raise_if_failed(_SEC)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

__all__ = ["ValidationReport", "ValidationError"]


class ValidationError(AssertionError):
    """One or more validation checks failed. The message lists each failure."""


class ValidationReport:
    """
    Ordered ledger of pass/fail checks, grouped by section and stage.

    Attributes
    ----------
    rows : list of dict
        One dict per check, in the order recorded, with keys ``section``,
        ``stage``, ``check``, ``passed`` (bool), ``detail`` (str) and
        ``error`` (str; empty unless the check raised).

    Notes
    -----
    Re-recording a check with the same (section, stage, check) replaces the
    earlier row, so re-running a notebook cell does not duplicate entries.
    """

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def check(
        self,
        section: str,
        stage: str,
        name: str,
        fn: Callable[[], Any],
        detail: str = "",
    ) -> bool:
        """
        Run one check and record the result.

        Parameters
        ----------
        section : str
            Group label, e.g. ``"3. Output contracts"``.
        stage : str
            Which stage is audited, e.g. ``"Stage 1"``.
        name : str
            What is being asserted, phrased as the passing condition.
        fn : callable
            Zero-argument; truthy means pass. Bind loop variables as default
            arguments (``lambda s=s: ...``) — a bare closure sees only the last
            loop value.
        detail : str, optional
            Context shown beside the result (a count, a value).

        Returns
        -------
        bool
            Whether the check passed.
        """
        error = ""
        try:
            passed = bool(fn())
        except Exception as exc:  # noqa: BLE001 — a crash is a failed check
            passed, error = False, f"{type(exc).__name__}: {exc}"

        key = (section, stage, name)
        self.rows = [r for r in self.rows
                     if (r["section"], r["stage"], r["check"]) != key]
        self.rows.append({"section": section, "stage": stage, "check": name,
                          "passed": passed, "detail": detail, "error": error})
        return passed

    # ------------------------------------------------------------------
    def frame(self, section: Optional[str] = None) -> pd.DataFrame:
        """
        Results as a DataFrame.

        Parameters
        ----------
        section : str, optional
            Restrict to one section. All sections when omitted.

        Returns
        -------
        pandas.DataFrame
            Columns ``section``, ``stage``, ``check``, ``passed``, ``detail``,
            ``error``.
        """
        df = pd.DataFrame(self.rows, columns=["section", "stage", "check",
                                              "passed", "detail", "error"])
        return df if section is None else df[df["section"] == section]

    def show(self, section: Optional[str] = None) -> None:
        """
        Display results as a table with ✅ / ❌.

        Parameters
        ----------
        section : str, optional
            One section, or everything when omitted.

        Notes
        -----
        Uses ``IPython.display`` in a notebook, plain ``print`` elsewhere.
        """
        df = self.frame(section).copy()
        n_pass = int(df["passed"].sum())
        title = section or "All sections"
        df.insert(0, "", df.pop("passed").map({True: "✅", False: "❌"}))
        df["detail"] = [e or d for d, e in zip(df["detail"], df["error"])]
        df = df.drop(columns=["section", "error"]).reset_index(drop=True)

        print(f"{title} — {n_pass}/{len(df)} passed")
        try:
            from IPython.display import display
            display(df.style.hide(axis="index").set_properties(**{"text-align": "left"}))
        except Exception:  # noqa: BLE001 — not in a notebook
            print(df.to_string(index=False))

    def raise_if_failed(self, section: Optional[str] = None) -> None:
        """
        Stop the notebook if any check in ``section`` failed.

        Parameters
        ----------
        section : str, optional
            One section, or everything when omitted.

        Raises
        ------
        ValidationError
            Listing each failed check and its detail or error.
        """
        df = self.frame(section)
        bad = df[~df["passed"]]
        if len(bad):
            lines = [f"  [{r.stage}] {r.check} — {r.error or r.detail}"
                     for r in bad.itertuples()]
            raise ValidationError(
                f"{len(bad)} check(s) failed in {section or 'validation'}:\n"
                + "\n".join(lines)
            )

    def summary(self) -> pd.DataFrame:
        """
        Pass counts per section and stage.

        Returns
        -------
        pandas.DataFrame
            Indexed by (section, stage); columns ``passed``, ``total``.
        """
        df = self.frame()
        return (df.groupby(["section", "stage"], sort=False)["passed"]
                  .agg(passed="sum", total="count"))

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = len(self.rows)
        ok = sum(r["passed"] for r in self.rows)
        return f"ValidationReport({ok}/{n} passed)"
