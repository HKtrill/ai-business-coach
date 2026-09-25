"""
blackbox_pipeline.models.rf_router.bands

Band confidence: how sure the router is about a decision it has made.

A threshold says which side of a cut a row falls on; it says nothing about how
reliable that side is. These classes measure it empirically from training
out-of-fold data — the observed precision of each band — and hand it back as the
``conf`` element of the Stage 2 triple.

Two implementations
-------------------
:class:`BandConfidence`
    One number per band. Every Pass 1 decision carries the band's NPV, every
    Pass 2 decision its precision. This is what both routers use.
:class:`BinnedBandConfidence`
    Confidence varies within a band, by score quantile. A row scoring far past
    the threshold gets more confidence than one scraping it.

Not to be confused with :mod:`~.threshhold.bandscan`, which enumerates which
bands a score can realise. That module is geometry over a sorted score vector;
this one measures outcomes.

Notes
-----
Confidence is measured on TRAINING OOF data only. ``from_oof`` takes the same
arrays the thresholds were solved from, so a band's confidence is the precision
the operating point was chosen under, not a fresh estimate from the test split.

**``BinnedBandConfidence`` and :func:`make_band_confidence` are currently
unwired.** ``_BaseRouter._make_bands`` always constructs ``BandConfidence``,
``RFRouterConfig`` exposes no bin-count knob, and the package's ``__init__``
exports only ``BandConfidence``. Before wiring the binned variant up, note that
``SingleRFRouter.predict`` and ``RFRouter.predict_train_oof`` call
``bands_.assign(decisions)`` with no scores, which :meth:`BinnedBandConfidence.assign`
rejects — those call sites need the scores threaded through first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = ["BandConfidence", "BinnedBandConfidence", "make_band_confidence"]


@dataclass
class BandConfidence:
    """
    One confidence value per band, measured over training OOF.

    Attributes
    ----------
    pass1_confidence : float
        Observed NPV of the Pass 1 band — ``P(y = 0 | p1 < t1)``.
    pass2_confidence : float
        Observed precision of the Pass 2 band — ``P(y = 1 | p2 > t2)``.
    pass1_support, pass2_support : int
        Rows in each band. Small support means a noisy confidence, which is why
        ``from_oof`` enforces a floor.
    ABSTAIN_CONFIDENCE : float, default 0.0
        Value assigned to abstained rows. The router expressed no opinion, so
        zero is the honest reading rather than a low-but-real probability.
    kind : str
        ``"flat"``. Distinguishes the serialised form from the binned variant.
    """

    pass1_confidence: float
    pass2_confidence: float
    pass1_support: int
    pass2_support: int
    ABSTAIN_CONFIDENCE: float = 0.0
    kind: str = "flat"

    @classmethod
    def from_oof(cls, *, pass1_proba, pass1_y, t1, pass2_proba, pass2_y, t2,
                 min_support: int = 1) -> "BandConfidence":
        """
        Measure both bands from training out-of-fold scores.

        Parameters
        ----------
        pass1_proba, pass1_y : array-like
            Pass 1 OOF scores and labels over the population it ran on.
        t1 : float
            Pass 1 cut. The band is ``p1 < t1``.
        pass2_proba, pass2_y : array-like
            Pass 2 OOF scores and labels. In the cascade these cover the
            remainder only, not the full training split.
        t2 : float
            Pass 2 cut. The band is ``p2 > t2``.
        min_support : int, default 1
            Minimum rows a band must contain.

        Returns
        -------
        BandConfidence

        Raises
        ------
        ValueError
            If either band holds fewer than ``min_support`` rows — a confidence
            estimated from a handful of rows is noise, and it would propagate
            into every ``predict_proba`` output.
        """
        p1 = np.asarray(pass1_proba, float); y1 = np.asarray(pass1_y).astype(int)
        p2 = np.asarray(pass2_proba, float); y2 = np.asarray(pass2_y).astype(int)
        routed, flagged = p1 < t1, p2 > t2
        n1, n2 = int(routed.sum()), int(flagged.sum())
        if n1 < min_support or n2 < min_support:
            raise ValueError(f"band support too small: pass1={n1}, pass2={n2} "
                             f"(min_support={min_support})")
        return cls(float((y1[routed] == 0).mean()), float((y2[flagged] == 1).mean()),
                   n1, n2)

    def assign(self, decisions, p1=None, p2=None) -> np.ndarray:
        """
        Confidence for each decision.

        Parameters
        ----------
        decisions : array-like
            Decision labels from ``apply_cuts``.
        p1, p2 : optional
            Ignored. Accepted so this is interchangeable with
            :meth:`BinnedBandConfidence.assign`, which needs them.

        Returns
        -------
        numpy.ndarray of float
            The band's confidence per row; ``ABSTAIN_CONFIDENCE`` where the
            router abstained.
        """
        conf = np.full(len(decisions), self.ABSTAIN_CONFIDENCE, dtype=float)
        conf[decisions == "pass1"] = self.pass1_confidence
        conf[decisions == "pass2"] = self.pass2_confidence
        return conf

    def to_dict(self) -> dict:
        """Serialise for the fit report and the saved artifact."""
        return {"kind": "flat",
                "pass1_confidence": float(self.pass1_confidence),
                "pass2_confidence": float(self.pass2_confidence),
                "pass1_support": int(self.pass1_support),
                "pass2_support": int(self.pass2_support)}

    @classmethod
    def from_dict(cls, d: dict) -> "BandConfidence":
        """Rebuild from :meth:`to_dict` output."""
        return cls(float(d["pass1_confidence"]), float(d["pass2_confidence"]),
                   int(d["pass1_support"]), int(d["pass2_support"]))


def _fit_bins(scores, labels, target_class, n_bins, min_per_bin):
    """
    Quantile-bin one band's scores and measure precision per bin.

    Returns ``(edges, precision, support)``. Interior quantile edges only — the
    outer edges are implicit, since every score in the band is on the band's
    side of the threshold by construction.

    Bins holding fewer than ``min_per_bin`` rows are merged into their left
    neighbour and the scan repeats, so no reported cell's precision is estimated
    from noise. The bin count is also capped at ``n // min_per_bin`` up front,
    and duplicate quantile edges (ties in the score) are dropped, which is why
    the realised bin count can be lower than ``n_bins``.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(int)
    n = len(s)
    n_bins = max(1, min(int(n_bins), n // max(1, min_per_bin)))

    if n_bins <= 1:
        return np.array([], float), np.array([float((y == target_class).mean())]), \
               np.array([n], int)

    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(s, qs))
    idx = np.digitize(s, edges, right=False)

    prec, supp, keep = [], [], []
    for b in range(len(edges) + 1):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        keep.append(b)
        prec.append(float((y[m] == target_class).mean()))
        supp.append(c)

    # Drop edges whose bin vanished, then merge undersized bins leftward.
    if len(keep) < len(edges) + 1:
        edges = np.array([edges[b - 1] for b in keep if b > 0], float)
        idx = np.digitize(s, edges, right=False)
        prec, supp = [], []
        for b in range(len(edges) + 1):
            m = idx == b
            prec.append(float((y[m] == target_class).mean()) if m.any() else 0.0)
            supp.append(int(m.sum()))

    while len(supp) > 1 and min(supp) < min_per_bin:
        b = int(np.argmin(supp))
        drop = b - 1 if b > 0 else 0
        edges = np.delete(edges, drop)
        idx = np.digitize(s, edges, right=False)
        prec, supp = [], []
        for k in range(len(edges) + 1):
            m = idx == k
            prec.append(float((y[m] == target_class).mean()) if m.any() else 0.0)
            supp.append(int(m.sum()))

    return np.asarray(edges, float), np.asarray(prec, float), np.asarray(supp, int)


@dataclass
class BinnedBandConfidence:
    """
    Confidence that varies within a band, by score quantile.

    A row well past the threshold is more reliable than one scraping it, and a
    single number per band throws that away. Each band is split into quantile
    bins and each bin carries its own measured precision.

    Attributes
    ----------
    pass1_edges, pass2_edges : numpy.ndarray
        Interior bin edges per band, ascending.
    pass1_precision, pass2_precision : numpy.ndarray
        Measured precision per bin. One longer than the edges.
    pass1_support, pass2_support : numpy.ndarray
        Rows per bin.
    ABSTAIN_CONFIDENCE : float, default 0.0
        Value for abstained rows.
    kind : str
        ``"binned"``.

    Notes
    -----
    Unwired at present — see the module docstring for what must change first.
    """

    pass1_edges: np.ndarray
    pass1_precision: np.ndarray
    pass1_support: np.ndarray
    pass2_edges: np.ndarray
    pass2_precision: np.ndarray
    pass2_support: np.ndarray
    ABSTAIN_CONFIDENCE: float = 0.0
    kind: str = "binned"

    @classmethod
    def from_oof(cls, *, pass1_proba, pass1_y, t1, pass2_proba, pass2_y, t2,
                 n_bins_pass1: int, n_bins_pass2: int,
                 min_per_bin: int = 50) -> "BinnedBandConfidence":
        """
        Bin both bands from training out-of-fold scores.

        Parameters are as :meth:`BandConfidence.from_oof`, plus the per-band bin
        counts and ``min_per_bin``, the floor below which neighbouring bins are
        merged.

        Raises
        ------
        ValueError
            If either band is empty.
        """
        p1 = np.asarray(pass1_proba, float); y1 = np.asarray(pass1_y).astype(int)
        p2 = np.asarray(pass2_proba, float); y2 = np.asarray(pass2_y).astype(int)
        routed, flagged = p1 < t1, p2 > t2
        if not routed.any() or not flagged.any():
            raise ValueError("empty band; cannot fit binned confidence")

        e1, c1, s1 = _fit_bins(p1[routed], y1[routed], 0, n_bins_pass1, min_per_bin)
        e2, c2, s2 = _fit_bins(p2[flagged], y2[flagged], 1, n_bins_pass2, min_per_bin)
        return cls(e1, c1, s1, e2, c2, s2)

    @property
    def pass1_confidence(self) -> float:
        """Support-weighted mean precision across the Pass 1 bins."""
        return float(np.average(self.pass1_precision, weights=self.pass1_support))

    @property
    def pass2_confidence(self) -> float:
        """Support-weighted mean precision across the Pass 2 bins."""
        return float(np.average(self.pass2_precision, weights=self.pass2_support))

    @property
    def n_distinct(self) -> int:
        """Distinct confidence values this object can emit, abstain included."""
        return len(self.pass1_precision) + len(self.pass2_precision) + 1

    def assign(self, decisions, p1=None, p2=None) -> np.ndarray:
        """
        Confidence for each decision, looked up by the row's own score.

        Parameters
        ----------
        decisions : array-like
            Decision labels from ``apply_cuts``.
        p1, p2 : array-like
            Per-row scores. **Required** — unlike the flat variant, this one
            cannot answer without them.

        Returns
        -------
        numpy.ndarray of float

        Raises
        ------
        ValueError
            If ``p1`` or ``p2`` is omitted.

        Notes
        -----
        A NaN Pass 2 score falls into the lowest bin rather than raising. That
        only arises for rows Pass 2 never scored, which cannot be flagged
        anyway, so the value is never read.
        """
        if p1 is None or p2 is None:
            raise ValueError("BinnedBandConfidence.assign needs p1 and p2 scores")
        conf = np.full(len(decisions), self.ABSTAIN_CONFIDENCE, dtype=float)

        m1 = decisions == "pass1"
        if m1.any():
            b = np.digitize(np.asarray(p1, float)[m1], self.pass1_edges, right=False)
            conf[m1] = self.pass1_precision[np.clip(b, 0, len(self.pass1_precision) - 1)]

        m2 = decisions == "pass2"
        if m2.any():
            s = np.asarray(p2, float)[m2]
            s = np.where(np.isnan(s), self.pass2_edges[0] if len(self.pass2_edges) else 0.0, s)
            b = np.digitize(s, self.pass2_edges, right=False)
            conf[m2] = self.pass2_precision[np.clip(b, 0, len(self.pass2_precision) - 1)]

        return conf

    def to_dict(self) -> dict:
        """Serialise for the fit report and the saved artifact."""
        return {"kind": "binned",
                "pass1_edges": self.pass1_edges.tolist(),
                "pass1_precision": self.pass1_precision.tolist(),
                "pass1_support": self.pass1_support.tolist(),
                "pass2_edges": self.pass2_edges.tolist(),
                "pass2_precision": self.pass2_precision.tolist(),
                "pass2_support": self.pass2_support.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "BinnedBandConfidence":
        """Rebuild from :meth:`to_dict` output."""
        return cls(np.asarray(d["pass1_edges"], float),
                   np.asarray(d["pass1_precision"], float),
                   np.asarray(d["pass1_support"], int),
                   np.asarray(d["pass2_edges"], float),
                   np.asarray(d["pass2_precision"], float),
                   np.asarray(d["pass2_support"], int))


def make_band_confidence(*, pass1_proba, pass1_y, t1, pass2_proba, pass2_y, t2,
                         n_bins_pass1: Optional[int] = None,
                         n_bins_pass2: Optional[int] = None,
                         min_support: int = 1, min_per_bin: int = 50):
    """
    Build whichever band-confidence variant the bin counts imply.

    Returns a :class:`BandConfidence` when either bin count is ``None``, and a
    :class:`BinnedBandConfidence` when both are given.

    Notes
    -----
    Unwired at present — ``_BaseRouter._make_bands`` calls
    ``BandConfidence.from_oof`` directly. See the module docstring.
    """
    if n_bins_pass1 is None or n_bins_pass2 is None:
        return BandConfidence.from_oof(
            pass1_proba=pass1_proba, pass1_y=pass1_y, t1=t1,
            pass2_proba=pass2_proba, pass2_y=pass2_y, t2=t2,
            min_support=min_support)
    return BinnedBandConfidence.from_oof(
        pass1_proba=pass1_proba, pass1_y=pass1_y, t1=t1,
        pass2_proba=pass2_proba, pass2_y=pass2_y, t2=t2,
        n_bins_pass1=n_bins_pass1, n_bins_pass2=n_bins_pass2,
        min_per_bin=min_per_bin)
