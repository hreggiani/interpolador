"""Solvers turning tabulated grid data into a correction for a measured abundance.

Two problems show up:

1. Published Delta = A(NLTE) - A(LTE) tables are indexed by the *model* (non-LTE)
   abundance, but a measurement gives the LTE abundance.  Reading Delta straight
   off at A(LTE) is only correct where Delta is flat in abundance; elsewhere the
   answer must satisfy ``A_nlte - Delta(A_nlte) = A_lte``.  See :func:`solve_delta`.

2. Some grids (Nordlander & Lind 2017 Al, Osorio 2016 Mg) publish LTE and non-LTE
   equivalent widths rather than corrections, so the correction has to be derived
   by matching equivalent widths.  See :func:`solve_ew_match`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

DEFAULT_TOL = 1e-4
DEFAULT_MAX_ITER = 30


@dataclass
class DeltaSolution:
    """Result of the self-consistent abundance solve, vectorised over points."""

    a_nlte: np.ndarray
    delta: np.ndarray
    converged: np.ndarray
    n_iter: int

    def item(self):
        return (
            float(self.a_nlte.reshape(-1)[0]),
            float(self.delta.reshape(-1)[0]),
            bool(self.converged.reshape(-1)[0]),
        )


def solve_delta(delta_fn, a_lte, bounds=None, mode: str = "selfconsistent",
                tol: float = DEFAULT_TOL, max_iter: int = DEFAULT_MAX_ITER) -> DeltaSolution:
    """Solve ``A_nlte - Delta(A_nlte) = A_lte`` for the non-LTE abundance.

    Parameters
    ----------
    delta_fn
        Callable mapping an array of trial abundances to an array of Delta
        values.  May return NaN where the grid has no coverage.
    a_lte
        Measured LTE abundance(s), log eps.
    bounds
        ``(lo, hi)`` abundance range of the grid, used by the bracketing
        fallback.  Defaults to the range spanned by the fixed-point iterates.
    mode
        ``"selfconsistent"`` (default) iterates; ``"direct"`` evaluates Delta once
        at ``a_lte``, which is what a naive table lookup does and is kept for
        comparison work.
    """
    a_lte = np.atleast_1d(np.asarray(a_lte, dtype=float))

    if mode == "direct":
        delta = np.asarray(delta_fn(a_lte), dtype=float).reshape(a_lte.shape)
        return DeltaSolution(
            a_nlte=a_lte + delta,
            delta=delta,
            converged=np.isfinite(delta),
            n_iter=0,
        )
    if mode != "selfconsistent":
        raise ValueError(f"unknown mode {mode!r}")

    current = a_lte.copy()
    delta = np.full_like(a_lte, np.nan)
    converged = np.zeros(a_lte.shape, dtype=bool)
    used = 0

    for used in range(1, max_iter + 1):
        d = np.asarray(delta_fn(current), dtype=float).reshape(a_lte.shape)
        nxt = a_lte + d
        step = np.abs(nxt - current)
        # Under-relax where the iteration overshoots; |dDelta/dA| > 1 happens on
        # saturated lines and would otherwise oscillate forever.
        oscillating = step > 0.5
        nxt = np.where(oscillating, current + 0.5 * (nxt - current), nxt)

        newly_done = np.isfinite(d) & (step < tol) & ~converged
        delta = np.where(np.isfinite(d), d, delta)
        converged |= newly_done
        current = np.where(np.isfinite(d), nxt, current)
        if converged.all():
            break

    stuck = ~converged & np.isfinite(delta)
    if stuck.any():
        lo, hi = _default_bounds(bounds, a_lte, delta)
        for i in np.flatnonzero(stuck):
            root, ok = _bracketed_solve(delta_fn, float(a_lte[i]), lo, hi)
            if ok:
                current[i] = root
                delta[i] = root - a_lte[i]
                converged[i] = True

    a_nlte = np.where(converged, a_lte + delta, np.nan)
    delta = np.where(converged, delta, np.nan)
    return DeltaSolution(a_nlte=a_nlte, delta=delta, converged=converged, n_iter=used)


def _default_bounds(bounds, a_lte, delta):
    if bounds is not None:
        return float(bounds[0]), float(bounds[1])
    pad = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1.0
    return float(np.min(a_lte) - 2 * pad - 1), float(np.max(a_lte) + 2 * pad + 1)


def _bracketed_solve(delta_fn, a_lte: float, lo: float, hi: float):
    """Root-find ``A - Delta(A) - a_lte`` on a grid of samples between lo and hi."""

    def f(a):
        d = np.asarray(delta_fn(np.array([a], dtype=float)), dtype=float).reshape(-1)[0]
        return np.nan if not np.isfinite(d) else a - d - a_lte

    xs = np.linspace(lo, hi, 41)
    fs = np.array([f(x) for x in xs])
    ok = np.isfinite(fs)
    for i in range(len(xs) - 1):
        if ok[i] and ok[i + 1] and fs[i] == 0:
            return float(xs[i]), True
        if ok[i] and ok[i + 1] and fs[i] * fs[i + 1] < 0:
            try:
                return float(brentq(f, xs[i], xs[i + 1], xtol=DEFAULT_TOL)), True
            except (ValueError, RuntimeError):
                return float("nan"), False
    return float("nan"), False


@dataclass
class EWMatch:
    """Result of equivalent-width matching for a single point."""

    a_nlte: float
    delta: float
    ok: bool
    note: str = ""


def _longest_increasing_run(y: np.ndarray) -> slice:
    """Longest strictly increasing run in ``y`` (assumed finite)."""
    best_start = best_len = 0
    start = 0
    for i in range(1, y.size):
        if y[i] <= y[i - 1]:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = i
    if y.size - start > best_len:
        best_start, best_len = start, y.size - start
    return slice(best_start, best_start + best_len)


def solve_ew_match(abund_nodes, logew_lte, logew_nlte, a_lte: float) -> EWMatch:
    """Derive Delta by matching equivalent widths.

    The LTE curve gives the equivalent width a measurement of ``a_lte`` implies;
    inverting the non-LTE curve at that same width gives the abundance a non-LTE
    analysis would have reported.  Curves are restricted to their longest
    strictly increasing run, because strong lines saturate and the growth curve
    flattens (and occasionally reverses through interpolation noise).
    """
    abund = np.asarray(abund_nodes, dtype=float)
    lte = np.asarray(logew_lte, dtype=float)
    nlte = np.asarray(logew_nlte, dtype=float)

    m_lte = np.isfinite(lte)
    m_nlte = np.isfinite(nlte)
    if m_lte.sum() < 2 or m_nlte.sum() < 2:
        return EWMatch(np.nan, np.nan, False, "insufficient EW coverage")

    a_l, y_l = abund[m_lte], lte[m_lte]
    run_l = _longest_increasing_run(y_l)
    a_l, y_l = a_l[run_l], y_l[run_l]
    if a_l.size < 2:
        return EWMatch(np.nan, np.nan, False, "LTE curve not monotonic")
    if not (a_l[0] - 1e-9 <= a_lte <= a_l[-1] + 1e-9):
        return EWMatch(np.nan, np.nan, False, "A(LTE) outside grid abundance range")

    target = float(np.interp(a_lte, a_l, y_l))

    a_n, y_n = abund[m_nlte], nlte[m_nlte]
    run_n = _longest_increasing_run(y_n)
    a_n, y_n = a_n[run_n], y_n[run_n]
    if a_n.size < 2:
        return EWMatch(np.nan, np.nan, False, "non-LTE curve not monotonic")
    if target < y_n[0] or target > y_n[-1]:
        return EWMatch(np.nan, np.nan, False, "EW outside invertible non-LTE range (saturated)")

    a_nlte = float(np.interp(target, y_n, a_n))
    return EWMatch(a_nlte=a_nlte, delta=a_nlte - a_lte, ok=True)
