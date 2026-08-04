"""NaN-aware multilinear interpolation over correction grids.

Published non-LTE grids are heavily masked: ``abcor_c1_1d-1dl.sav`` is 78%
sentinel values, and the Amarsi 2020 CSVs run 13-20%.  Feeding those to
``scipy.interpolate.RegularGridInterpolator`` would return NaN for any cell
touching a single bad corner, which is the common case rather than the edge
case.  Both interpolators here instead drop invalid corners and renormalise the
surviving weights, reporting how much weight survived so the caller can decide
whether to trust the result.

Two geometries are supported:

``RectGrid``
    Fully rectilinear axes (most sources).

``HybridGrid``
    A scattered block tensored with rectilinear axes.  The Amarsi 2016 iron
    tables sample only 54 of 104 (teff, logg) combinations but carry the full
    product in (vmic, abundance).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import Delaunay

#: Fraction of interpolation weight that must land on valid corners.
DEFAULT_MIN_VALID_WEIGHT = 0.5

#: Absolute tolerance when deciding whether a query sits on an axis endpoint.
_EDGE_ATOL = 1e-9


@dataclass
class InterpResult:
    """Outcome of an interpolation, vectorised over query points."""

    value: np.ndarray
    in_range: np.ndarray
    valid_weight: np.ndarray

    def item(self) -> tuple[float, bool, float]:
        """Scalar view, for single-point queries."""
        return (
            float(self.value.reshape(-1)[0]),
            bool(self.in_range.reshape(-1)[0]),
            float(self.valid_weight.reshape(-1)[0]),
        )


def _as_query(points, ndim: int) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[-1] != ndim:
        raise ValueError(f"expected {ndim} coordinates per point, got {pts.shape[-1]}")
    return pts


def _bracket(axis: np.ndarray, x: np.ndarray, clip: bool):
    """Cell index and fractional position of ``x`` within a monotonic ``axis``."""
    lo, hi = axis[0], axis[-1]
    inside = (x >= lo - _EDGE_ATOL) & (x <= hi + _EDGE_ATOL)
    xq = np.clip(x, lo, hi)
    idx = np.searchsorted(axis, xq, side="right") - 1
    idx = np.clip(idx, 0, axis.size - 2)
    span = axis[idx + 1] - axis[idx]
    t = np.where(span > 0, (xq - axis[idx]) / np.where(span > 0, span, 1.0), 0.0)
    return idx, np.clip(t, 0.0, 1.0), (np.ones_like(inside) if clip else inside)


def _combine(weights: np.ndarray, values: np.ndarray, min_valid_weight: float):
    """Drop non-finite corners and renormalise ``weights`` over the survivors.

    ``weights`` and ``values`` are both ``(n_corners, n_points)``.
    """
    valid = np.isfinite(values)
    w = np.where(valid, weights, 0.0)
    total = w.sum(axis=0)
    safe = total > 0
    acc = (w * np.where(valid, values, 0.0)).sum(axis=0)
    out = np.divide(acc, total, out=np.full_like(acc, np.nan), where=safe)
    out = np.where(total >= min_valid_weight, out, np.nan)
    return out, total


class RectGrid:
    """Multilinear interpolation on a rectilinear grid, tolerant of NaN nodes.

    Parameters
    ----------
    axes
        Strictly increasing 1-D coordinate arrays, one per dimension.  Length-1
        axes are rejected; readers squeeze those out and record them as fixed
        parameters instead.
    values
        Array shaped like the axes.  Any non-finite entry is treated as missing;
        readers map their source's sentinel (``-1e29``, ``-1e30``, ...) to NaN.
    clip_axes
        Names or indices of axes where out-of-range queries are clamped to the
        edge rather than rejected.  Intended for coarse axes such as vmic.
    """

    def __init__(self, axes, values, names=None, clip_axes=(),
                 min_valid_weight: float = DEFAULT_MIN_VALID_WEIGHT):
        self.axes = [np.asarray(a, dtype=float) for a in axes]
        for i, a in enumerate(self.axes):
            if a.ndim != 1 or a.size < 2:
                raise ValueError(f"axis {i} must be 1-D with at least 2 nodes")
            if not np.all(np.diff(a) > 0):
                raise ValueError(f"axis {i} must be strictly increasing")
        self.values = np.asarray(values, dtype=float)
        expected = tuple(a.size for a in self.axes)
        if self.values.shape != expected:
            raise ValueError(
                f"values shape {self.values.shape} does not match axes {expected}"
            )
        self.names = list(names) if names is not None else [f"x{i}" for i in range(len(self.axes))]
        self.min_valid_weight = float(min_valid_weight)
        self._clip = self._resolve_clip(clip_axes)
        self._corners = np.array(list(product((0, 1), repeat=len(self.axes))), dtype=np.intp)

    def _resolve_clip(self, clip_axes) -> list[bool]:
        flags = [False] * len(self.axes)
        for key in clip_axes:
            i = self.names.index(key) if isinstance(key, str) else int(key)
            flags[i] = True
        return flags

    @property
    def ndim(self) -> int:
        return len(self.axes)

    def bounds(self) -> dict:
        return {n: (float(a[0]), float(a[-1])) for n, a in zip(self.names, self.axes)}

    def __call__(self, points) -> InterpResult:
        pts = _as_query(points, self.ndim)
        n_pts = pts.shape[0]

        idx = np.empty((self.ndim, n_pts), dtype=np.intp)
        frac = np.empty((self.ndim, n_pts), dtype=float)
        in_range = np.ones(n_pts, dtype=bool)
        for d, axis in enumerate(self.axes):
            i, t, ok = _bracket(axis, pts[:, d], self._clip[d])
            idx[d], frac[d] = i, t
            in_range &= ok

        n_corners = self._corners.shape[0]
        weights = np.ones((n_corners, n_pts), dtype=float)
        gather = np.empty((self.ndim, n_corners, n_pts), dtype=np.intp)
        for c in range(n_corners):
            for d in range(self.ndim):
                off = self._corners[c, d]
                weights[c] *= frac[d] if off else (1.0 - frac[d])
                gather[d, c] = idx[d] + off

        corner_values = self.values[tuple(gather)]
        value, total = _combine(weights, corner_values, self.min_valid_weight)
        value = np.where(in_range, value, np.nan)
        return InterpResult(value=value, in_range=in_range, valid_weight=total)


class HybridGrid:
    """Barycentric interpolation over scattered axes, tensored with rect axes.

    ``nodes`` are the scattered coordinates (``n_nodes x n_scattered``); a
    Delaunay triangulation is built on range-normalised coordinates so that
    axes with wildly different units -- teff in kelvin against log g in dex --
    triangulate sensibly.  ``values`` is ``(n_nodes, *rect_shape)``.
    """

    def __init__(self, nodes, rect_axes, values, names=None, rect_names=None,
                 clip_axes=(), min_valid_weight: float = DEFAULT_MIN_VALID_WEIGHT):
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.ndim != 2:
            raise ValueError("nodes must be 2-D (n_nodes, n_scattered)")
        self.rect_axes = [np.asarray(a, dtype=float) for a in rect_axes]
        for i, a in enumerate(self.rect_axes):
            if a.size < 2 or not np.all(np.diff(a) > 0):
                raise ValueError(f"rect axis {i} must be strictly increasing with >=2 nodes")
        self.values = np.asarray(values, dtype=float)
        expected = (self.nodes.shape[0],) + tuple(a.size for a in self.rect_axes)
        if self.values.shape != expected:
            raise ValueError(
                f"values shape {self.values.shape} does not match {expected}"
            )

        n_scat = self.nodes.shape[1]
        self.names = list(names) if names is not None else [f"s{i}" for i in range(n_scat)]
        self.rect_names = (
            list(rect_names) if rect_names is not None
            else [f"x{i}" for i in range(len(self.rect_axes))]
        )
        self.min_valid_weight = float(min_valid_weight)

        self._lo = self.nodes.min(axis=0)
        span = self.nodes.max(axis=0) - self._lo
        self._scale = np.where(span > 0, span, 1.0)
        self._tri = Delaunay((self.nodes - self._lo) / self._scale)

        self._clip = [False] * len(self.rect_axes)
        for key in clip_axes:
            if isinstance(key, str):
                if key in self.rect_names:
                    self._clip[self.rect_names.index(key)] = True
            else:
                self._clip[int(key)] = True
        self._corners = np.array(
            list(product((0, 1), repeat=len(self.rect_axes))), dtype=np.intp
        )

    @property
    def ndim(self) -> int:
        return self.nodes.shape[1] + len(self.rect_axes)

    def bounds(self) -> dict:
        out = {
            n: (float(self.nodes[:, i].min()), float(self.nodes[:, i].max()))
            for i, n in enumerate(self.names)
        }
        out.update({n: (float(a[0]), float(a[-1])) for n, a in zip(self.rect_names, self.rect_axes)})
        return out

    def __call__(self, points) -> InterpResult:
        pts = _as_query(points, self.ndim)
        n_scat = self.nodes.shape[1]
        n_pts = pts.shape[0]

        scaled = (pts[:, :n_scat] - self._lo) / self._scale
        simplex = self._tri.find_simplex(scaled)
        found = simplex >= 0
        safe_simplex = np.where(found, simplex, 0)

        transform = self._tri.transform[safe_simplex]
        delta = scaled - transform[:, n_scat, :]
        bary = np.einsum("pij,pj->pi", transform[:, :n_scat, :], delta)
        bary = np.concatenate([bary, 1.0 - bary.sum(axis=1, keepdims=True)], axis=1)
        vertices = self._tri.simplices[safe_simplex]          # (n_pts, n_scat + 1)

        in_range = found.copy()
        n_rect = len(self.rect_axes)
        idx = np.empty((n_rect, n_pts), dtype=np.intp)
        frac = np.empty((n_rect, n_pts), dtype=float)
        for d, axis in enumerate(self.rect_axes):
            i, t, ok = _bracket(axis, pts[:, n_scat + d], self._clip[d])
            idx[d], frac[d] = i, t
            in_range &= ok

        n_vert = vertices.shape[1]
        n_corners = self._corners.shape[0]
        weights = np.empty((n_vert * n_corners, n_pts), dtype=float)
        gathered = np.empty((n_vert * n_corners, n_pts), dtype=float)

        for c in range(n_corners):
            corner_w = np.ones(n_pts, dtype=float)
            rect_idx = []
            for d in range(n_rect):
                off = self._corners[c, d]
                corner_w *= frac[d] if off else (1.0 - frac[d])
                rect_idx.append(idx[d] + off)
            for v in range(n_vert):
                row = c * n_vert + v
                weights[row] = np.clip(bary[:, v], 0.0, None) * corner_w
                gathered[row] = self.values[(vertices[:, v], *rect_idx)]

        value, total = _combine(weights, gathered, self.min_valid_weight)
        value = np.where(in_range, value, np.nan)
        return InterpResult(value=value, in_range=in_range, valid_weight=total)


def build_grid(axes: dict, values: np.ndarray, scattered: tuple = (), **kwargs):
    """Build a :class:`RectGrid`, or a :class:`HybridGrid` if ``scattered`` is given.

    ``axes`` maps axis name to node values in the order matching ``values``.
    ``scattered`` names the leading axis (already collapsed to a node list by the
    caller) that is not rectilinear.
    """
    names = list(axes)
    if not scattered:
        return RectGrid(list(axes.values()), values, names=names, **kwargs)
    rect_names = [n for n in names if n not in scattered]
    return HybridGrid(
        nodes=axes["__nodes__"] if "__nodes__" in axes else np.column_stack(
            [axes[n] for n in scattered]
        ),
        rect_axes=[axes[n] for n in rect_names],
        values=values,
        names=list(scattered),
        rect_names=rect_names,
        **kwargs,
    )
