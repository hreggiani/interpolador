"""The common representation every grid is loaded into.

A :class:`GridSource` is one element, one correction flavour, one line list, and
one interpolator per line.  Everything downstream -- line matching, the solvers,
:func:`nlte.correct`, the CLI -- consumes only this class, so adding a table
never requires touching the query code.

Queries are always expressed in canonical units: teff in K, logg in dex, feh in
dex, vmic in km/s, and abundance as A(X) = log eps.  Sources whose abundance axis
is [X/Fe] or an offset grid convert the *query* rather than their axis nodes,
because the abundance axis is only rectilinear in its native parametrisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interp import HybridGrid, RectGrid
from .solar import format_species, solar_abundance

#: Canonical stellar-parameter axis names, in the order interpolators expect.
PARAM_AXES = ("teff", "logg", "feh", "vmic")

#: Abundance axis parametrisations.
ABUND_KINDS = ("logeps", "xfe", "offset")


@dataclass(frozen=True)
class Line:
    """One spectral line in a grid.

    ``match`` holds every wavelength a query may use to reach this entry, which
    is how blended entries work: Amarsi's oxygen grid tabulates the 777 nm
    triplet as one entry, so all three components map to it.
    """

    wavelength: float          # Angstrom; representative value for display
    label: str                 # as published, e.g. "505.2nm" or "517nm(combined)"
    blend: bool = False
    match: tuple = ()

    def __post_init__(self):
        if not self.match:
            object.__setattr__(self, "match", (self.wavelength,))


@dataclass
class GridSource:
    """A queryable non-LTE correction table."""

    id: str
    reference: str
    element: str
    ion: int
    flavor: str                       # e.g. "1D NLTE - 1D LTE"
    kind: str                         # "delta" | "ew_pair"
    axis_names: tuple                 # interpolator coordinate order, ends in "abund"
    lines: list                       # list[Line]
    grids: list                       # per line: interpolator, or (lte, nlte) pair
    abund_kind: str = "logeps"
    abund_solar: float | None = None  # A(X)_sun used by an "xfe" axis
    abund_zero_point: float | None = None   # zero point of an "offset" axis
    #: Which abundance the tabulated Delta is indexed by.  Most grids tabulate
    #: against the model (non-LTE) abundance, so a measured LTE abundance has to
    #: be solved for; a grid indexed by the LTE abundance is read directly.
    indexed_by: str = "nlte"
    solar_table: str | None = None
    scattering: bool | None = None
    fixed: dict = field(default_factory=dict)   # collapsed length-1 axes
    alias: dict = field(default_factory=dict)   # axis name -> query parameter it reads
    abund_nodes: np.ndarray | None = None       # native nodes, for "ew_pair"
    #: Delta does not depend on abundance.  The grid still carries an abundance
    #: axis -- flat and clipped -- so every source has the same layout; the
    #: abundance it was computed at, if the table says, is in ``fixed``.
    abund_independent: bool = False
    notes: str = ""
    path: str = ""

    def __post_init__(self):
        if self.abund_kind not in ABUND_KINDS:
            raise ValueError(f"{self.id}: unknown abund_kind {self.abund_kind!r}")
        if self.indexed_by not in ("nlte", "lte"):
            raise ValueError(f"{self.id}: indexed_by must be 'nlte' or 'lte'")
        if self.axis_names[-1] != "abund":
            raise ValueError(f"{self.id}: abundance must be the last axis")
        if len(self.lines) != len(self.grids):
            raise ValueError(f"{self.id}: {len(self.lines)} lines but {len(self.grids)} grids")
        if self.abund_kind == "xfe" and self.abund_solar is None:
            raise ValueError(f"{self.id}: an 'xfe' abundance axis needs abund_solar")
        if self.abund_kind == "offset" and self.abund_zero_point is None:
            raise ValueError(f"{self.id}: an 'offset' abundance axis needs abund_zero_point")

    # -- parameters ------------------------------------------------------

    @property
    def species(self) -> str:
        """Element with its ionisation stage, e.g. ``"FeII"``."""
        return format_species(self.element, self.ion)

    @property
    def param_axes(self) -> tuple:
        """Stellar-parameter axes this source actually interpolates over."""
        return tuple(n for n in self.axis_names if n != "abund")

    @property
    def required_params(self) -> tuple:
        """Query parameters a caller must supply, after alias resolution.

        The Amarsi ``.sav`` grids carry separate microturbulence axes for the
        non-LTE and LTE sides; both read the single ``vmic`` a caller passes.
        """
        seen = []
        for n in self.param_axes:
            q = self.alias.get(n, n)
            if q not in seen:
                seen.append(q)
        return tuple(seen)

    @property
    def needs_feh(self) -> bool:
        """Whether a query must supply [Fe/H] -- as an axis or for the conversion."""
        return "feh" in self.axis_names or self.abund_kind in ("xfe", "offset")

    def _zero(self) -> float:
        return self.abund_solar if self.abund_kind == "xfe" else self.abund_zero_point

    def to_native_abundance(self, a_logeps, feh=None):
        """Convert A(X) = log eps to this grid's abundance parametrisation."""
        a = np.asarray(a_logeps, dtype=float)
        if self.abund_kind == "logeps":
            return a
        if feh is None:
            raise ValueError(f"{self.id}: [Fe/H] is required to convert a '{self.abund_kind}' axis")
        return a - self._zero() - np.asarray(feh, dtype=float)

    def to_logeps(self, a_native, feh=None):
        """Inverse of :meth:`to_native_abundance`."""
        a = np.asarray(a_native, dtype=float)
        if self.abund_kind == "logeps":
            return a
        if feh is None:
            raise ValueError(f"{self.id}: [Fe/H] is required to convert a '{self.abund_kind}' axis")
        return a + self._zero() + np.asarray(feh, dtype=float)

    # -- geometry --------------------------------------------------------

    def _first_grid(self):
        g = self.grids[0]
        return g[0] if isinstance(g, tuple) else g

    def bounds(self) -> dict:
        """Axis bounds, in native units, from the first line's interpolator."""
        return self._first_grid().bounds()

    def abundance_bounds(self, feh=None) -> tuple:
        """Abundance range of the grid expressed as log eps.

        For an [X/Fe] or offset axis the range depends on [Fe/H], so passing an
        array of metallicities returns arrays.
        """
        lo, hi = self.bounds()["abund"]
        return self.to_logeps(lo, feh), self.to_logeps(hi, feh)

    def geometry(self) -> str:
        grid = self._first_grid()
        label = getattr(grid, "geometry_label", None)
        if label:
            return label
        return "hybrid" if isinstance(grid, HybridGrid) else "rectilinear"

    def coverage(self) -> float:
        """Fraction of grid nodes that carry a usable value, averaged over lines."""
        fracs = []
        for g in self.grids:
            for part in (g if isinstance(g, tuple) else (g,)):
                # Model-backed sources have no tabulated nodes to count.
                if getattr(part, "values", None) is None:
                    return float("nan")
                fracs.append(float(np.isfinite(part.values).mean()))
        return float(np.mean(fracs)) if fracs else float("nan")

    def describe(self) -> str:
        b = self.bounds()
        if self.abund_independent:
            b.pop("abund", None)
        axes = ", ".join(f"{k}=[{v[0]:g}, {v[1]:g}]" for k, v in b.items())
        fixed = "".join(f"\n  fixed:      {k} = {v:g}" for k, v in self.fixed.items())
        wl = [ln.wavelength for ln in self.lines if np.isfinite(ln.wavelength)]
        span = f"{min(wl):.1f}-{max(wl):.1f} A" if wl else "n/a"
        if self.abund_independent:
            abundance = "not an axis (Delta does not depend on it)"
        else:
            abundance = self.abund_kind + (
                f" (zero point {self._zero():g})" if self.abund_kind != "logeps" else ""
            )
        return (
            f"{self.id}\n"
            f"  reference:  {self.reference}\n"
            f"  species:    {self.species}   flavour: {self.flavor}"
            f"   scattering: {self.scattering}\n"
            f"  kind:       {self.kind}   geometry: {self.geometry()}\n"
            f"  abundance:  {abundance}"
            + f"\n  axes:       {axes}{fixed}\n"
            f"  lines:      {len(self.lines)} ({span})\n"
            + (f"  coverage:   {self.coverage() * 100:.1f}% of nodes valid"
               if np.isfinite(self.coverage()) else "  coverage:   n/a (model-backed)")
            + (f"\n  notes:      {self.notes}" if self.notes else "")
        )

    # -- evaluation ------------------------------------------------------

    def _point(self, params: dict, a_native) -> np.ndarray:
        a_native = np.atleast_1d(np.asarray(a_native, dtype=float))
        cols = []
        for name in self.axis_names[:-1]:
            key = self.alias.get(name, name)
            v = params.get(key)
            if v is None:
                raise ValueError(f"{self.id}: query is missing required parameter {key!r}")
            cols.append(np.broadcast_to(np.asarray(v, dtype=float), a_native.shape))
        cols.append(a_native)
        return np.column_stack(cols)

    def delta_at(self, line_index: int, params: dict, a_logeps, feh=None):
        """Delta evaluated at a trial non-LTE abundance (``kind == "delta"`` only)."""
        if self.kind != "delta":
            raise TypeError(f"{self.id}: not a delta grid")
        native = self.to_native_abundance(a_logeps, feh)
        res = self.grids[line_index](self._point(params, native))
        return res

    def ew_curves(self, line_index: int, params: dict):
        """LTE and non-LTE log EW curves over the abundance nodes at fixed params."""
        if self.kind != "ew_pair":
            raise TypeError(f"{self.id}: not an equivalent-width grid")
        lte_grid, nlte_grid = self.grids[line_index]
        nodes = np.asarray(self.abund_nodes, dtype=float)
        pts = self._point(params, nodes)
        return nodes, lte_grid(pts).value, nlte_grid(pts).value


def make_grid(axes: dict, values: np.ndarray, scattered_names=(), scattered_nodes=None,
              clip_axes=(), min_valid_weight=None, max_node_distance=None):
    """Build the right interpolator for a set of axes.

    ``axes`` maps axis name to nodes, in the order matching ``values``.  If
    ``scattered_names`` is given, ``scattered_nodes`` holds the (n_nodes, k)
    coordinates of the leading scattered block and ``values`` is
    ``(n_nodes, *rect_shape)``.
    """
    kw = {} if min_valid_weight is None else {"min_valid_weight": min_valid_weight}
    if not scattered_names:
        return RectGrid(list(axes.values()), values, names=list(axes),
                        clip_axes=clip_axes, **kw)
    if max_node_distance is not None:
        kw["max_node_distance"] = max_node_distance
    rect_names = [n for n in axes if n not in scattered_names]
    return HybridGrid(
        nodes=scattered_nodes,
        rect_axes=[axes[n] for n in rect_names],
        values=values,
        names=list(scattered_names),
        rect_names=rect_names,
        clip_axes=clip_axes,
        **kw,
    )


def resolve_solar(element: str, spec: dict) -> float | None:
    """Resolve the solar A(X) a manifest asks for, if any."""
    if "solar" in spec:
        return float(spec["solar"])
    table = spec.get("solar_table")
    return solar_abundance(element, table) if table else None
