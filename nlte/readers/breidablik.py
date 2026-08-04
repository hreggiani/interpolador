"""Reader for the Breidablik 3D non-LTE lithium models (Wang et al. 2021, 2024).

Breidablik ships trained networks rather than a table, and exposes them through
its own package (``pip install breidablik``; the models come with the wheel).
Its ``Nlte.nlte_correction`` already returns exactly what this project wants --
a correction to a 1D LTE abundance -- so the wrapper is thin: it evaluates the
same networks over whole arrays at once, converts the vacuum line centres to
air, and turns Breidablik's out-of-grid warning into an ``in_range`` flag.

Coverage comes from the Stagger grid the models were trained on: Teff 3940-7050,
log g 1.5-5.0, [Fe/H] -2.0 to +0.25, A(Li) -0.5 to 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..interp import InterpResult
from ..registry import register_reader
from ..solar import normalize_element
from ..source import GridSource, Line
from ..wavelengths import vacuum_to_air

#: Line centres Breidablik models, in nm and in vacuum.
LINE_CENTRES_NM_VAC = (610.5298, 670.9659, 812.8606)

#: Breidablik's own bound on the abundance the networks were trained over.
ABUND_RANGE = (-0.5, 4.0)

_MODELS = {}


def _load_breidablik():
    """Load the Breidablik model set once per process."""
    if "nlte" not in _MODELS:
        try:
            from breidablik.interpolate.nlte import Nlte
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise FileNotFoundError(
                "the 'breidablik' package is required for this source "
                "(pip install breidablik)"
            ) from exc
        _MODELS["nlte"] = Nlte()
    return _MODELS["nlte"]


def _load_grid_snapshot():
    """The Stagger nodes and scalings Breidablik uses to judge coverage."""
    if "grid" not in _MODELS:
        import breidablik

        path = Path(breidablik.__file__).parent / "grid_snapshot.txt"
        with open(path) as fh:
            t_step, m_step = np.float64(fh.readline().split())
            nodes = np.loadtxt(fh)
        nodes[:, 0] = np.round(nodes[:, 0] * 2) / 2
        _MODELS["grid"] = (nodes, float(t_step), float(m_step))
    return _MODELS["grid"]


#: Breidablik flags a query as extrapolated beyond this distance from any node,
#: in its normalised parameter space.
_MAX_NODE_DISTANCE = np.sqrt(3 * 0.25 ** 2)


def _within_grid(teff, logg, feh) -> np.ndarray:
    """Breidablik's own coverage test, as a boolean instead of a warning."""
    nodes, t_step, m_step = _load_grid_snapshot()
    query = np.column_stack([
        np.asarray(teff, dtype=float) * t_step,
        np.asarray(logg, dtype=float),
        np.asarray(feh, dtype=float) * m_step,
    ])
    distance = np.sqrt(((query[:, None, :] - nodes[None, :, :]) ** 2).sum(axis=2))
    return distance.min(axis=1) <= _MAX_NODE_DISTANCE


class BreidablikGrid:
    """One Breidablik line, presented as an interpolator.

    Evaluates the network over all query points at once.  Breidablik's public
    ``nlte_correction`` takes scalars only; the underlying model is vectorised,
    so this calls it with a stacked array and is checked against the public
    entry point in the tests.
    """

    geometry_label = "neural network (Breidablik)"
    values = None

    def __init__(self, centre_nm_vac: float, axis_bounds: dict,
                 min_valid_weight: float = 0.5):
        self.centre_nm_vac = float(centre_nm_vac)
        self.axis_bounds = dict(axis_bounds)
        self.min_valid_weight = float(min_valid_weight)
        self._index = int(np.argmin(np.abs(np.array(LINE_CENTRES_NM_VAC) - centre_nm_vac)))

    def bounds(self) -> dict:
        return dict(self.axis_bounds)

    def __call__(self, points) -> InterpResult:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        teff, logg, feh, abund = (pts[:, i] for i in range(4))

        lo, hi = self.axis_bounds["abund"]
        in_range = _within_grid(teff, logg, feh) & (abund >= lo - 1e-9) & (abund <= hi + 1e-9)

        value = np.full(pts.shape[0], np.nan)
        weight = np.zeros(pts.shape[0])
        if in_range.any():
            model = _load_breidablik()
            idx = np.flatnonzero(in_range)
            scaled = model.scalars[self._index].transform(
                np.column_stack([teff[idx], logg[idx], feh[idx], abund[idx]])
            )
            value[idx] = np.asarray(model.models[self._index](scaled)).reshape(-1)
            weight[idx] = 1.0
        return InterpResult(value=value, in_range=in_range, valid_weight=weight)


@register_reader("breidablik_nlte")
def read_breidablik(manifest: dict, path: Path) -> list:
    """Build the lithium source from the installed Breidablik models."""
    element = normalize_element(manifest["element"])
    _load_breidablik()          # fail early, with a clear message, if absent

    axis_bounds = {
        name: tuple(float(v) for v in manifest["axes"][name])
        for name in ("teff", "logg", "feh")
    }
    axis_bounds["abund"] = tuple(manifest["axes"].get("abund", ABUND_RANGE))

    lines, grids = [], []
    for centre in LINE_CENTRES_NM_VAC:
        air = float(vacuum_to_air(centre * 10.0))
        lines.append(Line(wavelength=round(air, 3), label=f"{centre:g}nm(vac)"))
        grids.append(BreidablikGrid(centre, axis_bounds,
                                    manifest.get("min_valid_weight", 0.5)))

    return [GridSource(
        id=manifest["id"],
        reference=manifest.get("reference", ""),
        element=element,
        ion=int(manifest.get("ion", 1)),
        flavor=manifest.get("flavor", "3D NLTE - 1D LTE"),
        kind="delta",
        axis_names=("teff", "logg", "feh", "abund"),
        lines=lines,
        grids=grids,
        abund_kind="logeps",
        indexed_by="lte",
        notes=manifest.get("notes", ""),
    )]
