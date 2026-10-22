"""Reader for the Amarsi et al. 2022 3D non-LTE iron networks.

Upstream (https://github.com/sliljegren/1L-3NErrors, MIT) publishes three
trained networks rather than a table: Fe I below 2 eV, Fe I at or above 2 eV,
and Fe II.  Unlike every other source here they take the *line's* atomic data as
inputs -- lower and upper excitation potential and log gf -- alongside the
stellar parameters, so there is no line list inside the models.  A vendored line
list supplies that per line, which keeps queries looking like every other
source: ``correct("FeI", 5232.94, ...)``.

Two conventions matter, and both are checked in the tests:

* Upstream returns ``aberr`` = A(1D LTE) - A(3D NLTE), the opposite sign to this
  project's ``delta`` = A(NLTE) - A(LTE).  The values are negated here.
* The corrections are functions of the 3D non-LTE abundance, so the correction
  for a measured LTE abundance has to be solved for.  That is this project's
  default (``indexed_by = "nlte"``), and it is what upstream's README asks the
  user to do by hand.

See grids/amarsi2022_fe/PROVENANCE.md for how the vendored files were produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..interp import InterpResult
from ..registry import register_reader
from ..solar import normalize_element
from ..source import GridSource, Line

#: Lower excitation potential, in eV, splitting the two Fe I networks.  Upstream
#: branches on `< 2.0` and `> 2.0`, which leaves a line at exactly 2 eV with no
#: model and a silent -999; `>=` here closes that gap.
FE1_ELO_SPLIT = 2.0

#: Feature order the networks were trained on.
FEATURES = ("teff", "logg", "abund", "vmic", "elo", "eup", "loggf")

_CACHE: dict = {}


def _load_network(path: Path):
    """Layer weights and input scaling for one vendored network."""
    key = str(path)
    if key not in _CACHE:
        with np.load(path, allow_pickle=True) as z:
            n = int(z["n_layers"])
            layers = [(z[f"weight_{i}"], z[f"bias_{i}"]) for i in range(n)]
            _CACHE[key] = (layers, z["scaler_mean"], z["scaler_scale"])
    return _CACHE[key]


def _predict(path: Path, features: np.ndarray) -> np.ndarray:
    """Evaluate a ReLU network with a linear output, as scikit-learn would.

    Reproduces ``MLPRegressor.predict`` on the scaled inputs bit-for-bit; see
    PROVENANCE.md for why the weights are evaluated directly.
    """
    layers, mean, scale = _load_network(path)
    h = (features - mean) / scale
    for weight, bias in layers[:-1]:
        h = np.maximum(h @ weight + bias, 0.0)
    weight, bias = layers[-1]
    return (h @ weight + bias).reshape(-1)


class IronNetworkGrid:
    """One Fe line, presented as an interpolator.

    Exposes the same surface the tabulated interpolators do -- ``__call__``
    returning an :class:`~nlte.interp.InterpResult`, and ``bounds()`` -- so the
    solvers, the query API and the CLI need no special case.
    """

    geometry_label = "neural network (Amarsi 2022)"
    values = None               # no tabulated nodes to report coverage over

    def __init__(self, model_path: Path, atomic: dict, axis_bounds: dict,
                 extrapolate: bool = False, min_valid_weight: float = 0.5):
        self.model_path = Path(model_path)
        self.atomic = dict(atomic)
        self.axis_bounds = dict(axis_bounds)
        self.extrapolate = bool(extrapolate)
        self.min_valid_weight = float(min_valid_weight)

    def bounds(self) -> dict:
        return dict(self.axis_bounds)

    def __call__(self, points) -> InterpResult:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        teff, logg, vmic, abund = (pts[:, i] for i in range(4))

        in_range = np.ones(pts.shape[0], dtype=bool)
        for i, name in enumerate(("teff", "logg", "vmic", "abund")):
            lo, hi = self.axis_bounds[name]
            in_range &= (pts[:, i] >= lo - 1e-9) & (pts[:, i] <= hi + 1e-9)

        value = np.full(pts.shape[0], np.nan)
        weight = np.zeros(pts.shape[0])
        # Upstream can extrapolate on request, but its authors explicitly
        # decline to vouch for those numbers, so it is opt-in per source.
        usable = np.ones_like(in_range) if self.extrapolate else in_range
        if usable.any():
            idx = np.flatnonzero(usable)
            n = idx.size
            features = np.column_stack([
                teff[idx], logg[idx], abund[idx], vmic[idx],
                np.full(n, self.atomic["elo"]),
                np.full(n, self.atomic["eup"]),
                np.full(n, self.atomic["loggf"]),
            ])
            # Negated: upstream reports A(1D LTE) - A(3D NLTE).
            value[idx] = -_predict(self.model_path, features)
            weight[idx] = 1.0
        return InterpResult(value=value, in_range=in_range, valid_weight=weight)


def _read_line_list(path: Path):
    """The vendored Fe line list, as ``{species: [(wavelength, atomic), ...]}``."""
    import pandas as pd

    table = pd.read_csv(path)
    missing = {"species", "wavelength", "elo", "eup", "loggf"} - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: line list is missing columns {sorted(missing)}")
    grouped = {}
    for row in table.itertuples():
        grouped.setdefault(str(row.species), []).append(
            (float(row.wavelength),
             {"elo": float(row.elo), "eup": float(row.eup), "loggf": float(row.loggf)})
        )
    return grouped


@register_reader("amarsi2022_fe_ffnn")
def read_amarsi2022_fe(manifest: dict, path: Path) -> list:
    """Build one source per ionisation stage from the vendored networks."""
    element = normalize_element(manifest["element"])
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"{manifest['id']}: {root} not found")

    axis_bounds = {name: tuple(float(v) for v in manifest["axes"][name])
                   for name in ("teff", "logg", "vmic", "abund")}
    lines_by_species = _read_line_list(root / manifest.get("line_list", "fe_lines.csv"))
    extrapolate = bool(manifest.get("extrapolate", False))
    weight = manifest.get("min_valid_weight", 0.5)

    sources = []
    for entry in manifest["species"]:
        tag, ion = str(entry["tag"]), int(entry["ion"])
        missing = [m for m in entry["models"].values() if not (root / m).exists()]
        if missing:
            raise FileNotFoundError(f"{manifest['id']}: missing network(s) {missing}")

        lines, grids = [], []
        for wavelength, atomic in lines_by_species.get(tag, []):
            if len(entry["models"]) == 1:
                model = next(iter(entry["models"].values()))
            else:
                # Fe I is split by lower excitation potential.
                model = entry["models"]["lt" if atomic["elo"] < FE1_ELO_SPLIT else "ge"]
            lines.append(Line(wavelength=wavelength, label=f"{wavelength:.3f}",
                              atomic=atomic))
            grids.append(IronNetworkGrid(root / model, atomic, axis_bounds,
                                         extrapolate=extrapolate, min_valid_weight=weight))
        if not grids:
            continue
        sources.append(GridSource(
            id=f"{manifest['id']}_{tag.lower()}",
            reference=manifest.get("reference", ""),
            element=element,
            ion=ion,
            flavor=manifest.get("flavor", "3D NLTE - 1D LTE"),
            kind="delta",
            axis_names=("teff", "logg", "vmic", "abund"),
            lines=lines,
            grids=grids,
            abund_kind="logeps",
            indexed_by="nlte",
            notes=manifest.get("notes", ""),
        ))
    return sources
