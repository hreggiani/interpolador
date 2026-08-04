"""Reader for grids published as trained neural networks rather than tables.

Canocchi et al. (2026) release their 3D non-LTE sodium grid as feedforward
networks that map ``(Teff, log g, [Fe/H], REW)`` to A(Na), one per line and per
model flavour, instead of a cube of corrections.  Getting a correction out of
them takes two steps, following the authors' own ``interp_tools.nlte_correction``:

1. Build the 1D LTE curve of growth for the star by asking the 1D LTE network
   for the abundance at a ladder of reduced equivalent widths, then invert it to
   turn the measured 1D LTE abundance into a REW.
2. Ask the 3D non-LTE network what abundance that REW implies.

The difference is the correction, which is therefore indexed by the *1D LTE*
abundance.

Networks are loaded lazily, per line and flavour, and cached: the full set is
240 MB, and a query touches two of the twenty files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..interp import InterpResult
from ..registry import register_reader
from ..solar import normalize_element
from ..source import GridSource, Line

#: Sampling of the curve of growth used to invert abundance -> REW, matching the
#: authors' 0.1 dex ladder.  Per-line limits come from the manifest.
REW_STEP = 0.1

_MODEL_CACHE: dict = {}


def _network_present(directory: Path) -> bool:
    """Whether a trained network and its scalers are on disk."""
    stem = directory / "Abna_noholdout_seed28"
    return ((directory / "ffnn.joblib").exists()
            and Path(f"{stem}_scalar_fullset.npy").exists()
            and Path(f"{stem}_y_scalar.npy").exists())


def _extract_weights(model):
    """Pull the layer weights out of a fitted network, if we can evaluate it.

    The released models are plain ReLU multilayer perceptrons with a linear
    output, so evaluating them is a few matrix products.  Doing that directly
    rather than through ``predict`` keeps results independent of the version of
    scikit-learn installed -- the files were pickled with 1.6.1, and unpickling
    them elsewhere warns that results may differ.  Verified to reproduce
    ``predict`` exactly; anything unfamiliar falls back to it.
    """
    if getattr(model, "activation", None) != "relu":
        return None
    if getattr(model, "out_activation_", None) != "identity":
        return None
    coefs = getattr(model, "coefs_", None)
    intercepts = getattr(model, "intercepts_", None)
    if not coefs or not intercepts:
        return None
    return [(np.asarray(w, dtype=float), np.asarray(b, dtype=float))
            for w, b in zip(coefs, intercepts)]


def _load_network(directory: Path):
    """Load a trained network and its input/output scalers, once per directory."""
    key = str(directory)
    if key not in _MODEL_CACHE:
        import warnings

        from joblib import load

        stem = directory / "Abna_noholdout_seed28"
        x_mean, x_std = np.load(f"{stem}_scalar_fullset.npy")
        y_mean, y_std = np.load(f"{stem}_y_scalar.npy")
        with warnings.catch_warnings():
            # The pickles carry their training version; we evaluate the weights
            # ourselves, so a version mismatch does not affect the result.
            warnings.simplefilter("ignore")
            model = load(directory / "ffnn.joblib")
        _MODEL_CACHE[key] = (model, _extract_weights(model), x_mean, x_std, y_mean, y_std)
    return _MODEL_CACHE[key]


def _forward(layers, x: np.ndarray) -> np.ndarray:
    """ReLU multilayer perceptron with a linear output."""
    h = x
    for weight, bias in layers[:-1]:
        h = np.maximum(h @ weight + bias, 0.0)
    weight, bias = layers[-1]
    return (h @ weight + bias).reshape(-1)


def _predict(directory: Path, columns: list) -> np.ndarray:
    """Run a network on stacked inputs, applying the published scalings."""
    model, layers, x_mean, x_std, y_mean, y_std = _load_network(directory)
    x = np.column_stack([np.asarray(c, dtype=float) for c in columns])
    scaled = (x - x_mean) / x_std
    raw = _forward(layers, scaled) if layers else np.asarray(model.predict(scaled)).reshape(-1)
    return raw * y_std + y_mean


class NetworkGrid:
    """A grid backed by trained networks, callable like an interpolator.

    Presents the same surface the tabulated interpolators do -- ``__call__``
    returning an :class:`~nlte.interp.InterpResult`, plus ``bounds()`` -- so the
    solvers, the query API and the CLI need no special case.
    """

    geometry_label = "neural network"
    values = None          # no tabulated nodes to report coverage over

    def __init__(self, root: Path, line_tag: str, axis_bounds: dict, rew_range,
                 target: str = "3DNLTE", reference: str = "1DLTE",
                 min_valid_weight: float = 0.5):
        self.root = Path(root)
        self.line_tag = line_tag
        self.axis_bounds = dict(axis_bounds)
        self.rew_range = tuple(rew_range)
        self.target = target
        self.reference = reference
        self.min_valid_weight = float(min_valid_weight)

    # -- the interpolator surface ----------------------------------------

    def bounds(self) -> dict:
        return dict(self.axis_bounds)

    @property
    def _target_dir(self) -> Path:
        return self.root / f"line{self.line_tag}" / self.target

    @property
    def _reference_dir(self) -> Path:
        return self.root / f"line{self.line_tag}" / self.reference

    def _curve_of_growth(self, teff, logg, feh, vmic):
        """1D LTE abundance at each rung of the REW ladder, per query point."""
        rews = np.arange(self.rew_range[0], self.rew_range[1], REW_STEP)
        n, r = len(teff), rews.size
        tiled = [np.repeat(teff, r), np.repeat(logg, r), np.repeat(feh, r),
                 np.repeat(vmic, r), np.tile(rews, n)]
        abunds = _predict(self._reference_dir, tiled).reshape(n, r)
        return rews, abunds

    def __call__(self, points) -> InterpResult:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        teff, logg, feh, vmic, abund = (pts[:, i] for i in range(5))

        in_range = np.ones(pts.shape[0], dtype=bool)
        for i, name in enumerate(("teff", "logg", "feh", "vmic", "abund")):
            lo, hi = self.axis_bounds[name]
            in_range &= (pts[:, i] >= lo - 1e-9) & (pts[:, i] <= hi + 1e-9)

        value = np.full(pts.shape[0], np.nan)
        weight = np.zeros(pts.shape[0])
        if not in_range.any():
            return InterpResult(value=value, in_range=in_range, valid_weight=weight)

        idx = np.flatnonzero(in_range)
        rews, abunds = self._curve_of_growth(teff[idx], logg[idx], feh[idx], vmic[idx])

        # Invert the curve of growth point by point.  A cubic spline, as in the
        # authors' routine: on their 0.1 dex ladder, linear interpolation here
        # shifts the correction by up to 0.01 dex.  A measurement off the end of
        # the curve is not extrapolated.
        from scipy.interpolate import CubicSpline

        rew = np.full(idx.size, np.nan)
        for k in range(idx.size):
            curve = abunds[k]
            order = np.argsort(curve)
            ordered_abund, ordered_rew = curve[order], rews[order]
            keep = np.concatenate([[True], np.diff(ordered_abund) > 0])
            ordered_abund, ordered_rew = ordered_abund[keep], ordered_rew[keep]
            if ordered_abund.size < 4:
                continue
            if ordered_abund[0] - 1e-9 <= abund[idx[k]] <= ordered_abund[-1] + 1e-9:
                rew[k] = float(CubicSpline(ordered_abund, ordered_rew)(abund[idx[k]]))

        usable = np.isfinite(rew)
        if usable.any():
            sel = idx[usable]
            predicted = _predict(
                self._target_dir,
                [teff[sel], logg[sel], feh[sel], rew[usable]],
            )
            value[sel] = predicted - abund[sel]
            weight[sel] = 1.0

        # A point inside the axes whose abundance misses the curve of growth is
        # genuinely uncovered, not out of range.
        return InterpResult(value=value, in_range=in_range, valid_weight=weight)


@register_reader("canocchi_ffnn")
def read_canocchi_ffnn(manifest: dict, path: Path) -> list:
    """Build sources from a directory of trained networks."""
    element = normalize_element(manifest["element"])
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"{manifest['id']}: network directory {root} not found")

    axis_bounds = {
        name: tuple(float(v) for v in manifest["axes"][name])
        for name in ("teff", "logg", "feh", "vmic", "abund")
    }
    entries = manifest["lines"]
    variants = manifest.get("variants") or [{"target": "3DNLTE", "flavor": "3D NLTE - 1D LTE"}]

    sources = []
    for variant in variants:
        target = variant.get("target", "3DNLTE")
        reference_model = variant.get("reference", manifest.get("reference_model", "1DLTE"))
        lines, grids, skipped = [], [], []
        for entry in entries:
            tag = str(entry["tag"])
            # Not every line was trained for every flavour -- 5688 has no 3D LTE
            # network -- so drop those here rather than failing at query time.
            if not all(_network_present(root / f"line{tag}" / model)
                       for model in (target, reference_model)):
                skipped.append(tag)
                continue
            lines.append(Line(wavelength=float(entry["wavelength"]), label=tag))
            grids.append(NetworkGrid(
                root=root,
                line_tag=tag,
                axis_bounds=axis_bounds,
                rew_range=entry["rew_range"],
                target=target,
                reference=reference_model,
                min_valid_weight=manifest.get("min_valid_weight", 0.5),
            ))
        if not grids:
            continue
        suffix = variant.get("id_suffix", "")
        sources.append(GridSource(
            id=manifest["id"] + (f"_{suffix}" if suffix else ""),
            reference=manifest.get("reference", ""),
            element=element,
            ion=int(manifest.get("ion", 1)),
            flavor=variant.get("flavor", "NLTE - LTE"),
            kind="delta",
            axis_names=("teff", "logg", "feh", "vmic", "abund"),
            lines=lines,
            grids=grids,
            abund_kind="logeps",
            indexed_by="lte",
            notes=" ".join(filter(None, [
                manifest.get("notes", ""),
                f"No network for line(s) {', '.join(skipped)} in this flavour."
                if skipped else "",
            ])),
        ))
    return sources
