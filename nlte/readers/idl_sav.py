"""Readers for the IDL ``.sav`` grids.

Three schemas live in these files:

``amarsi_sav``
    ``abcor_*.sav`` -- an ``ABCOR`` cube indexed by
    ``(line, vturb_top, vturb_base, feh, logg, teff, lgeps)``.  Two
    microturbulence axes appear because the non-LTE and LTE sides may use
    different model atmospheres; both read the caller's single ``vmic``.

``oxygen_sav``
    ``oxygen_marcs.sav`` -- a different field layout, with an [O/Fe] abundance
    axis and a single microturbulence axis.

``nordlander_sav``
    The Nordlander & Lind aluminium grids, which publish LTE and non-LTE
    ``LOGEW`` cubes rather than corrections.

IDL stores arrays in column-major order, so ``scipy.io.readsav`` hands back the
axes reversed relative to the IDL declaration.  Each reader asserts the expected
shape against the ``N*`` counters in the struct before transposing, because a
silent axis swap here would poison every result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import readsav

from ..registry import register_reader
from ..solar import normalize_element
from ..source import GridSource, Line, make_grid

#: Order the interpolators expect once the line axis has been peeled off.
CANONICAL_ORDER = ("teff", "logg", "feh", "vmic", "vmic_lte", "abund")

SENTINEL = -1e28   # grids use -1e29/-1e30; anything below this is "no value"


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _axis(value) -> np.ndarray:
    """A grid axis as a 1-D array, tolerating IDL scalars."""
    return np.atleast_1d(np.asarray(value, dtype=float))


def _decode(values) -> list:
    out = []
    for v in np.atleast_1d(np.asarray(values)):
        out.append(v.decode() if isinstance(v, (bytes, bytearray)) else str(v))
    return out


def _mask_sentinels(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float)
    values[~np.isfinite(values) | (values <= SENTINEL)] = np.nan
    return values


def _arrange(values: np.ndarray, source_axes: list):
    """Squeeze degenerate axes out and transpose the rest into canonical order.

    ``source_axes`` is ``[(name, nodes), ...]`` in the order of ``values``'s
    dimensions.  Returns ``(axes, values, fixed)``.
    """
    fixed, keep = {}, []
    for i, (name, nodes) in enumerate(source_axes):
        if nodes.size == 1:
            fixed[name] = float(nodes[0])
        else:
            keep.append((i, name, nodes))
    values = values.reshape([n.size for _, n in source_axes])
    values = values[tuple(slice(None) if any(i == k for k, _, _ in keep) else 0
                          for i in range(len(source_axes)))]

    names = [name for _, name, _ in keep]
    order = sorted(range(len(names)), key=lambda j: CANONICAL_ORDER.index(names[j]))
    values = np.transpose(values, order)
    axes = {names[j]: keep[j][2] for j in order}
    return axes, values, fixed


def _lines_from_labels(labels, wavelengths=None) -> list:
    lines = []
    for i, label in enumerate(labels):
        if wavelengths is not None:
            wl = float(wavelengths[i])
        else:
            text = "".join(c for c in label if c.isdigit() or c == ".")
            wl = float(text) * 10.0 if "nm" in label.lower() else float(text)
        lines.append(Line(wavelength=wl, label=str(label)))
    return lines


def _oxygen_lines(labels, lmin, lmax) -> list:
    """Line list for ``oxygen_marcs.sav``, resolving its blended entry.

    Entries are labelled in nm ("777.2"), and one of them ("777 all") covers the
    whole triplet.  Its integration window contains the windows of the resolved
    components, so the blend is detected from the data and made reachable by a
    query at any component's wavelength.
    """
    centres = []
    for label in labels:
        digits = "".join(c for c in label if c.isdigit() or c == ".").strip(".")
        centres.append(float(digits) * 10.0)

    lines = []
    for i, label in enumerate(labels):
        inside = [
            j for j in range(len(labels))
            if j != i and lmin[i] <= lmin[j] and lmax[j] <= lmax[i]
        ]
        match = tuple(centres[j] for j in inside) if inside else (centres[i],)
        lines.append(Line(
            wavelength=centres[i], label=str(label), blend=bool(inside), match=match,
        ))
    return lines


def _abundance_kwargs(manifest: dict, element: str, grid_solar: float | None) -> dict:
    ab = manifest.get("abundance", {})
    kind = ab.get("kind", "logeps")
    solar = ab.get("solar", grid_solar)
    return {
        "abund_kind": kind,
        "abund_solar": float(solar) if kind == "xfe" and solar is not None else None,
        "abund_zero_point": (
            float(ab["zero_point"]) if kind == "offset" and "zero_point" in ab
            else (float(solar) if kind == "offset" and solar is not None else None)
        ),
        "solar_table": ab.get("solar_table"),
    }


def _base_kwargs(manifest: dict, element: str) -> dict:
    return {
        "reference": manifest.get("reference", ""),
        "element": element,
        "ion": int(manifest.get("ion", 1)),
        "flavor": manifest.get("flavor", "NLTE - LTE"),
        "scattering": manifest.get("scattering"),
        "notes": manifest.get("notes", ""),
    }


# ------------------------------------------------------------ abcor_*.sav


@register_reader("amarsi_sav")
def read_amarsi_sav(manifest: dict, path: Path) -> list:
    """``abcor_*.sav``: a struct named ``grid`` holding an ``ABCOR`` cube."""
    element = normalize_element(manifest["element"])
    data = readsav(str(path), verbose=False)
    g = np.asarray(data[manifest.get("struct", "grid")]).reshape(-1)[0]

    teff, logg, feh = _axis(g["TEFF"]), _axis(g["LGG"]), _axis(g["FEH"])
    vt_base, vt_top = _axis(g["VTURB_BASE"]), _axis(g["VTURB_TOP"])
    lgeps = _axis(g["LGEPS"])
    labels = _decode(g["LINE"])
    cube = _mask_sentinels(np.asarray(g["ABCOR"]))

    expected = (len(labels), vt_top.size, vt_base.size, feh.size,
                logg.size, teff.size, lgeps.size)
    if cube.shape != expected:
        raise ValueError(f"{path.name}: ABCOR shape {cube.shape} != expected {expected}")

    source_axes = [
        ("vmic_lte", vt_top), ("vmic", vt_base), ("feh", feh),
        ("logg", logg), ("teff", teff), ("abund", lgeps),
    ]
    clip = tuple(manifest.get("clip_axes", ()))
    grids, axes, fixed = [], None, {}
    for i in range(len(labels)):
        axes, values, fixed = _arrange(cube[i], source_axes)
        grids.append(make_grid(axes, values, clip_axes=clip,
                               min_valid_weight=manifest.get("min_valid_weight")))

    return [GridSource(
        id=manifest["id"],
        kind="delta",
        axis_names=tuple(axes),
        lines=_lines_from_labels(labels),
        grids=grids,
        fixed=fixed,
        alias={"vmic_lte": "vmic"},
        **_base_kwargs(manifest, element),
        **_abundance_kwargs(manifest, element, _scalar(g["SOLAR"])),
    )]


# ------------------------------------------------------- oxygen_marcs.sav


@register_reader("oxygen_sav")
def read_oxygen_sav(manifest: dict, path: Path) -> list:
    """``oxygen_marcs.sav``: ABCOR indexed by (line, ofe, vturb, feh, logg, teff)."""
    element = normalize_element(manifest["element"])
    data = readsav(str(path), verbose=False)
    g = np.asarray(data[manifest.get("struct", "grid")]).reshape(-1)[0]

    teff, logg, feh = _axis(g["TEFF"]), _axis(g["LGG"]), _axis(g["FEH"])
    vturb, ofe = _axis(g["VTURB"]), _axis(g["OFE"])
    labels = _decode(g["LINE"])
    cube = _mask_sentinels(np.asarray(g["ABCOR"]))

    expected = (len(labels), ofe.size, vturb.size, feh.size, logg.size, teff.size)
    if cube.shape != expected:
        raise ValueError(f"{path.name}: ABCOR shape {cube.shape} != expected {expected}")

    source_axes = [
        ("abund", ofe), ("vmic", vturb), ("feh", feh), ("logg", logg), ("teff", teff),
    ]
    clip = tuple(manifest.get("clip_axes", ()))
    grids, axes, fixed = [], None, {}
    for i in range(len(labels)):
        axes, values, fixed = _arrange(cube[i], source_axes)
        grids.append(make_grid(axes, values, clip_axes=clip,
                               min_valid_weight=manifest.get("min_valid_weight")))

    lines = _oxygen_lines(labels, _axis(g["LMIN"]), _axis(g["LMAX"]))
    return [GridSource(
        id=manifest["id"],
        kind="delta",
        axis_names=tuple(axes),
        lines=lines,
        grids=grids,
        fixed=fixed,
        **_base_kwargs(manifest, element),
        **_abundance_kwargs(manifest, element, _scalar(g["LGEPS_SOLAR"])),
    )]


# --------------------------------------------------- Nordlander & Lind Al


@register_reader("nordlander_sav")
def read_nordlander_sav(manifest: dict, path: Path) -> list:
    """Aluminium grids storing paired LTE / non-LTE ``LOGEW`` cubes."""
    element = normalize_element(manifest["element"])
    data = readsav(str(path), verbose=False)
    lte = np.asarray(data["ltegrid"]).reshape(-1)[0]
    nlte = np.asarray(data["nltegrid"]).reshape(-1)[0]

    teff, logg, feh = _axis(nlte["TEFF"]), _axis(nlte["LOGG"]), _axis(nlte["FEH"])
    abund = _axis(nlte["ABUND"])
    waves = np.asarray(nlte["WAVELENGTH"], dtype=float)

    cubes = {}
    for tag, struct in (("lte", lte), ("nlte", nlte)):
        cube = np.array(np.asarray(struct["LOGEW"]), dtype=float)
        expected = (waves.size, feh.size, logg.size, teff.size, abund.size)
        if cube.shape != expected:
            raise ValueError(f"{path.name}: {tag} LOGEW shape {cube.shape} != {expected}")
        cubes[tag] = cube

    source_axes = [("feh", feh), ("logg", logg), ("teff", teff), ("abund", abund)]
    clip = tuple(manifest.get("clip_axes", ()))
    grids, axes = [], None
    for i in range(waves.size):
        pair = []
        for tag in ("lte", "nlte"):
            axes, values, fixed = _arrange(cubes[tag][i], source_axes)
            pair.append(make_grid(axes, values, clip_axes=clip,
                                  min_valid_weight=manifest.get("min_valid_weight")))
        grids.append(tuple(pair))

    return [GridSource(
        id=manifest["id"],
        kind="ew_pair",
        axis_names=tuple(axes),
        lines=[Line(wavelength=float(w), label=f"{w:.3f}") for w in waves],
        grids=grids,
        abund_nodes=abund,
        **_base_kwargs(manifest, element),
        **_abundance_kwargs(manifest, element, None),
    )]
