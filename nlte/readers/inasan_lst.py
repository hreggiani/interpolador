"""Reader for the INASAN ``.lst`` correction tables (Sitnova et al. 2022, zinc).

The Moscow group distributes its grids as plain text in blocks, one block per
line and effective temperature::

     Zn lines, NLTE EW(mA) and NLTE abundance corrections
      481.0528 nm  Zn1  Eexc =  4.080  log gf =    -0.150
     Teff= 4000   0.5     1.0     1.5  ...          <- log g nodes
        -2.0          52      49      45  ...       <- [Fe/H], then EW (mA) per log g
        -2.0       0.090   0.110   0.130  ...       <- [Fe/H], then Delta per log g

so each [Fe/H] row appears twice, equivalent widths first.  ``EW = -1`` (with
``Delta = -1.000``) marks a line too weak to measure, below 5 mA; those nodes
are masked.  The corrections are tabulated at a fixed [Zn/Fe] = 0, so the
source is abundance-independent.

Parsing produces a long table that goes through the tabular reader's gridding,
so axis handling -- masked nodes, collapsed axes, the flat abundance axis -- is
shared rather than reimplemented.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..registry import register_reader
from ..solar import normalize_element
from ..source import GridSource, make_grid
from .tabular import _build_arrays, _independent_kwargs, _ion_number, _make_lines, _shape_axes

_LINE = re.compile(r"^\s*(?P<wl>\d+(?:\.\d+)?)\s*nm\s+(?P<species>[A-Z][a-z]?\d)\s+Eexc")
_TEFF = re.compile(r"^\s*Teff=\s*(?P<teff>\d+(?:\.\d+)?)\s+(?P<logg>.*)$")


def parse_lst(path: Path) -> pd.DataFrame:
    """Long table ``(line, species, teff, logg, feh, ew, delta)`` from an ``.lst`` file."""
    rows = []
    line = species = teff = logg = None
    seen = {}                    # [Fe/H] row label -> its equivalent widths, awaiting Delta
    for raw in Path(path).read_text().splitlines():
        if m := _LINE.match(raw):
            line, species = f"{m.group('wl')}nm", m.group("species")
            teff, logg = None, None
            continue
        if m := _TEFF.match(raw):
            teff = float(m.group("teff"))
            logg = [float(x) for x in m.group("logg").split()]
            seen = {}
            continue
        fields = raw.split()
        if teff is None or not fields:
            continue
        feh, values = fields[0], [float(x) for x in fields[1:]]
        if len(values) != len(logg):
            raise ValueError(f"{path}: row {raw!r} has {len(values)} values for {len(logg)} log g")
        if feh not in seen:
            seen[feh] = values
            continue
        ews = seen.pop(feh)
        for g, ew, d in zip(logg, ews, values):
            rows.append((line, species, teff, g, float(feh), ew, d))
    df = pd.DataFrame(rows, columns=["line", "species", "teff", "logg", "feh", "ew", "delta"])
    df.loc[df["ew"] < 0, ["ew", "delta"]] = float("nan")
    return df


@register_reader("inasan_lst")
def read_inasan_lst(manifest: dict, path: Path) -> list:
    """One abundance-independent source per ionisation stage in the file."""
    element = normalize_element(manifest["element"])
    df = parse_lst(path)
    layout = {**manifest, "columns": {"teff": "teff", "logg": "logg", "feh": "feh",
                                      "line": "line"},
              "abundance": {**manifest.get("abundance", {}), "independent": True}}
    clip = tuple(manifest.get("clip_axes", ())) + ("abund",)
    sources = []
    for species, sub in df.groupby("species"):
        ion = _ion_number(species)
        nodes, scattered, scat_nodes, labels, arrays = _build_arrays(
            sub.reset_index(drop=True), layout, {"delta": "delta"}
        )
        nodes, arrays, fixed, _ = _shape_axes(nodes, scattered, arrays, layout)
        grids = [
            make_grid(nodes, arrays["delta"][i], clip_axes=clip,
                      min_valid_weight=manifest.get("min_valid_weight"))
            for i in range(len(labels))
        ]
        sources.append(GridSource(
            id=f"{manifest['id']}_{element.lower()}{ion}",
            ion=ion,
            flavor=manifest.get("flavor", "1D NLTE - 1D LTE"),
            kind="delta",
            axis_names=tuple(nodes),
            lines=_make_lines(labels, manifest.get("wavelengths", {})),
            grids=grids,
            fixed=fixed,
            **_independent_kwargs(manifest, element, True),
        ))
    return sources
