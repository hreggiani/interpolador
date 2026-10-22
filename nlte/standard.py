"""The standard output: a short, fixed set of columns from a chosen few grids.

The full output of :func:`nlte.correct_frame` answers with every grid covering a
line, with all the diagnostics needed to see why one failed.  The standard
output is what goes into an abundance table instead: one row per measurement
and recommended source, ten columns, nothing else.  Where a recommended source
has no usable value the row stays, with NaN, so the full output can be checked
for why.

Columns:

``id``          the star, from the input's ``id`` (or ``star``) column
``wavelength``  as measured, in Angstrom
``species``     numeric code, atomic number + (stage - 1) / 10: Fe II = 26.1
``ep``          lower excitation potential, eV
``gf``          log gf
``A(X)``        the measured 1D LTE abundance
``A(X)_corr``   the corrected abundance
``DeltaA(X)``   A(X)_corr - A(X), the same sign convention as ``delta``
``source``      the paper, e.g. "Amarsi et al. 2019"
``flavor``      what the correction is, e.g. "3D NLTE - 1D LTE (ATLAS)"

``ep`` and ``gf`` come from the input when it has them (``ep``/``elo`` and
``gf``/``loggf`` columns), else from the line list of a grid that carries atomic
data -- today only the Amarsi 2022 iron networks -- else NaN.
"""

from __future__ import annotations

import numpy as np

from .api import canonical_columns, sources
from .solar import ATOMIC_NUMBER, format_species, parse_species

#: Recommended sources per species, in output order.  Each entry is one output
#: row; an entry naming several ids is one paper split across tables that cover
#: different stars (Sitnova's MARCS and hot-star LLmodels grids), and the first
#: with a usable value is reported.
STANDARD_SOURCES = {
    "LiI": [("wang2021_li",)],
    "CI": [("amarsi_c1_3d",), ("amarsi_c1_1d",)],
    "OI": [("amarsi_o1_3d",), ("amarsi_o1_1d",)],
    "NaI": [("canocchi2026_na_3dnlte",), ("lind2022_na1",)],
    "SiI": [("amarsi2017_si",)],
    "KI": [("reggiani2019_k",)],
    "TiII": [("sitnova2016_ti2",)],
    "FeI": [("amarsi2022_fe_fe1",), ("amarsi2016_fe_nmarcs_lmarcs_fe1",)],
    "FeII": [("amarsi2022_fe_fe2",), ("amarsi2016_fe_nmarcs_lmarcs_fe2",)],
    "CuI": [("caliskan2025_cu",)],
    "ZnI": [("sitnova2022_zn_marcs_zn1", "sitnova2022_zn_llhot_zn1")],
    "ZnII": [("sitnova2022_zn_marcs_zn2", "sitnova2022_zn_llhot_zn2")],
    "SrII": [("mashonkina2022_sr",)],
    "BaII": [("mashonkina2019_ba",), ("gallagher2020_ba_3dnlte",), ("gallagher2020_ba_1dnlte",)],
}

STANDARD_COLUMNS = ("id", "wavelength", "species", "ep", "gf",
                    "A(X)", "A(X)_corr", "DeltaA(X)", "source", "flavor")

_ID_COLUMNS = ("id", "star")
_EP_COLUMNS = ("ep", "elo")
_GF_COLUMNS = ("gf", "loggf")


def species_code(element, ion) -> float:
    """Numeric species, atomic number + (stage - 1) / 10: ``("Fe", 2)`` -> 26.1."""
    if element is None or ion is None:
        return float("nan")
    return round(ATOMIC_NUMBER[element] + (int(ion) - 1) / 10, 1)


def paper_label(reference: str) -> str:
    """The citation without the journal: "Amarsi et al. 2019, A&A ..." -> "Amarsi et al. 2019"."""
    return reference.split(",")[0].strip()


def _first(record, names):
    for name in names:
        if name in record.index:
            value = record[name]
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return value
    return float("nan")


class _Lookup:
    """Sources by id for one species, loaded once from the registry."""

    def __init__(self, registry):
        self.registry = registry
        self._by_species = {}

    def source(self, element, ion, source_id):
        key = (element, ion)
        if key not in self._by_species:
            self._by_species[key] = {s.id: s for s in sources(element, ion, self.registry)}
        return self._by_species[key].get(source_id)

    def atomic(self, element, ion, source_id, wavelength_grid):
        src = self.source(element, ion, source_id)
        if src is None:
            return None
        for line in src.lines:
            if line.atomic and abs(line.wavelength - wavelength_grid) < 1e-6:
                return line.atomic
        return None


def standard_frame(full, frame, columns=None, registry=None):
    """Reduce the full output of :func:`nlte.correct_frame` to the standard columns.

    ``frame`` and ``columns`` are what ``correct_frame`` was given; ``full`` is
    what it returned.  Every input row appears at least once: once per
    recommended source for its species, or once with NaN when the species has
    none.  Rows follow the input order.
    """
    import pandas as pd

    df = canonical_columns(frame, columns)
    lookup = _Lookup(registry)
    empty = full.iloc[0:0]
    by_row = {r: block for r, block in full.groupby("row", sort=False)}

    rows = []
    for idx in df.index:
        record = df.loc[idx]
        matched = by_row.get(idx, empty)
        matched = matched[matched.source != ""]
        try:
            element, ion = parse_species(record["element"])
        except KeyError:
            element, ion = None, None
        # A bare label ("C") takes its stage from the grids it matched.
        stages = [ion] if ion is not None else sorted(int(i) for i in matched.ion.unique())

        ep, gf = _first(record, _EP_COLUMNS), _first(record, _GF_COLUMNS)
        if np.isnan(ep) or np.isnan(gf):
            for hit in matched.itertuples():
                atomic = lookup.atomic(hit.element, hit.ion, hit.source, hit.wavelength_grid)
                if atomic:
                    ep = atomic["elo"] if np.isnan(ep) else ep
                    gf = atomic["loggf"] if np.isnan(gf) else gf
                    break

        base = {
            "id": _first(record, _ID_COLUMNS),
            "wavelength": float(record["wavelength"]),
            "ep": float(ep),
            "gf": float(gf),
            "A(X)": float(record["abund"]),
        }
        slots = [(stage, ids) for stage in stages if element is not None
                 for ids in STANDARD_SOURCES.get(format_species(element, stage), ())]
        if not slots:
            rows.append({**base, "species": species_code(element, ion),
                         "A(X)_corr": np.nan, "DeltaA(X)": np.nan,
                         "source": "", "flavor": ""})
            continue

        for stage, ids in slots:
            hits = {h.source: h for h in matched[matched.source.isin(ids)].itertuples()}
            usable = [hits[i] for i in ids if i in hits and hits[i].ok]
            pick = usable[0] if usable else None
            source_id = pick.source if pick else next((i for i in ids if i in hits), ids[0])
            src = lookup.source(element, stage, source_id)
            rows.append({
                **base,
                "species": species_code(element, stage),
                "A(X)_corr": pick.a_nlte if pick else np.nan,
                "DeltaA(X)": pick.delta if pick else np.nan,
                # A recommended source that did not load (missing optional
                # models) is still named, by id.
                "source": paper_label(src.reference) if src else source_id,
                "flavor": src.flavor if src else "",
            })

    return pd.DataFrame(rows, columns=list(STANDARD_COLUMNS))
