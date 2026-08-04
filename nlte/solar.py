"""Solar chemical composition references.

Grids are published against different solar scales, so every source carries its
own reference (the IDL ``.sav`` grids literally ship a ``SOLAR`` field).  These
tables are only the fallback for sources that do not state one.
"""

from __future__ import annotations

import re

# Asplund, Grevesse, Sauval & Scott (2009), photospheric A(X) = log eps.
ASPLUND2009 = {
    "H": 12.00, "He": 10.93, "Li": 1.05, "Be": 1.38, "B": 2.70,
    "C": 8.43, "N": 7.83, "O": 8.69, "F": 4.56, "Ne": 7.93,
    "Na": 6.24, "Mg": 7.60, "Al": 6.45, "Si": 7.51, "P": 5.41,
    "S": 7.12, "Cl": 5.50, "Ar": 6.40, "K": 5.03, "Ca": 6.34,
    "Sc": 3.15, "Ti": 4.95, "V": 3.93, "Cr": 5.64, "Mn": 5.43,
    "Fe": 7.50, "Co": 4.99, "Ni": 6.22, "Cu": 4.19, "Zn": 4.56,
    "Sr": 2.87, "Y": 2.21, "Zr": 2.58, "Ba": 2.18, "La": 1.10,
    "Ce": 1.58, "Nd": 1.42, "Sm": 0.96, "Eu": 0.52, "Dy": 1.10,
}

# Asplund, Amarsi & Grevesse (2021), photospheric A(X).
ASPLUND2021 = dict(ASPLUND2009)
ASPLUND2021.update({
    "Li": 0.96, "Be": 1.38, "B": 2.70, "C": 8.46, "N": 7.83, "O": 8.69,
    "Ne": 8.06, "Na": 6.22, "Mg": 7.55, "Al": 6.43, "Si": 7.51, "P": 5.41,
    "S": 7.12, "K": 5.07, "Ca": 6.30, "Sc": 3.14, "Ti": 4.97, "V": 3.90,
    "Cr": 5.62, "Mn": 5.42, "Fe": 7.46, "Co": 4.94, "Ni": 6.20, "Cu": 4.18,
    "Zn": 4.56, "Sr": 2.83, "Y": 2.21, "Zr": 2.59, "Ba": 2.27, "Eu": 0.52,
})

# Grevesse & Sauval (1998) -- still the reference for several older grids.
GREVESSE1998 = dict(ASPLUND2009)
GREVESSE1998.update({
    "C": 8.52, "N": 7.92, "O": 8.83, "Na": 6.33, "Mg": 7.58, "Al": 6.47,
    "Si": 7.55, "K": 5.12, "Ca": 6.36, "Ti": 5.02, "Cr": 5.67, "Mn": 5.39,
    "Fe": 7.50, "Ni": 6.25, "Ba": 2.13, "Li": 1.10,
})

SOLAR_TABLES = {
    "asplund2009": ASPLUND2009,
    "asplund2021": ASPLUND2021,
    "grevesse1998": GREVESSE1998,
}

DEFAULT_TABLE = "asplund2009"

ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Sr": 38, "Y": 39, "Zr": 40, "Ba": 56, "La": 57, "Ce": 58,
    "Nd": 60, "Sm": 62, "Eu": 63, "Dy": 66,
}

_CANON = {sym.lower(): sym for sym in ATOMIC_NUMBER}


def normalize_element(element: str) -> str:
    """Return the canonical element symbol, e.g. ``"fe"`` or ``"FE"`` -> ``"Fe"``."""
    key = str(element).strip().lower()
    if key not in _CANON:
        raise KeyError(f"unknown element {element!r}")
    return _CANON[key]


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}

_SPECIES = re.compile(r"^([A-Z][a-z]?)[\s_\-]*([IVX]+|\d+)?$")


def parse_species(species: str):
    """Split a species label into ``(element, ion)``.

    Accepts the forms line lists actually use -- ``"Fe"``, ``"FeI"``, ``"Fe II"``,
    ``"Fe_2"``, ``"BaII"`` -- returning ``ion=None`` when the label names only an
    element.  Parsing is case-sensitive on the symbol, which is what separates
    ``"Si"`` (silicon) from ``"SI"`` (S I).
    """
    text = str(species).strip()
    match = _SPECIES.match(text)
    if match:
        symbol, stage = match.group(1), match.group(2)
        if symbol in ATOMIC_NUMBER:
            if stage is None:
                return symbol, None
            if stage.isdigit():
                return symbol, int(stage)
            if stage in _ROMAN:
                return symbol, _ROMAN[stage]
    # Fall back to the whole label as an element, so "FE" or "fe" still resolve.
    return normalize_element(text), None


def solar_abundance(element: str, table: str = DEFAULT_TABLE) -> float:
    """Solar A(X) on the named scale."""
    sym = normalize_element(element)
    try:
        values = SOLAR_TABLES[table]
    except KeyError:
        raise KeyError(
            f"unknown solar table {table!r}; choose from {sorted(SOLAR_TABLES)}"
        ) from None
    try:
        return values[sym]
    except KeyError:
        raise KeyError(f"no solar abundance for {sym} in table {table!r}") from None
