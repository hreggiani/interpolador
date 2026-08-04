"""Wavelength conventions.

This project speaks air wavelengths throughout, because that is what the
published line lists and the other grids use.  Some sources tabulate vacuum
wavelengths instead -- Breidablik's lithium line centres, for one -- so those
are converted at the source boundary.
"""

from __future__ import annotations

import numpy as np


def vacuum_to_air(wavelength):
    """Air wavelength in Angstrom from a vacuum wavelength in Angstrom.

    Uses the IAU standard dispersion relation of Birch & Downs (1994), as given
    by Morton (2000).  Below 2000 A the relation is not applicable and the input
    is returned unchanged, matching the usual convention that such lines are
    quoted in vacuum.
    """
    wl = np.asarray(wavelength, dtype=float)
    s2 = (1e4 / wl) ** 2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return np.where(wl >= 2000.0, wl / n, wl)


def air_to_vacuum(wavelength):
    """Vacuum wavelength in Angstrom from an air wavelength in Angstrom.

    The inverse relation of Ciddor (1996), as given by Morton (2000).
    """
    wl = np.asarray(wavelength, dtype=float)
    s2 = (1e4 / wl) ** 2
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) \
        + 0.0001599740894897 / (38.92568793293 - s2)
    return np.where(wl >= 2000.0, wl * n, wl)
