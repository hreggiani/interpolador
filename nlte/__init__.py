"""Unified interpolation over published non-LTE abundance-correction grids.

    >>> from nlte import correct
    >>> for c in correct("MgI", 5711.09, abund=7.60, teff=5777, logg=4.44,
    ...                  feh=0.0, vmic=1.0):
    ...     print(c.source, c.delta)

Every grid covering the line answers, so different literature sources can be
compared directly.  See ``docs/adding_a_table.md`` for adding new tables.
"""

from .api import (
    Correction,
    correct,
    correct_frame,
    find_lines,
    line_index,
    sources,
    to_frame,
)
from .registry import register_reader, register_source, set_grid_root
from .solve import solve_delta, solve_ew_match
from .standard import STANDARD_SOURCES, standard_frame
from .source import GridSource, Line

__all__ = [
    "correct",
    "correct_frame",
    "find_lines",
    "line_index",
    "sources",
    "to_frame",
    "standard_frame",
    "STANDARD_SOURCES",
    "Correction",
    "GridSource",
    "Line",
    "register_reader",
    "register_source",
    "set_grid_root",
    "solve_delta",
    "solve_ew_match",
]

__version__ = "0.1.0"
