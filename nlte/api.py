"""The query interface: ask every grid that covers a line what it says.

Several elements are covered by more than one published grid -- magnesium by
three, oxygen by three, iron and carbon by two -- computed with different model
atmospheres and different assumptions.  A query therefore returns one
:class:`Correction` per matching source rather than picking a winner, so the
spread between literature sources stays visible.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np

from .registry import get_registry
from .solar import normalize_element, parse_species
from .solve import solve_delta, solve_ew_match

#: Default line-matching tolerance, in Angstrom.  Grids round their line labels
#: to varying precision (5684 for 5684.48), so this is looser than it looks.
DEFAULT_TOLERANCE = 1.0

QUERY_PARAMS = ("teff", "logg", "feh", "vmic")


@dataclass
class Correction:
    """What one grid says about one line for one star."""

    source: str
    reference: str
    element: str
    ion: int
    flavor: str
    wavelength: float           # as queried
    wavelength_grid: float      # as tabulated
    line_label: str
    delta: float                # A(NLTE) - A(LTE)
    a_nlte: float
    a_lte: float
    ok: bool                    # a usable number came back
    in_range: bool              # query inside the grid's axes
    valid_weight: float         # interpolation weight landing on valid nodes
    converged: bool
    scattering: bool | None = None
    blend: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Match:
    source: object
    line_index: int
    grid_wavelength: float


def sources(element=None, ion=None, registry=None) -> list:
    """Available sources, optionally filtered by species.

    ``element`` accepts a species label carrying the ionisation stage --
    ``"CI"``, ``"FeII"``, ``"Ba 2"`` -- as well as a bare symbol.
    """
    reg = registry or get_registry()
    if element is None:
        return reg.all()
    return reg.for_element(element, ion)


def find_lines(element, wavelength, ion=None, tol=DEFAULT_TOLERANCE,
               include_blends=True, source_ids=None, registry=None) -> list:
    """Every (source, line) pair matching a wavelength, nearest line per source.

    ``element`` may carry the ionisation stage -- ``"FeII"``, ``"Fe 2"`` -- which
    is how line lists usually label a species.  An explicit ``ion`` wins.
    """
    element, parsed_ion = parse_species(element)
    ion = parsed_ion if ion is None else ion
    matches = []
    for src in sources(element, ion, registry):
        if source_ids is not None and src.id not in source_ids:
            continue
        best = None
        for i, line in enumerate(src.lines):
            if line.blend and not include_blends:
                continue
            for candidate in line.match:
                if not np.isfinite(candidate):
                    continue
                d = abs(candidate - wavelength)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, i, candidate)
        if best is not None:
            matches.append(_Match(src, best[1], best[2]))
    return matches


def _decimals(value: float) -> int:
    """How precisely a wavelength was published, for picking a representative."""
    text = f"{value:.6f}".rstrip("0")
    return len(text.partition(".")[2])


def _cluster_lines(entries, tol: float) -> list:
    """Group ``(wavelength, source, blend)`` entries into physical lines.

    Two entries from the same grid are always different lines -- the iron list
    holds lines far closer together than the matching tolerance -- so groups are
    merged only when they share no source.  Merging is agglomerative and takes
    the closest admissible pair first, because a greedy left-to-right pass
    misassigns a coarser grid's value: with oxygen entries at 6156, 6157 and
    6158, the 6158 from one grid would otherwise be swept into the 6157 group
    and the line would appear twice in the output.
    """
    groups = {}
    for wl, source_id, blend in entries:
        groups.setdefault(round(wl, 6), []).append((wl, source_id, blend))
    ordered = [groups[k] for k in sorted(groups)]

    while len(ordered) > 1:
        best, best_gap = None, None
        for i in range(len(ordered) - 1):
            left, right = ordered[i], ordered[i + 1]
            gap = right[0][0] - left[0][0]
            if gap > tol:
                continue
            if {s for _, s, _ in left} & {s for _, s, _ in right}:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = i, gap
        if best is None:
            break
        ordered[best : best + 2] = [ordered[best] + ordered[best + 1]]
    return ordered


def line_index(element=None, ion=None, tol=DEFAULT_TOLERANCE, include_blends=True,
               registry=None):
    """Every line that can be corrected, merged across grids.

    ``element`` accepts a species label with the ionisation stage, as elsewhere.

    Returns a DataFrame of ``element, ion, wavelength, n_sources, blend,
    sources``.  ``blend`` marks lines where at least one grid tabulates the
    feature together with its neighbours rather than on its own.
    The same physical line is tabulated at different precisions by different
    grids -- 5711.0 here, 5711.09 there -- so wavelengths within ``tol`` are
    merged into one row and the most precisely published value is reported.

    Loading every source is unavoidable here, which means reading the large iron
    tables; they come from the on-disk cache after the first run.
    """
    import pandas as pd

    found = {}
    for src in sources(element, ion, registry):
        for line in src.lines:
            if line.blend and not include_blends:
                continue
            for wl in line.match:
                if np.isfinite(wl):
                    found.setdefault((src.element, src.ion), []).append(
                        (float(wl), src.id, line.blend)
                    )

    rows = []
    for (sym, stage), entries in found.items():
        for cluster in _cluster_lines(entries, tol):
            ids = sorted({source_id for _, source_id, _ in cluster})
            best = max((wl for wl, _, _ in cluster), key=lambda w: (_decimals(w), w))
            rows.append({
                "element": sym,
                "ion": stage,
                "wavelength": round(best, 3),
                "n_sources": len(ids),
                "blend": any(b for _, _, b in cluster),
                "sources": " ".join(ids),
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["element", "ion", "wavelength"]).reset_index(drop=True)


def _missing_params(src, params: dict) -> list:
    return [
        p for p in src.required_params
        if params.get(p) is None or not np.all(np.isfinite(np.asarray(params[p], dtype=float)))
    ]


def _blank(match, params, a_lte, note) -> Correction:
    src, line = match.source, match.source.lines[match.line_index]
    return Correction(
        source=src.id, reference=src.reference, element=src.element, ion=src.ion,
        flavor=src.flavor, wavelength=float("nan"), wavelength_grid=match.grid_wavelength,
        line_label=line.label, delta=float("nan"), a_nlte=float("nan"),
        a_lte=float(np.asarray(a_lte).reshape(-1)[0]), ok=False, in_range=False,
        valid_weight=0.0, converged=False, scattering=src.scattering,
        blend=line.blend, note=note,
    )


def _evaluate_delta(src, line_index, params, a_lte, mode):
    """Vectorised solve for a delta grid; returns arrays.

    Trial abundances are clamped to the grid's abundance range.  The iterates
    are internal, not user data, and without this a measurement sitting on the
    top node -- solar A(Mg) is exactly the top of the Matsuno grid -- would step
    a hundredth of a dex outside on the first iteration and return nothing.
    Whether the *measurement* is in range is judged separately, below.
    """
    feh = params.get("feh")
    lo, hi = src.abundance_bounds(feh)

    def delta_fn(trial):
        return src.delta_at(line_index, params, np.clip(trial, lo, hi), feh).value

    # A grid already indexed by the LTE abundance needs no solve -- iterating
    # would move the lookup off the abundance the table was built against.
    effective_mode = "direct" if src.indexed_by == "lte" else mode
    sol = solve_delta(delta_fn, a_lte, bounds=(np.min(lo), np.max(hi)), mode=effective_mode)
    solved = np.where(np.isfinite(sol.a_nlte), sol.a_nlte, a_lte)
    probe = src.delta_at(line_index, params, np.clip(solved, lo, hi), feh)

    abund_in_range = (a_lte >= np.asarray(lo) - 1e-9) & (a_lte <= np.asarray(hi) + 1e-9)
    notes = np.full(np.shape(sol.delta), "", dtype=object)
    edge = np.abs(solved - np.clip(solved, lo, hi)) > 1e-6
    notes[edge & np.isfinite(sol.delta)] = "Delta read at the grid's abundance edge"
    return (
        sol.delta, sol.a_nlte, sol.converged,
        probe.in_range & abund_in_range, probe.valid_weight, notes,
    )


def _evaluate_ew(src, line_index, params, a_lte):
    """Equivalent-width matching, point by point."""
    a_lte = np.atleast_1d(a_lte)
    n = a_lte.size
    delta = np.full(n, np.nan)
    a_nlte = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    notes = np.full(n, "", dtype=object)
    in_range = np.zeros(n, dtype=bool)
    weight = np.zeros(n)

    for i in range(n):
        one = {k: (np.asarray(v).reshape(-1)[min(i, np.asarray(v).size - 1)]
                   if np.ndim(v) else v)
               for k, v in params.items() if v is not None}
        nodes, lte, nlte = src.ew_curves(line_index, one)
        native = src.to_native_abundance(a_lte[i], one.get("feh"))
        m = solve_ew_match(nodes, lte, nlte, float(native))
        finite = np.isfinite(lte).sum() + np.isfinite(nlte).sum()
        in_range[i] = finite > 0
        weight[i] = finite / (2 * nodes.size)
        if m.ok:
            delta[i] = m.delta
            a_nlte[i] = src.to_logeps(m.a_nlte, one.get("feh"))
            ok[i] = True
        else:
            notes[i] = m.note
    return delta, a_nlte, ok, in_range, weight, notes


@contextmanager
def _weight_threshold(src, line_index, min_valid_weight):
    """Temporarily relax how much valid interpolation weight a cell must carry."""
    if min_valid_weight is None:
        yield
        return
    grids = src.grids[line_index]
    grids = grids if isinstance(grids, tuple) else (grids,)
    saved = [g.min_valid_weight for g in grids]
    try:
        for g in grids:
            g.min_valid_weight = float(min_valid_weight)
        yield
    finally:
        for g, old in zip(grids, saved):
            g.min_valid_weight = old


def _corrections_for(match, params, a_lte, mode, wavelength, min_valid_weight=None) -> list:
    src, li = match.source, match.line_index
    line = src.lines[li]
    a_lte = np.atleast_1d(np.asarray(a_lte, dtype=float))

    with _weight_threshold(src, li, min_valid_weight):
        if src.kind == "delta":
            delta, a_nlte, conv, in_range, weight, notes = _evaluate_delta(
                src, li, params, a_lte, mode
            )
        else:
            delta, a_nlte, conv, in_range, weight, notes = _evaluate_ew(src, li, params, a_lte)

    wavelength = np.broadcast_to(np.asarray(wavelength, dtype=float), a_lte.shape)
    out = []
    for i in range(a_lte.size):
        note = str(notes[i])
        if not note and not in_range[i]:
            note = "outside grid coverage"
        elif not note and not np.isfinite(delta[i]):
            # Distinguish "the grid has nothing here" from "the grid has values
            # but no abundance in range satisfies A - Delta(A) = A_lte".
            note = (
                "no self-consistent solution within the grid's abundance range"
                if weight[i] > 0 else "grid nodes here are masked"
            )
        out.append(Correction(
            source=src.id, reference=src.reference, element=src.element, ion=src.ion,
            flavor=src.flavor, wavelength=float(wavelength[i]),
            wavelength_grid=match.grid_wavelength, line_label=line.label,
            delta=float(delta[i]), a_nlte=float(a_nlte[i]), a_lte=float(a_lte[i]),
            ok=bool(np.isfinite(delta[i])), in_range=bool(in_range[i]),
            valid_weight=float(weight[i]), converged=bool(conv[i]),
            scattering=src.scattering, blend=line.blend, note=note,
        ))
    return out


def correct(element, wavelength, abund, teff=None, logg=None, feh=None, vmic=None,
            ion=None, sources=None, tol=DEFAULT_TOLERANCE, mode="selfconsistent",
            include_blends=True, min_valid_weight=None, registry=None) -> list:
    """Non-LTE corrections for one line, from every grid that covers it.

    Parameters
    ----------
    element, wavelength
        Species and air wavelength in Angstrom.
    abund
        The measured **LTE** abundance, A(X) = log eps.
    teff, logg, feh, vmic
        Stellar parameters.  Which ones matter depends on the grid: the Amarsi
        2016 iron tables have no [Fe/H] axis, and several grids have no
        microturbulence axis.  A grid needing a parameter you did not supply is
        returned with ``ok=False`` and an explanatory note, never silently.
    mode
        ``"selfconsistent"`` solves ``A_nlte - Delta(A_nlte) = abund``;
        ``"direct"`` reads Delta at ``abund`` without iterating.
    tol
        Line-matching tolerance in Angstrom.

    Returns
    -------
    list of :class:`Correction`, one per matching source.
    """
    params = {"teff": teff, "logg": logg, "feh": feh, "vmic": vmic}
    matches = find_lines(element, wavelength, ion=ion, tol=tol,
                         include_blends=include_blends, source_ids=sources,
                         registry=registry)
    results = []
    for match in matches:
        missing = _missing_params(match.source, params)
        if missing:
            results.append(_blank(match, params, abund,
                                  f"missing parameter(s): {', '.join(missing)}"))
            continue
        results.append(
            _corrections_for(match, params, abund, mode, wavelength, min_valid_weight)[0]
        )
    return results


def correct_frame(frame, mode="selfconsistent", tol=DEFAULT_TOLERANCE,
                  include_blends=True, sources=None, min_valid_weight=None,
                  registry=None, columns=None, keep=()):
    """Corrections for a table of measurements, one output row per source.

    ``frame`` needs ``element``, ``wavelength`` and ``abund`` columns plus
    whatever stellar parameters the grids require; ``columns`` renames them
    (``{"abund": "A_Fe", "vmic": "xi"}``).  ``keep`` names input columns to carry
    through to the output, e.g. a star identifier.

    Rows sharing a source, line and parameter set are solved together, which is
    what makes a full survey table practical.
    """
    import pandas as pd

    df = frame.rename(columns={v: k for k, v in (columns or {}).items()})
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    # "species" is the usual column name in a line list, and carries the
    # ionisation stage; "element" is accepted as a plainer alias.
    if "element" not in df.columns and "species" in df.columns:
        # Copy rather than rename, so `keep=["species"]` still finds it.
        df = df.assign(element=df["species"])
    required = {"element", "wavelength", "abund"}
    if not required.issubset(df.columns):
        raise ValueError(f"frame is missing columns: {sorted(required - set(df.columns))}")

    rows = []
    for (element, wavelength), block in df.groupby(["element", "wavelength"], sort=False):
        try:
            matches = find_lines(element, float(wavelength), tol=tol,
                                 include_blends=include_blends, source_ids=sources,
                                 registry=registry)
        except KeyError:
            # An unrecognised species must not abandon the rest of the table.
            matches = []
            note = f"unrecognised species {element!r}"
        else:
            note = "no grid covers this line"
        if not matches:
            for idx in block.index:
                rows.append((idx, Correction(
                    source="", reference="", element=str(element), ion=0, flavor="",
                    wavelength=float(wavelength), wavelength_grid=float("nan"),
                    line_label="", delta=float("nan"), a_nlte=float("nan"),
                    a_lte=float(df.at[idx, "abund"]), ok=False, in_range=False,
                    valid_weight=0.0, converged=False, note=note,
                )))
            continue

        params = {p: (block[p].to_numpy(dtype=float) if p in block else None)
                  for p in QUERY_PARAMS}
        a_lte = block["abund"].to_numpy(dtype=float)
        for match in matches:
            missing = _missing_params(match.source, params)
            if missing:
                note = f"missing parameter(s): {', '.join(missing)}"
                for idx in block.index:
                    rows.append((idx, _blank(match, params, df.at[idx, "abund"], note)))
                continue
            solved = _corrections_for(match, params, a_lte, mode, wavelength,
                                      min_valid_weight)
            for idx, corr in zip(block.index, solved):
                rows.append((idx, corr))

    out = pd.DataFrame([c.as_dict() for _, c in rows])
    out.insert(0, "row", [i for i, _ in rows])
    for col in keep:
        out[col] = [df.at[i, col] for i, _ in rows]
    return out.sort_values(["row", "source"], kind="stable").reset_index(drop=True)


def to_frame(corrections):
    """Turn a list of :class:`Correction` into a DataFrame."""
    import pandas as pd

    return pd.DataFrame([c.as_dict() for c in corrections])
