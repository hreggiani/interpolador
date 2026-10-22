"""Command-line entry points.

``nlte-correct``
    Corrects a table of measurements, writing one row per measurement and
    source (or one column per source with ``--wide``), and beside it the
    standard output, ``<output>_std``: the recommended sources only, in a
    fixed set of columns.

``nlte-validate``
    Loads manifests, prints what each source covers, and re-queries the grids at
    their own nodes to prove the axes were assembled correctly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .api import DEFAULT_TOLERANCE, correct_frame, line_index
from .interp import RectGrid
from .registry import get_registry, set_grid_root
from .standard import standard_frame


def _read_any(path: Path):
    import pandas as pd

    if path.suffix.lower() in (".fits", ".fit", ".fts"):
        from astropy.table import Table

        return Table.read(path).to_pandas()
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else None
    # utf-8-sig strips the byte-order mark VizieR and Excel exports carry.
    return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")


def _write_any(df, path: Path) -> None:
    if path.suffix.lower() in (".fits", ".fit", ".fts"):
        from astropy.table import Table

        Table.from_pandas(df).write(path, overwrite=True)
    elif path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _parse_col_map(text) -> dict:
    if not text:
        return {}
    out = {}
    for piece in text.split(","):
        canonical, _, actual = piece.partition("=")
        if not actual:
            raise SystemExit(f"--col-map entry {piece!r} must look like canonical=column")
        out[canonical.strip()] = actual.strip()
    return out


def main_correct(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="nlte-correct",
        description="Apply published non-LTE abundance corrections to a table of "
                    "measurements, reporting every grid that covers each line.",
    )
    p.add_argument("input", type=Path, help="CSV / TSV / FITS / Parquet table")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--col-map", default="",
                   help="rename input columns, e.g. 'abund=A_Fe,vmic=xi,wavelength=lambda'. "
                        "An 'element' or 'species' column may carry the ionisation "
                        "stage, as in FeII or Fe 2.")
    p.add_argument("--keep", default="",
                   help="comma-separated input columns to carry through, e.g. 'star,snr'")
    p.add_argument("--sources", default="",
                   help="comma-separated source ids to restrict to (default: all)")
    p.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE,
                   help="line-matching tolerance in Angstrom (default: %(default)s)")
    p.add_argument("--mode", choices=("selfconsistent", "direct"), default="selfconsistent",
                   help="solve A_nlte - Delta(A_nlte) = A_lte, or read Delta at A_lte")
    p.add_argument("--min-valid-weight", type=float, default=None,
                   help="fraction of interpolation weight that must land on valid "
                        "grid nodes (default: 0.5)")
    p.add_argument("--no-blends", action="store_true",
                   help="skip grid entries that represent several lines at once")
    p.add_argument("--wide", action="store_true",
                   help="pivot to one delta_<source> column per source")
    p.add_argument("--grids", type=Path, default=None, help="grid root directory")
    args = p.parse_args(argv)

    if args.grids:
        set_grid_root(args.grids)

    df = _read_any(args.input)
    col_map = _parse_col_map(args.col_map)
    out = correct_frame(
        df,
        mode=args.mode,
        tol=args.tol,
        include_blends=not args.no_blends,
        sources=[s.strip() for s in args.sources.split(",") if s.strip()] or None,
        min_valid_weight=args.min_valid_weight,
        columns=col_map,
        keep=[c.strip() for c in args.keep.split(",") if c.strip()],
    )
    std = standard_frame(out, df, columns=col_map)

    if args.wide:
        # Pivot on the row number alone.  Including element or wavelength in the
        # index would collide with the input's own columns on the merge back,
        # and dropping all-NaN groups would silently lose every measurement no
        # grid covers -- the output must keep one row per input row.
        wide = (
            out[out.source != ""]
            .pivot_table(index="row", columns="source", values="delta", aggfunc="first")
            .add_prefix("delta_")
        )
        out = (
            df.reset_index(drop=True)
            .join(wide.reindex(range(len(df))))
        )

    _write_any(out, args.output)
    usable = int(out.filter(like="delta").notna().to_numpy().sum()) if args.wide else int(out.ok.sum())
    print(f"wrote {args.output} ({len(out)} rows, {usable} usable corrections)")
    std_path = standard_path(args.output)
    _write_any(std, std_path)
    print(f"wrote {std_path} ({len(std)} rows, "
          f"{int(std['DeltaA(X)'].notna().sum())} usable corrections)")
    return 0


def standard_path(output: Path) -> Path:
    """Where the standard output goes: ``corrected.csv`` -> ``corrected_std.csv``."""
    return output.with_name(f"{output.stem}_std{output.suffix}")


MAX_MARKDOWN_LINES = 60


def _markdown_table(frame) -> list:
    head = ["| Wavelength (A) | Grids | Sources |", "|---:|---:|---|"]
    return head + [
        f"| {r.wavelength:.3f}{' *' if r.blend else ''} | {r.n_sources} | "
        f"{r.sources.replace(' ', ', ')} |"
        for r in frame.itertuples()
    ]


def _line_coverage_markdown(frame, csv_name: str) -> str:
    """A readable coverage document: a summary, then per-species line lists.

    Iron alone carries 2334 lines, which is unreadable as Markdown, so long
    species point at the CSV instead of being dumped inline.
    """
    out = [
        "# Line coverage",
        "",
        "Every line that can be corrected, across all grids in this project.",
        "Wavelengths are air, in Angstrom. Lines within the matching tolerance",
        "are merged across grids and reported at the most precisely published value.",
        "An asterisk marks a line at least one grid tabulates as part of a blend",
        "rather than on its own.",
        "",
        f"Generated by `nlte-lines`; the complete machine-readable list is [{csv_name}]({csv_name}).",
        "",
        "## Summary",
        "",
        "| Species | Lines | Range (A) | Sources |",
        "|---|---:|---|---|",
    ]
    for species, block in frame.groupby("species", sort=False):
        ids = sorted({s for row in block.sources for s in row.split()})
        out.append(
            f"| {species} | {len(block)} | "
            f"{block.wavelength.min():.1f}-{block.wavelength.max():.1f} | {len(ids)} |"
        )
    out += ["", f"**{len(frame)} lines** over {frame.species.nunique()} species.", ""]

    for species, block in frame.groupby("species", sort=False):
        out += [f"## {species}", ""]
        if len(block) > MAX_MARKDOWN_LINES:
            out += [
                f"{len(block)} lines spanning {block.wavelength.min():.1f}-"
                f"{block.wavelength.max():.1f} A -- too many to list here; "
                f"see [{csv_name}]({csv_name}).",
                "",
            ]
        else:
            out += _markdown_table(block) + [""]
    return "\n".join(out)


def main_lines(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="nlte-lines",
        description="List every line that can be corrected, across all grids.",
    )
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="write here (.csv, or .md for the coverage document)")
    p.add_argument("--element", default=None, help="restrict to one element")
    p.add_argument("--ion", type=int, default=None)
    p.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE,
                   help="merge wavelengths within this many Angstrom (default: %(default)s)")
    p.add_argument("--no-blends", action="store_true",
                   help="skip entries that represent several lines at once")
    p.add_argument("--grids", type=Path, default=None, help="grid root directory")
    args = p.parse_args(argv)

    if args.grids:
        set_grid_root(args.grids)

    frame = line_index(element=args.element, ion=args.ion, tol=args.tol,
                       include_blends=not args.no_blends)
    if frame.empty:
        print("no lines found", file=sys.stderr)
        return 1

    if args.output is None:
        print(frame.to_string(index=False))
        return 0

    if args.output.suffix.lower() in (".md", ".markdown"):
        csv_path = args.output.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        args.output.write_text(_line_coverage_markdown(frame, csv_path.name) + "\n")
        print(f"wrote {args.output} and {csv_path} ({len(frame)} lines)")
    else:
        frame.to_csv(args.output, index=False)
        print(f"wrote {args.output} ({len(frame)} lines)")
    return 0


def _node_round_trip(grid, rng, n=200):
    """Query a grid at its own nodes; returns the largest disagreement."""
    if isinstance(grid, RectGrid):
        shape = grid.values.shape
        idx = np.column_stack([rng.integers(0, s, n) for s in shape])
        pts = np.column_stack([grid.axes[d][idx[:, d]] for d in range(len(shape))])
        truth = grid.values[tuple(idx.T)]
    else:
        rshape = grid.values.shape[1:]
        ni = rng.integers(0, grid.nodes.shape[0], n)
        ri = np.column_stack([rng.integers(0, s, n) for s in rshape])
        pts = np.column_stack(
            [grid.nodes[ni]] + [grid.rect_axes[d][ri[:, d]] for d in range(len(rshape))]
        )
        truth = grid.values[(ni, *ri.T)]
    got = grid(pts).value
    m = np.isfinite(truth)
    if not m.any():
        return None
    return float(np.nanmax(np.abs(got[m] - truth[m])))


def main_validate(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="nlte-validate",
        description="Load grid manifests, report coverage, and verify that each "
                    "grid reproduces its own tabulated values at its nodes.",
    )
    p.add_argument("manifest", nargs="*", type=Path,
                   help="manifests to check (default: every one under the grid root)")
    p.add_argument("--grids", type=Path, default=None, help="grid root directory")
    p.add_argument("--quick", action="store_true", help="skip the node round-trip check")
    args = p.parse_args(argv)

    reg = set_grid_root(args.grids) if args.grids else get_registry()
    wanted = {m.resolve() for m in args.manifest} or None
    manifests = [m for m in reg.manifests if wanted is None or m.path.resolve() in wanted]
    if not manifests:
        print("no manifests found", file=sys.stderr)
        return 1

    rng = np.random.default_rng(0)
    failures = 0
    for manifest in manifests:
        try:
            loaded = reg._load(manifest)
        except Exception as exc:                      # a bad manifest must be loud
            print(f"FAIL {manifest.base_id}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if not loaded:
            why = reg.missing.get(manifest.key, "reader returned no sources")
            print(f"FAIL {manifest.base_id}: {why}\n")
            failures += 1
            continue
        for src in loaded:
            print(src.describe())
            if args.quick:
                print()
                continue
            if not np.isfinite(src.coverage()) and src.geometry() not in ("rectilinear", "hybrid"):
                # Networks and package-backed models have no nodes to re-query.
                print("  round-trip: n/a (model-backed)\n")
                continue
            errs = []
            for grid in src.grids[:8]:
                for part in (grid if isinstance(grid, tuple) else (grid,)):
                    e = _node_round_trip(part, rng)
                    if e is not None:
                        errs.append(e)
            if not errs:
                print("  round-trip: no finite nodes sampled\n")
            elif max(errs) < 1e-6:
                print(f"  round-trip: OK (max |error| = {max(errs):.2e})\n")
            else:
                print(f"  round-trip: FAILED (max |error| = {max(errs):.2e})\n")
                failures += 1

    print(f"{len(manifests)} manifest(s) checked, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_correct())
