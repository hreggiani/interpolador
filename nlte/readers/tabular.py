"""Generic reader for grids stored as text tables (CSV / TSV / whitespace).

This one reader covers the Amarsi 2020 CSVs, the silicon CSV, the Amarsi 2016
iron tables, the Osorio Mg VizieR table and the Matsuno Mg VizieR table -- and,
importantly, most tables that will be published next.  Everything it needs comes
from the manifest, so a new tabular grid is a TOML file and no Python.

Rows are scattered into an array indexed by the grid axes, with missing
combinations left as NaN, so partially populated tables (Matsuno) and tables
that are sparse in a plane (iron, 54 of 104 teff-logg pairs) both load without
special cases.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..registry import cache_dir, register_reader
from ..solar import normalize_element, parse_species
from ..source import GridSource, Line, make_grid, resolve_solar

CANONICAL_PARAMS = ("teff", "logg", "feh", "vmic")

_DELIMITERS = {
    "whitespace": r"\s+",
    "tab": "\t",
    "comma": ",",
    "semicolon": ";",
}

_CACHE_VERSION = 3


# ---------------------------------------------------------------- parsing


def _delimiter(spec: dict):
    d = spec.get("delimiter", ",")
    return _DELIMITERS.get(d, d)


def _ensure_extracted(path: Path) -> Path:
    """Extract ``<name>.txt`` from ``<name>.tar.gz`` on first use."""
    if path.exists():
        return path
    archive = path.with_suffix(path.suffix + ".tar.gz") if path.suffix else None
    candidates = [p for p in (archive, path.parent / (path.name + ".tar.gz"),
                              path.with_suffix(".tar.gz")) if p and p.exists()]
    if not candidates:
        raise FileNotFoundError(path)
    with tarfile.open(candidates[0]) as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == path.name), None)
        if member is None:
            raise FileNotFoundError(f"{path.name} not found inside {candidates[0].name}")
        member.name = path.name
        tar.extract(member, path.parent, filter="data")
    return path


def _is_separator_row(values) -> bool:
    """VizieR tables put a row of dashes between the header and the data."""
    cells = [str(v).strip() for v in values]
    return all(c == "" or set(c) <= {"-"} for c in cells)


_CDS_FIELD = re.compile(
    r"^\s*(?:(?P<start>\d+)\s*-\s*)?(?P<end>\d+)\s+(?P<fmt>[AIFE][\d.]*)\s+\S+\s+(?P<label>\S+)"
)


def cds_layout(readme: Path, filename: str):
    """Column names and byte ranges for ``filename`` from a CDS ReadMe.

    VizieR ReadMes describe each table in a "Byte-by-byte Description of file:"
    block whose header may name several files or a pattern (``tablea[12345].dat``,
    ``*.dat``).  Fixed-width parsing is the only safe way to read these tables:
    optional fields are left blank, which shifts every later column under a
    whitespace split, and flags can be glued to numbers (``-2.0a``).
    """
    name = Path(filename).name
    if name.endswith(".gz"):
        name = name[:-3]
    lines = Path(readme).read_text(encoding="latin-1").splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("Byte-by-byte Description of file:"):
            continue
        patterns = line.split(":", 1)[1].split()
        if not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        names, colspecs, rules = [], [], 0
        for row in lines[i + 1:]:
            if set(row.strip()) == {"-"}:
                rules += 1
                if rules == 3:          # header rule, column-title rule, closing rule
                    break
                continue
            m = _CDS_FIELD.match(row)
            if rules == 2 and m:
                end = int(m.group("end"))
                start = int(m.group("start") or end)
                names.append(m.group("label"))
                colspecs.append((start - 1, end))
        if names:
            return names, colspecs
    raise ValueError(f"{readme}: no byte-by-byte description covers {name}")


def read_table(path: Path, spec: dict, usecols=None) -> pd.DataFrame:
    """Read the data file into a DataFrame, honouring the manifest's read rules.

    ``usecols`` is passed through by callers that know which columns they need;
    the iron tables are 2.4 GB on disk and carry four columns of line data that
    the interpolator never touches.
    """
    path = _ensure_extracted(Path(path))
    if spec.get("format") == "cds":
        names, colspecs = cds_layout(path.parent / spec.get("readme", "ReadMe"), path.name)
        # VizieR serves large tables gzipped; read_fwf does not infer that here.
        df = pd.read_fwf(path, colspecs=colspecs, names=names, header=None,
                         dtype=str, keep_default_na=False,
                         compression="gzip" if path.suffix == ".gz" else None)
        df = df.apply(lambda c: c.str.strip())
        if usecols:
            df = df[[c for c in df.columns if c in set(usecols)]]
        return df
    kwargs = {
        "sep": _delimiter(spec),
        "comment": spec.get("comment"),
        "skiprows": spec.get("skiprows", 0),
        # The C engine handles r"\s+" natively and is far faster than the Python
        # engine, which matters at tens of millions of rows (the iron tables).
        "engine": "c",
    }
    if spec.get("header") == "none":
        kwargs["header"] = None
        kwargs["names"] = spec["names"]
    if usecols:
        kwargs["usecols"] = lambda c: str(c).strip() in set(usecols)
    with warnings.catch_warnings():
        # A leftover column-name row inside the data makes pandas chunk-infer
        # mixed dtypes; those rows are dropped as non-numeric axis rows later,
        # and low_memory=False is not an option on a 2.4 GB file.
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        df = pd.read_csv(path, **{k: v for k, v in kwargs.items() if v is not None})
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) and _is_separator_row(df.iloc[0].values):
        df = df.iloc[1:]
    return df.reset_index(drop=True)


def data_files(manifest: dict, path: Path) -> list:
    """Every file a manifest reads, in a stable order.

    ``files`` lists names or glob patterns relative to the manifest, for grids
    published as one file per line (Merle et al. 2011) or per (log g, [Fe/H])
    pair (Gallagher et al. 2020).  Without it the single ``file`` is used, as
    before.
    """
    spec = manifest.get("files")
    if spec is None:
        return [_ensure_extracted(Path(path))]
    base = Path(path) if Path(path).is_dir() else Path(path).parent
    found = []
    for pattern in [spec] if isinstance(spec, str) else spec:
        hits = sorted(base.glob(pattern))
        if not hits:
            raise FileNotFoundError(base / pattern)
        found.extend(h for h in hits if h not in found)
    return found


def _file_params(manifest: dict, path: Path) -> dict:
    """Columns carried by a file's name, e.g. ``g45mm10`` -> log g 4.5, [Fe/H] -1.0."""
    spec = manifest.get("file_params")
    if not spec:
        return {}
    m = re.search(spec["pattern"], Path(path).name)
    if not m:
        raise ValueError(f"{Path(path).name} does not match file_params {spec['pattern']!r}")
    scale = spec.get("scale", {})
    # "+ 0.0" turns the -0.0 of a negative scale on "00" into a plain zero.
    return {k: (float(v) * float(scale[k]) + 0.0 if k in scale else v)
            for k, v in m.groupdict().items()}


def _melt(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Stack columns that each hold one (line, abundance) combination.

    Korotin et al. 2015 publish one correction column per line and [Ba/Fe]
    (``d1-0.2``, ``d1+0.1``, ...).  The pattern's named groups become columns of
    the long table and the stacked values go into ``value``.
    """
    rx = re.compile(spec["pattern"])
    matched = {c: rx.fullmatch(c) for c in df.columns}
    value_cols = [c for c, m in matched.items() if m]
    if not value_cols:
        raise ValueError(f"melt pattern {spec['pattern']!r} matches no column")
    keep = [c for c in df.columns if c not in value_cols]
    long = df.melt(id_vars=keep, value_vars=value_cols, var_name="__col",
                   value_name=spec.get("value", "value"))
    groups = pd.DataFrame([matched[c].groupdict() for c in long["__col"]], index=long.index)
    labels = {str(k): v for k, v in spec.get("line_labels", {}).items()}
    if "line" in groups and labels:
        groups["line"] = groups["line"].map(lambda g: labels.get(g, g))
    return pd.concat([long.drop(columns="__col"), groups], axis=1)


def load_frame(manifest: dict, path: Path, usecols=None) -> pd.DataFrame:
    """Read every data file of a manifest into one frame, before any cleaning."""
    spec = manifest.get("read", {})
    if manifest.get("melt") or spec.get("where"):
        # The columns named in [columns] only exist after melting, and a row
        # filter may test columns the grid itself does not use.
        usecols = None
    frames = []
    for f in data_files(manifest, path):
        df = read_table(f, spec, usecols=usecols)
        for col, value in _file_params(manifest, f).items():
            df[col] = value
        frames.append(df)
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    if spec.get("where"):
        # A pandas query on the raw table, for files that tabulate one node twice
        # under different assumptions (Korotin et al. 2015 give [Fe/H] = -0.5 with
        # and without alpha enhancement); the manifest picks one.
        df = df.query(spec["where"]).reset_index(drop=True)
    if manifest.get("melt"):
        df = _melt(df, manifest["melt"])
    return df


_WL_IN_LABEL = re.compile(r"(\d+(?:\.\d+)?)\s*(nm|a|ang|angstrom)?", re.IGNORECASE)


def parse_wavelength(label, unit: str = "angstrom") -> float:
    """Wavelength in Angstrom from a numeric value or a label like ``'505.2nm'``."""
    if isinstance(label, (int, float, np.floating, np.integer)):
        value, tag = float(label), None
    else:
        m = _WL_IN_LABEL.search(str(label))
        if not m:
            return float("nan")
        value, tag = float(m.group(1)), (m.group(2) or "").lower()
    if tag == "nm":
        return value * 10.0
    if tag:
        return value
    if unit == "nm":
        return value * 10.0
    if unit == "auto":
        # Mixed columns: Amarsi's 2020 CSVs list blended entries in nm (777.0)
        # alongside single lines in Angstrom (6588.0).
        return value * 10.0 if value < 1000 else value
    return value


def _make_lines(labels, spec: dict) -> list:
    """Turn the file's line identifiers into :class:`Line` entries.

    An alias overrides the parsed value: it lists the real wavelengths an entry
    covers, which handles both blends (one entry, several components) and labels
    that are not wavelengths at all, such as the potassium grid's ``151631`` for
    15163.1 A.  The first alias becomes the representative wavelength.
    """
    unit = spec.get("unit", "angstrom")
    aliases = {str(k): v for k, v in spec.get("aliases", {}).items()}
    lines = []
    for label in labels:
        text = label if isinstance(label, str) else f"{label:g}"
        if text in aliases:
            match = tuple(float(x) for x in np.atleast_1d(aliases[text]))
            wl = match[0]
        else:
            wl = parse_wavelength(label, unit)
            match = (wl,)
        blend = bool(re.search(r"combin|blend", text, re.IGNORECASE)) or len(match) > 1
        lines.append(Line(wavelength=wl, label=text, blend=blend, match=match))
    return lines


# ---------------------------------------------------------------- gridding


def _abundance_independent(manifest: dict) -> bool:
    return bool(manifest.get("abundance", {}).get("independent", False))


def _axis_columns(manifest: dict) -> dict:
    cols = manifest["columns"]
    axes = {p: cols[p] for p in CANONICAL_PARAMS if p in cols}
    if "abund" in cols:
        axes["abund"] = cols["abund"]
    elif not _abundance_independent(manifest):
        raise ValueError(
            "manifest [columns] must map 'abund' (or set [abundance] independent = true)"
        )
    return axes


def _to_native_abundance(df: pd.DataFrame, manifest: dict, element: str) -> pd.DataFrame:
    """Reparametrise the abundance column into the axis the grid is rectilinear in.

    The silicon table stores log eps, but its nodes shift with [Fe/H] (log eps -
    [Fe/H] is always 7.01 ... 9.01), so it is only a rectangular grid when read
    as [Si/Fe].  Declaring ``stored = "logeps"`` alongside ``kind = "xfe"`` asks
    for that conversion; tables already storing [X/Fe] need nothing.
    """
    ab = manifest.get("abundance", {})
    kind = ab.get("kind", "logeps")
    stored = ab.get("stored", kind)
    if stored == kind:
        return df

    if stored != "logeps":
        raise ValueError(f"cannot convert a '{stored}' abundance column to '{kind}'")
    zero = resolve_solar(element, ab) if kind == "xfe" else ab.get("zero_point")
    if zero is None:
        raise ValueError(f"converting log eps to '{kind}' needs a solar value or zero_point")
    feh_col = manifest["columns"].get("feh")
    if feh_col is None:
        raise ValueError(f"converting log eps to '{kind}' needs a [Fe/H] column")

    col = manifest["columns"]["abund"]
    df = df.copy()
    df[col] = df[col].to_numpy(dtype=float) - float(zero) - df[feh_col].to_numpy(dtype=float)
    return df


def _build_arrays(df: pd.DataFrame, manifest: dict, value_columns: dict):
    """Scatter table rows into per-line arrays indexed by the grid axes.

    Returns ``(axis_nodes, scattered_names, scattered_nodes, labels, arrays)``
    where ``arrays`` maps each value column to ``(n_lines, *grid_shape)``.
    """
    axis_cols = _axis_columns(manifest)
    scattered = tuple(manifest.get("scattered", ()))
    line_col = manifest["columns"].get("line")

    labels = (
        list(pd.unique(df[line_col])) if line_col is not None else [manifest.get("line", "all")]
    )
    line_index = {lab: i for i, lab in enumerate(labels)}

    rect_names = [n for n in axis_cols if n not in scattered]
    nodes = {n: np.sort(pd.unique(df[axis_cols[n]].to_numpy(dtype=float))) for n in rect_names}

    if scattered:
        cols = [df[axis_cols[n]].to_numpy(dtype=float) for n in scattered]
        pairs = np.column_stack(cols)
        scat_nodes, scat_idx = np.unique(pairs, axis=0, return_inverse=True)
        lead = (len(scat_nodes),)
    else:
        scat_nodes, scat_idx, lead = None, None, ()

    shape = lead + tuple(nodes[n].size for n in rect_names)
    rect_idx = [
        np.searchsorted(nodes[n], df[axis_cols[n]].to_numpy(dtype=float)) for n in rect_names
    ]
    row_idx = ([scat_idx] if scattered else []) + rect_idx
    line_idx = (
        df[line_col].map(line_index).to_numpy() if line_col is not None
        else np.zeros(len(df), dtype=int)
    )

    arrays = {}
    for key, col in value_columns.items():
        out = np.full((len(labels),) + shape, np.nan)
        out[(line_idx, *row_idx)] = df[col].to_numpy(dtype=float)
        arrays[key] = out

    ordered = {n: nodes[n] for n in rect_names}
    return ordered, scattered, scat_nodes, labels, arrays


def _clean_values(df: pd.DataFrame, manifest: dict, columns) -> pd.DataFrame:
    quality = manifest.get("quality", {})
    sentinel = quality.get("sentinel")
    df = df.copy()
    for col in columns:
        v = pd.to_numeric(df[col], errors="coerce")
        if sentinel is not None:
            v = v.mask(v <= float(sentinel) * (1 - 1e-6) if sentinel < 0 else v == sentinel)
        df[col] = v

    # Axis columns must be numeric: a stray header or ruler row left in the file
    # would otherwise introduce a NaN node and corrupt every axis lookup.
    axis_cols = [c for c in _axis_columns(manifest).values() if c in df.columns]
    if axis_cols:
        for col in axis_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=axis_cols)

    # Axes printed with too few decimals: Lind et al. 2022 tabulate [X/Fe] in
    # steps of 0.25 in an F4.1 column, so -0.75 reads as -0.8.  [snap] rounds an
    # axis back onto its true step.
    for axis, step in manifest.get("snap", {}).items():
        col = _axis_columns(manifest)[axis]
        df[col] = np.round(df[col].to_numpy(dtype=float) / float(step)) * float(step) + 0.0

    flag_col = quality.get("flag_column")
    if flag_col and manifest.get("require_flag", False):
        good = str(quality.get("flag_good", "yes"))
        df = df[df[flag_col].astype(str).str.strip() == good]
    return df


def _common_kwargs(manifest: dict, element: str) -> dict:
    ab = manifest.get("abundance", {})
    kind = ab.get("kind", "logeps")
    return {
        "reference": manifest.get("reference", ""),
        "element": element,
        "indexed_by": ab.get("indexed_by", "nlte"),
        "abund_kind": kind,
        "abund_solar": resolve_solar(element, ab) if kind == "xfe" else None,
        "abund_zero_point": ab.get("zero_point") if kind == "offset" else None,
        "solar_table": ab.get("solar_table"),
        "scattering": manifest.get("scattering"),
        "notes": manifest.get("notes", ""),
    }


# ---------------------------------------------------------------- caching


def _cache_file(files, manifest: dict) -> Path:
    stamps = "|".join(
        f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in map(Path, files)
    )
    payload = json.dumps(manifest, sort_keys=True, default=str)
    digest = hashlib.sha1(f"{_CACHE_VERSION}|{stamps}|{payload}".encode()).hexdigest()[:16]
    return cache_dir(Path(files[0]).parent) / f"{manifest['id']}-{digest}.npz"


def _load_cache(files, manifest: dict):
    """Parsed arrays from a previous load, if the files and manifest are unchanged."""
    if not manifest.get("cache", False):
        return None
    f = _cache_file(files, manifest)
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def _store_cache(files, manifest: dict, payload: dict) -> None:
    if not manifest.get("cache", False):
        return
    try:
        np.savez(_cache_file(files, manifest), **payload)
    except Exception:  # a cache failure must never break a load
        pass


def _prepare(manifest: dict, path: Path, value_columns: dict, element: str, ion_col=None):
    """Read, clean and grid a table, going through the on-disk cache if enabled.

    The Amarsi 2016 iron tables are 8.2M rows each; parsing them takes seconds,
    while reloading the gridded arrays takes milliseconds.
    """
    files = data_files(manifest, path)    # extracts archives before stat()-ing them
    cached = _load_cache(files, manifest)

    def needed():
        cols = set(_axis_columns(manifest).values()) | set(value_columns.values())
        if manifest["columns"].get("line"):
            cols.add(manifest["columns"]["line"])
        if ion_col:
            cols.add(ion_col)
        flag = manifest.get("quality", {}).get("flag_column")
        if flag and manifest.get("require_flag", False):
            cols.add(flag)
        return cols

    if cached is not None:
        blocks = []
        for gi in range(int(cached["n_groups"])):
            nodes = {
                n: cached[f"g{gi}_axis_{n}"]
                for n in cached[f"g{gi}_rect_names"].tolist()
            }
            scattered = tuple(cached[f"g{gi}_scattered"].tolist())
            scat = cached[f"g{gi}_scat_nodes"] if scattered else None
            arrays = {k: cached[f"g{gi}_val_{k}"] for k in value_columns}
            blocks.append((
                int(cached[f"g{gi}_ion"]),
                nodes, scattered, scat,
                cached[f"g{gi}_labels"].tolist(),
                arrays,
            ))
        return blocks

    df = load_frame(manifest, path, usecols=needed())
    df = _clean_values(df, manifest, value_columns.values())
    df = _to_native_abundance(df, manifest, element)
    groups = (
        [(_ion_number(k), g) for k, g in df.groupby(ion_col)]
        if ion_col else [(int(manifest.get("ion", 1)), df)]
    )

    blocks = []
    payload = {"n_groups": np.array(len(groups))}
    for gi, (ion, sub) in enumerate(groups):
        nodes, scattered, scat_nodes, labels, arrays = _build_arrays(sub, manifest, value_columns)
        blocks.append((ion, nodes, scattered, scat_nodes, labels, arrays))
        payload[f"g{gi}_ion"] = np.array(ion)
        payload[f"g{gi}_rect_names"] = np.array(list(nodes), dtype=object)
        payload[f"g{gi}_scattered"] = np.array(list(scattered), dtype=object)
        payload[f"g{gi}_scat_nodes"] = scat_nodes if scattered else np.array([])
        payload[f"g{gi}_labels"] = np.array(labels, dtype=object)
        for n, v in nodes.items():
            payload[f"g{gi}_axis_{n}"] = v
        for k, v in arrays.items():
            payload[f"g{gi}_val_{k}"] = v
    _store_cache(files, manifest, payload)
    return blocks


def _ion_number(value) -> int:
    """Ionisation stage from an ion-column value: ``Fe1``, ``Fe 2``, ``Ca II``."""
    _, ion = parse_species(str(value).strip())
    if ion is None:
        m = re.search(r"(\d+)$", str(value).strip())
        if not m:
            raise ValueError(f"cannot read an ionisation stage from {value!r}")
        ion = int(m.group(1))
    return int(ion)


# ---------------------------------------------------------------- axis shaping

#: Abundance nodes given to a grid that does not depend on abundance.  The axis
#: is clipped, so any query lands on the (duplicated) values; the span only has
#: to be wider than any real abundance in any parametrisation.
FREE_ABUNDANCE_SPAN = (-99.0, 99.0)


def _shape_axes(nodes: dict, scattered: tuple, arrays: dict, manifest: dict):
    """Fold length-1 axes into fixed parameters and handle abundance-free grids.

    Returns ``(nodes, arrays, fixed, independent)``.  A grid whose abundance
    axis has a single node -- the INASAN Fe I grid is tabulated at [Fe/Fe] = 0
    only -- or that declares ``[abundance] independent = true`` gets a flat,
    clipped abundance axis, so the rest of the package sees the usual layout.
    """
    lead = 1 + (1 if scattered else 0)          # line axis, then the scattered block
    nodes, fixed = dict(nodes), {}
    for j in reversed(range(len(nodes))):
        name = list(nodes)[j]
        if nodes[name].size == 1:
            fixed[name] = float(nodes[name][0])
            arrays = {k: np.take(v, 0, axis=lead + j) for k, v in arrays.items()}
            del nodes[name]
    independent = _abundance_independent(manifest) or "abund" in fixed
    if independent:
        if "abund" in nodes:
            raise ValueError(
                f"{manifest['id']}: declared abundance-independent, but its abundance "
                f"column has {nodes['abund'].size} nodes"
            )
        nodes["abund"] = np.array(FREE_ABUNDANCE_SPAN)
        arrays = {k: np.repeat(v[..., None], 2, axis=-1) for k, v in arrays.items()}
    return nodes, arrays, dict(reversed(list(fixed.items()))), independent


def _independent_kwargs(manifest: dict, element: str, independent: bool) -> dict:
    kw = _common_kwargs(manifest, element)
    if independent:
        # Nothing is solved and nothing is converted: Delta is read directly and
        # the query abundance only rides along.
        kw.update(indexed_by="lte", abund_kind="logeps", abund_solar=None,
                  abund_zero_point=None, abund_independent=True)
    return kw


# ---------------------------------------------------------------- reader


@register_reader("tabular")
def read_tabular(manifest: dict, path: Path) -> list:
    """Build sources from a text table described by ``manifest``."""
    element = normalize_element(manifest["element"])
    kind = manifest.get("kind", "delta")
    ion_col = manifest.get("ion_column")

    if kind == "ew_pair":
        return _read_ew_pair(manifest, path, element)

    variants = manifest.get("variants") or [
        {"column": manifest["columns"]["value"], "flavor": manifest.get("flavor", "NLTE - LTE")}
    ]
    value_columns = {str(i): v["column"] for i, v in enumerate(variants)}
    blocks = _prepare(manifest, path, value_columns, element, ion_col)

    sources = []
    for ion, nodes, scattered, scat_nodes, labels, arrays in blocks:
        nodes, arrays, fixed, independent = _shape_axes(nodes, scattered, arrays, manifest)
        lines = _make_lines(labels, manifest.get("wavelengths", {}))
        clip = tuple(manifest.get("clip_axes", ())) + (("abund",) if independent else ())
        axis_names = tuple(scattered) + tuple(nodes)

        for vi, variant in enumerate(variants):
            values = arrays[str(vi)]
            grids = [
                make_grid(nodes, values[i], scattered_names=scattered,
                          scattered_nodes=scat_nodes, clip_axes=clip,
                          min_valid_weight=manifest.get("min_valid_weight"),
                          max_node_distance=manifest.get("max_node_distance"))
                for i in range(len(labels))
            ]
            suffix = variant.get("id_suffix", "")
            sid = manifest["id"] + (f"_{suffix}" if suffix else "")
            if ion_col:
                sid = f"{sid}_{element.lower()}{ion}"
            sources.append(GridSource(
                id=sid,
                ion=ion,
                flavor=variant.get("flavor", manifest.get("flavor", "NLTE - LTE")),
                kind="delta",
                axis_names=axis_names,
                lines=lines,
                grids=grids,
                fixed=fixed,
                **_independent_kwargs(manifest, element, independent),
            ))
    return sources


def _read_ew_pair(manifest: dict, path: Path, element: str) -> list:
    """Grids publishing LTE and non-LTE equivalent widths per line as columns."""
    if "ew_lte" in manifest["columns"]:
        return _read_ew_pair_long(manifest, path, element)
    df = read_table(path, manifest.get("read", {}))
    pattern = manifest.get("ew_pattern", {})
    if pattern:
        lte_re, nlte_re = re.compile(pattern["lte"]), re.compile(pattern["nlte"])
        found = {}
        for col in df.columns:
            for tag, rx in (("lte", lte_re), ("nlte", nlte_re)):
                m = rx.fullmatch(str(col))
                if m:
                    found.setdefault(m.group("line"), {})[tag] = col
        entries = [
            {"label": k, "lte": v["lte"], "nlte": v["nlte"]}
            for k, v in found.items() if "lte" in v and "nlte" in v
        ]
    else:
        entries = manifest["ew_lines"]

    value_columns = {}
    for i, e in enumerate(entries):
        value_columns[f"lte{i}"] = e["lte"]
        value_columns[f"nlte{i}"] = e["nlte"]
    df = _clean_values(df, manifest, value_columns.values())
    df = _to_native_abundance(df, manifest, element)

    nodes, scattered, scat_nodes, _, arrays = _build_arrays(
        df, {**manifest, "columns": {k: v for k, v in manifest["columns"].items() if k != "line"}},
        value_columns,
    )
    clip = tuple(manifest.get("clip_axes", ()))
    lines = _make_lines([e["label"] for e in entries], manifest.get("wavelengths", {}))
    grids = [
        (
            make_grid(nodes, arrays[f"lte{i}"][0], scattered_names=scattered,
                      scattered_nodes=scat_nodes, clip_axes=clip,
                      min_valid_weight=manifest.get("min_valid_weight")),
            make_grid(nodes, arrays[f"nlte{i}"][0], scattered_names=scattered,
                      scattered_nodes=scat_nodes, clip_axes=clip,
                      min_valid_weight=manifest.get("min_valid_weight")),
        )
        for i in range(len(entries))
    ]
    return [GridSource(
        id=manifest["id"],
        ion=int(manifest.get("ion", 1)),
        flavor=manifest.get("flavor", "NLTE - LTE"),
        kind="ew_pair",
        axis_names=tuple(scattered) + tuple(nodes),
        lines=lines,
        grids=grids,
        abund_nodes=nodes["abund"],
        **_common_kwargs(manifest, element),
    )]


def _read_ew_pair_long(manifest: dict, path: Path, element: str) -> list:
    """Equivalent-width grids with one row per line (Lind et al. 2022).

    ``[columns]`` names ``ew_lte`` and ``ew_nlte`` -- log10 equivalent widths,
    as the matching solver expects -- alongside a ``line`` column.  Unlike the
    wide layout this goes through the same gridding and cache as delta tables,
    which matters for the 3.3M-row magnesium file.
    """
    cols = manifest["columns"]
    value_columns = {"lte": cols["ew_lte"], "nlte": cols["ew_nlte"]}
    (ion, nodes, scattered, scat_nodes, labels, arrays), = _prepare(
        manifest, path, value_columns, element
    )
    nodes, arrays, fixed, independent = _shape_axes(nodes, scattered, arrays, manifest)
    if independent:
        raise ValueError(f"{manifest['id']}: an equivalent-width grid needs an abundance axis")
    clip = tuple(manifest.get("clip_axes", ()))
    kw = dict(scattered_names=scattered, scattered_nodes=scat_nodes, clip_axes=clip,
              min_valid_weight=manifest.get("min_valid_weight"),
              max_node_distance=manifest.get("max_node_distance"))
    grids = [
        (make_grid(nodes, arrays["lte"][i], **kw), make_grid(nodes, arrays["nlte"][i], **kw))
        for i in range(len(labels))
    ]
    return [GridSource(
        id=manifest["id"],
        ion=ion,
        flavor=manifest.get("flavor", "NLTE - LTE"),
        kind="ew_pair",
        axis_names=tuple(scattered) + tuple(nodes),
        lines=_make_lines(labels, manifest.get("wavelengths", {})),
        grids=grids,
        abund_nodes=nodes["abund"],
        fixed=fixed,
        **_common_kwargs(manifest, element),
    )]
