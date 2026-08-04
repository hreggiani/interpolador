# Adding a new correction table

New non-LTE grids appear regularly, and most are not new *formats* — they are
another VizieR export or another column layout over the same handful of axes.
So the usual answer is: **write a manifest, no Python.**

Put the data file anywhere under the grid root (`grids/`, or `$NLTE_GRIDS`) with
a `.toml` manifest beside it. The registry globs `**/*.toml` and loads lazily,
so a new file costs nothing until something queries that element.

Check your work with:

```bash
nlte-validate grids/path/to/your.toml
```

which prints the axes, bounds, line count and node coverage it inferred, then
re-queries the grid at its own nodes to prove the axes were assembled correctly.
Once it loads, regenerate the line list so the new grid shows up there too:

```bash
nlte-lines -o docs/line_coverage.md
```

## A tabular grid

```toml
id        = "matsuno2024_mg"          # source id; must be unique
format    = "tabular"
reference = "Matsuno et al. 2024, A&A 688, A72"
element   = "Mg"
ion       = 1
file      = "asu.tsv"                 # relative to the manifest
kind      = "delta"                   # "delta" | "ew_pair"
flavor    = "1D NLTE - 1D LTE"        # what the correction actually is
scattering = false                    # optional, if the source distinguishes it
clip_axes = ["vmic"]                  # clamp instead of rejecting out-of-range
notes     = "No [Fe/H] axis."         # shown by nlte-validate and sources()

[read]
delimiter = ","                       # "," ";" "tab" "whitespace", or a literal
comment   = "#"
skiprows  = 0
# header  = "none"                    # plus names = [...] for headerless files

[columns]                             # canonical axis -> column in the file
teff  = "Teff"
logg  = "logg"
feh   = "[Fe/H]"                      # omit if the grid has no metallicity axis
vmic  = "vmic"
abund = "A(Mg)"
line  = "line"                        # omit for a single-line table (set `line = 5711.0`)
value = "abcor"                       # or use [[variants]], below

[abundance]
kind   = "logeps"                     # "logeps" | "xfe" | "offset"
# stored = "logeps"                   # if the column is log eps but the axis is [X/Fe]
# solar  = 7.51                       # or solar_table = "asplund2009"
# zero_point = 6.43                   # for an "offset" axis
# indexed_by = "nlte"                 # "lte" if Delta is tabulated against A(LTE)

[wavelengths]
unit = "nm"                           # "angstrom" | "nm" | "auto"
# aliases."777.0" = [7771.94, 7774.17, 7775.39]   # blended entry -> its components

[quality]
sentinel    = -1e30                   # mapped to NaN
flag_column = "clean"                 # optional quality flag
flag_good   = "yes"

[[variants]]                          # one source per correction column
id_suffix = "1n1l"
flavor    = "1D NLTE - 1D LTE"
column    = "1N-1L"
[[variants]]
id_suffix = "3n1l"
flavor    = "3D NLTE - 1D LTE"
column    = "3N-1L"
```

### Things worth knowing

**Missing rows are fine.** Rows are scattered into an array indexed by the axes;
combinations absent from the file stay NaN, and interpolation renormalises around
them. You do not need a complete rectangle.

**Grids sparse in a plane** — the iron tables sample 54 of 104 (teff, logg) pairs
— need `scattered = ["teff", "logg"]`. Those axes are then triangulated instead
of gridded, and the rest stays rectilinear.

**Abundance axes that shift with [Fe/H].** If the file's abundance column is log
eps but its nodes track metallicity (the silicon grid runs log eps − [Fe/H] =
7.01 … 9.01), set `kind = "xfe"` with `stored = "logeps"`. The reader converts
the column to [X/Fe], which *is* rectilinear; queries still arrive as log eps.
Getting this wrong is not loud — the grid loads either way — so check the
coverage figure `nlte-validate` prints: silicon reads 100% when parametrised
correctly and 29% when not.

**Splitting by ionisation stage:** `ion_column = "species"` yields one source per
ion, with the ion parsed from values like `Fe1`, `Fe2`.

**Which abundance is Delta indexed by?** Most published grids tabulate Δ against
the *model* (non-LTE) abundance, so a measured LTE abundance has to be solved
for — that is the default, `indexed_by = "nlte"`. Some tabulate against the LTE
abundance instead (the potassium grid's column is literally `abund_LTE`); set
`indexed_by = "lte"` and Δ is read directly, with the solve mode ignored for
that source.

If the table ships `ew_lte` and `ew_nlte` alongside the correction you can settle
this from the data rather than guessing. Take one (Teff, log g, [Fe/H], vmic,
line) block and test both inversions against the tabulated Δ:

```python
A, lte, nlte, abcor = block.abund, np.log10(block.ew_lte), np.log10(block.ew_nlte), block.abcor
indexed_by_lte  = np.interp(lte, nlte, A) - A     # solve ew_nlte(A') = ew_lte(A)
indexed_by_nlte = A - np.interp(nlte, lte, A)     # solve ew_lte(A'') = ew_nlte(A)
```

Whichever reproduces `abcor` with the smaller RMS is the convention. For the
potassium grid this gave 0.023 dex against 0.058, which is decisive.

**Irregular sampling that blocks interpolation:** `min_valid_weight` (default
0.5) sets how much of a cell's interpolation weight must land on valid nodes.
Grids whose abundance nodes shift between metallicities, or that are missing
scattered points, leave many cells valid on one side only and get rejected.
Lower it per source — and justify the number by measuring, as
`grids/reggiani2019_k.toml` does, rather than picking one that makes the output
look full. A query can override it with `min_valid_weight=`.

**Slow files:** set `cache = true` to store the gridded arrays in `.cache/` next
to the data, keyed by file mtime and manifest content. The iron tables go from
about 60 s to about 5 s.

**Archives:** if the manifest names `foo.txt` and only `foo.tar.gz` is present,
it is extracted on first use.

## Several tables in one manifest

`[[tables]]` describes a family that differs in only a few keys — the 22 Amarsi
2020 CSVs differ by file, element and scattering flag. Each entry overrides the
top-level keys:

```toml
id = "amarsi2020"
format = "tabular"
# ... shared settings ...

[[tables]]
id = "amarsi2020_mg1"
element = "Mg"
file = "mg1_abcor.csv"
scattering = false
```

## Grids that publish equivalent widths

Set `kind = "ew_pair"` and give a regex pairing the LTE and non-LTE columns per
line. Corrections are then derived by matching equivalent widths, and saturated
lines are flagged rather than extrapolated:

```toml
kind = "ew_pair"

[ew_pattern]
lte  = "(?P<line>[0-9]+)LTE"
nlte = "(?P<line>[0-9]+)NLTE"
```

Use `[[ew_lines]]` entries (`label`, `lte`, `nlte`) instead if the column names
do not follow a pattern.

## Grids that are not tables at all

Some grids ship trained models instead of a cube of numbers. `format =
"canocchi_ffnn"` handles the Canocchi et al. 2026 sodium networks, and its
reader shows the pattern: anything exposing `__call__(points) -> InterpResult`
and `bounds()` plugs into the solvers, the query API and the CLI unchanged. Set
`geometry_label` and leave `values = None` so coverage reporting knows there are
no nodes to count.

If a model-backed grid indexes its correction by the LTE abundance, as that one
does, set `indexed_by = "lte"` on the source.

## A genuinely new format

Write a reader returning `GridSource` objects and register it:

```python
from nlte import register_reader
from nlte.source import GridSource, Line, make_grid

@register_reader("myformat")
def read_myformat(manifest: dict, path) -> list[GridSource]:
    ...
    return [GridSource(
        id=manifest["id"], reference=..., element=..., ion=1,
        flavor="1D NLTE - 1D LTE", kind="delta",
        axis_names=("teff", "logg", "feh", "abund"),
        lines=[Line(wavelength=5711.09, label="5711")],
        grids=[make_grid(axes, values)],       # one interpolator per line
    )]
```

Manifests can then say `format = "myformat"`. A separate package can ship readers
through the `nlte.readers` entry-point group without touching this repository,
and `nlte.register_source(GridSource(...))` adds one built entirely in memory.

Conventions a reader must honour:

- wavelengths in **Angstrom**, `axis_names` ending in `"abund"`;
- sentinels mapped to **NaN** (`make_grid` treats any non-finite node as missing);
- length-1 axes collapsed into `fixed` rather than kept as degenerate axes;
- `alias` if one query parameter feeds two axes (the Amarsi `.sav` grids carry
  separate microturbulence axes for the non-LTE and LTE sides, both fed by `vmic`).
