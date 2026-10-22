# Integrate the NLiTE grids into the interpolator

## Context

The previous session downloaded ~30 papers' NLTE correction tables from the NLiTE list
(https://nlite.pythonanywhere.com/) into `grids/<author><year>_<el>/`, documented in
`grids/NLITE_SOURCES.md`. None has a `.toml` manifest, so the registry (which globs
`**/*.toml`) ignores them. Goal: make every *grid* among them queryable through
`correct()` / `nlte-correct`, alongside the existing 44 sources, without breaking those.

Most new data does not fit the tabular reader as it stands. The gaps are generic, so the
plan extends `nlte/readers/tabular.py` with a few manifest features rather than writing
per-paper readers, plus one new reader for the INASAN `.lst` block format.

## Scope

**Integrate (grids):**

| Source (folder) | Species | Route |
|---|---|---|
| INASAN DB rebuild (`alexeeva2014_na`, `mashonkina2013_mg`, `mashonkina2017_ca`, `sitnova2016_ti`, `mashonkina2011_fe`, `mashonkina2022_sr`, `mashonkina2019_ba`, `mashonkina2000_eu`) | Na I, Mg I, Ca I, Ti II, Fe I, Sr II, Ba II, Eu II | tabular; [X/Fe] axis; Fe/Mg/Ti abundance-independent (single [X/Fe] node) |
| Lind+ 2009 (`lind2009_li`) | Li I 6707.8, 6103.6 | tabular, CDS fixed-width, `indexed_by="lte"` (column is LTE A(Li); verify with its EW columns per docs/adding_a_table.md) |
| Lind+ 2022 (`lind2022_namgal`) | Na I, Mg I, Al I | **long-format ew_pair** (log EW LTE/NLTE per row), [X/Fe] + vmic axes, `cache=true` (Mg 3.3M rows); line ids → wavelengths via aliases from the paper's line list (arXiv source) |
| Spite+ 2012 (`spite2012_ca`) | Ca I, Ca II | tabular, 5 files concatenated, roman-numeral ion column, [Ca/Fe] axis {0, 0.3} |
| Merle+ 2011 (`merle2011_mgca`) | Mg I, Ca I, Ca II | tabular, one file per line (line from filename); [α/Fe] tracks [Fe/H] → abundance-independent |
| Korotin+ 2015 (`korotin2015_ba`) | Ba II | tabular, CDS fixed-width (`-2.0a` flag), **wide→long melt** of `d<line><[Ba/Fe]>` columns |
| Gallagher+ 2020 (`gallagher2020_ba`) | Ba II | tabular over the `g<logg>mm<feh>_*.txt` summary files (params from filename), `scattered=["teff","logg"]`; variants `3D_corr` (3D NLTE − 1D LTE) and `1D_corr` (1D NLTE − 1D LTE), indexed by the model abundance (default solve); [X/Fe] axis, solar 2.27 (from the files) |
| Sitnova+ 2022 (`sitnova2022_zn`) | Zn I, Zn II | **new reader `inasan_lst`** (Teff blocks × logg columns × [Fe/H] rows, EW/Δ row pairs, −1 = missing); two sources: MARCS and LL hot-star grids |
| Takeda+ 2005 (`takeda2005_s`) | S I, Zn I | tabular over the PASJ TSVs; variant Δ₀ (adopted h=0); abundance tied to [Fe/H] → abundance-independent; ξ=2 collapsed to fixed |
| Takeda+ 2002 (`takeda2002_k`) | K I 7698 | tidy CSV regenerated from the LaTeX; [K/Fe] axis (nodes −0.3/0/+0.3 at [Fe/H]=0, +0.2/+0.5/+0.8 below) |
| Bergemann+ 2012 (`bergemann2012_sr`) | Sr I 4607, Sr II 4077, 10036 | tidy CSV regenerated from the LaTeX; abundance-independent |
| Bergemann+ 2013 (`bergemann2013_si`) | Si I J-band (RSG) | tidy CSV regenerated from the LaTeX; abundance-independent |

**Leave out (state why in the manifest-less folders' entry in NLITE_SOURCES.md):**
per-star tables (Skúladóttir 2015, Shi 2018, Xu 2022, Andrievsky 2018); Bergemann 2019 Mn
(2 models), Mashonkina 2013 cool table (3 models), Sitnova 2016 Ti appendix (single
logg/[Fe/H], hot stars — superseded by the INASAN Ti II rebuild); Amarsi+ 2022 Fe
(sklearn-1.0.2 pickles that need per-line Elow/Eup/log gf inputs, not a wavelength —
needs a line list and unpickling third-party code; separate decision).

## Code changes

All in `nlte/readers/tabular.py` unless noted; each is a manifest key documented in
`docs/adding_a_table.md`.

1. **Multiple files** — `files = ["a.dat", ...]` or a glob (`files = "g*mm*_5601_*.txt"`),
   resolved relative to the manifest; frames concatenated before gridding.
   `[file_params]` regex with named groups adds columns from the filename
   (`line`, `logg`, `feh`), with optional `scale` per group (e.g. `g45` → 4.5,
   `mm10` → −1.0). `datafile` for a manifest without `file` is the manifest path
   (`nlte/registry.py:110`), so the reader resolves against `path.parent`.
   Cache key (`_cache_file`) must hash every file's mtime/size.
2. **CDS fixed-width read** — `[read] format = "cds"`, `readme = "ReadMe"`: parse the
   ReadMe "Byte-by-byte Description of file:" block matching the file name (incl.
   `tablea[12345].dat`, `*.dat`), build `colspecs`/`names`, read with `pd.read_fwf`.
   Covers Lind 2009, Spite, Merle, Korotin, Lind 2022 (and the `-2.0a` flag column).
3. **Wide→long melt** — `[melt] pattern = "d(?P<line>\\d)(?P<abund>[+-]\\d\\.\\d)"`,
   `value = "delta"`, optional `line_labels = {1 = "4554", ...}`: stacks matching
   columns into `line`/`abund`/`delta` columns before gridding.
4. **Collapse length-1 axes** into `GridSource.fixed` (as `idl_sav._arrange` does);
   today `RectGrid` would raise. If the *abundance* axis collapses (or the manifest
   omits `abund` with `[abundance] independent = true`), build an
   **abundance-independent** source: synthesise a 2-node abundance axis over a wide
   span with duplicated values, clip it, set `indexed_by="lte"` (no solve), and record
   the fixed value in `fixed`. Add `GridSource.abund_independent` (`nlte/source.py`) so
   `describe()` prints "abundance: not an axis" instead of the synthetic bounds, and
   `_evaluate_delta` (`nlte/api.py:226`) never flags the abundance as out of range.
5. **Long-format ew_pair** — `kind = "ew_pair"` with `[columns] ew_lte`, `ew_nlte`,
   `line` (instead of `[ew_pattern]`); optional `[read] log_ew = true` when values are
   already log10. Reuses `_build_arrays` and the existing `solve_ew_match` path.
6. **Roman-numeral ion column** — `ion_column` values like `Ca I`/`Ca II`: parse with
   `nlte.solar.parse_species` instead of the `(\d+)$` regex in `_prepare`.
7. **New reader `inasan_lst`** — `nlte/readers/inasan_lst.py`, registered in
   `nlte/readers/__init__.py`; yields one source per ion; abundance-independent via
   the same helper as (4).

## Data preparation (scripts tracked in `tools/`, outputs stay in gitignored `grids/`)

- `tools/inasan_pull.py` — the INASAN rebuild script from last session (provenance).
- `tools/latex_tables.py` — regenerate tidy CSVs for Takeda 2002 K, Bergemann 2012 Sr,
  Bergemann 2013 Si from the arXiv LaTeX (arXiv ids recorded in the script); replaces
  the `.txt` files with `*.csv` that the tabular reader reads directly.
- Lind 2022 line list: wavelengths for the `Line` ids from the paper's arXiv source,
  written into the manifest as `[wavelengths.aliases]`.

## Manifests

One `.toml` per folder, headed by a provenance comment in the style of
`grids/reggiani2019_k.toml` (reference, origin URL, conventions, caveats). Solar
abundances for [X/Fe] axes are taken from each paper (INASAN: Mashonkina+ 2023;
Lind 2022; Korotin 2015) — verify each, fall back to `solar_table = "asplund2009"`
only where the paper says so. Ba/Sr INASAN notes record the dwarf/giant [X/Fe] union.

## Docs and bookkeeping

- `docs/adding_a_table.md`: the new manifest keys (files/file_params, cds read, melt,
  long ew_pair, abundance-independent, roman ions) and the `inasan_lst` format.
- `README.md`: source/element counts; regenerate `docs/line_coverage.md` / `.csv`
  (`nlte-lines -o docs/line_coverage.md`).
- `docs/references.bib`: add the new papers. **It has uncommitted user edits — read
  the diff first and append, never overwrite.**
- `grids/NLITE_SOURCES.md`: mark which folders are now manifested.
- Save this plan to `docs/plans/nlite-grids.md` and commit it (per project practice).

## Verification

- `nlte-validate` on each new manifest: loads, prints axes/coverage, node round-trip
  must be exact (also exercised by `tests/test_sources.py::test_every_source_loads_and_round_trips`).
- New tests (`tests/test_sources.py`, `tests/test_extensibility.py`): multi-file +
  filename params, CDS fwf parsing, melt, collapsed axes → `fixed`,
  abundance-independent source returns the same Δ at any abundance and is never
  "out of range" on abundance, long ew_pair, roman ions, `inasan_lst` parsing.
- Spot checks against known values: INASAN Fe I 5171.61 at 5000/2.0/−3.0 → +0.156
  (server value from last session); Lind 2009 vs Amarsi 2020 Li 6707.8 agree within
  ~0.05 dex where both cover; INASAN Ba II vs Gallagher 1D_corr and Korotin within
  ~0.1 dex (NLiTE paper's comparison); Gallagher 3D_corr at the solar model
  reproduces the per-model file value.
- Full suite: `pytest` (and `NLTE_TEST_ALL=1 pytest` for the big grids, marking
  Lind 2022 Mg as big in `tests/conftest.py::manifest_is_big` if needed); existing
  sources' results unchanged.
- `nlte-correct teste_interpolador_in.csv` end to end: new sources appear, no errors.

## Commits

1. Plan doc. 2. Reader features + tests + docs. 3. `inasan_lst` reader. 4. Manifests
(INASAN; CDS family; Gallagher; Takeda/Bergemann tidy CSVs + `tools/`). 5. README,
line coverage, references.
