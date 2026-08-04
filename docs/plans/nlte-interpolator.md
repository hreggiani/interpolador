# NLTE correction interpolator

## Context

`/Users/hreggiani/Trabalho/projetos/interpolador` currently contains only `grids/` — a collection of
published non-LTE abundance-correction tables in eight mutually incompatible formats (IDL `.sav`
structs, VizieR TSVs with three different delimiters, plain CSVs, whitespace tables inside tarballs).
Using any of them today means writing bespoke parsing and interpolation code per table, per project.

The goal is one Python package that loads every table behind a common interface and answers a single
question: *given (element, line, Teff, log g, [Fe/H], vmic, measured LTE abundance), what is the non-LTE
correction?* — returning an answer from **every** grid that covers that line, so different literature
sources can be compared directly.

Python env: `/Users/hreggiani/Trabalho/CODES/uv/new/bin/python` (3.14, numpy 2.3.5, scipy 1.16.3,
pandas 2.3.3, astropy 7.2). No numba — stay on numpy/scipy.

Decisions already made: library **and** CLI; corrections solved **self-consistently** in abundance;
EW-based grids (Al, Osorio Mg) included from the start; a query **returns all matching sources**.

## Grid inventory (verified by inspection)

| Source | Files | Kind | Axes | Abundance axis | Notes |
|---|---|---|---|---|---|
| Amarsi 2020 CSV | `grids/abcor2020_csv/*.csv` (22 = 11 elements × {LTE-scatt, no-scatt}) | Δ | teff(27), logg(13), feh(15), xfe(7; Li 23) | `xfe` = [X/Fe] | cols `teff,logg,feh,xfe,line,ew_nlte,ew_lte,abcor`; bad = `-1e30` (13–20%); line in Å (some as nm×10 e.g. `476.0`, `568.0` — needs per-file unit fix) |
| Amarsi C/O `.sav` | `abcor_{c1,o1}_{1d-1dl,3d-atl}.sav` | Δ | teff, logg, feh, vturb_base, vturb_top, lgeps | `LGEPS` = log ε | struct `grid`; `ABCOR` shape `(nline, nvt_top, nvt_base, nfeh, nlogg, nteff, nlgeps)` (IDL order reversed); bad = `-1e29` (up to 78%); `SOLAR` scalar carried |
| Amarsi Fe II `.sav` | `abcor_g_fe2_3dl-atl.sav` | Δ | same, but feh & vturb_top are **scalars** | log ε | 142 lines; degenerate axes need `atleast_1d` |
| Amarsi O MARCS | `oxygen_marcs.sav` | Δ (+`ABERR`, `EQW`) | teff(17), logg(9), feh(19), vturb(2), ofe(9) | `OFE` = [O/Fe] | 5 lines; different struct schema from the `abcor_*` ones |
| Amarsi 2016 Fe | `iron_abundancecorr/*.txt` (+2 still `.tar.gz`) | Δ | teff, logg, abnd, vmic — **sparse in (teff, logg)** | `abnd` = log ε | 8.2M rows, 2201 lines; **no [Fe/H] axis**; 54 of 104 (teff,logg) nodes populated, full product in vmic(3)×abnd(23); `clean` yes/no flag (28% yes) |
| Amarsi Si | `nlte_silicon.csv` | Δ | line(11), teff(9), logg(9), feh(12), vt(2), logeps(9 **per feh**) | log ε | logeps nodes track feh: `logeps − feh ∈ {7.01…9.01}` → reparametrise to `[Si/Fe]` to get a rectilinear grid |
| Nordlander & Lind 2017 Al | `NLTE_Al_{MARCS,mean3D}_*.sav` | **EW pair** | teff, logg, feh, abund(21) | offset grid `−2.06…1.94` | two structs `ltegrid`/`nltegrid`, `LOGEW` shape `(22 lines, nfeh, nlogg, nteff, nabund)`; 28–34% NaN |
| Osorio 2016 Mg | `mg1_osorio2016.tsv` | **EW pair** | Teff, logg, [Fe/H], Vturb, A(Mg) | log ε | VizieR, **`;`-delimited**, 109 header lines; 16 lines × {LTE,NLTE} columns of log(EW/λ) |
| Matsuno 2024 Mg | `mg_3d_nlte/asu.tsv` | Δ ×2 | Teff(7), logg(3), A(Mg)(4), vmic(3) | log ε | VizieR but **`,`-delimited**; **no [Fe/H] axis**; 2645 rows (slightly sparse); two columns `1N-1L` and `3N-1L`; line ids like `517nm(combined)` |

Three structural consequences drive the design: (1) sentinels differ per source and masked fractions
are large, so interpolation must be NaN-aware rather than assuming clean boxes; (2) two grids are not
rectilinear (Fe sparse in teff–logg, Si abundance axis sheared by feh); (3) two grids give EWs, not
corrections.

## Design

```
nlte/
  solar.py       # solar abundance refs (Asplund 2009/2021), Z/symbol tables
  interp.py      # NaN-aware linear interpolation: RectGrid + HybridGrid
  solve.py       # self-consistent Δ solver; EW-matching solver
  source.py      # GridSource dataclass/protocol + metadata
  readers/       # one module per format family (8 modules)
  registry.py    # lazy discovery + per-element/line index over ./grids
  api.py         # correct(), correct_frame()
  cli.py         # batch driver
tests/
```

### 1. `interp.py` — masked linear interpolation (the core)

`RectGrid(axes: list[np.ndarray], values: np.ndarray)`:
- locate the bracketing cell per axis via `np.searchsorted`, build the `2**n` corner weights;
- **drop NaN corners and renormalise the surviving weights**; return NaN if surviving weight
  < `min_valid_weight` (default 0.5). Return `n_bad_corners` alongside the value.
- Do not use `RegularGridInterpolator` directly — a single `-1e29` corner would poison results, and at
  78% masking in `abcor_c1_1d-1dl.sav` that is the common case, not the edge case.
- Vectorised over query points; out-of-range → NaN + `in_range=False` flag, with an opt-in
  `clip_axes` (intended for `vmic`, where grids are coarse and clamping is standard practice).

`HybridGrid(scattered_axes, regular_axes)` for the Fe tables: `scipy.spatial.Delaunay` over the 54
normalised (teff, logg) nodes → barycentric weights, tensored with linear weights in (vmic, abnd),
then the same drop-and-renormalise rule. Values stored as `[n_nodes, n_vmic, n_abnd]` per line.

The Si grid becomes rectilinear by using `[Si/Fe] = logeps − feh − A_sun(Si)` as the abundance axis;
handled in its reader, not the engine.

### 2. `solve.py`

- `solve_delta(delta_fn, a_lte, bounds)` — Δ is tabulated against the *model* (non-LTE) abundance, so
  fixed-point iterate `A ← a_lte + Δ(A)` from `A₀ = a_lte`; tol 1e-4 dex, max 30 iters, falling back to
  `scipy.optimize.brentq` on `f(A) = A − Δ(A) − a_lte` over the grid's abundance range when the
  iteration oscillates. Returns `(a_nlte, converged, n_iter)`. `mode='direct'` skips the loop for
  comparison work.
- `solve_ew_match(logew_lte, logew_nlte, a_lte)` for Al/Osorio: interpolate the LTE log EW curve at
  `a_lte` to get the target EW, then invert the NLTE curve `logEW(A)`. Mask NaN nodes, restrict to the
  longest strictly-increasing segment (strong lines saturate and flatten), `np.interp`, and flag
  `saturated` / `extrapolated` when the target falls outside it.

### 3. Abundance bookkeeping — `solar.py` + readers

Everything is converted to **A(X) = log ε** at the source boundary:
- `[X/Fe]` grids (Amarsi 2020 CSV `xfe`, `oxygen_marcs` `OFE`): `A = A_sun(X) + feh + xfe`.
- log ε grids (Si, Fe, Osorio, Matsuno, all `abcor_*.sav` `LGEPS`): used as-is.
- Al offset grid: `A = A_sun(Al) + feh + abund`, with `A_sun(Al)` a per-source metadata constant.
- Each source declares its **own** solar reference — the `.sav` files carry a `SOLAR` field
  (C 8.43/8.39, O 8.69/8.66, Fe 7.5); the CSV/TSV sources get Asplund 2009 in metadata. Never mix a
  global solar table into a source that ships its own.

> One convention to confirm during implementation: the Al `ABUND` axis (`−2.06 … 1.94`, step 0.2,
> zero-node at `−0.06`) is an offset grid whose zero point is not round — consistent with a solar
> A(Al) of 6.37 vs 6.43. The reader will expose `abund_zero_point` as metadata; validate by checking
> that solar-parameter corrections for 3961 Å reproduce Nordlander & Lind 2017's published values, and
> adjust the constant if not.

Grids without a `feh` axis (Fe 2016, Matsuno Mg) declare `axes` without it; the query's `feh` is then
unused and reported as ignored in the result, rather than silently dropped.

### 4. `registry.py` + `api.py`

`GridSource` metadata: `id`, `reference`, `element`, `ion`, `flavor` (`1D NLTE − 1D LTE`,
`<3D> NLTE − 1D LTE`, `3D NLTE − 1D LTE`, …), `scattering`, `lines_angstrom`, `axes`, `kind`
(`delta` | `ew_pair`), `solar`. All wavelengths normalised to Å at load (the `nm` string ids like
`'505.2nm'`, and the Amarsi-2020 CSV files whose `line` column is nm×10 for some elements).

Sources load lazily on first use — `readsav` on the 263 MB files is only ~0.2 s, but the 8.2M-row Fe
tables are slow to parse, so those get an `.npz` cache in `grids/.cache/` keyed by mtime, and the
reader auto-extracts `nmarcs_lmarcs.tar.gz` / `nmtd_lmtd.tar.gz` on first access.

```python
from nlte import correct
results = correct(element="Mg", wavelength=5711.09, teff=5777, logg=4.44,
                  feh=0.0, vmic=1.0, abund=7.60)          # abund = measured LTE log ε
```
Returns `list[Correction]` — one per source containing that line (per the "return all matches"
decision), each with `source, reference, flavor, scattering, wavelength_grid, delta, a_nlte,
converged, in_range, n_bad_corners, notes`. Line matching is by `|λ − λ_grid| ≤ tol` (default 0.5 Å);
blended entries such as `517nm(combined)` are parsed but excluded unless `include_blends=True`.
Helpers: `correct_frame(df, ...)` → tidy DataFrame, and `sources(element=...)` to introspect coverage.

### 5. Adding new tables (must stay cheap — new grids appear constantly)

New tables arrive regularly, and most of them are *not* new formats — they are another VizieR TSV or
another column layout over the same five axes. So the registry is driven by **declarative manifests**,
not by code, and only genuinely novel binary formats require a new reader.

**Common case — a tabular grid (CSV/TSV/whitespace), zero code.** Drop the file anywhere under
`grids/` next to a `<name>.toml` manifest; `registry.py` globs `grids/**/*.toml` at import and builds
the source from it. The generic `readers/tabular.py` handles delimiter, comment/skip rules, and column
mapping:

```toml
id        = "matsuno2024_mg"
reference = "Matsuno et al. 2024, A&A 688, A72"
element   = "Mg"; ion = 1
file      = "asu.tsv"
format    = "tabular"
delimiter = ","          # ";" for Osorio, whitespace for the Fe tables
skiprows  = 3            # or comment = "#"
[columns]                 # grid column -> canonical axis name
teff = "Teff"; logg = "logg"; abund = "A(Mg)"; vmic = "vmic"; line = "line"
[abundance]
kind = "logeps"           # or "xfe" / "offset"; "solar"/"zero_point" when needed
[[variants]]              # one entry per correction column in the file
name = "1D NLTE - 1D LTE"; column = "1N-1L"
[[variants]]
name = "<3D> NLTE - 1D LTE"; column = "3N-1L"
[quality]
sentinel = -1e30          # optional; mapped to NaN
wavelength_unit = "nm"    # normalised to Å on load
```

The tabular reader infers whether the axes form a full rectilinear product and picks `RectGrid` or
`HybridGrid` automatically, reporting which it chose (and which axes are sparse) via
`sources(id).describe()`. A manifest for an EW-pair table declares `kind = "ew_pair"` with
`lte_column`/`nlte_column` per line instead of `variants`, and routes to `solve_ew_match` unchanged.

**Rarer case — a new binary/bespoke format.** Write a reader function, decorate it with
`@register_reader("myformat")`, and manifests can then say `format = "myformat"`. Third-party packages
can register through the `nlte.readers` entry-point group without touching this repo, and
`nlte.register_source(GridSource(...))` allows building one entirely in memory.

Everything downstream — line matching, both solvers, `correct()`, the CLI — consumes `GridSource`
only, so a new table is queryable the moment its manifest exists, with no changes to `api.py`.
The eight existing sources ship as manifests themselves (with `format = "amarsi_sav"` etc. for the
IDL ones), so the extension path is the same path the built-ins use and cannot silently rot.
`docs/adding_a_table.md` records the recipe plus a `nlte-validate <manifest>` command that loads the
table, prints the inferred axes/coverage, and runs the node round-trip check from §1 of Verification.

### 6. `cli.py`

`nlte-correct input.csv -o out.csv` — input is one row per (star, line) measurement with
teff/logg/feh/vmic/element/wavelength/abund columns (`--col-map` for renames). Output defaults to long
format (one row per measurement × source); `--wide` pivots to `delta_<source_id>` columns. CSV and
FITS via astropy. Flags: `--sources`, `--tol`, `--mode {selfconsistent,direct}`, `--require-clean`
(Fe only), `--clip-vmic`.

## Verification

1. **Node round-trip, every source**: sample ~200 random grid nodes per source, query at exactly those
   coordinates with `mode='direct'`, assert the returned Δ equals the stored value to 1e-6. Catches
   axis-order mistakes in the transposed IDL arrays — the single most likely bug here.
2. **Masking**: assert queries inside cells whose corners are all sentinel return NaN with
   `in_range=True`, and that partially-masked cells renormalise (hand-checked case in
   `abcor_c1_1d-1dl.sav`).
3. **Solver**: for random queries assert `|a_nlte − delta(a_nlte) − a_lte| < 1e-4`; check
   self-consistent vs direct differ most for Li 6708 and metal-poor O 7774, as expected.
4. **EW grids**: verify Al and Osorio Mg reproduce a hand-computed EW match at a grid node, and that
   saturated strong lines (Mg b 5172/5183) raise the `saturated` flag rather than returning garbage.
5. **Physical sanity**: solar-parameter corrections have the right sign and rough magnitude against the
   source papers (O 777 nm ≈ −0.2 dex; Na D negative; Fe I positive and growing toward low [Fe/H]).
6. **End-to-end**: run the CLI over a ~50-row demo table spanning dwarfs/giants/metal-poor stars and
   confirm multi-source rows appear for Mg, O, C, Fe, Si; inspect a Δ-vs-[Fe/H] plot per source.
7. **Extensibility**: add a small hand-written table via manifest only (no code) in a test, and assert
   it becomes queryable through `correct()`; run `nlte-validate` over all shipped manifests in CI so a
   malformed or moved grid fails loudly rather than at query time.

Run: `/Users/hreggiani/Trabalho/CODES/uv/new/bin/python -m pytest tests/`
