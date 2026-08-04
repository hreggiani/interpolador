# nlte

Interpolation over published non-LTE abundance-correction grids, behind one
interface. Eight incompatible on-disk formats — IDL `.sav` structs, VizieR
exports with three different delimiters, plain CSVs, whitespace tables inside
tarballs — become 40 queryable sources covering 12 elements.

Several elements are covered by more than one grid, computed with different
model atmospheres and assumptions, so a query returns **every** source that
covers the line rather than picking a winner.

```python
from nlte import correct

for c in correct("O", 7771.94, abund=8.69, teff=5777, logg=4.44, feh=0.0, vmic=1.0):
    print(f"{c.source:22s} {c.flavor:26s} {c.delta:+.3f}")
```

```
amarsi_o1_1d           1D NLTE - 1D LTE           -0.134
amarsi_o1_3d           3D NLTE - 1D LTE (ATLAS)   -0.176
amarsi2016_o1_marcs    1D NLTE - 1D LTE           -0.138
```

`abund` is the **measured LTE abundance** as A(X) = log ε. Each result carries
`delta`, `a_nlte`, `ok`, `in_range`, `valid_weight`, `converged`, the reference,
and a `note` explaining any failure.

## Batch use

```bash
nlte-correct measurements.csv -o corrected.csv --keep star
nlte-correct measurements.csv -o wide.csv --wide          # one column per source
```

The input needs `element`, `wavelength`, `abund` and whatever stellar parameters
the grids require (`--col-map abund=A_Fe,vmic=xi` renames them). Output is one
row per measurement and source. See `examples/demo_measurements.csv`.

```bash
nlte-validate                    # load every manifest, report coverage, verify axes
nlte-lines -o coverage.md        # every line that can be corrected, per element
```

## What it does that a lookup table does not

**Corrections are solved self-consistently.** Published Δ = A(NLTE) − A(LTE) is
tabulated against the *model* (non-LTE) abundance, but a measurement gives you
the LTE one. Reading Δ straight off at A(LTE) is only right where Δ is flat in
abundance; otherwise the answer has to satisfy `A_nlte − Δ(A_nlte) = A_lte`.
Pass `mode="direct"` to compare against the naive lookup.

**Masked nodes are handled, not propagated.** These grids are heavily masked —
`abcor_c1_1d-1dl.sav` is 78% sentinel values. Invalid corners are dropped and
the surviving interpolation weights renormalised, with `valid_weight` reporting
how much weight survived. A cell that is mostly missing returns NaN and says so
rather than quietly averaging two corners out of sixteen.

**Correction conventions are per grid, not assumed.** Most tables index Δ by the
model abundance and get the self-consistent solve; the Reggiani 2019 potassium
grid indexes it by the LTE abundance and is read directly. Which convention a
table uses can be settled from its own equivalent-width columns — see
[docs/adding_a_table.md](docs/adding_a_table.md).

**Grids that publish equivalent widths are usable.** The aluminium and Osorio
magnesium grids give LTE and non-LTE EWs, not corrections; those are derived by
matching equivalent widths, and saturated lines are flagged instead of
extrapolated.

**Awkward geometries load correctly.** The iron tables sample 54 of 104
(teff, log g) pairs and are triangulated in that plane; the silicon grid's
abundance axis shifts with [Fe/H] and is reparametrised to [Si/Fe] to become
rectilinear.

## Coverage

| Element | Sources |
|---|---|
| Li, Mn, Ba, Ca, Al (6699/7835) | Amarsi et al. 2020, with and without scattering |
| K | Amarsi 2020; Reggiani et al. 2019 (resonance doublet + 1.5 µm lines) |
| C | Amarsi 2020; Amarsi et al. 2019 1D and 3D |
| O | Amarsi 2020; Amarsi et al. 2019 1D and 3D; Amarsi et al. 2016 MARCS |
| Na | Amarsi 2020 |
| Mg | Amarsi 2020; Osorio et al. 2016 (EW-matched); Matsuno et al. 2024 1D and 3D |
| Al (3961 etc.) | Nordlander & Lind 2017, MARCS and ⟨3D⟩ (EW-matched) |
| Si | Amarsi 2020; Amarsi & Asplund 2017 |
| Fe I, Fe II | Amarsi et al. 2016, three atmosphere combinations; Amarsi et al. 2019 Fe II 3D LTE |

**2338 lines over 13 species.** [docs/line_coverage.md](docs/line_coverage.md)
lists every one with the grids that cover it, and
[docs/line_coverage.csv](docs/line_coverage.csv) is the machine-readable version;
both are regenerated with `nlte-lines`. From Python, `nlte.line_index()` returns
the same table and `nlte.sources()` lists the grids with axes and coverage.

## Install

```bash
pip install -e .
```

Needs numpy, scipy and pandas (astropy only for FITS input). Grid data is not
distributed with the package: point `$NLTE_GRIDS` at your grid directory, or
keep it in `grids/` beside the package. Each manifest names the exact file it
expects.

## Adding a table

New grids usually need a TOML manifest and no code. See
[docs/adding_a_table.md](docs/adding_a_table.md), and
[docs/plans/nlte-interpolator.md](docs/plans/nlte-interpolator.md) for the
design and the full grid inventory.

## Caveats

- The Nordlander & Lind aluminium `ABUND` axis is an offset grid whose zero node
  sits at −0.06 rather than 0. `zero_point = 6.43` in its manifest encodes the
  assumed solar reference; a shift there moves the abundance at which Δ is read
  by the same amount, which matters only where Δ varies steeply.
- `amarsi_fe2_3d` is a **3D LTE − 1D LTE** correction, not a non-LTE one.
- Some published nodes are masked in places you would not expect: Amarsi 2020's
  Mg 5711 grid has no value at solar parameters and [Mg/Fe] = 0, so that source
  returns `ok=False` there while Osorio and Matsuno answer.

## Tests

```bash
python -m pytest tests/            # NLTE_TEST_ALL=1 to include the 2.4 GB iron tables
```

The load-bearing test re-queries every grid at its own nodes: the IDL cubes
arrive with axes reversed, and a silent transpose error would produce
plausible-looking corrections that are wrong everywhere.
