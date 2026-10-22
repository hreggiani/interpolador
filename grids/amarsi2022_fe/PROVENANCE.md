# Amarsi et al. 2022 iron networks — provenance

Vendored from **https://github.com/sliljegren/1L-3NErrors** (MIT, see `LICENSE`),
commit `707f3fe4e7b6c7fe267cc88554d6fe013d0b7778` (2022-09-28), which accompanies
Amarsi et al. 2022 ([arXiv:2209.13449](https://arxiv.org/abs/2209.13449)).

## What is here, and why it is not the upstream files

Upstream ships each model as a pickled `(StandardScaler, MLPRegressor)` pair.
Unpickling executes arbitrary code and ties the file to scikit-learn 1.0.2, so
instead of committing the pickles this directory holds only what evaluating the
networks needs:

| file | contents |
|---|---|
| `fe1_lt02.npz` | Fe I, Elo < 2 eV: layer weights, biases, scaler mean and scale |
| `fe1_gt02.npz` | Fe I, Elo >= 2 eV: likewise |
| `fe2.npz` | Fe II: likewise |
| `fe_lines.csv` | Fe I/Fe II line list: wavelength (air, A), Elo (eV), Eup (eV), log gf |

The networks are plain ReLU multilayer perceptrons with a linear output
(7 -> 200 -> 200 -> 200 -> 1), so the reader reproduces scikit-learn's own
`predict()` from these arrays bit-for-bit — verified in the tests against the
value upstream's README documents (`aberr = -0.136`).

Regenerate with:

```bash
python3 tools/vendor_amarsi2022_fe.py /path/to/1L-3NErrors
```

## The line list

The networks take per-line atomic data as *inputs* (Elo, Eup, log gf) rather
than identifying a line, so a line list is needed to make them reachable by
wavelength. `fe_lines.csv` is derived from the Amarsi et al. 2016 iron tables
already used by this project (`grids/iron_abundancecorr/`), which tabulate
`wave_nm`, `E_low` in eV and `loggf` for 2201 lines. Eup is not tabulated and is
computed exactly from the transition:

    Eup = Elo + 12398.419 / lambda(Angstrom)

Any other Fe line can be used by passing `elo`, `eup` and `loggf` to the query
directly; the bundled list is a convenience, not a limit of the grid.

## Sign convention

Upstream returns `aberr` = A(1D LTE) − A(3D NLTE) — the repository is named
"1L-3NErrors". This project's `delta` is A(NLTE) − A(LTE), so the reader negates
it. Checked against the Amarsi et al. 2016 grid on lines of both ions: the
signs agree once negated.
