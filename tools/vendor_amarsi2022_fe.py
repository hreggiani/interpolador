#!/usr/bin/env python3
"""Vendor the Amarsi et al. 2022 iron networks into grids/amarsi2022_fe/.

The upstream release (https://github.com/sliljegren/1L-3NErrors, MIT) ships each
model as a pickled (StandardScaler, MLPRegressor) pair.  Unpickling executes
code and binds the file to a scikit-learn version, so rather than committing the
pickles this script extracts what evaluating the network actually needs -- the
layer weights and the scaler's mean and scale -- into a plain .npz.  The reader
reproduces scikit-learn's own predict() from those arrays bit-for-bit.

Usage:  python3 tools/vendor_amarsi2022_fe.py /path/to/1L-3NErrors
"""

from __future__ import annotations

import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "fe1_lt02": "fe1_model_lt02.p",
    "fe1_gt02": "fe1_model_gt02.p",
    "fe2": "fe2_model.p",
}
#: The 7 network inputs, in order.
FEATURES = ("teff", "logg", "abund", "vmic", "elo", "eup", "loggf")

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "grids" / "amarsi2022_fe"
#: Iron line list already in the repo; supplies Elo (eV) and log gf per line.
LINE_SOURCE = HERE / "grids" / "iron_abundancecorr" / "nmtd_lmarcs.txt"
#: hc in eV A, for Eup = Elo + hc / lambda.
HC_EV_ANGSTROM = 12398.419


def convert_models(source: Path) -> None:
    for key, name in MODELS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # pickled under scikit-learn 1.0.2
            scaler, model = pickle.load(open(source / name, "rb"))
        if getattr(model, "activation", None) != "relu" or model.out_activation_ != "identity":
            raise SystemExit(f"{name}: unexpected network ({model.activation}/{model.out_activation_})")
        arrays = {
            "scaler_mean": np.asarray(scaler.mean_, dtype=float),
            "scaler_scale": np.asarray(scaler.scale_, dtype=float),
            "n_layers": np.array(len(model.coefs_)),
            "features": np.array(FEATURES, dtype=object),
        }
        for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
            arrays[f"weight_{i}"] = np.asarray(w, dtype=float)
            arrays[f"bias_{i}"] = np.asarray(b, dtype=float)
        path = OUT / f"{key}.npz"
        np.savez_compressed(path, **arrays)
        print(f"  {name:22s} -> {path.name}  ({path.stat().st_size/1e6:.2f} MB, "
              f"{len(model.coefs_)} layers)")


def convert_line_list() -> None:
    """Extract the Fe line list, so the vendored source needs no other grid."""
    table = pd.read_csv(LINE_SOURCE, sep=r"\s+", skiprows=1).drop_duplicates("wave_nm")
    lines = pd.DataFrame({
        "species": table.species.to_numpy(),
        "wavelength": (table.wave_nm.to_numpy() * 10.0).round(4),   # nm -> Angstrom
        "elo": table.E_low.to_numpy().round(6),                     # eV
        "loggf": table.loggf.to_numpy().round(4),
    })
    lines["eup"] = (lines.elo + HC_EV_ANGSTROM / lines.wavelength).round(6)
    lines = lines.sort_values(["species", "wavelength"]).reset_index(drop=True)
    path = OUT / "fe_lines.csv"
    lines.to_csv(path, index=False)
    print(f"  line list -> {path.name}  ({len(lines)} lines, "
          f"{(lines.species == 'Fe1').sum()} Fe I / {(lines.species == 'Fe2').sum()} Fe II)")


def main(argv) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    source = Path(argv[1]).expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"vendoring from {source}")
    convert_models(source)
    convert_line_list()
    shutil.copy(source / "LICENSE", OUT / "LICENSE")
    print(f"  LICENSE -> {OUT/'LICENSE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
