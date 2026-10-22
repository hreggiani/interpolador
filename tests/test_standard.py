"""The standard output: fixed columns, recommended sources only."""

import warnings

import numpy as np
import pandas as pd
import pytest

from nlte import STANDARD_SOURCES, correct_frame, standard_frame
from nlte.standard import STANDARD_COLUMNS, paper_label, species_code

SUN = dict(teff=5777, logg=4.44, feh=0.0, vmic=1.0)
HOT_DWARF = dict(teff=7000, logg=4.0, feh=0.0, vmic=2.0)


def test_species_code_is_atomic_number_plus_stage():
    assert species_code("Fe", 2) == 26.1
    assert species_code("Fe", 1) == 26.0
    assert species_code("Ba", 2) == 56.1
    assert np.isnan(species_code("Fe", None))


def test_paper_label_drops_the_journal():
    assert paper_label("Amarsi et al. 2019, A&A 630, A104") == "Amarsi et al. 2019"
    assert paper_label("Mashonkina et al. 2022, MNRAS 509, 3626 (via INASAN_NLTE)") \
        == "Mashonkina et al. 2022"


def _standard(rows, registry, **kw):
    frame = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = correct_frame(frame, registry=registry, **kw)
    return standard_frame(full, frame, registry=registry, columns=kw.get("columns"))


@pytest.mark.grids
def test_recommended_sources_all_load(registry):
    optional = {"wang2021_li", "canocchi2026_na_3dnlte"}       # need extra packages
    for species, slots in STANDARD_SOURCES.items():
        ids = {s.id for s in registry.for_element(species)}
        wanted = {i for ids_ in slots for i in ids_} - optional
        assert wanted <= ids, (species, wanted - ids)


@pytest.mark.grids
def test_columns_are_exactly_the_standard_set(registry):
    std = _standard([{"id": "Sun", "species": "OI", "wavelength": 7771.94,
                      "abund": 8.69, **SUN}], registry)
    assert tuple(std.columns) == STANDARD_COLUMNS


@pytest.mark.grids
def test_one_row_per_recommended_source(registry):
    std = _standard([{"id": "Sun", "species": "FeI", "wavelength": 5232.94,
                      "abund": 7.50, **SUN}], registry)
    assert list(std.source) == ["Amarsi et al. 2022", "Amarsi et al. 2016"]
    assert list(std.flavor) == ["3D NLTE - 1D LTE", "1D NLTE - 1D LTE"]
    assert (std.species == 26.0).all()
    assert std["DeltaA(X)"].notna().all()
    np.testing.assert_allclose(std["A(X)_corr"] - std["A(X)"], std["DeltaA(X)"])
    # Atomic data from the Amarsi 2022 line list, on every row of the line.
    np.testing.assert_allclose(std.ep, 2.93987)
    np.testing.assert_allclose(std.gf, -0.057)


@pytest.mark.grids
def test_a_failed_recommended_source_stays_as_nan(registry):
    # Amarsi 2022 covers dwarfs only, so a giant gets NaN there and a value
    # from Amarsi 2016.
    giant = dict(teff=4800, logg=2.5, feh=-0.5, vmic=1.5)
    std = _standard([{"id": "g", "species": "FeI", "wavelength": 5232.94,
                      "abund": 7.0, **giant}], registry)
    by_source = std.set_index("source")
    assert np.isnan(by_source.loc["Amarsi et al. 2022", "DeltaA(X)"])
    assert np.isfinite(by_source.loc["Amarsi et al. 2016", "DeltaA(X)"])


@pytest.mark.grids
def test_species_without_a_recommended_source_is_one_nan_row(registry):
    std = _standard([
        {"id": "Sun", "species": "MgI", "wavelength": 5711.09, "abund": 7.60, **SUN},
        {"id": "Sun", "species": "ScII", "wavelength": 4314.08, "abund": 3.1, **SUN},
    ], registry)
    assert len(std) == 2
    assert list(std.species) == [12.0, 21.1]
    assert std[["A(X)_corr", "DeltaA(X)"]].isna().all().all()
    assert (std.source == "").all()


@pytest.mark.grids
def test_one_paper_split_across_tables_is_one_row(registry):
    # Sitnova 2022 Zn: MARCS for FGK stars, LLmodels above 6600 K.
    std = _standard([
        {"id": "Sun", "species": "ZnI", "wavelength": 4810.528, "abund": 4.56, **SUN},
        {"id": "hot", "species": "ZnI", "wavelength": 4810.528, "abund": 4.56, **HOT_DWARF},
    ], registry)
    assert len(std) == 2
    assert (std.source == "Sitnova et al. 2022").all()
    assert std["DeltaA(X)"].notna().all()


@pytest.mark.grids
def test_input_atomic_data_wins_over_the_grid_line_list(registry):
    std = _standard([{"star": "Sun", "species": "FeI", "wavelength": 5232.94,
                      "abund": 7.50, "elo": 2.94, "log_gf": -0.06, **SUN}],
                    registry, columns={"gf": "log_gf"})
    assert (std.id == "Sun").all()
    np.testing.assert_allclose(std.ep, 2.94)
    np.testing.assert_allclose(std.gf, -0.06)


@pytest.mark.grids
def test_bare_label_takes_its_stage_from_the_grid(registry):
    std = _standard([{"id": "Sun", "species": "O", "wavelength": 7771.94,
                      "abund": 8.69, **SUN}], registry)
    assert list(std.source) == ["Amarsi et al. 2019"] * 2
    assert (std.species == 8.0).all()


@pytest.mark.grids
def test_cli_writes_the_standard_output_beside_the_full_one(tmp_path, registry):
    from nlte.cli import main_correct

    src = tmp_path / "in.csv"
    src.write_text(
        "id,species,wavelength,abund,teff,logg,feh,vmic\n"
        "Sun,OI,7771.94,8.69,5777,4.44,0.0,1.0\n"
        "Sun,Sc,4314.08,3.1,5777,4.44,0.0,1.0\n"
    )
    out = tmp_path / "corrected.csv"
    assert main_correct([str(src), "-o", str(out), "--wide"]) == 0

    std = pd.read_csv(tmp_path / "corrected_std.csv")
    assert tuple(std.columns) == STANDARD_COLUMNS
    assert len(std) == 3                       # two O I sources, one empty Sc row
    assert len(pd.read_csv(out)) == 2          # the full output, still wide


@pytest.mark.grids
def test_barium_at_solar_metallicity_comes_from_gallagher(registry):
    # Mashonkina & Belyaev 2019 stops at [Fe/H] = -2; Gallagher 2020 reaches 0.
    std = _standard([{"id": "Sun", "species": "BaII", "wavelength": 6496.9,
                      "abund": 2.18, **SUN}], registry)
    assert list(std.flavor) == ["1D NLTE - 1D LTE", "3D NLTE - 1D LTE", "1D NLTE - 1D LTE"]
    by_row = std.set_index(["source", "flavor"])["DeltaA(X)"]
    assert np.isnan(by_row[("Mashonkina & Belyaev 2019", "1D NLTE - 1D LTE")])
    assert np.isfinite(by_row[("Gallagher et al. 2020", "3D NLTE - 1D LTE")])
    assert np.isfinite(by_row[("Gallagher et al. 2020", "1D NLTE - 1D LTE")])
