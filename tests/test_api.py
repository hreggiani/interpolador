"""Query behaviour, and physical sanity against published values."""

import numpy as np
import pandas as pd
import pytest

from nlte import correct, correct_frame, find_lines

pytestmark = pytest.mark.grids

SUN = dict(teff=5777, logg=4.44, feh=0.0, vmic=1.0)
METAL_POOR_DWARF = dict(teff=6300, logg=4.0, feh=-2.2, vmic=1.5)


def _by_source(results):
    return {c.source: c for c in results}


def test_returns_every_source_covering_a_line(registry):
    results = correct("O", 7771.94, abund=8.69, registry=registry, **SUN)
    ids = {c.source for c in results}
    assert {"amarsi_o1_1d", "amarsi_o1_3d", "amarsi2016_o1_marcs"} <= ids


def test_solar_oxygen_triplet_matches_the_literature(registry):
    # The solar 777 nm triplet non-LTE correction is about -0.15 dex, and two
    # independent grids should agree closely.
    results = _by_source(correct("O", 7771.94, abund=8.69, registry=registry, **SUN))
    one_d = [results[k].delta for k in ("amarsi_o1_1d", "amarsi2016_o1_marcs")]
    assert all(-0.25 < d < -0.05 for d in one_d), one_d
    assert abs(one_d[0] - one_d[1]) < 0.02


def test_iron_corrections_are_positive_and_grow_toward_low_metallicity(registry):
    solar = _by_source(correct("Fe", 5232.94, abund=7.50, registry=registry, **SUN))
    poor = _by_source(correct("Fe", 5232.94, abund=5.30, registry=registry, **METAL_POOR_DWARF))
    key = "amarsi2016_fe_nmarcs_lmarcs_fe1"
    if key not in poor:
        pytest.skip("iron tables not loaded")
    assert poor[key].delta > 0
    assert poor[key].delta > solar[key].delta


def test_aluminium_resonance_line_needs_a_large_positive_correction(registry):
    results = _by_source(correct("Al", 3961.52, abund=3.0, teff=4600, logg=1.4,
                                 feh=-2.7, vmic=1.9, registry=registry))
    corr = results["nordlander2017_al_marcs"]
    assert corr.ok and corr.delta > 0.4


def test_missing_parameters_are_reported_not_guessed(registry):
    results = correct("Si", 3905.52, abund=5.3, teff=6300, logg=4.0, feh=-2.2,
                      registry=registry)          # no vmic
    assert results
    for c in results:
        assert not c.ok and "missing parameter" in c.note


def test_unknown_line_returns_nothing(registry):
    assert correct("Fe", 1234.5, abund=7.5, registry=registry, **SUN) == []


def test_line_matching_respects_the_tolerance(registry):
    assert find_lines("Mg", 5711.09, tol=1.0, registry=registry)
    assert not find_lines("Mg", 5711.09, tol=0.001, registry=registry)


def test_self_consistent_and_direct_modes_differ_where_delta_varies(registry):
    kwargs = dict(element="Li", wavelength=6707.76, abund=2.2, teff=6100, logg=4.3,
                  feh=-2.0, vmic=1.3, registry=registry)
    exact = _by_source(correct(mode="selfconsistent", **kwargs))
    direct = _by_source(correct(mode="direct", **kwargs))
    shared = set(exact) & set(direct)
    assert shared
    for key in shared:
        if exact[key].ok and direct[key].ok:
            assert np.isfinite(exact[key].delta) and np.isfinite(direct[key].delta)


def test_self_consistent_result_satisfies_its_definition(registry):
    from nlte.api import find_lines as _find

    match = [m for m in _find("Fe", 5232.94, registry=registry)
             if m.source.id.endswith("nmarcs_lmarcs_fe1")]
    if not match:
        pytest.skip("iron tables not loaded")
    src, li = match[0].source, match[0].line_index
    corr = _by_source(correct("Fe", 5232.94, abund=5.30, registry=registry,
                              **METAL_POOR_DWARF))[src.id]
    delta_at_solution = src.delta_at(li, METAL_POOR_DWARF, corr.a_nlte, feh=-2.2).value[0]
    assert abs(corr.a_nlte - delta_at_solution - corr.a_lte) < 1e-3


def test_results_carry_provenance(registry):
    corr = correct("O", 7771.94, abund=8.69, registry=registry, **SUN)[0]
    assert corr.reference and corr.flavor and corr.line_label
    assert corr.element == "O" and corr.ion == 1


def test_correct_frame_produces_one_row_per_source(registry):
    frame = pd.DataFrame({
        "star": ["Sun", "Sun"],
        "element": ["O", "Mg"],
        "wavelength": [7771.94, 5711.09],
        "abund": [8.69, 7.60],
        "teff": [5777, 5777], "logg": [4.44, 4.44],
        "feh": [0.0, 0.0], "vmic": [1.0, 1.0],
    })
    out = correct_frame(frame, registry=registry, keep=["star"])
    assert len(out) > len(frame)
    assert {"source", "delta", "a_nlte", "ok", "note", "star"} <= set(out.columns)
    assert out.ok.any()
    assert set(out.row) == {0, 1}


def test_correct_frame_renames_columns(registry):
    frame = pd.DataFrame({
        "element": ["O"], "lambda": [7771.94], "A_X": [8.69],
        "teff": [5777], "logg": [4.44], "feh": [0.0], "xi": [1.0],
    })
    out = correct_frame(frame, registry=registry,
                        columns={"wavelength": "lambda", "abund": "A_X", "vmic": "xi"})
    assert out.ok.any()


def test_correct_frame_marks_lines_no_grid_covers(registry):
    frame = pd.DataFrame({
        "element": ["Fe"], "wavelength": [1234.5], "abund": [7.5],
        "teff": [5777], "logg": [4.44], "feh": [0.0], "vmic": [1.0],
    })
    out = correct_frame(frame, registry=registry)
    assert len(out) == 1 and not out.ok[0]
    assert "no grid" in out.note[0]


def test_line_index_lists_every_correctable_line(registry):
    from nlte import line_index

    frame = line_index(registry=registry)
    assert {"element", "ion", "wavelength", "n_sources", "blend", "sources"} <= set(frame.columns)
    assert (frame.wavelength > 1000).all()
    assert (frame.n_sources >= 1).all()
    # One row per physical line: a wavelength must not appear twice for a species.
    assert not frame.duplicated(["element", "ion", "wavelength"]).any()


def test_line_index_merges_the_same_line_across_grids(registry):
    from nlte import line_index

    mg = line_index("Mg", registry=registry)
    row = mg[np.isclose(mg.wavelength, 5711.09, atol=0.2)]
    assert len(row) == 1, "5711 should be one line, not one per grid"
    assert row.iloc[0].n_sources >= 3


def test_line_index_keeps_distinct_lines_from_one_grid_apart(registry):
    # The Amarsi oxygen grids list 6156, 6157, 6158 and 6160 -- neighbours closer
    # together than the matching tolerance, which must not be merged.
    from nlte import line_index

    o = line_index("O", registry=registry)
    near = o[(o.wavelength > 6150) & (o.wavelength < 6165)]
    assert len(near) == 4


def test_line_index_can_be_restricted(registry):
    from nlte import line_index

    assert set(line_index("Si", registry=registry).element) == {"Si"}


def test_species_labels_carry_the_ionisation_stage():
    from nlte.solar import parse_species

    assert parse_species("FeII") == ("Fe", 2)
    assert parse_species("MnI") == ("Mn", 1)
    assert parse_species("Fe 2") == ("Fe", 2)
    assert parse_species("Fe_II") == ("Fe", 2)
    assert parse_species("C") == ("C", None)
    # Case on the symbol is what separates silicon from S I.
    assert parse_species("Si") == ("Si", None)
    assert parse_species("SI") == ("S", 1)
    with pytest.raises(KeyError):
        parse_species("Xx")


def test_ion_stage_selects_the_right_sources(registry):
    # One manifest yields both stages of iron, so filtering has to happen on the
    # loaded sources: Fe I measurements must never get an Fe II correction.
    fe1 = {c.source for c in correct("FeI", 5232.94, abund=5.3, registry=registry,
                                     **METAL_POOR_DWARF)}
    fe2 = {c.source for c in correct("FeII", 4923.92, abund=5.3, registry=registry,
                                     **METAL_POOR_DWARF)}
    assert fe1 and not any(s.endswith("fe2") for s in fe1)
    assert fe2 and all("fe1" not in s for s in fe2)


def test_barium_lines_are_reachable_as_ba_two(registry):
    # The Amarsi file is named "ba1" but 4554/5854/6142/6497 are Ba II lines.
    got = correct("BaII", 6496.9, abund=1.3, teff=5966, logg=3.76, feh=-0.51,
                  vmic=1.27, registry=registry)
    assert got and any(c.ok for c in got)


def test_correct_frame_accepts_a_species_column(registry):
    frame = pd.DataFrame({
        "species": ["FeI", "BaII", "Sc"],
        "wavelength": [5232.94, 6496.9, 4314.08],
        "abund": [6.9, 1.3, 2.8],
        "teff": [5966] * 3, "logg": [3.76] * 3,
        "feh": [-0.51] * 3, "vmic": [1.27] * 3,
    })
    out = correct_frame(frame, registry=registry)
    assert out.ok.any()
    # An element no grid covers must not abandon the rest of the table.
    scandium = out[out.row == 2]
    assert len(scandium) == 1 and not scandium.ok.iloc[0]
    assert "no grid" in scandium.note.iloc[0]


def test_correct_frame_reports_an_unrecognised_species(registry):
    frame = pd.DataFrame({
        "species": ["Xx"], "wavelength": [5000.0], "abund": [1.0],
        "teff": [5777], "logg": [4.4], "feh": [0.0], "vmic": [1.0],
    })
    out = correct_frame(frame, registry=registry)
    assert len(out) == 1 and not out.ok[0]
    assert "unrecognised species" in out.note[0]


def test_correct_frame_handles_many_rows_and_species(registry):
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "teste_interpolador_in.csv"
    if not example.exists():
        pytest.skip("example input not present")
    frame = pd.read_csv(example, encoding="utf-8-sig")
    out = correct_frame(frame, registry=registry, keep=["id"])
    assert len(out) >= len(frame)
    assert out.ok.sum() > 200
    assert set(out.row) == set(range(len(frame)))


def test_cli_wide_output_keeps_every_input_row(tmp_path, registry):
    # The wide pivot must not drop measurements no grid covers, and must not
    # collide with the input's own wavelength column.
    from nlte.cli import main_correct

    src = tmp_path / "in.csv"
    src.write_text(
        "species,wavelength,abund,teff,logg,feh,vmic\n"
        "FeI,5232.94,6.9,5966,3.76,-0.51,1.27\n"
        "Sc,4314.08,2.8,5966,3.76,-0.51,1.27\n"
        "BaII,6496.9,1.3,5966,3.76,-0.51,1.27\n"
    )
    out = tmp_path / "out.csv"
    assert main_correct([str(src), "-o", str(out), "--wide"]) == 0

    got = pd.read_csv(out)
    assert len(got) == 3
    assert not any(c.endswith(("_x", "_y")) for c in got.columns)
    assert "wavelength" in got.columns
    delta_cols = [c for c in got.columns if c.startswith("delta_")]
    assert delta_cols
    assert got.loc[got.species == "Sc", delta_cols].isna().all().all()
    assert got.loc[got.species == "FeI", delta_cols].notna().any().any()
