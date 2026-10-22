"""Every shipped grid loads, and reproduces its own values at its own nodes.

The node round-trip is the load-bearing test here: the IDL cubes arrive with
their axes reversed and have to be transposed into canonical order, and a silent
axis swap would produce plausible-looking but wrong corrections everywhere.
"""

import numpy as np
import pytest

from nlte.interp import HybridGrid, RectGrid
from nlte.source import ABUND_KINDS

from .conftest import manifest_is_big

pytestmark = pytest.mark.grids


def _all_manifests(registry, test_all):
    return [m for m in registry.manifests if test_all or not manifest_is_big(m)]


def _round_trip_error(grid, rng, n=200):
    if isinstance(grid, RectGrid):
        shape = grid.values.shape
        idx = np.column_stack([rng.integers(0, s, n) for s in shape])
        pts = np.column_stack([grid.axes[d][idx[:, d]] for d in range(len(shape))])
        truth = grid.values[tuple(idx.T)]
    else:
        rshape = grid.values.shape[1:]
        ni = rng.integers(0, grid.nodes.shape[0], n)
        ri = np.column_stack([rng.integers(0, s, n) for s in rshape])
        pts = np.column_stack([grid.nodes[ni]]
                              + [grid.rect_axes[d][ri[:, d]] for d in range(len(rshape))])
        truth = grid.values[(ni, *ri.T)]
    got = grid(pts).value
    m = np.isfinite(truth)
    if not m.any():
        return None
    assert np.isfinite(got[m]).all(), "a valid node came back masked"
    return float(np.abs(got[m] - truth[m]).max())


def test_manifests_are_discovered(registry):
    assert len(registry.manifests) >= 30
    assert {"Mg", "Fe", "O", "Al", "Si", "Li"} <= set(registry.elements())


def test_every_source_loads_and_round_trips(registry, test_all):
    checked = 0
    for manifest in _all_manifests(registry, test_all):
        for src in registry._load(manifest):
            if getattr(src._first_grid(), "values", None) is None:
                continue          # model-backed source; no nodes to round-trip
            errors = [
                e for grid in src.grids[:6]
                for part in (grid if isinstance(grid, tuple) else (grid,))
                if (e := _round_trip_error(part, np.random.default_rng(1))) is not None
            ]
            assert errors, f"{src.id}: no finite nodes to check"
            assert max(errors) < 1e-6, f"{src.id}: node round-trip error {max(errors):.2e}"
            checked += 1
    assert checked > 20


def test_source_metadata_is_coherent(registry, test_all):
    for manifest in _all_manifests(registry, test_all):
        for src in registry._load(manifest):
            assert src.axis_names[-1] == "abund"
            assert src.abund_kind in ABUND_KINDS
            assert src.kind in ("delta", "ew_pair")
            assert len(src.lines) == len(src.grids)
            assert src.reference, f"{src.id} has no reference"
            bounds = src.bounds()
            for name, (lo, hi) in bounds.items():
                assert lo < hi, f"{src.id}: axis {name} is degenerate"
            if src.kind == "ew_pair":
                assert src.abund_nodes is not None


def test_wavelengths_are_in_angstrom(registry, test_all):
    for manifest in _all_manifests(registry, test_all):
        for src in registry._load(manifest):
            wl = np.array([ln.wavelength for ln in src.lines])
            wl = wl[np.isfinite(wl)]
            assert (wl > 1000).all(), f"{src.id}: wavelengths look like nm, not Angstrom"
            assert (wl < 250000).all()


def test_degenerate_axes_become_fixed_parameters(registry):
    # The Fe II grid holds a single [Fe/H] and a single upper microturbulence.
    src = registry.get("amarsi_fe2_3d")
    assert "feh" in src.fixed and np.isclose(src.fixed["feh"], -1.0)
    assert "feh" not in src.axis_names


def test_the_iron_tables_are_sparse_in_the_teff_logg_plane(registry, test_all):
    if not test_all:
        pytest.skip("iron tables are large; set NLTE_TEST_ALL to include them")
    src = registry.get("amarsi2016_fe_nmtd_lmarcs_fe1")
    assert isinstance(src.grids[0], HybridGrid)
    assert src.grids[0].nodes.shape[0] < 13 * 8      # not the full rectangle


def test_silicon_is_reparametrised_to_x_over_fe(registry):
    src = registry.get("amarsi2017_si")
    lo, hi = src.bounds()["abund"]
    assert -1.0 < lo < 0.0 and 1.0 < hi < 2.0        # [Si/Fe], not log eps
    assert src.coverage() > 0.9                      # rectilinear once reparametrised


def test_blended_entries_are_reachable_by_their_components(registry):
    src = registry.get("amarsi2016_o1_marcs")
    blends = [ln for ln in src.lines if ln.blend]
    assert blends, "the oxygen grid's combined triplet entry was not detected"
    assert len(blends[0].match) > 1


def test_potassium_grid_is_indexed_by_the_lte_abundance(registry):
    # Verified numerically against the tabulated equivalent widths; see the
    # manifest.  A grid indexed this way must ignore the solve mode.
    src = registry.get("reggiani2019_k")
    assert src.indexed_by == "lte"

    from nlte.api import correct
    star = dict(teff=4600, logg=1.5, feh=-2.5, vmic=1.8, abund=2.8)
    exact = [c for c in correct("K", 7698.96, mode="selfconsistent", registry=registry, **star)
             if c.source == "reggiani2019_k"][0]
    direct = [c for c in correct("K", 7698.96, mode="direct", registry=registry, **star)
              if c.source == "reggiani2019_k"][0]
    assert exact.ok and exact.delta == direct.delta


def test_potassium_labels_map_to_real_wavelengths(registry):
    # The published labels drop the decimal on the infrared lines: 151631 is
    # 15163.1 A, not 151631 A.
    src = registry.get("reggiani2019_k")
    by_label = {ln.label: ln.wavelength for ln in src.lines}
    assert by_label["151631"] == pytest.approx(15163.1)
    assert by_label["151632"] == pytest.approx(15163.2)
    assert by_label["7698"] == pytest.approx(7698.96)


def test_two_potassium_grids_agree_on_the_resonance_line(registry):
    # Reggiani 2019 and Amarsi 2020 are independent calculations; agreement is
    # evidence the units, abundance conversion and indexing are all right.
    from nlte.api import correct

    for star in (dict(teff=5777, logg=4.44, feh=0.0, vmic=1.0, abund=5.03),
                 dict(teff=4600, logg=1.5, feh=-2.5, vmic=1.8, abund=2.8)):
        got = {c.source: c.delta for c in correct("K", 7698.96, registry=registry, **star)
               if c.ok}
        if {"reggiani2019_k", "amarsi2020_k1"} <= set(got):
            assert abs(got["reggiani2019_k"] - got["amarsi2020_k1"]) < 0.05
            assert got["reggiani2019_k"] < 0        # K I resonance line, always negative


def _sodium_networks_available(registry) -> bool:
    try:
        return bool(registry.get("canocchi2026_na_3dnlte"))
    except KeyError:
        return False


def test_network_backed_source_presents_the_interpolator_surface(registry):
    # The sodium grid is trained networks, not a table; it must still describe
    # itself and answer queries like any other source.
    if not _sodium_networks_available(registry):
        pytest.skip("Canocchi sodium networks not present on this machine")
    src = registry.get("canocchi2026_na_3dnlte")
    assert src.geometry() == "neural network"
    assert np.isnan(src.coverage())          # no tabulated nodes to count
    assert src.indexed_by == "lte"
    assert set(src.bounds()) == {"teff", "logg", "feh", "vmic", "abund"}
    assert "coverage:   n/a" in src.describe()


def test_sodium_network_corrections_are_sane(registry):
    if not _sodium_networks_available(registry):
        pytest.skip("Canocchi sodium networks not present on this machine")
    from nlte.api import correct

    # 3D NLTE - 1D LTE for Na is negative for all but the most metal-poor stars.
    got = [c for c in correct("Na", 5682.67, abund=6.22, teff=6000, logg=4.5,
                              feh=0.0, vmic=1.0, registry=registry)
           if c.source == "canocchi2026_na_3dnlte"]
    assert got and got[0].ok
    assert -0.5 < got[0].delta < 0.0


def test_sodium_network_rejects_queries_outside_training_range(registry):
    if not _sodium_networks_available(registry):
        pytest.skip("Canocchi sodium networks not present on this machine")
    from nlte.api import correct

    got = [c for c in correct("Na", 5682.67, abund=6.22, teff=8000, logg=4.5,
                              feh=0.0, vmic=1.0, registry=registry)
           if c.source == "canocchi2026_na_3dnlte"]
    assert got and not got[0].ok and not got[0].in_range


def test_missing_external_data_is_skipped_not_fatal(tmp_path):
    # A manifest pointing at a distribution that is not on this machine must not
    # break every query for that element.
    from nlte.registry import Registry

    (tmp_path / "gone.toml").write_text(
        'id = "gone_na"\nformat = "canocchi_ffnn"\nelement = "Na"\n'
        'file = "no/such/directory"\n'
        '[axes]\nteff = [4000, 6500]\nlogg = [1.5, 5]\nfeh = [-4, 0.5]\n'
        'vmic = [0.5, 3]\nabund = [1.72, 7.72]\n'
        '[[lines]]\ntag = 5682\nwavelength = 5682.67\nrew_range = [-9, -3.5]\n'
    )
    reg = Registry(tmp_path)
    with pytest.warns(RuntimeWarning, match="data not found"):
        assert reg.for_element("Na") == []
    assert reg.missing


def _breidablik_available() -> bool:
    try:
        import breidablik  # noqa: F401
    except ImportError:
        return False
    return True


def test_breidablik_matches_its_own_entry_point(registry):
    if not _breidablik_available():
        pytest.skip("breidablik is not installed")
    from breidablik.interpolate.nlte import Nlte

    from nlte.api import correct

    reference = Nlte()
    cases = [
        (5353.0, 4.25, -1.53, 2.85, 670.9659, 6707.807),
        (5688.0, 4.53, -0.90, 1.66, 610.5298, 6103.608),
        (6100.0, 4.10, -0.50, 2.20, 812.8606, 8126.371),
    ]
    for teff, logg, feh, abund, centre, air in cases:
        theirs = reference.nlte_correction(teff, logg, feh, abund, center=centre)
        mine = [c for c in correct("Li", air, abund=abund, teff=teff, logg=logg,
                                   feh=feh, registry=registry) if c.source == "wang2021_li"]
        assert mine and mine[0].ok
        assert mine[0].delta == pytest.approx(theirs, abs=1e-12)


def test_breidablik_coverage_test_matches_theirs(registry):
    if not _breidablik_available():
        pytest.skip("breidablik is not installed")
    import warnings as _warnings

    from breidablik.interpolate.grid_check import _grid_check

    from nlte.readers.breidablik import _within_grid

    rng = np.random.default_rng(11)
    for _ in range(60):
        teff = float(rng.uniform(3500, 7500))
        logg = float(rng.uniform(1.0, 5.5))
        feh = float(rng.uniform(-3.0, 0.8))
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _grid_check(teff, logg, feh)
            theirs = not any("outside of the grid" in str(c.message) for c in caught)
        assert bool(_within_grid([teff], [logg], [feh])[0]) == theirs


def test_lithium_line_centres_convert_from_vacuum_to_air(registry):
    if not _breidablik_available():
        pytest.skip("breidablik is not installed")
    src = registry.get("wang2021_li")
    got = sorted(ln.wavelength for ln in src.lines)
    assert got == pytest.approx([6103.608, 6707.807, 8126.371], abs=0.01)


def test_copper_grid_is_indexed_by_the_lte_abundance(registry):
    src = registry.get("caliskan2025_cu")
    assert src.indexed_by == "lte"
    assert src.abund_kind == "xfe" and src.abund_solar == pytest.approx(4.18)
    assert len(src.lines) == 12


def test_copper_grid_refuses_the_gap_between_giants_and_dwarfs(registry):
    # The grid was computed as two disjoint blocks; nothing lies between them.
    from nlte.api import correct

    def ask(teff, logg):
        got = [c for c in correct("Cu", 5105.5, abund=3.18, teff=teff, logg=logg,
                                  feh=-1.0, vmic=1.0, registry=registry)
               if c.source == "caliskan2025_cu"]
        assert got
        return got[0]

    assert ask(4500, 1.5).ok            # giant block
    assert ask(6000, 4.5).ok            # dwarf block
    assert not ask(6000, 2.0).ok        # nothing was computed here
    assert not ask(6000, 2.0).in_range


# ---------------------------------------------------------------- NLiTE-list grids


def _point(src, **params):
    return np.array([[params[n] for n in src.axis_names]])


def test_inasan_grid_reproduces_the_server_at_a_node(registry):
    # +0.156 is what spectrum.inasan.ru/nLTE2 returned for this node when the
    # grid was rebuilt; Fe I there does not depend on abundance.
    from nlte.api import correct

    for abund in (4.0, 4.5, 6.0):
        (c,) = correct("FeI", 5171.61, abund=abund, teff=5000, logg=2.0, feh=-3.0,
                       sources=["mashonkina2011_fe"], registry=registry)
        assert c.ok and c.in_range and c.delta == pytest.approx(0.156)


def test_inasan_sodium_joins_the_dwarf_and_giant_halves(registry):
    # The server lists each Na I line twice: one entry is the dwarf grid, the
    # other the giant grid.  Merged, the lines reach the cool giants.
    src = registry.get("alexeeva2014_na")
    assert sorted(ln.wavelength for ln in src.lines) == [5682.64, 5688.2, 5889.95, 5895.92]
    assert src.bounds()["teff"] == (4000.0, 6500.0) and src.bounds()["logg"] == (0.5, 5.0)
    d1 = src.lines.index(next(ln for ln in src.lines if ln.wavelength == 5895.92))
    giant = src.grids[d1](_point(src, teff=4500, logg=1.5, feh=-3.0, abund=0.0))
    assert np.isfinite(giant.value).all()


def test_gallagher_reproduces_its_solar_model(registry):
    # barium_corrections_v5: Teff 5770, log g 4.44, [M/H] 0, 4554 A, [Ba/Fe] = 0
    # -> 3D_corr = 0.049, 1D_corr = -0.028.
    for suffix, expected in (("3dnlte", 0.049), ("1dnlte", -0.028)):
        src = registry.get(f"gallagher2020_ba_{suffix}")
        i = next(k for k, ln in enumerate(src.lines) if abs(ln.wavelength - 4554.03) < 0.1)
        got = src.grids[i](_point(src, teff=5770, logg=4.4, feh=0.0, abund=0.0)).value[0]
        assert got == pytest.approx(expected)


def test_korotin_keeps_the_alpha_enhanced_models_at_minus_half(registry):
    # table5 lists [Fe/H] = -0.5 twice; the "a" (alpha-enhanced) row has -0.09
    # for 5853 A at [Ba/Fe] = +0.1, the solar-composition row -0.08.
    src = registry.get("korotin2015_ba")
    i = next(k for k, ln in enumerate(src.lines) if ln.label == "5853.70")
    got = src.grids[i](_point(src, teff=5000, logg=2.0, feh=-0.5, vmic=1.0, abund=0.1)).value[0]
    assert got == pytest.approx(-0.09)


def test_lind2022_resolves_shared_line_identifiers(registry):
    # 7877 is Mg II 7877.054 and 7878 is Mg I 7877.48: the first strengthens with
    # Teff like the other Mg II lines, the second weakens like Mg I.
    mg1, mg2 = registry.get("lind2022_mg1"), registry.get("lind2022_mg2")
    assert 7877.054 in [ln.wavelength for ln in mg2.lines]
    assert 7877.48 in [ln.wavelength for ln in mg1.lines]
    assert {ln.wavelength for ln in mg1.lines}.isdisjoint(ln.wavelength for ln in mg2.lines)
    # [X/Fe] is printed with one decimal; the axis is snapped back to 0.25 steps.
    nodes = mg1.abund_nodes
    assert np.allclose(nodes / 0.25, np.round(nodes / 0.25))


def test_abundance_independent_sources_say_so(registry):
    for sid in ("mashonkina2011_fe", "merle2011_mg1", "takeda2005_s1",
                "bergemann2012_sr_sr2", "sitnova2022_zn_marcs_zn1"):
        src = registry.get(sid)
        assert src.abund_independent and src.indexed_by == "lte", sid
