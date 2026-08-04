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
