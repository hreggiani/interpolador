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
