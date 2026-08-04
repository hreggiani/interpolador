"""Interpolation engine: exactness, node round-trip, and masking behaviour."""

import numpy as np
import pytest

from nlte.interp import HybridGrid, RectGrid


@pytest.fixture
def linear_grid():
    axes = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0]), np.array([0.0, 1.0, 2.0, 3.0])]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return RectGrid(axes, 2 * x + 0.1 * y - 3 * z, names=["a", "b", "c"])


def test_reproduces_a_linear_function(linear_grid):
    pts = np.array([[1.5, 15.0, 0.5], [3.0, 20.0, 3.0], [2.2, 12.0, 1.7]])
    expected = 2 * pts[:, 0] + 0.1 * pts[:, 1] - 3 * pts[:, 2]
    result = linear_grid(pts)
    assert np.allclose(result.value, expected)
    assert result.in_range.all()


def test_round_trips_every_node(linear_grid):
    nodes = np.array([[a, b, c] for a in linear_grid.axes[0]
                      for b in linear_grid.axes[1] for c in linear_grid.axes[2]])
    assert np.allclose(linear_grid(nodes).value, linear_grid.values.ravel())


def test_flags_out_of_range_queries(linear_grid):
    result = linear_grid(np.array([[0.5, 15.0, 1.0], [1.5, 15.0, 9.0]]))
    assert not result.in_range.any()
    assert np.isnan(result.value).all()


def test_clipped_axis_accepts_out_of_range(linear_grid):
    clipped = RectGrid(linear_grid.axes, linear_grid.values,
                       names=["a", "b", "c"], clip_axes=["b"])
    result = clipped([[2.0, 50.0, 1.0]])
    assert result.in_range.all()
    assert np.isclose(result.value[0], 2 * 2.0 + 0.1 * 20.0 - 3 * 1.0)


def test_clip_ignores_axes_the_grid_does_not_have(linear_grid):
    # Manifests clip vmic even for grids that collapsed that axis away.
    RectGrid(linear_grid.axes, linear_grid.values, names=["a", "b", "c"],
             clip_axes=["vmic"])


def test_renormalises_around_masked_corners(linear_grid):
    values = linear_grid.values.copy()
    values[0, 0, 0] = np.nan
    grid = RectGrid(linear_grid.axes, values, names=["a", "b", "c"])
    result = grid([[1.2, 12.0, 0.3]])
    assert np.isfinite(result.value[0])
    assert 0 < result.valid_weight[0] < 1


def test_rejects_cells_below_the_weight_threshold(linear_grid):
    values = np.full_like(linear_grid.values, np.nan)
    values[0, 0, 0] = 1.0
    grid = RectGrid(linear_grid.axes, values, names=["a", "b", "c"])
    result = grid([[1.5, 15.0, 0.5]])          # only 1 of 8 corners survives
    assert np.isnan(result.value[0])
    assert result.in_range[0]                  # in range, just unusable
    assert result.valid_weight[0] < 0.5


def test_fully_masked_cell_returns_nan(linear_grid):
    grid = RectGrid(linear_grid.axes, np.full_like(linear_grid.values, np.nan))
    result = grid([[1.5, 15.0, 0.5]])
    assert np.isnan(result.value[0]) and result.valid_weight[0] == 0


def test_rejects_degenerate_axes():
    with pytest.raises(ValueError, match="at least 2 nodes"):
        RectGrid([np.array([1.0])], np.array([0.0]))


def test_rejects_unsorted_axes():
    with pytest.raises(ValueError, match="strictly increasing"):
        RectGrid([np.array([2.0, 1.0])], np.array([0.0, 1.0]))


@pytest.fixture
def hybrid():
    nodes = np.array([[t, l] for t in (4000.0, 5000.0, 6000.0) for l in (1.5, 3.0, 4.5)])
    rect = [np.array([0.75, 1.5, 3.0]), np.array([2.5, 3.0, 3.5, 4.0])]
    values = np.empty((len(nodes), 3, 4))
    for i, (t, l) in enumerate(nodes):
        for j, v in enumerate(rect[0]):
            for k, a in enumerate(rect[1]):
                values[i, j, k] = 1e-3 * t + 2 * l + 0.5 * v - a
    return HybridGrid(nodes, rect, values, names=["teff", "logg"],
                      rect_names=["vmic", "abund"])


def test_hybrid_reproduces_a_linear_function(hybrid):
    pts = np.array([[4500.0, 2.0, 1.0, 3.2], [6000.0, 4.5, 3.0, 4.0]])
    expected = 1e-3 * pts[:, 0] + 2 * pts[:, 1] + 0.5 * pts[:, 2] - pts[:, 3]
    assert np.allclose(hybrid(pts).value, expected)


def test_hybrid_round_trips_nodes(hybrid):
    rng = np.random.default_rng(0)
    ni = rng.integers(0, hybrid.nodes.shape[0], 50)
    ri = np.column_stack([rng.integers(0, s, 50) for s in hybrid.values.shape[1:]])
    pts = np.column_stack([hybrid.nodes[ni]]
                          + [hybrid.rect_axes[d][ri[:, d]] for d in range(2)])
    assert np.allclose(hybrid(pts).value, hybrid.values[(ni, *ri.T)])


def test_hybrid_rejects_points_outside_the_hull(hybrid):
    result = hybrid([[3000.0, 1.0, 1.0, 3.0]])
    assert not result.in_range[0]
    assert np.isnan(result.value[0])
    assert result.valid_weight[0] == 0     # no meaningless weight is reported


def test_vacuum_to_air_reproduces_known_line_centres():
    from nlte.wavelengths import air_to_vacuum, vacuum_to_air

    # Li I 6707.81, 6103.6 and 8126.4 in air; Breidablik quotes them in vacuum.
    assert vacuum_to_air(6709.659) == pytest.approx(6707.81, abs=0.01)
    assert vacuum_to_air(6105.298) == pytest.approx(6103.61, abs=0.01)
    assert vacuum_to_air(8128.606) == pytest.approx(8126.37, abs=0.01)


def test_vacuum_air_conversions_round_trip():
    from nlte.wavelengths import air_to_vacuum, vacuum_to_air

    wl = np.array([3000.0, 5000.0, 6709.659, 10000.0, 25000.0])
    assert np.allclose(air_to_vacuum(vacuum_to_air(wl)), wl, atol=1e-6)


def test_below_2000_angstrom_is_left_in_vacuum():
    from nlte.wavelengths import vacuum_to_air

    assert vacuum_to_air(1500.0) == 1500.0


def test_max_node_distance_refuses_to_fill_a_hole(hybrid):
    # A triangulation fills its convex hull. For a grid computed as two disjoint
    # blocks, that means answering confidently where nothing was computed.
    nodes = np.array([[4000.0, 1.5], [4500.0, 1.5], [4000.0, 2.0], [4500.0, 2.0],
                      [6000.0, 4.5], [6500.0, 4.5], [6000.0, 5.0], [6500.0, 5.0]])
    rect = [np.array([0.0, 1.0])]
    values = np.tile(np.array([1.0, 2.0]), (len(nodes), 1))

    unguarded = HybridGrid(nodes, rect, values, names=["teff", "logg"],
                           rect_names=["abund"])
    guarded = HybridGrid(nodes, rect, values, names=["teff", "logg"],
                         rect_names=["abund"], max_node_distance=0.15)

    gap = [[5200.0, 3.0, 0.5]]
    assert unguarded(gap).in_range[0]              # hull-filling, no warning
    assert not guarded(gap).in_range[0]
    assert np.isnan(guarded(gap).value[0])

    # Points on the blocks themselves are unaffected.
    on_node = [[4000.0, 1.5, 0.5]]
    assert guarded(on_node).in_range[0]
    assert np.isclose(guarded(on_node).value[0], 1.5)
