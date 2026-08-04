"""The two solvers: self-consistent abundance, and equivalent-width matching."""

import numpy as np
import pytest

from nlte.solve import solve_delta, solve_ew_match


def test_self_consistent_solution_satisfies_its_own_equation():
    delta_fn = lambda a: -0.3 * (np.asarray(a) - 7.0)   # noqa: E731
    a_lte = np.array([6.0, 7.0, 8.0])
    sol = solve_delta(delta_fn, a_lte)
    assert sol.converged.all()
    residual = sol.a_nlte - delta_fn(sol.a_nlte) - a_lte
    assert np.abs(residual).max() < 1e-4


def test_direct_mode_reads_delta_at_the_lte_abundance():
    delta_fn = lambda a: -0.3 * (np.asarray(a) - 7.0)   # noqa: E731
    sol = solve_delta(delta_fn, [6.0], mode="direct")
    assert np.isclose(sol.delta[0], 0.3)
    assert sol.n_iter == 0


def test_direct_and_self_consistent_differ_when_delta_varies():
    delta_fn = lambda a: -0.3 * (np.asarray(a) - 7.0)   # noqa: E731
    direct = solve_delta(delta_fn, [6.0], mode="direct")
    exact = solve_delta(delta_fn, [6.0])
    assert abs(direct.delta[0] - exact.delta[0]) > 0.05


def test_steep_dependence_falls_back_to_bracketing():
    # |dDelta/dA| > 1: plain fixed-point iteration would oscillate away.
    delta_fn = lambda a: -1.6 * (np.asarray(a) - 7.0)   # noqa: E731
    sol = solve_delta(delta_fn, [6.5], bounds=(4.0, 10.0))
    assert sol.converged.all()
    assert abs(sol.a_nlte[0] - delta_fn(sol.a_nlte)[0] - 6.5) < 1e-4


def test_reports_failure_where_the_grid_has_no_values():
    delta_fn = lambda a: np.where(np.asarray(a) < 7.5, np.nan, 0.1)   # noqa: E731
    sol = solve_delta(delta_fn, [6.0])
    assert not sol.converged[0]
    assert np.isnan(sol.delta[0]) and np.isnan(sol.a_nlte[0])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        solve_delta(lambda a: np.zeros_like(a), [7.0], mode="magic")


# -- equivalent-width matching ------------------------------------------

def _curves(offset=0.15, n=17):
    """A linear curve of growth, with non-LTE stronger by `offset` dex."""
    abund = np.linspace(5.0, 9.0, n)
    lte = abund - 9.0
    nlte = abund - 9.0 + offset
    return abund, lte, nlte


def test_ew_matching_recovers_a_known_offset():
    abund, lte, nlte = _curves(0.15)
    match = solve_ew_match(abund, lte, nlte, 7.0)
    assert match.ok
    assert np.isclose(match.delta, -0.15, atol=1e-6)


def test_ew_matching_tolerates_masked_nodes():
    abund, lte, nlte = _curves(0.15)
    lte[:5] = np.nan
    match = solve_ew_match(abund, lte, nlte, 7.0)
    assert match.ok and np.isclose(match.delta, -0.15, atol=1e-6)


def test_ew_matching_flags_saturation_rather_than_extrapolating():
    abund, lte, nlte = _curves(0.15)
    nlte[12:] = nlte[12]                 # growth curve flattens
    match = solve_ew_match(abund, lte, nlte, 8.8)
    assert not match.ok
    assert "saturated" in match.note


def test_ew_matching_rejects_an_abundance_off_the_grid():
    abund, lte, nlte = _curves()
    match = solve_ew_match(abund, lte, nlte, 3.0)
    assert not match.ok
    assert "outside grid" in match.note


def test_ew_matching_needs_enough_coverage():
    abund, lte, nlte = _curves()
    lte[:] = np.nan
    match = solve_ew_match(abund, lte, nlte, 7.0)
    assert not match.ok
    assert "insufficient" in match.note
