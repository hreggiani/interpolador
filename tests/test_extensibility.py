"""Adding a table must stay cheap: a manifest, no Python.

These tests build small grids in a temporary directory and query them through
the public API, so they exercise the same path the shipped tables use without
needing the real data.
"""

import numpy as np
import pytest

from nlte.api import correct, find_lines
from nlte.registry import Registry, register_reader
from nlte.source import GridSource, Line, make_grid

MANIFEST = """
id        = "demo2030_ca"
format    = "tabular"
reference = "Demo et al. 2030"
element   = "Ca"
ion       = 1
file      = "demo.csv"
kind      = "delta"
flavor    = "1D NLTE - 1D LTE"

[read]
delimiter = ","

[columns]
teff  = "teff"
logg  = "logg"
feh   = "feh"
abund = "xfe"
line  = "line"
value = "abcor"

[abundance]
kind  = "xfe"
solar = 6.34

[wavelengths]
unit = "angstrom"
"""


@pytest.fixture
def demo_root(tmp_path):
    """A grid root holding one small, entirely synthetic table."""
    rows = ["teff,logg,feh,xfe,line,abcor"]
    for teff in (5000, 6000):
        for logg in (2.0, 4.0):
            for feh in (-2.0, 0.0):
                for xfe in (-0.5, 0.0, 0.5):
                    # A correction that varies with abundance, so the
                    # self-consistent solve has something to do.
                    value = -0.1 * xfe + 0.05 * feh
                    rows.append(f"{teff},{logg},{feh},{xfe},4226.73,{value:.6f}")
    (tmp_path / "demo.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "demo.toml").write_text(MANIFEST)
    return tmp_path


# With [Fe/H] = -1 and a solar A(Ca) of 6.34, the grid's [Ca/Fe] axis of
# -0.5 .. 0.5 covers A(Ca) = 4.84 .. 5.84.
STAR = dict(teff=5500, logg=3.0, feh=-1.0)
IN_RANGE_ABUND = 5.5


def test_a_manifest_alone_makes_a_table_queryable(demo_root):
    registry = Registry(demo_root)
    assert registry.elements() == ["Ca"]

    results = correct("Ca", 4226.73, abund=IN_RANGE_ABUND, registry=registry, **STAR)
    assert len(results) == 1
    corr = results[0]
    assert corr.source == "demo2030_ca"
    assert corr.reference == "Demo et al. 2030"
    assert corr.ok and corr.in_range and np.isfinite(corr.delta)


def test_the_new_table_solves_self_consistently(demo_root):
    registry = Registry(demo_root)
    src = registry.get("demo2030_ca")
    corr = correct("Ca", 4226.73, abund=IN_RANGE_ABUND, registry=registry, **STAR)[0]
    delta = src.delta_at(0, STAR, corr.a_nlte, feh=STAR["feh"])
    assert abs(corr.a_nlte - delta.value[0] - corr.a_lte) < 1e-3


def test_an_abundance_past_the_axis_end_is_flagged(demo_root):
    registry = Registry(demo_root)
    corr = correct("Ca", 4226.73, abund=8.0, registry=registry, **STAR)[0]
    assert not corr.in_range
    assert "edge" in corr.note or not corr.ok


def test_the_abundance_axis_is_converted_from_log_eps(demo_root):
    registry = Registry(demo_root)
    src = registry.get("demo2030_ca")
    # [Ca/Fe] = A(Ca) - 6.34 - [Fe/H]; the axis spans -0.5 .. 0.5
    assert np.isclose(src.to_native_abundance(6.34, feh=-1.0), 1.0)
    lo, hi = src.abundance_bounds(feh=-1.0)
    assert np.isclose(lo, 6.34 - 1.0 - 0.5) and np.isclose(hi, 6.34 - 1.0 + 0.5)


def test_an_unknown_format_names_the_formats_that_do_exist(tmp_path):
    (tmp_path / "bad.toml").write_text(
        'id = "x"\nformat = "no-such-format"\nelement = "Fe"\n'
    )
    registry = Registry(tmp_path)
    with pytest.raises(KeyError, match="no reader for format"):
        registry.for_element("Fe")


def test_a_custom_reader_can_be_registered(tmp_path):
    @register_reader("demo-inline")
    def _read(manifest, path):
        axes = {"teff": np.array([5000.0, 6000.0]), "abund": np.array([0.0, 1.0])}
        values = np.array([[0.1, 0.2], [0.3, 0.4]])
        return [GridSource(
            id=manifest["id"], reference="inline", element="Ba", ion=2,
            flavor="test", kind="delta", axis_names=("teff", "abund"),
            lines=[Line(wavelength=4554.03, label="4554")],
            grids=[make_grid(axes, values)],
        )]

    (tmp_path / "inline.toml").write_text(
        'id = "inline_ba"\nformat = "demo-inline"\nelement = "Ba"\nion = 2\n'
    )
    registry = Registry(tmp_path)
    corr = correct("Ba", 4554.03, abund=0.5, teff=5500, registry=registry)
    assert len(corr) == 1 and corr[0].ok


def test_in_memory_sources_can_be_registered(tmp_path):
    axes = {"teff": np.array([5000.0, 6000.0]), "abund": np.array([0.0, 1.0])}
    src = GridSource(
        id="scratch_eu", reference="built in a script", element="Eu", ion=2,
        flavor="test", kind="delta", axis_names=("teff", "abund"),
        lines=[Line(wavelength=4129.72, label="4129")],
        grids=[make_grid(axes, np.array([[0.1, 0.2], [0.3, 0.4]]))],
    )
    registry = Registry(tmp_path)
    registry.add(src)
    assert find_lines("Eu", 4129.72, registry=registry)
    assert correct("Eu", 4129.72, abund=0.5, teff=5500, registry=registry)[0].ok


def test_one_manifest_can_describe_a_family_of_tables(tmp_path):
    (tmp_path / "a.csv").write_text("teff,abund,value\n5000,0,0.1\n6000,0,0.2\n"
                                    "5000,1,0.3\n6000,1,0.4\n")
    (tmp_path / "b.csv").write_text("teff,abund,value\n5000,0,0.5\n6000,0,0.6\n"
                                    "5000,1,0.7\n6000,1,0.8\n")
    (tmp_path / "fam.toml").write_text("""
id = "fam"
format = "tabular"
element = "K"
reference = "family test"
[read]
delimiter = ","
[columns]
teff = "teff"
abund = "abund"
value = "value"
[abundance]
kind = "logeps"
[[tables]]
id = "fam_a"
file = "a.csv"
line = 7698.96
[[tables]]
id = "fam_b"
file = "b.csv"
line = 7698.96
""")
    registry = Registry(tmp_path)
    ids = {s.id for s in registry.for_element("K")}
    assert ids == {"fam_a", "fam_b"}
