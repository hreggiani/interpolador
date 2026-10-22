"""Manifest features added for the NLiTE-list grids, on small synthetic tables.

Each test writes a few files into a temporary grid root and queries them through
the public API, so the features are covered without the published data.
"""

import numpy as np
import pytest

from nlte.api import correct
from nlte.readers.inasan_lst import parse_lst
from nlte.readers.tabular import cds_layout
from nlte.registry import Registry

STAR = dict(teff=5500, logg=3.0, feh=-1.5)


def _grid_rows(value, lines=(5000.0,), xfe=(0.0,), fehs=(-2.0, -1.0)):
    """Rows teff,logg,feh,xfe,line,delta over a small rectangle."""
    for line in lines:
        for teff in (5000, 6000):
            for logg in (2.0, 4.0):
                for feh in fehs:
                    for x in xfe:
                        yield teff, logg, feh, x, line, value(teff, logg, feh, x)


def _write_csv(path, rows, header="teff,logg,feh,xfe,line,delta"):
    path.write_text(header + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")


def manifest(top: str, body: str, read: bool = True, kind: str = "delta", ion=1) -> str:
    """A synthetic manifest: top-level keys first, then the tables a test needs."""
    head = (f'format = "tabular"\nreference = "Synthetic"\nelement = "Ca"\n'
            f'kind = "{kind}"\nflavor = "1D NLTE - 1D LTE"\n'
            + (f"ion = {ion}\n" if ion is not None else ""))
    tables = ('[read]\ndelimiter = ","\n' if read else "") + '[wavelengths]\nunit = "angstrom"\n'
    return head + top.strip() + "\n" + tables + body


# ---------------------------------------------------------------- several files


def test_files_are_concatenated_and_named_parameters_come_from_file_names(tmp_path):
    for logg in (2.0, 4.0):
        for feh in (0.0, -1.0):
            rows = [(teff, x, 5000.0, 0.01 * logg + 0.1 * feh)
                    for teff in (5000, 6000) for x in (-0.5, 0.5)]
            name = f"g{int(logg * 10)}mm{int(-feh * 10):02d}.csv"
            _write_csv(tmp_path / name, rows, header="teff,xfe,line,delta")
    (tmp_path / "m.toml").write_text(manifest("""
id    = "multi"
files = "g*mm*.csv"
""", """
[file_params]
pattern = '^g(?P<logg>\\d+)mm(?P<feh>\\d+)\\.csv$'
scale   = { logg = 0.1, feh = -0.1 }
[columns]
teff = "teff"
logg = "logg"
feh  = "feh"
abund = "xfe"
line  = "line"
value = "delta"
[abundance]
kind  = "xfe"
solar = 6.34
"""))
    src = Registry(tmp_path).get("multi")
    assert src.bounds()["logg"] == (2.0, 4.0)
    assert src.bounds()["feh"] == (-1.0, 0.0)
    assert not np.signbit(src.bounds()["feh"][1])          # a plain 0, not -0.0
    (c,) = correct("CaI", 5000.0, abund=6.34 - 0.5, teff=5500, logg=3.0, feh=-0.5,
                   registry=Registry(tmp_path))
    assert c.ok and np.isclose(c.delta, 0.01 * 3.0 + 0.1 * -0.5)


# ---------------------------------------------------------------- CDS fixed width

README = """\
Byte-by-byte Description of file: t*.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  4  I4    K       Teff      Effective temperature
   6-  8  F3.1 [cm/s2]  logg      Surface gravity
  10- 13  F4.1 [Sun]    [Fe/H]    Metallicity
      14  A1    ---     a         [a] Alpha flag
  16- 20  F5.2  ---     Delta     ? Correction, blank when not computed
  22- 25  F4.1  ---     Other     A column after the optional one
--------------------------------------------------------------------------------
"""


def test_cds_layout_reads_the_byte_by_byte_block(tmp_path):
    (tmp_path / "ReadMe").write_text(README)
    names, colspecs = cds_layout(tmp_path / "ReadMe", "t1.dat.gz")
    assert names == ["Teff", "logg", "[Fe/H]", "a", "Delta", "Other"]
    assert colspecs[0] == (0, 4) and colspecs[3] == (13, 14)


def test_cds_fixed_width_survives_blank_fields_and_glued_flags(tmp_path):
    (tmp_path / "ReadMe").write_text(README)
    lines = []
    for teff in (5000, 6000):
        for logg in (2.0, 4.0):
            for feh in (-2.0, -1.0):
                blank = teff == 6000 and logg == 4.0 and feh == -1.0
                delta = "     " if blank else f"{0.01 * logg:5.2f}"
                lines.append(f"{teff:4d} {logg:3.1f} {feh:4.1f}a {delta} {9.9:4.1f}")
    (tmp_path / "t1.dat").write_text("\n".join(lines) + "\n")
    (tmp_path / "m.toml").write_text(manifest("""
id   = "cds"
file = "t1.dat"
line = 5000.0
""", """
[read]
format = "cds"
readme = "ReadMe"
[columns]
teff  = "Teff"
logg  = "logg"
feh   = "[Fe/H]"
value = "Delta"
[abundance]
independent = true
""", read=False))
    registry = Registry(tmp_path)
    values = registry.get("cds").grids[0].values
    assert np.isnan(values).sum() == 2               # the blank node, on both abundance slabs
    assert np.nanmax(values) == pytest.approx(0.04)  # "Other" (9.9) was not read as Delta

    # A cell touching the blank node, with every corner required: no answer, and
    # since this grid is read directly, no talk of a failed solve either.
    (c,) = correct("CaI", 5000.0, abund=5.0, teff=5900, logg=3.9, feh=-1.1,
                   min_valid_weight=0.999, registry=registry)
    assert not c.ok and c.note == "too few valid grid nodes around this point"


# ---------------------------------------------------------------- melt and where


def test_wide_columns_are_melted_into_line_and_abundance(tmp_path):
    header = "teff,logg,feh,flag,d1-0.2,d1+0.4,d2-0.2,d2+0.4"
    rows = []
    for teff in (5000, 6000):
        for logg in (2.0, 4.0):
            for feh in (-2.0, -1.0):
                rows.append((teff, logg, feh, "a", 0.1, 0.2, 0.3, 0.4))
    rows.append((5000, 2.0, -1.0, "", 9, 9, 9, 9))  # a duplicate node the manifest drops
    _write_csv(tmp_path / "wide.csv", rows, header=header)
    (tmp_path / "m.toml").write_text(manifest("""
id   = "wide"
file = "wide.csv"
""", """
[read]
delimiter = ","
where     = "flag == 'a'"
[melt]
pattern     = 'd(?P<line>\\d)(?P<abund>[-+]\\d\\.\\d)'
value       = "delta"
line_labels = { 1 = "4554.03", 2 = "5853.70" }
[columns]
line  = "line"
teff  = "teff"
logg  = "logg"
feh   = "feh"
abund = "abund"
value = "delta"
[abundance]
kind  = "xfe"
solar = 2.17
""", read=False))
    src = Registry(tmp_path).get("wide")
    assert [ln.wavelength for ln in src.lines] == [4554.03, 5853.70]
    assert src.bounds()["abund"] == (-0.2, 0.4)
    assert np.nanmax(src.grids[0].values) == pytest.approx(0.2)    # the dropped 9s are gone
    assert np.nanmin(src.grids[1].values) == pytest.approx(0.3)


# ---------------------------------------------------------------- collapsed axes


def test_single_node_axes_become_fixed_parameters(tmp_path):
    rows = [(5000 + 1000 * i, 2.0 + 2 * j, feh, 2.0, 5000.0, 0.1)
            for i in (0, 1) for j in (0, 1) for feh in (-2.0, -1.0)]
    _write_csv(tmp_path / "v.csv", rows, header="teff,logg,feh,vt,line,delta")
    (tmp_path / "m.toml").write_text(manifest("""
id   = "fixedvt"
file = "v.csv"
""", """
[columns]
teff  = "teff"
logg  = "logg"
feh   = "feh"
vmic  = "vt"
line  = "line"
value = "delta"
[abundance]
independent = true
"""))
    src = Registry(tmp_path).get("fixedvt")
    assert src.fixed == {"vmic": 2.0}
    assert "vmic" not in src.axis_names
    assert "vmic" not in src.required_params


def test_a_single_abundance_node_makes_the_source_abundance_independent(tmp_path):
    _write_csv(tmp_path / "g.csv", _grid_rows(lambda t, g, f, x: 0.01 * g + 0.1 * f))
    (tmp_path / "m.toml").write_text(manifest("""
id   = "flat"
file = "g.csv"
""", """
[columns]
teff  = "teff"
logg  = "logg"
feh   = "feh"
abund = "xfe"
line  = "line"
value = "delta"
[abundance]
kind  = "xfe"
solar = 6.34
"""))
    registry = Registry(tmp_path)
    src = registry.get("flat")
    assert src.abund_independent and src.fixed == {"abund": 0.0}
    assert "not an axis" in src.describe()
    expected = 0.01 * 3.0 + 0.1 * -1.5
    for abund in (1.0, 4.84, 9.0):                    # anywhere, including absurd values
        (c,) = correct("CaI", 5000.0, abund=abund, registry=registry, **STAR)
        assert c.ok and c.in_range and np.isclose(c.delta, expected)
        assert np.isclose(c.a_nlte, abund + expected)


def test_declared_independence_with_several_abundance_nodes_is_an_error(tmp_path):
    _write_csv(tmp_path / "g.csv", _grid_rows(lambda t, g, f, x: 0.1, xfe=(0.0, 0.4)))
    (tmp_path / "m.toml").write_text(manifest("""
id   = "contradiction"
file = "g.csv"
""", """
[columns]
teff  = "teff"
logg  = "logg"
feh   = "feh"
abund = "xfe"
line  = "line"
value = "delta"
[abundance]
independent = true
"""))
    with pytest.raises(ValueError, match="abundance-independent"):
        Registry(tmp_path).get("contradiction")


# ---------------------------------------------------------------- snap, ions, ew pairs


def test_snap_restores_an_axis_printed_with_too_few_decimals(tmp_path):
    printed = (-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8)       # 0.25 steps at one decimal
    _write_csv(tmp_path / "g.csv", _grid_rows(lambda t, g, f, x: 0.1, xfe=printed))
    (tmp_path / "m.toml").write_text(manifest("""
id   = "snapped"
file = "g.csv"
""", """
[columns]
teff  = "teff"
logg  = "logg"
feh   = "feh"
abund = "xfe"
line  = "line"
value = "delta"
[abundance]
kind  = "xfe"
solar = 6.34
[snap]
abund = 0.25
"""))
    nodes = Registry(tmp_path).get("snapped").grids[0].axes[-1]
    assert np.allclose(nodes, [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])


def test_roman_numeral_ion_columns_split_the_stages(tmp_path):
    rows = [(*r[:5], "Ca I" if r[4] < 6000 else "Ca II", r[5])
            for r in _grid_rows(lambda t, g, f, x: 0.1, lines=(5000.0, 8498.0))]
    _write_csv(tmp_path / "g.csv", rows, header="teff,logg,feh,xfe,line,ion,delta")
    (tmp_path / "m.toml").write_text(manifest("""
id         = "stages"
file       = "g.csv"
ion_column = "ion"
""", """
[columns]
teff  = "teff"
logg  = "logg"
feh   = "feh"
line  = "line"
value = "delta"
[abundance]
independent = true
""", ion=None))
    registry = Registry(tmp_path)
    assert {s.id: s.ion for s in registry.for_element("Ca")} == {"stages_ca1": 1, "stages_ca2": 2}
    assert [ln.wavelength for ln in registry.get("stages_ca2").lines] == [8498.0]


def test_long_format_equivalent_widths_are_matched(tmp_path):
    # Curves of growth on the linear part, with the non-LTE curve 0.1 dex
    # stronger: the same width needs 0.1 dex less abundance in non-LTE.
    rows = []
    for teff in (5000, 6000):
        for logg in (2.0, 4.0):
            for feh in (-2.0, -1.0):
                for x in (-0.5, 0.0, 0.5):
                    lte = 1.0 + x
                    rows.append((teff, logg, feh, x, 5000.0, lte, lte + 0.1))
    _write_csv(tmp_path / "ew.csv", rows, header="teff,logg,feh,xfe,line,lwl,lwn")
    (tmp_path / "m.toml").write_text(manifest("""
id   = "ewlong"
file = "ew.csv"
""", """
[columns]
teff    = "teff"
logg    = "logg"
feh     = "feh"
abund   = "xfe"
line    = "line"
ew_lte  = "lwl"
ew_nlte = "lwn"
[abundance]
kind  = "xfe"
solar = 6.34
""", kind="ew_pair"))
    registry = Registry(tmp_path)
    assert registry.get("ewlong").kind == "ew_pair"
    (c,) = correct("CaI", 5000.0, abund=6.34 - 1.5 + 0.1, registry=registry, **STAR)
    assert c.ok and np.isclose(c.delta, -0.1)


# ---------------------------------------------------------------- inasan .lst

LST = """\
 Zn lines, NLTE EW(mA) and NLTE abundance corrections
  481.0528 nm  Zn1  Eexc =  4.080  log gf =    -0.150
 Teff= 5000   2.0     4.0
   -1.0          52      -1
   -1.0       0.090  -1.000
   -2.0          31      28
   -2.0       0.120   0.140
 Teff= 6000   2.0     4.0
   -1.0          40      35
   -1.0       0.100   0.110
   -2.0          20      18
   -2.0       0.130   0.150
  206.2001 nm  Zn2  Eexc =  0.000  log gf =    -0.300
 Teff= 5000   2.0     4.0
   -1.0          80      70
   -1.0      -0.010  -0.020
   -2.0          60      50
   -2.0      -0.030  -0.040
 Teff= 6000   2.0     4.0
   -1.0          75      65
   -1.0      -0.015  -0.025
   -2.0          55      45
   -2.0      -0.035  -0.045
"""


def test_inasan_lst_blocks_parse_into_a_long_table(tmp_path):
    (tmp_path / "zn.lst").write_text(LST)
    df = parse_lst(tmp_path / "zn.lst")
    assert len(df) == 16 and set(df.species) == {"Zn1", "Zn2"}
    row = df[(df.line == "481.0528nm") & (df.teff == 6000) & (df.logg == 4.0) & (df.feh == -2.0)]
    assert row.delta.item() == pytest.approx(0.150) and row.ew.item() == 18
    weak = df[(df.line == "481.0528nm") & (df.teff == 5000) & (df.logg == 4.0) & (df.feh == -1.0)]
    assert np.isnan(weak.delta.item())                # EW = -1 masks the node


def test_inasan_lst_yields_one_source_per_stage(tmp_path):
    (tmp_path / "zn.lst").write_text(LST)
    (tmp_path / "zn.toml").write_text("""
id        = "zn"
format    = "inasan_lst"
reference = "Synthetic"
element   = "Zn"
file      = "zn.lst"
[wavelengths]
unit = "nm"
""")
    registry = Registry(tmp_path)
    zn2 = registry.get("zn_zn2")
    assert zn2.ion == 2 and zn2.abund_independent
    assert zn2.lines[0].wavelength == pytest.approx(2062.001)
    (c,) = correct("ZnI", 4810.53, abund=3.0, teff=6000, logg=4.0, feh=-2.0, registry=registry)
    assert c.ok and c.delta == pytest.approx(0.150)
