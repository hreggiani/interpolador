"""Turn correction grids printed in papers into tidy CSVs the tabular reader can load.

Three grids in the NLiTE list exist only as typeset tables.  Their arXiv sources
carry the LaTeX, so this script fetches each source, parses the tables, and writes
one long CSV per paper into its grids/ folder:

    takeda2002_k/k1_7698_nlte_corrections.csv      Takeda et al. 2002, PASJ 54, 275
    bergemann2012_sr/sr_nlte_corrections.csv       Bergemann et al. 2012, A&A 546, A90
    bergemann2013_si/si1_jband_nlte_corrections.csv Bergemann et al. 2013, ApJ 764, 115

usage: python tools/latex_tables.py [GRIDS_DIR]

The numbers are copied, not recomputed; missing entries ("--") stay empty.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ARXIV = {
    "takeda2002_k": "astro-ph/0110165",
    "bergemann2012_sr": "1207.2451",
    "bergemann2013_si": "1212.2649",
}


def fetch_source(arxiv_id: str, dest: Path) -> Path:
    """Download and unpack an arXiv e-print into ``dest``."""
    req = urllib.request.Request(f"https://arxiv.org/e-print/{arxiv_id}",
                                 headers={"User-Agent": "nlte-grids/latex_tables"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def environments(tex: str, anchor: str) -> list:
    """Bodies of the table environments whose text matches ``anchor``."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    pattern = r"\\begin\{((?:table|deluxetable)\*?)\}(.*?)\\end\{\1\}"
    return [m.group(2) for m in re.finditer(pattern, tex, re.S) if re.search(anchor, m.group(0))]


def rows_of(body: str) -> list:
    """Cells of each tabular row, with math and spacing markup removed."""
    if "\\startdata" in body:
        body = body.split("\\startdata", 1)[1].split("\\enddata", 1)[0]
    else:
        body = re.split(r"\\begin\{tabular\*?\}", body, maxsplit=1)[1]
        body = re.sub(r"^\s*(\{[^}]*\})+", "", body)        # the column spec
        body = body.split("\\end{tabular}", 1)[0]
    body = re.sub(r"\\(hline|noalign\{[^}]*\}|smallskip|tableline)", "", body)
    out = []
    for raw in re.split(r"\\\\", body):
        raw = re.sub(r"^\s*\[[-+\d.]+\s*[a-z]*\]", "", raw)
        cells = [clean(c) for c in raw.split("&")]
        if any(cells):
            out.append(cells)
    return out


def clean(cell: str) -> str:
    cell = cell.replace("$-$", "-").replace("$+$", "+")
    cell = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}", r"\1", cell)
    cell = re.sub(r"\\log\b", "log", cell)
    cell = re.sub(r"\\[a-zA-Z]+", " ", cell)
    cell = re.sub(r"[${}()~]", " ", cell)
    cell = " ".join(cell.split())
    return "" if cell in ("--", "---", "...") else cell


def number(cell: str):
    try:
        return float(cell)
    except ValueError:
        return None


def takeda2002(src: Path) -> list:
    """K I 7698.98, xi = 2 km/s: three abundance blocks per model row."""
    tex = (src / "msv2.tex").read_text(errors="replace")
    bodies = (environments(tex, r"NLTE abundance correction for K I 7698")
              + environments(tex, r"caption\{\(Continued\.\)\}"))
    out = []
    for body in bodies:
        for cells in rows_of(body):
            m = re.fullmatch(r"t(\d\d)g(\d\d)m(\d)", cells[0])
            if not m:
                continue
            teff, logg, feh = int(m.group(1)) * 100, int(m.group(2)) / 10, -int(m.group(3))
            values = [number(c) for c in cells[1:]]
            for b in range(3):
                a_model, w_lte, w_nlte, a_nlte, a_lte, delta = values[6 * b:6 * b + 6]
                out.append({"line": 7698.98, "teff": teff, "logg": logg, "feh": feh,
                            "a_model": a_model, "w_lte": w_lte, "w_nlte": w_nlte,
                            "a_nlte": a_nlte, "a_lte": a_lte, "delta": delta})
    return out


def bergemann2012(src: Path) -> list:
    """Sr I 4607 and Sr II 4077, 10036: rows (Teff, [Fe/H]), columns log g."""
    out = []
    for name, label, line, species in (("table3.tex", "grid1", 4607.33, "Sr1"),
                                       ("table4.tex", "grid2", 4077.71, "Sr2"),
                                       ("table5.tex", "grid3", 10036.65, "Sr2")):
        (body,) = environments((src / name).read_text(errors="replace"), rf"label\{{{label}\}}")
        rows = rows_of(body)
        logg = [number(c) for c in next(r for r in rows if r[0] == "log g")[2:]]
        for cells in rows:
            teff, feh = number(cells[0]), number(cells[1]) if len(cells) > 1 else None
            if teff is None or feh is None:
                continue
            for g, d in zip(logg, cells[2:]):
                out.append({"line": line, "species": species, "teff": teff, "logg": g,
                            "feh": feh, "delta": number(d)})
    return out


def bergemann2013(src: Path) -> list:
    """Si I J-band lines for red supergiants, xi = 2 km/s."""
    (body,) = environments((src / "tab2.tex").read_text(errors="replace"),
                           r"Non-LTE abundance corrections")
    lines = [float(x) for x in re.findall(r"\\colhead\{\$(1\d{4}\.\d+)\$\}", body)]
    out = []
    for cells in rows_of(body):
        values = [number(c) for c in cells]
        if len(values) != 3 + len(lines) or None in values[:3]:
            continue
        teff, logg, feh = values[:3]
        for line, d in zip(lines, values[3:]):
            out.append({"line": line, "teff": teff, "logg": logg, "feh": feh, "delta": d})
    return out


def write(rows: list, path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print(f"{path}: {len(rows)} rows")


def main():
    grids = Path(sys.argv[1] if len(sys.argv) > 1 else "grids")
    jobs = {
        "takeda2002_k": (takeda2002, "k1_7698_nlte_corrections.csv"),
        "bergemann2012_sr": (bergemann2012, "sr_nlte_corrections.csv"),
        "bergemann2013_si": (bergemann2013, "si1_jband_nlte_corrections.csv"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for folder, (parse, name) in jobs.items():
            src = fetch_source(ARXIV[folder], Path(tmp) / folder)
            write(parse(src), grids / folder / name)


if __name__ == "__main__":
    main()
