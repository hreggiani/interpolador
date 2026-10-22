"""Rebuild the INASAN_NLTE correction grids by querying the batch form at every grid node.

The server interpolates linearly in its grids, so at the nodes it returns the
tabulated values.  Nodes follow Mashonkina et al. 2023 (MNRAS 524, 3526): MARCS
grid, Teff step 250 K, log g step 0.5, [Fe/H] step 0.5, plus the per-species
[X/Fe] values listed there.  Nodes without a model come back as
"Error: Out of parameters" and are dropped.

usage: inasan_pull.py GRIDS_DIR WORK_DIR
Per-line raw responses are cached in WORK_DIR so an interrupted run resumes.
"""
import itertools
import os
import re
import subprocess
import sys
import time

URL = 'https://spectrum.inasan.ru/nLTE2/nLTE2.cgi'
PAGE = 'https://spectrum.inasan.ru/nLTE2/'

# species: (output folder, reference, [Fe/H] range, [X/Fe] nodes)
SPECIES = {
    'Na1': ('alexeeva2014_na', 'Alexeeva, Pakhomov & Mashonkina 2014, Astron. Lett. 40, 406', (-5, -2), [-0.6, -0.3, 0.0, 0.3, 0.6]),
    'Mg1': ('mashonkina2013_mg', 'Mashonkina 2013, A&A 550, A28', (-5, -2), [0.0]),
    'Ca1': ('mashonkina2017_ca', 'Mashonkina, Sitnova & Belyaev 2017, A&A 605, A53', (-5, -2), [0.0, 0.4]),
    'Ti2': ('sitnova2016_ti', 'Sitnova, Mashonkina & Ryabchikova 2016, MNRAS 461, 1000', (-4, 0), [0.0]),
    'Fe1': ('mashonkina2011_fe', 'Mashonkina et al. 2011, A&A 528, A87', (-5, -2), [0.0]),
    'Sr2': ('mashonkina2022_sr', 'Mashonkina et al. 2022, MNRAS 509, 3626', (-5, -2), [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]),
    'Ba2': ('mashonkina2019_ba', 'Mashonkina & Belyaev 2019, Astron. Lett. 45, 341', (-5, -2), [-1.5, -1.0, -0.5, 0.0, 0.5]),
    'Eu2': ('mashonkina2000_eu', 'Mashonkina & Gehren 2000, A&A 364, 249; Mashonkina & Yakovleva 2026 (subm.)', (-4, 0), [-0.5, 0.0, 0.5, 1.0, 1.5]),
}


def curl(*args):
    return subprocess.run(['curl', '-sS', '-m', '600', '--retry', '3', '--retry-delay', '20', *args],
                          check=True, capture_output=True, text=True).stdout


def nodes(feh_range, xfe):
    teff = range(4000, 6501, 250)
    logg = [0.5 * i for i in range(1, 11)]
    lo, hi = feh_range
    feh = [lo + 0.5 * i for i in range(int(round((hi - lo) / 0.5)) + 1)]
    return [f'{t} {g:.2f} {f:.2f} {x:.2f}' for t, g, f, x in itertools.product(teff, logg, feh, xfe)]


def merge_split_lines(rows, gap=0.1):
    """Join entries the server lists twice under near-identical wavelengths.

    The Na I lines appear as 5682.64 and 5682.6401, 5895.92 and 5895.8398, ...:
    one entry holds the dwarf grid (Teff >= 5000 K, log g >= 2), the other the
    giant grid, and where both define a node they agree exactly.  Each pair is
    merged under the wavelength of its larger entry -- the giant-grid copy of
    Na I D1 is listed at 5895.8398, 0.08 A off the line -- and a disagreement on
    a shared node is an error, not something to average.
    """
    by_line = {}
    for lam, f in rows:
        by_line.setdefault(lam, {})[tuple(f[:4])] = f[4:]
    lams = sorted(by_line, key=float)
    clusters = [[lams[0]]]
    for lam in lams[1:]:
        if float(lam) - float(clusters[-1][-1]) < gap:
            clusters[-1].append(lam)
        else:
            clusters.append([lam])
    out = []
    for cluster in clusters:
        label = max(cluster, key=lambda lam: len(by_line[lam]))
        merged = {}
        for lam in cluster:
            for node, values in by_line[lam].items():
                if node in merged and merged[node] != values:
                    sys.exit(f'{cluster}: entries disagree at {node}: {merged[node]} vs {values}')
                merged[node] = values
        out.extend((label, list(node) + values) for node, values in merged.items())
    return out


def main():
    grids, work = sys.argv[1:3]
    os.makedirs(work, exist_ok=True)
    page = curl(PAGE)
    lams = {sp: re.findall(r"'([\d.]+)'", m) for sp, m in re.findall(r"var lams_(\w+) = \[(.*?)\];", page)}
    for sp, (folder, ref, feh_range, xfe) in SPECIES.items():
        star_file = os.path.join(work, f'{sp}_nodes.txt')
        rows = nodes(feh_range, xfe)
        open(star_file, 'w').write('\n'.join(rows) + '\n')
        out_rows, dropped, odd = [], 0, []
        for lam in lams[sp]:
            raw_path = os.path.join(work, f'{sp}_{lam}.raw')
            if not os.path.exists(raw_path):
                body = curl('-F', f'elem2={sp}', '-F', f'lam2={lam}', '-F', 'parlist=',
                            '-F', f'parfile=@{star_file};type=text/plain', '-F', 'res2=file',
                            '-H', f'Referer: {PAGE}', URL)
                if body.count('\n') < len(rows) * 0.9:
                    sys.exit(f'{sp} {lam}: short response ({body.count(chr(10))} lines)\n{body[:500]}')
                open(raw_path, 'w').write(body)
                print(sp, lam, 'fetched', flush=True)
                time.sleep(2)
            for line in open(raw_path):
                f = line.split()
                if len(f) == 6:
                    out_rows.append((lam, f))
                elif 'Out of parameters' in line:
                    dropped += 1
                elif line.strip():
                    odd.append(f'{lam} {line.rstrip()}')
        out_rows = [f'{lam:>10} ' + ' '.join(f) for lam, f in merge_split_lines(out_rows)]
        os.makedirs(os.path.join(grids, folder), exist_ok=True)
        dest = os.path.join(grids, folder, f'inasan_nlte_{sp}.dat')
        with open(dest, 'w') as fh:
            fh.write(f'# INASAN_NLTE grid for {sp}: {ref}\n'
                     f'# Database: Mashonkina, Pakhomov, Sitnova et al. 2023, MNRAS 524, 3526; {PAGE}\n'
                     f'# Rebuilt {time.strftime("%Y-%m-%d")} by querying the batch form (nLTE2.cgi) at the grid nodes\n'
                     f'# (MARCS: Teff step 250 K, log g step 0.5, [Fe/H] step 0.5; [X/Fe] = {xfe}).\n'
                     f'# Nodes the server reports as "Out of parameters" are omitted ({dropped} of {len(rows) * len(lams[sp])}).\n'
                     f'# Delta = A(NLTE) - A(LTE) in dex; EW in mA as returned by the server.\n'
                     f'# line_A teff logg feh xfe delta ew\n')
            fh.write('\n'.join(out_rows) + '\n')
        print(f'{sp}: {len(lams[sp])} lines, {len(out_rows)} rows -> {dest}; dropped {dropped}; odd {len(odd)}', flush=True)
        for o in odd[:20]:
            print('   ODD', o, flush=True)


if __name__ == '__main__':
    main()
