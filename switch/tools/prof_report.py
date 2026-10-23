#!/usr/bin/env python3
"""prof_report.py LOG_OR_TABLE burnout3.elf [top]: console profiler samples with function names.

Lines of prof.txt: "T thread samples", then "S func count" (self), "I func count" (inclusive), "P pc count" (by
instruction), offsets from the module's start. Functions come from nm of the same build's burnout3.elf."""
import bisect, collections, os, subprocess, sys

table, elf = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
out = subprocess.run([os.path.join(os.environ.get('DEVKITPRO', '/opt/devkitpro'), 'devkitA64/bin/aarch64-none-elf-nm'), '-n', '-C', '--defined-only', elf],
                     capture_output=True, text=True, check=True).stdout
starts, names = [], []
for line in out.splitlines():
    p = line.split(' ', 2)
    if len(p) == 3 and p[1] in 'TtWw':
        a = int(p[0], 16)
        if starts and starts[-1] == a:
            if len(p[2]) < len(names[-1]):
                names[-1] = p[2]
            continue
        starts.append(a)
        names.append(p[2])

def name(off):
    if off == 0xFFFFFFFF:
        return '(outside the module)'
    i = bisect.bisect_right(starts, off) - 1
    return names[i] if i >= 0 else '?'

threads = collections.OrderedDict()
cur = None
for line in open(table):
    if 'profdata: ' in line:
        line = line.split('profdata: ', 1)[1]
    parts = line.split()
    if not parts or parts[0] not in ('T', 'S', 'I', 'P'):
        continue
    for offset in range(0, len(parts), 3):
        k, a, b = parts[offset:offset + 3]
        if k == 'T':
            # Streamed snapshots are cumulative: the newest replaces the old one.
            cur = threads[a] = {'n': int(b), 'S': collections.Counter(), 'I': collections.Counter(), 'P': collections.Counter()}
        elif cur is not None:
            cur[k][int(a, 16)] += int(b)
for tid, t in threads.items():
    print(f'thread {tid}: {t["n"]} samples')
    for k, what in (('S', 'self'), ('I', 'incl')):
        agg = collections.Counter()
        for off, c in t[k].items():
            agg[name(off)] += c
        for n, c in agg.most_common(top):
            print(f'  {what} {100.0 * c / t["n"]:5.1f}%  {n[:110]}')
