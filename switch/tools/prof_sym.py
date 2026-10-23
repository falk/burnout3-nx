#!/usr/bin/env python3
"""prof_sym.py burnout3.elf burnout3.sym: the function table for the console profiler (switch/headless/b3_prof.cpp)."""
import os, struct, subprocess, sys

elf, out = sys.argv[1], sys.argv[2]
nm = os.path.join(os.environ.get('DEVKITPRO', '/opt/devkitpro'), 'devkitA64/bin/aarch64-none-elf-nm')
lines = subprocess.run([nm, '-n', '-C', '--defined-only', elf], capture_output=True, text=True, check=True).stdout
syms = {}
anchor = None
for line in lines.splitlines():
    parts = line.split(' ', 2)
    if len(parts) < 3 or parts[1] not in 'TtWw':
        continue
    addr = int(parts[0], 16)
    name = parts[2]
    if name == 'b3_prof_anchor':
        anchor = addr
    if addr >= 1 << 32 or name.startswith('$'):
        continue
    # several names at one address: keep the shortest readable one (sub_X over __imp__sub_X aliases)
    if addr not in syms or len(name) < len(syms[addr]):
        syms[addr] = name[:120]
if anchor is None:
    sys.exit('no b3_prof_anchor in ' + elf)
starts = sorted(syms)
names = b''
pairs = []
for a in starts:
    pairs.append((a, len(names)))
    names += syms[a].encode() + b'\0'
with open(out, 'wb') as f:
    f.write(struct.pack('<4sII', b'BSYM', len(starts), anchor))
    for a, o in pairs:
        f.write(struct.pack('<II', a, o))
    f.write(names)
print(f'{len(starts)} functions, anchor {anchor:X}, {out}')
