#!/usr/bin/env python3
"""Differential hot-path tests against the current generated originals.

Usage: test_hot_candidates.py GENERATED SDK_RUNTIME OUTPUT [--arch x86|a64]
Tests raw float bits, guest scratch/register state and alias fallbacks.
Bench numbers on Linux/qemu are diagnostic, not estimates of Switch performance.
"""
import argparse
import pathlib
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('generated', type=pathlib.Path)
p.add_argument('runtime', type=pathlib.Path)
p.add_argument('output', type=pathlib.Path)
p.add_argument('--arch', choices=['x86', 'a64'], default='x86')
p.add_argument('--bench', action='store_true')
p.add_argument('--newlib', type=pathlib.Path, help='Link Switch libm for ARM64 particle trig validation')
a = p.parse_args()
repo = pathlib.Path(__file__).resolve().parents[2]
a.output.mkdir(parents=True, exist_ok=True)
functions = {}
needed = {'0015D320', '0015B380', '0015BF30', '00011DD0', '0015CFF0', '0015C620', '00011C10', '00140960', '000A34D4', '000205A0', '000CCB30', '000CC6C0', '000CFE40', '00011D40', '000A20F0', '000A2080', '00156D80', '0026C410', '000D1D10', '000CD040'}
for file in a.generated.glob('x86_recomp.*.cpp'):
    text = file.read_text()
    for match in re.finditer(r'X86_FUNC\(__imp__sub_([0-9A-F]+)\)', text):
        name = match[1]
        if name in needed:
            end = text.find('\nX86_FUNC(', match.end())
            functions[name] = text[match.start():end if end >= 0 else len(text)]
if needed - functions.keys():
    raise SystemExit(f'Missing generated functions: {needed - functions.keys()}')
includes = ['-I' + str(repo / 'src'), '-I' + str(a.runtime.resolve()), '-I' + str(a.generated.resolve())]
flags = ['-std=c++20', '-O2', '-fno-strict-aliasing', '-ffp-contract=off', '-ffunction-sections', '-fdata-sections']
if a.newlib:
    if a.arch != 'a64':
        p.error('--newlib requires --arch a64')
    flags += ['-DB3_TEST_NEWLIB', '-fno-builtin-sin', '-fno-builtin-cos']
for name, funcs, source in [
    ('mesh_ray', ['000D1D10', '000CD040', '000205A0', '00011C10'], 'hot_paths.cpp'),
    ('resource_release', ['0026C410'], None),
    ('render_queue', ['000A20F0', '000A2080', '00156D80'], 'hot_render_queue.cpp'),
    ('collision_tree', ['000CFE40', '000205A0', '000CCB30', '000CC6C0', '00011D40'], 'hot_paths.cpp'),
    ('decode', ['000205A0', '000CCB30', '000CC6C0', '00011D40'], 'hot_paths.cpp'),
    ('ftol', ['000A34D4'], 'hot_ftol.cpp'),
    ('adpcm', ['00140960'], 'hot_adpcm.cpp'),
    ('quad', ['0015CFF0', '0015C620', '00011C10'], 'hot_quad.cpp'),
    ('billboard', ['0015D320'], 'hot_billboard.cpp'),
    ('particle_batch', ['0015D320', '0015B380', '0015BF30', '00011DD0'], 'hot_particles.cpp'),
]:
    original = '#include "cpu/x86_ops.h"\n#include <cstdlib>\n'
    if a.newlib:
        original += '#include <cerrno>\nextern "C" int* __errno() { return &errno; }\n'
    for fn in funcs + ['0002AC00', '0026C360', '000205A0', '000CCB30', '00011C10']:
        original += f'X86_EXTERN_FUNC(sub_{fn});\nX86_EXTERN_FUNC(__imp__sub_{fn});\n'
    original += '\n'.join(functions[fn] for fn in funcs)
    if name == 'collision_tree':
        for fn in ['000CFE40', '000CC6C0']:
            original = original.replace(' sub_' + fn + '(ctx, base);', ' __imp__sub_' + fn + '(ctx, base);')
    if name == 'render_queue':
        original = original.replace(' sub_000A2080(ctx, base);', ' __imp__sub_000A2080(ctx, base);')
    if name == 'collision_tree':
        baseline = re.sub(r'X86_WEAK_ALIAS[^\n]*', '', functions['000CFE40'])
        baseline = baseline.replace('__imp__sub_000CFE40', 'baseline_collision_walk')
        baseline = baseline.replace('sub_000CFE40(ctx, base);', 'baseline_collision_walk(ctx, base);')
        original += '\n' + baseline
    if name == 'resource_release':
        # Exercise the actual generic call wrapper, not an imitation in the test.
        dispatch = (a.runtime / 'cpu/dispatch.cpp').read_text()
        start = dispatch.index('uint32_t CallGuest(X86Context&')
        end = dispatch.index('\n}\n', start) + 2
        original += '\n#include "kernel/kernel.h"\nnamespace kernel {\n' + dispatch[start:end] + '\n}\n'
    path = a.output / (name + '-original.cpp')
    path.write_text(original)
    exe = a.output / (name + '-' + a.arch)
    # Compile the reference with clang, as in the Switch build.
    obj = a.output / (name + '-original.o')
    compiler = ['clang++']
    if a.arch == 'a64':
        compiler += ['--target=aarch64-linux-gnu', '-mcpu=cortex-a57']
    subprocess.run(compiler + flags + includes + ['-c', str(path), '-o', str(obj)], check=True)
    compiler = ['g++'] if a.arch == 'x86' else ['aarch64-linux-gnu-g++', '-static', '-mcpu=cortex-a57']
    subprocess.run(compiler + flags + includes + [str(repo / 'tests' / ('hot_' + name + '_test.cpp')), str(obj),
        *([str(repo / 'src/game' / source)] if source else []), *([str(repo / 'src/game/hot_billboard.cpp')] if name == 'particle_batch' else []), *([str(a.newlib.resolve())] if a.newlib else []), '-Wl,--gc-sections', '-o', str(exe)], check=True)
    run = ([str(exe.resolve())] if a.arch == 'x86' else ['qemu-aarch64', str(exe.resolve())])
    subprocess.run(run + (['bench'] if a.bench else []), check=True)

if a.newlib:
    exe = a.output / 'trig-newlib-a64'
    subprocess.run(['aarch64-linux-gnu-g++', '-static', '-mcpu=cortex-a57'] + flags + includes +
        [str(repo / 'tests/hot_trig_test.cpp'), str(a.newlib.resolve()), '-o', str(exe)], check=True)
    subprocess.run(['qemu-aarch64', str(exe.resolve())] + (['bench'] if a.bench else []), check=True)
