#!/usr/bin/env python3
"""Replay production state deltas and check captured payloads on x86 or AArch64."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('runtime', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--arch', choices=['x86', 'a64'], default='x86')
p.add_argument('--source', type=Path)
p.add_argument('--bench', action='store_true')
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
source = (a.source or repo / 'src/d3d/render_thread.cpp').read_text()
a.output.mkdir(parents=True, exist_ok=True)
def function(signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]
# Exercise the same SIMD comparison as production, without the OS/backend.
start = source.index('#ifdef __SWITCH__\n// the bits')
end = source.index('\nusing Clock', start)
functions = '\n'.join(function(s) for s in [
    'size_t At(', 'size_t SimpleBytes(', 'std::vector<Range> StateRanges(', 'std::vector<Range> FixedRanges('
])
if 'size_t ResourceSlot(' in source:
    functions += '\n' + function('size_t ResourceSlot(')
if 'bool SameHeader(' in source:
    functions += '\n' + function('bool SameHeader(')
methods = ['uint32_t Put(', 'void Delta(']
if 'void StoreDelta(' in source:
    methods.append('void StoreDelta(')
methods.append('void SendRange(')
text = '''#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <mutex>
#include <vector>
#include <unordered_map>
#include "gpu/state.h"
#include "gpu/shaders.h"
#include "gpu/texture.h"
#ifdef __aarch64__
#define __SWITCH__
#include <arm_neon.h>
#else
#include <smmintrin.h>
#endif
using namespace gpu;
struct Range { size_t off, len; };
''' + functions + '\n' + source[start:end] + '''
struct Batch { uint64_t id = 1; std::vector<uint8_t> bytes; };
struct Capture {
    State sent_{};
    Batch batch;
    Batch* cur_ = &batch;
''' + '\n'.join(function(s) for s in methods) + '\n};\n'
d3d_source = (repo / 'src/d3d/d3d8.cpp').read_text()
saved_source = source
source = d3d_source
text += """
static State g_state;
static std::recursive_mutex g_lock;
static std::vector<uint8_t> guest(2 << 20);
constexpr uint32_t kRenderState = 0, kTextureState = 4096, kStageStates = 32, kVsConstants = 192;
constexpr uint32_t kDevice = 0x100000, XDK_CDevice_ScreenSpaceOffset = 0xEF8, XDK_CDevice_Viewport = 0xEE0;
constexpr uint32_t kViewportScale = 58, kViewportOffset = 59;
float g_view_scale = 1.0f;
float g_near_depth = 0.0f;
void* HostPtr(uint32_t p) { return guest.data() + p; }
uint32_t R32(uint32_t p) { uint32_t v; std::memcpy(&v, HostPtr(p), 4); return v; }
void W32(uint32_t p, uint32_t v) { std::memcpy(HostPtr(p), &v, 4); }
void WriteFloat(uint32_t p, float v) { std::memcpy(HostPtr(p), &v, 4); }
float ArgF(const uint32_t* a, unsigned i) { float v; std::memcpy(&v, a + i, 4); return v; }
float ZScale(const Surface&) { return 16777215.0f; }
#define HLE_FASTCALL(name, n) uint32_t name(const uint32_t* a)
#define HLE_STDCALL(name, n) uint32_t name(const uint32_t* a)
"""
text += '\n'.join(function(s) for s in ['void UpdateViewportConstants(',
    'HLE_STDCALL(D3DDevice_SetScreenSpaceOffset,', 'void SyncDeferred(',
    'HLE_FASTCALL(D3DDevice_SetVertexShaderConstant4,',
    'HLE_FASTCALL(D3DDevice_SetVertexShaderConstantNotInlineFast,'])
text += """
struct X86Context {};
namespace d3d {
gpu::State& State() { return g_state; }
std::recursive_mutex& Lock() { return g_lock; }
}
"""
source = (repo / 'src/game/hooks.cpp').read_text()
text += function('void StoreViewport(')
source = saved_source
text += """
namespace kernel {
uint32_t R32(uint32_t p) { return ::R32(p); }
void* HostPtr(uint32_t p) { return ::HostPtr(p); }
}
constexpr uint32_t kHeaderBytes = 20, kHeaderSlot = 32, kHeaderRing = 1u << 20;
struct HeaderCapture {
    struct HeaderCopy { uint32_t at = 0; uint64_t made = 0; uint8_t bytes[20]; };
    template<class T> struct Recent { uint32_t key = 0; T* value = nullptr; };
    std::unordered_map<uint32_t, HeaderCopy> headers_;
    std::array<Recent<HeaderCopy>, 64> recent_headers_{};
    std::array<Recent<uint64_t>, 64> recent_uses_{};
    std::unordered_map<uint32_t, uint64_t> last_use_;
    Batch batch;
    Batch* cur_ = &batch;
    uint32_t ring_ = 0x10000;
    uint64_t ring_pos_ = 0, header_copies_ = 0;
""" + '\n'.join(function(s) for s in ['uint32_t Scratch(', 'uint32_t Header(', 'void Use(']) + '\n};\n'
source = (repo / 'switch/deko3d/deko3d_backend.cpp').read_text()
text += '\n'.join(function(s) for s in ['uint32_t Simple(', 'bool IsGlowComposite(', 'void AlignGlow('])
text += (repo / 'tests/render_capture_test.inc').read_text()
path = a.output / 'capture.cpp'
path.write_text(text)
exe = a.output / ('capture-' + a.arch)
compiler = ['g++', '-msse4.1'] if a.arch == 'x86' else ['aarch64-linux-gnu-g++', '-static', '-mcpu=cortex-a57']
subprocess.run(compiler + ['-std=c++20', '-O2', *(['-DCAPTURE_BENCH'] if a.bench else []), '-I' + str(a.runtime.resolve()), str(path), '-o', str(exe)], check=True)
subprocess.run(([str(exe.resolve())] if a.arch == 'x86' else ['qemu-aarch64', str(exe.resolve())]), check=True)
