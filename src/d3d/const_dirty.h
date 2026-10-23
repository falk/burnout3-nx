// The constant range of State (vsc, psc and the viewport, up to rt) in 64 byte pieces, a bit each: d3d8.cpp marks
// the pieces it writes, the render thread (render_thread.cpp) sends only those instead of comparing the 3.3 KB on
// every draw. A write that misses MarkConstants is found by the render thread's periodic full comparison.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "gpu/state.h"

namespace gpu {

constexpr size_t kConstPiece = 64;
static_assert((offsetof(State, rt) - offsetof(State, vsc) + kConstPiece - 1) / kConstPiece <= 64);

extern std::atomic<uint64_t> g_const_dirty;

// n bytes at p, inside s's constant range
inline void MarkConstants(const State& s, const void* p, size_t n) {
    if (!n) return;
    size_t off = size_t(static_cast<const uint8_t*>(p) - reinterpret_cast<const uint8_t*>(&s.vsc));
    size_t first = off / kConstPiece, last = (off + n - 1) / kConstPiece;
    uint64_t bits = (last >= 63 ? ~0ull : (2ull << last) - 1) & ~((1ull << first) - 1);
    g_const_dirty.fetch_or(bits, std::memory_order_relaxed);
}

inline void MarkAllConstants() { g_const_dirty.store(~0ull, std::memory_order_relaxed); }

}  // namespace gpu
