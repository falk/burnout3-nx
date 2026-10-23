// Shared by the native rewrites of hot recompiled functions (src/game/hot_*.cpp)
//
// A rewrite leaves the guest exactly as the original does: the same result, the same registers and the same stack
// bytes, since other code can read stack slots a function leaves behind. Build with B3_VERIFY_HOT to run the
// original after every call (Verify) and count the calls where anything differs.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "cpu/x86_ops.h"

// fused multiply-add rounds differently from the separate SSE multiply and add: no contraction here (GCC: build with
// -ffp-contract=off)
#ifdef __clang__
#pragma clang fp contract(off)
#endif

namespace hot {

typedef float f4 __attribute__((vector_size(16)));  // SSE on x86, NEON on ARM, with clang and GCC

inline f4 Splat(float f) { return f4{f, f, f, f}; }
inline float F32(uint8_t* base, uint32_t va) { return X86_MEMF32(va); }
inline uint32_t Bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline void PutXmm(X86Xmm& x, f4 v) { std::memcpy(&x, &v, 16); }
inline f4 Load4(uint8_t* base, uint32_t va) {
    f4 v;
    std::memcpy(&v, X86_PTR(uint8_t, va), 16);
    return v;
}
inline void Store4(uint8_t* base, uint32_t va, f4 v) { std::memcpy(X86_PTR(uint8_t, va), &v, 16); }
// shufps a, b, imm: two lanes of a, then two of b
template <int I>
inline f4 Shuf(f4 a, f4 b) {
    return f4{a[I & 3], a[(I >> 2) & 3], b[(I >> 4) & 3], b[(I >> 6) & 3]};
}
inline f4 Scalar(float f) { return f4{f, 0, 0, 0}; }  // movss from memory: the upper lanes cleared
inline f4 Lane0(f4 v, float f) {                      // addss, mulss, divss: lane 0 replaced
    v[0] = f;
    return v;
}
// comiss a, b then jnb: taken when a >= b, never when either is NaN; then jnbe: when a > b
inline bool Jnb(float a, float b) { return a >= b; }
inline bool Jnbe(float a, float b) { return a > b; }

}  // namespace hot

#ifdef B3_VERIFY_HOT
namespace hot {

// runs the native function, then the original from the same state, and compares registers and the given memory
struct Region {
    uint32_t va, size;
};

template <size_t N>
void Verify(const char* name, X86Context& ctx, uint8_t* base, void (*native)(X86Context&, uint8_t*),
            void (*original)(X86Context&, uint8_t*), const Region (&regions)[N], std::atomic<uint64_t>& calls,
            std::atomic<uint64_t>& bad) {
    uint8_t before[N][256], after[N][256];  // regions up to 256 bytes
    for (size_t i = 0; i < N; i++) std::memcpy(before[i], X86_PTR(uint8_t, regions[i].va), regions[i].size);
    const X86Context start = ctx;

    native(ctx, base);
    const X86Context mine = ctx;
    for (size_t i = 0; i < N; i++) std::memcpy(after[i], X86_PTR(uint8_t, regions[i].va), regions[i].size);

    for (size_t i = 0; i < N; i++) std::memcpy(X86_PTR(uint8_t, regions[i].va), before[i], regions[i].size);
    ctx = start;
    original(ctx, base);

    bool same = std::memcmp(&mine, &ctx, offsetof(X86Context, cf)) == 0 &&
                std::memcmp(&mine.xmm, &ctx.xmm, sizeof ctx.xmm) == 0;
    for (size_t i = 0; i < N; i++)
        same = same && std::memcmp(after[i], X86_PTR(uint8_t, regions[i].va), regions[i].size) == 0;

    uint64_t n = ++calls;
    if (!same && ++bad <= 8) std::fprintf(stderr, "verify %s: mismatch at call %llu\n", name, (unsigned long long)n);
    if ((n & (n - 1)) == 0 && n >= 1024)  // at each power of two
        std::fprintf(stderr, "verify %s: %llu calls, %llu mismatches\n", name, (unsigned long long)n,
                     (unsigned long long)bad.load());
}
}  // namespace hot
#endif
