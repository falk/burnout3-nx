// Verification of native rewrites whose written memory is only known at run time (src/game/hot_*.cpp)
//
// VerifyAll runs the native function, then the original from the same state, and compares the registers, the xmm
// registers, the x87 stack (top, status word, the eight slots) and every given memory range. Ranges of any size are
// compared in pieces of at most 256 bytes. PathCounts counts the ways through a native function; HOT_PATH compiles
// to nothing outside B3_VERIFY_HOT.
#pragma once
#include "game/hot_paths.h"

#ifdef B3_VERIFY_HOT
#include <vector>

namespace hot {

struct PathCounts {
    static constexpr int kMax = 24;
    std::atomic<uint64_t> n[kMax];
    const char* names[kMax];
};
#define HOT_PATH(counts, i) (++(counts).n[i])

// ranges of guest memory a call may write, cut into pieces of at most 256 bytes
struct Ranges {
    std::vector<Region> pieces;
    void Add(uint32_t va, uint32_t size) {
        while (size) {
            const uint32_t n = size < 256 ? size : 256;
            pieces.push_back({va, n});
            va += n;
            size -= n;
        }
    }
};

inline void PrintPaths(const char* name, const PathCounts& p) {
    for (int i = 0; i < PathCounts::kMax; i++)
        if (p.names[i])
            std::fprintf(stderr, "verify %s: path %s %llu\n", name, p.names[i], (unsigned long long)p.n[i].load());
}

inline bool VerifyAll(const char* name, X86Context& ctx, uint8_t* base, void (*native)(X86Context&, uint8_t*),
                      void (*original)(X86Context&, uint8_t*), const Ranges& ranges, std::atomic<uint64_t>& calls,
                      std::atomic<uint64_t>& bad, const PathCounts* paths) {
    const size_t n = ranges.pieces.size();
    std::vector<uint8_t> before(n * 256), after(n * 256);
    for (size_t i = 0; i < n; i++)
        std::memcpy(&before[i * 256], X86_PTR(uint8_t, ranges.pieces[i].va), ranges.pieces[i].size);
    const X86Context start = ctx;

    native(ctx, base);
    const X86Context mine = ctx;
    for (size_t i = 0; i < n; i++)
        std::memcpy(&after[i * 256], X86_PTR(uint8_t, ranges.pieces[i].va), ranges.pieces[i].size);

    for (size_t i = 0; i < n; i++)
        std::memcpy(X86_PTR(uint8_t, ranges.pieces[i].va), &before[i * 256], ranges.pieces[i].size);
    ctx = start;
    original(ctx, base);

    bool regs = std::memcmp(&mine, &ctx, offsetof(X86Context, cf)) == 0 &&
                std::memcmp(&mine.xmm, &ctx.xmm, sizeof ctx.xmm) == 0;
    bool x87 = mine.top == ctx.top && mine.fsw == ctx.fsw && std::memcmp(mine.st, ctx.st, sizeof ctx.st) == 0;
    bool mem = true;
    for (size_t i = 0; i < n; i++)
        mem = mem &&
              std::memcmp(&after[i * 256], X86_PTR(uint8_t, ranges.pieces[i].va), ranges.pieces[i].size) == 0;

    uint64_t c = ++calls;
    if (!(regs && x87 && mem) && ++bad <= 8) {
        std::fprintf(stderr, "verify %s: mismatch at call %llu (registers %d, x87 %d, memory %d)\n", name,
                     (unsigned long long)c, regs, x87, mem);
        const uint32_t* a = &mine.eax;
        const uint32_t* b = &ctx.eax;
        for (int i = 0; i < 8; i++)
            if (a[i] != b[i]) std::fprintf(stderr, "  gpr %d: %08X, original %08X\n", i, a[i], b[i]);
        for (int i = 0; i < 8; i++)
            if (std::memcmp(&mine.xmm[i], &ctx.xmm[i], 16))
                std::fprintf(stderr, "  xmm%d: %08X %08X %08X %08X, original %08X %08X %08X %08X\n", i,
                             mine.xmm[i].u[0], mine.xmm[i].u[1], mine.xmm[i].u[2], mine.xmm[i].u[3],
                             ctx.xmm[i].u[0], ctx.xmm[i].u[1], ctx.xmm[i].u[2], ctx.xmm[i].u[3]);
        for (size_t i = 0; i < n; i++)
            for (uint32_t k = 0; k < ranges.pieces[i].size; k += 4) {
                uint32_t x, y;
                std::memcpy(&x, &after[i * 256 + k], 4);
                std::memcpy(&y, X86_PTR(uint8_t, ranges.pieces[i].va + k), 4);
                if (x != y)
                    std::fprintf(stderr, "  [%08X]: %08X, original %08X\n", ranges.pieces[i].va + k, x, y);
            }
    }
    if ((c & (c - 1)) == 0 && c >= 1024) {
        std::fprintf(stderr, "verify %s: %llu calls, %llu mismatches\n", name, (unsigned long long)c,
                     (unsigned long long)bad.load());
        if (paths) PrintPaths(name, *paths);
    }
    return regs && x87 && mem;
}

// a synthetic call in the dead stack below the live esp: setup writes the operands into [lo, lo + size) and sets the
// registers (esp inside the range, with room for the frames below); native and original are compared as in
// VerifyAll, then the bytes are put back
template <class Setup>
inline void VerifyCase(const char* name, const X86Context& live, uint8_t* base,
                       void (*native)(X86Context&, uint8_t*), void (*original)(X86Context&, uint8_t*), uint32_t lo,
                       uint32_t size, Setup setup) {
    std::vector<uint8_t> keep(X86_PTR(uint8_t, lo), X86_PTR(uint8_t, lo) + size);
    X86Context c = live;
    setup(c);
    Ranges r;
    r.Add(lo, size);
    std::atomic<uint64_t> calls{0}, bad{0};
    const bool same = VerifyAll(name, c, base, native, original, r, calls, bad, nullptr);
    std::fprintf(stderr, "verify %s: synthetic case %s\n", name, same ? "same" : "DIFFERENT");
    std::memcpy(X86_PTR(uint8_t, lo), keep.data(), size);
}

// synthetic NaN operands: when both operands of addps/mulps (or the ss forms) are NaN the result is the first one's,
// so each case puts NaNs of distinct bits in the lanes its mask names (bit i: lane i) of every input vector
struct NanCase {
    const char* what;
    uint32_t lanes;
    bool signaling;
};
inline constexpr NanCase kNanCases[] = {
    {"no NaN", 0, false},          {"NaN w lanes", 8, false},           {"NaN x lanes", 1, false},
    {"NaN y and z lanes", 6, false}, {"NaN in every lane", 15, false},  {"signaling NaN w lanes", 8, true},
    {"signaling NaN x lanes", 1, true},
};
inline constexpr int kNanCaseCount = sizeof kNanCases / sizeof kNanCases[0];
// a NaN for input v, lane l, case k: quiet (or signaling, the quiet bit clear), the sign from v + l
inline uint32_t NanBits(uint32_t v, uint32_t l, uint32_t k, bool signaling) {
    return (((v + l) & 1u) << 31) | (signaling ? 0x7F800000u : 0x7FC00000u) | ((v & 0xFFu) << 12) | (l << 8) |
           ((k & 0xFu) << 4) | 1u;
}
// a 16-byte vector at va: x, y, z, w, with case k's NaNs in place of the lanes its mask names
inline void PutNanVector(uint8_t* base, uint32_t va, float x, float y, float z, float w, uint32_t v, int k) {
    const float f[4] = {x, y, z, w};
    for (uint32_t l = 0; l < 4; l++) {
        if (kNanCases[k].lanes >> l & 1) X86_MEM32(va + 4 * l) = NanBits(v, l, k, kNanCases[k].signaling);
        else X86_MEMF32(va + 4 * l) = f[l];
    }
}

}  // namespace hot
#else
#define HOT_PATH(counts, i) ((void)0)
#endif
