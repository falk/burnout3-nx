// sub_00013CA0: out (eax) = v.x * m[0] + m[3] + v.y * m[1] + v.z * m[2], a point through a 4x4 matrix (cdecl: m, v),
// RenderWare's, the same code as Black's sub_00012600 (Black src/game/hot_vec.cpp). The rewrite keeps the original's
// operand order, its stores (the saved registers, the frame, the result) and the registers and xmm it leaves.
// Build with B3_VERIFY_HOT to run the original after every call and count the calls where anything differs.
#include <atomic>

#include "game/hot_ordered.h"
#include "game/hot_verify.h"

using namespace hot;

X86_EXTERN_FUNC(__imp__sub_00013CA0);

namespace {

#ifdef B3_VERIFY_HOT
PathCounts g_point_paths{{}, {"native", "operands-in-frame"}};
#endif

inline bool Overlap(uint32_t a, uint64_t an, uint32_t b, uint64_t bn) {
    return uint64_t(a) < uint64_t(b) + bn && uint64_t(b) < uint64_t(a) + an;
}

// Frame: push ebp, the alignment, sub esp, 0x18, push esi, push edi; the sum at [esp+0x10], copied to out a dword at
// a time
void TransformPoint(X86Context& __restrict ctx, uint8_t* base) {
    const uint32_t e = ctx.esp;
    const uint32_t s = ((e - 4) & ~15u) - 0x20;
    const uint32_t m = X86_MEM32(e + 4), v = X86_MEM32(e + 8), out = ctx.eax;
    if (Overlap(m, 0x40, s, e - s) || Overlap(v, 12, s, e - s) || Overlap(out, 16, s, e - s)) {
        HOT_PATH(g_point_paths, 1);
        __imp__sub_00013CA0(ctx, base);
        return;
    }
    HOT_PATH(g_point_paths, 0);
    X86_MEM32(e - 4) = ctx.ebp;
    X86_MEM32(s + 4) = ctx.esi;
    X86_MEM32(s) = ctx.edi;
    f4 x0 = VMul(Splat(F32(base, v)), Load4(base, m));
    x0 = VAdd(x0, Load4(base, m + 0x30));
    f4 x1 = VMul(Splat(F32(base, v + 4)), Load4(base, m + 0x10));
    x0 = VAdd(x0, x1);
    x1 = VMul(Splat(F32(base, v + 8)), Load4(base, m + 0x20));
    x0 = VAdd(x0, x1);
    Store4(base, s + 0x10, x0);
    Store4(base, out, x0);
    ctx.ecx = out;
    ctx.edx = X86_MEM32(s + 0x1C);
    PutXmm(ctx.xmm[0], x0);
    PutXmm(ctx.xmm[1], x1);
    ctx.ebp = X86_MEM32(e - 4);
    ctx.esp = e + 4;
}

#ifdef B3_VERIFY_HOT
std::atomic<uint64_t> g_point_calls{0}, g_point_bad{0};
#endif
}  // namespace

X86_FUNC(sub_00013CA0) {
#ifndef B3_VERIFY_HOT
    TransformPoint(ctx, base);
#else
    Ranges r;
    r.Add(ctx.esp - 0x40, 0x40);
    r.Add(ctx.eax, 16);
    VerifyAll("sub_00013CA0", ctx, base, TransformPoint, __imp__sub_00013CA0, r, g_point_calls, g_point_bad, &g_point_paths);
#endif
}
