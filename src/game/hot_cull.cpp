// sub_001B23F0 (cdecl: planes, corners): the box test of the world renderer's culling (sub_001AD5E0 for every
// object, about 3% of the game thread in a race). The 8 corners of a box (16 bytes each) go through two sets of 4
// planes, the first set given as rows at planes + 0x20..0x50, the second built from the plane array at planes and
// planes + 0x10 with shufps. Per corner: x * r0 + y * r1 + z * r2 - r3, movmskps of the 4 lanes. Any set where some
// corner is outside every plane of a lane (the or of the masks not 0xF) returns 0 (outside); after both sets, 1 when
// some corner was not inside all 4 planes (bl, from the last cmovne), else 2 (inside).
//
// The rewrite leaves what the original leaves: eax, ecx (its high half and the counters), edx (the last or), the
// xmm registers of the last corner and pass, the four pushed registers below esp. Build with B3_VERIFY_HOT to run
// the original after every call and count the calls where anything differs.
#include <atomic>

#include "game/hot_ordered.h"
#include "game/hot_verify.h"

using namespace hot;

X86_EXTERN_FUNC(__imp__sub_001B23F0);

namespace {

#ifdef B3_VERIFY_HOT
PathCounts g_box_paths{{}, {"outside first set", "outside second set", "partly", "inside"}};
#endif

inline uint32_t MoveMask(f4 v) {
    return Bits(v[0]) >> 31 | (Bits(v[1]) >> 31) << 1 | (Bits(v[2]) >> 31) << 2 | (Bits(v[3]) >> 31) << 3;
}

void BoxTest(X86Context& __restrict ctx, uint8_t* base) {
    const uint32_t e = ctx.esp;
    // push ebp, mov ebp, esp, push ebx, push esi, push edi
    X86_MEM32(e - 4) = ctx.ebp;
    X86_MEM32(e - 8) = ctx.ebx;
    X86_MEM32(e - 12) = ctx.esi;
    X86_MEM32(e - 16) = ctx.edi;
    const uint32_t planes = X86_MEM32(e + 4), corners = X86_MEM32(e + 8);
    uint32_t ebx = 0, ecx = (ctx.ecx & 0xFFFF0000u) | 0x200u, edx = 0, eax = 0;
    f4 r0 = Load4(base, planes + 0x20), r1 = Load4(base, planes + 0x30), r2 = Load4(base, planes + 0x40),
       r3 = Load4(base, planes + 0x50);
    f4 x0{}, x1{}, x2{}, x3{};
    for (;;) {
        edx = 0;
        ecx = (ecx & 0xFFFFFF00u) | 8u;
        uint32_t c = corners;
        for (int i = 0; i < 8; i++, c += 16) {
            x0 = Load4(base, c);
            x1 = VMul(Splat(x0[0]), r0);
            x2 = VMul(Splat(x0[1]), r1);
            x3 = VMul(Splat(x0[2]), r2);
            x1 = VAdd(x1, x2);
            x1 = VAdd(x1, x3);
            x1 = VSub(x1, r3);
            eax = MoveMask(x1);
            if (eax != 0xF) ebx = ecx;
            edx |= eax;
            ecx = (ecx & 0xFFFFFF00u) | ((ecx - 1) & 0xFFu);
        }
        eax = 0;
        if (edx != 0xF) {
            HOT_PATH(g_box_paths, (ecx >> 8 & 0xFF) == 2 ? 0 : 1);
            break;
        }
        const uint32_t ch = ((ecx >> 8) - 1) & 0xFF;
        ecx = (ecx & 0xFFFF00FFu) | ch << 8;
        if (!ch) {
            eax = (ebx & 0xFF) ? 1 : 2;
            HOT_PATH(g_box_paths, eax == 1 ? 2 : 3);
            ecx = 1;
            break;
        }
        x0 = Load4(base, planes);
        x1 = Load4(base, planes + 0x10);
        r0 = f4{x0[0], x0[0], x1[0], x1[0]};
        r1 = f4{x0[1], x0[1], x1[1], x1[1]};
        r2 = f4{x0[2], x0[2], x1[2], x1[2]};
        r3 = f4{x0[3], x0[3], x1[3], x1[3]};
    }
    ctx.eax = eax;
    ctx.ecx = ecx;
    ctx.edx = edx;
    PutXmm(ctx.xmm[0], x0);
    PutXmm(ctx.xmm[1], x1);
    PutXmm(ctx.xmm[2], x2);
    PutXmm(ctx.xmm[3], x3);
    PutXmm(ctx.xmm[4], r0);
    PutXmm(ctx.xmm[5], r1);
    PutXmm(ctx.xmm[6], r2);
    PutXmm(ctx.xmm[7], r3);
    ctx.esp = e + 4;  // ret: ebp, ebx, esi and edi as they were
}

#ifdef B3_VERIFY_HOT
std::atomic<uint64_t> g_box_calls{0}, g_box_bad{0};
#endif
}  // namespace

X86_FUNC(sub_001B23F0) {
#ifndef B3_VERIFY_HOT
    BoxTest(ctx, base);
#else
    Ranges r;
    r.Add(ctx.esp - 0x10, 0x10);
    VerifyAll("sub_001B23F0", ctx, base, BoxTest, __imp__sub_001B23F0, r, g_box_calls, g_box_bad, &g_box_paths);
#endif
}
