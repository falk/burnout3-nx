#pragma once
#include "cpu/x86_ops.h"

namespace d3d {
// Host-side equivalent of the non-destructive D3D_InternalRelease call. Its
// generic HLE wrapper restores the guest GPRs but leaves these two stack words
// and last_icall observable. Keep those effects without the dispatch round trip.
inline bool TryReleaseHeld(X86Context& ctx, uint8_t* base, uint32_t resource, uint32_t target) {
    constexpr uint32_t internal = 0x80000, internal_mask = 0x780000;
    const uint32_t e = ctx.esp;
    // The wrapper writes its return address before reading the header. Leave
    // unusual headers inside that frame to the original, ordered implementation.
    if (uint64_t(resource) < uint64_t(e) && uint64_t(e - 8) < uint64_t(resource) + 24)
        return false;
    const uint32_t common = X86_MEM32(resource);
    if ((common & internal_mask) == internal) {
        if (!(common & 0xFFFF))
            return false; // destruction
        if ((common & 0x70000) == 0x50000 && X86_MEM32(resource + 0x14))
            return false; // parent release
    }
    X86_MEM32(e - 4) = 0xFFFFFFF0;
    X86_MEM32(e - 8) = ctx.edi;
    ctx.last_icall = target;
    X86_MEM32(resource) = common - internal;
    return true;
}
} // namespace d3d
