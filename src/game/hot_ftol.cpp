// The compiler's x87-to-integer helper (sub_00244DF0), called from the physics, the effects and the HUD. The same
// code as Black's sub_000A34D4 (Black src/game/hot_ftol.cpp): collapse its push/pop sequence while retaining the
// popped slots, precision-control behavior and bitwise carry correction.
#include "game/hot_paths.h"
#include "game/hot_verify.h"

X86_EXTERN_FUNC(__imp__sub_00244DF0);

namespace {

void ConvertInteger(X86Context& ctx, uint8_t* base) {
    const uint32_t e = ctx.esp, s = ((e - 4) & ~15u) - 32;
    const uint32_t top = ctx.top;
    const double value = ctx.st[top];
    X86_MEM32(e - 4) = ctx.ebp;
    const float initial = float(value);
    X86_MEMF32(s + 24) = initial;
    const uint64_t rounded = uint64_t(x87_to_i64(ctx, value));
    X86_MEM64(s + 16) = rounded;
    const double integer = double(int64_t(rounded));
    ctx.st[(top - 1) & 7] = integer;
    uint64_t result = rounded;
    if (uint32_t(rounded) || (uint32_t(rounded >> 32) & 0x7fffffff)) {
        const double residual = x87_prec(ctx, value - integer);
        ctx.st[top] = residual;
        const float fraction = float(residual);
        X86_MEMF32(s) = fraction;
        uint32_t bits = hot::Bits(fraction);
        const bool negative = hot::Bits(initial) >> 31;
        if (negative) bits ^= 0x80000000u;
        const uint64_t sum = uint64_t(bits) + 0x7fffffffu;
        ctx.ecx = uint32_t(sum);
        if (negative) result += sum >> 32;
        else result -= sum >> 32;
    }
    ctx.eax = uint32_t(result);
    ctx.edx = uint32_t(result >> 32);
    ctx.top = (top + 1) & 7;
    ctx.esp = e + 4;
}
}  // namespace

X86_FUNC(sub_00244DF0) {
#ifndef B3_VERIFY_HOT
    ConvertInteger(ctx, base);
#else
    hot::Ranges ranges;
    ranges.Add(ctx.esp - 0x40, 0x40);
    static std::atomic<uint64_t> calls{0}, bad{0};
    hot::VerifyAll("sub_00244DF0", ctx, base, ConvertInteger, __imp__sub_00244DF0, ranges, calls, bad, nullptr);
#endif
}
