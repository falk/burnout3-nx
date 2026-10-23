#pragma once
#include <cstdint>
#include <cstring>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace gpu::dk {
inline bool SameFourConstants(const uint8_t* a, const uint8_t* b) {
#ifdef __aarch64__
    const auto d0 = veorq_u8(vld1q_u8(a), vld1q_u8(b));
    const auto d1 = veorq_u8(vld1q_u8(a + 16), vld1q_u8(b + 16));
    const auto d2 = veorq_u8(vld1q_u8(a + 32), vld1q_u8(b + 32));
    const auto d3 = veorq_u8(vld1q_u8(a + 48), vld1q_u8(b + 48));
    return vmaxvq_u32(vreinterpretq_u32_u8(vorrq_u8(vorrq_u8(d0, d1), vorrq_u8(d2, d3)))) == 0;
#else
    return std::memcmp(a, b, 64) == 0;
#endif
}

// Find changed 16-byte registers, preserving bit patterns and limiting each
// upload to 1 KiB. Most registers are unchanged between adjacent draws.
// The callback uploads and copies the emitted run into the shadow.
template<class Emit>
void ChangedConstantRuns(const void* shadow, const void* now, uint32_t bytes, bool valid, Emit emit) {
    const auto* sh = static_cast<const uint8_t*>(shadow);
    const auto* nw = static_cast<const uint8_t*>(now);
    const uint32_t regs = bytes / 16;
    for (uint32_t r = 0; r < regs;) {
        if (valid && r + 4 <= regs && SameFourConstants(sh + r * 16, nw + r * 16)) {
            r += 4;
            continue;
        }
        if (valid && !std::memcmp(sh + r * 16, nw + r * 16, 16)) {
            ++r;
            continue;
        }
        uint32_t end = r + 1;
        while (end < regs && end - r < 64 && !(valid && !std::memcmp(sh + end * 16, nw + end * 16, 16))) ++end;
        emit(r * 16, (end - r) * 16);
        r = end;
    }
}
} // namespace gpu::dk
