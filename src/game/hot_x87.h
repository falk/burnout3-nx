// x87 arithmetic for the native rewrites (src/game/hot_*.cpp), on the doubles the generated code keeps for st(i)
//
// The operation runs in the instruction's operand order (asm, as hot_ordered.h does for SSE: a compiler may swap the
// operands of * and +, which changes which NaN comes out), and a NaN result is the one the x87 gives: with one NaN
// operand that one, quieted; with two the one with the larger significand (the destination's on a tie); an invalid
// operation (0 * inf, inf - inf) the real indefinite, a negative quiet NaN. The generated code takes plain double
// arithmetic instead, so its NaNs can differ from these (and from the console's); other results are the same.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace hot {

inline uint64_t DBits(double d) {
    uint64_t u;
    std::memcpy(&u, &d, 8);
    return u;
}
inline double DFromBits(uint64_t u) {
    double d;
    std::memcpy(&d, &u, 8);
    return d;
}
inline double X87NaN(double d, double s) {
    constexpr uint64_t kSignificand = 0x000FFFFFFFFFFFFFull, kQuiet = 0x0008000000000000ull;
    const bool dn = std::isnan(d), sn = std::isnan(s);
    if (!dn && !sn) return DFromBits(0xFFF8000000000000ull);
    uint64_t pick = dn ? DBits(d) : DBits(s);
    if (dn && sn && ((DBits(s) | kQuiet) & kSignificand) > ((DBits(d) | kQuiet) & kSignificand)) pick = DBits(s);
    return DFromBits(pick | kQuiet);
}

#if defined(__aarch64__)
#define HOT_X87_OP(name, insn)                                                     \
    inline double name(double d, double s) {                                       \
        double r;                                                                  \
        __asm__(insn " %d0, %d1, %d2" : "=w"(r) : "w"(d), "w"(s));                 \
        return std::isnan(r) ? X87NaN(d, s) : r;                                   \
    }
HOT_X87_OP(X87Mul, "fmul")
HOT_X87_OP(X87Add, "fadd")
HOT_X87_OP(X87Sub, "fsub")
#else
#define HOT_X87_OP(name, insn)                                                     \
    inline double name(double d, double s) {                                       \
        double r = d;                                                              \
        __asm__(insn " %1, %0" : "+x"(r) : "x"(s));                                \
        return std::isnan(r) ? X87NaN(d, s) : r;                                   \
    }
HOT_X87_OP(X87Mul, "mulsd")
HOT_X87_OP(X87Add, "addsd")
HOT_X87_OP(X87Sub, "subsd")
#endif
#undef HOT_X87_OP

}  // namespace hot
