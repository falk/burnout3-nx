// Packed float operations with their operands kept in order, and shufps as a permute (src/game/hot_*.cpp)
//
// When both operands of addps, subps or mulps are NaN, the result is the first operand's NaN (x86: the destination;
// A64 FADD, FSUB, FMUL: Operand1). Compilers treat + and * as commutative and may swap the operands, which changes
// the NaN a lane ends with (seen on garbage w columns of matrices). These keep the instruction's operand order:
// V* packed (addps, subps, mulps, divps; VMulS by a splat second operand), S* scalar (addss, mulss), AddPair lane 0 + lane 1.
// They are needed only where both operands can be NaN; an operand that is a constant of .rdata, or a copy of the
// other operand, cannot change the result. The asm is not volatile: unused results are still removed and common
// ones shared; what it gives up is constant folding and the A64 multiply by element (fmul v, v, v.s[i]).
// What order cannot fix: A64 makes a new NaN (0 * inf, inf - inf) as 0x7FC00000 where x86 makes 0xFFC00000, and
// with a quiet NaN first and a signaling NaN second A64 returns the second (quieted), x86 the first.
#pragma once
#include "game/hot_paths.h"

namespace hot {

// shufps a, b, imm as one permute the compiler can map to ext, zip, uzp, rev, dup or tbl
template <int I>
inline f4 Sh(f4 a, f4 b) {
#if defined(__clang__)
    return __builtin_shufflevector(a, b, I & 3, (I >> 2) & 3, 4 + ((I >> 4) & 3), 4 + ((I >> 6) & 3));
#else
    typedef int32_t i4 __attribute__((vector_size(16)));
    return __builtin_shuffle(a, b, i4{I & 3, (I >> 2) & 3, 4 + ((I >> 4) & 3), 4 + ((I >> 6) & 3)});
#endif
}

#if defined(__x86_64__) || defined(__i386__)
inline f4 VAdd(f4 a, f4 b) {
    __asm__("addps %1, %0" : "+x"(a) : "x"(b));
    return a;
}
inline f4 VSub(f4 a, f4 b) {
    __asm__("subps %1, %0" : "+x"(a) : "x"(b));
    return a;
}
inline f4 VMul(f4 a, f4 b) {
    __asm__("mulps %1, %0" : "+x"(a) : "x"(b));
    return a;
}
inline f4 VDiv(f4 a, f4 b) {
    __asm__("divps %1, %0" : "+x"(a) : "x"(b));
    return a;
}
inline float SAdd(float a, float b) {
    __asm__("addss %1, %0" : "+x"(a) : "x"(b));
    return a;
}
inline float SMul(float a, float b) {
    __asm__("mulss %1, %0" : "+x"(a) : "x"(b));
    return a;
}
inline float AddPair(f4 v) { return SAdd(v[0], v[1]); }
inline f4 VMulS(f4 a, float b) { return VMul(a, Splat(b)); }
#elif defined(__aarch64__)
inline f4 VAdd(f4 a, f4 b) {
    f4 r;
    __asm__("fadd %0.4s, %1.4s, %2.4s" : "=w"(r) : "w"(a), "w"(b));
    return r;
}
inline f4 VSub(f4 a, f4 b) {
    f4 r;
    __asm__("fsub %0.4s, %1.4s, %2.4s" : "=w"(r) : "w"(a), "w"(b));
    return r;
}
inline f4 VMul(f4 a, f4 b) {
    f4 r;
    __asm__("fmul %0.4s, %1.4s, %2.4s" : "=w"(r) : "w"(a), "w"(b));
    return r;
}
inline f4 VDiv(f4 a, f4 b) {
    f4 r;
    __asm__("fdiv %0.4s, %1.4s, %2.4s" : "=w"(r) : "w"(a), "w"(b));
    return r;
}
inline float SAdd(float a, float b) {
    float r;
    __asm__("fadd %s0, %s1, %s2" : "=w"(r) : "w"(a), "w"(b));
    return r;
}
inline float SMul(float a, float b) {
    float r;
    __asm__("fmul %s0, %s1, %s2" : "=w"(r) : "w"(a), "w"(b));
    return r;
}
// a * {b, b, b, b} as FMUL (by element): Operand1 is a, as in mulps a, splat(b), without the dup
inline f4 VMulS(f4 a, float b) {
    f4 r;
    __asm__("fmul %0.4s, %1.4s, %2.s[0]" : "=w"(r) : "w"(a), "w"(b));
    return r;
}
// FADDP (scalar) is one IEEE addition with element 0 as Operand1: the same result as v[0] + v[1] in that order,
// without moving lane 1 out first
inline float AddPair(f4 v) {
    float r;
    __asm__("faddp %s0, %1.2s" : "=w"(r) : "w"(v));
    return r;
}
#else
inline f4 VAdd(f4 a, f4 b) { return a + b; }
inline f4 VSub(f4 a, f4 b) { return a - b; }
inline f4 VMul(f4 a, f4 b) { return a * b; }
inline f4 VDiv(f4 a, f4 b) { return a / b; }
inline float SAdd(float a, float b) { return a + b; }
inline float SMul(float a, float b) { return a * b; }
inline float AddPair(f4 v) { return v[0] + v[1]; }
inline f4 VMulS(f4 a, float b) { return a * Splat(b); }
#endif

}  // namespace hot
