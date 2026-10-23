// Synthetic calls for the verify builds of the native rewrites (src/game/hot_*.cpp, B3_VERIFY_HOT only)
//
// As VerifyCase (hot_verify.h), for calls that also write outside the stack (a vertex buffer, statics): those
// ranges are saved, compared between the rewrite and the original, and put back afterwards with the stack around t.
#pragma once
#include "game/hot_verify.h"

#ifdef B3_VERIFY_HOT
#include <initializer_list>
#include <string>
#include <vector>

namespace hot {

// > 0 while a synthetic call runs: the verify wrappers of functions it calls keep their live counts clean
inline std::atomic<int>& SyntheticDepth() {
    static std::atomic<int> depth{0};
    return depth;
}

template <class Setup>
inline bool SyntheticCall(const std::string& name, const X86Context& live, uint8_t* base,
                          void (*native)(X86Context&, uint8_t*), void (*original)(X86Context&, uint8_t*), uint32_t t,
                          std::initializer_list<Region> extra, Setup setup) {
    std::vector<std::vector<uint8_t>> keep;
    Ranges r;
    r.Add(t - 0x400, 0x800);
    for (const Region& x : extra) r.Add(x.va, x.size);
    for (const Region& p : r.pieces) keep.emplace_back(X86_PTR(uint8_t, p.va), X86_PTR(uint8_t, p.va) + p.size);
    X86Context c = live;
    setup(c);
    std::atomic<uint64_t> calls{0}, bad{0};
    ++SyntheticDepth();
    const bool same = VerifyAll(name.c_str(), c, base, native, original, r, calls, bad, nullptr);
    --SyntheticDepth();
    std::fprintf(stderr, "verify %s: synthetic case %s\n", name.c_str(), same ? "same" : "DIFFERENT");
    for (size_t i = 0; i < r.pieces.size(); i++) std::memcpy(X86_PTR(uint8_t, r.pieces[i].va), keep[i].data(), r.pieces[i].size);
    return same;
}

}  // namespace hot
#endif
