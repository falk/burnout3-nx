// Hooks for default.xbe
#include <cstdlib>
#include <cstring>
#include <string>
#include <strings.h>

#include "kernel/kernel.h"
#include "xdk/hle.h"

using namespace kernel;

// D3D8LTCG's tile setup, inlined into the game where it allocates a tiled surface (sub_0003D500): it programs the
// NV2A's tile and Z compression registers through the device's register base and polls them. There are no registers
// here, and D3DDevice_SetTile, which follows it, is the SDK's no-op. Returns TRUE and pops its two arguments as it does.
X86_FUNC(sub_003568C5) { Return(ctx, 1, 2); }

// The race and crash training films (D:\ovid\TrD_<language>.xmv, TrC_<language>.xmv) play before a profile's first
// race of each kind and cannot be skipped then; they open the CRI logo film instead, a few seconds long (a film that
// fails to open crashes the game). B3_TRAINING_FILMS=1 keeps them.
namespace {
const char* ShortTrainingFilm(const char* name) {
    static const bool keep = [] {
        const char* e = std::getenv("B3_TRAINING_FILMS");
        return e && *e == '1';
    }();
    if (keep) return nullptr;
    const char* base = std::strrchr(name, '\\');
    base = base ? base + 1 : name;
    bool film = (!strncasecmp(base, "TrD_", 4) || !strncasecmp(base, "TrC_", 4)) && std::strlen(base) == 11;
    if (!film) return nullptr;
    // the same folder, as the name came
    static thread_local std::string other;
    other.assign(name, size_t(base - name));
    other += "cri_rw30.xmv";
    return other.c_str();
}
const bool g_short_training = (kernel::g_redirect_file = ShortTrainingFilm, true);
}  // namespace
