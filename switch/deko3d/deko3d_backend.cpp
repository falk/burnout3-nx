// The title's Direct3D 8 calls drawn with deko3d on the Switch's Maxwell GPU. The same console path as the SDK's
// D3D12 backend (runtime/gpu/d3d12.cpp): guest surfaces become render targets at the console size, guest textures are
// decoded on the CPU and uploaded, the shaders are the SDK's translation as GLSL (nv2a_glsl.cpp) compiled by UAM.
//
// Frames: kFrames frames in flight. Each frame slot owns its command memory, its upload ring and the resources freed
// while it recorded; all of it is reused once the slot's fence (signalled after its present) has passed.
// Hazards: images carry the epoch of their last render, copy write and read; a write after a read or a read after a
// write in the same epoch takes a full barrier with image and L2 invalidation, which starts a new epoch.
// Descriptors: one image descriptor per image for its life, samplers cached by state; both tables written by the CPU,
// slots never reused before every frame that used them is done, new ones made visible by a descriptor invalidate.
#include <switch.h>
#include <deko3d.h>
#include <arm_acle.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "app/app.h"
#include "core/log.h"
#include "d3d/gpu_frame.h"
#include "d3d/occlusion.h"
#include "gpu/backend.h"
#include "gpu/shaders.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "harness.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"
#include "nv2a_glsl.h"
#include "shader_cache.h"
#include "draw_cache.h"
#include "constant_runs.h"

namespace gpu {

// presents so far, for the pad script's frame-relative presses (switch_input.cpp)
std::atomic<uint64_t> g_presented{0};

namespace {

constexpr uint32_t kFrames = 2;
constexpr uint32_t kRingChunk = 8u << 20;   // upload ring chunk, per frame slot as needed
constexpr uint32_t kCmdChunk = 1u << 20;    // command memory chunk
constexpr uint32_t kImageDescs = 8192;
constexpr uint32_t kSamplerDescs = 1024;
constexpr uint32_t kReportSlots = 4096;     // counter reports in flight: two 16 byte reports per counted draw
constexpr uint64_t kStaleFrames = 8;        // frames a target may go unused and still be read back
constexpr uint32_t kSpanSlots = 16;        // GPU timestamps per frame around each submitted list (start, end pairs)
constexpr uint32_t kSubmitDraws = 200;      // a pass ending after this many unsubmitted draws is submitted early
constexpr uint32_t kMarks = 192;            // GPU timestamps per frame at render target changes (per pass GPU time)

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

float Bits(uint32_t v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

// n / 255 for each byte, the same correctly rounded quotients a division gives: SetConstants converts some twenty
// colours per draw, and the A57 divides slowly
constexpr std::array<float, 256> kUnorm8 = [] {
    std::array<float, 256> t{};
    for (int i = 0; i < 256; i++) t[size_t(i)] = float(i) / 255.0f;
    return t;
}();

void Color(uint32_t argb, float* o) {
    o[0] = kUnorm8[(argb >> 16) & 255];
    o[1] = kUnorm8[(argb >> 8) & 255];
    o[2] = kUnorm8[argb & 255];
    o[3] = kUnorm8[argb >> 24];
}

uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

// bytes per row of a 32 bit surface in guest memory, rounded to 64
uint32_t GuestPitch(uint32_t width) { return (width * 4 + 63) & ~63u; }

// --- console enums (GL values) to deko3d ---

DkBlendFactor Blend(uint32_t v, bool alpha) {
    switch (v) {
        case 0: return DkBlendFactor_Zero;
        case 1: return DkBlendFactor_One;
        case 0x300: return alpha ? DkBlendFactor_SrcAlpha : DkBlendFactor_SrcColor;
        case 0x301: return alpha ? DkBlendFactor_InvSrcAlpha : DkBlendFactor_InvSrcColor;
        case 0x302: return DkBlendFactor_SrcAlpha;
        case 0x303: return DkBlendFactor_InvSrcAlpha;
        case 0x304: return DkBlendFactor_DstAlpha;
        case 0x305: return DkBlendFactor_InvDstAlpha;
        case 0x306: return alpha ? DkBlendFactor_DstAlpha : DkBlendFactor_DstColor;
        case 0x307: return alpha ? DkBlendFactor_InvDstAlpha : DkBlendFactor_InvDstColor;
        case 0x308: return DkBlendFactor_SrcAlphaSaturate;
        case 0x8001: return DkBlendFactor_ConstColor;
        case 0x8002: return DkBlendFactor_InvConstColor;
        case 0x8003: return DkBlendFactor_ConstAlpha;
        case 0x8004: return DkBlendFactor_InvConstAlpha;
        default: return DkBlendFactor_One;
    }
}

DkBlendOp BlendOp(uint32_t v) {
    switch (v) {
        case 0x800A: return DkBlendOp_Sub;
        case 0x800B: return DkBlendOp_RevSub;
        case 0x8007: return DkBlendOp_Min;
        case 0x8008: return DkBlendOp_Max;
        default: return DkBlendOp_Add;
    }
}

// D3DCMP_NEVER 0x200 to D3DCMP_ALWAYS 0x207, deko3d's order from 1
DkCompareOp Cmp(uint32_t v) { return DkCompareOp(v >= 0x200 && v <= 0x207 ? v - 0x200 + 1 : 8); }

DkStencilOp StencilOp(uint32_t v) {
    switch (v) {
        case 0: return DkStencilOp_Zero;
        case 0x1E01: return DkStencilOp_Replace;
        case 0x1E02: return DkStencilOp_Incr;
        case 0x1E03: return DkStencilOp_Decr;
        case 0x150A: return DkStencilOp_Invert;
        case 0x8507: return DkStencilOp_IncrWrap;
        case 0x8508: return DkStencilOp_DecrWrap;
        default: return DkStencilOp_Keep;
    }
}

// D3DTADDRESS: 1 wrap, 2 mirror, 3 clamp, 4 border, 5 clamp to edge
DkWrapMode Address(uint32_t v) {
    switch (v) {
        case 2: return DkWrapMode_MirroredRepeat;
        case 3: case 5: return DkWrapMode_ClampToEdge;
        case 4: return DkWrapMode_ClampToBorder;
        default: return DkWrapMode_Repeat;
    }
}

DkImageFormat HostImageFormat(Host h) {
    switch (h) {
        case Host::BC1: return DkImageFormat_RGBA_BC1;
        case Host::BC2: return DkImageFormat_RGBA_BC2;
        case Host::BC3: return DkImageFormat_RGBA_BC3;
        case Host::R32F: return DkImageFormat_R32_Float;
        case Host::R16Unorm: return DkImageFormat_R16_Unorm;
        default: return DkImageFormat_BGRA8_Unorm;
    }
}

// a vertex element's DXGI_FORMAT (as gpu::Assemble names them) as a deko3d attribute
bool VertexFormat(uint8_t dxgi, DkVtxAttribState& a) {
    a.isBgra = 0;
    switch (dxgi) {
        case 2: a.size = DkVtxAttribSize_4x32; a.type = DkVtxAttribType_Float; return true;   // R32G32B32A32_FLOAT
        case 6: a.size = DkVtxAttribSize_3x32; a.type = DkVtxAttribType_Float; return true;   // R32G32B32_FLOAT
        case 16: a.size = DkVtxAttribSize_2x32; a.type = DkVtxAttribType_Float; return true;  // R32G32_FLOAT
        case 41: a.size = DkVtxAttribSize_1x32; a.type = DkVtxAttribType_Float; return true;  // R32_FLOAT
        case 42: a.size = DkVtxAttribSize_1x32; a.type = DkVtxAttribType_Uint; return true;   // R32_UINT
        case 87: a.size = DkVtxAttribSize_4x8; a.type = DkVtxAttribType_Unorm; a.isBgra = 1; return true;  // B8G8R8A8
        case 28: a.size = DkVtxAttribSize_4x8; a.type = DkVtxAttribType_Unorm; return true;   // R8G8B8A8_UNORM
        case 49: a.size = DkVtxAttribSize_2x8; a.type = DkVtxAttribType_Unorm; return true;   // R8G8_UNORM
        case 61: a.size = DkVtxAttribSize_1x8; a.type = DkVtxAttribType_Unorm; return true;   // R8_UNORM
        case 13: a.size = DkVtxAttribSize_4x16; a.type = DkVtxAttribType_Snorm; return true;  // R16G16B16A16_SNORM
        case 37: a.size = DkVtxAttribSize_2x16; a.type = DkVtxAttribType_Snorm; return true;  // R16G16_SNORM
        case 58: a.size = DkVtxAttribSize_1x16; a.type = DkVtxAttribType_Snorm; return true;  // R16_SNORM
        case 14: a.size = DkVtxAttribSize_4x16; a.type = DkVtxAttribType_Sint; return true;   // R16G16B16A16_SINT
        case 38: a.size = DkVtxAttribSize_2x16; a.type = DkVtxAttribType_Sint; return true;   // R16G16_SINT
        case 59: a.size = DkVtxAttribSize_1x16; a.type = DkVtxAttribType_Sint; return true;   // R16_SINT
        default: return false;
    }
}

// --- memory ---

struct Mem {
    DkMemBlock block = nullptr;
    uint32_t offset = 0, size = 0, chunk = 0;
    uint8_t* cpu = nullptr;
    DkGpuAddr gpu = 0;
    explicit operator bool() const { return block != nullptr; }
};

// free spans over memblocks grown on demand (after deko9's Heap)
class Heap {
public:
    void Init(DkDevice dev, uint32_t flags, uint32_t chunk, uint32_t tail = 0) {
        dev_ = dev, flags_ = flags, chunk_ = chunk, tail_ = tail;
    }
    bool Alloc(uint32_t size, uint32_t align, Mem& out) {
        if (!size) return false;
        align = std::max<uint32_t>(align, 256);
        size = AlignUp(size, 256);
        for (int attempt = 0; attempt < 2; attempt++) {
            for (uint32_t c = 0; c < chunks_.size(); c++) {
                Chunk& ch = chunks_[c];
                for (auto it = ch.free.begin(); it != ch.free.end(); ++it) {
                    uint32_t span = it->first, span_size = it->second, start = AlignUp(span, align);
                    if (uint64_t(start - span) + size > span_size) continue;
                    uint32_t end = span + span_size;
                    ch.free.erase(it);
                    if (start > span) ch.free[span] = start - span;
                    if (start + size < end) ch.free[start + size] = end - (start + size);
                    out.block = ch.block;
                    out.offset = start;
                    out.size = size;
                    out.chunk = c;
                    out.gpu = dkMemBlockGetGpuAddr(ch.block) + start;
                    void* cpu = (flags_ & DkMemBlockFlags_CpuAccessMask) ? dkMemBlockGetCpuAddr(ch.block) : nullptr;
                    out.cpu = cpu ? static_cast<uint8_t*>(cpu) + start : nullptr;
                    return true;
                }
            }
            if (attempt) break;
            uint32_t bytes = AlignUp(std::max(chunk_, size + align + tail_), DK_MEMBLOCK_ALIGNMENT);
            DkMemBlockMaker m;
            dkMemBlockMakerDefaults(&m, dev_, bytes);
            m.flags = flags_;
            DkMemBlock block = dkMemBlockCreate(&m);
            if (!block) {
                LOGE("deko3d: no memblock of %u KiB (flags %X)", bytes >> 10, flags_);
                return false;
            }
            Chunk ch{block, {}};
            ch.free[0] = bytes - tail_;
            chunks_.push_back(std::move(ch));
        }
        return false;
    }
    void Free(const Mem& m) {
        if (!m.block) return;
        auto& free = chunks_[m.chunk].free;
        uint32_t start = m.offset, size = m.size;
        auto next = free.lower_bound(start);
        if (next != free.end() && start + size == next->first) {
            size += next->second;
            next = free.erase(next);
        }
        if (next != free.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == start) {
                start = prev->first;
                size += prev->second;
                free.erase(prev);
            }
        }
        free[start] = size;
    }

private:
    struct Chunk {
        DkMemBlock block;
        std::map<uint32_t, uint32_t> free;  // offset -> size
    };
    DkDevice dev_ = nullptr;
    uint32_t flags_ = 0, chunk_ = 0, tail_ = 0;
    std::vector<Chunk> chunks_;
};

// --- images ---

// an image with its memory, its descriptor slot and its hazard epochs
struct Img {
    DkImage image;
    DkImageLayout layout;
    Mem mem;
    uint32_t desc = ~0u;
    uint32_t desc_opaque = ~0u;  // the same view with alpha read as 1, for a target sampled as an alpha-less format
    DkImageFormat fmt = DkImageFormat_None;
    DkImageType type = DkImageType_2D;
    uint32_t w = 0, h = 0, d = 1, levels = 1;
    bool depth = false;
    uint64_t render = 0, copy_write = 0, read = 0;
    bool live() const { return mem.block != nullptr; }
};

enum class Access : uint8_t { Sample, Render, CopyRead, CopyWrite };

struct Target {
    Img img;
    bool depth = false;
    uint32_t gw = 0, gh = 0;
    uint32_t data = 0, bytes = 0;
    uint64_t epoch = 0, sig = 0;  // last lock seen, guest memory as last matched
    bool drawn = false;           // the GPU wrote it since the last read back
    uint32_t aw = 0, ah = 0;      // host pixels its picture covers (dynamic resolution), set when drawn
    std::vector<uint8_t> read;    // what Sync left in guest memory
    std::unique_ptr<Img> fb;      // copy read by draws that sample the target they render to
    uint64_t version = 1;         // counts the GPU's writes (binds, clears, copies, uploads)
    std::unique_ptr<Img> small;   // its picture at the console's size, for passes shrinking an upscaled target
    uint64_t small_version = 0;   // the version small holds
};

struct Tex {
    Img img;
    TextureDesc d;
    uint32_t levels = 1;
    uint64_t sig = 0, frame = 0, palette = 0;
};

// MemSig hashes a surface up to kSigWholeMax bytes whole, a bigger one as evenly spaced samples
constexpr uint32_t kSigWholeMax = 65536;
constexpr uint32_t kSigSamples = 64;
constexpr uint32_t kSigSampleBytes = 64;

// A change check of n bytes: four CRC32C streams over interleaved 8 byte words, the A57 retiring one crc32cx a cycle
// while gpu::Hash's chain of 128 bit multiplies takes several per 16 bytes (it ran over every texture every frame)
uint64_t Sig(const uint8_t* p, size_t n, uint64_t seed) {
    uint32_t a = uint32_t(seed), b = uint32_t(seed >> 32), c = ~a, d = ~b;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        uint64_t w[4];
        std::memcpy(w, p + i, 32);
        a = __crc32cd(a, w[0]);
        b = __crc32cd(b, w[1]);
        c = __crc32cd(c, w[2]);
        d = __crc32cd(d, w[3]);
    }
    for (; i < n; i++) a = __crc32cb(a, p[i]);
    a = __crc32cw(a, uint32_t(n));
    return (uint64_t(a ^ c) << 32 | (b ^ d)) ^ (uint64_t(c) << 7) ^ (uint64_t(d) << 39);
}

uint64_t MemSig(uint32_t data, uint32_t total) {
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(data));
    uint64_t sig = 0x9E3779B97F4A7C15ull ^ total;
    if (total <= kSigWholeMax) return Sig(p, total, sig);
    for (uint32_t i = 0; i < kSigSamples; i++) {
        const uint8_t* sample = p + uint64_t(total - kSigSampleBytes) * i / (kSigSamples - 1);
        sig = Sig(sample, kSigSampleBytes, sig);
    }
    return sig;
}

// MemSig samples a big surface and misses small writes: after a Sync every byte is compared
bool Changed(const Target& t) {
    if (t.drawn || t.read.empty()) return false;
    uint32_t row = t.gw * 4, pitch = GuestPitch(t.gw);
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
    for (uint32_t y = 0; y < t.gh; y++)
        if (std::memcmp(p + size_t(y) * pitch, &t.read[size_t(y) * row], row)) return true;
    return false;
}

// a lock alone proves nothing, reads lock too: only changed bytes mean the CPU wrote it
bool LatchCpuWrite(Target& t) {
    uint64_t e = DirtyEpoch(t.data);
    if (e <= t.epoch) return false;
    t.epoch = e;
    uint64_t sig = MemSig(t.data, t.bytes);
    if (sig == t.sig && !Changed(t)) return false;
    t.sig = sig;
    return true;
}

uint64_t TargetMapKey(uint32_t data, bool depth) { return uint64_t(data) << 1 | (depth ? 1 : 0); }

constexpr uint32_t kFmtD16 = 0x2C;     // D3DFMT_D16
constexpr uint32_t kFmtLinD16 = 0x30;  // D3DFMT_LIN_D16
uint8_t ShadowDepthKind(uint32_t format) { return format == kFmtD16 || format == kFmtLinD16 ? 2 : 1; }

TexKind TextureKind(const TextureDesc& d) {
    if (d.cube) return kTexCube;
    if (d.depth > 1) return kTex3D;
    return kTex2D;
}

// D3DTRANSFORMSTATETYPE indices into s.transforms
constexpr size_t kTsView = 2;
constexpr size_t kTsProjection = 3;
constexpr size_t kTsTexture0 = 16;
constexpr size_t kTsWorld = 256;

void MatMul(const float* x, const float* y, float* o) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            o[r * 4 + c] = x[r * 4] * y[c] + x[r * 4 + 1] * y[4 + c] + x[r * 4 + 2] * y[8 + c] + x[r * 4 + 3] * y[12 + c];
}

// GLSL `v * M` over a column major block reads the transpose of the row major matrix
void Transpose(const float* m, float* o) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) o[c * 4 + r] = m[r * 4 + c];
}

void InverseTranspose3x3(const float* wv, float* n) {
    float det = wv[0] * (wv[5] * wv[10] - wv[6] * wv[9]) - wv[1] * (wv[4] * wv[10] - wv[6] * wv[8]) +
                wv[2] * (wv[4] * wv[9] - wv[5] * wv[8]);
    float id = det != 0 ? 1.0f / det : 0.0f;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            int r1 = (r + 1) % 3, r2 = (r + 2) % 3, c1 = (c + 1) % 3, c2 = (c + 2) % 3;
            n[r * 4 + c] = (wv[r1 * 4 + c1] * wv[r2 * 4 + c2] - wv[r1 * 4 + c2] * wv[r2 * 4 + c1]) * id;
        }
}

// D3DLIGHT8 as floats: Type, Diffuse[4], Specular[4], Ambient[4], Position[3], Direction[3], Range, Falloff,
// Attenuation0 to 2, Theta, Phi
void LightToView(const std::array<uint32_t, 26>& l, const float* vm, float (&o)[7][4]) {
    float f[26];
    std::memcpy(f, l.data(), sizeof(f));
    for (int j = 0; j < 4; j++) {
        o[0][j] = f[1 + j];
        o[1][j] = f[5 + j];
        o[2][j] = f[9 + j];
    }
    const float* pos = f + 13;
    const float* dir = f + 16;
    for (int j = 0; j < 3; j++) {
        o[3][j] = pos[0] * vm[j] + pos[1] * vm[4 + j] + pos[2] * vm[8 + j] + vm[12 + j];
        o[4][j] = dir[0] * vm[j] + dir[1] * vm[4 + j] + dir[2] * vm[8 + j];
    }
    o[3][3] = f[19];
    o[4][3] = f[20];
    o[5][0] = f[21];
    o[5][1] = f[22];
    o[5][2] = f[23];
    o[6][0] = std::cos(f[24] * 0.5f);
    o[6][1] = std::cos(f[25] * 0.5f);
}

uint32_t Simple(const State& s, uint32_t method) { return s.simple[method >> 2]; }

// console texture formats (XDK D3DFORMAT) that store an alpha channel; a read of any other gives alpha 1
bool GuestFormatHasAlpha(uint32_t f) {
    switch (f) {
        case 0x00: case 0x03: case 0x05: case 0x07: case 0x11: case 0x13: case 0x16: case 0x17: case 0x1C: case 0x1E:
        case 0x24: case 0x25: case 0x27: case 0x28: case 0x29:
            return false;
        default: return true;
    }
}

// Final glow composite signature from TGP482/BLACK-XBERecomp#39.
bool IsGlowComposite(const State& s, const PixelKey& pk, const TextureDesc& glow, uint32_t prim, uint32_t count,
                     uint32_t rt_w, uint32_t rt_h, uint32_t scene_w, uint32_t scene_h, bool texture_rt) {
    const uint32_t op = Simple(s, kMBlendOp);
    return scene_w && scene_h && prim == 5 && count == 3 && rt_w == scene_w && rt_h == scene_h &&
        !s.rs[kRsZEnable] && pk.programmable && (pk.count & 0xFF) == 1 && pk.tex_modes == 1 && !pk.alphakill &&
        pk.rgb_in[0] == 0xC8C10000 && pk.rgb_out[0] == 0xC0 && pk.final_abcd == 0x00000C00 &&
        pk.final_efg == 0x00001180 && Simple(s, kMBlendEnable) && Simple(s, kMSrcBlend) == 1 &&
        Simple(s, kMDestBlend) == 0x302 && op != 0x800A && op != 0x800B && op != 0x8007 && op != 0x8008 && texture_rt &&
        (Simple(s, kMPsFinalConstant0) >> 24) == 255 && !s.textures[1] && !s.textures[2] && !s.textures[3] &&
        glow.format == 0x12 && uint64_t(glow.width) * 4 == scene_w && uint64_t(glow.height) * 4 == scene_h;
}

void AlignGlow(float* texsize, float fx, float fy, uint32_t scene_w, uint32_t scene_h) {
    // Upstream calibration is in scene pixels. Account for the active part of
    // a cached image when dynamic resolution changes its drawn dimensions.
    texsize[2] += 7.38f / float(scene_w) * fx;
    texsize[3] += 10.77f / float(scene_h) * fy;
}

// the draw's shaders: what they are made from and their cache keys
struct DrawShaders {
    bool prog = false, quad = false, fixed_transform = false;
    FixedKey fk;
    uint64_t vkey = 0, pkey = 0;
    const DkShader* vs = nullptr;
    const DkShader* ps = nullptr;
};

constexpr uint32_t kQuadVertices = 6;
constexpr uint64_t kDeclarationSalt = 0x10000;
constexpr uint64_t kPretransformedSalt = 0x5A5A;
constexpr uint64_t kQuadSalt = 0xC3C30000;

// A small draw whose every float2 input lies in 0..1 is sampled clamped (see the D3D12 backend's UnitTexcoords)
bool UnitTexcoords(const Assembled& a) {
    constexpr float kLo = -1e-4f, kHi = 1.0001f;
    if (!a.vertices.empty() || !a.vertex_count || a.vertex_count > 4096) return false;
    bool any = false;
    for (int r = 0; r < 16; r++) {
        const VertexElement& e = a.elements[size_t(r)];
        if (!(a.regs >> r & 1) || e.format != 16) continue;  // DXGI_FORMAT_R32G32_FLOAT
        const VertexBuffer& vb = a.buffers[e.buffer];
        if (!vb.src || !vb.stride || uint64_t(a.vertex_count) * vb.stride > vb.bytes) return false;
        for (uint32_t v = 0; v < a.vertex_count; v++) {
            float uv[2];
            std::memcpy(uv, vb.src + size_t(v) * vb.stride + e.offset, sizeof(uv));
            if (!(uv[0] >= kLo && uv[0] <= kHi && uv[1] >= kLo && uv[1] <= kHi)) return false;
        }
        any = true;
    }
    return any;
}

// the present pass: the front buffer over the bound viewport, each channel through the gamma lookup
const char* kPresentVs = R"(#version 460
layout(location = 0) out vec2 uv;
layout(std140, binding = 0) uniform P { vec4 part; };
void main() {
    vec2 t = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = t * part.xy;
    gl_Position = vec4(t * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
}
)";
const char* kDepthCopyPs = R"(#version 460
layout(location = 0) in vec2 uv;
layout(binding = 0) uniform sampler2D img;
void main() { gl_FragDepth = texture(img, uv).r; }
)";
const char* kPresentPs = R"(#version 460
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
layout(binding = 0) uniform sampler2D img;
layout(binding = 1) uniform sampler2D lut;
void main() {
    ivec3 i = ivec3(clamp(texture(img, uv).rgb, 0.0, 1.0) * 255.0 + 0.5);
    o = vec4(texelFetch(lut, ivec2(i.r, 0), 0).r, texelFetch(lut, ivec2(i.g, 0), 0).g, texelFetch(lut, ivec2(i.b, 0), 0).b, 1.0);
}
)";

class DekoBackend;
DekoBackend* g_self = nullptr;

class DekoBackend final : public Backend {
public:
    bool Init(uint32_t width, uint32_t height) override;
    void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
               uint32_t stencil) override;
    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override;
    void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects, uint32_t dst_points) override;
    void Present(const Surface& front) override;
    void Sync(uint32_t data, uint32_t size) override;
    void SyncLate(uint32_t data) override;
    bool DeliverLate(uint32_t data) override;
    void RecordLate(uint32_t data) override;
    void Flush() override { Submit(true); }
    void SetGamma(const uint8_t* ramp) override {
        std::memcpy(ramp_, ramp, sizeof(ramp_));
        ramp_dirty_ = true;
    }
    void ClearReport() override { counted_.clear(); }
    void GetReport(uint32_t report) override {
        LOG_FIRST(4, "deko3d: visibility report %08X over %zu counted draws", report, counted_.size());
        reports_.push_back({report, counted_, frame_});
    }

    // until the GPU has drawn present number n, counted from 0
    void WaitFrame(uint64_t n);
    // present n drawn, not waiting: false while it is queued or not presented yet
    void ShutdownForExit();
    bool FrameDone(uint64_t n);
    bool ZeroCopy() const { return guest_mem_ != nullptr; }
    // for the title's busy checks (render_thread.cpp): what the GPU may still read in place
    static constexpr uint32_t kInPlaceSlots = 4096;
    std::atomic<uint64_t> in_place_va_[kInPlaceSlots] = {};  // by hash of the buffer's va: present + 1 last read
    std::atomic<uint64_t> in_place_any_[8] = {};             // by present % 8: present + 1 when it read anything in place
    static uint32_t InPlaceSlot(uint32_t va) { return (va * 0x9E3779B1u) >> 20; }
    uint64_t InPlaceFrame(uint32_t va) const { return in_place_va_[InPlaceSlot(va)].load(std::memory_order_acquire); }
    bool FrameReadInPlace(uint64_t n) const { return in_place_any_[n % 8].load(std::memory_order_acquire) == n + 1; }

private:
    struct Slot {
        DkFence fence{};
        bool pending = false;
        std::vector<Mem> cmd;  // command memory chunks; the first is added at frame start
        uint32_t cmd_next = 0;
        std::vector<Mem> ring;
        uint32_t ring_chunk = 0, ring_used = 0;
        std::vector<Mem> garbage;          // memory freed while the slot recorded
        std::vector<Mem> read_garbage;     // read back memory freed while the slot recorded
        std::vector<uint32_t> free_descs;  // image descriptors freed while the slot recorded
        std::vector<uint64_t> marks;       // the pass each GPU timestamp of the frame starts
        uint32_t spans = 0;                // span timestamps written this frame
        std::vector<uint32_t> mark_draws;  // draws (and blended draws << 16) recorded in each
    };
    struct Span {
        uint8_t* cpu;
        DkGpuAddr gpu;
    };

    static void AddCmdMemory(void* user, DkCmdBuf, size_t min);
    Slot& Cur() { return slots_[frame_ % kFrames]; }
    Span Upload(uint32_t size, uint32_t align = 256);
    void BeginFrame();
    void Submit(bool wait);
    uint32_t unsubmitted_draws_ = 0;  // draws recorded since the last submit
    void Barrier();
    void Hazard(Img& img, Access a);
    void HazardsDone(Img& img, Access a);

    // images
    bool CreateImage(Img& img, DkImageType type, DkImageFormat fmt, uint32_t w, uint32_t h, uint32_t d,
                     uint32_t levels, uint32_t flags);
    void FreeImage(Img& img);
    uint32_t AllocDesc(const Img& img, bool opaque = false);
    uint32_t SamplerSlot(const DkSampler& sm);
    void ImageView(const Img& img, DkImageView& v);
    void UploadPixels(Img& img, uint32_t level, uint32_t face, const uint8_t* px, uint32_t pitch, uint32_t w,
                      uint32_t h, uint32_t depth, uint32_t row_bytes, uint32_t rows);

    // targets
    Target* FindTarget(const Surface& s, bool depth);
    Target* EnsureTarget(const Surface& s, bool depth);
    void ReleaseTarget(std::unordered_map<uint64_t, Target>::iterator it);
    uint64_t SizedTargetKey(uint32_t data, bool depth, uint32_t gw, uint32_t gh) const;
    void UploadTarget(Target& t, const Surface& s);
    bool BindTargets(const State& s);
    void BindRenderTargets(Target* rt, Target* ds);
    bool DepthCopy(const State& s);
    bool TargetStale(uint32_t data) const;
    bool ReadTarget(Target& t, std::vector<uint8_t>& px);
    bool ReadTargetInto(Target& t, const Mem& m);

    // draws
    void EnsureShaders(const State& s, const Assembled& a, DrawShaders& sh);
    struct Stage {
        TexInfo info[4];
        const Img* img[4] = {};
        bool rt[4] = {};
        bool has[4] = {};    // desc[i] holds the stage's texture
        TextureDesc desc[4];
        float wide[4] = {};  // host texels per console texel of a target read into a smaller one, else 0
        float kx[4] = {1, 1, 1, 1}, ky[4] = {1, 1, 1, 1};  // host texels per guest texel of a sampled target
        float fx[4] = {1, 1, 1, 1}, fy[4] = {1, 1, 1, 1};  // the part of its image drawn this frame
        float hw[4] = {}, hh[4] = {};                       // its image's size
        bool blur[4] = {};
        bool opaque[4] = {};  // a target read as a format without alpha: alpha samples as 1 like the console's
        bool shrunk[4] = {};  // an upscaled target sampled through its console sized copy
        bool align_glow = false;
    };
    void BindTexture(const State& s, int i, Stage& st);
    const Img* BindGuestTexture(const State& s, int i, const TextureDesc& d, const FormatInfo& fi, Stage& st);
    void SetSamplersAndTextures(const State& s, const Assembled& a, const Stage& st);
    void SetPipelineState(const State& s, const Assembled& a);
    void SetConstants(const State& s, const DrawShaders& sh, const Stage& st, uint32_t gw, uint32_t gh, bool points);
    void SetViewportScissor(const State& s, uint32_t gw);
    void DrawGeometry(const Assembled& a);
    void DeliverReports();

    // present
    void CreateSwapchain(uint32_t w, uint32_t h);
    void DestroySwapchain();
    void Screenshot(const Surface& front);
    void WriteTargetTga(Target& t, const char* suffix);
    void LogPerf();
    void ReleaseOldTextures();
    void PresentPass(Target& t, uint32_t x, uint32_t y, uint32_t w, uint32_t h);

    DkDevice dev_ = nullptr;
    DkQueue queue_ = nullptr;
    DkCmdBuf cmd_ = nullptr;
    Heap image_heap_, upload_heap_, read_heap_;
    Mem descs_;  // image descriptors then sampler descriptors
    Mem reports_mem_;
    uint32_t report_next_ = 0;
    std::vector<uint32_t> free_descs_;
    std::unordered_map<uint64_t, uint32_t> samplers_;
    uint32_t sampler_next_ = 0;
    uint64_t sampler_epoch_ = 1;
    struct LastSampler {
        DkSampler value{};
        uint64_t epoch = 0;
        uint32_t slot = 0;
    } last_sampler_[4];
    static constexpr bool draw_fast_ = true;
    uint64_t draw_epoch_ = 1;
    template<class T> bool Changed(dk::DrawCache<T>& cache, const T& value) {
        perf_.state_checks++;
        bool bind = cache.Update(value, draw_epoch_, draw_fast_);
        perf_.state_emits += bind;
        return bind;
    }
    dk::DrawCache<DkColorState> color_cache_;
    dk::DrawCache<DkBlendState> blend_cache_;
    dk::DrawCache<DkColorWriteState> write_cache_;
    dk::DrawCache<DkDepthStencilState> depth_cache_;
    dk::DrawCache<DkRasterizerState> raster_cache_;
    dk::DrawCache<std::array<float, 4>> blend_const_cache_;
    dk::DrawCache<std::array<float, 3>> depth_bias_cache_;
    dk::DrawCache<std::array<uint32_t, 3>> stencil_cache_;
    dk::DrawCache<std::array<const DkShader*, 2>> shader_cache_;
    dk::DrawCache<std::array<DkResHandle, 4>> texture_cache_;
    dk::DrawCache<DkViewport> viewport_cache_;
    dk::DrawCache<DkScissor> scissor_cache_;
    struct Attributes { std::array<DkVtxAttribState, 16> data{}; uint32_t count = 0; };
    struct Buffers { std::array<DkVtxBufferState, 16> data{}; uint32_t count = 0; };
    dk::DrawCache<Attributes> attrib_cache_;
    dk::DrawCache<Buffers> buffer_cache_;
    bool descs_dirty_ = false;
    uint64_t clock_ = 1;
    bool list_work_ = false;

    Slot slots_[kFrames];
    uint64_t frame_ = 0;  // presents so far; the frame being recorded uses slot frame_ % kFrames
    std::atomic<uint64_t> done_through_{0};  // presents 0 .. done_through_ - 1 are drawn
    // WaitGpuFrame, from the title's thread: present n's fence, at n % 4 with n + 1 beside it
    std::mutex present_m_;
    DkFence present_fence_[4]{};
    uint64_t present_n_[4] = {};
    std::atomic<uint64_t> wait_ns_{0};
    DkFence sync_fence_{};

    DkSwapchain swap_ = nullptr;
    Img swap_img_[3];
    uint32_t disp_w_ = 0, disp_h_ = 0;
    NWindow* win_ = nullptr;

    std::unordered_map<uint64_t, Target> targets_;
    std::unordered_map<uint32_t, uint64_t> target_used_;
    std::unordered_map<uint64_t, Tex> textures_;
    Img dummy_[4];  // 2D, 2D, 3D, cube: a stage without a texture reads (0, 0, 0, 1)
    // the title's gamma ramp (red, green, blue, 256 each) as a 256 x 1 lookup the present pass reads, when not identity
    uint8_t ramp_[768] = {};
    bool ramp_dirty_ = false, gamma_ = false;
    Img lut_;
    const DkShader* present_vs_ = nullptr;
    const DkShader* present_ps_ = nullptr;
    Target* bound_rt_ = nullptr;
    Target* bound_ds_ = nullptr;
    uint32_t guest_w_ = 640, guest_h_ = 480;
    // host pixels per guest pixel of every target: the guest's 640x480 (anamorphic 16:9 in the title's widescreen
    // mode) drawn at the 1280x720 internal resolution
    float sx_ = 1.0f, sy_ = 1.0f;
    uint32_t aniso_ = 4;
    uint32_t HostW(uint32_t gw) const { return std::max(1u, uint32_t(std::lround(gw * sx_))); }
    uint32_t HostH(uint32_t gh) const { return std::max(1u, uint32_t(std::lround(gh * sy_))); }
    // which targets are drawn at the internal resolution: the guest's full screen ones (the scene, its depth, the
    // front buffers); the glow chain's and shadow maps' smaller targets stay at their guest size
    // dynamic resolution: a scaled target is drawn in its top left dyn_ part, set per frame from the GPU's time
    // samples scale their coordinates into that part; the
    // present stretches it over the display
    float dyn_ = 1.0f, dyn_target_ = 15.0f;
    double dyn_ms_ = 0;
    uint32_t dyn_frames_ = 0;
    // the part of a scaled target drawing writes now, at this frame's dyn_
    uint32_t NowW(const Target& t) const { return t.img.w == t.gw ? t.gw : std::max(1u, uint32_t(std::lround(t.img.w * dyn_))); }
    uint32_t NowH(const Target& t) const { return t.img.h == t.gh ? t.gh : std::max(1u, uint32_t(std::lround(t.img.h * dyn_))); }
    // the part holding the target's picture: as large as when it was last drawn (reads, copies out, samples, present)
    uint32_t AW(const Target& t) const { return t.aw ? t.aw : NowW(t); }
    uint32_t AH(const Target& t) const { return t.ah ? t.ah : NowH(t); }
    // the target is drawn into now: its picture's part becomes this frame's
    void Drawn(Target& t) { t.aw = NowW(t), t.ah = NowH(t), ++t.version; }
    bool EnsureFb(Target& t);
    Img* Shrunk(Target& t);
    void Rescale(Target& t);
    void DynamicResolution(double gpu_ms);
    Mem present_ubo_;
    // Eligible vertices use mem::GuestRamForGpu's mappable view: guest RAM is one GPU
    // memory block; a vertex range the guest has left alone for kZcStableFrames frames (same lock epoch, same sampled
    // bytes, a choice for speed only) is drawn in place, its bytes cleaned from the CPU's cache each frame; any other is
    // copied as before
    DkMemBlock guest_mem_ = nullptr;
    const uint8_t* guest_ptr_ = nullptr;
    uint32_t guest_size_ = 0;
    DkGpuAddr guest_gpu_ = 0;
    struct ZcRange {
        uint64_t sig = 0, epoch = 0, since = 0, seen = 0;
        uint64_t cleaned = 0;  // the frame (+1) its bytes were last cleaned to memory
    };
    std::unordered_map<uint64_t, ZcRange> zc_;
    bool zc_l2_ = false;          // this draw is the frame's first in place: the GPU's L2 may hold older bytes
    uint64_t zc_l2_frame_ = 0;    // the frame (+1) whose L2 invalidate was recorded
    bool InPlace(const VertexBuffer& b, DkGpuAddr& gpu);
    bool ScaleTarget(uint32_t gw, uint32_t gh) const { return gw == guest_w_ && gh == guest_h_; }
    // guest sized BGRA images that CPU pixels pass through on the way into a scaled target and out of it
    std::unordered_map<uint64_t, Img> staging_;
    Img* Staging(uint32_t w, uint32_t h);

    std::unordered_map<uint64_t, const DkShader*> vshaders_, pshaders_;
    uint64_t last_vs_key_ = 0, last_ps_key_ = 0;
    const DkShader* last_vs_ = nullptr;
    const DkShader* last_ps_ = nullptr;
    // the constant blocks last bound, in this frame's ring (0: none yet)
    std::array<std::array<float, 4>, 192> last_vc_{};
    DrawConstants last_dc_{};
    DkGpuAddr last_vc_gpu_ = 0, last_dc_gpu_ = 0;
    // the vertex constants and the draw constants live in one buffer each for the renderer's life; a draw pushes the
    // 16 byte registers that changed inline (pipelined: earlier draws keep the old values, after deko9's PushConstants)
    Mem vc_ubo_, dc_ubo_;
    bool consts_valid_ = false;
    void PushChanged(const Mem& ubo, void* shadow, const void* now, uint32_t bytes);
    ShaderCache shaders_;
    Assembled asm_;

    struct Counted {
        uint32_t before, after;  // report slots
        float area;              // host samples per guest pixel of the target drawn
    };
    struct Report {
        uint32_t va;
        std::vector<Counted> counted;
        uint64_t frame;
    };
    std::vector<Counted> counted_;
    std::vector<Report> reports_;

    struct Late {
        Mem buf[kFrames + 1];
        uint64_t frame[kFrames + 1] = {};
        uint32_t gw = 0, gh = 0, next = 0;
        uint64_t delivered = ~0ull;
        uint64_t delivered_sig = 0;  // MemSig of guest memory right after DeliverLate wrote it
        bool sig_ready = false;
    };
    std::unordered_map<uint32_t, Late> late_;
    std::mutex late_m_;  // DeliverLate runs on the title's thread

    struct Perf {
        double draw = 0, present = 0, sync = 0, wait = 0, texup = 0, compile = 0;
        double assemble = 0, bind = 0, consts = 0, geometry = 0;  // parts of draw
        uint64_t draws = 0, upload = 0, texups = 0, barriers = 0, syncs = 0, compiles = 0;
        uint64_t state_checks = 0, state_emits = 0, sampler_hits = 0;
    } perf_;
    // what the draws asked for, to find anything dropped or unsupported (logged with the perf lines)
    struct Census {
        uint64_t drop_asm = 0, drop_target = 0, drop_shader = 0, drop_vfmt = 0;
        uint64_t stage_tex[4] = {}, stage_unused[4] = {}, no_desc = 0, bad_fmt = 0, decode_fail = 0, dummy_kind_mismatch = 0;
        uint64_t cube = 0, vol = 0, p8 = 0, p8_nopal = 0, dxt = 0, linear = 0, mipped = 0, mips_dropped = 0, rt = 0, shadow = 0, wide = 0;
        uint64_t prog = 0, ff = 0, final_comb = 0, modes[32] = {}, ffops[32] = {};
        uint64_t alphatest = 0, blend = 0, points = 0, lines = 0, fog = 0, aniso = 0, lodbias = 0, specular = 0, wire = 0;
        uint32_t bad_formats[8] = {};
        uint64_t fb_copies = 0, fb_direct = 0;
        uint64_t zc_bytes = 0, zc_copied = 0, zc_cleaned = 0, zc_not_guest = 0, zc_misaligned = 0, zc_dynamic = 0;
        double fb_mb = 0;
    } census_;
    void CountDraw(const State& s, const Assembled& a, const PixelKey& pk, const Stage& st);
    void LogCensus(double n);
    Clock::time_point perf_t_ = Clock::now();
    Clock::time_point last_present_t_ = Clock::now();
    std::vector<float> intervals_;  // ms between presents since the last perf line
    // GPU timestamps of each slot's frame start and end, in reports_mem_ past the counter slots
    double gpu_ms_ = 0;
    // GPU ticks when the last frame's commands ended, and the ms between consecutive frames' ends since the last perf
    // line: the cadence the display sees (the present to present intervals above are taken on the render thread,
    // after a recording whose length varies with the draw count)
    uint64_t last_gpu_end_ = 0;
    std::vector<float> gpu_intervals_;
    std::unordered_map<uint64_t, double> pass_ms_;  // GPU ms by pass label since the last perf line
    std::string pass_seq_;                          // one frame's passes in order, for the log
    // B3_PERF_DIAG=1: the per pass GPU timestamps, the census of what draws use, the pass and draw descriptions;
    // without it only the fps, frame time and resolution lines
    bool diag_ = false;
    bool draw_dump_ = false;  // B3_DRAW_DUMP=1: diag_ plus the per-draw dump (~150 KB per dump, stalls a slow nxlink)
    bool roi_ = false;
    unsigned roi_window_ = 0, roi_samples_ = 0;
    double roi_elapsed_ = 0, roi_frames_ = 0, roi_gpu_ = 0, roi_gpu_frames_ = 0;
    double roi_capture_ = 0, roi_record_ = 0;
    u64 roi_render_ticks_ = 0, roi_game_ticks_ = 0;
    Handle roi_game_handle_ = INVALID_HANDLE;

    bool draw_quad_ = false;
    // targets in a compressible memory kind (deko9's r_deko9RtCompression, on by default there): the memory controller
    // compresses colour and depth for every GPU engine, only the CPU would see other bytes and it never maps a target
    uint32_t rt_compress_ = DkImageFlags_HwCompression;
    void Mark(uint64_t label);
    uint32_t MarkSlot(uint32_t i) const { return kReportSlots + kSpanSlots * kFrames + uint32_t(frame_ % kFrames) * kMarks + i; }
    uint32_t SpanSlot(uint32_t i) const { return kReportSlots + uint32_t(frame_ % kFrames) * kSpanSlots + i; }
    // the GPU's time for the frame: timestamps around each list it runs, so neither a read back's wait nor the
    // recording after it counts
    void SpanTimestamp();
    uint32_t gpu_frames_ = 0;
    uint32_t shot_every_ = 0;
    Clock::time_point shot_check_ = Clock::now();
};

// --- device, frames, memory ---

bool DekoBackend::Init(uint32_t width, uint32_t height) {
    guest_w_ = width;
    guest_h_ = height;
    {
        unsigned rw = 1280, rh = 720;
#ifdef B3_RENDER_DIAGNOSTICS
        diag_ = true;
#endif
        // or B3_RENDER_DIAG=1 (env.txt): the same without a rebuild
        if (const char* e = std::getenv("B3_RENDER_DIAG"); e && *e == '1') diag_ = true;
        if (const char* e = std::getenv("B3_DRAW_DUMP"); e && *e == '1') diag_ = draw_dump_ = true;
#ifdef B3_ROI_BUILD
        roi_ = true;
        dyn_target_ = 0;
        diag_ = false;
        LOGI("ROI: fixed 1280x720; production draw cache enabled; manual input");
        LOGI("ROI: keep scene/camera repeatable; title_active_wall is wall time, not CPU time; GPU spans may overlap");
#endif
        sx_ = float(rw) / float(width);
        sy_ = float(rh) / float(height);
    }
    DkDeviceMaker dm;
    dkDeviceMakerDefaults(&dm);
    dm.flags = DkDeviceFlags_DepthZeroToOne | DkDeviceFlags_OriginUpperLeft;
    dm.cbDebug = [](void*, const char* ctx, DkResult r, const char* msg) { LOGE("deko3d: %s: %d %s", ctx, int(r), msg ? msg : ""); };
    dev_ = dkDeviceCreate(&dm);
    if (!dev_) return false;
    image_heap_.Init(dev_, DkMemBlockFlags_GpuCached | DkMemBlockFlags_Image, 64u << 20);
    upload_heap_.Init(dev_, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached, kRingChunk);
    read_heap_.Init(dev_, DkMemBlockFlags_CpuCached | DkMemBlockFlags_GpuCached, 4u << 20);

    DkQueueMaker qm;
    dkQueueMakerDefaults(&qm, dev_);
    qm.flags = DkQueueFlags_Graphics | DkQueueFlags_HighPrio;
    qm.commandMemorySize = 1u << 20;
    qm.flushThreshold = 128u << 10;
    queue_ = dkQueueCreate(&qm);
    if (!queue_) return false;

    DkCmdBufMaker cm;
    dkCmdBufMakerDefaults(&cm, dev_);
    cm.userData = this;
    cm.cbAddMem = AddCmdMemory;
    cmd_ = dkCmdBufCreate(&cm);

    uint32_t desc_bytes = kImageDescs * sizeof(DkImageDescriptor) + kSamplerDescs * sizeof(DkSamplerDescriptor);
    if (!upload_heap_.Alloc(desc_bytes, 256, descs_) || !upload_heap_.Alloc((kReportSlots + kSpanSlots * kFrames + kFrames * kMarks) * 16, 256, reports_mem_)) return false;
    if (!upload_heap_.Alloc(AlignUp(sizeof(last_vc_), 256), DK_UNIFORM_BUF_ALIGNMENT, vc_ubo_) ||
        !upload_heap_.Alloc(AlignUp(sizeof(DrawConstants), 256), DK_UNIFORM_BUF_ALIGNMENT, dc_ubo_) ||
        !upload_heap_.Alloc(256, DK_UNIFORM_BUF_ALIGNMENT, present_ubo_))
        return false;
    std::memset(descs_.cpu, 0, desc_bytes);
    std::memset(reports_mem_.cpu, 0, (kReportSlots + kSpanSlots * kFrames + kFrames * kMarks) * 16);
    for (uint32_t i = kImageDescs; i-- > 0;) free_descs_.push_back(i);

    {
        mem::GuestRam g = mem::GuestRamForGpu();
        if (g.gpu_mappable) {
            DkMemBlockMaker m;
            dkMemBlockMakerDefaults(&m, dev_, g.size);
            m.flags = DkMemBlockFlags_CpuCached | DkMemBlockFlags_GpuCached;
            m.storage = g.ptr;
            guest_mem_ = dkMemBlockCreate(&m);
        }
        if (guest_mem_) {
            guest_ptr_ = g.ptr;
            guest_size_ = g.size;
            guest_gpu_ = dkMemBlockGetGpuAddr(guest_mem_);
        }
        LOGI("deko3d: zero copy %s (guest RAM %p, %u MB, mappable %d)", guest_mem_ ? "on" : "off", static_cast<void*>(g.ptr),
             g.size >> 20, int(g.gpu_mappable));
    }
    shaders_.Init(dev_, "sdmc:/switch/burnout3/shaders", dk::kGeneratorVersion);
    win_ = nwindowGetDefault();
    bool docked = appletGetOperationMode() == AppletOperationMode_Console;
    CreateSwapchain(docked ? 1920 : 1280, docked ? 1080 : 720);

    BeginFrame();
    // unbound stages: 1x1 (0, 0, 0, 1) of each kind
    static const uint8_t black[4 * 6] = {0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255};
    for (int k = 0; k < 4; k++) {
        DkImageType type = k == kTex3D ? DkImageType_3D : k == kTexCube ? DkImageType_Cubemap : DkImageType_2D;
        CreateImage(dummy_[k], type, DkImageFormat_BGRA8_Unorm, 1, 1, type == DkImageType_Cubemap ? 6 : 1, 1,
                    type == DkImageType_3D ? 0 : DkImageFlags_Usage2DEngine);
        for (uint32_t f = 0; f < (type == DkImageType_Cubemap ? 6u : 1u); f++) UploadPixels(dummy_[k], 0, f, black, 4, 1, 1, 1, 4, 1);
    }
    CreateImage(lut_, DkImageType_2D, DkImageFormat_BGRA8_Unorm, 256, 1, 1, 1, DkImageFlags_Usage2DEngine);
    present_vs_ = shaders_.Get(ShaderCache::kVertex, kPresentVs);
    present_ps_ = shaders_.Get(ShaderCache::kFragment, kPresentPs);
    Submit(true);
    if (FILE* f = std::fopen("sdmc:/switch/burnout3/shots", "r")) {
        unsigned every = 0;
        if (std::fscanf(f, "%u", &every) == 1) shot_every_ = every;
        std::fclose(f);
    }
    g_self = this;
    LOGI("deko3d: renderer up, guest %ux%u drawn at %ux%u, display %ux%u, anisotropy %u", width, height, HostW(width), HostH(height),
         disp_w_, disp_h_, aniso_);
    return true;
}

void DekoBackend::CreateSwapchain(uint32_t w, uint32_t h) {
    for (Img& img : swap_img_) {
        DkImageLayoutMaker lm;
        dkImageLayoutMakerDefaults(&lm, dev_);
        lm.flags = DkImageFlags_UsageRender | DkImageFlags_UsagePresent | DkImageFlags_Usage2DEngine | DkImageFlags_HwCompression;
        lm.format = DkImageFormat_RGBA8_Unorm;
        lm.dimensions[0] = w;
        lm.dimensions[1] = h;
        dkImageLayoutInitialize(&img.layout, &lm);
        image_heap_.Alloc(uint32_t(dkImageLayoutGetSize(&img.layout)), dkImageLayoutGetAlignment(&img.layout), img.mem);
        dkImageInitialize(&img.image, &img.layout, img.mem.block, img.mem.offset);
        img.w = w;
        img.h = h;
        img.fmt = DkImageFormat_RGBA8_Unorm;
    }
    const DkImage* imgs[3] = {&swap_img_[0].image, &swap_img_[1].image, &swap_img_[2].image};
    DkSwapchainMaker sm;
    dkSwapchainMakerDefaults(&sm, dev_, win_, imgs, 3);
    swap_ = dkSwapchainCreate(&sm);
    dkSwapchainSetSwapInterval(swap_, 1);
    disp_w_ = w;
    disp_h_ = h;
}

// the queue is idle
void DekoBackend::DestroySwapchain() {
    dkSwapchainDestroy(swap_);
    swap_ = nullptr;
    for (Img& img : swap_img_) {
        image_heap_.Free(img.mem);
        img = Img{};
    }
}

void DekoBackend::AddCmdMemory(void* user, DkCmdBuf cmd, size_t min) {
    auto* self = static_cast<DekoBackend*>(user);
    Slot& slot = self->Cur();
    if (slot.cmd_next >= slot.cmd.size() || slot.cmd[slot.cmd_next].size < min) {
        Mem m;
        if (!self->upload_heap_.Alloc(std::max<uint32_t>(kCmdChunk, AlignUp(uint32_t(min), 4096)), 256, m)) return;
        slot.cmd.insert(slot.cmd.begin() + slot.cmd_next, m);
    }
    const Mem& m = slot.cmd[slot.cmd_next++];
    dkCmdBufAddMemory(cmd, m.block, m.offset, m.size);
}

// the slot of the frame about to be recorded: its previous use done, its memory free again
void DekoBackend::BeginFrame() {
    ++draw_epoch_;
    Slot& slot = Cur();
    if (slot.pending) {
        auto t = Clock::now();
        dkFenceWait(&slot.fence, -1);
        perf_.wait += Ms(t);
        slot.pending = false;
        // GPU timer ticks (614.4 MHz) to ms
        uint64_t b = 0, busy = 0;
        for (uint32_t i = 0; i + 1 < slot.spans; i += 2) {
            uint64_t t0, t1;
            std::memcpy(&t0, reports_mem_.cpu + SpanSlot(i) * 16 + 8, 8);
            std::memcpy(&t1, reports_mem_.cpu + SpanSlot(i + 1) * 16 + 8, 8);
            if (t1 > t0) busy += t1 - t0;
            b = t1;
        }
        if (busy) {
            double ms = double(busy) * 625.0 / 384.0 / 1e6;
            gpu_ms_ += ms, gpu_frames_++;
            DynamicResolution(ms);
        }
        if (b) {
            if (last_gpu_end_ && b > last_gpu_end_) gpu_intervals_.push_back(float(double(b - last_gpu_end_) * 625.0 / 384.0 / 1e6));
            last_gpu_end_ = b;
        }
        // each pass from its mark to the next, the last to the frame's end
        for (size_t m = 0; m < slot.marks.size(); m++) {
            uint64_t t0, t1;
            std::memcpy(&t0, reports_mem_.cpu + MarkSlot(uint32_t(m)) * 16 + 8, 8);
            if (m + 1 < slot.marks.size()) std::memcpy(&t1, reports_mem_.cpu + MarkSlot(uint32_t(m + 1)) * 16 + 8, 8);
            else t1 = b;
            if (t1 > t0) pass_ms_[slot.marks[m]] += double(t1 - t0) * 625.0 / 384.0 / 1e6;
            if (diag_ && frame_ % 300 == 0 && m < 48) {
                // a frame's passes in order, every few seconds
                char one[96];
                uint64_t l = slot.marks[m];
                std::snprintf(one, sizeof(one), " [%zu %ux%u%s%s %u/%u %.2f]", m, uint32_t(l >> 32), uint32_t(l >> 16) & 0xFFFF,
                              l & 4 ? "*" : "", l & 2 ? "+z" : "", slot.mark_draws[m] & 0xFFFF, slot.mark_draws[m] >> 16,
                              t1 > t0 ? double(t1 - t0) * 625.0 / 384.0 / 1e6 : 0.0);
                pass_seq_ += one;
            }
        }
        if (!pass_seq_.empty()) {
            LOGI("deko3d: passes (target draws/blended ms), wide %s:%s", frame_ / 600 % 2 ? "B" : "A", pass_seq_.c_str());
            pass_seq_.clear();
        }
    }
    if (frame_ >= kFrames) done_through_.store(std::max<uint64_t>(done_through_.load(), frame_ - kFrames + 1));
    for (const Mem& m : slot.garbage) {
        // command and ring chunks never come here; everything else is image memory or read back memory
        if (m.size) image_heap_.Free(m);
    }
    slot.garbage.clear();
    for (const Mem& m : slot.read_garbage) read_heap_.Free(m);
    slot.read_garbage.clear();
    for (uint32_t d : slot.free_descs) free_descs_.push_back(d);
    slot.free_descs.clear();
    slot.marks.clear();
    slot.mark_draws.clear();
    slot.ring_chunk = 0;
    slot.ring_used = 0;
    slot.cmd_next = 0;
    dkCmdBufClear(cmd_);
    AddCmdMemory(this, cmd_, kCmdChunk);
    dkCmdBufBindImageDescriptorSet(cmd_, descs_.gpu, kImageDescs);
    dkCmdBufBindSamplerDescriptorSet(cmd_, descs_.gpu + kImageDescs * sizeof(DkImageDescriptor), kSamplerDescs);
    bound_rt_ = bound_ds_ = nullptr;
    last_vc_gpu_ = last_dc_gpu_ = 0;
    dkCmdBufBindUniformBuffer(cmd_, DkStage_Vertex, 0, vc_ubo_.gpu, vc_ubo_.size);
    dkCmdBufBindUniformBuffer(cmd_, DkStage_Vertex, 1, dc_ubo_.gpu, dc_ubo_.size);
    dkCmdBufBindUniformBuffer(cmd_, DkStage_Fragment, 1, dc_ubo_.gpu, dc_ubo_.size);
    Cur().spans = 0;
    SpanTimestamp();
    DeliverReports();
}

DekoBackend::Span DekoBackend::Upload(uint32_t size, uint32_t align) {
    Slot& slot = Cur();
    perf_.upload += size;
    for (;;) {
        if (slot.ring_chunk < slot.ring.size()) {
            Mem& m = slot.ring[slot.ring_chunk];
            uint32_t at = AlignUp(slot.ring_used, align);
            if (uint64_t(at) + size <= m.size) {
                slot.ring_used = at + size;
                return {m.cpu + at, m.gpu + at};
            }
            if (slot.ring_used == 0 && size > m.size) {
                // a chunk too small for this alone: a bigger one in its place, the small one goes with the frame
                Mem big;
                if (!upload_heap_.Alloc(AlignUp(size, 4096), 256, big)) return {nullptr, 0};
                slot.ring.insert(slot.ring.begin() + slot.ring_chunk, big);
                continue;
            }
            slot.ring_chunk++;
            slot.ring_used = 0;
            continue;
        }
        Mem m;
        if (!upload_heap_.Alloc(std::max(kRingChunk, AlignUp(size, 4096)), 256, m)) {
            LOGE("deko3d: upload ring out of memory");
            return {nullptr, 0};
        }
        slot.ring.push_back(m);
    }
}

// the recorded commands to the GPU; wait: until it ran them
void DekoBackend::SpanTimestamp() {
    Slot& slot = Cur();
    if (slot.spans >= kSpanSlots) return;
    dkCmdBufReportCounter(cmd_, DkCounter_Timestamp, reports_mem_.gpu + SpanSlot(slot.spans++) * 16);
}

void DekoBackend::Submit(bool wait) {
    // the list's end; the commands after it open a new span (the last pair is cut when they would not fit)
    if (Cur().spans % 2 == 1 && Cur().spans + 2 < kSpanSlots) SpanTimestamp();
    DkCmdList list = dkCmdBufFinishList(cmd_);
    dkQueueSubmitCommands(queue_, list);
    unsubmitted_draws_ = 0;
    if (wait) {
        dkQueueSignalFence(queue_, &sync_fence_, true);
        dkQueueFlush(queue_);
        auto t = Clock::now();
        dkFenceWait(&sync_fence_, -1);
        perf_.wait += Ms(t);
    } else {
        dkQueueFlush(queue_);
    }
    if (Cur().spans % 2 == 0 && Cur().spans + 2 < kSpanSlots) SpanTimestamp();
}

void DekoBackend::Barrier() {
    dkCmdBufBarrier(cmd_, DkBarrier_Full, DkInvalidateFlags_Image | DkInvalidateFlags_L2Cache);
    clock_++;
    perf_.barriers++;
}

// before an access to img: a barrier when an access of this epoch conflicts with it (deko9's HazardCheck)
void DekoBackend::Hazard(Img& img, Access a) {
    uint64_t c = clock_;
    bool hit = false;
    switch (a) {
        case Access::Sample: hit = img.copy_write == c || img.render == c; break;
        case Access::Render: hit = img.copy_write == c || img.read == c; break;
        case Access::CopyRead: hit = img.render == c || img.copy_write == c; break;
        case Access::CopyWrite: hit = img.read == c || img.render == c; break;
    }
    if (hit) Barrier();
    HazardsDone(img, a);
}

void DekoBackend::HazardsDone(Img& img, Access a) {
    switch (a) {
        case Access::Sample: case Access::CopyRead: img.read = clock_; break;
        case Access::Render: img.render = clock_; break;
        case Access::CopyWrite: img.copy_write = clock_; break;
    }
}

// --- images ---

bool DekoBackend::CreateImage(Img& img, DkImageType type, DkImageFormat fmt, uint32_t w, uint32_t h, uint32_t d,
                              uint32_t levels, uint32_t flags) {
    DkImageLayoutMaker lm;
    dkImageLayoutMakerDefaults(&lm, dev_);
    lm.type = type;
    lm.flags = flags;
    lm.format = fmt;
    lm.dimensions[0] = w;
    lm.dimensions[1] = h;
    lm.dimensions[2] = type == DkImageType_Cubemap ? 6 : type == DkImageType_3D ? d : 1;
    lm.mipLevels = levels;
    bool bc = fmt == DkImageFormat_RGBA_BC1 || fmt == DkImageFormat_RGBA_BC2 || fmt == DkImageFormat_RGBA_BC3;
    if (bc && type == DkImageType_2D) {
        // deko3d's own tile height for block compressed images disagrees with the copy engine's for some heights
        // (6 to 8 block rows): the standard block linear height (from deko9)
        uint32_t gobs = ((h + 3) / 4 + 7) / 8;
        uint32_t tile = DkTileSize_SixteenGobs;
        while (tile && (1u << (tile - 1)) >= gobs) --tile;
        lm.flags |= DkImageFlags_CustomTileSize;
        lm.tileSize = DkTileSize(tile);
    }
    dkImageLayoutInitialize(&img.layout, &lm);
    uint64_t size = dkImageLayoutGetSize(&img.layout);
    if (size > 0xFFFFFFFFull || !image_heap_.Alloc(uint32_t(size), dkImageLayoutGetAlignment(&img.layout), img.mem)) {
        LOGE("deko3d: no memory for a %ux%u image", w, h);
        return false;
    }
    dkImageInitialize(&img.image, &img.layout, img.mem.block, img.mem.offset);
    img.fmt = fmt;
    img.type = type;
    img.w = w;
    img.h = h;
    img.d = d;
    img.levels = levels;
    img.depth = fmt == DkImageFormat_Z24S8;
    img.desc = AllocDesc(img);
    return img.desc != ~0u;
}

// memory and descriptor go once the frames using them are done
void DekoBackend::FreeImage(Img& img) {
    if (!img.live()) return;
    Cur().garbage.push_back(img.mem);
    if (img.desc != ~0u) Cur().free_descs.push_back(img.desc);
    if (img.desc_opaque != ~0u) Cur().free_descs.push_back(img.desc_opaque);
    img = Img{};
}

void DekoBackend::ImageView(const Img& img, DkImageView& v) {
    dkImageViewDefaults(&v, &img.image);
    if (img.depth) v.dsSource = DkDsSource_Depth;
}

uint32_t DekoBackend::AllocDesc(const Img& img, bool opaque) {
    if (free_descs_.empty()) {
        LOGE("deko3d: out of image descriptors");
        return ~0u;
    }
    uint32_t d = free_descs_.back();
    free_descs_.pop_back();
    DkImageView v;
    ImageView(img, v);
    if (opaque) v.swizzle[3] = DkImageSwizzle_One;
    auto* table = reinterpret_cast<DkImageDescriptor*>(descs_.cpu);
    dkImageDescriptorInitialize(&table[d], &v, false, false);
    descs_dirty_ = true;
    return d;
}

uint32_t DekoBackend::SamplerSlot(const DkSampler& sm) {
    uint64_t key = Hash(&sm, sizeof(sm));
    auto it = samplers_.find(key);
    if (it != samplers_.end()) return it->second;
    if (sampler_next_ >= kSamplerDescs) {
        // every slot taken: frames in flight may still use them, so wait for the GPU before writing over them
        Submit(true);
        samplers_.clear();
        ++sampler_epoch_;
        sampler_next_ = 0;
    }
    uint32_t slot = sampler_next_++;
    auto* table = reinterpret_cast<DkSamplerDescriptor*>(descs_.cpu + kImageDescs * sizeof(DkImageDescriptor));
    dkSamplerDescriptorInitialize(&table[slot], &sm);
    samplers_[key] = slot;
    descs_dirty_ = true;
    return slot;
}

// rows of px (pitch bytes apart, row_bytes each, rows per slice) into level and face of img through the ring
void DekoBackend::UploadPixels(Img& img, uint32_t level, uint32_t face, const uint8_t* px, uint32_t pitch, uint32_t w,
                               uint32_t h, uint32_t depth, uint32_t row_bytes, uint32_t rows) {
    uint32_t bytes = row_bytes * rows * depth;
    Span sp = Upload(bytes, 256);
    if (!sp.cpu) return;
    for (uint32_t z = 0; z < depth; z++)
        for (uint32_t y = 0; y < rows; y++)
            std::memcpy(sp.cpu + (size_t(z) * rows + y) * row_bytes, px + (size_t(z) * rows + y) * pitch, row_bytes);
    Hazard(img, Access::CopyWrite);
    DkImageView v;
    dkImageViewDefaults(&v, &img.image);
    v.mipLevelOffset = uint8_t(level);
    v.mipLevelCount = 1;
    DkImageRect r{0, 0, 0, w, h, depth};
    if (img.type == DkImageType_Cubemap) r.z = face, r.depth = 1;
    DkCopyBuf cb{sp.gpu, 0, 0};
    dkCmdBufCopyBufferToImage(cmd_, &cb, &v, &r, 0);
    list_work_ = true;
}

// --- targets ---

constexpr int kKeyWidthShift = 34;
constexpr int kKeyHeightShift = 48;

// a smaller depth surface laid on the depth buffer's memory is a target of its own
uint64_t DekoBackend::SizedTargetKey(uint32_t data, bool depth, uint32_t gw, uint32_t gh) const {
    uint64_t key = TargetMapKey(data, depth);
    if (!(depth && (gw != guest_w_ || gh != guest_h_))) return key;
    return key | uint64_t(gw) << kKeyWidthShift | uint64_t(gh) << kKeyHeightShift;
}

Target* DekoBackend::FindTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    uint32_t data = mem::VirtualFromPhysical(s.data), gw, gh;
    SurfaceSize(s, gw, gh);
    auto it = targets_.find(SizedTargetKey(data, depth, gw, gh));
    if (it == targets_.end() || it->second.gw != gw || it->second.gh != gh) return nullptr;
    target_used_[data] = frame_;
    return &it->second;
}

void DekoBackend::ReleaseTarget(std::unordered_map<uint64_t, Target>::iterator it) {
    if (bound_rt_ == &it->second || bound_ds_ == &it->second) bound_rt_ = bound_ds_ = nullptr;
    FreeImage(it->second.img);
    if (it->second.fb) FreeImage(*it->second.fb);
    if (it->second.small) FreeImage(*it->second.small);
    targets_.erase(it);
}

Target* DekoBackend::EnsureTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    uint32_t data = mem::VirtualFromPhysical(s.data), gw, gh;
    SurfaceSize(s, gw, gh);
    uint64_t key = SizedTargetKey(data, depth, gw, gh);
    target_used_[data] = frame_;
    auto it = targets_.find(key);
    if (it != targets_.end() && it->second.gw == gw && it->second.gh == gh) return &it->second;
    if (it != targets_.end()) ReleaseTarget(it);
    if (!gw || !gh || gw > 4096 || gh > 4096) return nullptr;
    Target t;
    t.depth = depth;
    t.gw = gw;
    t.gh = gh;
    bool up = ScaleTarget(gw, gh);
    if (!CreateImage(t.img, DkImageType_2D, depth ? DkImageFormat_Z24S8 : DkImageFormat_BGRA8_Unorm, up ? HostW(gw) : gw, up ? HostH(gh) : gh, 1, 1,
                     DkImageFlags_UsageRender | DkImageFlags_Usage2DEngine | rt_compress_))
        return nullptr;
    t.data = data;
    t.bytes = depth ? gw * gh * 4 : GuestPitch(gw) * gh;
    t.epoch = DirtyEpoch(data);
    t.sig = MemSig(data, t.bytes);
    Target& ref = targets_[key] = std::move(t);
    // a new target starts cleared, as the D3D12 backend's do
    BindRenderTargets(depth ? nullptr : &ref, depth ? &ref : nullptr);
    DkScissor sc{0, 0, ref.img.w, ref.img.h};
    ++draw_epoch_;  // target initialization changes dynamic scissor state
    dkCmdBufSetScissors(cmd_, 0, &sc, 1);
    if (depth) dkCmdBufClearDepthStencil(cmd_, true, 1.0f, 0xFF, 0);
    else dkCmdBufClearColorFloat(cmd_, 0, DkColorMask_RGBA, 0, 0, 0, 0);
    list_work_ = true;
    return &ref;
}

// the target's copy for draws that sample the target they render to, made on first use
bool DekoBackend::EnsureFb(Target& t) {
    if (t.fb) return true;
    t.fb = std::make_unique<Img>();
    if (CreateImage(*t.fb, DkImageType_2D, t.img.fmt, t.img.w, t.img.h, 1, 1, DkImageFlags_Usage2DEngine)) return true;
    t.fb.reset();
    return false;
}

// dynamic resolution changed since the target was last drawn: its picture, in the part of the old scale, stretched
// over the part of this frame's. Feedback buffers (the motion blur, which blends each frame over the last) otherwise
// keep a stale band at the bottom and right edges after the scale grows, and read a shifted picture after it shrinks.
void DekoBackend::Rescale(Target& t) {
    const uint32_t w = NowW(t), h = NowH(t);
    if (t.depth || !t.aw || (t.aw == w && t.ah == h) || !EnsureFb(t)) return;
    DkImageView iv, fv;
    ImageView(t.img, iv);
    ImageView(*t.fb, fv);
    DkImageRect old{0, 0, 0, t.aw, t.ah, 1}, now{0, 0, 0, w, h, 1};
    Hazard(t.img, Access::CopyRead);
    Hazard(*t.fb, Access::CopyWrite);
    dkCmdBufBlitImage(cmd_, &iv, &old, &fv, &old, DkBlitFlag_FilterNearest, 0);
    Hazard(*t.fb, Access::CopyRead);
    Hazard(t.img, Access::CopyWrite);
    dkCmdBufBlitImage(cmd_, &fv, &old, &iv, &now, DkBlitFlag_FilterLinear, 0);
    list_work_ = true;
}

// the target's picture at the console's size, remade when the target was written since: the drawn part through the
// 2D engine's linear filter, at 720p's 2x a 2x2 box of host texels per console texel, at dynamic resolution's smaller
// scales the nearest four weighted. A pass shrinking the target then samples console texels with the title's own
// taps, as the console did
Img* DekoBackend::Shrunk(Target& t) {
    if (t.depth) return nullptr;
    if (!t.small) {
        t.small = std::make_unique<Img>();
        if (!CreateImage(*t.small, DkImageType_2D, t.img.fmt, t.gw, t.gh, 1, 1, DkImageFlags_Usage2DEngine)) {
            t.small.reset();
            return nullptr;
        }
    }
    if (t.small_version != t.version) {
        Hazard(t.img, Access::CopyRead);
        Hazard(*t.small, Access::CopyWrite);
        DkImageView sv, dv;
        ImageView(t.img, sv);
        ImageView(*t.small, dv);
        DkImageRect sr{0, 0, 0, AW(t), AH(t), 1}, dr{0, 0, 0, t.gw, t.gh, 1};
        dkCmdBufBlitImage(cmd_, &sv, &sr, &dv, &dr, DkBlitFlag_FilterLinear, 0);
        t.small_version = t.version;
        list_work_ = true;
    }
    return t.small.get();
}

void DekoBackend::BindRenderTargets(Target* rt, Target* ds) {
    if (rt) Rescale(*rt);
    if (rt) Drawn(*rt);
    if (ds) Drawn(*ds);
    if (rt) Hazard(rt->img, Access::Render);
    if (ds) Hazard(ds->img, Access::Render);
    if (rt == bound_rt_ && ds == bound_ds_) return;
    // a pass of many draws is done (the scene, before its post passes): the GPU starts on it now instead of at Present
    if (unsubmitted_draws_ >= kSubmitDraws) Submit(false);
    DkImageView cv, dv;
    if (rt) ImageView(rt->img, cv);
    if (ds) dkImageViewDefaults(&dv, &ds->img.image);
    const DkImageView* colors[1] = {&cv};
    dkCmdBufBindRenderTargets(cmd_, colors, rt ? 1 : 0, ds ? &dv : nullptr);
    // the pass label: the colour target's guest size (else the depth target's), host scaled, depth bound
    const Target* lt = rt ? rt : ds;
    Mark(uint64_t(lt->gw) << 32 | uint64_t(lt->gh) << 16 | (lt->img.w != lt->gw ? 4 : 0) | (ds ? 2 : 0) | (rt ? 0 : 8));
    bound_rt_ = rt;
    bound_ds_ = ds;
}

// CPU pixels of a colour surface into its target
void DekoBackend::UploadTarget(Target& t, const Surface& s) {
    TextureDesc d;
    if (!Describe(s.header, d) || Format(d.format).depth || d.width != t.gw || d.height != t.gh) return;
    if (diag_) LOG_FIRST(200, "deko3d: CPU pixels over target %08X (%ux%u) at frame %llu", t.data, t.gw, t.gh, (unsigned long long)frame_);
    std::vector<uint8_t> px;
    uint32_t pitch;
    DecodeLevel(d, 0, 0, static_cast<const uint8_t*>(kernel::HostPtr(d.data)), nullptr, px, pitch);
    if (bound_rt_ == &t) bound_rt_ = nullptr, bound_ds_ = nullptr;
    Drawn(t);
    if (t.img.w == t.gw && t.img.h == t.gh) {
        UploadPixels(t.img, 0, 0, px.data(), pitch, t.gw, t.gh, 1, t.gw * 4, t.gh);
    } else if (Img* st = Staging(t.gw, t.gh)) {
        // guest pixels into the staging image, then stretched over the target by the 2D engine
        UploadPixels(*st, 0, 0, px.data(), pitch, t.gw, t.gh, 1, t.gw * 4, t.gh);
        Hazard(*st, Access::CopyRead);
        Hazard(t.img, Access::CopyWrite);
        DkImageView sv, dv;
        ImageView(*st, sv);
        ImageView(t.img, dv);
        DkImageRect sr{0, 0, 0, t.gw, t.gh, 1}, dr{0, 0, 0, AW(t), AH(t), 1};
        dkCmdBufBlitImage(cmd_, &sv, &sr, &dv, &dr, DkBlitFlag_FilterLinear, 0);
    }
    t.epoch = DirtyEpoch(t.data);
    t.sig = MemSig(t.data, t.bytes);
}

bool DekoBackend::BindTargets(const State& s) {
    Target* rt = EnsureTarget(s.rt, false);
    Target* ds = s.rs[kRsZEnable] || s.rs[kRsStencilEnable] ? EnsureTarget(s.zs, true) : nullptr;
    if (rt && ds && (rt->gw != ds->gw || rt->gh != ds->gh)) ds = nullptr;
    if (!rt && !ds) return false;
    if (rt && LatchCpuWrite(*rt)) UploadTarget(*rt, s.rt);
    if (rt) rt->drawn = true;
    if (ds) ds->drawn = true;
    BindRenderTargets(rt, ds);
    return true;
}

// long unused: the title freed it and other data is at this address now
bool DekoBackend::TargetStale(uint32_t data) const {
    auto it = target_used_.find(data);
    uint64_t used = it == target_used_.end() ? 0 : it->second;
    return frame_ - used > kStaleFrames;
}

// a copy of the target's pixels, rows packed BGRA at its guest size, into m, recorded
bool DekoBackend::ReadTargetInto(Target& t, const Mem& m) {
    Img* src = &t.img;
    if (t.img.w != t.gw || t.img.h != t.gh) {
        // shrunk to guest size by the 2D engine first
        Img* st = Staging(t.gw, t.gh);
        if (!st) return false;
        Hazard(t.img, Access::CopyRead);
        Hazard(*st, Access::CopyWrite);
        DkImageView sv, dv;
        ImageView(t.img, sv);
        ImageView(*st, dv);
        DkImageRect sr{0, 0, 0, AW(t), AH(t), 1}, dr{0, 0, 0, t.gw, t.gh, 1};
        dkCmdBufBlitImage(cmd_, &sv, &sr, &dv, &dr, DkBlitFlag_FilterLinear, 0);
        src = st;
    }
    Hazard(*src, Access::CopyRead);
    DkImageView v;
    ImageView(*src, v);
    DkImageRect r{0, 0, 0, t.gw, t.gh, 1};
    DkCopyBuf cb{m.gpu, 0, 0};
    dkCmdBufCopyImageToBuffer(cmd_, &v, &r, &cb, 0);
    list_work_ = true;
    return true;
}

// the target's pixels, rows packed BGRA; waits for the GPU
bool DekoBackend::ReadTarget(Target& t, std::vector<uint8_t>& px) {
    uint32_t bytes = t.gw * t.gh * 4;
    Mem m;
    if (!read_heap_.Alloc(bytes, 256, m)) return false;
    if (!ReadTargetInto(t, m)) {
        read_heap_.Free(m);
        return false;
    }
    Submit(true);
    armDCacheFlush(m.cpu, bytes);
    px.assign(m.cpu, m.cpu + bytes);
    read_heap_.Free(m);
    return true;
}

Img* DekoBackend::Staging(uint32_t w, uint32_t h) {
    Img& img = staging_[uint64_t(w) << 32 | h];
    if (!img.live() && !CreateImage(img, DkImageType_2D, DkImageFormat_BGRA8_Unorm, w, h, 1, 1, DkImageFlags_Usage2DEngine)) {
        staging_.erase(uint64_t(w) << 32 | h);
        return nullptr;
    }
    return &img;
}

// --- shaders ---

void DekoBackend::EnsureShaders(const State& s, const Assembled& a, DrawShaders& sh) {
    if (last_vs_ && last_vs_key_ == sh.vkey) {
        sh.vs = last_vs_;
        return;
    }
    auto vit = vshaders_.find(sh.vkey);
    if (vit != vshaders_.end()) {
        sh.vs = vit->second;
    } else {
        auto t = Clock::now();
        std::string src = sh.prog ? dk::VertexProgramGlsl(s.vs->program, a.regs, a.decode, sh.quad)
                                  : dk::FixedVertexGlsl(a.regs, a.pretransformed, a.decode, sh.fk, sh.quad);
        sh.vs = shaders_.Get(ShaderCache::kVertex, src);
        vshaders_[sh.vkey] = sh.vs;
        perf_.compile += Ms(t);
        perf_.compiles++;
    }
    last_vs_key_ = sh.vkey;
    last_vs_ = sh.vs;
}

// --- textures ---

const Img* DekoBackend::BindGuestTexture(const State& s, int i, const TextureDesc& d, const FormatInfo& fi, Stage& st) {
    // a P8 texture's palette: its colours (256, fewer by the size code in the top two bits) are part of its key
    uint32_t colours[256];
    uint32_t pal_va = 0;
    uint64_t pal_hash = 0;
    if (d.format == 0x0B && s.palettes[size_t(i)]) {  // D3DFMT_P8
        std::memset(colours, 0, sizeof(colours));
        uint32_t header = s.palettes[size_t(i)];
        pal_va = mem::VirtualFromPhysical(kernel::R32(header + 4));
        size_t bytes = (256u >> (kernel::R32(header) >> 30)) * 4;
        std::memcpy(colours, kernel::HostPtr(pal_va), bytes);
        pal_hash = Hash(colours, sizeof(colours));
    }
    uint64_t key = uint64_t(pal_va) << 32 | d.data;
    Tex& tx = textures_[key];
    bool same = tx.img.live() && tx.d.format == d.format && tx.d.width == d.width && tx.d.height == d.height &&
                tx.d.depth == d.depth && tx.d.levels == d.levels && tx.d.cube == d.cube && tx.d.pitch == d.pitch;
    if (!same) {
        FreeImage(tx.img);
        tx = Tex{};
        tx.d = d;
        DkImageFormat fmt = HostImageFormat(fi.host);
        uint32_t w = d.width, h = d.height;
        tx.levels = d.levels;
        if (fi.compressed && ((w & 3) || (h & 3))) {
            w = (w + 3) & ~3u;
            h = (h + 3) & ~3u;
            tx.levels = 1;
        }
        DkImageType type = d.cube ? DkImageType_Cubemap : d.depth > 1 ? DkImageType_3D : DkImageType_2D;
        if (!CreateImage(tx.img, type, fmt, w, h, d.depth, tx.levels, type == DkImageType_3D ? 0 : DkImageFlags_Usage2DEngine)) {
            textures_.erase(key);
            return nullptr;
        }
    }
    // cheap change check once per frame: lock epochs plus a sample of the bytes
    if (tx.frame != frame_ + 1 || !same || tx.palette != pal_hash) {
        tx.frame = frame_ + 1;
        tx.palette = pal_hash;
        uint64_t sig = MemSig(d.data, TotalSize(d)) ^ DirtyEpoch(d.data) ^ pal_hash;
        if (sig != tx.sig || !same) {
            auto t = Clock::now();
            tx.sig = sig;
            const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(d.data));
            std::vector<uint8_t> px;
            for (uint32_t f = 0; f < (d.cube ? 6u : 1u); f++)
                for (uint32_t l = 0; l < tx.levels; l++) {
                    uint32_t pitch;
                    DecodeLevel(d, l, f, p, pal_va ? colours : nullptr, px, pitch);
                    uint32_t lw = std::max(1u, tx.img.w >> l), lh = std::max(1u, tx.img.h >> l);
                    uint32_t ld = d.depth > 1 ? std::max(1u, d.depth >> l) : 1;
                    uint32_t rows = fi.compressed ? (lh + 3) / 4 : lh;
                    uint32_t row_bytes = fi.compressed ? (lw + 3) / 4 * (fi.host == Host::BC1 ? 8 : 16)
                                                       : lw * (fi.host == Host::R16Unorm ? 2 : 4);
                    if (!pitch || px.size() < size_t(pitch) * rows * ld) {
                        census_.decode_fail++;
                        LOG_FIRST(8, "deko3d: texture %08X format %02X %ux%u level %u not decoded", d.data, d.format, d.width, d.height, l);
                        continue;
                    }
                    UploadPixels(tx.img, l, f, px.data(), pitch, lw, lh, ld, std::min(row_bytes, pitch), rows);
                }
            perf_.texup += Ms(t);
            perf_.texups++;
        }
    }
    if (fi.depth) st.info[i].shadow = ShadowDepthKind(d.format);
    return &tx.img;
}

void DekoBackend::BindTexture(const State& s, int i, Stage& st) {
    TexInfo& info = st.info[i];
    info = {};
    st.img[i] = nullptr;
    st.rt[i] = false;
    st.has[i] = false;
    st.kx[i] = st.ky[i] = st.fx[i] = st.fy[i] = 1;
    TextureDesc& d = st.desc[i];
    if (!s.textures[size_t(i)]) return;
    if (!Describe(s.textures[size_t(i)], d)) {
        census_.no_desc++;
        return;
    }
    st.has[i] = true;
    const FormatInfo& fi = Format(d.format);
    if (fi.host == Host::None || !fi.bits) {
        census_.bad_fmt++;
        for (uint32_t& f : census_.bad_formats)
            if (!f || f == d.format + 1) {
                f = d.format + 1;
                break;
            }
    }
    info.kind = TextureKind(d);
    info.rect = d.linear;
    // rendered by the GPU: sample the target itself; DXT data loaded over a target's memory is not it
    auto tit = targets_.find(SizedTargetKey(d.data, fi.depth, d.width, d.height));
    if (tit != targets_.end() && !d.cube && d.depth == 1 && !fi.compressed) {
        Target& t = tit->second;
        if (&t == bound_ds_) return;
        st.rt[i] = true;
        if (!t.depth && LatchCpuWrite(t)) {
            Surface su;
            su.header = s.textures[size_t(i)];
            UploadTarget(t, su);
        }
        info.kind = kTex2D;
        if (t.depth) info.shadow = ShadowDepthKind(d.format);
        // every colour target is BGRA8 here; the console stores no alpha for R5G6B5 or X8R8G8B8 surfaces and a
        // texture read of one gives 1 (the boost blur's blend alpha is its 320x240 R5G6B5 copy's alpha times the
        // vertex alpha, with the frame's alpha it faded out)
        st.opaque[i] = !t.depth && !GuestFormatHasAlpha(d.format);
        // a pass shrinking an upscaled target averages each tap over a box of console texels, else its taps skip
        // host texels and the glow ghosts (the D3D12 backend's kTexWide, src/d3d/d3d12.cpp)
        constexpr float kUpscaledMin = 1.01f;
        st.kx[i] = float(AW(t)) / float(t.gw);
        st.ky[i] = float(AH(t)) / float(t.gh);
        st.fx[i] = float(AW(t)) / float(t.img.w);
        st.fy[i] = float(AH(t)) / float(t.img.h);
        if (st.fx[i] < 1.0f || st.fy[i] < 1.0f) info.flags |= dk::kTexPart;
        st.hw[i] = float(t.img.w);
        st.hh[i] = float(t.img.h);
        float k = std::max(st.kx[i], st.ky[i]);
        Target* rt = bound_rt_;
        bool blur = rt && t.gw < guest_w_ && rt->gw < guest_w_;
        st.wide[i] = 0;
        if (!t.depth && k > kUpscaledMin && rt && rt != &t && (blur || t.gw > rt->gw || t.gh > rt->gh)) {
            // sampled at the console's size: the title's taps land on console texels (a tap per upscaled texel
            // skipped host texels and thinned the glow; a box of host texels per tap, the spread below, cost the
            // post passes 36 reads a pixel). The spread stays for a target whose copy cannot be made
            if (Img* small = Shrunk(t)) {
                st.img[i] = small;
                st.kx[i] = st.ky[i] = st.fx[i] = st.fy[i] = 1.0f;
                st.hw[i] = float(t.gw);
                st.hh[i] = float(t.gh);
                st.shrunk[i] = true;
                info.flags &= ~dk::kTexPart;
                return;
            }
            st.wide[i] = k;
            st.blur[i] = blur;
            info.flags |= kTexWide;
        }
        if (&t == bound_rt_) {
            // the console reads the pixels it is about to overwrite: the draw samples a copy
            if (!EnsureFb(t)) return;
            Hazard(t.img, Access::CopyRead);
            Hazard(*t.fb, Access::CopyWrite);
            DkImageView sv, dv;
            ImageView(t.img, sv);
            ImageView(*t.fb, dv);
            DkImageRect r{0, 0, 0, AW(t), AH(t), 1};
            // the 2D engine: dkCmdBufCopyImage goes through the copy engine, slow on block linear images (the pass
            // holding this copy and two full screen draws measured 5 ms at 720p on the OLED)
            dkCmdBufBlitImage(cmd_, &sv, &r, &dv, &r, DkBlitFlag_FilterNearest, 0);
            // the draw renders into the target the copy just read: the barrier BindRenderTargets would take
            Hazard(t.img, Access::Render);
            census_.fb_copies++;
            census_.fb_mb += AW(t) * AH(t) * 4 / 1048576.0;
            st.img[i] = t.fb.get();
            return;
        }
        st.img[i] = &t.img;
        return;
    }
    st.img[i] = BindGuestTexture(s, i, d, fi, st);
}

// D3DTEXF: 1 point, 2 linear, 3 anisotropic
void DekoBackend::SetSamplersAndTextures(const State& s, const Assembled& a, const Stage& st) {
    bool clamp = UnitTexcoords(a);
    std::array<DkResHandle, 4> handles;
    for (int i = 0; i < 4; i++) {
        const auto& t = s.tss[size_t(i)];
        DkSampler sm;
        dkSamplerDefaults(&sm);
        uint32_t mag = t[kTssMagFilter], min = t[kTssMinFilter], mip = t[kTssMipFilter];
        sm.minFilter = min >= 2 ? DkFilter_Linear : DkFilter_Nearest;
        sm.magFilter = mag >= 2 ? DkFilter_Linear : DkFilter_Nearest;
        sm.mipFilter = mip == 2 ? DkMipFilter_Linear : mip == 1 ? DkMipFilter_Nearest : DkMipFilter_None;
        if (min == 3 || mag == 3) sm.maxAnisotropy = float(std::clamp<uint32_t>(t[kTssMaxAnisotropy], 1, 16));
        // Mipmapped linear textures filtered anisotropically,
        // the floors and walls seen at a slant stay sharp at 720p
        if (aniso_ > 1 && mip && min >= 2 && !st.rt[i]) sm.maxAnisotropy = std::max(sm.maxAnisotropy, float(aniso_));
        sm.wrapMode[0] = Address(clamp && t[kTssAddressU] == 1 ? 3 : t[kTssAddressU]);
        sm.wrapMode[1] = Address(clamp && t[kTssAddressV] == 1 ? 3 : t[kTssAddressV]);
        sm.wrapMode[2] = Address(t[kTssAddressW]);
        sm.lodBias = Bits(t[kTssMipLodBias]);
        sm.lodClampMin = float(t[kTssMaxMipLevel]);
        sm.lodClampMax = mip ? 1000.0f : sm.lodClampMin;
        float bc[4];
        Color(t[kTssBorderColor], bc);
        for (int c = 0; c < 4; c++) sm.borderColor[c].value_f = bc[c];
        if (st.info[i].shadow) {
            // the console tests texel OP r, deko3d tests r OP texel
            static const uint8_t swap[8] = {0, 4, 2, 6, 1, 5, 3, 7};
            sm.compareEnable = true;
            sm.compareOp = DkCompareOp(swap[Cmp(s.rs[kRsShadowFunc]) - 1] + 1);
            sm.mipFilter = DkMipFilter_None;
            sm.maxAnisotropy = 1.0f;
        }
        const Img* img = st.img[i];
        if (!img) img = &dummy_[st.info[i].kind];
        else Hazard(const_cast<Img&>(*img), Access::Sample);
        uint32_t desc = img->desc;
        if (st.opaque[i] && st.img[i]) {
            Img& im = const_cast<Img&>(*img);
            if (im.desc_opaque == ~0u) im.desc_opaque = AllocDesc(im, true);
            if (im.desc_opaque != ~0u) desc = im.desc_opaque;
        }
        LastSampler& last = last_sampler_[i];
        uint32_t slot;
        if (last.epoch == sampler_epoch_ && !std::memcmp(&last.value, &sm, sizeof(sm))) {
            slot = last.slot;
            perf_.sampler_hits++;
        } else {
            slot = SamplerSlot(sm);
            last.value = sm;
            last.slot = slot;
            last.epoch = sampler_epoch_;
        }
        handles[i] = dkMakeTextureHandle(desc, slot);
    }
    if (descs_dirty_) {
        dkCmdBufBarrier(cmd_, DkBarrier_None, DkInvalidateFlags_Descriptors);
        descs_dirty_ = false;
    }
    if (Changed(texture_cache_, handles)) dkCmdBufBindTextures(cmd_, DkStage_Fragment, 0, handles.data(), 4);
}

// --- draws ---

constexpr uint32_t kFillWireframe = 0x1B01;  // D3DFILL_WIREFRAME
constexpr uint32_t kCullCw = 0x900;          // D3DCULL_CW
constexpr float kDepth24Max = 16777215.0f;

void DekoBackend::SetPipelineState(const State& s, const Assembled& a) {
    // blend and colour writes
    DkColorState cs;
    dkColorStateDefaults(&cs);
    bool blend = Simple(s, kMBlendEnable) != 0;
    dkColorStateSetBlendEnable(&cs, 0, blend);
    if (Changed(color_cache_, cs)) dkCmdBufBindColorState(cmd_, &cs);
    if (blend) {
        DkBlendState bs;
        dkBlendStateDefaults(&bs);
        uint32_t src = Simple(s, kMSrcBlend), dst = Simple(s, kMDestBlend), op = Simple(s, kMBlendOp);
        dkBlendStateSetOps(&bs, BlendOp(op), BlendOp(op));
        dkBlendStateSetFactors(&bs, Blend(src, false), Blend(dst, false), Blend(src, true), Blend(dst, true));
        if (Changed(blend_cache_, bs)) dkCmdBufBindBlendState(cmd_, 0, &bs);
        float bf[4];
        Color(Simple(s, kMBlendColor), bf);
        if (Changed(blend_const_cache_, std::array<float, 4>{bf[0], bf[1], bf[2], bf[3]}))
            dkCmdBufSetBlendConst(cmd_, bf[0], bf[1], bf[2], bf[3]);
    }
    uint32_t cw = Simple(s, kMColorWriteEnable);
    // console ARGB bytes to RGBA bits
    uint32_t mask = ((cw >> 16) & 1) | ((cw >> 8) & 1) << 1 | (cw & 1) << 2 | ((cw >> 24) & 1) << 3;
    DkColorWriteState ws;
    dkColorWriteStateDefaults(&ws);
    dkColorWriteStateSetMask(&ws, 0, mask);
    if (Changed(write_cache_, ws)) dkCmdBufBindColorWriteState(cmd_, &ws);

    // depth and stencil
    DkDepthStencilState ds;
    dkDepthStencilStateDefaults(&ds);
    bool has_ds = bound_ds_ != nullptr;
    ds.depthTestEnable = has_ds && s.rs[kRsZEnable];
    ds.depthWriteEnable = ds.depthTestEnable && Simple(s, kMZWriteEnable);
    ds.depthCompareOp = Cmp(Simple(s, kMZFunc));
    ds.stencilTestEnable = has_ds && s.rs[kRsStencilEnable];
    if (ds.stencilTestEnable) {
        DkStencilOp fail = StencilOp(s.rs[kRsStencilFail]), zfail = StencilOp(Simple(s, kMStencilZFail)),
                    pass = StencilOp(Simple(s, kMStencilPass));
        DkCompareOp func = Cmp(Simple(s, kMStencilFunc));
        ds.stencilFrontFailOp = ds.stencilBackFailOp = fail;
        ds.stencilFrontDepthFailOp = ds.stencilBackDepthFailOp = zfail;
        ds.stencilFrontPassOp = ds.stencilBackPassOp = pass;
        ds.stencilFrontCompareOp = ds.stencilBackCompareOp = func;
        std::array<uint32_t, 3> stencil{uint8_t(Simple(s, kMStencilWriteMask)), uint8_t(Simple(s, kMStencilRef)),
                                        uint8_t(Simple(s, kMStencilMask))};
        if (Changed(stencil_cache_, stencil))
            dkCmdBufSetStencil(cmd_, DkFace_FrontAndBack, stencil[0], stencil[1], stencil[2]);
    }
    if (Changed(depth_cache_, ds)) dkCmdBufBindDepthStencilState(cmd_, &ds);

    // rasterizer: polygon offset in depth buffer units (24 bit), slope per pixel
    DkRasterizerState rs;
    dkRasterizerStateDefaults(&rs);
    uint32_t cull = s.rs[kRsCullMode];
    rs.cullMode = cull != 0 && a.topology == Topology::Triangles ? DkFace_Back : DkFace_None;
    rs.frontFace = cull == kCullCw ? DkFrontFace_CCW : DkFrontFace_CW;
    rs.polygonModeFront = rs.polygonModeBack = s.rs[kRsFillMode] == kFillWireframe ? DkPolygonMode_Line : DkPolygonMode_Fill;
    rs.provokingVertex = DkProvokingVertex_First;
    // without a depth test the console draws whatever depth a vertex has; the host would clip it at the near and far
    // planes (Burnout 3's full screen passes put the scene on the back buffer from outside them)
    rs.depthClampEnable = !s.rs[kRsZEnable];
    if (has_ds && a.topology == Topology::Triangles && Simple(s, kMSolidOffsetEnable)) {
        rs.depthBiasEnableMask = DkPolygonFlag_All;
        // the slope is per pixel, and an upscaled target's pixels are smaller
        std::array<float, 3> bias{std::round(Bits(Simple(s, kMDepthOffset)) * kDepth24Max / s.zscale), 0.0f,
            Bits(Simple(s, kMSlopeScale)) * std::max(float(AW(*bound_ds_)) / float(bound_ds_->gw), float(AH(*bound_ds_)) / float(bound_ds_->gh))};
        if (Changed(depth_bias_cache_, bias)) dkCmdBufSetDepthBias(cmd_, bias[0], bias[1], bias[2]);
    }
    if (Changed(raster_cache_, rs)) dkCmdBufBindRasterizerState(cmd_, &rs);
}

void DekoBackend::SetConstants(const State& s, const DrawShaders& sh, const Stage& st, uint32_t gw, uint32_t gh, bool points) {
    // the vertex constants and the draw constants go up only when they differ from the last draw's this frame
    static_assert(sizeof(last_vc_) == sizeof(s.vsc));
    PushChanged(vc_ubo_, last_vc_.data(), s.vsc.data(), sizeof(s.vsc));
    DrawConstants c{};
    c.surf[0] = float(gw);
    c.surf[1] = float(gh);
    c.surf[2] = 1.0f / s.zscale;
    c.surf[3] = points ? 1.0f : 0.0f;
    Color(s.rs[kRsFogColor], c.fog_color);
    c.misc[0] = float(Simple(s, kMAlphaRef) & 0xFF);
    c.misc[1] = Simple(s, kMAlphaTestEnable) ? float(Cmp(Simple(s, kMAlphaFunc)) - 1) : 7.0f;
    c.misc[2] = float(Cmp(s.rs[kRsShadowFunc]) - 1);
    c.misc[3] = s.rs[kRsSpecularEnable] ? 1.0f : 0.0f;
    // fog: table mode none under a vertex program makes oFog the blend factor, linear from 1 to 0
    if (s.rs[kRsFogEnable]) {
        if (sh.prog && s.rs[kRsFogTableMode] == 0) {
            c.fog[0] = 1.0f, c.fog[1] = 0.0f, c.fog[2] = 0.0f, c.fog[3] = 3.0f;
        } else {
            c.fog[0] = Bits(s.rs[kRsFogStart]);
            c.fog[1] = Bits(s.rs[kRsFogEnd]);
            c.fog[2] = Bits(s.rs[kRsFogDensity]);
            c.fog[3] = float(s.rs[kRsFogTableMode]);
        }
    }
    for (int i = 0; i < 8; i++) {
        Color(Simple(s, kMPsConstant0 + i * 4), c.pc[i * 2]);
        Color(Simple(s, kMPsConstant1 + i * 4), c.pc[i * 2 + 1]);
    }
    Color(Simple(s, kMPsFinalConstant0), c.pc[16]);
    Color(Simple(s, kMPsFinalConstant1), c.pc[17]);
    Color(s.rs[kRsTextureFactor], c.tfactor);
    for (int i = 0; i < 4; i++) {
        if (st.has[i]) {
            const TextureDesc& d = st.desc[i];
            c.texsize[i][0] = float(d.width);
            c.texsize[i][1] = float(d.height);
            // the console samples pixels at integer positions, Maxwell half a pixel on: targets are read half a texel back
            // coordinates over the drawn part of a target (bumplum.zw multiplies them, nv2a_glsl)
            c.bumpl[i][2] = st.fx[i];
            c.bumpl[i][3] = st.fy[i];
            if (st.rt[i]) {
                c.texsize[i][2] = -0.5f / (st.hw[i] ? st.hw[i] : float(d.width) * st.kx[i]);
                c.texsize[i][3] = -0.5f / (st.hh[i] ? st.hh[i] : float(d.height) * st.ky[i]);
            }
            // a 3 console texel box per tap keeps the console chain's glow width and brightness; a blur's taps are
            // 7/6 texel apart, so each covers that much, a sample per host texel
            if (st.wide[i] > 0) {
                constexpr float kGlowBoxTexels = 3.0f, kBlurTapTexels = 7.0f / 6;
                float fw = st.blur[i] ? kBlurTapTexels : kGlowBoxTexels;
                float n = std::ceil(fw * st.wide[i] * (st.blur[i] ? 1.0f : 0.5f));
                // B3_WIDE_AB=1 (measuring): a single tap every other 10 s, the filter's cost in one run
                static const bool wide_ab = [] { const char* e = std::getenv("B3_WIDE_AB"); return e && *e == '1'; }();
                if (wide_ab && frame_ / 600 % 2) n = 1;
                c.spread[i][0] = fw / (n * float(d.width)) * st.fx[i];
                c.spread[i][1] = fw / (n * float(d.height)) * st.fy[i];
                c.spread[i][2] = n;
            }
        }
        const auto& t = s.tss[size_t(i)];
        c.bump[i][0] = Bits(t[kTssBumpEnvMat00]);
        c.bump[i][1] = Bits(t[kTssBumpEnvMat01]);
        c.bump[i][2] = Bits(t[kTssBumpEnvMat10]);
        c.bump[i][3] = Bits(t[kTssBumpEnvMat11]);
        if (!st.has[i]) c.bumpl[i][2] = c.bumpl[i][3] = 1.0f;
        c.bumpl[i][0] = Bits(t[kTssBumpEnvLScale]);
        c.bumpl[i][1] = Bits(t[kTssBumpEnvLOffset]);
    }
    if (st.align_glow) AlignGlow(c.texsize[0], st.fx[0], st.fy[0], guest_w_, guest_h_);
    if (sh.fixed_transform) {
        float wv[16], wvp[16], full[16];
        MatMul(s.transforms[kTsWorld].data(), s.transforms[kTsView].data(), wv);
        MatMul(wv, s.transforms[kTsProjection].data(), wvp);
        const auto& v = s.viewport;
        float sx = float(v.w) / gw, sy = float(v.h) / gh;
        float ox = (2.0f * v.x + v.w) / gw - 1, oy = 1 - (2.0f * v.y + v.h) / gh;
        float vpm[16] = {sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, v.max_z - v.min_z, 0, ox, oy, v.min_z, 1};
        MatMul(wvp, vpm, full);
        Transpose(full, c.wvp);
        Transpose(wv, c.wv);
        float n[16] = {};
        InverseTranspose3x3(wv, n);
        Transpose(n, c.wvn);
        for (int t = 0; t < 4; t++) Transpose(s.transforms[kTsTexture0 + size_t(t)].data(), c.tm[t]);
        std::memcpy(c.mat, s.material.data(), sizeof(float) * 16);
        c.mat[4][0] = s.material[16];
        Color(s.rs[kRsAmbient], c.amb);
        const float* vm = s.transforms[kTsView].data();
        for (int i = 0; i < 8; i++)
            if ((sh.fk.types >> (i * 2)) & 3) LightToView(s.lights[size_t(i)], vm, c.lt[i]);
    }
    PushChanged(dc_ubo_, &last_dc_, &c, sizeof(c));
    consts_valid_ = true;
}

// the 16 byte registers of now that differ from shadow, in runs of at most 1 KiB, into ubo; shadow then holds now
void DekoBackend::PushChanged(const Mem& ubo, void* shadow, const void* now, uint32_t bytes) {
    auto* sh = static_cast<uint8_t*>(shadow);
    auto* nw = static_cast<const uint8_t*>(now);
    dk::ChangedConstantRuns(sh, nw, bytes, consts_valid_, [&](uint32_t off, uint32_t n) {
        dkCmdBufPushConstants(cmd_, ubo.gpu, ubo.size, off, n, nw + off);
        std::memcpy(sh + off, nw + off, n);
        perf_.upload += n;
    });
}

// the viewport covers the target, the guest viewport is in the shader; the scissor clips to it
void DekoBackend::SetViewportScissor(const State& s, uint32_t gw) {
    Target* t = bound_rt_ ? bound_rt_ : bound_ds_;
    float W = float(AW(*t)), H = float(AH(*t));
    DkViewport vp{0, 0, W, H, 0, 1};
    if (Changed(viewport_cache_, vp)) dkCmdBufSetViewports(cmd_, 0, &vp, 1);
    // guest pixels of the bound target to its host pixels
    float scx = W / float(t->gw), scy = H / float(t->gh);
    (void)gw;
    int x0 = int(s.viewport.x * scx), y0 = int(s.viewport.y * scy);
    int x1 = int((s.viewport.x + s.viewport.w) * scx), y1 = int((s.viewport.y + s.viewport.h) * scy);
    if (s.scissor_count) {
        x0 = std::max(x0, int(s.scissors[0] * scx));
        y0 = std::max(y0, int(s.scissors[1] * scy));
        x1 = std::min(x1, int(s.scissors[2] * scx));
        y1 = std::min(y1, int(s.scissors[3] * scy));
    }
    x0 = std::clamp(x0, 0, int(W)), x1 = std::clamp(x1, 0, int(W));
    y0 = std::clamp(y0, 0, int(H)), y1 = std::clamp(y1, 0, int(H));
    DkScissor sc_rect{uint32_t(x0), uint32_t(y0), uint32_t(std::max(x1 - x0, 0)), uint32_t(std::max(y1 - y0, 0))};
    if (Changed(scissor_cache_, sc_rect)) dkCmdBufSetScissors(cmd_, 0, &sc_rect, 1);
}

constexpr uint64_t kZcStableFrames = 3;
constexpr uint32_t kZcMinBytes = 512;  // smaller ranges are cheaper to copy than to track

// b drawn in place: its GPU address in guest RAM; false: copy it
bool DekoBackend::InPlace(const VertexBuffer& b, DkGpuAddr& gpu) {
    Census& c = census_;
    if (!b.va || b.bytes < kZcMinBytes || b.src < guest_ptr_ || b.src + b.bytes > guest_ptr_ + guest_size_) {
        c.zc_not_guest++;
        return false;
    }
    uint64_t off = uint64_t(b.src - guest_ptr_);
    if ((off & 3) || (b.stride & 3)) {
        c.zc_misaligned++;
        return false;
    }
    // sixteen 32 byte samples of the range with the lock epoch of its buffer: a rewrite shows in either
    uint64_t sig = 0x9E3779B97F4A7C15ull ^ b.bytes;
    for (uint32_t k = 0; k < 16; k++) sig = Sig(b.src + uint64_t(b.bytes - 32) * k / 15, 32, sig);
    uint64_t epoch = DirtyEpoch(b.va);
    ZcRange& z = zc_[off << 24 ^ b.bytes];
    if (z.seen != frame_ + 1) {
        z.seen = frame_ + 1;
        if (z.sig != sig || z.epoch != epoch || !z.since) {
            z.sig = sig, z.epoch = epoch, z.since = frame_ + 1;
        }
    } else if (z.sig != sig || z.epoch != epoch) {
        z.sig = sig, z.epoch = epoch, z.since = frame_ + 1;  // rewritten within the frame
    }
    if (frame_ + 1 - z.since < kZcStableFrames) {
        c.zc_dynamic++;
        return false;
    }
    // every frame that draws it, the CPU's cached bytes out to memory before the GPU reads them (the title may rewrite
    // it any time, through a NOOVERWRITE lock or a kept pointer, which no epoch shows), and once a frame the GPU's L2
    // told to read memory again; the title's fences and busy checks wait for the GPU (render_thread.cpp)
    if (z.cleaned != frame_ + 1) {
        dkMemBlockFlushCpuCache(guest_mem_, uint32_t(off), b.bytes);
        z.cleaned = frame_ + 1;
        c.zc_cleaned += b.bytes;
    }
    if (zc_l2_frame_ != frame_ + 1) {
        zc_l2_frame_ = frame_ + 1;
        zc_l2_ = true;
    }
    gpu = guest_gpu_ + off;
    in_place_va_[InPlaceSlot(b.va)].store(frame_ + 1, std::memory_order_release);
    in_place_any_[frame_ % 8].store(frame_ + 1, std::memory_order_release);
    return true;
}

void DekoBackend::DrawGeometry(const Assembled& a) {
    // attributes at their register's location; the registers between take a fixed value
    Attributes attributes{};
    auto* attribs = attributes.data.data();
    uint32_t n = 0;
    for (uint32_t r = 0; r < 16; r++) {
        if (!((a.regs >> r) & 1)) continue;
        while (n < r) {
            attribs[n] = DkVtxAttribState{};
            attribs[n].isFixed = 1;
            attribs[n].size = DkVtxAttribSize_1x32;
            attribs[n].type = DkVtxAttribType_Float;
            n++;
        }
        const VertexElement& e = a.elements[r];
        DkVtxAttribState at{};
        at.bufferId = e.buffer;
        at.offset = e.offset;
        if (!VertexFormat(e.format, at)) {
            LOG_FIRST(8, "deko3d: vertex format %u not mapped", e.format);
            census_.drop_vfmt++;
            return;
        }
        attribs[n++] = at;
    }
    attributes.count = n;
    if (Changed(attrib_cache_, attributes)) dkCmdBufBindVtxAttribState(cmd_, attribs, n);
    Buffers buffers{};
    auto* bufs = buffers.data.data();
    for (uint32_t i = 0; i < a.buffer_count; i++) bufs[i] = {a.buffers[i].stride, 0};
    buffers.count = a.buffer_count;
    if (Changed(buffer_cache_, buffers)) dkCmdBufBindVtxBufferState(cmd_, bufs, a.buffer_count);
    // each buffer bound alone: a multi-buffer bind can write past its command chunk (deko9_vtxbind.h)
    zc_l2_ = false;
    DkGpuAddr place[16];
    bool in_place[16];
    for (uint32_t i = 0; i < a.buffer_count; i++) in_place[i] = guest_mem_ && InPlace(a.buffers[i], place[i]);
    // an element at an offset Maxwell may not fetch from in place sends its buffer through the copy
    for (uint32_t r = 0; r < 16; r++)
        if ((a.regs >> r) & 1 && (a.elements[r].offset & 3)) in_place[a.elements[r].buffer] = false;
    if (zc_l2_) dkCmdBufBarrier(cmd_, DkBarrier_None, DkInvalidateFlags_L2Cache);
    for (uint32_t i = 0; i < a.buffer_count; i++) {
        const VertexBuffer& b = a.buffers[i];
        if (in_place[i]) {
            census_.zc_bytes += b.bytes;
            dkCmdBufBindVtxBuffer(cmd_, i, place[i], b.bytes);
            continue;
        }
        if (guest_mem_) census_.zc_copied += b.bytes;
        Span sp = Upload(b.bytes, 16);
        if (!sp.cpu) return;
        std::memcpy(sp.cpu, b.src, b.bytes);
        dkCmdBufBindVtxBuffer(cmd_, i, sp.gpu, b.bytes);
    }
    DkPrimitive prim = a.topology == Topology::Points  ? DkPrimitive_Points
                       : a.topology == Topology::Lines ? (a.strip ? DkPrimitive_LineStrip : DkPrimitive_Lines)
                                                       : (a.strip ? DkPrimitive_TriangleStrip : DkPrimitive_Triangles);
    if (a.index_count) {
        uint32_t ibytes = a.index_count * (a.index16 ? 2 : 4);
        Span ib = Upload(ibytes, 4);
        if (!ib.cpu) return;
        std::memcpy(ib.cpu, a.index_src, ibytes);
        dkCmdBufBindIdxBuffer(cmd_, a.index16 ? DkIdxFormat_Uint16 : DkIdxFormat_Uint32, ib.gpu);
        dkCmdBufDrawIndexed(cmd_, prim, a.index_count, 1, 0, a.base_vertex, 0);
    } else {
        dkCmdBufDraw(cmd_, prim, a.draw_count, 1, 0, 0);
    }
    list_work_ = true;
}

// The Xbox also copies depth through colour aliases. Keep the depth image current,
// as D3D12Backend::DepthCopy does, rather than drawing into a separate colour image.
bool DekoBackend::DepthCopy(const State& s) {
    TextureDesc d;
    if (!s.rt.header || !s.textures[0] || !Describe(s.textures[0], d) || Format(d.format).depth || d.cube || d.depth != 1)
        return false;
    uint32_t w, h;
    SurfaceSize(s.rt, w, h);
    auto di = targets_.find(SizedTargetKey(mem::VirtualFromPhysical(s.rt.data), true, w, h));
    auto si = targets_.find(SizedTargetKey(d.data, true, d.width, d.height));
    if (di == targets_.end() || si == targets_.end() || di == si) return false;
    Target& dst = di->second;
    Target& src = si->second;
    const DkShader* ps = shaders_.Get(ShaderCache::kFragment, kDepthCopyPs);
    if (!present_vs_ || !ps) return false;
    ++draw_epoch_;  // depth-copy pass overwrites shaders, textures, layouts and pipeline state
    Hazard(src.img, Access::Sample);
    BindRenderTargets(nullptr, &dst);
    const DkShader* stages[] = {present_vs_, ps};
    dkCmdBufBindShaders(cmd_, DkStageFlag_Vertex | DkStageFlag_Fragment, stages, 2);
    DkSampler sampler;
    dkSamplerDefaults(&sampler);
    sampler.wrapMode[0] = sampler.wrapMode[1] = sampler.wrapMode[2] = DkWrapMode_ClampToEdge;
    DkResHandle handle = dkMakeTextureHandle(src.img.desc, SamplerSlot(sampler));
    if (descs_dirty_) {
        dkCmdBufBarrier(cmd_, DkBarrier_None, DkInvalidateFlags_Descriptors);
        descs_dirty_ = false;
    }
    dkCmdBufBindTextures(cmd_, DkStage_Fragment, 0, &handle, 1);
    DkDepthStencilState ds;
    dkDepthStencilStateDefaults(&ds);
    ds.depthTestEnable = ds.depthWriteEnable = true;
    ds.depthCompareOp = DkCompareOp_Always;
    ds.stencilTestEnable = false;
    dkCmdBufBindDepthStencilState(cmd_, &ds);
    DkRasterizerState rs;
    dkRasterizerStateDefaults(&rs);
    rs.cullMode = DkFace_None;
    dkCmdBufBindRasterizerState(cmd_, &rs);
    float part[4] = {float(AW(src)) / src.img.w, float(AH(src)) / src.img.h, 0, 0};
    dkCmdBufPushConstants(cmd_, present_ubo_.gpu, present_ubo_.size, 0, sizeof(part), part);
    dkCmdBufBindUniformBuffer(cmd_, DkStage_Vertex, 0, present_ubo_.gpu, present_ubo_.size);
    dkCmdBufBindVtxAttribState(cmd_, nullptr, 0);
    dkCmdBufBindVtxBufferState(cmd_, nullptr, 0);
    DkViewport vp{0, 0, float(AW(dst)), float(AH(dst)), 0, 1};
    DkScissor sc{0, 0, AW(dst), AH(dst)};
    dkCmdBufSetViewports(cmd_, 0, &vp, 1);
    dkCmdBufSetScissors(cmd_, 0, &sc, 1);
    dkCmdBufDraw(cmd_, DkPrimitive_Triangles, 3, 1, 0, 0);
    // zcull cannot follow depth a fragment shader writes: later draws testing against it must not cull on stale data
    dkCmdBufBarrier(cmd_, DkBarrier_None, DkInvalidateFlags_Zcull);
    dkCmdBufBindUniformBuffer(cmd_, DkStage_Vertex, 0, vc_ubo_.gpu, vc_ubo_.size);
    dst.drawn = true;
    list_work_ = true;
    LOG_FIRST(4, "deko3d: depth copied %08X to %08X", src.data, dst.data);
    return true;
}

void DekoBackend::Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    auto t0 = Clock::now();
    perf_.draws++;
    unsubmitted_draws_++;
    // diagnostics: every draw of one frame in 600 with what became of it, the passes' order on the console
    const char* fate = "drawn";
    struct Trace {
        DekoBackend& b;
        const State& s;
        uint32_t prim, count;
        const char*& fate;
        ~Trace() {
            if (!b.diag_ || b.frame_ % 600 != 2) return;
            uint32_t tex[4] = {};
            for (int i = 0; i < 4; i++) {
                TextureDesc d;
                if (s.textures[size_t(i)] && Describe(s.textures[size_t(i)], d)) tex[i] = d.data;
            }
            LOGI("deko3d: trace rt %08X ds %08X prim %u count %u vs %08X z %u tex %08X %08X %08X %08X: %s",
                 s.rt.header ? mem::VirtualFromPhysical(s.rt.data) : 0, s.zs.header ? mem::VirtualFromPhysical(s.zs.data) : 0,
                 prim, count, s.vs_handle, s.rs[kRsZEnable], tex[0], tex[1], tex[2], tex[3], fate);
        }
    } trace{*this, s, prim, count, fate};
    if (DepthCopy(s)) {
        fate = "depth copy";
        return;
    }
    Assembled& a = asm_;
    bool assembled = Assemble(s, prim, start, count, indices, a);
    perf_.assemble += Ms(t0);
    if (!assembled) {
        census_.drop_asm++;
        fate = "not assembled";
        return;
    }
    if (!BindTargets(s)) {
        census_.drop_target++;
        fate = "no target";
        return;
    }

    DrawShaders sh;
    sh.prog = (s.vs_handle & 1) && s.vs && !s.vs->program.empty();
    bool points = a.topology == Topology::Points;
    sh.quad = count <= kQuadVertices && !s.rs[kRsZEnable] && !points;
    draw_quad_ = sh.quad;
    sh.fixed_transform = !sh.prog && !a.pretransformed;
    if (!sh.prog) sh.fk = MakeFixedKey(s);
    if (sh.prog && !s.vs->hash) s.vs->hash = Hash(s.vs->program.data(), s.vs->program.size() * 4) | 1;
    uint64_t seed = sh.prog ? s.vs->hash ^ a.regs : Hash(&s.vs_handle, 4, a.regs | kDeclarationSalt);
    sh.vkey = Hash(&a.decode, 8, seed);
    if (!sh.prog && a.pretransformed) sh.vkey ^= kPretransformedSalt;
    if (sh.quad) sh.vkey ^= kQuadSalt;
    if (!sh.prog) sh.vkey = Hash(&sh.fk, sizeof(sh.fk), sh.vkey);
    EnsureShaders(s, a, sh);

    Stage st;
    auto tb = Clock::now();
    for (int i = 0; i < 4; i++) BindTexture(s, i, st);
    perf_.bind += Ms(tb);
    // an upload over the bound target (UploadTarget) unbinds it; a feedback copy re-arms the render hazard itself.
    // Otherwise the binding stands and the second BindTargets (four map lookups and a change check per draw) is skipped
    if (!bound_rt_ && !bound_ds_ && !BindTargets(s)) {
        fate = "target lost";
        return;
    }
    PixelKey pk = MakePixelKey(s, st.info);
    st.align_glow = IsGlowComposite(s, pk, st.desc[0], prim, count, bound_rt_ ? bound_rt_->gw : 0,
        bound_rt_ ? bound_rt_->gh : 0, guest_w_, guest_h_, st.rt[0]);
    pk.quad = sh.quad;
    bool atest = Simple(s, kMAlphaTestEnable) != 0;
    sh.pkey = Hash(&pk, sizeof(pk)) ^ (atest ? 0 : 0xA1FA7E57ull);
    if (last_ps_ && last_ps_key_ == sh.pkey) {
        sh.ps = last_ps_;
    } else {
        auto pit = pshaders_.find(sh.pkey);
        if (pit != pshaders_.end()) {
            sh.ps = pit->second;
        } else {
            auto t = Clock::now();
            sh.ps = shaders_.Get(ShaderCache::kFragment, dk::PixelGlsl(pk, atest));
            pshaders_[sh.pkey] = sh.ps;
            perf_.compile += Ms(t);
            perf_.compiles++;
        }
        last_ps_key_ = sh.pkey;
        last_ps_ = sh.ps;
    }
    if (!sh.vs || !sh.ps) {
        census_.drop_shader++;
        fate = !sh.vs ? "no vertex shader" : "no pixel shader";
        return;
    }
    if (diag_) CountDraw(s, a, pk, st);
    // every ten seconds, the draws of one frame outside the scene pass, described
    if (draw_dump_ && frame_ % 600 == 1 && bound_rt_ && (!bound_ds_ || bound_rt_->gw != guest_w_)) {
        char tx[384];
        size_t o = 0;
        for (int i = 0; i < 4 && o < sizeof(tx) - 96; i++) {
            if (!st.has[i]) continue;
            const TextureDesc& d = st.desc[i];
            const auto& t = s.tss[size_t(i)];
            o += std::snprintf(tx + o, sizeof(tx) - o, " t%d %08X %ux%u f%02X%s%s k%.2f mode %u filt %u/%u/%u addr %u/%u", i, d.data, d.width,
                               d.height, d.format, st.rt[i] ? " rt" : "", st.img[i] && bound_rt_->fb && st.img[i] == bound_rt_->fb.get() ? " fb" : "",
                               st.kx[i], pk.programmable ? (pk.tex_modes >> (i * 5)) & 0x1F : pk.ff[i][0], t[kTssMinFilter], t[kTssMagFilter],
                               t[kTssMipFilter], t[kTssAddressU], t[kTssAddressV]);
        }
        LOGI("deko3d: draw into %ux%u%s%s prim %u count %u verts %u idx %u quad %d blend %u %X/%X ztest %u prog %u combiners %X final %X/%X%s",
             bound_rt_->gw, bound_rt_->gh, bound_rt_->img.w != bound_rt_->gw ? "*" : "", bound_ds_ ? "+z" : "", prim, count, a.vertex_count,
             a.index_count, int(sh.quad), Simple(s, kMBlendEnable), Simple(s, kMSrcBlend), Simple(s, kMDestBlend), s.rs[kRsZEnable],
             pk.programmable, pk.count, pk.final_abcd, pk.final_efg, tx);
    }
    if (diag_ && !Cur().mark_draws.empty()) Cur().mark_draws.back() += Simple(s, kMBlendEnable) ? 0x10001 : 1;
    std::array<const DkShader*, 2> shaders{sh.vs, sh.ps};
    if (Changed(shader_cache_, shaders))
        dkCmdBufBindShaders(cmd_, DkStageFlag_Vertex | DkStageFlag_Fragment, shaders.data(), 2);

    uint32_t gw, gh;
    SurfaceSize(bound_rt_ ? s.rt : s.zs, gw, gh);
    SetPipelineState(s, a);
    auto tc = Clock::now();
    SetConstants(s, sh, st, gw, gh, points);
    perf_.consts += Ms(tc);
    SetSamplersAndTextures(s, a, st);
    SetViewportScissor(s, gw);
    bool counting = s.simple[kMCountEnable >> 2] != 0;
    uint32_t before = 0;
    if (counting) {
        before = report_next_++ % kReportSlots;
        dkCmdBufReportCounter(cmd_, DkCounter_SamplesPassed, reports_mem_.gpu + before * 16);
    }
    auto tg = Clock::now();
    DrawGeometry(a);
    perf_.geometry += Ms(tg);
    if (counting) {
        uint32_t after = report_next_++ % kReportSlots;
        dkCmdBufReportCounter(cmd_, DkCounter_SamplesPassed, reports_mem_.gpu + after * 16);
        Target* ct = bound_rt_ ? bound_rt_ : bound_ds_;
        counted_.push_back({before, after, float(AW(*ct)) * AH(*ct) / (float(ct->gw) * ct->gh)});
    }
    perf_.draw += Ms(t0);
}

// reports whose frame the GPU has finished, written as the console's GPU writes them: time, count, then 0 for done
void DekoBackend::DeliverReports() {
    uint64_t done = done_through_.load();
    size_t kept = 0;
    for (size_t i = 0; i < reports_.size(); i++) {
        Report& r = reports_[i];
        if (r.frame >= done) {
            reports_[kept++] = std::move(r);
            continue;
        }
        uint64_t pixels = 0;
        for (const Counted& c : r.counted) {
            uint64_t a, b;
            std::memcpy(&a, reports_mem_.cpu + c.before * 16, 8);
            std::memcpy(&b, reports_mem_.cpu + c.after * 16, 8);
            pixels += uint64_t(double(b - a) / c.area + 0.5);
        }
        kernel::W32(r.va + 0, uint32_t(frame_));
        kernel::W32(r.va + 4, 0);
        // host samples to guest pixels
        kernel::W32(r.va + 8, uint32_t(pixels));
        std::atomic_thread_fence(std::memory_order_release);
        kernel::W32(r.va + 12, 0);
    }
    reports_.resize(kept);
}

// --- clear, copy ---

void DekoBackend::Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
                        uint32_t stencil) {
    if (flags & kClearOcclusion) return;  // ambient occlusion: the console picture has none
    Target* rt = (flags & kClearTarget) ? EnsureTarget(s.rt, false) : nullptr;
    Target* ds = (flags & (kClearZ | kClearStencil)) ? EnsureTarget(s.zs, true) : nullptr;
    if (rt && ds && (rt->gw != ds->gw || rt->gh != ds->gh)) ds = nullptr;
    if (!rt && !ds) return;
    ++draw_epoch_;  // clears overwrite the scissor
    // pixels the CPU wrote before this clear go in first, a later upload would paint over the clear
    if (rt && LatchCpuWrite(*rt)) UploadTarget(*rt, s.rt);
    BindRenderTargets(rt, ds);
    Target* t = rt ? rt : ds;
    std::vector<DkScissor> rs;
    int W = int(AW(*t)), H = int(AH(*t));
    float scx = float(W) / float(t->gw), scy = float(H) / float(t->gh);
    auto add = [&](int x0, int y0, int x1, int y1) {
        x0 = std::clamp(int(x0 * scx), 0, W), x1 = std::clamp(int(x1 * scx), 0, W);
        y0 = std::clamp(int(y0 * scy), 0, H), y1 = std::clamp(int(y1 * scy), 0, H);
        if (x1 > x0 && y1 > y0) rs.push_back({uint32_t(x0), uint32_t(y0), uint32_t(x1 - x0), uint32_t(y1 - y0)});
    };
    if (count && rects) {
        for (uint32_t i = 0; i < count; i++) {
            int32_t r[4];
            std::memcpy(r, rects + size_t(i) * sizeof(r), sizeof(r));
            add(r[0], r[1], r[2], r[3]);
        }
    } else {
        const auto& v = s.viewport;
        add(int(v.x), int(v.y), int(v.x + v.w), int(v.y + v.h));
    }
    float col[4];
    Color(color, col);
    // D3DCLEAR_TARGET_R/G/B/A (0x10 to 0x80): only the channels asked for, as the console clears
    const uint32_t mask = (flags & 0x10 ? DkColorMask_R : 0) | (flags & 0x20 ? DkColorMask_G : 0) |
                          (flags & 0x40 ? DkColorMask_B : 0) | (flags & 0x80 ? DkColorMask_A : 0);
    for (const DkScissor& sc : rs) {
        dkCmdBufSetScissors(cmd_, 0, &sc, 1);
        if (rt) dkCmdBufClearColor(cmd_, 0, mask, col);
        if (ds) dkCmdBufClearDepthStencil(cmd_, (flags & kClearZ) != 0, std::clamp(z, 0.0f, 1.0f),
                                          (flags & kClearStencil) ? 0xFF : 0, uint8_t(stencil));
    }
    if (rt) rt->drawn = true;
    if (ds) ds->drawn = true;
    list_work_ = true;
}

void DekoBackend::Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects, uint32_t dst_points) {
    Target* src_t = FindTarget(src, false);
    // pixels the CPU made: the copy happens in memory, textures and targets pick it up from there
    if (!src_t || !src_t->drawn) {
        if (!CopySurfaceMemory(src, dst, count, src_rects, dst_points)) {
            LOG_FIRST(8, "deko3d: CopyRects %08X to %08X skipped", src.data, dst.data);
        } else if (Target* dst_t = FindTarget(dst, false)) {
            UploadTarget(*dst_t, dst);
        }
        return;
    }
    Target* dst_t = EnsureTarget(dst, false);
    if (!dst_t || src_t == dst_t) return;
    Hazard(src_t->img, Access::CopyRead);
    Hazard(dst_t->img, Access::CopyWrite);
    if (bound_rt_ == dst_t) bound_rt_ = bound_ds_ = nullptr;
    Drawn(*dst_t);
    DkImageView sv, dv;
    ImageView(src_t->img, sv);
    ImageView(dst_t->img, dv);
    bool listed = count && src_rects;
    uint32_t n = listed ? count : 1;
    for (uint32_t i = 0; i < n; i++) {
        int32_t l = 0, t = 0, r = int32_t(std::min(src_t->gw, dst_t->gw)), b = int32_t(std::min(src_t->gh, dst_t->gh));
        int32_t dx = 0, dy = 0;
        if (listed) {
            uint32_t p = src_rects + i * 16;
            l = int32_t(kernel::R32(p)), t = int32_t(kernel::R32(p + 4)), r = int32_t(kernel::R32(p + 8)), b = int32_t(kernel::R32(p + 12));
            dx = l, dy = t;
            if (dst_points) dx = int32_t(kernel::R32(dst_points + i * 8)), dy = int32_t(kernel::R32(dst_points + i * 8 + 4));
        }
        l = std::max(l, 0), t = std::max(t, 0);
        r = std::min(r, int32_t(src_t->gw)), b = std::min(b, int32_t(src_t->gh));
        if (r <= l || b <= t || dx < 0 || dy < 0 || dx >= int32_t(dst_t->gw) || dy >= int32_t(dst_t->gh)) continue;
        uint32_t w = std::min<uint32_t>(r - l, dst_t->gw - dx), h = std::min<uint32_t>(b - t, dst_t->gh - dy);
        // guest pixels to each target's host pixels
        auto hx = [this](const Target& g, uint32_t v) { return uint32_t(uint64_t(v) * AW(g) / g.gw); };
        auto hy = [this](const Target& g, uint32_t v) { return uint32_t(uint64_t(v) * AH(g) / g.gh); };
        uint32_t sl = hx(*src_t, l), st = hy(*src_t, t), dl = hx(*dst_t, dx), dt = hy(*dst_t, dy);
        DkImageRect sr{sl, st, 0, hx(*src_t, l + w) - sl, hy(*src_t, t + h) - st, 1};
        DkImageRect dr{dl, dt, 0, hx(*dst_t, dx + w) - dl, hy(*dst_t, dy + h) - dt, 1};
        if (!sr.width || !sr.height || !dr.width || !dr.height) continue;
        // the 2D engine either way, the copy engine is slow on block linear images
        bool same = sr.width == dr.width && sr.height == dr.height;
        dkCmdBufBlitImage(cmd_, &sv, &sr, &dv, &dr, same ? DkBlitFlag_FilterNearest : DkBlitFlag_FilterLinear, 0);
    }
    dst_t->drawn = true;
    list_work_ = true;
}

// --- read backs ---

// the CPU reads a surface the GPU drew: bring the pixels back
void DekoBackend::Sync(uint32_t data, uint32_t) {
    auto it = targets_.find(TargetMapKey(data, false));
    if (it == targets_.end() || !it->second.drawn || TargetStale(data)) return;
    auto t0 = Clock::now();
    perf_.syncs++;
    LOG_FIRST(8, "deko3d: read back of %08X (%ux%u)", data, it->second.gw, it->second.gh);
    Target& t = it->second;
    std::vector<uint8_t> px;
    if (!ReadTarget(t, px)) return;
    // colour targets are 32 bit linear in guest memory, the pitch rounds to 64 bytes
    uint32_t pitch = GuestPitch(t.gw);
    uint8_t* guest = static_cast<uint8_t*>(kernel::HostPtr(data));
    for (uint32_t y = 0; y < t.gh; y++) std::memcpy(guest + size_t(y) * pitch, &px[size_t(y) * t.gw * 4], t.gw * 4);
    t.read = std::move(px);
    t.drawn = false;
    t.sig = MemSig(t.data, t.bytes);
    perf_.sync += Ms(t0);
}

// A NOOVERWRITE lock: the newest copy the GPU has finished (a frame or two old) goes to guest memory and a new copy is
// recorded for a later lock, so the CPU never waits; the first lock of a surface waits in Sync
void DekoBackend::SyncLate(uint32_t data) {
    if (!DeliverLate(data)) Sync(data, 0);
    RecordLate(data);
}

// the newest finished copy into guest memory; any thread (the title's, with the render thread)
bool DekoBackend::DeliverLate(uint32_t data) {
    std::lock_guard lock(late_m_);
    auto it = late_.find(data);
    if (it == late_.end()) return false;
    Late& l = it->second;
    uint64_t done = done_through_.load();
    int best = -1;
    for (int i = 0; i < int(kFrames + 1); i++)
        if (l.buf[i] && l.frame[i] && l.frame[i] - 1 < done && (best < 0 || l.frame[i] > l.frame[best])) best = i;
    if (best < 0) return false;
    if (l.frame[best] != l.delivered) {
        armDCacheFlush(l.buf[best].cpu, l.gw * l.gh * 4);
        uint32_t pitch = GuestPitch(l.gw);
        uint8_t* guest = static_cast<uint8_t*>(kernel::HostPtr(data));
        for (uint32_t y = 0; y < l.gh; y++) std::memcpy(guest + size_t(y) * pitch, l.buf[best].cpu + size_t(y) * l.gw * 4, l.gw * 4);
        l.delivered = l.frame[best];
    }
    // the signature of what guest memory holds now, taken here on the title's thread before it writes more; RecordLate
    // gives it to the target (computed there, on the render thread, it raced the title's writes)
    l.delivered_sig = MemSig(data, GuestPitch(l.gw) * l.gh);
    l.sig_ready = true;
    return true;
}

// a copy for a later lock, in order with the draws, into a slot no frame in flight writes
void DekoBackend::RecordLate(uint32_t data) {
    auto it = targets_.find(TargetMapKey(data, false));
    if (it == targets_.end() || TargetStale(data)) return;
    Target& t = it->second;
    LOG_FIRST(8, "deko3d: late read back of %08X (%ux%u)", data, t.gw, t.gh);
    t.read.clear();
    std::lock_guard lock(late_m_);
    Late& l = late_[data];
    // what DeliverLate wrote is no CPU write to upload back (a Sync instead set t.sig itself)
    if (l.sig_ready && l.gw == t.gw && l.gh == t.gh) t.sig = l.delivered_sig;
    l.sig_ready = false;
    if (l.gw != t.gw || l.gh != t.gh) {
        for (Mem& m : l.buf)
            if (m) Cur().read_garbage.push_back(m), m = Mem{};
        l = Late{};
        l.gw = t.gw;
        l.gh = t.gh;
    }
    uint64_t done = done_through_.load();
    uint32_t slot = l.next;
    if (l.frame[slot] && l.frame[slot] - 1 >= done) return;
    if (!l.buf[slot] && !read_heap_.Alloc(t.gw * t.gh * 4, 256, l.buf[slot])) return;
    if (!ReadTargetInto(t, l.buf[slot])) return;
    l.frame[slot] = frame_ + 1;  // done once present frame_ is
    l.next = (slot + 1) % (kFrames + 1);
}

// --- present ---

void DekoBackend::WaitFrame(uint64_t n) {
    DkFence fence;
    {
        std::lock_guard lock(present_m_);
        uint32_t i = uint32_t(n % std::size(present_n_));
        // not presented yet, or so long ago that its slot holds a later present: nothing to wait for
        if (present_n_[i] != n + 1) return;
        fence = present_fence_[i];
    }
    auto t = Clock::now();
    dkFenceWait(&fence, -1);
    wait_ns_ += uint64_t(Ms(t) * 1e6);
}

bool DekoBackend::FrameDone(uint64_t n) {
    if (n < done_through_.load()) return true;
    DkFence fence;
    {
        std::lock_guard lock(present_m_);
        uint32_t i = uint32_t(n % std::size(present_n_));
        if (present_n_[i] > n + 1) return true;   // four presents later: the title waited for it long ago
        if (present_n_[i] != n + 1) return false;  // not presented yet
        fence = present_fence_[i];
    }
    return dkFenceWait(&fence, 0) == DkResult_Success;
}

// the front buffer as a TGA, top down
void DekoBackend::Screenshot(const Surface& front) {
    Target* t = FindTarget(front, false);
    if (!t) return;
    WriteTargetTga(*t, "");
    // diagnostics: every colour target too, shot_N_<data>.tga, the front buffer's own data named
    if (diag_) {
        std::vector<Target*> all;
        for (auto& [key, other] : targets_)
            if (!other.depth && other.gw >= 64) all.push_back(&other);
        LOGI("deko3d: shot front %08X", t->data);
        for (Target* o : all) {
            char suffix[16];
            std::snprintf(suffix, sizeof(suffix), "_%08X", o->data);
            WriteTargetTga(*o, suffix);
        }
    }
}

void DekoBackend::WriteTargetTga(Target& tg, const char* suffix) {
    Target* t = &tg;
    std::vector<uint8_t> px;
    if (!ReadTarget(*t, px)) return;
    char name[96];
    std::snprintf(name, sizeof(name), "sdmc:/switch/burnout3/shot_%05llu%s.tga", (unsigned long long)frame_, suffix);
    FILE* f = std::fopen(name, "wb");
    if (!f) return;
    uint8_t hd[18] = {0, 0, 2};
    hd[12] = uint8_t(t->gw), hd[13] = uint8_t(t->gw >> 8), hd[14] = uint8_t(t->gh), hd[15] = uint8_t(t->gh >> 8);
    hd[16] = 32;
    hd[17] = 0x28;  // 8 alpha bits, top left origin
    for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
    std::fwrite(hd, 1, sizeof(hd), f);
    std::fwrite(px.data(), 1, px.size(), f);
    std::fclose(f);
    LOGI("deko3d: screenshot %s", name);
}

void DekoBackend::Present(const Surface& front) {
    auto t0 = Clock::now();
    Target* t = EnsureTarget(front, false);
    if (t && LatchCpuWrite(*t)) UploadTarget(*t, front);

    // shots of the title's front buffer (shot_N.tga) and of the display picture (shot_N_display.tga):
    // sdmc:/switch/burnout3/snap asks for one of each (removed after), sdmc:/switch/burnout3/shots holding N for a front
    // buffer shot every N frames and a display shot every 4N
    bool shot = shot_every_ && frame_ % shot_every_ == 0 && frame_;
    bool shot_out = shot && frame_ % (shot_every_ * 4) == 0;
    if (Clock::now() >= shot_check_) {
        shot_check_ = Clock::now() + std::chrono::seconds(1);
        if (FILE* f = std::fopen("sdmc:/switch/burnout3/snap", "r")) {
            std::fclose(f);
            std::remove("sdmc:/switch/burnout3/snap");
            shot = shot_out = true;
        }
    }
    if (shot && t) Screenshot(front);

    // the docked display is 1080p, handheld 720p
    bool docked = appletGetOperationMode() == AppletOperationMode_Console;
    uint32_t want_w = docked ? 1920 : 1280, want_h = docked ? 1080 : 720;
    if (want_w != disp_w_) {
        Submit(true);
        dkQueueWaitIdle(queue_);
        DestroySwapchain();
        CreateSwapchain(want_w, want_h);
        LOGI("deko3d: display %ux%u", want_w, want_h);
    }

    if (ramp_dirty_) {
        ramp_dirty_ = false;
        uint8_t px[256 * 4];
        bool identity = true;
        for (int i = 0; i < 256; i++) {
            px[i * 4] = ramp_[512 + i], px[i * 4 + 1] = ramp_[256 + i], px[i * 4 + 2] = ramp_[i], px[i * 4 + 3] = 255;
            identity &= ramp_[i] == i && ramp_[256 + i] == i && ramp_[512 + i] == i;
        }
        gamma_ = !identity;
        if (gamma_) UploadPixels(lut_, 0, 0, px, sizeof(px), 256, 1, 1, sizeof(px), 1);
    }

    int slot_img = -1;
    DkFence acquire{};
    dkSwapchainAcquireImage(swap_, &slot_img, &acquire);
    dkQueueWaitFence(queue_, &acquire);
    Img& out = swap_img_[slot_img];
    // black bars, then the picture at 4:3 (16:9 when the console reports a widescreen display), as large as fits
    DkImageView ov;
    dkImageViewDefaults(&ov, &out.image);
    const DkImageView* colors[1] = {&ov};
    dkCmdBufBindRenderTargets(cmd_, colors, 1, nullptr);
    Mark(1);  // the present pass
    bound_rt_ = bound_ds_ = nullptr;
    DkScissor full{0, 0, disp_w_, disp_h_};
    dkCmdBufSetScissors(cmd_, 0, &full, 1);
    float aspect = app::Widescreen() ? 16.0f / 9.0f : 4.0f / 3.0f;
    float vw = std::min(float(disp_w_), disp_h_ * aspect), vh = vw / aspect;
    // bars only when the picture leaves some: a full display is all written by the blit
    if (!t || uint32_t(vw) < disp_w_ || uint32_t(vh) < disp_h_) dkCmdBufClearColorFloat(cmd_, 0, DkColorMask_RGBA, 0, 0, 0, 1);
    if (t) {
        uint32_t x = uint32_t((disp_w_ - vw) / 2), y = uint32_t((disp_h_ - vh) / 2);
        if (gamma_ && present_vs_ && present_ps_) {
            PresentPass(*t, x, y, uint32_t(vw), uint32_t(vh));
        } else {
            // the 2D engine reads the picture after the 3D engine finished it and cleared the bars
            Barrier();
            HazardsDone(t->img, Access::CopyRead);
            DkImageView sv;
            ImageView(t->img, sv);
            DkImageRect sr{0, 0, 0, AW(*t), AH(*t), 1}, dr{x, y, 0, uint32_t(vw), uint32_t(vh), 1};
            dkCmdBufBlitImage(cmd_, &sv, &sr, &ov, &dr, DkBlitFlag_FilterLinear, 0);
        }
    }
    Slot& slot = Cur();
    Mem out_mem;
    if (shot_out && read_heap_.Alloc(disp_w_ * disp_h_ * 4, 256, out_mem)) {
        // the finished picture, bars included, as the display gets it
        Barrier();
        DkImageRect r{0, 0, 0, disp_w_, disp_h_, 1};
        DkCopyBuf cb{out_mem.gpu, 0, 0};
        dkCmdBufCopyImageToBuffer(cmd_, &ov, &r, &cb, 0);
    }
    if (slot.spans % 2 == 1) SpanTimestamp();
    DkCmdList list = dkCmdBufFinishList(cmd_);
    dkQueueSubmitCommands(queue_, list);
    unsubmitted_draws_ = 0;
    dkQueueSignalFence(queue_, &slot.fence, true);
    slot.pending = true;
    {
        std::lock_guard lock(present_m_);
        present_fence_[frame_ % 4] = slot.fence;
        present_n_[frame_ % 4] = frame_ + 1;
    }
    dkQueuePresentImage(queue_, swap_, slot_img);
    if (out_mem) {
        dkFenceWait(&slot.fence, -1);
        armDCacheFlush(out_mem.cpu, disp_w_ * disp_h_ * 4);
        char name[96];
        std::snprintf(name, sizeof(name), "sdmc:/switch/burnout3/shot_%05llu_display.tga", (unsigned long long)frame_);
        if (FILE* f = std::fopen(name, "wb")) {
            uint8_t hd[18] = {0, 0, 2};
            hd[12] = uint8_t(disp_w_), hd[13] = uint8_t(disp_w_ >> 8), hd[14] = uint8_t(disp_h_), hd[15] = uint8_t(disp_h_ >> 8);
            hd[16] = 32;
            hd[17] = 0x28;
            std::fwrite(hd, 1, sizeof(hd), f);
            // RGBA rows to TGA's BGRA
            std::vector<uint8_t> row(disp_w_ * 4);
            for (uint32_t y = 0; y < disp_h_; y++) {
                const uint8_t* p = out_mem.cpu + size_t(y) * disp_w_ * 4;
                for (uint32_t x = 0; x < disp_w_; x++)
                    row[x * 4] = p[x * 4 + 2], row[x * 4 + 1] = p[x * 4 + 1], row[x * 4 + 2] = p[x * 4], row[x * 4 + 3] = 255;
                std::fwrite(row.data(), 1, row.size(), f);
            }
            std::fclose(f);
        }
        read_heap_.Free(out_mem);
    }
    frame_++;
    g_presented.store(frame_, std::memory_order_relaxed);
    auto now = Clock::now();
    intervals_.push_back(float(std::chrono::duration<double, std::milli>(now - last_present_t_).count()));
    last_present_t_ = now;
    harness::g_frames = frame_;  // the pad script times its presses in presented frames
    perf_.present += Ms(t0);
    LogPerf();
    BeginFrame();
    ReleaseOldTextures();
}

// the front buffer drawn over the display rect through the gamma lookup; the display image is bound
void DekoBackend::ShutdownForExit() {
    if (queue_) {
        dkQueueWaitIdle(queue_);
        if (swap_) dkSwapchainDestroy(swap_);
        swap_ = nullptr;
        dkQueueDestroy(queue_);
        queue_ = nullptr;
    }
    if (dev_) dkDeviceDestroy(dev_);
    dev_ = nullptr;
}

void DekoBackend::PresentPass(Target& t, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    ++draw_epoch_;
    Hazard(t.img, Access::Sample);
    Hazard(lut_, Access::Sample);
    const DkShader* shaders[2] = {present_vs_, present_ps_};
    dkCmdBufBindShaders(cmd_, DkStageFlag_Vertex | DkStageFlag_Fragment, shaders, 2);
    DkSampler lin;
    dkSamplerDefaults(&lin);
    lin.minFilter = lin.magFilter = DkFilter_Linear;
    lin.wrapMode[0] = lin.wrapMode[1] = lin.wrapMode[2] = DkWrapMode_ClampToEdge;
    DkSampler point;
    dkSamplerDefaults(&point);
    point.wrapMode[0] = point.wrapMode[1] = point.wrapMode[2] = DkWrapMode_ClampToEdge;
    DkResHandle handles[2] = {dkMakeTextureHandle(t.img.desc, SamplerSlot(lin)), dkMakeTextureHandle(lut_.desc, SamplerSlot(point))};
    if (descs_dirty_) {
        dkCmdBufBarrier(cmd_, DkBarrier_None, DkInvalidateFlags_Descriptors);
        descs_dirty_ = false;
    }
    dkCmdBufBindTextures(cmd_, DkStage_Fragment, 0, handles, 2);
    DkColorState cs;
    dkColorStateDefaults(&cs);
    dkCmdBufBindColorState(cmd_, &cs);
    DkColorWriteState ws;
    dkColorWriteStateDefaults(&ws);
    dkCmdBufBindColorWriteState(cmd_, &ws);
    DkDepthStencilState ds;
    dkDepthStencilStateDefaults(&ds);
    ds.depthTestEnable = ds.depthWriteEnable = false;
    dkCmdBufBindDepthStencilState(cmd_, &ds);
    DkRasterizerState rs;
    dkRasterizerStateDefaults(&rs);
    rs.cullMode = DkFace_None;
    dkCmdBufBindRasterizerState(cmd_, &rs);
    float part[4] = {float(AW(t)) / float(t.img.w), float(AH(t)) / float(t.img.h), 0, 0};
    dkCmdBufPushConstants(cmd_, present_ubo_.gpu, present_ubo_.size, 0, sizeof(part), part);
    dkCmdBufBindUniformBuffer(cmd_, DkStage_Vertex, 0, present_ubo_.gpu, present_ubo_.size);  // vertex slot 0 is the constants' again from BeginFrame
    dkCmdBufBindVtxAttribState(cmd_, nullptr, 0);
    dkCmdBufBindVtxBufferState(cmd_, nullptr, 0);
    DkViewport vp{float(x), float(y), float(w), float(h), 0, 1};
    dkCmdBufSetViewports(cmd_, 0, &vp, 1);
    DkScissor sc{x, y, w, h};
    dkCmdBufSetScissors(cmd_, 0, &sc, 1);
    dkCmdBufDraw(cmd_, DkPrimitive_Triangles, 3, 1, 0, 0);
}

// textures unused for ten seconds go once there are many: levels stream theirs in and out
void DekoBackend::ReleaseOldTextures() {
    constexpr uint64_t kSweepEvery = 60, kIdleFrames = 600;
    // zero copy ranges not drawn for ten seconds
    if (frame_ % kSweepEvery == 0)
        for (auto it = zc_.begin(); it != zc_.end();) it = it->second.seen + kIdleFrames < frame_ ? zc_.erase(it) : std::next(it);
    constexpr size_t kTexturesKept = 2048;
    if (frame_ % kSweepEvery != 0 || textures_.size() <= kTexturesKept) return;
    for (auto it = textures_.begin(); it != textures_.end();) {
        if (it->second.frame + kIdleFrames < frame_) {
            FreeImage(it->second.img);
            it = textures_.erase(it);
        } else {
            ++it;
        }
    }
}

// every 15 frames toward the GPU time to hold: the time goes with the pixels, the square of dyn_; at most 0.05 a step,
// between 0.75 (540p of 720p) and 1, inside 85% to 100% of the target left alone
void DekoBackend::DynamicResolution(double gpu_ms) {
    if (dyn_target_ <= 0) return;
    dyn_ms_ += gpu_ms;
    if (++dyn_frames_ < 15) return;
    double avg = dyn_ms_ / dyn_frames_;
    dyn_ms_ = 0;
    dyn_frames_ = 0;
    if (avg <= dyn_target_ && avg >= dyn_target_ * 0.85) return;
    float want = dyn_ * float(std::sqrt(dyn_target_ * 0.93 / avg));
    want = std::clamp(want, dyn_ - 0.05f, dyn_ + 0.05f);
    want = std::clamp(std::round(want * 64.0f) / 64.0f, 0.75f, 1.0f);
    if (want != dyn_) LOG_FIRST(64, "deko3d: dynamic resolution %.3f (gpu %.2f ms)", want, avg);
    dyn_ = want;
}

void DekoBackend::Mark(uint64_t label) {
    if (!diag_ && !roi_) return;
    Slot& slot = Cur();
    if (slot.marks.size() >= kMarks) return;
    // diagnostics: the earlier passes finished first, so each pass's time is its own work, not the previous pass's
    // tail still in the pipeline (unfenced, a small pass after the scene took the scene's last 3 ms)
    if (diag_) dkCmdBufBarrier(cmd_, DkBarrier_Full, 0);
    dkCmdBufReportCounter(cmd_, DkCounter_Timestamp, reports_mem_.gpu + MarkSlot(uint32_t(slot.marks.size())) * 16);
    slot.marks.push_back(label);
    slot.mark_draws.push_back(0);
}


void DekoBackend::CountDraw(const State& s, const Assembled& a, const PixelKey& pk, const Stage& st) {
    Census& c = census_;
    bool prog = pk.programmable != 0;
    (prog ? c.prog : c.ff)++;
    if (prog && (pk.final_abcd || pk.final_efg)) c.final_comb++;
    for (int i = 0; i < 4; i++) {
        uint32_t mode = prog ? (pk.tex_modes >> (i * 5)) & 0x1F : 0;
        if (prog) c.modes[mode]++;
        else if (pk.ff[i][0] > 1) c.ffops[pk.ff[i][0] & 31]++;
        bool used = prog ? mode != 0 : pk.ff[i][0] > 1;
        if (!st.has[i]) continue;
        c.stage_tex[i]++;
        if (!used) c.stage_unused[i]++;
        const TextureDesc& d = st.desc[i];
        const FormatInfo& fi = Format(d.format);
        if (d.cube) c.cube++;
        if (d.depth > 1) c.vol++;
        if (d.format == 0x0B) c.p8++;
        if (fi.compressed) c.dxt++;
        if (d.linear) c.linear++;
        if (d.levels > 1) c.mipped++;
        if (d.levels > 1 && fi.compressed && ((d.width & 3) || (d.height & 3))) c.mips_dropped++;
        if (st.rt[i]) c.rt++;
        if (st.info[i].shadow) c.shadow++;
        if (st.wide[i] > 0 || st.shrunk[i]) c.wide++;
        // a sampler of a kind the stage's texture is not (a cube read as 2D, ...)
        if (prog && used && st.img[i] && pk.tex[i].kind != TextureKind(d) && !st.rt[i]) c.dummy_kind_mismatch++;
        const auto& t = s.tss[size_t(i)];
        if (t[kTssMinFilter] == 3 || t[kTssMagFilter] == 3) c.aniso++;
        if (t[kTssMipLodBias]) {
            c.lodbias++;
            static uint32_t seen[8];
            for (uint32_t& v : seen) {
                if (v == t[kTssMipLodBias]) break;
                if (!v) {
                    v = t[kTssMipLodBias];
                    LOGI("census: stage %d lod bias %08X (%g), filters min %u mag %u mip %u, max mip level %u, format %02X %ux%u levels %u",
                         i, v, Bits(v), t[kTssMinFilter], t[kTssMagFilter], t[kTssMipFilter], t[kTssMaxMipLevel], d.format, d.width,
                         d.height, d.levels);
                    break;
                }
            }
        }
        if (d.format == 0x0B && !s.palettes[size_t(i)]) c.p8_nopal++;
        // the first few P8 stages: the palette and the combiners reading it
        static int p8_logged = 0;
        if (d.format == 0x0B && s.palettes[size_t(i)] && p8_logged < 3) {
            p8_logged++;
            uint32_t header = s.palettes[size_t(i)];
            const uint32_t* pal = static_cast<const uint32_t*>(kernel::HostPtr(mem::VirtualFromPhysical(kernel::R32(header + 4))));
            uint32_t colours = 256u >> (kernel::R32(header) >> 30), last = colours - 1;  // 256, 128, 64 or 32 entries
            LOGI("census: P8 stage %d %ux%u palette common %08X, %u colours: %08X %08X %08X %08X .. [%u] %08X [%u] %08X", i, d.width, d.height,
                 kernel::R32(header), colours, pal[0], pal[1], pal[2], pal[3], last / 2, pal[last / 2], last, pal[last]);
            LOGI("census: P8 draw combiners prog %u count %X modes %X final %08X %08X | in/out %08X/%08X %08X/%08X a %08X/%08X %08X/%08X",
                 pk.programmable, pk.count, pk.tex_modes, pk.final_abcd, pk.final_efg, pk.rgb_in[0], pk.rgb_out[0], pk.rgb_in[1], pk.rgb_out[1],
                 pk.a_in[0], pk.a_out[0], pk.a_in[1], pk.a_out[1]);
        }
    }
    if (Simple(s, kMAlphaTestEnable)) c.alphatest++;
    if (Simple(s, kMBlendEnable)) c.blend++;
    if (a.topology == Topology::Points) c.points++;
    if (a.topology == Topology::Lines) c.lines++;
    if (s.rs[kRsFogEnable]) c.fog++;
    if (s.rs[kRsSpecularEnable]) c.specular++;
    if (s.rs[kRsFillMode] == kFillWireframe) c.wire++;
}

void DekoBackend::LogCensus(double n) {
    Census& c = census_;
    auto f = [n](uint64_t v) { return double(v) / n; };
    LOGI("census: per frame dropped asm %.1f target %.1f shader %.1f vfmt %.1f | prog %.0f ff %.0f final %.0f | stage tex %.0f %.0f %.0f %.0f, "
         "unused %.0f %.0f %.0f %.0f | no desc %.1f, bad fmt %.1f (%X %X %X %X), decode fail %llu, kind mismatch %.1f",
         f(c.drop_asm), f(c.drop_target), f(c.drop_shader), f(c.drop_vfmt), f(c.prog), f(c.ff), f(c.final_comb), f(c.stage_tex[0]),
         f(c.stage_tex[1]), f(c.stage_tex[2]), f(c.stage_tex[3]), f(c.stage_unused[0]), f(c.stage_unused[1]), f(c.stage_unused[2]),
         f(c.stage_unused[3]), f(c.no_desc), f(c.bad_fmt), c.bad_formats[0] ? c.bad_formats[0] - 1 : 0,
         c.bad_formats[1] ? c.bad_formats[1] - 1 : 0, c.bad_formats[2] ? c.bad_formats[2] - 1 : 0, c.bad_formats[3] ? c.bad_formats[3] - 1 : 0,
         (unsigned long long)c.decode_fail, f(c.dummy_kind_mismatch));
    char modes[256] = "", ffops[256] = "";
    size_t m = 0, o = 0;
    for (int i = 0; i < 32; i++) {
        if (c.modes[i] && m < sizeof(modes) - 16) m += std::snprintf(modes + m, sizeof(modes) - m, " %X:%.0f", i, f(c.modes[i]));
        if (c.ffops[i] && o < sizeof(ffops) - 16) o += std::snprintf(ffops + o, sizeof(ffops) - o, " %d:%.0f", i, f(c.ffops[i]));
    }
    LOGI("census: feedback copies %.1f per frame, %.1f MB, sampled directly %.1f", f(c.fb_copies), c.fb_mb / n, f(c.fb_direct));
    if (guest_mem_)
        LOGI("census: zero copy per frame %.2f MB in place, %.2f MB copied, %.3f MB cleaned | copied ranges: %.0f not guest RAM or small, "
             "%.1f misaligned, %.0f changing",
             f(c.zc_bytes) / 1048576.0, f(c.zc_copied) / 1048576.0, f(c.zc_cleaned) / 1048576.0, f(c.zc_not_guest), f(c.zc_misaligned),
             f(c.zc_dynamic));
    LOGI("census: tex modes%s | ff ops%s | cube %.1f vol %.1f p8 %.1f dxt %.0f linear %.0f mipped %.0f (mips dropped %.1f) rt %.1f "
         "shadow %.1f wide %.1f p8 without palette %.1f | aniso %.0f lodbias %.0f | alphatest %.0f blend %.0f fog %.0f specular %.0f points %.1f lines %.1f wire %.1f",
         modes, ffops, f(c.cube), f(c.vol), f(c.p8), f(c.dxt), f(c.linear), f(c.mipped), f(c.mips_dropped), f(c.rt), f(c.shadow),
         f(c.wide), f(c.p8_nopal), f(c.aniso), f(c.lodbias), f(c.alphatest), f(c.blend), f(c.fog), f(c.specular), f(c.points), f(c.lines), f(c.wire));
    census_ = Census{};
}

void DekoBackend::LogPerf() {
    double el = Ms(perf_t_);
    if (el < 2000.0) return;
    static uint64_t last_frame = 0;
    double n = double(frame_ - last_frame);
    last_frame = frame_;
    if (n <= 0) return;
    double title_wait = wait_ns_.exchange(0) / 1e6 / n;
    if (!roi_) {
        LOGI("deko3d: draw parts per frame: assemble %.2f ms, textures %.2f ms, constants %.2f ms, geometry %.2f ms",
             perf_.assemble / n, perf_.bind / n, perf_.consts / n, perf_.geometry / n);
        LOGI("deko3d: %.1f fps | per frame: %.0f draws %.2f ms, present %.2f ms, gpu wait %.2f ms (title %.2f), %.1f syncs %.2f ms, "
             "%.1f tex uploads %.2f ms, %.2f MB upload, %.1f barriers | %llu shaders %.0f ms, %zu targets %zu textures",
             n * 1000.0 / el, perf_.draws / n, perf_.draw / n, perf_.present / n, perf_.wait / n, title_wait, perf_.syncs / n, perf_.sync / n,
             perf_.texups / n, perf_.texup / n, perf_.upload / n / 1048576.0, perf_.barriers / n, (unsigned long long)perf_.compiles,
             perf_.compile, targets_.size(), textures_.size());
    }
    // frame times (present to present) as percentiles, the GPU's time per frame, and with the render thread the
    // title's capture and wait per frame
    if (!intervals_.empty()) {
        std::sort(intervals_.begin(), intervals_.end());
        auto pct = [&](double p) { return intervals_[std::min(intervals_.size() - 1, size_t(p * intervals_.size()))]; };
        size_t slow = size_t(intervals_.end() - std::upper_bound(intervals_.begin(), intervals_.end(), 17.5f));
        if (!roi_) LOGI("deko3d: resolution %ux%u (%.3f)", uint32_t(std::lround(HostW(guest_w_) * dyn_)), uint32_t(std::lround(HostH(guest_h_) * dyn_)), dyn_);
        // the title's thread: its frame less its waits for the render thread and the GPU
        double capture = g_perf_capture_ns.exchange(0) / n / 1e6, rt_wait = g_perf_wait_ns.exchange(0) / n / 1e6;
        if (!roi_) LOGI("deko3d: frame ms p50 %.2f p90 %.2f p99 %.2f max %.2f, %zu of %zu over 17.5 | gpu %.2f ms | title busy %.2f ms (capture %.2f), "
             "waits render thread %.2f gpu %.2f",
             pct(0.5), pct(0.9), pct(0.99), intervals_.back(), slow, intervals_.size(), gpu_frames_ ? gpu_ms_ / gpu_frames_ : 0.0,
             el / n - rt_wait - title_wait, capture, rt_wait, title_wait);
        if (!roi_ && !gpu_intervals_.empty()) {
            // frame end to frame end on the GPU: late frames here can miss a vertical blank, late presents above need not
            std::sort(gpu_intervals_.begin(), gpu_intervals_.end());
            auto gp = [&](double p) { return gpu_intervals_[std::min(gpu_intervals_.size() - 1, size_t(p * gpu_intervals_.size()))]; };
            size_t late = size_t(gpu_intervals_.end() - std::upper_bound(gpu_intervals_.begin(), gpu_intervals_.end(), 17.5f));
            LOGI("deko3d: gpu frame end intervals ms p50 %.2f p90 %.2f p99 %.2f max %.2f, %zu of %zu over 17.5", gp(0.5), gp(0.9), gp(0.99),
                 gpu_intervals_.back(), late, gpu_intervals_.size());
            gpu_intervals_.clear();
        }
        if (roi_) {
            // Service counters, not sampling: no guest suspension or stack walking.
            u64 render_ticks = 0, game_ticks = 0;
            Result render_rc = svcGetInfo(&render_ticks, InfoType_ThreadTickCount, CUR_THREAD_HANDLE, UINT64_MAX);
            for (const auto& t : kernel::Threads()) {
                if (t->id == 0x14 && t->host_thread != roi_game_handle_) {
                    roi_game_handle_ = Handle(t->host_thread);
                    roi_game_ticks_ = 0;
                }
            }
            Result game_rc = roi_game_handle_ ? svcGetInfo(&game_ticks, InfoType_ThreadTickCount, roi_game_handle_, UINT64_MAX) : 1;
            const double tick_ms = 1000.0 / double(armGetSystemTickFreq()) / n;
            const double render_cpu = R_SUCCEEDED(render_rc) && roi_render_ticks_ && render_ticks >= roi_render_ticks_
                ? (render_ticks - roi_render_ticks_) * tick_ms : -1;
            const double game_cpu = R_SUCCEEDED(game_rc) && roi_game_ticks_ && game_ticks >= roi_game_ticks_
                ? (game_ticks - roi_game_ticks_) * tick_ms : -1;
            roi_render_ticks_ = R_SUCCEEDED(render_rc) ? render_ticks : 0;
            roi_game_ticks_ = R_SUCCEEDED(game_rc) ? game_ticks : 0;
            LOGI("ROI_CPU: window=%u render_cpu_ms=%.2f guest_14_cpu_ms=%.2f (-1 unavailable)", roi_window_, render_cpu, game_cpu);
            const bool settle = roi_window_ == 0 && roi_samples_ < 2;
            LOGI("ROI: window=%u draw_fast=%u res=%ux%u state=%s frames=%.0f elapsed_ms=%.1f fps=%.2f p50_ms=%.2f p90_ms=%.2f p99_ms=%.2f over17_5=%zu gpu_ms=%.2f gpu_frames=%u title_active_wall_ms=%.2f capture_ms=%.2f record_ms=%.2f rt_wait_ms=%.2f gpu_wait_ms=%.2f draws=%.0f upload_mb=%.3f shaders=%llu compile_ms=%.2f",
                 roi_window_, unsigned(draw_fast_), uint32_t(std::lround(HostW(guest_w_) * dyn_)), uint32_t(std::lround(HostH(guest_h_) * dyn_)),
                 settle ? "settle" : "measure", n, el, n * 1000 / el, pct(0.5), pct(0.9), pct(0.99), slow,
                 gpu_frames_ ? gpu_ms_ / gpu_frames_ : 0, gpu_frames_, el / n - rt_wait - title_wait,
                 capture, perf_.draw / n, rt_wait, title_wait, perf_.draws / n, perf_.upload / n / 1048576.,
                 (unsigned long long)perf_.compiles, perf_.compile);
            LOGI("ROI: parts_ms assemble=%.2f textures=%.2f constants=%.2f geometry=%.2f present=%.2f sync=%.2f tex_upload=%.2f",
                 perf_.assemble/n, perf_.bind/n, perf_.consts/n, perf_.geometry/n, perf_.present/n, perf_.sync/n, perf_.texup/n);
            LOGI("ROI_DRAW: draw_fast=%u state_checks=%.0f state_emits=%.0f sampler_hits=%.0f", unsigned(draw_fast_),
                 perf_.state_checks/n, perf_.state_emits/n, perf_.sampler_hits/n);
            LOGI("ROI_WAIT: readback_ms=%.2f resource_ms=%.2f fence_ms=%.2f flush_ms=%.2f copy_ms=%.2f next_frame_ms=%.2f waits=%.1f blocking_readbacks=%.1f late_readbacks=%.1f",
                 g_perf_wait_kind_ns[0].exchange(0)/n/1e6, g_perf_wait_kind_ns[1].exchange(0)/n/1e6,
                 g_perf_wait_kind_ns[2].exchange(0)/n/1e6, g_perf_wait_kind_ns[3].exchange(0)/n/1e6,
                 g_perf_wait_kind_ns[4].exchange(0)/n/1e6, g_perf_wait_kind_ns[5].exchange(0)/n/1e6,
                 g_perf_waits.exchange(0)/n, g_perf_sync_block.exchange(0)/n, g_perf_sync_late.exchange(0)/n);
            if (!settle) {
                roi_elapsed_ += el; roi_frames_ += n; roi_gpu_ += gpu_ms_; roi_gpu_frames_ += gpu_frames_;
                roi_capture_ += capture * n; roi_record_ += perf_.draw;
            }
        }
        intervals_.clear();
    }
    if ((diag_ || roi_) && gpu_frames_) {
        std::vector<std::pair<double, uint64_t>> passes;
        for (auto& [label, ms] : pass_ms_) passes.push_back({ms / gpu_frames_, label});
        std::sort(passes.rbegin(), passes.rend());
        char line[512] = {};
        size_t o = 0;
        for (size_t i = 0; i < passes.size() && i < (roi_ ? 5u : 10u) && o < sizeof(line) - 48; i++) {
            uint64_t l = passes[i].second;
            if (l == 1) o += std::snprintf(line + o, sizeof(line) - o, " present %.2f", passes[i].first);
            else if (l == 2) o += std::snprintf(line + o, sizeof(line) - o, " feedback-copy %.2f", passes[i].first);
            else if (l == 3) o += std::snprintf(line + o, sizeof(line) - o, " fullscreen-draw %.2f", passes[i].first);
            else
                o += std::snprintf(line + o, sizeof(line) - o, " %ux%u%s%s%s %.2f", uint32_t(l >> 32), uint32_t(l >> 16) & 0xFFFF,
                                   l & 4 ? "*" : "", l & 2 ? "+z" : "", l & 8 ? "(z only)" : "", passes[i].first);
        }
        LOGI("deko3d: gpu ms per frame by target (* scaled):%s", line);
    }
    pass_ms_.clear();
    if (diag_) LogCensus(n);
    else census_ = Census{};
    gpu_ms_ = 0;
    gpu_frames_ = 0;
    perf_ = Perf{};
#ifdef B3_ROI_BUILD
    if (roi_ && ++roi_samples_ >= 10) {
        if (roi_frames_ > 0) LOGI("ROI_SUMMARY: window=%u draw_fast=%u res=%s frames=%.0f fps=%.2f gpu_ms=%.2f capture_ms=%.2f record_ms=%.2f",
            roi_window_, unsigned(draw_fast_), "1280x720", roi_frames_, roi_frames_ * 1000 / roi_elapsed_,
            roi_gpu_frames_ ? roi_gpu_ / roi_gpu_frames_ : 0, roi_capture_ / roi_frames_, roi_record_ / roi_frames_);
        ++roi_window_; roi_samples_ = 0;
        roi_elapsed_ = roi_frames_ = roi_gpu_ = roi_gpu_frames_ = roi_capture_ = roi_record_ = 0;
        LOGI("ROI: window=%u draw_fast=%u fixed 1280x720", roi_window_, unsigned(draw_fast_));
    }
#endif
    perf_t_ = Clock::now();
}

}  // namespace

std::unique_ptr<Backend> CreateBackend() { return std::make_unique<DekoBackend>(); }

// until the GPU has drawn present number n, counted from 0; the title's D3D thread
void WaitGpuFrame(uint64_t n) {
    if (g_self) g_self->WaitFrame(n);
}

bool GpuFrameDone(uint64_t n) { return !g_self || g_self->FrameDone(n); }

bool ZeroCopyActive() { return g_self && g_self->ZeroCopy(); }

uint64_t InPlaceFrame(uint32_t va) { return g_self ? g_self->InPlaceFrame(va) : 0; }

bool FrameReadInPlace(uint64_t n) { return g_self && g_self->FrameReadInPlace(n); }

void ShutdownForExit() {
    if (g_self) g_self->ShutdownForExit();
}

}  // namespace gpu
