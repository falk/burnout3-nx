// The Direct3D 12 backend on a thread of its own. The title's render thread records each call with what the console
// takes at call time (state changes, resource headers, indices, rects) and goes on; vertex, texture and surface memory
// are read when the draw runs, as the console's GPU reads them after the call returns. The XDK's fences, busy checks
// and locks wait for that.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef __SWITCH__
#include <arm_neon.h>
#include <switch.h>
#else
#include <windows.h>
#include <smmintrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/diag.h"
#include "core/log.h"
#include "d3d/const_dirty.h"
#include "d3d/gpu_frame.h"
#include "gpu/backend.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"

namespace gpu {

namespace {

// commands handed over at once, so the render thread keeps close behind and a wait is short
constexpr uint32_t kBatchCommands = 64;

// resource headers the render thread reads: common, data, lock, format, size
constexpr uint32_t kHeaderBytes = 20;
constexpr uint32_t kHeaderSlot = 32;
constexpr uint32_t kHeaderRing = 1u << 20;  // guest memory for header copies; a copy is renewed after half a ring

enum class WaitFor : uint8_t { ReadBack, Resource, Fence, Flush, Copy, Frame, Count };

enum class Op : uint8_t { State, Draw, Clear, Copy, Present, Sync, RecordLate, Flush, Gamma, ClearReport, GetReport };

struct Cmd {
    Op op;
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t d = 0;
    float f = 0;
    // data and size: bytes in Batch::bytes
    uint32_t data = 0;
    uint32_t size = 0;
    // index in Batch::vs / ps: the shader changed
    int32_t vs = -1;
    int32_t ps = -1;
    Surface s1;
    Surface s2;
};

struct Batch {
    uint64_t id = 0;
    std::vector<Cmd> cmds;
    std::vector<uint8_t> bytes;
    std::vector<std::shared_ptr<VertexShader>> vs;
    std::vector<std::shared_ptr<PixelShader>> ps;
};

// the plain data in State, around its shader pointers and the resource headers (sent as copies); of the transforms only
// view, projection, texture and world are drawn with, of the simple render states only those src/d3d/d3d12.cpp and
// MakePixelKey read: alpha test to depth offset, the combiner constants and the final combiner constants
struct Range {
    size_t off;
    size_t len;
};
size_t At(const State& s, const void* p) {
    return size_t(static_cast<const uint8_t*>(p) - reinterpret_cast<const uint8_t*>(&s));
}
// Resource headers are aligned; mixing higher address bits avoids mapping
// whole groups of neighbouring allocations onto the same recent-cache slots.
size_t ResourceSlot(uint32_t header) { return ((header >> 4) ^ (header >> 10)) & 63; }

// These snapshots are exactly 20 bytes. Compare them without an out-of-line
// memcmp call or reading beyond a short resource's padded snapshot.
bool SameHeader(const uint8_t* a, const uint8_t* b) {
    uint64_t ax[2], bx[2];
    uint32_t at, bt;
    std::memcpy(ax, a, 16); std::memcpy(bx, b, 16);
    std::memcpy(&at, a + 16, 4); std::memcpy(&bt, b + 16, 4);
    return ((ax[0] ^ bx[0]) | (ax[1] ^ bx[1]) | uint64_t(at ^ bt)) == 0;
}

// bytes of the simple render states from method first to method last
size_t SimpleBytes(uint32_t first, uint32_t last) { return size_t(last - first) + sizeof(uint32_t); }
std::vector<Range> StateRanges() {
    static const State s;
    return {
        {At(s, &s.simple[kMAlphaTestEnable >> 2]), SimpleBytes(kMAlphaTestEnable, kMDepthOffset)},
        {At(s, &s.simple[kMPsConstant0 >> 2]), SimpleBytes(kMPsConstant0, kMPsAlphaOutputs - 4)},
        {At(s, &s.simple[kMPsFinalConstant0 >> 2]), SimpleBytes(kMPsFinalConstant0, kMPsFinalConstant1)},
        {At(s, &s.rs), At(s, &s.textures) - At(s, &s.rs)},
        {At(s, &s.index_buffer), At(s, &s.vs) - At(s, &s.index_buffer)},
        {At(s, &s.vsc), At(s, &s.rt) - At(s, &s.vsc)},
        {At(s, &s.scissor_count), At(s, &s.transforms) - At(s, &s.scissor_count)},
        {At(s, &s.light_enable), sizeof(State) - At(s, &s.light_enable)},
    };
}

// the fixed function vertex state: transforms, material and lights, read by draws without a vertex program only, so
// sent with those (a vertex program draw leaves them as last sent; the next fixed function draw sends what changed)
std::vector<Range> FixedRanges() {
    static const State s;
    return {
        {At(s, &s.transforms[2]), 2 * sizeof(s.transforms[0])},
        {At(s, &s.transforms[16]), 4 * sizeof(s.transforms[0])},
        {At(s, &s.transforms[256]), sizeof(s.transforms[0])},
        {At(s, &s.material), At(s, &s.light_enable) - At(s, &s.material)},
    };
}

#ifdef __SWITCH__
// the bits where 16 bytes of a and b differ; none set: equal
inline uint8x16_t Diff16(const uint8_t* a, const uint8_t* b) { return veorq_u8(vld1q_u8(a), vld1q_u8(b)); }
inline bool Zero(uint8x16_t d) { return vmaxvq_u32(vreinterpretq_u32_u8(d)) == 0; }

// equal bytes, 16 at a time: most of the state is compared on every draw
bool Same(const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;
    uint8x16_t diff = vdupq_n_u8(0);
    for (; i + 16 <= n; i += 16) diff = vorrq_u8(diff, Diff16(a + i, b + i));
    if (!Zero(diff)) return false;
    return i == n || !std::memcmp(a + i, b + i, n - i);
}

constexpr size_t kPiece = 64;  // bytes compared and sent at once

// equal 64 byte pieces, the scan over the state on every draw
bool SamePiece(const uint8_t* a, const uint8_t* b) {
    uint8x16_t diff = vorrq_u8(vorrq_u8(Diff16(a, b), Diff16(a + 16, b + 16)), vorrq_u8(Diff16(a + 32, b + 32), Diff16(a + 48, b + 48)));
    return Zero(diff);
}
#else
// equal bytes, 16 at a time: most of the state is compared on every draw
bool Same(const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;
    __m128i diff = _mm_setzero_si128();
    for (; i + 16 <= n; i += 16)
        diff = _mm_or_si128(diff, _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)),
                                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i))));
    if (!_mm_testz_si128(diff, diff)) return false;
    return i == n || !std::memcmp(a + i, b + i, n - i);
}

constexpr size_t kPiece = 64;  // bytes compared and sent at once

// equal 64 byte pieces, the scan over the state on every draw
bool SamePiece(const uint8_t* a, const uint8_t* b) {
    __m128i diff = _mm_setzero_si128();
    for (size_t i = 0; i < kPiece; i += 16) {
        __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        diff = _mm_or_si128(diff, _mm_xor_si128(x, y));
    }
    return _mm_testz_si128(diff, diff);
}
#endif

using Clock = std::chrono::steady_clock;
uint64_t Ns(Clock::time_point t) {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count());
}

class Threaded final : public Backend {
public:
    explicit Threaded(std::unique_ptr<Backend> inner) : inner_(std::move(inner)), ranges_(StateRanges()), fixed_(FixedRanges()) {
        ring_ = mem::AllocSystem(kHeaderRing, 64);
        cur_ = NewBatch();
        std::thread([this] { Run(); }).detach();
    }
    ~Threaded() override {
        {
            std::lock_guard lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        std::unique_lock lock(m_);
        // at exit the worker may be gone mid-batch: then the inner backend is left, not freed under it
        if (!done_cv_.wait_for(lock, std::chrono::milliseconds(500), [&] { return exited_; })) (void)inner_.release();
    }

    bool Init(uint32_t width, uint32_t height) override { return inner_->Init(width, height); }

    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override {
        auto t = Clock::now();
        SendState(s, true);
        state_ns_ += Ns(t);
        draws_++;
        Cmd c{Op::Draw};
        c.a = prim;
        c.b = start;
        c.c = count;
        if (indices) {
            c.d = 1;
            c.data = Put(indices, size_t(count) * 2);
            index_bytes_ += size_t(count) * 2;
        }
        cur_->cmds.push_back(c);
        for (uint32_t tex : s.textures) Use(tex);
        for (uint32_t pal : s.palettes) Use(pal);
        for (const auto& st : s.streams) Use(st.vb);
        Use(s.rt.header);
        Use(s.zs.header);
        Written(s.rt);
        if (cur_->cmds.size() >= kBatchCommands) Submit();
        uint64_t ns = Ns(t);
        g_perf_capture_ns += ns;
        draw_ns_ += ns;
    }

    void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
               uint32_t stencil) override {
        auto t = Clock::now();
        SendState(s);
        Cmd c{Op::Clear};
        c.a = count;
        c.b = flags;
        c.c = color;
        c.d = stencil;
        c.f = z;
        if (count && rects) {
            c.data = Put(rects, size_t(count) * 16);
            c.size = count * 16;
        }
        cur_->cmds.push_back(c);
        Use(s.rt.header);
        Use(s.zs.header);
        if (flags & kClearTarget) Written(s.rt);
        g_perf_capture_ns += Ns(t);
    }

    // rects and points are taken now; the copy itself, CPU or GPU, runs in order like the console's
    void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects,
              uint32_t dst_points) override {
        auto t = Clock::now();
        Cmd c{Op::Copy};
        c.a = count;
        c.b = count && src_rects ? Scratch(kernel::HostPtr(src_rects), count * 16) : 0;
        c.c = count && dst_points ? Scratch(kernel::HostPtr(dst_points), count * 8) : 0;
        c.s1 = src;
        c.s1.header = Header(src.header);
        c.s2 = dst;
        c.s2.header = Header(dst.header);
        cur_->cmds.push_back(c);
        Use(src.header);
        Use(dst.header);
        Written(dst);
        g_perf_capture_ns += Ns(t);
    }

    void Present(const Surface& front) override {
        auto t = Clock::now();
        Cmd c{Op::Present};
        c.s1 = front;
        c.s1.header = Header(front.header);
        cur_->cmds.push_back(c);
        Use(front.header);
        // where the capture goes, every 300 frames
        static const bool diag = [] {
            const char* e = std::getenv("B3_PERF_DIAG");
            return e && *e == '1';
        }();
        if (++presents_ % 300 == 0 && draws_ && diag) {
            LOGI("render thread: capture per draw %.2f us, of it state %.2f us; %.0f header copies per frame, %.0f index bytes per draw",
                 draw_ns_ / 1e3 / draws_, state_ns_ / 1e3 / draws_, double(header_copies_) / 300, double(index_bytes_) / draws_);
            draws_ = draw_ns_ = state_ns_ = header_copies_ = index_bytes_ = 0;
        }
        uint64_t id = Submit();
        g_perf_capture_ns += Ns(t);
        // the title records one frame while the one before it draws
        Wait(last_present_, WaitFor::Frame, nullptr);
        last_present_ = id;
    }

    void Sync(uint32_t data, uint32_t) override { ReadBack(data, false); }

    void SyncLate(uint32_t data) override {
        if (!inner_->DeliverLate(data)) {
            if (diag::Perf()) {
                LOG_FIRST(16, "render thread: no finished copy of %08X for a NOOVERWRITE lock, reading it back now", data);
            }
            written_.insert(data);
            last_written_ = ~0u;
            ReadBack(data, true);
        }
        Cmd c{Op::RecordLate};
        c.a = data;
        cur_->cmds.push_back(c);
    }

    void Flush() override {
        cur_->cmds.push_back(Cmd{Op::Flush});
        Wait(Submit(), WaitFor::Flush, "flush");
    }

    void SetGamma(const uint8_t* ramp) override {
        Cmd c{Op::Gamma};
        c.data = Put(ramp, 768);
        cur_->cmds.push_back(c);
    }

    void ClearReport() override { cur_->cmds.push_back(Cmd{Op::ClearReport}); }
    void GetReport(uint32_t report) override {
        Cmd c{Op::GetReport};
        c.a = report;
        cur_->cmds.push_back(c);
    }

    // A batch is done once the render thread ran it. With zero copy the GPU reads guest memory when it draws: a
    // vertex buffer it drew in place is free only once the GPU is past that frame, and a fence only once the GPU is
    // past its frame when that frame read anything in place
    uint32_t Fence() override { return uint32_t(cur_->id); }
    bool FencePending(uint32_t fence) override { return fence > done_.load() || !GpuDone(FenceFrame(fence)); }
    void WaitFence(uint32_t fence) override {
        Wait(fence, WaitFor::Fence, "fence");
        WaitGpu(FenceFrame(fence));
    }
    bool Busy(uint32_t resource) override {
        auto it = last_use_.find(resource);
        return it != last_use_.end() && (it->second > done_.load() || !GpuDone(ResourceFrame(resource)));
    }
    void WaitResource(uint32_t resource) override {
        auto it = last_use_.find(resource);
        if (it == last_use_.end()) return;
        Wait(it->second, WaitFor::Resource, "resource");
        WaitGpu(ResourceFrame(resource));
    }

private:
    static constexpr uint64_t kFrameRing = 1u << 16;  // batches remembered with their frame
    // the present number the batch's draws belong to (presents the render thread ran before it)
    uint64_t BatchFrame(uint64_t id) const { return batch_frame_[id & (kFrameRing - 1)].load(std::memory_order_acquire); }
    // the present (+1) the GPU must be past before the CPU may write what the fence or resource covers, 0 none
    uint64_t FenceFrame(uint64_t id) {
#ifdef __SWITCH__
        if (!id || !gpu::ZeroCopyActive() || id + kFrameRing <= next_id_) return 0;
        uint64_t f = BatchFrame(id);
        return gpu::FrameReadInPlace(f) ? f + 1 : 0;
#else
        (void)id;
        return 0;
#endif
    }
    uint64_t ResourceFrame(uint32_t resource) {
#ifdef __SWITCH__
        if (!gpu::ZeroCopyActive()) return 0;
        return gpu::InPlaceFrame(mem::VirtualFromPhysical(kernel::R32(resource + 4)));
#else
        (void)resource;
        return 0;
#endif
    }
    bool GpuDone(uint64_t frame1) {
#ifdef __SWITCH__
        return !frame1 || gpu::GpuFrameDone(frame1 - 1);
#else
        (void)frame1;
        return true;
#endif
    }
    // until the GPU drew present frame1 - 1; one not presented yet is drawn by a flush
    void WaitGpu(uint64_t frame1) {
#ifdef __SWITCH__
        if (GpuDone(frame1)) return;
        uint64_t f = frame1 - 1;
        if (f >= presents_run_.load()) {
            cur_->cmds.push_back(Cmd{Op::Flush});
            Wait(Submit(), WaitFor::Flush, "zero copy flush");
        } else {
            gpu::WaitGpuFrame(f);
        }
#else
        (void)id;
#endif
    }

    Batch* NewBatch() {
        Batch* b;
        {
            std::lock_guard lock(m_);
            if (free_.empty()) {
                b = new Batch;
            } else {
                b = free_.back();
                free_.pop_back();
            }
        }
        b->id = ++next_id_;
        return b;
    }

    uint32_t Put(const void* p, size_t n) {
        size_t at = (cur_->bytes.size() + 3) & ~size_t(3);
        cur_->bytes.resize(at + n);
        std::memcpy(cur_->bytes.data() + at, p, n);
        return uint32_t(at);
    }

    void ReadBack(uint32_t data, bool late) {
        // only surfaces the GPU wrote since the last read-back hold anything newer than guest memory
        last_written_ = ~0u;
        if (!written_.erase(data)) return;
        (late ? g_perf_sync_late : g_perf_sync_block)++;
        if (diag::Perf() && logged_.insert(uint64_t(data) << 1 | (late ? 1 : 0)).second) {
            const char* why = late ? "a NOOVERWRITE lock without a finished copy" : "a lock / BlockOnResource";
            LOGI("render thread: read-back of %08X for %s", data, why);
        }
        Cmd c{Op::Sync};
        c.a = data;
        cur_->cmds.push_back(c);
        Wait(Submit(), WaitFor::ReadBack, "read-back");
    }

    // bytes the render thread reads through a guest address, in the header ring
    uint32_t Scratch(const void* p, uint32_t n) {
        uint32_t size = (n + kHeaderSlot - 1) & ~(kHeaderSlot - 1);
        if (ring_pos_ % kHeaderRing + size > kHeaderRing) ring_pos_ += kHeaderRing - ring_pos_ % kHeaderRing;
        uint32_t at = ring_ + uint32_t(ring_pos_ % kHeaderRing);
        ring_pos_ += size;
        std::memcpy(kernel::HostPtr(at), p, n);
        return at;
    }

    // A header as it is now. The console's push buffer takes a texture's format and address, and a surface's or vertex
    // buffer's address, when they are set; the title points the same header elsewhere before the GPU gets there.
    uint32_t Header(uint32_t header) {
        if (!header || !ring_) return header;
        // vertex buffer, index buffer and palette headers are 12 bytes: reading 20 can run off the end of the heap
        const uint32_t type = kernel::R32(header) & 0x70000;  // 0 vertex, 0x10000 index, 0x30000 palette
        uint8_t now[kHeaderBytes];
        if (type < 0x20000 || type == 0x30000) {
            std::memcpy(now, kernel::HostPtr(header), 12);
            std::memset(now + 12, 0, kHeaderBytes - 12);
        } else {
            std::memcpy(now, kernel::HostPtr(header), kHeaderBytes);
        }
        Recent<HeaderCopy>& r = recent_headers_[ResourceSlot(header)];
        HeaderCopy& c = r.key == header ? *r.value : headers_[header];
        r = {header, &c};
        if (c.made && SameHeader(c.bytes, now) && ring_pos_ - c.made < kHeaderRing / 2) return c.at;
        c.at = Scratch(now, kHeaderBytes);
        header_copies_++;
        c.made = ring_pos_;
        std::memcpy(c.bytes, now, kHeaderBytes);
        return c.at;
    }

    // a field of the render thread's State, when it differs from what was sent
    void Delta(const void* field, const void* value, size_t n) {
        uint8_t* sent = reinterpret_cast<uint8_t*>(&sent_) + At(sent_, field);
        if (Same(sent, static_cast<const uint8_t*>(value), n)) return;
        StoreDelta(field, value, n);
    }

    // SendRange already compared a full piece; avoid scanning changed bytes twice.
    void StoreDelta(const void* field, const void* value, size_t n) {
        uint8_t* sent = reinterpret_cast<uint8_t*>(&sent_) + At(sent_, field);
        std::memcpy(sent, value, n);
        uint32_t head[2] = {uint32_t(At(sent_, field)), uint32_t(n)};
        size_t at = cur_->bytes.size();
        cur_->bytes.resize(at + 8 + n);
        std::memcpy(cur_->bytes.data() + at, head, 8);
        std::memcpy(cur_->bytes.data() + at + 8, value, n);
    }

    // the pieces of a range that changed, the last one shorter when the range ends inside it
    void SendRange(const uint8_t* now, const Range& r) {
        const uint8_t* sent = reinterpret_cast<const uint8_t*>(&sent_);
        size_t off = r.off;
        size_t end = r.off + r.len;
        for (; off + kPiece <= end; off += kPiece) {
            if (!SamePiece(sent + off, now + off)) StoreDelta(sent + off, now + off, kPiece);
        }
        if (off < end) Delta(sent + off, now + off, end - off);
    }

    // the constant range's pieces d3d8.cpp marked written (const_dirty.h); every 1024th time the whole range is
    // compared too, and a change nothing marked is logged
    void SendConstants(const uint8_t* now, const Range& r) {
        uint64_t dirty = g_const_dirty.exchange(0, std::memory_order_relaxed);
        const uint8_t* sent = reinterpret_cast<const uint8_t*>(&sent_);
        for (; dirty; dirty &= dirty - 1) {
            size_t off = r.off + size_t(__builtin_ctzll(dirty)) * kConstPiece;
            if (off >= r.off + r.len) break;
            size_t n = std::min(kConstPiece, r.off + r.len - off);
            if (n == kPiece ? !SamePiece(sent + off, now + off) : !Same(sent + off, now + off, n))
                StoreDelta(sent + off, now + off, n);
        }
        if (++const_checks_ % 1024) return;
        size_t before = cur_->bytes.size();
        SendRange(now, r);
        if (cur_->bytes.size() != before)
            LOG_FIRST(4, "render thread: constants written without MarkConstants (%zu bytes)", cur_->bytes.size() - before);
    }

    // what changed since the last command, in 64 byte pieces
    void SendState(const State& s, bool fixed = false) {
        const uint8_t* now = reinterpret_cast<const uint8_t*>(&s);
        Cmd c{Op::State};
        size_t first = cur_->bytes.size();
        for (const Range& r : ranges_) {
            if (r.off == At(s, &s.vsc)) SendConstants(now, r);
            else SendRange(now, r);
        }
        // Clear and the like read none of it; a draw with a vertex program neither
        bool prog = (s.vs_handle & 1) && s.vs && !s.vs->program.empty();
        if (fixed && !prog)
            for (const Range& r : fixed_) SendRange(now, r);
        std::array<uint32_t, 4> textures;
        for (size_t i = 0; i < 4; i++) textures[i] = Header(s.textures[i]);
        Delta(&sent_.textures, &textures, sizeof(textures));
        std::array<uint32_t, 4> palettes;
        for (size_t i = 0; i < 4; i++) palettes[i] = Header(s.palettes[i]);
        Delta(&sent_.palettes, &palettes, sizeof(palettes));
        auto streams = s.streams;
        for (auto& st : streams) st.vb = Header(st.vb);
        Delta(&sent_.streams, &streams, sizeof(streams));
        Surface rt = s.rt;
        Surface zs = s.zs;
        rt.header = Header(rt.header);
        zs.header = Header(zs.header);
        Delta(&sent_.rt, &rt, sizeof(rt));
        Delta(&sent_.zs, &zs, sizeof(zs));
        if (s.vs.get() != sent_.vs.get()) {
            sent_.vs = s.vs;
            c.vs = int32_t(cur_->vs.size());
            cur_->vs.push_back(s.vs);
        }
        if (s.ps.get() != sent_.ps.get()) {
            sent_.ps = s.ps;
            c.ps = int32_t(cur_->ps.size());
            cur_->ps.push_back(s.ps);
        }
        if (cur_->bytes.size() == first && c.vs < 0 && c.ps < 0) return;
        c.data = uint32_t(first);
        c.size = uint32_t(cur_->bytes.size() - first);
        cur_->cmds.push_back(c);
    }

    // resources by guest header: the batch that last read them
    uint64_t const_checks_ = 0;

    void Use(uint32_t header) {
        if (!header) return;
        Recent<uint64_t>& r = recent_uses_[ResourceSlot(header)];
        if (r.key != header) r = {header, &last_use_[header]};
        *r.value = cur_->id;
    }
    void Written(const Surface& s) {
        // most draws go to the target the last one did: the set is touched when it changes
        if (!s.header || s.data == last_written_) return;
        last_written_ = s.data;
        written_.insert(mem::VirtualFromPhysical(s.data));
    }

    uint64_t Submit() {
        uint64_t id = cur_->id;
        {
            std::lock_guard lock(m_);
            queue_.push_back(cur_);
        }
        cv_.notify_one();
        cur_ = NewBatch();
        return id;
    }

    void Wait(uint64_t id, WaitFor kind, const char* why) {
        if (!id || done_.load() >= id) return;
        if (id >= cur_->id) id = Submit();
        auto t = Clock::now();
        {
            std::unique_lock lock(m_);
            done_cv_.wait(lock, [&] { return done_.load() >= id; });
        }
        uint64_t ns = Ns(t);
        g_perf_wait_ns += ns;
        g_perf_wait_kind_ns[size_t(kind)] += ns;
        if (why) {
            g_perf_waits++;
            if (diag::Perf()) LOG_FIRST(32, "render thread: the title waited %.2f ms for a %s", ns / 1e6, why);
        }
    }

    void Run() {
#ifdef __SWITCH__
        // core 2 preferred, any core of the process allowed (XBE_PIN placement may give the game thread core 2 too)
        u64 mask = 0;
        svcGetInfo(&mask, InfoType_CoreMask, CUR_PROCESS_HANDLE, 0);
        if (mask >> 2 & 1) svcSetThreadCoreMask(CUR_THREAD_HANDLE, 2, u32(mask));
#else
        using SetDescription = HRESULT(WINAPI*)(HANDLE, PCWSTR);
        if (auto set = reinterpret_cast<SetDescription>(
                reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription"))))
            set(GetCurrentThread(), L"render");
#endif
        for (;;) {
            Batch* b;
            {
                std::unique_lock lock(m_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) break;
                b = queue_.front();
                queue_.pop_front();
            }
            batch_frame_[b->id & (kFrameRing - 1)].store(presents_run_.load(), std::memory_order_release);
            for (const Cmd& c : b->cmds) Execute(*b, c);
            uint64_t id = b->id;
            b->cmds.clear();
            b->bytes.clear();
            b->vs.clear();
            b->ps.clear();
            {
                std::lock_guard lock(m_);
                free_.push_back(b);
                done_.store(id);
            }
            done_cv_.notify_all();
        }
        std::lock_guard lock(m_);
        exited_ = true;
        done_cv_.notify_all();
    }

    void Execute(const Batch& b, const Cmd& c) {
        const uint8_t* data = b.bytes.data() + c.data;
        switch (c.op) {
            case Op::State: {
                uint8_t* s = reinterpret_cast<uint8_t*>(&state_);
                for (const uint8_t* p = data; p < data + c.size;) {
                    uint32_t head[2];
                    std::memcpy(head, p, 8);
                    std::memcpy(s + head[0], p + 8, head[1]);
                    p += 8 + head[1];
                }
                if (c.vs >= 0) state_.vs = b.vs[size_t(c.vs)];
                if (c.ps >= 0) state_.ps = b.ps[size_t(c.ps)];
                break;
            }
            case Op::Draw: inner_->Draw(state_, c.a, c.b, c.c, c.d ? data : nullptr); break;
            case Op::Clear: inner_->Clear(state_, c.a, c.size ? data : nullptr, c.b, c.c, c.f, c.d); break;
            case Op::Copy: inner_->Copy(c.s1, c.s2, c.a, c.b, c.c); break;
            case Op::Present:
                inner_->Present(c.s1);
                presents_run_++;
                break;
            case Op::Sync: inner_->Sync(c.a, 0); break;
            case Op::RecordLate: inner_->RecordLate(c.a); break;
            case Op::Flush: inner_->Flush(); break;
            case Op::Gamma: inner_->SetGamma(data); break;
            case Op::ClearReport: inner_->ClearReport(); break;
            case Op::GetReport: inner_->GetReport(c.a); break;
        }
    }

    std::unique_ptr<Backend> inner_;
    std::vector<Range> ranges_;
    std::vector<Range> fixed_;

    // the title's thread
    struct HeaderCopy {
        uint32_t at = 0;
        uint64_t made = 0;  // ring_pos_ after it was written
        uint8_t bytes[kHeaderBytes];
    };
    std::unordered_map<uint32_t, HeaderCopy> headers_;
    // the last few lookups, in front of the maps (their elements stay put)
    template <typename T>
    struct Recent {
        uint32_t key = 0;
        T* value = nullptr;
    };
    std::array<Recent<HeaderCopy>, 64> recent_headers_{};
    std::array<Recent<uint64_t>, 64> recent_uses_{};
    uint32_t ring_ = 0;
    uint64_t ring_pos_ = 0;
    Batch* cur_ = nullptr;
    State sent_;  // the state as the render thread will have it
    std::unordered_map<uint32_t, uint64_t> last_use_;
    std::unordered_set<uint32_t> written_;  // surfaces drawn since their last read-back, by data address
    uint32_t last_written_ = ~0u;           // the physical address Written last put in written_
    std::unordered_set<uint64_t> logged_;   // perf diagnostic: read-backs named once, by address and kind
    uint64_t next_id_ = 0;
    uint64_t presents_ = 0, draws_ = 0, draw_ns_ = 0, state_ns_ = 0, header_copies_ = 0, index_bytes_ = 0;  // capture diagnostics
    uint64_t last_present_ = 0;

    // the render thread
    State state_;

    std::mutex m_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::deque<Batch*> queue_;
    std::vector<Batch*> free_;
    std::atomic<uint64_t> done_{0};
    std::atomic<uint64_t> presents_run_{0};                // presents the render thread ran
    std::unique_ptr<std::atomic<uint64_t>[]> batch_frame_{new std::atomic<uint64_t>[kFrameRing]()};
    bool stop_ = false;
    bool exited_ = false;
};

}  // namespace

std::unique_ptr<Backend> CreateThreadedBackend(std::unique_ptr<Backend> inner) {
    return std::make_unique<Threaded>(std::move(inner));
}

}  // namespace gpu
