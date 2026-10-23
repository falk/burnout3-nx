// B3_PROF=1 (or sdmc:/switch/burnout3/prof_on holding 1): a sampling profiler of the game thread on the console.
// A thread of its own (core 0) pauses the guest thread about every millisecond (svcSetThreadActivity), reads its
// registers (svcGetThreadContext3) and the return addresses of a few frames (the x29 chain, while the thread is still
// paused), lets it run again and counts the sample against the function holding each address. B3_PROF=2 samples
// every guest thread; B3_PROF_HZ sets the rate (default 1000).
//
// Frames: the D3D device's swap count marks the end of each frame. Per frame the profiler takes the wall time and the
// game thread's CPU time (svcGetInfo ThreadTickCount); every 10 s their percentiles go to the log with each guest
// thread's share of the CPU. A frame longer than B3_PROF_SLOW ms (default 18) logs its own top functions.
//
// Function names come from sdmc:/switch/burnout3/burnout3.sym (switch/tools/prof_sym.py, made from burnout3.elf of the same
// build); without it, or for another build, the counts are by raw text offset. Every 10 s the top functions of the
// last 10 s by self and by inclusive samples go to the log, followed by the whole-run histograms.
// switch/tools/prof_report.py names the offsets offline using the matching ELF; no SD writes.
#include <switch.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/log.h"
#include "image_symbols.h"
#include "kernel/kernel.h"
#include "xdk/hle.h"
#include "xdk/layout.h"

extern "C" void b3_prof_anchor() {}
extern "C" int __system_argc;
extern "C" char** __system_argv;

namespace {

constexpr int kFrames = 12;  // return addresses walked per sample
constexpr uint32_t kSwapCount = SYM(D3D_Device) + XDK_CDevice_SwapCount;
constexpr uint32_t kNone = 0xFFFFFFFFu;

struct Sym {
    uint32_t start;
    const char* name;
};

std::vector<Sym> g_syms;
std::vector<char> g_names;
uint64_t g_text = 0, g_text_size = 0;

uint32_t Off(uint64_t pc) { return pc >= g_text && pc < g_text + g_text_size ? uint32_t(pc - g_text) : kNone; }

// burnout3.sym: "BSYM", count, the anchor's offset, then count (start, name offset) pairs and the names; the copy in
// the NRO's romfs (build.sh puts it there) always matches the build, the SD card's is the fallback
void LoadSyms() {
    FILE* f = nullptr;
    if (R_SUCCEEDED(romfsInit())) f = std::fopen("romfs:/burnout3.sym", "rb");
    if (!f) f = std::fopen("sdmc:/switch/burnout3/burnout3.sym", "rb");
    if (!f) {
        LOGW("prof: no burnout3.sym, counting by text offset");
        return;
    }
    uint32_t hdr[3] = {};
    std::vector<uint32_t> pairs;
    bool ok = std::fread(hdr, 4, 3, f) == 3 && hdr[0] == 0x4D595342u;
    uint32_t anchor = Off(reinterpret_cast<uint64_t>(&b3_prof_anchor));
    if (ok && hdr[2] != anchor) {
        LOGW("prof: burnout3.sym is of another build (anchor %X, this build %X)", hdr[2], anchor);
        ok = false;
    }
    if (ok) {
        pairs.resize(size_t(hdr[1]) * 2);
        ok = std::fread(pairs.data(), 4, pairs.size(), f) == pairs.size();
    }
    if (ok) {
        long here = std::ftell(f);
        std::fseek(f, 0, SEEK_END);
        long end = std::ftell(f);
        std::fseek(f, here, SEEK_SET);
        g_names.resize(size_t(end - here) + 1);
        ok = std::fread(g_names.data(), 1, size_t(end - here), f) == size_t(end - here);
    }
    std::fclose(f);
    if (!ok) {
        LOGW("prof: burnout3.sym unreadable, counting by text offset");
        return;
    }
    g_syms.reserve(hdr[1]);
    for (uint32_t i = 0; i < hdr[1]; i++) g_syms.push_back({pairs[i * 2], g_names.data() + pairs[i * 2 + 1]});
    LOGI("prof: %zu symbols", g_syms.size());
}

// the function holding a text offset: its start (the key the counts use)
uint32_t FuncOf(uint32_t off) {
    if (off == kNone) return off;
    if (g_syms.empty()) return off & ~3u;
    auto it = std::upper_bound(g_syms.begin(), g_syms.end(), off, [](uint32_t o, const Sym& s) { return o < s.start; });
    return it == g_syms.begin() ? kNone : (it - 1)->start;
}

const char* NameOf(uint32_t start) {
    if (start == kNone) return "(outside the module: svc stubs, other)";
    auto it = std::lower_bound(g_syms.begin(), g_syms.end(), start, [](const Sym& s, uint32_t o) { return s.start < o; });
    return it != g_syms.end() && it->start == start ? it->name : "?";
}

using Map = std::unordered_map<uint32_t, uint32_t>;

struct Counts {
    uint64_t samples = 0;
    Map self, incl, pcs;
    void Clear() {
        samples = 0;
        self.clear();
        incl.clear();
        pcs.clear();
    }
};

struct Target {
    uint32_t id;
    Handle handle;
    uint64_t stack_lo = 0, stack_hi = 0;
    Counts total, window, frame;  // the whole run, the last 10 s, the current frame (game thread)
};

void StackOf(Target& t, uint64_t sp) {
    if (sp >= t.stack_lo && sp < t.stack_hi) return;
    MemoryInfo mi{};
    u32 pi = 0;
    if (R_SUCCEEDED(svcQueryMemory(&mi, &pi, sp))) {
        t.stack_lo = mi.addr;
        t.stack_hi = mi.addr + mi.size;
    }
}

void Count(Counts& c, uint32_t pc, uint32_t self, const uint32_t* funcs, int n) {
    c.samples++;
    c.pcs[pc == kNone ? pc : pc & ~3u]++;
    c.self[self]++;
    for (int i = 0; i < n; i++) c.incl[funcs[i]]++;
}

// one sample of a thread: registers and return addresses while it is paused, counted after it runs again
bool Sample(Target& t, bool per_frame) {
    if (R_FAILED(svcSetThreadActivity(t.handle, ThreadActivity_Paused))) return false;
    ThreadContext ctx;
    Result rc = svcGetThreadContext3(&ctx, t.handle);
    uint64_t addrs[kFrames + 2];
    int n = 0;
    if (R_SUCCEEDED(rc)) {
        addrs[n++] = ctx.pc.x;
        addrs[n++] = ctx.lr;  // a leaf function keeps its return address in x30
        uint64_t fp = ctx.fp;
        uint64_t lo = t.stack_lo, hi = t.stack_hi;
        if (!(ctx.sp >= lo && ctx.sp < hi)) lo = hi = 0;  // the bounds are looked up again below, after the resume
        for (int i = 0; i < kFrames && fp >= lo && fp + 16 <= hi && (fp & 7) == 0; i++) {
            const uint64_t* rec = reinterpret_cast<const uint64_t*>(fp);
            addrs[n++] = rec[1];
            if (rec[0] <= fp) break;
            fp = rec[0];
        }
    }
    svcSetThreadActivity(t.handle, ThreadActivity_Runnable);
    if (R_FAILED(rc)) return false;
    StackOf(t, ctx.sp);
    uint32_t pc = Off(addrs[0]);
    uint32_t self = FuncOf(pc);
    // inclusive: each function once per sample (a return address points after its call: one instruction back)
    uint32_t funcs[kFrames + 2];
    int nf = 0;
    for (int i = 0; i < n; i++) {
        uint32_t f = i == 0 ? self : FuncOf(Off(addrs[i] - 4));
        if (f == kNone && i) continue;
        if (std::find(funcs, funcs + nf, f) != funcs + nf) continue;
        funcs[nf++] = f;
    }
    Count(t.total, pc, self, funcs, nf);
    Count(t.window, pc, self, funcs, nf);
    if (per_frame) Count(t.frame, pc, self, funcs, nf);
    return true;
}

void Top(const char* what, const Map& m, uint64_t total, int count) {
    std::vector<std::pair<uint32_t, uint32_t>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
    for (int i = 0; i < count && i < int(v.size()); i++)
        LOGI("prof:   %s %5.1f%%  +%06X %s", what, 100.0 * v[i].second / double(total), v[i].first, NameOf(v[i].first));
}

// Stream cumulative histograms in bounded lines. Keep profiling off the SD card;
// prof_report.py resolves raw offsets using the matching ELF on the host.
void WriteTable(std::vector<Target>& targets) {
    for (Target& t : targets) {
        if (!t.total.samples) continue;
        LOGI("profdata: T %X %llu", t.id, (unsigned long long)t.total.samples);
        auto stream = [](char kind, const Map& values) {
            char line[512];
            size_t used = 0;
            for (auto& [address, count] : values) {
                used += size_t(std::snprintf(line + used, sizeof(line) - used, "%c %X %u ", kind, address, count));
                if (used > sizeof(line) - 40) {
                    LOGI("profdata: %s", line);
                    used = 0;
                }
            }
            if (used) LOGI("profdata: %s", line);
        };
        stream('S', t.total.self);
        stream('I', t.total.incl);
        stream('P', t.total.pcs);
    }
}

uint64_t ThreadTicks(Handle h) {
    u64 v = 0;
    return R_SUCCEEDED(svcGetInfo(&v, InfoType_ThreadTickCount, h, UINT64_MAX)) ? v : 0;
}

float Pct(std::vector<float>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size())))];
}

struct Frames {
    uint32_t last_swap = 0;
    uint64_t last_tick = 0, last_cpu = 0;
    std::vector<float> wall, cpu;
    int slow_logged = 0;
};

char g_mode = 0;  // '1' the game thread, '2' every guest thread; set before the thread starts

void Run(void*) {
    svcSleepThread(3'000'000'000ll);  // the log and the guest threads come up in main
    const char* mode = &g_mode;
    const bool all = *mode == '2';
    const char* hz_env = std::getenv("B3_PROF_HZ");
    const int hz = hz_env ? std::max(50, std::min(4000, std::atoi(hz_env))) : 1000;
    const char* slow_env = std::getenv("B3_PROF_SLOW");
    const float slow_ms = slow_env ? float(std::atof(slow_env)) : 18.0f;
    MemoryInfo mi{};
    u32 pi = 0;
    svcQueryMemory(&mi, &pi, reinterpret_cast<uint64_t>(&b3_prof_anchor));
    g_text = mi.addr;
    g_text_size = mi.size;
    LoadSyms();
    LOGI("prof: sampling %s at %d Hz, frames over %.0f ms logged, text %lX size %lX", all ? "every guest thread" : "the game thread",
         hz, slow_ms, g_text, g_text_size);
    std::vector<Target> targets;
    std::unordered_map<uint32_t, std::pair<Handle, uint64_t>> cpu_of;  // guest thread: handle, ticks at the last report
    Frames fr;
    const uint64_t freq = armGetSystemTickFreq();
    uint64_t start = armGetSystemTick(), last_dump = start, last_scan = 0;
    for (;;) {
        uint64_t now = armGetSystemTick();
        if (now - last_scan > freq) {  // threads come and go: look again each second
            last_scan = now;
            for (auto& t : kernel::Threads()) {
                if (!t->host_thread || !t->ctx) continue;
                if (!cpu_of.count(t->id)) cpu_of[t->id] = {Handle(t->host_thread), ThreadTicks(Handle(t->host_thread))};
                if (!all && t->id != 0x14) continue;
                bool known = std::any_of(targets.begin(), targets.end(), [&](Target& x) { return x.id == t->id; });
                if (!known) targets.push_back({t->id, Handle(t->host_thread)});
            }
        }
        Target* game = nullptr;
        for (Target& t : targets) {
            Sample(t, t.id == 0x14);
            if (t.id == 0x14) game = &t;
        }
        // the frame ends when the device's swap count moves (the Swap on the game thread)
        if (game && kernel::g_base) {
            uint32_t swap = kernel::R32(kSwapCount);
            if (swap != fr.last_swap) {
                uint64_t cpu = ThreadTicks(game->handle);
                if (fr.last_tick) {
                    float wall = float(double(now - fr.last_tick) * 1000.0 / double(freq));
                    float busy = float(double(cpu - fr.last_cpu) * 1000.0 / double(freq));
                    fr.wall.push_back(wall);
                    fr.cpu.push_back(busy);
                    if ((wall > slow_ms || busy > slow_ms) && fr.slow_logged < 8 && game->frame.samples) {
                        fr.slow_logged++;
                        LOGI("prof: slow frame %u: %.1f ms, game thread CPU %.1f ms, %llu samples", swap, wall, busy,
                             (unsigned long long)game->frame.samples);
                        Top("frame incl", game->frame.incl, game->frame.samples, 10);
                        Top("frame self", game->frame.self, game->frame.samples, 6);
                    }
                }
                game->frame.Clear();
                fr.last_swap = swap;
                fr.last_tick = now;
                fr.last_cpu = cpu;
            }
        }
        if (now - last_dump > 10 * freq) {
            double secs = double(now - last_dump) / double(freq);
            LOGI("prof: %.0f s, %zu frames: wall ms p50 %.2f p90 %.2f p99 %.2f max %.2f | game thread CPU ms p50 %.2f p90 %.2f "
                 "p99 %.2f max %.2f",
                 double(now - start) / double(freq), fr.wall.size(), Pct(fr.wall, 0.5), Pct(fr.wall, 0.9), Pct(fr.wall, 0.99),
                 fr.wall.empty() ? 0.0f : fr.wall.back(), Pct(fr.cpu, 0.5), Pct(fr.cpu, 0.9), Pct(fr.cpu, 0.99),
                 fr.cpu.empty() ? 0.0f : fr.cpu.back());
            std::string shares;
            for (auto& [id, hc] : cpu_of) {
                uint64_t ticks = ThreadTicks(hc.first);
                if (ticks < hc.second) continue;
                char buf[32];
                std::snprintf(buf, sizeof buf, " %X %.1f%%", id, 100.0 * double(ticks - hc.second) / double(freq) / secs);
                shares += buf;
                hc.second = ticks;
            }
            LOGI("prof: CPU per guest thread:%s", shares.c_str());
            fr.wall.clear();
            fr.cpu.clear();
            fr.slow_logged = 0;
            for (Target& t : targets) {
                if (!t.window.samples) continue;
                LOGI("prof: thread %X, last %.0f s: %llu samples", t.id, secs,
                     (unsigned long long)t.window.samples);
                Top("self", t.window.self, t.window.samples, 25);
                Top("incl", t.window.incl, t.window.samples, 25);
                t.window.Clear();
            }
            WriteTable(targets);
            last_dump = armGetSystemTick();
        }
        svcSleepThread(1'000'000'000ll / hz);
    }
}

// B3_PROF=<mode> among the NAME=value arguments (argv is built before the constructors run), else the file
// sdmc:/switch/burnout3/prof_on holding the mode; no thread at all otherwise
char ProfMode() {
#if !defined(B3_PROF_BUILD) || defined(B3_ROI_BUILD)
    return 0;
#else
    for (int i = 1; i < __system_argc; i++)
        if (__system_argv[i] && std::strncmp(__system_argv[i], "B3_PROF=", 8) == 0) return __system_argv[i][8];
    char m = 0;
    if (FILE* f = std::fopen("sdmc:/switch/burnout3/prof_on", "r")) {
        m = char(std::fgetc(f));
        std::fclose(f);
    }
    return m;
#endif
}

struct Starter {
    Starter() {
        g_mode = ProfMode();
        if (g_mode != '1' && g_mode != '2') return;
        if (!envIsSyscallHinted(0x32) || !envIsSyscallHinted(0x33)) {
            g_mode = 0;  // logged once the log is up
            return;
        }
        // the profiler runs until the process exits: core 0, above the guest threads so the samples come on time;
        // it sleeps between them
        static Thread thread;
        if (R_SUCCEEDED(threadCreate(&thread, Run, nullptr, nullptr, 0x10000, 0x21, 0))) threadStart(&thread);
    }
} g_starter;

}  // namespace
