// A clean return to the homebrew menu, for unattended test runs: B3_RUN_SECONDS=<n> ends the program after n
// seconds and the loader (hbloader) starts the next program in this same process, sphaira by default
// (B3_NEXT_NRO=<path> another one), so the next nxlink push needs no button presses.
//
// The loader reuses this process and its heap, so nothing may be left running or holding heap pages:
// - every thread but this one paused (guest threads cannot be ended from outside on Horizon), and its stack mirror
//   (svcMapMemory of heap pages, made by libnx's threadCreate) unmapped; the threads are found by wrapping libnx's
//   threadCreate and threadClose (-Wl,--wrap), which std::thread and pthreads go through too
// - the audio renderer closed (its memory pool is mapped by the service), after its thread ends on its own
// - the GPU idle and its device closed (nvdrv maps heap pages for every memory block)
// - the guest space's aliases of heap pages unmapped (kernel/memory.cpp)
// - sockets closed by userAppExit, the services by libnx's __appExit, then the return to the loader.
// The paused threads stay in the process (their count is logged against the process's limit). A watchdog ends
// the process (svcExitProcess, back to the HOME menu) if the return has not happened within 3 s.
#include <switch.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

#include "core/log.h"
#include "kernel/memory.h"

extern "C" {
Result __real_threadCreate(Thread* t, ThreadFunc entry, void* arg, void* stack_mem, size_t stack_sz, int prio, int cpuid);
Result __real_threadClose(Thread* t);
void __appExit(void);
void NX_NORETURN __nx_exit(Result rc, LoaderReturnFn retaddr);
}

namespace gpu {
__attribute__((weak)) void ShutdownForExit();  // deko3d_backend.cpp; absent in the headless build
}
namespace audio {
void StopOutput();  // switch_audio.cpp
}

namespace {

// the threads libnx made for this program, not yet closed
constexpr int kMaxThreads = 512;
Thread* g_threads[kMaxThreads];
int g_count = 0;
Mutex g_threads_lock = 0;

void Out(const char* fmt, ...) {
    char line[256];
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    size_t len = n < int(sizeof(line)) ? size_t(n) : sizeof(line) - 1;
    write(STDOUT_FILENO, line, len);
    svcOutputDebugString(line, len);
}

std::atomic<bool> g_done{false};
Thread g_watchdog;

void Watchdog(void*) {
    for (int i = 0; i < 300 && !g_done; i++) svcSleepThread(10'000'000);
    if (!g_done) {
        Out("exit: no clean return within 3 s, ending the process\n");
        svcExitProcess();
    }
}

}  // namespace

extern "C" {

Result __wrap_threadCreate(Thread* t, ThreadFunc entry, void* arg, void* stack_mem, size_t stack_sz, int prio, int cpuid) {
    Result rc = __real_threadCreate(t, entry, arg, stack_mem, stack_sz, prio, cpuid);
    if (R_SUCCEEDED(rc)) {
        mutexLock(&g_threads_lock);
        if (g_count < kMaxThreads) g_threads[g_count++] = t;
        mutexUnlock(&g_threads_lock);
    }
    return rc;
}

Result __wrap_threadClose(Thread* t) {
    Result rc = __real_threadClose(t);
    if (R_SUCCEEDED(rc)) {
        mutexLock(&g_threads_lock);
        for (int i = 0; i < g_count; i++)
            if (g_threads[i] == t) {
                g_threads[i] = g_threads[--g_count];
                break;
            }
        mutexUnlock(&g_threads_lock);
    }
    return rc;
}

}  // extern "C"

namespace app {

// on the main thread (the one the loader entered): never returns
void NX_NORETURN CleanExit() {
    LoaderReturnFn back = envGetExitFuncPtr();
    // SetThreadActivity, UnmapMemory, CloseHandle: the loader must allow them (any other SVC ends the process)
    if (!back || envIsNso() || !envIsSyscallHinted(0x32) || !envIsSyscallHinted(0x05)) {
        LOGW("exit: no return to a loader here, ending the process");
        logx::Flush();
        svcExitProcess();
    }
    LOGI("exit: returning to the homebrew menu");
    logx::Flush();

    // the audio thread ends on its own, while the guest threads (its mixer's callers) still run
    audio::StopOutput();

    if (R_SUCCEEDED(__real_threadCreate(&g_watchdog, Watchdog, nullptr, nullptr, 0x4000, 0x2B, -2)))
        threadStart(&g_watchdog);

    // every other thread paused, then given a moment to leave the cores it runs on
    Thread* self = threadGetSelf();
    mutexLock(&g_threads_lock);
    int paused = 0;
    for (int i = 0; i < g_count; i++) {
        Thread* t = g_threads[i];
        if (t == self || t == &g_watchdog) continue;
        if (R_SUCCEEDED(svcSetThreadActivity(t->handle, ThreadActivity_Paused))) paused++;
    }
    const int threads = g_count;
    mutexUnlock(&g_threads_lock);
    svcSleepThread(20'000'000);
    Out("exit: %d of %d threads paused\n", paused, threads);

    // the GPU: idle, then the device closed (and with it every GPU mapping of heap pages)
    if (gpu::ShutdownForExit) gpu::ShutdownForExit();

    // the guest space's aliases of heap pages
    bool mem_ok = mem::Shutdown();

    // the threads' stack mirrors (svcMapMemory of heap pages) and handles; the threads stay paused
    int unmapped = 0;
    for (int i = 0; i < g_count; i++) {
        Thread* t = g_threads[i];
        if (t == self || t == &g_watchdog || !t->stack_mirror) continue;
        MemoryInfo mi{};
        u32 pi = 0;
        if (R_SUCCEEDED(svcQueryMemory(&mi, &pi, u64(t->stack_mirror))) && mi.addr == u64(t->stack_mirror) &&
            mi.type == MemType_MappedMemory &&
            R_SUCCEEDED(svcUnmapMemory(t->stack_mirror, t->stack_mem, mi.size)))
            unmapped++;
        svcCloseHandle(t->handle);
    }
    g_count = 0;

    // how many threads the paused ones leave the next programs
    s64 used = 0, limit = 0;
    Handle rl = INVALID_HANDLE;
    u64 rlh = 0;
    if (envIsSyscallHinted(0x30) && envIsSyscallHinted(0x31) &&
        R_SUCCEEDED(svcGetInfo(&rlh, InfoType_ResourceLimit, INVALID_HANDLE, 0)) && (rl = Handle(rlh))) {
        svcGetResourceLimitCurrentValue(&used, rl, LimitableResource_Threads);
        svcGetResourceLimitLimitValue(&limit, rl, LimitableResource_Threads);
        svcCloseHandle(rl);
    }
    Out("exit: guest space %s, %d stacks unmapped, threads %lld of %lld in use\n", mem_ok ? "unmapped" : "NOT unmapped",
        unmapped, (long long)used, (long long)limit);

    // the next program: sphaira (its nxlink server), unless another is given
    const char* next = std::getenv("B3_NEXT_NRO");
    if (!next) next = "sdmc:/switch/sphaira/sphaira.nro";
    struct stat st {};
    if (envHasNextLoad() && stat(next, &st) == 0) {
        envSetNextLoad(next, next);
        Out("exit: next %s\n", next);
    }
    if (!mem_ok) {
        Out("exit: guest space still mapped, ending the process\n");
        svcExitProcess();
    }

    g_done = true;
    threadWaitForExit(&g_watchdog);
    __real_threadClose(&g_watchdog);
    Out("exit: back to the loader\n");
    __appExit();  // userAppExit (sockets), then the services
    __nx_exit(0, back);
}

}  // namespace app
