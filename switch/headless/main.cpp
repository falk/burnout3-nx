// Switch build: the recompiled game, the SDK kernel and Black's host code on Horizon. build.sh links it with the
// deko3d renderer (burnout3.nro, the build to play); build-headless.sh with no renderer, to measure the game thread.
//
// sdmc:/switch/burnout3/burnout3.nro, the extracted disc in sdmc:/switch/burnout3/game.
// Run from the homebrew menu, or with: nxlink -s burnout3.nro [NAME=value ...]; the log streams back over nxlink when
// nxlink started it, else it goes to svcOutputDebugString, and always to sdmc:/switch/burnout3/logs.
// Arguments of the form NAME=value are set in the environment before boot, e.g. XBE_PIN=off or XBE_PIN=split:1,0x2C
// for the guest threads' placement (kernel/threads.cpp), B3_SKIP_FILMS=1 to end every film as it starts.
// The game quits through its own menu; the HOME menu closes it too.
#include <netinet/in.h>
#include <switch.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "app/app.h"
#include "core/input.h"
#include "core/log.h"
#include "image_symbols.h"
#include "kernel/kernel.h"
#include "switch_clocks.h"

namespace game { bool BenchHotCandidates(); }

namespace app {
void NX_NORETURN CleanExit();  // clean_exit.cpp
}

namespace {

constexpr const char* kRoot = "sdmc:/switch/burnout3";

// XBE_THREAD_DUMP=<seconds>: each guest thread's last indirect call and the code addresses near the top of its guest
// stack, to see where threads wait or spin. Written straight to stdout from a thread of its own: the logger's lock may
// be held by a guest thread that does not get to run.
void DumpThreads(int seconds) {
    std::thread([seconds] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(seconds));
            for (auto& t : kernel::Threads()) {
                if (!t->ctx) continue;
                const X86Context& c = *t->ctx;
                char line[400];
                int n = std::snprintf(line, sizeof(line), "dump: thread %X esp %08X icall %08X eax %08X ecx %08X stack",
                                      t->id, c.esp, c.last_icall, c.eax, c.ecx);
                // only a stack inside the console's 64 MB: a thread that has not run yet has none
                const bool stack_ok = c.esp >= 0x10000 && c.esp < 0x4000000 - 0x200;
                for (uint32_t off = 0; stack_ok && off < 0x200 && n < 360; off += 4) {
                    uint32_t v = kernel::R32(c.esp + off);
                    if (v >= 0x11000 && v < 0x36C000) n += std::snprintf(line + n, sizeof(line) - n, " +%X:%08X", off, v);
                }
                line[n++] = '\n';
                write(STDOUT_FILENO, line, size_t(n));
            }
        }
    }).detach();
}

}  // namespace

extern "C" {

// sockets only for nxlink: a launch from the menu (no nxlink host) logs to svcOutputDebugString
bool g_sockets = false;

void userAppInit(void) {
    if (__nxlink_host.s_addr && R_SUCCEEDED(socketInitializeDefault())) {
        g_sockets = true;
        if (nxlinkStdio() >= 0) return;
    }
    consoleDebugInit(debugDevice_SVC);
}

void userAppExit(void) {
    platform::CloseClocks();
    if (g_sockets) socketExit();
}

}  // extern "C"

int main(int argc, char** argv) {
    std::set_terminate([] {
        try {
            if (auto e = std::current_exception()) std::rethrow_exception(e);
            LOGE("terminate: no exception");
        } catch (const std::exception& e) {
            LOGE("terminate: %s", e.what());
        } catch (...) {
            LOGE("terminate: an exception of another type");
        }
        logx::Flush();
        std::_Exit(3);
    });
    for (int i = 1; i < argc; i++)
        if (std::strchr(argv[i], '=')) putenv(argv[i]);
    // sdmc:/switch/burnout3/env.txt: NAME=value lines as well, for launchers that pass no arguments (Ryujinx)
    if (FILE* f = std::fopen("sdmc:/switch/burnout3/env.txt", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            line[std::strcspn(line, "\r\n")] = 0;
            if (char* eq = std::strchr(line, '='); eq && eq != line && line[0] != '#') {
                *eq = 0;
                setenv(line, eq + 1, 0);
            }
        }
        std::fclose(f);
    }
#if defined(B3_ROI_BUILD) || defined(B3_PROF_BUILD)
    setenv("B3_SKIP_FILMS", "1", 0);
#endif
    chdir(kRoot);
    std::string game = std::string(kRoot) + "/game";
    setenv("GAME_DIR", game.c_str(), 1);

    logx::Init((app::ExeDir() / "logs").string(), app::kGame.log);
    LOGI("black: %s, log %s", game.c_str(), g_sockets ? "over nxlink" : "to the debug output");
    platform::ReportClocks();
#ifdef B3_ROI_BUILD
    if (const char* b = std::getenv("B3_BENCH_HOT"); b && *b == '1' && !game::BenchHotCandidates()) return 1;
#endif
    input::Init();
    app::Load();
    if (app::kGame.init) app::kGame.init();
    if (app::kGame.window) app::kGame.window(nullptr);

    // in one read: an ifstream reads 1 KB per fs request here, some 2800 for the image
    std::vector<uint8_t> xbe;
    if (FILE* f = std::fopen((game + "/" + IMAGE_XBE).c_str(), "rb")) {
        std::setvbuf(f, nullptr, _IONBF, 0);
        struct stat st {};
        if (fstat(fileno(f), &st) == 0 && st.st_size > 0) {
            xbe.resize(size_t(st.st_size));
            xbe.resize(std::fread(xbe.data(), 1, xbe.size(), f));
        }
        std::fclose(f);
    }
    if (xbe.empty()) {
        LOGE("no %s in %s", IMAGE_XBE, game.c_str());
        logx::Flush();
        return 1;
    }
    if (!kernel::Boot(std::move(xbe), IMAGE_XBE, game, (app::ExeDir() / "hdd").string())) {
        logx::Flush();
        return 1;
    }
    if (const char* dump = std::getenv("XBE_THREAD_DUMP")) DumpThreads(std::max(1, std::atoi(dump)));

    // B3_RUN_SECONDS=<n>: back to the homebrew menu after n seconds (clean_exit.cpp), for unattended test runs
    const char* run = std::getenv("B3_RUN_SECONDS");
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(run ? std::max(1, std::atoi(run)) : 0);
    // the main thread keeps the applet messages (focus, exit requests) flowing
    auto next_clocks = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (appletMainLoop()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        logx::Flush();
        if (std::chrono::steady_clock::now() >= next_clocks) {
            platform::ReportClocks();
            next_clocks = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        }
        if (run && std::chrono::steady_clock::now() >= end) app::CleanExit();
    }
    LOGI("black: exit requested");
    logx::Flush();
    std::_Exit(0);  // the guest threads never return
}
