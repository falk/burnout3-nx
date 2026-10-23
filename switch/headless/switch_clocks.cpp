// Same gameplay APM policy as the CoD4 Switch port. This requests an official
// handheld operating point; it does not use FastLoad (which throttles the GPU).
#include <switch.h>

#include "core/log.h"
#include "switch_clocks.h"

#ifdef B3_DEKO3D
extern "C" {
u32 __nx_applet_type = AppletType_Application;
// Normal = handheld; Boost = docked. Zero keeps Horizon's docked policy.
u32 __nx_applet_PerformanceConfiguration[2] = {0x92220007u, 0};
}
#endif

namespace platform {
namespace {

struct Sample {
    u32 cpu = 0, gpu = 0, memory = 0;
    Result rc = 0;
    int operation = -1, performance = -1;
};

bool g_initialized = false, g_have_sample = false;
Result g_init_rc = 0;
Sample g_last;

Result ReadRate(PcvModuleId module, PcvModule legacy, u32* rate) {
    if (!hosversionAtLeast(8, 0, 0)) return pcvGetClockRate(legacy, rate);
    ClkrstSession session;
    Result rc = clkrstOpenSession(&session, module, 3);
    if (R_FAILED(rc)) return rc;
    rc = clkrstGetClockRate(&session, rate);
    clkrstCloseSession(&session);
    return rc;
}

}  // namespace

void ReportClocks() {
    if (!g_initialized) {
        g_initialized = true;
        g_init_rc = hosversionAtLeast(8, 0, 0) ? clkrstInitialize() : pcvInitialize();
#ifdef B3_DEKO3D
        LOGI("clocks: gameplay APM request handheld 92220007, docked Horizon default");
#endif
    }
    Sample s;
    s.operation = int(appletGetOperationMode());
    s.performance = int(appletGetPerformanceMode());
    s.rc = g_init_rc;
    if (R_SUCCEEDED(s.rc)) s.rc = ReadRate(PcvModuleId_CpuBus, PcvModule_CpuBus, &s.cpu);
    if (R_SUCCEEDED(s.rc)) s.rc = ReadRate(PcvModuleId_GPU, PcvModule_GPU, &s.gpu);
    if (R_SUCCEEDED(s.rc)) s.rc = ReadRate(PcvModuleId_EMC, PcvModule_EMC, &s.memory);
    if (g_have_sample && s.cpu == g_last.cpu && s.gpu == g_last.gpu && s.memory == g_last.memory &&
        s.rc == g_last.rc && s.operation == g_last.operation && s.performance == g_last.performance)
        return;
    g_have_sample = true;
    g_last = s;
    const char* mode = s.operation == AppletOperationMode_Handheld ? "handheld" :
                       s.operation == AppletOperationMode_Console ? "docked" : "unknown";
    if (R_FAILED(s.rc) || !s.cpu || !s.gpu || !s.memory) {
        LOGW("clocks: actual rates unavailable rc=%X rates=%u/%u/%u Hz mode=%s perf=%d", s.rc,
             s.cpu, s.gpu, s.memory, mode, s.performance);
        return;
    }
    LOGI("clocks: actual CPU %.1f MHz GPU %.1f MHz memory %.1f MHz mode=%s perf=%d",
         s.cpu / 1e6, s.gpu / 1e6, s.memory / 1e6, mode, s.performance);
}

void CloseClocks() {
    if (g_initialized && R_SUCCEEDED(g_init_rc)) {
        if (hosversionAtLeast(8, 0, 0)) clkrstExit();
        else pcvExit();
    }
    g_initialized = false;
    g_have_sample = false;
}

}  // namespace platform
