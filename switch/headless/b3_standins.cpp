// headless harness only: stand ins for Black's Windows only sources (menu, mouse and keyboard, monitor, focus pause,
// the D3D12 frame wait); no window, no keyboard, no panel
#include <switch.h>

#include <chrono>
#include <cstdlib>
#include <thread>

#include "audio/output.h"
#include "common/menu.h"
#include "common/monitor.h"
#include "common/mouse_keyboard.h"
#include "common/options.h"
#include "core/log.h"
#include "d3d/gpu_frame.h"
#include "image_symbols.h"
#include "kernel/kernel.h"
#include "xdk/hle.h"
#include "xdk/layout.h"

extern const char kImageSha1[] = "82d0588124e70102ee650f8969c0a56374476234";  // assets/default.xbe, USA

namespace app {
bool MenuOpen() { return false; }
bool Capturing() { return false; }
void ToggleMenu() {}
void DrawUi(double) {}
// The Xbox shows a swap at a vertical blank and holds the title in its next Swap until the one before is on screen;
// Black presents at interval 0x80000002 (two or immediate): a flip every second vertical blank, at the first one when
// the frame is late. Headless there is no display, so the game thread waits here on the vertical blank count
// (D3D_Device + VBlankCount, kept by the vertical blank thread) as it would in Swap, and threads below its priority
// run meanwhile. The playable renderer uses swapchain pacing.
void PaceFrame() {
#ifdef B3_DEKO3D
    return; // the swapchain controls presentation cadence
#else
    constexpr uint32_t kInterval = 2;
    const uint32_t count_at = SYM(D3D_Device) + XDK_CDevice_VBlankCount;
    static uint32_t shown = 0;  // the vertical blank that shows the previous swap
    // the previous swap is shown first
    while (int32_t(kernel::R32(count_at) - shown) < 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    uint32_t now = kernel::R32(count_at);
    shown = int32_t(now - shown) >= int32_t(kInterval) ? now : shown + kInterval;
#endif
}
void* CursorImage() { return nullptr; }
namespace rows {
void Defaults(int) {}
bool Enabled(const Row&) { return false; }
const Row* Page(int, size_t* count) {
    *count = 0;
    return nullptr;
}
std::string Value(const Row&, int) { return {}; }
}  // namespace rows
}  // namespace app

namespace game {
void UpdateSplashMonitor() {}
void UpdateWindowMonitor(void*) {}
// Out of focus (the HOME menu, a sleep, an overlay applet) the game thread waits, the sound pauses and the vertical
// blank count stands still (hooks.cpp), as for a window in the background on Windows. Horizon suspends the process
// in the HOME menu besides; this covers the applets that leave it running.
namespace {
// a library applet launch reports no focus state (0) until a change
bool InFocus() {
    AppletFocusState f = appletGetFocusState();
    return f == AppletFocusState_InFocus || f == AppletFocusState(0);
}
}  // namespace

void WaitWhileUnfocused() {
    if (InFocus()) return;
    LOGI("focus lost: paused");
    g_focus_paused = true;
    audio::SetPaused(true);
    while (!InFocus()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    audio::SetPaused(false);
    g_focus_pauses++;
    g_focus_paused = false;
    LOGI("focus back: resumed");
}
}  // namespace game

// the deko3d build waits for its own frames
#ifndef B3_DEKO3D
namespace gpu {
void WaitGpuFrame(uint64_t) {}
}  // namespace gpu
#endif

namespace input {
void StartMouse(void*) {}
void OnRawInput(void*) {}
void ReleaseCursor() {}
void Keyboard(Pad&) {}
bool TakeMouse(float&, float&) { return false; }
bool TakeGyro(float&, float&) { return false; }
void Bind(size_t, int, int) {}
void Unbind(size_t) {}
bool TakeDeletePress() { return false; }
void LoadBindings() {}
int PressedKey() { return 0; }
bool PanelKey(int) { return false; }
bool HudKey(int) { return false; }
std::string ButtonKey(int) { return {}; }
void SeenPad(const Pad&) {}
void SetMenu(bool) {}
bool KeyboardActive() { return false; }
bool CursorShown() { return false; }
bool Cursor(float&, float&, bool&) { return false; }
int TakeWheel() { return 0; }
bool LeftButton() { return false; }
void TapConfirm() {}
void SetTextEntry(bool) {}
void OnChar(unsigned) {}
std::u16string TakeChars() { return {}; }
}  // namespace input
