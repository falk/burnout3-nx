#pragma once

namespace platform {
// Main thread only: report actual clocks at startup and when clocks/mode change.
void ReportClocks();
void CloseClocks();
}  // namespace platform
