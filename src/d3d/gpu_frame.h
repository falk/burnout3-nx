#pragma once
#include <cstdint>

namespace gpu {
// until the GPU has drawn present number n, counted from 0; any thread
void WaitGpuFrame(uint64_t n);
#ifdef __SWITCH__
// the deko3d backend (switch/deko3d): present number n drawn, without waiting; true when it never will be (too old)
bool GpuFrameDone(uint64_t n);
// draws read eligible guest memory in place: a resource is free for the CPU only once the GPU is past it
bool ZeroCopyActive();
// the present (+1) that last drew the vertex buffer whose data is at guest address va in place, 0 none (a hash slot:
// a collision only waits longer)
uint64_t InPlaceFrame(uint32_t va);
// present n drew anything in place
bool FrameReadInPlace(uint64_t n);
#endif
}  // namespace gpu
