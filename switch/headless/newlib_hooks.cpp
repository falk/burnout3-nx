// newlib hooks this libnx does not provide, and its exit replaced
#include <switch.h>

#include <cstdio>

// libsysbase's pthread_detach (std::thread::detach too) calls __syscall_thread_detach, which libnx leaves out: without
// it every detach fails with ENOSYS, and std::thread throws. Detached threads here are the runtime's, which run for the
// life of the process, and the guest's; the block of one that ends is not reclaimed.
extern "C" int __syscall_thread_detach(struct __pthread_t*) { return 0; }

// Every way out (return from main, exit, _Exit, abort, the crash handler's _exit) ends here. libnx's own __libnx_exit
// returns to the loader (hbloader) inside this process, with the guest threads still running and the guest's code
// memory still mapped over heap pages; the loader then unmaps this program under those threads and starts hbmenu on
// that heap. The process ends instead, and Horizon releases its threads, memory, files and sockets.
extern "C" void NX_NORETURN __libnx_exit(int) {
    std::fflush(stdout);
    std::fflush(stderr);
    svcExitProcess();
}
