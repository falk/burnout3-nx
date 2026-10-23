// Logs where each C++ exception is thrown (linked with -Wl,--wrap=__cxa_throw): its type and caller, for addr2line
// against black-headless.elf (minus the load base logged with it)
#include <switch.h>

#include <stdexcept>
#include <system_error>
#include <typeinfo>

#include "core/log.h"

extern "C" {
[[noreturn]] void __real___cxa_throw(void*, std::type_info*, void (*)(void*));

[[noreturn]] void __wrap___cxa_throw(void* obj, std::type_info* type, void (*dtor)(void*)) {
    MemoryInfo info{};
    u32 page = 0;
    void* caller = __builtin_return_address(0);
    svcQueryMemory(&info, &page, reinterpret_cast<u64>(caller));
    const char* what = "";
    if (type && *type == typeid(std::system_error)) what = static_cast<std::system_error*>(obj)->what();
    LOGW("throw %s (%s) from %p (image at %llX, offset %llX)", type ? type->name() : "?", what, caller,
         (unsigned long long)info.addr, (unsigned long long)(reinterpret_cast<u64>(caller) - info.addr));
    // the callers above, through the frame records (x29 chain)
    auto* fp = static_cast<uintptr_t*>(__builtin_frame_address(0));
    for (int i = 0; i < 8 && fp && reinterpret_cast<u64>(fp) > 0x1000; i++) {
        u64 lr = fp[1];
        if (lr < info.addr || lr > info.addr + 0x10000000) break;
        LOGW("  at offset %llX", (unsigned long long)(lr - info.addr));
        fp = reinterpret_cast<uintptr_t*>(fp[0]);
    }
    __real___cxa_throw(obj, type, dtor);
}
}
