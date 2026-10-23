// A crash on Horizon: the host registers, the guest registers of the thread and the guest memory they point at go
// to stdout (nxlink) and the debug log, then the process exits, back to the menu instead of Atmosphère's report.
// Runs on the faulting thread: no allocation, no locks, raw writes only.
#include <switch.h>

#include <cstdarg>
#include <cstdio>
#include <unistd.h>

#include "kernel/kernel.h"

namespace {

constexpr size_t kExceptionStackSize = 0x10000;
char g_line[512];

void Out(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(g_line, sizeof(g_line), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    size_t len = n < int(sizeof(g_line)) ? size_t(n) : sizeof(g_line) - 1;
    write(STDOUT_FILENO, g_line, len);
    svcOutputDebugString(g_line, len);
}

bool Readable(uint64_t addr, size_t size) {
    MemoryInfo info{};
    u32 page = 0;
    if (R_FAILED(svcQueryMemory(&info, &page, addr))) return false;
    return info.type != MemType_Unmapped && (info.perm & Perm_R) && addr + size <= info.addr + info.size;
}

// 32 bytes of guest memory at va, as dwords, when they are mapped
void DumpGuest(const char* name, uint32_t va) {
    uint8_t* base = kernel::g_base;
    if (!base || !Readable(uint64_t(base) + va, 32)) {
        Out("  %-4s %08X: unmapped\n", name, va);
        return;
    }
    const uint32_t* p = reinterpret_cast<const uint32_t*>(base + va);
    Out("  %-4s %08X: %08X %08X %08X %08X %08X %08X %08X %08X\n", name, va, p[0], p[1], p[2], p[3], p[4], p[5], p[6],
        p[7]);
}

}  // namespace

extern "C" {

alignas(16) u8 __nx_exception_stack[kExceptionStackSize];
u64 __nx_exception_stack_size = sizeof(__nx_exception_stack);

void __libnx_exception_handler(ThreadExceptionDump* ctx) {
    MemoryInfo text{};
    u32 page = 0;
    svcQueryMemory(&text, &page, reinterpret_cast<uintptr_t>(&__libnx_exception_handler));
    const uint64_t b = text.addr;
    Out("\ncrash: error %X, pc +%lX, lr +%lX, far %lX, esr %X, guest base %p\n", ctx->error_desc, ctx->pc.x - b,
        ctx->lr.x - b, ctx->far.x, ctx->esr, kernel::g_base);
    for (int i = 0; i < 29; i += 4)
        Out("  x%-2d %016lX %016lX %016lX %016lX\n", i, ctx->cpu_gprs[i].x, i + 1 < 29 ? ctx->cpu_gprs[i + 1].x : 0,
            i + 2 < 29 ? ctx->cpu_gprs[i + 2].x : 0, i + 3 < 29 ? ctx->cpu_gprs[i + 3].x : 0);
    // return addresses on the host stack, inside this module
    uint64_t fp = ctx->fp.x;
    for (int i = 0; i < 24 && fp && Readable(fp, 16); i++) {
        const uint64_t* frame = reinterpret_cast<const uint64_t*>(fp);
        if (frame[1] >= b && frame[1] < b + text.size) Out("  frame +%lX\n", frame[1] - b);
        fp = frame[0];
    }
    // the guest thread: registers as last stored to its context (a function may hold newer values in host registers)
    if (kernel::Thread* t = kernel::Current(); t && t->ctx) {
        const X86Context& g = *t->ctx;
        Out("guest thread %X: eax %08X ecx %08X edx %08X ebx %08X esp %08X ebp %08X esi %08X edi %08X icall %08X\n",
            t->id, g.eax, g.ecx, g.edx, g.ebx, g.esp, g.ebp, g.esi, g.edi, g.last_icall);
        const char* names[] = {"eax", "ecx", "edx", "ebx", "ebp", "esi", "edi"};
        const uint32_t regs[] = {g.eax, g.ecx, g.edx, g.ebx, g.ebp, g.esi, g.edi};
        for (int i = 0; i < 7; i++) DumpGuest(names[i], regs[i]);
        for (uint32_t off = 0; off < 0x80; off += 0x20) DumpGuest("esp", g.esp + off);
    }
    // the host registers that look like guest addresses
    for (int i = 0; i < 29; i++) {
        uint64_t v = ctx->cpu_gprs[i].x;
        if (v >= 0x10000 && v < 0x100000000ull) {
            char name[8];
            std::snprintf(name, sizeof(name), "x%d", i);
            DumpGuest(name, uint32_t(v));
        }
    }
    _exit(1);
}

}  // extern "C"
