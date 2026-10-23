#pragma once
// headless harness only: Win32 names Black's portable sources use
#include <strings.h>
#ifdef __cplusplus
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define _stricmp strcasecmp
#define _strnicmp strncasecmp

enum : unsigned { WM_CREATE = 0x0001, WM_CLOSE = 0x0010, WM_CHAR = 0x0102, WM_SYSCOMMAND = 0x0112, WM_APP = 0x8000 };
enum : unsigned { SC_KEYMENU = 0xF100 };
enum : unsigned { MB_ICONERROR = 0x10, MB_ICONWARNING = 0x30, MB_YESNO = 0x4, MB_DEFBUTTON2 = 0x100, IDYES = 6, IDNO = 7 };

// quitting from the game's menu ends the run
inline int PostMessageW(void*, unsigned, uintptr_t, intptr_t) { std::_Exit(0); }
// no window to ask in: log the text, answer no
inline int MessageBoxW(void*, const wchar_t* text, const wchar_t*, unsigned) {
    std::fprintf(stderr, "MessageBox: %ls\n", text);
    return IDNO;
}
#endif
