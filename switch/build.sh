#!/bin/bash
# build.sh: burnout3.nro, the game drawn with deko3d (switch/deko3d), the Switch build to play
#   -s SDK folder (default ../XboxXBE-Recomp)   -g generated code of default.xbe   -o out folder (default out/switch-deko)
#   PROF=0: without the console profiler (in by default, idle unless B3_PROF=1 is passed)
#   RENDER_THREAD=1: the renderer on a thread of its own (src/d3d/render_thread.cpp), else on the title's D3D thread
# The headless build (build-headless.sh, null renderer) is the same program but this folder's renderer: generated code
# with clang, host code with devkitA64's GCC, linked by devkitA64 with libnx, deko3d and UAM (build-uam.sh first).
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
SDK=$REPO/../XboxXBE-Recomp; OUT=$REPO/out/switch-deko; GEN=$REPO/generated/game
while getopts s:g:o: opt; do
    case $opt in s) SDK=$OPTARG ;; g) GEN=$OPTARG ;; o) OUT=$OPTARG ;; *) sed -n 2,5p "$0"; exit 2 ;; esac
done
[ -f "$GEN/x86_func_mapping.cpp" ] || { echo "no generated code in $GEN: run switch/recomp.sh"; exit 1; }
: "${DEVKITPRO:?devkitPro is needed}"
UAM=${UAM:-$REPO/out/switch-deko/uam}
[ -f "$UAM/libuam.a" ] || { echo "no $UAM/libuam.a: run switch/build-uam.sh"; exit 1; }
SDK=$(realpath "$SDK"); GEN=$(realpath "$GEN"); mkdir -p "$OUT/gen"; OUT=$(realpath "$OUT")
A64=$DEVKITPRO/devkitA64; NX=$DEVKITPRO/libnx; H=$SDK/harness; RT=$SDK/runtime; SW=$REPO/switch/headless; DK=$REPO/switch/deko3d
GCCVER=$("$A64/bin/aarch64-none-elf-gcc" -dumpversion)
[ -z "$TOML" ] && [ -d "$REPO/out/switch-deko/toml/include/toml++" ] && TOML=$REPO/out/switch-deko/toml/include
TOML=${TOML:-$REPO/out/switch/tools/_deps/tomlplusplus-src/include}
[ -d "$TOML" ] || { echo "toml++ headers not found: set TOML to tomlplusplus/include"; exit 1; }
JOBS=${JOBS:-$(nproc)}
RTHREAD=${RENDER_THREAD:-1}; HOST=$OUT/host-rt$RTHREAD; mkdir -p "$HOST"
unset CC CXX AR CFLAGS CXXFLAGS CPPFLAGS LDFLAGS

COMMON="-std=gnu++20 -O2 -g ${EXTRA_CXXFLAGS:-} -fno-omit-frame-pointer -fPIE -ffunction-sections -fdata-sections -fno-strict-aliasing -ffp-contract=off -D__SWITCH__"
CLANG="clang++ --target=aarch64-none-elf -mcpu=cortex-a57 $COMMON -fno-math-errno -w -nostdinc++
    -isystem $A64/aarch64-none-elf/include/c++/$GCCVER -isystem $A64/aarch64-none-elf/include/c++/$GCCVER/aarch64-none-elf
    -isystem $A64/aarch64-none-elf/include"
GXX="$A64/bin/aarch64-none-elf-g++ -march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft $COMMON -fexceptions"
# PROF=0 leaves out the console sampling profiler (switch/headless/b3_prof.cpp); in, it starts only for B3_PROF=<mode>
[ "${PROF:-1}" = 1 ] && GXX="$GXX -DB3_PROF_BUILD"
HOSTINC="-I$REPO/src -I$RT -I$GEN -I$H -I$H/compat -I$SW -I$DK -I$NX/include -I$TOML -DXBE_RAW_VERTICES=1 -DXBE_RENDER_THREAD=$RTHREAD
    -DB3_DEKO3D -include $H/compat/windows.h -include $SW/b3_compat.h"
UAMINC="-I$UAM/src/source -I$UAM/src/mesa-imported"

# Cache both the compile configuration and headers, so verification flags and
# SDK/header edits cannot silently link stale objects from an earlier build.
gen_key=$({ printf '%s\n' "$CLANG" "$RT" "$GEN"; find "$RT/cpu" "$GEN" -maxdepth 1 -name '*.h' -type f -print0 | sort -z | xargs -0 sha256sum; } | sha256sum | cut -c1-16)
GENOBJ=$OUT/gen-$gen_key
host_key=$({ printf '%s\n' "$GXX" "$HOSTINC"; find "$RT" "$REPO/src" "$SW" "$DK" "$H" "$GEN" -type f \( -name '*.h' -o -name '*.inc' \) -print0 | sort -z | xargs -0 sha256sum; } | sha256sum | cut -c1-16)
HOST=$OUT/host-$host_key
mkdir -p "$GENOBJ" "$HOST"

# generated code: an object is rebuilt when its partition is newer
export CLANG RT GEN OUT GENOBJ
ls "$GEN"/x86_recomp.*.cpp "$GEN"/x86_func_mapping.cpp | xargs -P "$JOBS" -I{} sh -c '
    o="$GENOBJ/$(basename {} .cpp).o"; [ "$o" -nt {} ] && exit 0
    $CLANG -I"$RT" -I"$GEN" -c {} -o "$o.tmp" && mv "$o.tmp" "$o"'

# host code: the headless build's sources with this renderer for the null one
SRC="$(ls "$RT"/*/*.cpp | grep -Ev '/runtime/app/|/core/input.cpp|/gpu/(d3d12|render_thread).cpp|/audio/xaudio2.cpp|/xdk/(d3d8|dsound).cpp')
     $RT/app/game_files.cpp
     $(ls "$REPO"/src/*/*.cpp | grep -Ev '/src/(audio/xaudio2|d3d/d3d12|common/(monitor|cursor|mouse_keyboard|gyro|focus_pause|menu|game_files)).cpp' |
       { [ "$RTHREAD" = 1 ] && cat || grep -v /src/d3d/render_thread.cpp; })
     $H/app_headless.cpp $H/reference_renderer.cpp
     $(ls "$SW"/*.cpp) $(ls "$DK"/*.cpp)"
objs=(); compile_jobs=()
for f in $SRC; do
    o=$HOST/$(echo "${f#/}" | tr / _ | sed 's/\.cpp$/.o/'); objs+=("$o")
    # headers of this renderer count for its own sources
    if [ -s "$o" ] && [ "$o" -nt "$f" ]; then
        case $f in "$DK"/*) [ -z "$(find "$DK" -name '*.h' -newer "$o")" ] && continue ;; *) continue ;; esac
    fi
    echo "compile ${f#$REPO/}"
    extra=""; case $f in */shader_cache.cpp) extra=$UAMINC ;; esac
    ( $GXX $HOSTINC $extra -c "$f" -o "$o.tmp" && mv "$o.tmp" "$o" ) &
    compile_jobs+=("$!")
    [ "$(jobs -r | wc -l)" -ge "$JOBS" ] && wait -n
done
for j in "${compile_jobs[@]}"; do wait "$j" || exit 1; done

"$A64/bin/aarch64-none-elf-g++" -specs="$NX/switch.specs" -g -march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft -fPIE \
    -Wl,--gc-sections -Wl,--wrap=__cxa_throw -Wl,--wrap=threadCreate -Wl,--wrap=threadClose -Wl,-Map,"$OUT/burnout3.map" "${objs[@]}" "$GENOBJ"/*.o "$UAM/libuam.a" \
    -L"$NX/lib" -ldeko3d -lnx -o "$OUT/burnout3.elf"
"$DEVKITPRO/tools/bin/nacptool" --create "Burnout 3" "burnout3-nx" "0.1" "$OUT/burnout3.nacp" >/dev/null
# the profiler's function table of this very build in the romfs (switch/headless/b3_prof.cpp reads it there)
mkdir -p "$OUT/romfs" && python3 "$REPO/switch/tools/prof_sym.py" "$OUT/burnout3.elf" "$OUT/romfs/burnout3.sym" >/dev/null
"$DEVKITPRO/tools/bin/elf2nro" "$OUT/burnout3.elf" "$OUT/burnout3.nro" --nacp="$OUT/burnout3.nacp" --romfsdir="$OUT/romfs" >/dev/null
echo "built $OUT/burnout3.nro"
