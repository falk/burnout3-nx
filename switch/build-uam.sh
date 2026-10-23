#!/bin/bash
# build-uam.sh: libuam.a, the deko3d shader compiler (GLSL -> DKSH) as a library for the renderer build
#   -o out folder (default out/switch-deko/uam)   UAM_REPO: git source (default github devkitPro/uam)
#   HOST=1 builds it for this machine instead (prebaking shader caches on the host)
# UAM 5a5afc2 with switch/deko3d/uam-library.patch (OutputDkshToMemory, pthread locks for the shared GLSL caches, Maxwell
# scheduling fixes; from the KisakCOD Switch port). Needs bison, flex, python3 with mako.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$REPO/out/switch-deko/uam
while getopts o: opt; do
    case $opt in o) OUT=$OPTARG ;; *) sed -n 2,6p "$0"; exit 2 ;; esac
done
REV=5a5afc2bae8b55409ab36ba45be63fcb73f68993
UAM_REPO=${UAM_REPO:-https://github.com/devkitPro/uam}
PATCH=$REPO/switch/deko3d/uam-library.patch
mkdir -p "$OUT"; OUT=$(realpath "$OUT"); SRC=$OUT/src; GEN=$OUT/gen
JOBS=${JOBS:-$(nproc)}

if [ ! -d "$SRC/.git" ]; then
    git clone -q "$UAM_REPO" "$SRC"
    git -C "$SRC" checkout -q "$REV"
fi
git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null || { git -C "$SRC" checkout -q -- .; git -C "$SRC" apply "$PATCH"; }

# generated parsers and expression tables, laid out as the lexers include them
G=$SRC/mesa-imported/glsl
mkdir -p "$GEN/glsl/glcpp"
[ -f "$GEN/glsl/glsl_parser.cpp" ] || (cd "$GEN/glsl" && bison -o glsl_parser.cpp -p _mesa_glsl_ --defines=glsl_parser.h "$G/glsl_parser.yy")
[ -f "$GEN/glsl/glsl_lexer.cpp" ] || (cd "$GEN/glsl" && flex -o glsl_lexer.cpp "$G/glsl_lexer.ll")
[ -f "$GEN/glsl/glcpp/glcpp-parse.c" ] || (cd "$GEN/glsl/glcpp" && bison -o glcpp-parse.c -p glcpp_parser_ --defines=glcpp-parse.h "$G/glcpp/glcpp-parse.y")
[ -f "$GEN/glsl/glcpp/glcpp-lex.c" ] || (cd "$GEN/glsl/glcpp" && flex -o glcpp-lex.c "$G/glcpp/glcpp-lex.l")
for k in enum:ir_expression_operation.h constant:ir_expression_operation_constant.h strings:ir_expression_operation_strings.h; do
    [ -f "$GEN/glsl/${k#*:}" ] || python3 "$G/ir_expression_operation.py" "${k%%:*}" > "$GEN/glsl/${k#*:}"
done

# sources: the quoted names in each directory's meson.build that exist, without the command line front end
SRCS=""
for d in source mesa-imported/codegen mesa-imported/tgsi mesa-imported/util mesa-imported/cso_cache mesa-imported/glsl \
         mesa-imported/glsl/glcpp mesa-imported/compiler mesa-imported/program mesa-imported/state_tracker mesa-imported/main; do
    for n in $(grep -oE "'[A-Za-z0-9_./-]+\.(cpp|c)'" "$SRC/$d/meson.build" | tr -d "'"); do
        [ -f "$SRC/$d/$n" ] && [ "$d/$n" != source/main.cpp ] && SRCS="$SRCS $SRC/$d/$n"
    done
done
SRCS="$SRCS $GEN/glsl/glsl_parser.cpp $GEN/glsl/glsl_lexer.cpp $GEN/glsl/glcpp/glcpp-parse.c $GEN/glsl/glcpp/glcpp-lex.c"

unset CC CXX AR CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
if [ "${HOST:-0}" = 1 ]; then
    CC="gcc"; CXX="g++"; AR=ar; ARCH="-fPIC"; LIB=$OUT/libuam-host.a; OBJ=$OUT/obj-host
else
    : "${DEVKITPRO:?devkitPro is needed}"
    A64=$DEVKITPRO/devkitA64/bin
    CC="$A64/aarch64-none-elf-gcc"; CXX="$A64/aarch64-none-elf-g++"; AR="$A64/aarch64-none-elf-ar"
    ARCH="-march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft -ftls-model=local-exec -fPIE -D__SWITCH__"
    LIB=$OUT/libuam.a; OBJ=$OUT/obj
fi
mkdir -p "$OBJ"
FLAGS="$ARCH -O3 -DNDEBUG -ffunction-sections -fdata-sections -DPACKAGE_STRING=\"uam-1.1.0\" -DDESKTOP -D_USE_MATH_DEFINES -D_GNU_SOURCE
       -DHAVE_POSIX_MEMALIGN -D_FILE_OFFSET_BITS=64 -I$GEN -I$GEN/glsl -I$GEN/glsl/glcpp -I$SRC -I$SRC/mesa-imported
       -I$SRC/mesa-imported/glsl -I$SRC/mesa-imported/glsl/glcpp -w"
objs=()
for f in $SRCS; do
    o=$OBJ/$(echo "${f#$OUT/}" | tr / _).o; objs+=("$o")
    [ "$o" -nt "$f" ] && continue
    case $f in
        *.c) $CC $FLAGS -std=c99 -Werror=implicit-function-declaration -c "$f" -o "$o" & ;;
        *) $CXX $FLAGS -std=c++11 -c "$f" -o "$o" & ;;
    esac
    [ "$(jobs -r | wc -l)" -ge "$JOBS" ] && wait -n
done
wait
rm -f "$LIB"; $AR rcs "$LIB" "${objs[@]}"
echo "built $LIB"
