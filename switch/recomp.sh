#!/bin/bash
# recomp.sh: generated/game, the C++ of assets/default.xbe (USA), for the Switch builds; run once, and again when the
# SDK's recompiler or config/game.toml changes. Builds xbe_recomp for this machine in out/switch/tools (also the
# toml++ headers the Switch builds use).   -s SDK folder (default ../XboxXBE-Recomp)
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
SDK=$REPO/../XboxXBE-Recomp
while getopts s: opt; do
    case $opt in s) SDK=$OPTARG ;; *) sed -n 2,4p "$0"; exit 2 ;; esac
done
SDK=$(realpath "$SDK"); T=$REPO/out/switch/tools
[ -f "$REPO/assets/default.xbe" ] || { echo "no assets/default.xbe: put your own USA game files in assets/"; exit 1; }
unset CC CXX CFLAGS CXXFLAGS LDFLAGS
cmake -S "$REPO/switch/tools" -B "$T" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSDK="$SDK" >/dev/null
cmake --build "$T"
cd "$REPO" && "$T/xbe_recomp" config/game.toml --xdk "$SDK/xdk"
echo "generated $REPO/generated/game"
