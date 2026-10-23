#!/bin/bash
# docker/build.sh: burnout3.nro built in Docker, so only Docker is needed on this machine. Needs your own game's
# default.xbe in assets/ (USA, sha256 9f497acd...). The image (burnout3-nx-builder) is made on first use and again
# when the pinned SDK commit changes; the recompiled code and objects stay in generated/ and out/ for the next build.
#   PROF=0, RENDER_THREAD=0 and JOBS are passed on to switch/build.sh
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$REPO/assets/default.xbe" ] || { echo "no assets/default.xbe: put the files of your own copy of the game in assets/"; exit 1; }
SDK_REV=$(sed -n 's/.*GIT_TAG \([a-f0-9]\{40\}\)).*/\1/p' "$REPO/CMakeLists.txt")
[ ${#SDK_REV} -eq 40 ] || { echo "no pinned SDK commit (GIT_TAG) in CMakeLists.txt"; exit 1; }
IMAGE=burnout3-nx-builder:${SDK_REV:0:12}
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" --build-arg SDK_REV="$SDK_REV" -f "$REPO/docker/Dockerfile" "$REPO"
docker run --rm -u "$(id -u):$(id -g)" -v "$REPO:/work" \
    -e PROF -e RENDER_THREAD -e JOBS "$IMAGE" bash -c '
    set -e
    # recompile when there is no recompiler or generated code yet, or default.xbe or the config is newer than it
    if [ ! -x out/switch/tools/xbe_recomp ] || [ ! -f generated/game/x86_func_mapping.cpp ] || [ assets/default.xbe -nt generated/game/x86_func_mapping.cpp ] ||
       [ config/game.toml -nt generated/game/x86_func_mapping.cpp ]; then
        switch/recomp.sh -s /opt/xbe-sdk
    fi
    UAM=/opt/uam switch/build.sh -s /opt/xbe-sdk'
