# The burnout3-nx build environment: devkitA64 with libnx and deko3d, the host tools, the recompiler SDK at the commit
# CMakeLists.txt pins, and UAM (the deko3d shader compiler) built as a library. Holds no game code: the game's
# default.xbe is mounted at run time (docker/build.sh).
FROM devkitpro/devkita64@sha256:1fc388c3a0d34bd2045a6dadcb1020e069d5f876a187fd705de14b4440c00282

RUN apt-get update \
 && apt-get install -y --no-install-recommends clang g++ cmake ninja-build git bison flex python3 python3-mako ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG SDK_REPO=https://github.com/falk/XboxXBERecomp.git
ARG SDK_REV
RUN test -n "$SDK_REV" \
 && git clone --filter=blob:none "$SDK_REPO" /opt/xbe-sdk \
 && git -C /opt/xbe-sdk checkout -q --detach "$SDK_REV"

COPY switch/build-uam.sh /opt/uam-build/switch/build-uam.sh
COPY switch/deko3d/uam-library.patch /opt/uam-build/switch/deko3d/uam-library.patch
RUN /opt/uam-build/switch/build-uam.sh -o /opt/uam && rm -rf /opt/uam/src/.git

ENV HOME=/tmp
WORKDIR /work
