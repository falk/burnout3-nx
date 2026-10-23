# burnout3-nx

An unofficial Nintendo Switch port of the original Xbox version of **Burnout 3: Takedown** (2004), made by static recompilation with native hot paths and a native deko3d renderer.

It is built on [Xbox XBE Recomp](https://github.com/TGP482/XboxXBE-Recomp), the same way as the [BLACK Switch port](https://github.com/falk/black-nx): the game's x86 code is translated to C++ ahead of time, the Xbox kernel, XDK and Direct3D 8 are provided by the runtime, and the NV2A draw state goes to the Switch GPU through deko3d.

> [!NOTE]
> This project does not include any game files. You need the files from your own legally acquired copy of the game, the **USA** release (`default.xbe` sha256 `9f497acd…`).

> [!WARNING]
> Work in progress. Races and menus run, but expect bugs. See [Status](#status).

## Installing

1. Build `burnout3.nro` from your own copy of the game (see [Building](#building)). No builds are distributed.
2. On the SD card, make this layout:
   ```
   sdmc:/switch/burnout3/burnout3.nro
   sdmc:/switch/burnout3/game/          the game disc's files: default.xbe, Data/, Graphics/, Tracks/, ovid/ ...
   ```
3. Start it from the Homebrew Menu opened through a game (hold **R** while launching a game). Applet mode (the Album) does not have enough memory.

The game makes `hdd/` (profiles and saves) and `shaders/` (the shader cache) next to the NRO on first run. The first visit to a scene can stutter while its shaders are compiled. After that they are cached.

### Options

You can put lines of the form `NAME=value` in `sdmc:/switch/burnout3/env.txt`. When launching over `nxlink`, pass them as arguments instead.

| Option | Effect |
| --- | --- |
| `B3_PERF_DIAG=1` | Frame time, GPU time and fps log |
| `B3_PROF=1` | Sampling profiler of the game thread. Function names come from the symbol table built into the NRO |
| `B3_RENDER_DIAG=1` | Per-pass GPU times and render diagnostics |
| `B3_DRAW_DUMP=1` | `B3_RENDER_DIAG` plus a dump of one frame's draws every ten seconds (large; can stall a slow `nxlink` link) |
| `B3_TRAINING_FILMS=1` | Play the long training films. By default each is replaced with a short one |

## Status

- Boots, runs the menus, and plays races at 60 fps on the console with the 3D scene, HUD and post-processing.
- The car and track previews in the menus may be missing.
- WMA music (the soundtrack) is silent. Sound effects and speech play.
- Some frames take longer than 16.7 ms. Performance work is ongoing.

## Building

The game's code is translated from your own `default.xbe` at build time. This repository and the build tools contain no game code, so every build needs your copy of the game.

Builds use Docker. Only [Docker](https://docs.docker.com/get-docker/) and git are needed, on Linux, macOS or Windows (WSL2).

```sh
git clone https://github.com/falk/burnout3-nx
cd burnout3-nx
mkdir -p assets
cp /path/to/your/game/default.xbe assets/    # from your own USA disc
docker/build.sh                              # -> out/switch-deko/burnout3.nro
```

The first run builds the `burnout3-nx-builder` image: devkitPro's devkitA64 with libnx and deko3d, the host tools, the recompiler SDK at the pinned commit, and the deko3d shader compiler. That takes a few minutes. Then it recompiles `default.xbe` into `generated/` and builds the NRO, which takes a while the first time. Later runs reuse the image and only rebuild what changed.

The image never contains the game: `default.xbe` is read from your `assets/` folder, mounted into the container while it builds. `PROF=0 docker/build.sh` leaves the profiler out. `JOBS=<n>` sets the number of parallel compiles.

### Run

Copy the NRO and the game folders to the SD card as described in [Installing](#installing). For quick testing, you can also send the NRO over the network: open the Homebrew Menu, press **Y** for the netloader, then run

```sh
$DEVKITPRO/tools/bin/nxlink -a <switch-ip> -s out/switch-deko/burnout3.nro B3_PERF_DIAG=1
```

The game files still need to be on the SD card. `-s` prints the game's log in your terminal.

## Layout

| Path | Contents |
| --- | --- |
| `config/game.toml` | Recompiler configuration: functions, partitions, hooks |
| `src/game/` | Native replacements for the hot functions (culling, transforms, physics helpers) and game hooks |
| `src/d3d/` | Direct3D 8 for the inlined XDK code and the render thread |
| `switch/deko3d/` | The deko3d renderer: NV2A shader translation, shader cache, draw state |
| `switch/headless/` | Switch platform layer: entry, input, audio, clocks, profiler |
| `docker/` | The Docker build environment and `docker/build.sh` |
| `switch/tools/` | Recompiler build, profiler symbol tools and reports |
| `tests/` | Pad scripts for automated runs |

## Credits

- [Xbox XBE Recomp](https://github.com/TGP482/XboxXBE-Recomp), the recompiler and runtime this is built on.
- [deko3d](https://github.com/devkitPro/deko3d) and [UAM](https://github.com/devkitPro/uam) from devkitPro, the GPU API and shader compiler. [libnx](https://github.com/switchbrew/libnx).
- [XenonRecomp](https://github.com/hedge-dev/XenonRecomp), [Zydis](https://github.com/zyantific/zydis) and [toml++](https://github.com/marzer/tomlplusplus).
- Burnout 3: Takedown is © Electronic Arts / Criterion Games. This project is not affiliated with or endorsed by them.
