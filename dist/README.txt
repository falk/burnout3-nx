Burnout 3 for Switch (burnout3-nx), test build
============================================

Copy the contents of sdcard/ to the root of the Switch's SD card, so it reads:

  sdmc:/switch/burnout3/burnout3.nro       the program
  sdmc:/switch/burnout3/game/              the game disc's files (default.xbe sha256 9f497acd...)

Launch burnout3 from the Homebrew Menu, opened by holding R while starting a game (title override) for the full
memory, not from the album applet.

Made by the game on first run:
  sdmc:/switch/burnout3/hdd/       the Xbox hard disk: profiles and saves
  sdmc:/switch/burnout3/shaders/   the shader cache (included, from a test run; grows as new scenes appear)

Optional: sdmc:/switch/burnout3/env.txt, lines NAME=value, for example B3_PERF_DIAG=1 for the frame-time log.
Known issues: the menu car and track previews can be missing; WMA music is silent.
