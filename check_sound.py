"""Diagnose the move sound end to end.

Run from the repo root:   python check_sound.py
Plays the sound once and prints exactly where the chain breaks, if it does.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PySide6.QtGui import QGuiApplication

app = QGuiApplication(sys.argv[:1])

print("python      :", sys.executable)

import PySide6

print("PySide6     :", PySide6.__version__)

try:
    from PySide6.QtMultimedia import QMediaDevices

    outputs = [d.description() for d in QMediaDevices.audioOutputs()]
    print("audio out   :", outputs or "NONE FOUND")
    print("default out :", QMediaDevices.defaultAudioOutput().description() or "NONE")
except Exception as exc:
    print("QtMultimedia: FAILED ->", exc)
    raise SystemExit(1)

from diamond.app.sounds import MOVE_SOUND, MovePlayer

print("sound file  :", MOVE_SOUND)
print("            exists:", MOVE_SOUND.is_file())

player = MovePlayer()
print("available   :", player.available)
print("status      :", player.status or "(healthy)")

states: list[str] = []
player._player.playbackStateChanged.connect(lambda s: states.append(str(s)))

print("\nplaying... (listen)")
player.play()
deadline = time.time() + 4
while time.time() < deadline:
    app.processEvents()
    time.sleep(0.02)

played = any("Playing" in s for s in states)
print("entered PlayingState:", played)
print("duration ms :", player._player.duration())
print("final error :", player._player.errorString() or "(none)")
print("status      :", player.status or "(healthy)")

print("\nRESULT:", "OK - audio pipeline works" if played else "FAILED - see status above")

# And the controller path the app actually uses.
from diamond.main import build_controller

controller = build_controller(seed=1, thinking_delay_ms=0)
print("\ncontroller.soundAvailable :", controller.soundAvailable)
print("controller.soundEnabled   :", controller.soundEnabled)
print("controller.soundStatus    :", controller.soundStatus or "(healthy)")
controller.shutdown()
