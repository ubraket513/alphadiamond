"""Move sound effect.

Kept out of :class:`GameController` so the turn state machine has no opinion
about audio: the controller announces that a move was committed, and this
object decides whether anything is audible.

Audio is deliberately best-effort.  A machine with no audio device, no codec
for the file, or a locked-down backend must still play a full match, so every
failure path here degrades to silence rather than raising.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

SOUND_DIR = Path(__file__).resolve().parent.parent / "assets" / "sounds"

MOVE_SOUND = SOUND_DIR / "move.m4a"

DEFAULT_VOLUME = 0.6
"""Well under full scale: this fires on every committed move."""


class MovePlayer(QObject):
    """Plays the move sound, one committed move at a time."""

    def __init__(self, parent: QObject | None = None, volume: float = DEFAULT_VOLUME) -> None:
        super().__init__(parent)
        self._available = MOVE_SOUND.is_file()
        self._muted = False
        self._player: QMediaPlayer | None = None
        self._output: QAudioOutput | None = None

        if not self._available:
            return

        try:
            self._output = QAudioOutput(self)
            self._output.setVolume(volume)
            self._player = QMediaPlayer(self)
            self._player.setAudioOutput(self._output)
            self._player.setSource(QUrl.fromLocalFile(str(MOVE_SOUND)))
        except Exception:  # pragma: no cover - depends on the audio backend
            self._available = False
            self._player = None
            self._output = None

    @property
    def available(self) -> bool:
        """True when a sound is loaded and could be played."""
        return self._available and self._player is not None

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    def play(self) -> bool:
        """Restart the move sound.  Returns whether anything was played.

        Seeking to zero first means rapid confirmations retrigger the sound
        instead of being swallowed while the previous one is still running.
        """
        if self._muted or not self.available:
            return False
        try:
            self._player.stop()
            self._player.setPosition(0)
            self._player.play()
        except Exception:  # pragma: no cover - depends on the audio backend
            return False
        return True


__all__ = ["MOVE_SOUND", "SOUND_DIR", "MovePlayer"]
