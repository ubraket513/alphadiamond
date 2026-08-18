"""Move sound effect.

Kept out of :class:`GameController` so the turn state machine has no opinion
about audio: the controller announces that a move was committed, and this
object decides whether anything is audible.

Audio is best-effort -- a machine with no device, no codec or a locked-down
backend must still play a full match -- but *silently* best-effort is a trap:
the first version reported failure nowhere, so a broken audio backend and a
missing feature looked identical from the outside.  Every failure here is
therefore recorded in :attr:`status` and surfaced by the controller.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal

try:  # Audio is optional; the GUI must still run if QtMultimedia is unavailable.
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except ImportError:  # pragma: no cover - platform/package dependent
    QAudioOutput = None
    QMediaPlayer = None

SOUND_DIR = Path(__file__).resolve().parent.parent / "assets" / "sounds"

MOVE_SOUND = SOUND_DIR / "move.m4a"

DEFAULT_VOLUME = 0.6
"""Well under full scale: this fires on every hop of every move."""


def _clamp(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_VOLUME
    return min(1.0, max(0.0, value))


class MovePlayer(QObject):
    """Plays the move sound, one committed move at a time."""

    statusChanged = Signal()
    """Emitted when :attr:`status` changes, so the UI can report a failure."""

    def __init__(self, parent: QObject | None = None, volume: float = DEFAULT_VOLUME) -> None:
        super().__init__(parent)
        self._muted = False
        self._volume = _clamp(volume)
        self._player: QMediaPlayer | None = None
        self._output: QAudioOutput | None = None
        self._status = ""
        # setSource() loads asynchronously.  A move confirmed before loading
        # finishes would otherwise be swallowed, so the request is remembered
        # and replayed once the media is ready.
        self._pending = False
        self._loaded = False

        if QMediaPlayer is None or QAudioOutput is None:
            self._status = "QtMultimedia is not available in this PySide6 build."
            return
        if not MOVE_SOUND.is_file():
            self._status = f"Sound file missing: {MOVE_SOUND}"
            return

        try:
            self._output = QAudioOutput(self)
            self._output.setVolume(self._volume)
            self._player = QMediaPlayer(self)
            self._player.setAudioOutput(self._output)
            self._player.mediaStatusChanged.connect(self._on_media_status)
            self._player.errorOccurred.connect(self._on_error)
            self._player.setSource(QUrl.fromLocalFile(str(MOVE_SOUND)))
        except Exception as exc:  # pragma: no cover - depends on the audio backend
            self._status = f"Audio backend unavailable: {exc}"
            self._player = None
            self._output = None

    # -- state -----------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when a player exists and has not reported a fatal error."""
        return self._player is not None and not self._status

    @property
    def status(self) -> str:
        """Empty when healthy, otherwise a human-readable reason."""
        return self._status

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    @property
    def volume(self) -> float:
        """Playback level, 0.0 to 1.0.  Retained while muted."""
        return self._volume

    def set_volume(self, volume: float) -> None:
        self._volume = _clamp(volume)
        if self._output is not None:
            self._output.setVolume(self._volume)

    # -- internals -------------------------------------------------------
    def _set_status(self, status: str) -> None:
        if status != self._status:
            self._status = status
            self.statusChanged.emit()

    def _on_media_status(self, status) -> None:
        name = str(status)
        if "InvalidMedia" in name:
            self._set_status(f"Cannot decode {MOVE_SOUND.name}; no codec for it.")
        elif "LoadedMedia" in name or "BufferedMedia" in name:
            self._loaded = True
            if self._pending:
                self._pending = False
                self._start()

    def _on_error(self, error, message: str) -> None:
        if "NoError" in str(error):
            return
        self._set_status(message or f"Audio error: {error}")

    def _start(self) -> None:
        # Rewinding first means rapid confirmations retrigger the sound rather
        # than being ignored while the previous one is still running.
        self._player.stop()
        self._player.setPosition(0)
        self._player.play()

    # -- playback --------------------------------------------------------
    def play(self) -> bool:
        """Restart the move sound.  Returns whether playback was started.

        A ``False`` here means muted, unavailable, or still loading -- never a
        silently dropped error; those land in :attr:`status`.
        """
        if self._muted or not self.available:
            return False
        if not self._loaded:
            self._pending = True  # replayed from _on_media_status
            return False
        try:
            self._start()
        except Exception as exc:  # pragma: no cover - depends on the audio backend
            self._set_status(f"Playback failed: {exc}")
            return False
        return True


__all__ = ["MOVE_SOUND", "SOUND_DIR", "DEFAULT_VOLUME", "MovePlayer"]
