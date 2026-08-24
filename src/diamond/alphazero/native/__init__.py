"""Optional native Soo backend: import guard and capability probe.

The extension is never required.  If it is missing, stale or built for another
board, :func:`native_module` returns ``None`` and every caller must fall back to
the Python backend, which stays the default and the parity oracle.

Importing this package configures the extension with the tables exported by
:mod:`.topology`, so a native handle is always topology-configured.
"""

from __future__ import annotations

import threading
from typing import Any

from ...contract.state import PlayerSpec
from .topology import CAMP_INDEX, CAMP_ORDER, player_table, topology_tables

_LOCK = threading.Lock()
_MODULE: Any | None = None
_ERROR: str | None = None
_LOADED = False


def _load() -> None:
    global _MODULE, _ERROR, _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        try:
            from . import _diamond_native as module  # type: ignore[attr-defined]
        except ImportError as exc:  # not built on this host
            _MODULE, _ERROR = None, f"native extension unavailable: {exc}"
        else:
            try:
                module.configure(topology_tables())
            except Exception as exc:  # noqa: BLE001 - the guard must never raise
                _MODULE, _ERROR = None, f"native extension failed to configure: {exc}"
            else:
                _MODULE, _ERROR = module, None
        _LOADED = True


def native_module() -> Any | None:
    """The configured extension, or ``None`` when it is unavailable."""
    _load()
    return _MODULE


def native_error() -> str | None:
    """Why the extension is unavailable, or ``None`` when it loaded."""
    _load()
    return _ERROR


def is_available() -> bool:
    return native_module() is not None


def require_native() -> Any:
    module = native_module()
    if module is None:
        raise RuntimeError(native_error() or "native extension unavailable")
    return module


def native_game(players: tuple[PlayerSpec, ...]) -> Any:
    """A native ``Game`` handle for a Python seat list, in turn order."""
    return require_native().Game(list(player_table(players)))


__all__ = [
    "CAMP_INDEX",
    "CAMP_ORDER",
    "is_available",
    "native_error",
    "native_game",
    "native_module",
    "player_table",
    "require_native",
    "topology_tables",
]
