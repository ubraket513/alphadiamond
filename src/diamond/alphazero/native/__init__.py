"""The native Soo backend: import guard and capability probe.

The extension is **required** for anything that executes games -- decision 1 in
docs/architecture/decisions.md retired the Python search and self-play backend,
and there is nothing to fall back to. What this guard still does is say plainly
why it is unavailable, because "no module named _diamond_native" from six frames
down is not a diagnosis.

The extension derives its own board tables at import (``ensure_topology_configured``
in ``native/src/topology_gen.cpp``), so a handle is always topology-configured
without Python handing it anything. :mod:`.topology` still generates the same
tables from the Python board -- that is the export the deployment artifact ships
and the golden corpus pins, and ``tests/native/test_topology_generation.py``
holds the two constructions to the same answer.
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
                if not module.is_configured():
                    # An older extension that does not configure itself.
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
