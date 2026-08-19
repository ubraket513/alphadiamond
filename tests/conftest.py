from __future__ import annotations

import os
import time

import pytest

from diamond.game.board import standard_board
from diamond.game.state import EMPTY, GameState


@pytest.fixture(scope="session")
def board():
    return standard_board()


@pytest.fixture(scope="session")
def qapp():
    """A Qt event loop for controller and font tests.

    QGuiApplication rather than QCoreApplication: the font database needs the
    GUI layer to register the bundled typeface.  ``offscreen`` keeps it
    headless, and only one application object may exist per process, so every
    Qt test shares this one.
    """
    from PySide6.QtGui import QGuiApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication.instance() or QGuiApplication([])
    yield app


def make_state(board, pieces: dict[int, int], current_player_id: int = 1, turn: int = 1):
    """Build an arbitrary state: ``pieces`` maps position id -> player id."""
    occupancy = [EMPTY] * len(board)
    for position_id, player_id in pieces.items():
        occupancy[position_id] = player_id
    return GameState(
        occupancy=tuple(occupancy), current_player_id=current_player_id, turn_number=turn
    )


def pump(app, predicate, timeout: float = 5.0, interval: float = 0.005) -> bool:
    """Spin the event loop until ``predicate()`` holds or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(interval)
    app.processEvents()
    return predicate()

# Modules whose tests need a real Qt stack.  Marking them here rather than in
# each file keeps the list in one place, so a GUI-free run (CI, a machine with
# no Qt platform plugin) is exactly ``-m "not gui"``.
_GUI_MODULES = frozenset(
    {
        "test_controller",
        "test_fonts",
        "test_icons",
        "test_integration",
        "test_sounds",
        "test_window_chrome",
    }
)


def pytest_collection_modifyitems(items):
    gui = pytest.mark.gui
    for item in items:
        if item.module.__name__.rsplit(".", 1)[-1] in _GUI_MODULES:
            item.add_marker(gui)
