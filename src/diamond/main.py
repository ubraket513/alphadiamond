"""Application entry point: build the controller, hand it to QML, run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from .agents.random_agent import RandomAgent
from .app.controller import GameController
from .app.fonts import load_bundled_fonts
from .app.icons import PROVIDER_ID, IconImageProvider, app_icon
from .app.native_chrome import NativeChrome
from .app.window_chrome import (
    apply_native_rounding,
    enable_shell_integration,
    remove_native_border,
    set_app_user_model_id,
)
from .game.state import DEFAULT_PLAYERS, PlayerKind

QML_DIR = Path(__file__).parent / "qml"


def build_controller(
    seed: int | None = None,
    thinking_delay_ms: int | None = None,
    sounds: bool = True,
) -> GameController:
    agents = {
        spec.id: RandomAgent(seed=seed)
        for spec in DEFAULT_PLAYERS
        if spec.kind is PlayerKind.AI
    }
    return GameController(
        DEFAULT_PLAYERS,
        agents=agents,
        thinking_delay_ms=thinking_delay_ms,
        sounds=sounds,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="diamond")
    parser.add_argument(
        "--seed", type=int, default=None, help="seed the RandomAgent for reproducible play"
    )
    parser.add_argument(
        "--thinking-delay",
        type=int,
        default=None,
        metavar="MS",
        help="artificial agent delay in milliseconds (default 400)",
    )
    return parser.parse_args(argv)


def create_native_chrome() -> NativeChrome | None:
    """The Windows non-client hook, for Snap Layouts and the size animation.

    Created before the QML loads so the context property is a real object from
    the outset; the window is attached afterwards.
    """
    return NativeChrome() if sys.platform == "win32" else None


def build_engine(controller, font_family: str) -> QQmlApplicationEngine:
    """Create the QML engine with everything Main.qml expects.

    Shared with `debug_qml.py` on purpose: the two drifted once already, and a
    harness missing the image provider renders every icon as a broken image
    while the real app looks fine.
    """
    engine = QQmlApplicationEngine()
    engine.addImportPath(str(QML_DIR))  # makes `import Style` resolvable
    engine.addImageProvider(PROVIDER_ID, IconImageProvider())
    engine.rootContext().setContextProperty("controller", controller)
    engine.rootContext().setContextProperty("appFontFamily", font_family)
    return engine


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Before any window exists: the shell reads this when the first top-level
    # window appears, and without it the taskbar shows python.exe's icon.
    set_app_user_model_id()

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Diamond Controller")
    app.setOrganizationName("alphadiamond")
    QQuickStyle.setStyle("Basic")  # deterministic look across platforms

    # Must happen before the QML loads: Theme reads `appFontFamily` at import.
    font_family = load_bundled_fonts()
    app.setFont(QFont(font_family))
    app.setWindowIcon(app_icon())

    controller = build_controller(seed=args.seed, thinking_delay_ms=args.thinking_delay)

    chrome = create_native_chrome()

    engine = build_engine(controller, font_family)
    engine.rootContext().setContextProperty("nativeChrome", chrome)
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Main.qml")))

    if not engine.rootObjects():
        print("failed to load QML", file=sys.stderr)
        return 1

    # A frameless window is a WS_POPUP as far as the shell is concerned, which
    # costs it the minimise/restore animation, taskbar click-to-minimise and
    # the rounded corners. Put all three back.
    root_window = engine.rootObjects()[0]

    # WS_THICKFRAME is what DWM wants before it will animate a resize, but it
    # also reserves non-client space; the filter answers WM_NCCALCSIZE to take
    # that space back, so the bit is only safe once the filter is running.
    if chrome is not None:
        chrome.attach(root_window)
        app.installNativeEventFilter(chrome)

    enable_shell_integration(root_window, sizing_frame=chrome is not None)
    apply_native_rounding(root_window)
    remove_native_border(root_window)

    app.aboutToQuit.connect(controller.shutdown)
    exit_code = app.exec()
    del engine
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
