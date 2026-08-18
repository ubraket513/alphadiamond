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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Diamond Controller")
    app.setOrganizationName("alphadiamond")
    QQuickStyle.setStyle("Basic")  # deterministic look across platforms

    # Must happen before the QML loads: Theme reads `appFontFamily` at import.
    font_family = load_bundled_fonts()
    app.setFont(QFont(font_family))

    controller = build_controller(seed=args.seed, thinking_delay_ms=args.thinking_delay)

    engine = QQmlApplicationEngine()
    engine.addImportPath(str(QML_DIR))  # makes `import Style` resolvable
    engine.rootContext().setContextProperty("controller", controller)
    engine.rootContext().setContextProperty("appFontFamily", font_family)
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Main.qml")))

    if not engine.rootObjects():
        print("failed to load QML", file=sys.stderr)
        return 1

    app.aboutToQuit.connect(controller.shutdown)
    exit_code = app.exec()
    del engine
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
