"""Run the app with full QML diagnostics.

    python debug_qml.py

Same app as `python -m diamond`, plus:
  * every Qt/QML message is printed with its category and source location;
  * a running tally is printed on exit, so repeated spam is summarised;
  * binding loops and type errors are made as loud as Qt can make them.

Use this to capture the exact message text when something misbehaves at
runtime, then quit the window normally.
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PySide6.QtCore import QtMsgType, QUrl, qInstallMessageHandler
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtQuickControls2 import QQuickStyle

_LEVEL = {
    QtMsgType.QtDebugMsg: "DEBUG",
    QtMsgType.QtInfoMsg: "INFO",
    QtMsgType.QtWarningMsg: "WARN",
    QtMsgType.QtCriticalMsg: "CRIT",
    QtMsgType.QtFatalMsg: "FATAL",
}

seen: collections.Counter = collections.Counter()


def handler(mode, context, message):
    seen[message] += 1
    where = ""
    if context.file:
        where = f"  [{context.file}:{context.line}]"
    # Print each distinct message once, then only every 25th repeat, so a
    # binding that re-fires every frame does not bury everything else.
    count = seen[message]
    if count == 1 or count % 25 == 0:
        suffix = f"  (x{count})" if count > 1 else ""
        print(f"{_LEVEL.get(mode, '?'):5} {message}{where}{suffix}", flush=True)


def main() -> int:
    qInstallMessageHandler(handler)

    from diamond.app.fonts import load_bundled_fonts
    from diamond.app.icons import app_icon
    from diamond.main import QML_DIR, build_controller, build_engine

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Diamond Controller (debug)")
    QQuickStyle.setStyle("Basic")

    family = load_bundled_fonts()
    app.setFont(QFont(family))
    app.setWindowIcon(app_icon())
    print(f"font: {family}")
    print(f"colorScheme: {app.styleHints().colorScheme()}")

    controller = build_controller()
    engine = build_engine(controller, family)
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Main.qml")))

    if not engine.rootObjects():
        print("QML failed to load", file=sys.stderr)
        return 1

    app.aboutToQuit.connect(controller.shutdown)
    code = app.exec()
    del engine

    print("\n=== message tally ===")
    for message, count in seen.most_common(30):
        print(f"{count:6}  {message}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
