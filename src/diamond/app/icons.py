"""Iconography, via QtAwesome.

QtAwesome bundles its own icon fonts, which is the reason to use it here rather
than drawing each glyph as a QML `Shape`: the earlier hand-drawn chevrons and
caption marks existed only because the text typeface has no such characters and
the Windows symbol font cannot be relied on cross-platform. A bundled font has
neither problem.

Two entry points:

* :func:`app_icon` — the window and taskbar icon.
* :class:`IconImageProvider` — lets QML ask for any icon by name and colour,
  as ``image://qta/<name>/<rrggbb>``.
"""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtQuick import QQuickImageProvider

import qtawesome as qta

APP_ICON = "fa6s.diamond"
"""A plain rhombus. The detailed gem cuts lose all definition at 16px."""

APP_ICON_COLOR = "#FF3B30"
"""systemRed — the app's most identifiable colour, and Player 1's."""

PROVIDER_ID = "qta"

DEFAULT_SIZE = 32


def app_icon(color: str = APP_ICON_COLOR) -> QIcon:
    """The window / taskbar icon, rendered at the sizes the shell asks for."""
    icon = QIcon()
    source = qta.icon(APP_ICON, color=color)
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(source.pixmap(size, size))
    return icon


class IconImageProvider(QQuickImageProvider):
    """Serves QtAwesome icons to QML.

    QML asks for ``image://qta/<name>/<rrggbb>`` — for example
    ``image://qta/msc.chrome-close/1B1B1B``. The colour is passed without its
    leading ``#`` because that character terminates a URL.
    """

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.ImageType.Pixmap)

    def requestPixmap(self, image_id: str, size: QSize, requested: QSize) -> QPixmap:
        name, _, color = image_id.partition("/")

        width = requested.width() if requested.width() > 0 else DEFAULT_SIZE
        height = requested.height() if requested.height() > 0 else DEFAULT_SIZE

        try:
            icon = qta.icon(name, color=f"#{color}" if color else None)
        except Exception:
            # An unknown name must not take the window down with it; an empty
            # pixmap shows up as a missing icon and nothing else breaks.
            return QPixmap()

        pixmap = icon.pixmap(width, height)
        if size is not None:
            size.setWidth(pixmap.width())
            size.setHeight(pixmap.height())
        return pixmap


__all__ = ["APP_ICON", "APP_ICON_COLOR", "PROVIDER_ID", "IconImageProvider", "app_icon"]
