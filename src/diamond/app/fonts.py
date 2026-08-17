"""Registration of the bundled application typeface.

Google Sans Flex ships with the app rather than being assumed present: no
desktop OS installs it, and a missing family falls back silently to a system
default, which would change every metric the QML layout was tuned against.

Loading here and handing the resolved family name to QML means the UI asks for
a family Qt has definitely got.  If the files are somehow unreadable we fall
back to a platform stack rather than failing to start -- an ugly console still
beats no console.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontDatabase

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

PRIMARY_FAMILY = "Google Sans Flex"

FALLBACK_FAMILY = "Segoe UI, Noto Sans, DejaVu Sans, sans-serif"
"""Used only when the bundled files cannot be loaded at all."""


def load_bundled_fonts() -> str:
    """Register the bundled TTFs and return the family name QML should use."""
    families: set[str] = set()
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id != -1:
            families.update(QFontDatabase.applicationFontFamilies(font_id))

    if PRIMARY_FAMILY in families:
        return PRIMARY_FAMILY
    if families:
        # A future re-export could rename the family; prefer it over the
        # platform stack regardless, since it is still the intended typeface.
        return sorted(families)[0]
    return FALLBACK_FAMILY


__all__ = ["FONT_DIR", "PRIMARY_FAMILY", "load_bundled_fonts"]
