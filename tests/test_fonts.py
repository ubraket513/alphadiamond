"""The bundled typeface must actually register with Qt.

Without this, a missing or corrupt font file degrades silently to a system
fallback and every metric in the QML layout shifts.
"""

from __future__ import annotations

from diamond.app.fonts import FONT_DIR, PRIMARY_FAMILY, load_bundled_fonts


def test_font_files_are_present():
    names = {path.name for path in FONT_DIR.glob("*.ttf")}
    assert names == {
        "GoogleSansFlex-Regular.ttf",
        "GoogleSansFlex-Medium.ttf",
        "GoogleSansFlex-Bold.ttf",
    }


def test_the_licence_ships_alongside_the_font():
    licence = (FONT_DIR / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE" in licence.upper()
    assert "Google Sans Flex" in licence


def test_loading_registers_google_sans_flex(qapp):
    assert load_bundled_fonts() == PRIMARY_FAMILY


def test_qt_resolves_the_family_after_loading(qapp):
    from PySide6.QtGui import QFont, QFontInfo

    family = load_bundled_fonts()
    assert QFontInfo(QFont(family)).family() == PRIMARY_FAMILY
