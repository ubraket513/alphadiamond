"""Iconography: the app icon and the QML image provider.

QtAwesome bundles its own fonts, so these can assert real rendering rather than
just that a name resolves.
"""

from __future__ import annotations

from PySide6.QtCore import QSize

from diamond.app.icons import APP_ICON, IconImageProvider, app_icon


def test_app_icon_covers_the_shell_sizes(qapp):
    icon = app_icon()
    sizes = {s.width() for s in icon.availableSizes()}
    assert {16, 24, 32, 48, 256} <= sizes
    for size in sizes:
        assert not icon.pixmap(size, size).isNull()


def test_app_icon_is_the_diamond_glyph():
    assert APP_ICON == "fa6s.diamond"


def test_provider_renders_a_named_icon(qapp):
    provider = IconImageProvider()
    size = QSize()
    pixmap = provider.requestPixmap("msc.chrome-close/1B1B1B", size, QSize(24, 24))
    assert not pixmap.isNull()
    assert pixmap.width() == 24
    assert size.width() == 24  # provider reports the size it actually produced


def test_provider_honours_the_requested_colour(qapp):
    provider = IconImageProvider()
    red = provider.requestPixmap("fa6s.diamond/FF0000", QSize(), QSize(32, 32))
    green = provider.requestPixmap("fa6s.diamond/00FF00", QSize(), QSize(32, 32))
    assert red.toImage() != green.toImage()


def test_an_unknown_icon_name_does_not_raise(qapp):
    """A typo must not take the window down; it shows as a missing icon."""
    provider = IconImageProvider()
    assert provider.requestPixmap("nope.not-an-icon/000000", QSize(), QSize(16, 16)).isNull()
