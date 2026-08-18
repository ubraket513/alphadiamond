"""Window chrome: the shell behaviours a frameless window opts out of.

These assert the contract rather than the pixels — whether corners actually
round, and whether minimise animates, is the shell's business and only
observable on a real desktop.
"""

from __future__ import annotations

import sys

import pytest

from diamond.app.window_chrome import (
    apply_native_rounding,
    enable_shell_integration,
    remove_native_border,
)

def _native_windows(qapp) -> bool:
    """True only with a real HWND behind the window.

    The suite runs on the `offscreen` platform plugin, which has no native
    window for user32/dwmapi to act on, so these checks only mean anything
    when a real windowing system is present.
    """
    return sys.platform == "win32" and qapp.platformName() == "windows"


windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows-only chrome"
)


@pytest.fixture
def window(qapp):
    from PySide6.QtQuick import QQuickWindow

    win = QQuickWindow()
    win.resize(320, 240)
    win.show()
    yield win
    win.close()


@windows_only
def test_shell_integration_sets_the_expected_style_bits(qapp, window):
    if not _native_windows(qapp):
        pytest.skip('needs a native window; the suite runs offscreen')

    import ctypes
    from ctypes import wintypes

    from diamond.app.window_chrome import (
        GWL_STYLE,
        WS_MAXIMIZEBOX,
        WS_MINIMIZEBOX,
        WS_SYSMENU,
        WS_THICKFRAME,
    )

    assert enable_shell_integration(window)

    get_long = ctypes.windll.user32.GetWindowLongPtrW
    get_long.restype = ctypes.c_ssize_t
    style = get_long(wintypes.HWND(int(window.winId())), GWL_STYLE)

    for bit in (WS_MINIMIZEBOX, WS_MAXIMIZEBOX, WS_SYSMENU, WS_THICKFRAME):
        assert style & bit == bit


@windows_only
def test_shell_integration_is_idempotent(qapp, window):
    if not _native_windows(qapp):
        pytest.skip('needs a native window; the suite runs offscreen')

    assert enable_shell_integration(window)
    assert enable_shell_integration(window)  # second call is a no-op, not a failure


@windows_only
def test_rounding_is_accepted_by_dwm(qapp, window):
    if not _native_windows(qapp):
        pytest.skip('needs a native window; the suite runs offscreen')

    assert apply_native_rounding(window)


@windows_only
def test_border_removal_is_accepted_by_dwm(qapp, window):
    if not _native_windows(qapp):
        pytest.skip("needs a native window; the suite runs offscreen")
    assert remove_native_border(window)


def test_all_are_safe_to_call_on_any_platform(window):
    """None may raise: an unavailable API degrades to a no-op."""
    assert enable_shell_integration(window) in (True, False)
    assert apply_native_rounding(window) in (True, False)
    assert remove_native_border(window) in (True, False)
