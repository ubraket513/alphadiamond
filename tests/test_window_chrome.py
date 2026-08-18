"""Window chrome: the shell behaviours a frameless window opts out of.

These assert the contract rather than the pixels — whether corners actually
round, and whether minimise animates, is the shell's business and only
observable on a real desktop.
"""

from __future__ import annotations

import ctypes.wintypes
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

    for bit in (WS_MINIMIZEBOX, WS_MAXIMIZEBOX, WS_SYSMENU):
        assert style & bit == bit

    # WS_THICKFRAME must stay off: it is the bit that can make Windows reserve
    # non-client space, which shows as a frame band and clipped caption buttons.
    assert style & WS_THICKFRAME == 0


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


# -- app identity ------------------------------------------------------------


@windows_only
def test_app_user_model_id_is_set_and_readable(qapp):
    """Without an explicit id the shell groups the window under python.exe and
    shows Python's icon, whatever the app sets as its window icon."""
    import ctypes

    from diamond.app.window_chrome import APP_USER_MODEL_ID, set_app_user_model_id

    assert set_app_user_model_id()

    buffer = ctypes.c_wchar_p()
    result = ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(
        ctypes.byref(buffer)
    )
    assert result == 0
    assert buffer.value == APP_USER_MODEL_ID


# -- native chrome -----------------------------------------------------------


def test_native_chrome_is_inert_until_attached(qapp):
    """Constructed before the QML loads, so it must tolerate having no window:
    the filter runs against every message in the process from the moment it is
    installed."""
    from diamond.app.native_chrome import NativeChrome

    chrome = NativeChrome()
    assert not chrome.maximiseHovered
    handled, result = chrome.nativeEventFilter(b"windows_generic_MSG", 0)
    assert handled is False
    assert result == 0


def test_native_chrome_ignores_foreign_event_types(qapp):
    from diamond.app.native_chrome import NativeChrome

    chrome = NativeChrome()
    assert chrome.nativeEventFilter(b"xcb_generic_event_t", 0) == (False, 0)


def test_maximise_button_rect_round_trips(qapp):
    from diamond.app.native_chrome import NativeChrome

    chrome = NativeChrome()
    chrome.setMaximiseButtonRect(1348, 0, 46, 44)
    assert chrome._rect == (1348, 0, 46, 44)


@windows_only
def test_hit_test_claims_the_maximise_button(qapp, window):
    """Windows offers Snap Layouts only to a window whose hit test answers
    HTMAXBUTTON, so this is the whole feature in one assertion."""
    import ctypes

    from diamond.app.native_chrome import HTCLIENT, HTMAXBUTTON, NativeChrome

    if not _native_windows(qapp):
        pytest.skip("needs a native window; the suite runs offscreen")

    chrome = NativeChrome()
    chrome.attach(window)
    qapp.installNativeEventFilter(chrome)
    try:
        chrome.setMaximiseButtonRect(10, 0, 40, 30)
        handle = int(window.winId())
        ratio = window.devicePixelRatio() or 1.0

        # The filter works in client coordinates. A plain test window still has
        # a native caption, so its client origin is not its window origin —
        # unlike the real frameless window, where WM_NCCALCSIZE makes them one.
        origin = ctypes.wintypes.POINT(0, 0)
        ctypes.windll.user32.ClientToScreen(
            ctypes.wintypes.HWND(handle), ctypes.byref(origin)
        )

        def hit(local_x, local_y):
            x = int(origin.x + local_x * ratio)
            y = int(origin.y + local_y * ratio)
            return ctypes.windll.user32.SendMessageW(
                ctypes.wintypes.HWND(handle), 0x0084, 0, (y << 16) | (x & 0xFFFF)
            )

        assert hit(30, 15) == HTMAXBUTTON     # centre of the button
        assert hit(200, 200) == HTCLIENT      # ordinary content
    finally:
        qapp.removeNativeEventFilter(chrome)
