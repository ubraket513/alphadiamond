"""Native window-corner rounding.

Windows 11 rounds window corners itself, through DWM, but only for windows it
considers ordinary. A frameless window (``Qt.FramelessWindowHint``) is a popup
as far as the shell is concerned, so it opts out and comes back square — which
is why this app's corners look unlike every other window on the desktop.

Asking DWM for rounding explicitly puts them back. That is preferable to
clipping the corners ourselves in QML: the rounding then matches the platform's
own radius exactly, follows it if the user changes their preferences, and keeps
the window's drop shadow, which a self-clipped translucent window loses.

Everything here degrades to a no-op: on a non-Windows platform, on Windows 10
(where the attribute predates the OS), or if ``dwmapi`` cannot be reached.
"""

from __future__ import annotations

import sys

GWL_STYLE = -16

WS_MAXIMIZEBOX = 0x00010000
WS_MINIMIZEBOX = 0x00020000
WS_THICKFRAME = 0x00040000
WS_SYSMENU = 0x00080000

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020

DWMWA_WINDOW_CORNER_PREFERENCE = 33
"""Attribute id; available from Windows 11 (build 22000) onward."""

DWMWA_BORDER_COLOR = 34
"""Colour of the window's outline, same availability."""

DWMWA_COLOR_NONE = 0xFFFFFFFE
"""Sentinel for DWMWA_BORDER_COLOR meaning "draw no border at all"."""

DWMWCP_DEFAULT = 0
DWMWCP_DONOTROUND = 1
DWMWCP_ROUND = 2
DWMWCP_ROUNDSMALL = 3

MIN_WINDOWS_BUILD = 22000


APP_USER_MODEL_ID = "Diamond.ControllerConsole"
"""Identity the Windows shell groups this app's windows under.

Without one, the shell falls back to the host executable -- `python.exe` -- so
the taskbar shows the Python icon and groups the window with any other Python
process, no matter what `QGuiApplication.setWindowIcon` says.
"""


def set_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Give the process its own shell identity.

    Must run before any window exists: the shell reads it when the first
    top-level window is created and does not re-read it afterwards.
    """
    if sys.platform != "win32":
        return False

    import ctypes

    try:
        result = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        return False
    return result == 0


def _supported() -> bool:
    if sys.platform != "win32":
        return False
    version = getattr(sys, "getwindowsversion", None)
    return version is not None and version().build >= MIN_WINDOWS_BUILD


def _set_dwm_attribute(handle: int, attribute: int, value: int) -> bool:
    import ctypes
    from ctypes import wintypes

    payload = ctypes.c_uint(value)
    try:
        result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(handle),
            ctypes.c_uint(attribute),
            ctypes.byref(payload),
            ctypes.sizeof(payload),
        )
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        return False
    return result == 0


def apply_native_rounding(window, *, small: bool = False) -> bool:
    """Ask DWM to round ``window``'s corners.  Returns whether it took effect.

    ``small`` selects the tighter radius the shell uses for menus and popups.
    """
    if not _supported():
        return False
    handle = int(window.winId())
    if not handle:
        return False
    return _set_dwm_attribute(
        handle,
        DWMWA_WINDOW_CORNER_PREFERENCE,
        DWMWCP_ROUNDSMALL if small else DWMWCP_ROUND,
    )


def remove_native_border(window) -> bool:
    """Stop DWM drawing its own outline around the window.

    With "show accent colour on title bars and window borders" enabled — the
    Windows 11 default — DWM paints the *active* window's outline in the accent
    colour. On an ordinary window that reads as a thin highlight around the
    title bar. Here there is no native title bar to frame, so it lands as a
    stray coloured band across the top of our own chrome.

    The window already draws its own 1px border in `Main.qml`, which is the one
    that should be visible.
    """
    if not _supported():
        return False
    handle = int(window.winId())
    if not handle:
        return False
    return _set_dwm_attribute(handle, DWMWA_BORDER_COLOR, DWMWA_COLOR_NONE)


def enable_shell_integration(window, *, sizing_frame: bool = False) -> bool:
    """Restore the shell behaviours a frameless window opts out of.

    ``Qt.FramelessWindowHint`` makes the window a ``WS_POPUP``, which the shell
    treats as a transient thing rather than an application window.  The visible
    consequences are that minimise and restore happen instantly instead of
    animating, and that clicking the taskbar button does not minimise the
    window the way it does for every other app.

    Putting the minimise/maximise boxes and the system menu back makes the
    shell treat it as a normal window again.

    ``sizing_frame`` adds ``WS_THICKFRAME``, which DWM requires before it will
    animate a maximise or restore.  Left off, that bit makes Windows reserve
    non-client space -- a band of frame across the top, caption buttons clipped
    at the edge -- so it may only be set by a caller that also answers
    ``WM_NCCALCSIZE`` to reclaim that space.  See :mod:`diamond.app.native_chrome`.

    Taskbar click-to-minimise and the minimise animation come from
    ``WS_MINIMIZEBOX`` and ``WS_SYSMENU``, and need no frame at all.
    """
    if sys.platform != "win32":
        return False

    import ctypes
    from ctypes import wintypes

    handle = int(window.winId())
    if not handle:
        return False

    user32 = ctypes.windll.user32
    # GetWindowLongPtrW only exists on 64-bit; the 32-bit name is the plain one.
    get_long = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
    set_long = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
    get_long.restype = ctypes.c_ssize_t
    set_long.restype = ctypes.c_ssize_t
    set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]

    try:
        style = get_long(wintypes.HWND(handle), GWL_STYLE)
        wanted = style | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU
        wanted = (wanted | WS_THICKFRAME) if sizing_frame else (wanted & ~WS_THICKFRAME)
        if wanted != style:
            set_long(wintypes.HWND(handle), GWL_STYLE, wanted)
            user32.SetWindowPos(
                wintypes.HWND(handle), None, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        return False
    return True


__all__ = [
    "APP_USER_MODEL_ID",
    "DWMWA_COLOR_NONE",
    "DWMWCP_ROUND",
    "DWMWCP_ROUNDSMALL",
    "apply_native_rounding",
    "enable_shell_integration",
    "remove_native_border",
    "set_app_user_model_id",
]
