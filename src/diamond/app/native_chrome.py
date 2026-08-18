"""Windows 11 custom window chrome: Snap Layouts and native resize behaviour.

A frameless Qt window is a `WS_POPUP`, and two things the shell does for
ordinary windows depend on it not being one:

* **Snap Layouts** — hovering the maximise button pops up the layout grid.
  Windows offers it only when the window answers ``WM_NCHITTEST`` with
  ``HTMAXBUTTON``, which a Qt window never does because Qt reports the whole
  surface as client area.
* **The maximise / restore size animation** — driven by DWM, and only for
  windows carrying ``WS_THICKFRAME``.

``WS_THICKFRAME`` normally drags a native frame back with it, which is what put
a band across the top of the chrome and clipped the caption buttons. Answering
``WM_NCCALCSIZE`` with the full window rectangle removes that frame while
keeping everything the style bit buys — the standard way custom chrome is done
on Windows.

The filter needs to know where the maximise button is, and the button needs to
know when the *system* is hovering it, since Windows owns the pointer once the
hit test says ``HTMAXBUTTON``. Both cross through :class:`NativeChrome`.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import (
    QAbstractNativeEventFilter,
    QObject,
    Property,
    Signal,
    Slot,
)

WM_NCCALCSIZE = 0x0083
WM_NCHITTEST = 0x0084
WM_NCMOUSEMOVE = 0x00A0
WM_NCLBUTTONDOWN = 0x00A1
WM_NCLBUTTONUP = 0x00A2
WM_NCMOUSELEAVE = 0x02A2

HTCLIENT = 1
HTMAXBUTTON = 9

SM_CXSIZEFRAME = 32
SM_CYSIZEFRAME = 33
SM_CXPADDEDBORDER = 92

MONITOR_DEFAULTTONEAREST = 2


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hWnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
    ]


class _NCCALCSIZE_PARAMS(ctypes.Structure):
    _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class NativeChrome(QObject, QAbstractNativeEventFilter):
    """Bridges the Windows non-client area to the QML title bar."""

    maximiseHoveredChanged = Signal()
    maximiseClicked = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        QObject.__init__(self, parent)
        QAbstractNativeEventFilter.__init__(self)
        self._window = None
        self._hwnd = 0
        self._maximise_hovered = False
        # Logical coordinates, as QML sees them; scaled on use.
        self._rect = (0, 0, 0, 0)

    def attach(self, window) -> None:
        """Bind to the window once QML has created it.

        Construction happens *before* the QML loads so the context property is
        a real object from the start: bound to a placeholder, the title bar
        would skip reporting the button rectangle at Component.onCompleted and
        never retry, leaving the hit test with nothing to match.
        """
        self._window = window
        self._hwnd = int(window.winId()) if sys.platform == "win32" else 0

    # -- QML interface ---------------------------------------------------
    def _get_maximise_hovered(self) -> bool:
        return self._maximise_hovered

    maximiseHovered = Property(bool, _get_maximise_hovered, notify=maximiseHoveredChanged)

    @Slot(float, float, float, float)
    def setMaximiseButtonRect(self, x: float, y: float, width: float, height: float) -> None:
        """Where the maximise button sits, in the window's logical coordinates."""
        self._rect = (x, y, width, height)

    # -- internals -------------------------------------------------------
    def _set_hovered(self, hovered: bool) -> None:
        if hovered != self._maximise_hovered:
            self._maximise_hovered = hovered
            self.maximiseHoveredChanged.emit()

    def _over_maximise(self, screen_x: int, screen_y: int) -> bool:
        x, y, w, h = self._rect
        if w <= 0 or h <= 0:
            return False
        point = wintypes.POINT(screen_x, screen_y)
        ctypes.windll.user32.ScreenToClient(wintypes.HWND(self._hwnd), ctypes.byref(point))
        ratio = self._window.devicePixelRatio() or 1.0
        local_x = point.x / ratio
        local_y = point.y / ratio
        return x <= local_x < x + w and y <= local_y < y + h

    def _is_maximised(self) -> bool:
        return bool(ctypes.windll.user32.IsZoomed(wintypes.HWND(self._hwnd)))

    def _work_area(self) -> wintypes.RECT | None:
        """The work area of the monitor this window is on."""
        monitor = ctypes.windll.user32.MonitorFromWindow(
            wintypes.HWND(self._hwnd), MONITOR_DEFAULTTONEAREST
        )
        if not monitor:
            return None
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if not ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        return info.rcWork

    # -- the filter ------------------------------------------------------
    def nativeEventFilter(self, event_type, message):
        if event_type != b"windows_generic_MSG":
            return False, 0

        if not self._hwnd:
            return False, 0

        msg = ctypes.cast(int(message), ctypes.POINTER(_MSG)).contents
        if msg.hWnd != self._hwnd:
            return False, 0

        if msg.message == WM_NCCALCSIZE and msg.wParam:
            # Claim the whole window as client area, so no frame is drawn even
            # though WS_THICKFRAME is set.
            #
            # Maximising a window with a sizing frame can size it to the work
            # area *plus* the frame, on the assumption the frame falls off
            # screen. With the frame reclaimed there is nothing to spare, and
            # the overhang would clip whatever sits at the edges -- the caption
            # buttons, here. Rather than assume it happens, clamp only when the
            # proposed rectangle actually exceeds the work area.
            if self._is_maximised():
                work = self._work_area()
                if work is not None:
                    params = ctypes.cast(
                        msg.lParam, ctypes.POINTER(_NCCALCSIZE_PARAMS)
                    ).contents
                    rect = params.rgrc[0]
                    rect.left = max(rect.left, work.left)
                    rect.top = max(rect.top, work.top)
                    rect.right = min(rect.right, work.right)
                    rect.bottom = min(rect.bottom, work.bottom)
            return True, 0

        if msg.message == WM_NCHITTEST:
            # Windows only offers Snap Layouts to a window that claims the
            # maximise button in its hit test.
            x = ctypes.c_short(msg.lParam & 0xFFFF).value
            y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
            if self._over_maximise(x, y):
                self._set_hovered(True)
                return True, HTMAXBUTTON
            self._set_hovered(False)
            return True, HTCLIENT

        if msg.message in (WM_NCMOUSELEAVE,):
            self._set_hovered(False)
            return False, 0

        if msg.message == WM_NCLBUTTONDOWN and msg.wParam == HTMAXBUTTON:
            # Swallow it: releasing over the button is what maximises, and
            # letting the default handler run would start a frame drag.
            return True, 0

        if msg.message == WM_NCLBUTTONUP and msg.wParam == HTMAXBUTTON:
            self.maximiseClicked.emit()
            return True, 0

        return False, 0


__all__ = ["NativeChrome"]
