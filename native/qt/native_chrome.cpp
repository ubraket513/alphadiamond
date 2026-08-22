#include "native_chrome.hpp"

#include <QWindow>

#include <algorithm>

#ifdef Q_OS_WIN
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <shobjidl.h>

namespace {
constexpr DWORD kDwmWindowCornerPreference = 33;
constexpr DWORD kDwmBorderColor = 34;
constexpr DWORD kDwmRound = 2;
constexpr DWORD kDwmColorNone = 0xFFFFFFFE;
}
#endif

NativeChrome::NativeChrome(QObject* parent) : QObject(parent) {}

void NativeChrome::attach(QWindow* window) {
    window_ = window;
    native_handle_ = window ? static_cast<quintptr>(window->winId()) : 0;
}

void NativeChrome::setMaximiseButtonRect(double x, double y, double width, double height) {
    maximise_rect_ = QRectF(x, y, width, height);
}

void NativeChrome::setHovered(bool hovered) {
    if (maximise_hovered_ == hovered) return;
    maximise_hovered_ = hovered;
    Q_EMIT maximiseHoveredChanged();
}

bool NativeChrome::nativeEventFilter(const QByteArray& eventType, void* message, qintptr* result) {
#ifdef Q_OS_WIN
    if (eventType != QByteArrayLiteral("windows_generic_MSG") || !native_handle_) return false;
    auto* msg = static_cast<MSG*>(message);
    const HWND hwnd = reinterpret_cast<HWND>(native_handle_);
    if (msg->hwnd != hwnd) return false;

    if (msg->message == WM_NCCALCSIZE && msg->wParam) {
        if (IsZoomed(hwnd)) {
            HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
            MONITORINFO info{sizeof(MONITORINFO)};
            if (monitor && GetMonitorInfoW(monitor, &info)) {
                auto* params = reinterpret_cast<NCCALCSIZE_PARAMS*>(msg->lParam);
                params->rgrc[0].left = std::max(params->rgrc[0].left, info.rcWork.left);
                params->rgrc[0].top = std::max(params->rgrc[0].top, info.rcWork.top);
                params->rgrc[0].right = std::min(params->rgrc[0].right, info.rcWork.right);
                params->rgrc[0].bottom = std::min(params->rgrc[0].bottom, info.rcWork.bottom);
            }
        }
        *result = 0;
        return true;
    }

    if (msg->message == WM_NCHITTEST) {
        POINT point{static_cast<short>(LOWORD(msg->lParam)),
                    static_cast<short>(HIWORD(msg->lParam))};
        ScreenToClient(hwnd, &point);
        const qreal ratio = window_ && window_->devicePixelRatio() > 0
            ? window_->devicePixelRatio() : 1.0;
        const bool over = maximise_rect_.contains(point.x / ratio, point.y / ratio);
        setHovered(over);
        *result = over ? HTMAXBUTTON : HTCLIENT;
        return true;
    }
    if (msg->message == WM_NCMOUSELEAVE) setHovered(false);
    if (msg->message == WM_NCLBUTTONDOWN && msg->wParam == HTMAXBUTTON) {
        *result = 0;
        return true;
    }
    if (msg->message == WM_NCLBUTTONUP && msg->wParam == HTMAXBUTTON) {
        Q_EMIT maximiseClicked();
        *result = 0;
        return true;
    }
#else
    Q_UNUSED(eventType)
    Q_UNUSED(message)
    Q_UNUSED(result)
#endif
    return false;
}

bool NativeChrome::setAppUserModelId() {
#ifdef Q_OS_WIN
    return SUCCEEDED(SetCurrentProcessExplicitAppUserModelID(L"Diamond.ControllerConsole"));
#else
    return false;
#endif
}

bool NativeChrome::enableShellIntegration(QWindow* window) {
#ifdef Q_OS_WIN
    if (!window) return false;
    const HWND hwnd = reinterpret_cast<HWND>(window->winId());
    LONG_PTR style = GetWindowLongPtrW(hwnd, GWL_STYLE);
    style |= WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU | WS_THICKFRAME;
    SetWindowLongPtrW(hwnd, GWL_STYLE, style);
    return SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED) != FALSE;
#else
    Q_UNUSED(window)
    return false;
#endif
}

bool NativeChrome::applyDwmAppearance(QWindow* window) {
#ifdef Q_OS_WIN
    if (!window) return false;
    const HWND hwnd = reinterpret_cast<HWND>(window->winId());
    const DWORD corner = kDwmRound;
    const DWORD border = kDwmColorNone;
    const HRESULT rounded = DwmSetWindowAttribute(hwnd,
                                                   static_cast<DWMWINDOWATTRIBUTE>(kDwmWindowCornerPreference),
                                                   &corner, sizeof(corner));
    const HRESULT borderless = DwmSetWindowAttribute(hwnd,
                                                      static_cast<DWMWINDOWATTRIBUTE>(kDwmBorderColor),
                                                      &border, sizeof(border));
    return SUCCEEDED(rounded) && SUCCEEDED(borderless);
#else
    Q_UNUSED(window)
    return false;
#endif
}
