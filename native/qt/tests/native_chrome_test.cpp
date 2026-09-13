#include "native_chrome.hpp"

#include <QGuiApplication>
#include <QTimer>
#include <QWindow>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdio>

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);
    QWindow window;
    window.setFlags(Qt::Window | Qt::FramelessWindowHint);
    window.resize(980, 640);
    NativeChrome chrome;
    chrome.attach(&window);
    const auto hwnd = reinterpret_cast<HWND>(window.winId());
    const auto event_type = QByteArrayLiteral("windows_generic_MSG");
    int clicks = 0;
    QObject::connect(&chrome, &NativeChrome::maximiseClicked, &app, [&] { ++clicks; });
    const auto require = [](bool ok, const char* description) {
        if (!ok)
            std::fprintf(stderr, "native chrome test failed: %s\n", description);
        return ok;
    };

    // Qt's message queue passes nullptr; the window procedure passes an output
    // pointer. Input messages may only visit the former, so both must be handled.
    for (bool with_result : {false, true}) {
        qintptr value = -1;
        qintptr* result = with_result ? &value : nullptr;
        NCCALCSIZE_PARAMS params{};
        params.rgrc[0] = {0, 0, 980, 640};
        MSG msg{};
        msg.hwnd = hwnd;
        msg.message = WM_NCCALCSIZE;
        msg.wParam = TRUE;
        msg.lParam = reinterpret_cast<LPARAM>(&params);
        if (!require(chrome.nativeEventFilter(event_type, &msg, result), "frame calculation") ||
            !require(!result || value == 0, "frame result"))
            return 1;

        POINT point{30, 15};
        ClientToScreen(hwnd, &point);
        msg.message = WM_NCHITTEST;
        msg.wParam = 0;
        msg.lParam = MAKELPARAM(point.x, point.y);
        chrome.setMaximiseButtonRect(0, 0, 100, 100);
        if (!require(chrome.nativeEventFilter(event_type, &msg, result), "caption hit test") ||
            !require(chrome.maximiseHovered(), "caption hover") ||
            !require(!result || value == HTMAXBUTTON, "caption hit result"))
            return 1;
        chrome.setMaximiseButtonRect(200, 200, 100, 100);
        if (!require(chrome.nativeEventFilter(event_type, &msg, result), "client hit test") ||
            !require(!chrome.maximiseHovered(), "client hover") ||
            !require(!result || value == HTCLIENT, "client hit result"))
            return 1;

        msg.message = WM_NCLBUTTONDOWN;
        msg.wParam = HTMAXBUTTON;
        if (!require(chrome.nativeEventFilter(event_type, &msg, result), "caption press") ||
            !require(!result || value == 0, "press result"))
            return 1;
        const int before = clicks;
        msg.message = WM_NCLBUTTONUP;
        if (!require(chrome.nativeEventFilter(event_type, &msg, result), "caption release") ||
            !require(clicks == before + 1, "one click per release") ||
            !require(!result || value == 0, "release result"))
            return 1;
        msg.message = WM_NULL;
        if (!require(!chrome.nativeEventFilter(event_type, &msg, result), "unhandled message"))
            return 1;
    }

    MSG stale_message{};
    {
        QWindow replacement;
        chrome.attach(&replacement);
        stale_message.hwnd = reinterpret_cast<HWND>(replacement.winId());
        stale_message.message = WM_NCHITTEST;
    }
    if (!require(!chrome.nativeEventFilter(event_type, &stale_message, nullptr),
                 "ignore destroyed preview window"))
        return 1;
    chrome.attach(&window);
    app.installNativeEventFilter(&chrome);
    if (!require(NativeChrome::enableShellIntegration(&window), "shell integration"))
        return 1;
    window.show();
    QObject::connect(&chrome, &NativeChrome::maximiseClicked, &window, [&] {
        if (window.windowState() == Qt::WindowMaximized)
            window.showNormal();
        else
            window.showMaximized();
    });
    int step = 0;
    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, &app, [&] {
        // Repeat queued caption clicks, minimise, and restore on a real HWND.
        const auto expected = step % 4 == 1   ? Qt::WindowMaximized
                              : step % 4 == 3 ? Qt::WindowMinimized
                                              : Qt::WindowNoState;
        if (!require(window.windowState() == expected, "native window transition")) {
            app.exit(1);
            return;
        }
        if (step == 20) {
            std::puts("native chrome: null/non-null messages and 20 window transitions passed");
            app.quit();
            return;
        }
        switch (step++ % 4) {
        case 0:
        case 1:
            PostMessageW(hwnd, WM_NCLBUTTONDOWN, HTMAXBUTTON, 0);
            PostMessageW(hwnd, WM_NCLBUTTONUP, HTMAXBUTTON, 0);
            break;
        case 2:
            window.showMinimized();
            break;
        case 3:
            window.showNormal();
            break;
        }
    });
    timer.start(150);
    QTimer::singleShot(15000, &app, [&] { app.exit(1); });
    return app.exec();
}
