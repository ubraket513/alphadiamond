#pragma once

#include <QAbstractNativeEventFilter>
#include <QObject>
#include <QRectF>

class QWindow;

class NativeChrome final : public QObject, public QAbstractNativeEventFilter {
    Q_OBJECT
    Q_PROPERTY(bool maximiseHovered READ maximiseHovered NOTIFY maximiseHoveredChanged)

  public:
    explicit NativeChrome(QObject* parent = nullptr);

    bool maximiseHovered() const { return maximise_hovered_; }
    void attach(QWindow* window);
    Q_INVOKABLE void setMaximiseButtonRect(double x, double y, double width, double height);

    bool nativeEventFilter(const QByteArray& eventType, void* message, qintptr* result) override;

    static bool setAppUserModelId();
    static bool enableShellIntegration(QWindow* window);
    static bool applyDwmAppearance(QWindow* window);

  Q_SIGNALS:
    void maximiseHoveredChanged();
    void maximiseClicked();

  private:
    void setHovered(bool hovered);

    QWindow* window_ = nullptr;
    quintptr native_handle_ = 0;
    QRectF maximise_rect_;
    bool maximise_hovered_ = false;
};
