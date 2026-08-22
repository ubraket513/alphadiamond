#include <QGuiApplication>
#include <QImage>
#include <QQuickImageProvider>
#include <QQuickWindow>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QPainter>
#include <QPen>
#include <QFile>
#include <QTextStream>
#include <QTimer>

#include <cstdlib>

#include "native_controller.hpp"

class PlaceholderIconProvider final : public QQuickImageProvider {
  public:
    PlaceholderIconProvider() : QQuickImageProvider(QQuickImageProvider::Image) {}

    QImage requestImage(const QString& imageId, QSize* size, const QSize& requestedSize) override {
        const QSize actual = requestedSize.isValid() ? requestedSize : QSize(16, 16);
        if (size) *size = actual;
        QImage image(actual, QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::transparent);
        const auto parts = imageId.split('/');
        const QString name = parts.value(0);
        const QColor color(parts.value(1).isEmpty() ? QStringLiteral("#6B7280")
                                                     : QStringLiteral("#") + parts.value(1));
        QPainter painter(&image);
        painter.setRenderHint(QPainter::Antialiasing);
        QPen pen(color, std::max(1, actual.width() / 10), Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        const QRectF r = QRectF(0, 0, actual.width(), actual.height()).adjusted(actual.width() * .22, actual.height() * .22,
                                                 -actual.width() * .22, -actual.height() * .22);
        if (name == QStringLiteral("msc.chrome-close")) {
            painter.drawLine(r.topLeft(), r.bottomRight()); painter.drawLine(r.topRight(), r.bottomLeft());
        } else if (name == QStringLiteral("msc.chrome-minimize")) {
            painter.drawLine(r.left(), r.center().y(), r.right(), r.center().y());
        } else if (name == QStringLiteral("msc.chrome-maximize")) {
            painter.drawRect(r);
        } else if (name == QStringLiteral("msc.chrome-restore")) {
            const QRectF back = r.adjusted(r.width() * .18, -r.height() * .18, r.width() * .18, -r.height() * .18);
            painter.drawRect(back); painter.drawRect(r.adjusted(-r.width() * .18, r.height() * .18, -r.width() * .18, r.height() * .18));
        } else if (name == QStringLiteral("msc.chevron-up") || name == QStringLiteral("msc.chevron-down")) {
            const bool up = name.endsWith(QStringLiteral("up"));
            QPolygonF chevron;
            chevron << QPointF(r.left(), up ? r.bottom() : r.top())
                    << QPointF(r.center().x(), up ? r.top() : r.bottom())
                    << QPointF(r.right(), up ? r.bottom() : r.top());
            painter.drawPolyline(chevron);
        } else if (name == QStringLiteral("msc.menu")) {
            for (int i = 0; i < 3; ++i) painter.drawLine(r.left(), r.top() + i * r.height() / 2, r.right(), r.top() + i * r.height() / 2);
        } else if (name == QStringLiteral("fa6s.diamond")) {
            QPolygonF diamond;
            diamond << QPointF(r.center().x(), r.top()) << QPointF(r.right(), r.center().y())
                    << QPointF(r.center().x(), r.bottom()) << QPointF(r.left(), r.center().y());
            painter.setBrush(color); painter.drawPolygon(diamond);
        }
        painter.end();
        return image;
    }
};

int main(int argc, char* argv[]) {
    bool smoke_mode = false;
    for (int i = 1; i < argc; ++i) smoke_mode = smoke_mode ||
        QString::fromLocal8Bit(argv[i]) == QStringLiteral("--smoke");
    if (!smoke_mode) qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("windows"));
    QGuiApplication app(argc, argv);
    app.setQuitOnLastWindowClosed(false);
    QFile startupLog(QCoreApplication::applicationDirPath() + QStringLiteral("/diamond_qt_startup.log"));
    startupLog.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);
    QTextStream startup(&startupLog);
    startup << "application-start\n";
    startup.flush();
    app.setApplicationName(QStringLiteral("Diamond"));
    startup << "after-application-name\n";
    startup.flush();
    app.setOrganizationName(QStringLiteral("alphadiamond"));
    startup << "after-organization-name\n";
    startup.flush();
    QQuickStyle::setStyle(QStringLiteral("Basic"));
    startup << "after-quick-style\n";
    startup.flush();

    NativeController controller;
    startup << "after-controller\n";
    startup.flush();
    QQmlApplicationEngine engine;
    startup << "after-engine\n";
    startup.flush();
    engine.addImportPath(QStringLiteral("qrc:/qml"));
    engine.addImageProvider(QStringLiteral("qta"), new PlaceholderIconProvider);
    engine.rootContext()->setContextProperty(QStringLiteral("controller"), &controller);
    engine.rootContext()->setContextProperty(QStringLiteral("appFontFamily"),
                                              QStringLiteral("Segoe UI"));
    startup << "before-qml-load\n";
    startup.flush();
    engine.load(QUrl(QStringLiteral("qrc:/qml/Main.qml")));
    startup << "after-qml-load roots=" << engine.rootObjects().size() << "\n";
    startup.flush();
    if (engine.rootObjects().isEmpty()) return 1;
    QObject* root = engine.rootObjects().constFirst();
    startup << "root=" << root->metaObject()->className() << " visibleProperty="
            << root->property("visible").toString() << "\n";
    root->setProperty("visible", true);
    if (auto* window = qobject_cast<QQuickWindow*>(engine.rootObjects().constFirst())) {
        window->setFlags(Qt::Window | Qt::FramelessWindowHint);
        window->resize(1440, 900);
        window->setMinimumSize(QSize(980, 640));
        window->setVisibility(QWindow::Windowed);
        window->setVisible(true);
        window->show();
        window->raise();
        window->requestActivate();
        QTimer::singleShot(100, window, [window] {
            if (!window->isVisible()) {
                window->setVisibility(QWindow::Windowed);
                window->showNormal();
                window->raise();
                window->requestActivate();
            }
        });
        startup << "window=QQuickWindow visible=" << window->isVisible()
                << " exposed=" << window->isExposed() << " size=" << window->size().width()
                << "x" << window->size().height() << "\n";
    } else {
        startup << "window=QQuickWindow cast failed\n";
    }
    startup.flush();
    startupLog.close();
    for (int i = 1; i < argc; ++i) {
        if (QString::fromLocal8Bit(argv[i]) == QStringLiteral("--game-smoke"))
            return controller.gameSmoke() ? 0 : 1;
        if (QString::fromLocal8Bit(argv[i]) == QStringLiteral("--worker-smoke"))
            return controller.workerSmoke() ? 0 : 1;
        if (QString::fromLocal8Bit(argv[i]) == QStringLiteral("--soo-smoke"))
            return controller.sooSmoke() ? 0 : 1;
        if (QString::fromLocal8Bit(argv[i]) == QStringLiteral("--smoke")) std::_Exit(0);
    }
    return app.exec();
}
