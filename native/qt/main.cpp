#include <QGuiApplication>
#include <QImage>
#include <QQuickImageProvider>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>

#include <cstdlib>

#include "native_controller.hpp"

class PlaceholderIconProvider final : public QQuickImageProvider {
  public:
    PlaceholderIconProvider() : QQuickImageProvider(QQuickImageProvider::Image) {}

    QImage requestImage(const QString&, QSize* size, const QSize& requestedSize) override {
        const QSize actual = requestedSize.isValid() ? requestedSize : QSize(16, 16);
        if (size) *size = actual;
        QImage image(actual, QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::transparent);
        return image;
    }
};

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("Diamond"));
    app.setOrganizationName(QStringLiteral("alphadiamond"));
    QQuickStyle::setStyle(QStringLiteral("Basic"));

    NativeController controller;
    QQmlApplicationEngine engine;
    engine.addImportPath(QStringLiteral("qrc:/qml"));
    engine.addImageProvider(QStringLiteral("qta"), new PlaceholderIconProvider);
    engine.rootContext()->setContextProperty(QStringLiteral("controller"), &controller);
    engine.rootContext()->setContextProperty(QStringLiteral("appFontFamily"),
                                              QStringLiteral("Segoe UI"));
    engine.load(QUrl(QStringLiteral("qrc:/qml/Main.qml")));
    if (engine.rootObjects().isEmpty()) return 1;
    for (int i = 1; i < argc; ++i) {
        if (QString::fromLocal8Bit(argv[i]) == QStringLiteral("--smoke")) std::_Exit(0);
    }
    return app.exec();
}
