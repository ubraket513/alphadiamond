#include <QGuiApplication>
#include <QFont>
#include <QFontDatabase>
#include <QIcon>
#include <QPixmap>
#include <QImage>
#include <QMouseEvent>
#include <QQuickImageProvider>
#include <QQuickItem>
#include <QQuickWindow>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQmlProperty>
#include <QQuickStyle>
#include <QPainter>
#include <QPen>
#include <QTimer>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "native_controller.hpp"
#include "native_chrome.hpp"

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
    for (int i = 1; i < argc; ++i)
        smoke_mode = smoke_mode || QString::fromLocal8Bit(argv[i]).endsWith(QStringLiteral("-smoke"));
    qputenv("QT_QPA_PLATFORM", smoke_mode ? QByteArrayLiteral("offscreen")
                                           : QByteArrayLiteral("windows"));
#ifdef Q_OS_WIN
    if (qEnvironmentVariableIsEmpty("QT_MEDIA_BACKEND"))
        qputenv("QT_MEDIA_BACKEND", QByteArrayLiteral("windows"));
#endif
    NativeChrome::setAppUserModelId();
    QGuiApplication app(argc, argv);
    app.setQuitOnLastWindowClosed(true);
    app.setApplicationName(QStringLiteral("Diamond Controller"));
    app.setOrganizationName(QStringLiteral("alphadiamond"));
    QQuickStyle::setStyle(QStringLiteral("Basic"));

    QStringList font_families;
    for (const QString& resource : {QStringLiteral(":/fonts/GoogleSansFlex-Regular.ttf"),
                                    QStringLiteral(":/fonts/GoogleSansFlex-Medium.ttf"),
                                    QStringLiteral(":/fonts/GoogleSansFlex-Bold.ttf")}) {
        const int id = QFontDatabase::addApplicationFont(resource);
        if (id >= 0) font_families.append(QFontDatabase::applicationFontFamilies(id));
    }
    const QString app_font = font_families.contains(QStringLiteral("Google Sans Flex"))
        ? QStringLiteral("Google Sans Flex")
        : font_families.value(0, QStringLiteral("Segoe UI"));
    app.setFont(QFont(app_font));

    QPixmap app_icon_pixmap(256, 256);
    app_icon_pixmap.fill(Qt::transparent);
    {
        QPainter painter(&app_icon_pixmap);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(QStringLiteral("#FF3B30")));
        QPolygon diamond;
        diamond << QPoint(128, 16) << QPoint(240, 128) << QPoint(128, 240) << QPoint(16, 128);
        painter.drawPolygon(diamond);
    }
    app.setWindowIcon(QIcon(app_icon_pixmap));

    NativeController controller;
    NativeChrome native_chrome;
    app.installNativeEventFilter(&native_chrome);
    QObject::connect(&app, &QCoreApplication::aboutToQuit, &controller, &NativeController::shutdown);
    QQmlApplicationEngine engine;
    engine.addImportPath(QStringLiteral("qrc:/qml"));
    engine.addImageProvider(QStringLiteral("qta"), new PlaceholderIconProvider);
    engine.rootContext()->setContextProperty(QStringLiteral("controller"), &controller);
    engine.rootContext()->setContextProperty(QStringLiteral("appFontFamily"),
                                              app_font);
    engine.rootContext()->setContextProperty(QStringLiteral("nativeChrome"), &native_chrome);
    engine.load(QUrl(QStringLiteral("qrc:/qml/Main.qml")));
    if (engine.rootObjects().isEmpty()) return 1;
    QObject* root = engine.rootObjects().constFirst();
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
        if (!smoke_mode) {
            native_chrome.attach(window);
            NativeChrome::enableShellIntegration(window);
            NativeChrome::applyDwmAppearance(window);
        }
        QTimer::singleShot(100, window, [window] {
            if (!window->isVisible()) {
                window->setVisibility(QWindow::Windowed);
                window->showNormal();
                window->raise();
                window->requestActivate();
            }
        });
    }
    for (int i = 1; i < argc; ++i) {
        const QString argument = QString::fromLocal8Bit(argv[i]);
        if (argument == QStringLiteral("--analysis-smoke")) {
            const auto require_analysis = [](bool condition, const char* message) {
                if (!condition) {
                    std::fprintf(stderr, "analysis smoke failed: %s\n", message);
                    std::fflush(stderr);
                }
                return condition;
            };
            QObject* console = root->findChild<QObject*>(QStringLiteral("analysisConsole"));
            QObject* footer = root->findChild<QObject*>(QStringLiteral("analysisFooter"));
            QObject* outlook = root->findChild<QObject*>(QStringLiteral("positionOutlookPanel"));
            QObject* decision = root->findChild<QObject*>(QStringLiteral("decisionValuePanel"));
            QObject* preference = root->findChild<QObject*>(QStringLiteral("movePreferencePanel"));
            QObject* outlook_chart = root->findChild<QObject*>(QStringLiteral("positionOutlookChart"));
            QObject* decision_chart = root->findChild<QObject*>(QStringLiteral("decisionValueChart"));
            QObject* preference_chart = root->findChild<QObject*>(QStringLiteral("movePreferenceChart"));
            QObject* compute = root->findChild<QObject*>(QStringLiteral("searchComputePanel"));
            QObject* title_bar = root->findChild<QObject*>(QStringLiteral("titleBar"));
            QObject* drawer = root->findChild<QObject*>(QStringLiteral("historyDrawer"));
            QObject* history = root->findChild<QObject*>(QStringLiteral("drawerHistoryPanel"));
            if (!require_analysis(console && footer && outlook && decision && preference && compute,
                                  "analysis layout components are missing") ||
                !require_analysis(outlook_chart && decision_chart && preference_chart,
                                  "analysis charts are missing") ||
                !require_analysis(title_bar && drawer && history, "history drawer is missing")) return 1;
            if (!require_analysis(
                    console->findChild<QObject*>(QStringLiteral("positionOutlookPanel")) == nullptr &&
                    console->findChild<QObject*>(QStringLiteral("decisionValuePanel")) == nullptr &&
                    console->findChild<QObject*>(QStringLiteral("movePreferencePanel")) == nullptr,
                    "analysis monitors still belong to the sidebar") ||
                !require_analysis(
                    footer->findChild<QObject*>(QStringLiteral("positionOutlookPanel")) &&
                    footer->findChild<QObject*>(QStringLiteral("decisionValuePanel")) &&
                    footer->findChild<QObject*>(QStringLiteral("movePreferencePanel")),
                    "analysis monitors are not in the footer") ||
                !require_analysis(outlook_chart->property("height").toDouble() >= 150.0,
                                  "footer analysis chart is too small")) return 1;
            if (!QQmlProperty::write(outlook_chart, QStringLiteral("points"), QVariantList{
                QVariantMap{{"available", true}, {"ply", 1}, {"nnEstimate", 0.50},
                            {"mctsEstimate", 0.40}},
                QVariantMap{{"available", true}, {"ply", 2}, {"nnEstimate", 0.60},
                            {"mctsEstimate", 0.35}},
            })) {
                std::fprintf(stderr, "analysis smoke failed: points property is not writable\n");
                return 1;
            }
            QCoreApplication::processEvents();
            const auto current_value = outlook_chart->property("firstValueText").toString();
            const auto current_delta = outlook_chart->property("firstDeltaText").toString();
            if (!require_analysis(current_value == QStringLiteral("60.00%"),
                                  qPrintable(QStringLiteral(
                                                 "analysis chart current value is not exact: value=%1 index=%2 key=%3 points=%4")
                                                 .arg(current_value)
                                                 .arg(outlook_chart->property("displayIndex").toInt())
                                                 .arg(outlook_chart->property("firstKey").toString())
                                                 .arg(outlook_chart->property("points").toList().size()))) ||
                !require_analysis(current_delta == QStringLiteral("+10.00% up"),
                                  qPrintable(QStringLiteral("analysis chart delta is not exact: %1")
                                                 .arg(current_delta)))) return 1;
            if (!require_analysis(
                    console->findChild<QObject*>(QStringLiteral("drawerHistoryPanel")) == nullptr,
                    "history still belongs to permanent analysis console")) return 1;
            if (!QMetaObject::invokeMethod(title_bar, "historyRequested")) {
                std::fprintf(stderr, "analysis smoke failed: history command is not wired\n");
                return 1;
            }
            QCoreApplication::processEvents();
            if (!require_analysis(drawer->property("open").toBool(),
                                  "history drawer did not open")) return 1;
            drawer->setProperty("open", false);
            QCoreApplication::processEvents();
            return require_analysis(!drawer->property("open").toBool(),
                                    "history drawer did not close") ? 0 : 1;
        }
        if (argument == QStringLiteral("--rotation-smoke")) {
            const auto require_rotation = [](bool condition, const char* message) {
                if (!condition) {
                    std::fprintf(stderr, "rotation smoke failed: %s\n", message);
                    std::fflush(stderr);
                }
                return condition;
            };
            QObject* rotation_control = root->findChild<QObject*>(
                QStringLiteral("boardRotationControl"));
            QObject* rotation_dial = root->findChild<QObject*>(
                QStringLiteral("boardRotationDial"));
            QObject* rotating_board = root->findChild<QObject*>(
                QStringLiteral("rotatingBoard"));
            if (!require_rotation(rotation_control && rotation_dial && rotating_board,
                                  "QML rotation objects are missing")) return 1;

            rotation_dial->setProperty("value", 137.5);
            QCoreApplication::processEvents();
            if (!require_rotation(
                    std::abs(rotating_board->property("rotation").toDouble() - 137.5) <= 0.01,
                    "clockwise angle did not reach the board")) return 1;

            rotation_dial->setProperty("value", -121.25);
            QCoreApplication::processEvents();
            if (!require_rotation(
                    std::abs(rotating_board->property("rotation").toDouble() + 121.25) <= 0.01,
                    "counterclockwise angle did not reach the board")) return 1;

            auto* board_model = qobject_cast<ContractListModel*>(controller.boardModel());
            if (!require_rotation(board_model, "board model is unavailable")) return 1;
            int occupied_human_position = -1;
            for (const QVariant& row : board_model->rows()) {
                const QVariantMap values = row.toMap();
                if (values.value(QStringLiteral("occupant")).toInt() ==
                    controller.currentPlayerId()) {
                    occupied_human_position = values.value(
                        QStringLiteral("positionId")).toInt();
                    break;
                }
            }
            auto* rotating_item = qobject_cast<QQuickItem*>(rotating_board);
            const QString hole_name = QStringLiteral("boardHole-%1").arg(
                occupied_human_position);
            const auto find_visual_item = [&hole_name](auto&& self, QQuickItem* parent)
                -> QQuickItem* {
                if (!parent) return nullptr;
                if (parent->objectName() == hole_name) return parent;
                for (QQuickItem* child : parent->childItems())
                    if (QQuickItem* match = self(self, child)) return match;
                return nullptr;
            };
            auto* hole = find_visual_item(find_visual_item, rotating_item);
            auto* window = qobject_cast<QQuickWindow*>(root);
            if (!require_rotation(occupied_human_position >= 0, "no human piece was found") ||
                !require_rotation(hole, "rotated hole delegate is unavailable") ||
                !require_rotation(window, "QML root is not a window")) return 1;

            rotation_dial->setProperty("value", 73.25);
            QCoreApplication::processEvents();
            const QPointF scene_position = hole->mapToScene(
                QPointF(hole->width() / 2.0, hole->height() / 2.0));
            QMouseEvent press(QEvent::MouseButtonPress, scene_position, scene_position,
                              Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
            QMouseEvent release(QEvent::MouseButtonRelease, scene_position, scene_position,
                                Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
            QCoreApplication::sendEvent(window, &press);
            QCoreApplication::sendEvent(window, &release);
            QCoreApplication::processEvents();
            if (!require_rotation(controller.selectedPosition() == occupied_human_position,
                                  "clicking the rotated hole did not select its piece")) return 1;

            QList<QQuickItem*> holes;
            const auto collect_holes = [&holes](auto&& self, QQuickItem* parent) -> void {
                if (!parent) return;
                if (parent->objectName().startsWith(QStringLiteral("boardHole-")))
                    holes.push_back(parent);
                for (QQuickItem* child : parent->childItems()) self(self, child);
            };
            collect_holes(collect_holes, rotating_item);
            QQuickItem* board_item = rotating_item->parentItem();
            if (!require_rotation(holes.size() == soo::kBoardSize && board_item,
                                  "not every board hole is present in the rotating surface")) return 1;
            for (int angle = -180; angle <= 180; angle += 15) {
                rotation_dial->setProperty("value", angle);
                QCoreApplication::processEvents();
                for (QQuickItem* candidate : holes) {
                    const QPointF center = candidate->mapToItem(
                        board_item, QPointF(candidate->width() / 2.0,
                                            candidate->height() / 2.0));
                    const double radius = candidate->property("socketRadius").toDouble();
                    if (!require_rotation(center.x() - radius >= 0.0 &&
                                          center.y() - radius >= 0.0 &&
                                          center.x() + radius <= board_item->width() &&
                                          center.y() + radius <= board_item->height(),
                                          "a rotated socket leaves the board panel")) return 1;
                }
            }

            if (!require_rotation(QMetaObject::invokeMethod(rotation_control, "resetRotation"),
                                  "resetRotation is not invokable")) return 1;
            QCoreApplication::processEvents();
            return require_rotation(
                std::abs(rotating_board->property("rotation").toDouble()) < 0.01,
                "reset did not return the board to zero") ? 0 : 1;
        }
        if (argument == QStringLiteral("--sound-smoke")) {
            QObject::connect(&controller, &NativeController::changed, &app, [&controller, &app] {
                if (controller.soundLoaded()) app.exit(0);
                else if (!controller.soundStatus().isEmpty()) app.exit(1);
            });
            QTimer::singleShot(5000, &app, [&app] { app.exit(1); });
            controller.previewSound();
            if (controller.soundLoaded()) return 0;
            if (!controller.soundStatus().isEmpty()) return 1;
            return app.exec();
        }
        if (argument == QStringLiteral("--failure-smoke")) {
            const int starts_before = controller.aiSearchStartCount();
            QObject::connect(&controller, &NativeController::changed, &app,
                [&controller, &app, starts_before] {
                    if (controller.aiStatus() != QStringLiteral("Error")) return;
                    QTimer::singleShot(250, &app, [&controller, &app, starts_before] {
                        app.exit(!controller.aiThinking() &&
                                 controller.aiStatus() == QStringLiteral("Error") &&
                                 controller.aiSearchStartCount() == starts_before + 1 ? 0 : 1);
                    });
                });
            if (!controller.failureSmoke()) return 1;
            QTimer::singleShot(5000, &app, [&app] { app.exit(1); });
            return app.exec();
        }
        if (argument == QStringLiteral("--game-smoke"))
            return controller.gameSmoke() ? 0 : 1;
        if (argument == QStringLiteral("--worker-smoke"))
            return controller.workerSmoke() ? 0 : 1;
        if (argument == QStringLiteral("--soo-smoke"))
            return controller.sooSmoke() ? 0 : 1;
        if (argument == QStringLiteral("--smoke")) std::_Exit(0);
    }
    return app.exec();
}
