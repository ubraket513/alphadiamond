#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSettings>
#include <QTemporaryDir>

#include <filesystem>
#include <iostream>

#include "rating_bridge.hpp"

namespace {
bool require(bool value, const char* message) {
    if (!value) std::cerr << "rating_bridge_test: " << message << '\n';
    return value;
}
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    qputenv("DIAMOND_RATING_SYNC_DISABLED", QByteArrayLiteral("1"));
    QTemporaryDir temporary;
    if (!require(temporary.isValid(), "temporary root unavailable")) return 1;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temporary.path());
    QCoreApplication::setOrganizationName(QStringLiteral("AlphaDiamondTest"));
    QCoreApplication::setApplicationName(QStringLiteral("RatingBridge"));

    const QString model = QDir(temporary.path()).filePath(QStringLiteral("model"));
    QDir().mkpath(model);
    QFile metadata{QDir(model).filePath(QStringLiteral("metadata.json"))};
    if (!require(metadata.open(QIODevice::WriteOnly), "metadata unavailable")) return 1;
    metadata.write(QJsonDocument(QJsonObject{{"model_sha256", QString(64, QLatin1Char('b'))},
                                               {"runtime_sha256", QString(64, QLatin1Char('a'))}}).toJson());
    metadata.close();

    RatingBridge bridge{std::filesystem::path(temporary.filePath(QStringLiteral("ratings")).toStdString())};
    const QVariantList order{1, 2};
    const bool published = bridge.recordTerminalSooMatch(
        QStringLiteral("local-game-1"), model, QStringLiteral("Soo/2.0.0"),
        QStringLiteral("Soo v2.0.0"), 2, 1, order);
    if (!require(published, "terminal model game did not publish a durable event")) {
        std::cerr << bridge.syncState().toStdString() << '\n';
        return 1;
    }
    if (!require(!bridge.recordTerminalSooMatch(QStringLiteral("local-game-1"), model,
                                                QStringLiteral("Soo/2.0.0"), QStringLiteral("Soo v2.0.0"),
                                                2, 1, order),
                 "same terminal game was not idempotent")) return 1;
    if (!require(bridge.pendingEventCount() == 1, "exactly one event was not retained for retry")) return 1;
    if (!require(!bridge.syncRunning() && bridge.syncState().contains(QStringLiteral("queued"), Qt::CaseInsensitive),
                 "missing sync launcher did not leave durable retry state")) return 1;
    return 0;
}
