#include <QCoreApplication>
#include <QStandardPaths>
#include <QElapsedTimer>
#include <QFile>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QThread>
#include <QUrl>
#include <QSettings>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDir>

#include <stdexcept>

#include "../model_catalog.hpp"

struct ModelCatalogTestAccess {
    static void releases(ModelCatalog& catalog, const QByteArray& payload) {
        catalog.parseGitHubReleases(payload);
    }
    static void localModels(ModelCatalog& catalog, const QStringList& paths) {
        catalog.local_models_.clear();
        catalog.local_paths_.clear();
        for (const auto& path : paths) {
            ModelCatalog::LocalModel model;
            if (!catalog.readLocalModel(path, &model)) throw std::runtime_error("invalid test model");
            catalog.local_models_.push_back(model);
            catalog.local_paths_.insert(model.id, model.path);
        }
        catalog.rebuildRows();
    }
    static bool checkIdentity(ModelCatalog& catalog, const QString& path, const QString& modelDigest) {
        ModelCatalog::LocalModel model;
        if (!catalog.readLocalModel(path, &model)) throw std::runtime_error("invalid identity fixture");
        insertArtifact(catalog, model.id, model.version, modelDigest, model.runtime_digest);
        catalog.download_id_ = model.id;
        QString error;
        return catalog.validateDownloadedModel(path, &error);
    }
    static void isolateDownload(ModelCatalog& catalog, const QString& root, bool github) {
        catalog.local_root_ = root;
        catalog.local_models_.clear();
        catalog.local_paths_.clear();
        auto& artifact = catalog.artifacts_[QStringLiteral("soo/2.0.0")];
        if (github) artifact.hugging_face = false;
        else artifact.github = false;
        catalog.rebuildRows();
    }
    static void downloadFile(ModelCatalog& catalog, const QString& staging, const QUrl& url) {
        catalog.download_staging_ = staging;
        // Leave the artifact transaction open: this test checks transport and saving.
        catalog.download_pending_ = 2;
        catalog.requestDownloadFile(QStringLiteral("metadata.json"), url);
    }
    static bool fileFinished(const ModelCatalog& catalog) { return catalog.download_pending_ == 1; }
    static QString downloadError(const ModelCatalog& catalog) { return catalog.download_error_; }
    static void addUnselectedCandidate(ModelCatalog& catalog) {
        ModelCatalog::LocalModel candidate = catalog.local_models_.front();
        candidate.id = QStringLiteral("soo/9.0.0");
        candidate.version = QStringLiteral("9.0.0");
        candidate.training_step = 999999;
        catalog.local_models_.push_back(candidate);
        catalog.local_paths_.insert(candidate.id, candidate.path);
        catalog.selected_id_.clear();
        catalog.selected_path_.clear();
    }
    static QStringList parseFileTree(ModelCatalog& catalog, const QByteArray& payload) {
        catalog.parseHuggingFaceTree(payload);
        return catalog.hugging_face_files_.value(QStringLiteral("min/2.0.0"));
    }
    static void insertArtifact(ModelCatalog& catalog, const QString& id, const QString& version,
                               const QString& modelDigest, const QString& runtimeDigest) {
        ModelCatalog::Artifact artifact;
        artifact.id = id;
        artifact.version = version;
        artifact.model_digest = modelDigest;
        artifact.runtime_digest = runtimeDigest;
        catalog.artifacts_.insert(artifact.id, artifact);
    }

    static bool parseRatings(ModelCatalog& catalog, const QByteArray& payload, bool persist) {
        return catalog.parseHuggingFaceRatings(payload, persist);
    }
};

namespace {
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
void waitForCatalog(ModelCatalog& catalog, int timeout = 60000) {
    QElapsedTimer timer;
    timer.start();
    while (catalog.busy() && timer.elapsed() < timeout) {
        QCoreApplication::processEvents();
        QThread::msleep(1);
    }
    require(!catalog.busy(), "catalog operation must finish within timeout");
}
void liveDownload(bool github) {
    QTemporaryDir destination;
    require(QDir::setCurrent(destination.path()), "isolate live download from packaged models");
    ModelCatalog catalog;
    catalog.refresh();
    waitForCatalog(catalog);
    bool historical = false;
    bool found = false;
    for (const auto& row : catalog.models()) {
        const auto model = row.toMap();
        if (model.value("id") == "soo/2.0.0-alpha.1")
            historical = model.value("github").toBool() && !model.value("compatible").toBool();
        if (model.value("id") == "soo/2.0.0") {
            require(model.value("github").toBool() && model.value("huggingFace").toBool(),
                    "live catalog must merge both sources into the Soo row");
            found = true;
        }
    }
    require(found, "live catalog lists released Soo model");
    require(historical,
            "live catalog must list historical GitHub checkpoints separately from native bundles");
    ModelCatalogTestAccess::isolateDownload(catalog, destination.path(), github);
    catalog.downloadModel(QStringLiteral("soo/2.0.0"));
    waitForCatalog(catalog);
    qInfo("live download: %s", qPrintable(catalog.status()));
    require(QFile::exists(destination.filePath(QStringLiteral("soo/2.0.0/metadata.json"))),
            "live download atomically installs validated model");
    catalog.selectModel(QStringLiteral("soo/2.0.0"));
    catalog.activateSelected(QStringLiteral("soo"));
    require(catalog.activeModelPath() == destination.filePath(QStringLiteral("soo/2.0.0")),
            "downloaded model can be selected and activated");
}
QString payload(const QString& modelDigest, const QString& runtimeDigest, double elo) {
    return QStringLiteral(
               R"({"schema_version":2,"ratings":[{"elo":%1,"full_identity":{"model_family":"soo","model_version":"2.0.0","model_sha256":"%2","runtime_sha256":"%3"}}]})")
        .arg(elo, 0, 'f', 2)
        .arg(modelDigest, runtimeDigest);
}
void downloadResponse(bool compressed) {
    QTcpServer server;
    require(server.listen(QHostAddress::LocalHost), "local HTTP server starts");
    QObject::connect(&server, &QTcpServer::newConnection, &server, [&] {
        auto* socket = server.nextPendingConnection();
        QObject::connect(socket, &QTcpSocket::readyRead, socket, [socket, compressed] {
            if (!socket->peek(socket->bytesAvailable()).contains("\r\n\r\n")) return;
            socket->readAll();
            if (compressed) {
                socket->write("HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 22\r\nConnection: close\r\n\r\n");
                socket->write(QByteArray::fromHex("1f8b0800000000000203abae050043bfa6a302000000"));
            } else {
                socket->write("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                              "2\r\n{}\r\n0\r\n\r\n");
            }
            socket->disconnectFromHost();
        });
    });
    QTemporaryDir staging;
    require(staging.isValid(), "download staging created");
    ModelCatalog catalog;
    ModelCatalogTestAccess::downloadFile(catalog, staging.path(),
        QUrl(QStringLiteral("http://127.0.0.1:%1/metadata.json").arg(server.serverPort())));
    QElapsedTimer timer;
    timer.start();
    while (!ModelCatalogTestAccess::fileFinished(catalog) && timer.elapsed() < 3000) {
        QCoreApplication::processEvents();
        QThread::msleep(1);
    }
    require(ModelCatalogTestAccess::fileFinished(catalog), "chunked download completes");
    require(ModelCatalogTestAccess::downloadError(catalog).isEmpty(),
            "compressed or missing Content-Length must not reject a valid download");
    QFile saved(staging.filePath(QStringLiteral("metadata.json")));
    require(saved.open(QIODevice::ReadOnly) && saved.readAll() == "{}",
            "chunked download saves the received bytes");
}
} // namespace

int main(int argc, char** argv) {
    QCoreApplication application(argc, argv);
    QStandardPaths::setTestModeEnabled(true);
    QTemporaryDir settings;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settings.path());
    try {
        if (application.arguments().contains(QStringLiteral("--live-github")) ||
            application.arguments().contains(QStringLiteral("--live-huggingface"))) {
            liveDownload(application.arguments().contains(QStringLiteral("--live-github")));
            return 0;
        }
        downloadResponse(false);
        downloadResponse(true);
        ModelCatalog treeCatalog;
        ModelCatalogTestAccess::releases(
            treeCatalog,
            R"([{"tag_name":"soo-v2.0.0-alpha.1","html_url":"https://github.com/ubraket513/alphadiamond/releases/tag/soo-v2.0.0-alpha.1"},{"tag_name":"soo-v9.0.0","draft":true}])");
        bool historicalFound = false;
        for (const auto& value : treeCatalog.models()) {
            const auto row = value.toMap();
            if (row.value("id") == "soo/2.0.0-alpha.1") {
                historicalFound = true;
                require(!row.value("compatible").toBool() && row.value("github").toBool(),
                        "training checkpoint must not be offered as a native runtime model");
            }
            require(row.value("id") != "soo/9.0.0", "draft releases are private to publishers");
        }
        require(historicalFound,
                "release discovery must include versions absent from the current index");
        require(ModelCatalogTestAccess::parseFileTree(treeCatalog,
                    R"([{"type":"directory","path":"models/min/2.0.0/weights"},{"type":"file","path":"models/min/2.0.0/weights/a.f32"}])")
                    == QStringList{QStringLiteral("models/min/2.0.0/weights/a.f32")},
                "bucket directories must not become downloadable files");
        const QString modelDigest(64, QLatin1Char('a'));
        const QString runtimeDigest(64, QLatin1Char('b'));
        ModelCatalog catalog;
        ModelCatalogTestAccess::localModels(catalog, {});
        ModelCatalogTestAccess::insertArtifact(catalog, QStringLiteral("soo/2.0.0"),
                                               QStringLiteral("2.0.0"), modelDigest, runtimeDigest);

        require(ModelCatalogTestAccess::parseRatings(
                    catalog, payload(modelDigest, runtimeDigest, 1203.5).toUtf8(), false),
                "matching ratings JSON is accepted");
        require(catalog.models().front().toMap().value(QStringLiteral("latestElo")).toString() ==
                    QStringLiteral("1203.50"),
                "rating matches full artifact identity");
        require(ModelCatalogTestAccess::parseRatings(
                    catalog, payload(QString(64, QLatin1Char('c')), runtimeDigest, 9999.0).toUtf8(),
                    false),
                "different artifact ratings JSON is accepted");
        require(catalog.models()
                    .front()
                    .toMap()
                    .value(QStringLiteral("latestElo"))
                    .toString()
                    .isEmpty(),
                "different digest never substitutes a rating");
        require(!ModelCatalogTestAccess::parseRatings(catalog, QByteArrayLiteral("{}"), false),
                "invalid response is rejected");
        const auto soo = QDir::current().filePath(QStringLiteral("models/soo/2.0.0"));
        ModelCatalog identityCatalog;
        require(!ModelCatalogTestAccess::checkIdentity(identityCatalog, soo, QString(64, 'f')),
                "matching runtime digest must not accept a different model identity");
        ModelCatalog selection;
        ModelCatalogTestAccess::localModels(selection, {soo, soo});
        require(selection.models().size() == 1, "duplicate local paths must list one model row");
        selection.selectModel(QStringLiteral("soo/2.0.0"));
        selection.activateSelected(QStringLiteral("soo"));
        selection.activateSelected(QStringLiteral("min"));
        require(selection.activeModelPath().isEmpty(), "Soo selection must not activate for Min");
        selection.activateSelected(QStringLiteral("soo"));
        require(selection.activeModelId() == QStringLiteral("soo/2.0.0"),
                "switching game families restores the compatible selection");
        QSettings().clear();
        ModelCatalogTestAccess::addUnselectedCandidate(selection);
        selection.activateSelected(QStringLiteral("soo"));
        require(selection.activeModelId() == QStringLiteral("soo/2.0.0"),
                "an unselected higher-step candidate must not override the indexed default");
        QTemporaryDir invalidIndex;
        QDir().mkpath(invalidIndex.filePath(QStringLiteral("models")));
        QFile index(invalidIndex.filePath(QStringLiteral("models/index.json")));
        require(index.open(QIODevice::WriteOnly) && index.write("{}") == 2,
                "malformed index fixture is written");
        index.close();
        const auto originalDirectory = QDir::currentPath();
        require(QDir::setCurrent(invalidIndex.path()), "malformed index directory opens");
        ModelCatalog invalidDefault;
        ModelCatalogTestAccess::localModels(invalidDefault, {soo});
        invalidDefault.activateSelected(QStringLiteral("soo"));
        require(invalidDefault.activeModelId().isEmpty(),
                "an invalid model index must not silently activate a local fallback");
        QDir::setCurrent(originalDirectory);
    } catch (const std::exception& error) {
        qCritical("model_catalog_test: %s", error.what());
        return 1;
    }
    return 0;
}
