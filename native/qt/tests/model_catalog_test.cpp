#include <QCoreApplication>
#include <QStandardPaths>

#include <stdexcept>

#include "../model_catalog.hpp"

struct ModelCatalogTestAccess {
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
QString payload(const QString& modelDigest, const QString& runtimeDigest, double elo) {
    return QStringLiteral(
               R"({"schema_version":2,"ratings":[{"elo":%1,"full_identity":{"model_family":"soo","model_version":"2.0.0","model_sha256":"%2","runtime_sha256":"%3"}}]})")
        .arg(elo, 0, 'f', 2)
        .arg(modelDigest, runtimeDigest);
}
} // namespace

int main(int argc, char** argv) {
    QCoreApplication application(argc, argv);
    QStandardPaths::setTestModeEnabled(true);
    try {
        const QString modelDigest(64, QLatin1Char('a'));
        const QString runtimeDigest(64, QLatin1Char('b'));
        ModelCatalog catalog;
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
    } catch (const std::exception& error) {
        qCritical("model_catalog_test: %s", error.what());
        return 1;
    }
    return 0;
}
