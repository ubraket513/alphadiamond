#include "model_catalog.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QSettings>
#include <QStandardPaths>
#include <QUrl>
#include <QUuid>

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

#ifdef DIAMOND_QT_HAS_SOO
#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/model_index.hpp"
#endif

namespace {
constexpr auto kRepository = "ubraket513/alphadiamond";
constexpr auto kHuggingFaceDataset = "ubraket513/AlphaDiamond";
const QRegularExpression kSemVer(
    QStringLiteral("^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(?:-(?:0|[1-9][0-9]*|["
                   "0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)(?:\\.(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-"
                   "Za-z-][0-9A-Za-z-]*))*)?(?:\\+[0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*)?$"));
const QRegularExpression kDigest(QStringLiteral("^[0-9a-fA-F]{64}$"));
QString modelLabel(const QString& id) {
    const QStringList parts = id.split('/');
    if (parts.size() != 2)
        return id;
    QString family = parts.at(0);
    if (!family.isEmpty())
        family[0] = family.at(0).toUpper();
    return QStringLiteral("%1 %2").arg(family, parts.at(1));
}
QByteArray readAll(const QString& path) {
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray{};
}
bool safeRelativePath(const QString& path) {
    if (path.isEmpty() || QDir::isAbsolutePath(path) || path.contains(QLatin1Char('\\')))
        return false;
    const QString clean = QDir::cleanPath(path);
    return clean == path && clean != QStringLiteral(".") && clean != QStringLiteral("..") && !clean.startsWith(QStringLiteral("../")) &&
           !clean.contains(QStringLiteral("/../"));
}
int metaInt(const QJsonObject& root, const QJsonObject& source, const char* key) {
    const QJsonValue value = source.value(QLatin1String(key));
    return value.isDouble() ? value.toInt() : root.value(QLatin1String(key)).toInt();
}
QString ratingIdentityKey(const QString& id, const QString& modelDigest,
                          const QString& runtimeDigest) {
    return id + QChar(0x1f) + modelDigest.toLower() + QChar(0x1f) + runtimeDigest.toLower();
}
QString ratingIdentityKey(const QJsonObject& identity) {
    QList<QJsonObject> candidates{identity};
    for (const char* key : {"artifact", "deployment", "model"}) {
        const QJsonObject nested = identity.value(QLatin1String(key)).toObject();
        if (!nested.isEmpty())
            candidates.push_back(nested);
    }
    for (const QJsonObject& candidate : candidates) {
        const QString family = candidate.value(QStringLiteral("model_family"))
                                   .toString(candidate.value(QStringLiteral("family")).toString());
        const QString version =
            candidate.value(QStringLiteral("model_version"))
                .toString(candidate.value(QStringLiteral("version")).toString());
        const QString modelDigest = candidate.value(QStringLiteral("model_sha256")).toString();
        const QString runtimeDigest = candidate.value(QStringLiteral("runtime_sha256")).toString();
        if (!family.isEmpty() && kSemVer.match(version).hasMatch() &&
            kDigest.match(modelDigest).hasMatch() && kDigest.match(runtimeDigest).hasMatch())
            return ratingIdentityKey(family + QLatin1Char('/') + version, modelDigest,
                                     runtimeDigest);
    }
    return {};
}
} // namespace

ModelCatalog::ModelCatalog(QObject* parent)
    : QObject(parent), network_(new QNetworkAccessManager(this)),
      local_root_(QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
                      .filePath(QStringLiteral("models"))) {
    network_->setTransferTimeout(30000);
    QDir().mkpath(local_root_);
    QSettings settings;
    selected_id_ = settings.value(QStringLiteral("models/selectedId")).toString();
    selected_path_ = settings.value(QStringLiteral("models/selectedPath")).toString();
    loadCachedRatings();
    scanLocal();
    active_id_ = selected_id_;
    active_path_ = selected_path_;
    rebuildRows();
}
QString ModelCatalog::selectedModelLabel() const {
    return selected_id_.isEmpty() ? QStringLiteral("None") : modelLabel(selected_id_);
}
QString ModelCatalog::activeModelLabel() const {
    return active_id_.isEmpty() ? QStringLiteral("None") : modelLabel(active_id_);
}
void ModelCatalog::setStatus(const QString& status) {
    status_ = status;
    Q_EMIT changed();
}
void ModelCatalog::beginWork() {
    ++busy_count_;
    Q_EMIT changed();
}
void ModelCatalog::endWork() {
    if (busy_count_ > 0)
        --busy_count_;
    if (!busy() && status_ == QStringLiteral("Refreshing GitHub and Hugging Face…"))
        status_ = QStringLiteral("Model catalog ready.");
    Q_EMIT changed();
}

bool ModelCatalog::readLocalModel(const QString& path, LocalModel* model) const {
    const QJsonDocument document =
        QJsonDocument::fromJson(readAll(QDir(path).filePath(QStringLiteral("metadata.json"))));
    if (!document.isObject())
        return false;
    const QJsonObject root = document.object(),
                      source = root.value(QStringLiteral("source")).toObject();
    const QString family = root.value(QStringLiteral("model_family")).toString();
    const QString version = root.value(QStringLiteral("model_version")).toString();
    if ((family != QStringLiteral("soo") && family != QStringLiteral("min")) ||
        !kSemVer.match(version).hasMatch())
        return false;
#ifdef DIAMOND_QT_HAS_SOO
    try {
        diamond_model::validate_deployment_artifact(std::filesystem::path(path.toStdString()),
                                                    family.toStdString());
    } catch (const std::exception&) {
        return false;
    }
#endif
    model->id = family + QLatin1Char('/') + version;
    model->path = QDir::cleanPath(path);
    model->version = version;
    model->model_digest = root.value(QStringLiteral("model_sha256")).toString();
    model->runtime_digest = root.value(QStringLiteral("runtime_sha256")).toString();
    model->training_step = metaInt(root, source, "training_step");
    model->training_simulations = metaInt(root, source, "training_simulations");
    return true;
}
void ModelCatalog::scanLocal() {
    local_models_.clear();
    local_paths_.clear();
    QSet<QString> visited;
    const QStringList bases = {
        QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("models")),
        QDir(QDir::currentPath()).filePath(QStringLiteral("models")), local_root_};
    for (const QString& base : bases) {
        QDirIterator it(base, QStringList{QStringLiteral("metadata.json")}, QDir::Files,
                        QDirIterator::Subdirectories);
        while (it.hasNext()) {
            const QString root = QFileInfo(it.next()).absolutePath();
            const QString canonical = QFileInfo(root).canonicalFilePath();
            const QString unique = canonical.isEmpty() ? QDir::cleanPath(root) : canonical;
            if (visited.contains(unique))
                continue;
            visited.insert(unique);
            LocalModel model;
            if (readLocalModel(root, &model)) {
                local_models_.push_back(model);
                if (!local_paths_.contains(model.id))
                    local_paths_.insert(model.id, model.path);
            }
        }
    }
    if (!local_paths_.contains(selected_id_) || !QFileInfo::exists(selected_path_)) {
        selected_id_.clear();
        selected_path_.clear();
    }
}
void ModelCatalog::rebuildRows() {
    QVariantList rows;
    QSet<QString> known;
    for (auto it = artifacts_.cbegin(); it != artifacts_.cend(); ++it) {
        const Artifact& a = it.value();
        const bool installed = local_paths_.contains(a.id);
        rows.push_back(
            QVariantMap{{"id", a.id},
                        {"name", modelLabel(a.id)},
                        {"version", a.version},
                        {"trainingStep", a.training_step},
                        {"trainingSimulations", a.training_simulations},
                        {"latestElo", latestRating(a.id, a.model_digest, a.runtime_digest)},
                        {"installed", installed},
                        {"compatible", true},
                        {"selected", a.id == selected_id_},
                        {"active", a.id == active_id_},
                        {"github", a.github},
                        {"huggingFace", a.hugging_face},
                        {"githubUrl", a.github_url},
                        {"huggingFaceUrl", a.hugging_face_url}});
        known.insert(a.id);
    }
    for (const LocalModel& a : local_models_)
        if (!known.contains(a.id)) {
            rows.push_back(
                QVariantMap{{"id", a.id},
                            {"name", modelLabel(a.id)},
                            {"version", a.version},
                            {"trainingStep", a.training_step},
                            {"trainingSimulations", a.training_simulations},
                            {"latestElo", latestRating(a.id, a.model_digest, a.runtime_digest)},
                            {"installed", true},
                            {"compatible", true},
                            {"selected", a.id == selected_id_},
                            {"active", a.id == active_id_},
                            {"github", false},
                            {"huggingFace", false},
                            {"githubUrl", QString()},
                            {"huggingFaceUrl", QString()}});
            known.insert(a.id);
        }
    std::sort(rows.begin(), rows.end(), [](const QVariant& left, const QVariant& right) {
        const auto a = left.toMap(), b = right.toMap();
        const auto af = a.value(QStringLiteral("id")).toString().section('/', 0, 0);
        const auto bf = b.value(QStringLiteral("id")).toString().section('/', 0, 0);
        if (af != bf) return af < bf;
        if (a.value(QStringLiteral("trainingStep")) != b.value(QStringLiteral("trainingStep")))
            return a.value(QStringLiteral("trainingStep")).toInt() > b.value(QStringLiteral("trainingStep")).toInt();
        return a.value(QStringLiteral("id")).toString() < b.value(QStringLiteral("id")).toString();
    });
    models_ = rows;
    Q_EMIT changed();
}

void ModelCatalog::refresh() {
    if (busy())
        return;
    artifacts_.clear();
    github_files_.clear();
    hugging_face_files_.clear();
    catalog_error_.clear();
    scanLocal();
    rebuildRows();
    setStatus(QStringLiteral("Refreshing GitHub and Hugging Face…"));
    fetchGitHub();
    fetchHuggingFace();
    fetchHuggingFaceRatings();
}
void ModelCatalog::fetchGitHub() {
    const auto request = [this](const QUrl& url, auto parser) {
        QNetworkRequest request(url);
        request.setRawHeader("Accept", "application/vnd.github+json");
        request.setRawHeader("User-Agent", "AlphaDiamond");
        beginWork();
        QNetworkReply* reply = network_->get(request);
        connect(reply, &QNetworkReply::finished, this, [this, reply, parser] {
            if (reply->error() == QNetworkReply::NoError)
                (this->*parser)(reply->readAll());
            else
                setStatus(QStringLiteral("GitHub refresh failed: %1").arg(reply->errorString()));
            reply->deleteLater();
            endWork();
        });
    };
    request(QUrl(QStringLiteral("https://raw.githubusercontent.com/%1/main/models/index.json")
                     .arg(kRepository)),
            &ModelCatalog::parseGitHubIndex);
    request(QUrl(QStringLiteral("https://api.github.com/repos/%1/git/trees/main?recursive=1")
                     .arg(kRepository)),
            &ModelCatalog::parseGitHubTree);
}
void ModelCatalog::fetchHuggingFace() {
    QNetworkRequest request(QUrl(QStringLiteral("https://huggingface.co/api/buckets/%1/"
                                                "tree?recursive=true&expand=false")
                                     .arg(kHuggingFaceDataset)));
    request.setRawHeader("User-Agent", "AlphaDiamond");
    beginWork();
    QNetworkReply* reply = network_->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        if (reply->error() == QNetworkReply::NoError)
            parseHuggingFaceTree(reply->readAll());
        else
            setStatus(QStringLiteral("Hugging Face refresh failed: %1").arg(reply->errorString()));
        reply->deleteLater();
        endWork();
    });
}
void ModelCatalog::fetchHuggingFaceRatings() {
    QNetworkRequest request(QUrl(QStringLiteral("https://huggingface.co/buckets/%1/resolve/ratings/"
                                                "ratings.json?download=true")
                                     .arg(kHuggingFaceDataset)));
    request.setRawHeader("User-Agent", "AlphaDiamond");
    beginWork();
    QNetworkReply* reply = network_->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        if (reply->error() == QNetworkReply::NoError) {
            if (!parseHuggingFaceRatings(reply->readAll(), true))
                setStatus(
                    QStringLiteral("Hugging Face ratings are invalid; cached ratings retained."));
        } else {
            setStatus(QStringLiteral("Hugging Face ratings unavailable; cached ratings retained."));
        }
        reply->deleteLater();
        endWork();
    });
}
bool ModelCatalog::parseHuggingFaceRatings(const QByteArray& payload, bool persist) {
    const QJsonDocument document = QJsonDocument::fromJson(payload);
    if (!document.isObject())
        return false;
    const QJsonObject root = document.object();
    if (root.value(QStringLiteral("schema_version")).toInt() != 2 ||
        !root.value(QStringLiteral("ratings")).isArray())
        return false;
    QHash<QString, QString> parsed;
    for (const QJsonValue& value : root.value(QStringLiteral("ratings")).toArray()) {
        const QJsonObject row = value.toObject();
        const QString key =
            ratingIdentityKey(row.value(QStringLiteral("full_identity")).toObject());
        if (key.isEmpty())
            continue; // Human and non-artifact participants have no model row.
        const QJsonValue elo = row.value(QStringLiteral("elo"));
        if (!elo.isDouble() || !std::isfinite(elo.toDouble()) || parsed.contains(key))
            return false;
        parsed.insert(key, QString::number(elo.toDouble(), 'f', 2));
    }
    ratings_by_identity_ = std::move(parsed);
    if (persist) {
        QSettings settings;
        settings.setValue(QStringLiteral("models/huggingFaceRatings/v2"), payload);
    }
    rebuildRows();
    return true;
}
void ModelCatalog::loadCachedRatings() {
    QSettings settings;
    parseHuggingFaceRatings(
        settings.value(QStringLiteral("models/huggingFaceRatings/v2")).toByteArray(), false);
}
QString ModelCatalog::latestRating(const QString& id, const QString& modelDigest,
                                   const QString& runtimeDigest) const {
    return ratings_by_identity_.value(ratingIdentityKey(id, modelDigest, runtimeDigest));
}
void ModelCatalog::parseGitHubIndex(const QByteArray& payload) {
    const QJsonArray models =
        QJsonDocument::fromJson(payload).object().value(QStringLiteral("models")).toArray();
    for (const QJsonValue& value : models) {
        const QJsonObject metadata = value.toObject();
        addArtifact(QStringLiteral("GitHub"), metadata,
                    QStringLiteral("https://github.com/%1/tree/main/models/%2")
                        .arg(kRepository, metadata.value(QStringLiteral("path")).toString()));
    }
    rebuildRows();
}
void ModelCatalog::parseGitHubTree(const QByteArray& payload) {
    const QJsonArray tree =
        QJsonDocument::fromJson(payload).object().value(QStringLiteral("tree")).toArray();
    for (const QJsonValue& value : tree) {
        const QJsonObject item = value.toObject();
        const QString path = item.value(QStringLiteral("path")).toString();
        const QStringList p = path.split('/');
        if (item.value(QStringLiteral("type")).toString() == QStringLiteral("blob") &&
            p.size() >= 4 && p.at(0) == QStringLiteral("models") && safeRelativePath(path))
            github_files_[p.at(1) + QLatin1Char('/') + p.at(2)].push_back(path);
    }
}
void ModelCatalog::parseHuggingFaceTree(const QByteArray& payload) {
    QSet<QString> ids;
    for (const QJsonValue& value : QJsonDocument::fromJson(payload).array()) {
        const auto item = value.toObject();
        if (item.value(QStringLiteral("type")).toString() != QStringLiteral("file"))
            continue;
        const QString path = item.value(QStringLiteral("path")).toString();
        const QStringList p = path.split('/');
        if (p.size() < 4 || p.at(0) != QStringLiteral("models") || !safeRelativePath(path))
            continue;
        const QString id = p.at(1) + QLatin1Char('/') + p.at(2);
        if (!hugging_face_files_[id].contains(path))
            hugging_face_files_[id].push_back(path);
        if (p.size() == 4 && p.at(3) == QStringLiteral("metadata.json"))
            ids.insert(id);
    }
    for (const QString& id : ids)
        fetchHuggingFaceMetadata(id);
}
void ModelCatalog::fetchHuggingFaceMetadata(const QString& modelId) {
    QNetworkRequest request(
        QUrl(QStringLiteral("https://huggingface.co/buckets/%1/resolve/models/%2/"
                            "metadata.json?download=true")
                 .arg(kHuggingFaceDataset, modelId)));
    request.setRawHeader("User-Agent", "AlphaDiamond");
    beginWork();
    QNetworkReply* reply = network_->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, modelId] {
        if (reply->error() == QNetworkReply::NoError) {
            addArtifact(QStringLiteral("Hugging Face"),
                        QJsonDocument::fromJson(reply->readAll()).object(),
                        QStringLiteral("https://huggingface.co/buckets/%1/tree/models/%2")
                            .arg(kHuggingFaceDataset, modelId));
            rebuildRows();
        } else
            setStatus(QStringLiteral("Hugging Face metadata failed: %1").arg(reply->errorString()));
        reply->deleteLater();
        endWork();
    });
}

void ModelCatalog::addArtifact(const QString& source, const QJsonObject& metadata,
                               const QString& webUrl) {
    const QString family = metadata.value(QStringLiteral("model_family"))
                               .toString(metadata.value(QStringLiteral("family")).toString());
    const QString version = metadata.value(QStringLiteral("model_version"))
                                .toString(metadata.value(QStringLiteral("version")).toString());
    const QString id = family + QLatin1Char('/') + version;
    const QString modelDigest = metadata.value(QStringLiteral("model_sha256")).toString();
    const QString digest = metadata.value(QStringLiteral("runtime_sha256")).toString();
    if ((family != QStringLiteral("soo") && family != QStringLiteral("min")) ||
        !kSemVer.match(version).hasMatch() || !kDigest.match(modelDigest).hasMatch() ||
        !kDigest.match(digest).hasMatch())
        return;
    if (artifacts_.contains(id) && (artifacts_.value(id).model_digest != modelDigest ||
                                    artifacts_.value(id).runtime_digest != digest)) {
        failDigestMismatch(id);
        return;
    }
    const QJsonObject provenance = metadata.value(QStringLiteral("source")).toObject();
    Artifact artifact = artifacts_.value(id);
    artifact.id = id;
    artifact.version = version;
    artifact.model_digest = modelDigest;
    artifact.runtime_digest = digest;
    artifact.training_step = metaInt(metadata, provenance, "training_step");
    artifact.training_simulations = metaInt(metadata, provenance, "training_simulations");
    if (source == QStringLiteral("GitHub")) {
        artifact.github = true;
        artifact.github_url = webUrl;
    } else {
        artifact.hugging_face = true;
        artifact.hugging_face_url = webUrl;
    }
    artifacts_.insert(id, artifact);
}
void ModelCatalog::failDigestMismatch(const QString& modelId) {
    catalog_error_ = QStringLiteral("Source digest mismatch for %1. Downloads are disabled.")
                         .arg(modelLabel(modelId));
    setStatus(catalog_error_);
}
QString ModelCatalog::destinationFor(const QString& modelId) const {
    return safeRelativePath(modelId) ? QDir(local_root_).filePath(modelId) : QString();
}
void ModelCatalog::selectModel(const QString& modelId) {
    const QString path = local_paths_.value(modelId);
    if (path.isEmpty()) {
        setStatus(QStringLiteral("Download %1 before selecting it.").arg(modelLabel(modelId)));
        return;
    }
    selected_id_ = modelId;
    selected_path_ = path;
    QSettings settings;
    settings.setValue(QStringLiteral("models/selectedId"), selected_id_);
    settings.setValue(QStringLiteral("models/selectedPath"), selected_path_);
    settings.setValue(QStringLiteral("models/selectedByFamily/") + modelId.section('/', 0, 0),
                      selected_id_);
    setStatus(QStringLiteral("%1 selected for next game.").arg(modelLabel(modelId)));
    rebuildRows();
}
bool ModelCatalog::activateSelected(const QString& family) {
    if (!family.isEmpty()) {
        const QString prefix = family + QLatin1Char('/');
        QSettings settings;
        QString id = selected_id_.startsWith(prefix) ? selected_id_ :
            settings.value(QStringLiteral("models/selectedByFamily/") + family).toString();
        if (!id.startsWith(prefix) || !local_paths_.contains(id)) {
            id.clear();
            const QStringList bases = {
                QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("models")),
                QDir::current().filePath(QStringLiteral("models")), local_root_};
            // Training step count is not a promotion decision. Honor the indexed default.
            bool invalidIndex = false;
            for (const auto& base : bases) {
                if (!QFile::exists(QDir(base).filePath(QStringLiteral("index.json")))) continue;
                QString candidate;
#ifdef DIAMOND_QT_HAS_SOO
                try {
                    const auto index = diamond_model::load_model_index(base.toStdString());
                    if (const auto* entry = index.default_for(family.toStdString()))
                        candidate = QString::fromStdString(entry->family + "/" + entry->version);
                } catch (const std::exception& error) {
                    invalidIndex = true;
                    setStatus(QStringLiteral("Invalid model index: %1").arg(error.what()));
                    break;
                }
#else
                const auto index = QJsonDocument::fromJson(
                    readAll(QDir(base).filePath(QStringLiteral("index.json")))).object();
                if (index.value(QStringLiteral("index_version")).toInt() != 1 ||
                    !index.value(QStringLiteral("models")).isArray() ||
                    !index.value(QStringLiteral("defaults")).isObject()) {
                    invalidIndex = true;
                    break;
                }
                candidate = index.value(QStringLiteral("defaults")).toObject()
                    .value(family).toString();
#endif
                if (!candidate.isEmpty() && !local_paths_.contains(candidate)) {
                    invalidIndex = true;
                    setStatus(QStringLiteral("The indexed model is not installed or valid."));
                    break;
                }
                if (candidate.startsWith(prefix) && local_paths_.contains(candidate)) {
                    id = candidate;
                    break;
                }
            }
            if (id.isEmpty() && !invalidIndex) {
                QSet<QString> available;
                for (const auto& model : local_models_)
                    if (model.id.startsWith(prefix)) available.insert(model.id);
                if (available.size() == 1) id = *available.cbegin();
            }
        }
        selected_id_ = id;
        selected_path_ = local_paths_.value(id);
    }
    if (!QFileInfo::exists(selected_path_)) {
        selected_id_.clear();
        selected_path_.clear();
    }
    const bool changed = active_id_ != selected_id_ || active_path_ != selected_path_;
    active_id_ = selected_id_;
    active_path_ = selected_path_;
    if (changed) {
        setStatus(active_id_.isEmpty() ? QStringLiteral("No model is active for this game. Select one in Models.") :
                  QStringLiteral("%1 is active.").arg(modelLabel(active_id_)));
        rebuildRows();
    }
    return changed;
}
void ModelCatalog::downloadModel(const QString& modelId) {
    if (busy())
        return;
    if (!catalog_error_.isEmpty()) {
        setStatus(catalog_error_);
        return;
    }
    if (local_paths_.contains(modelId)) {
        selectModel(modelId);
        return;
    }
    const Artifact artifact = artifacts_.value(modelId);
    if (artifact.id.isEmpty()) {
        setStatus(QStringLiteral("Model is not available from a trusted source."));
        return;
    }
    startDownload(modelId, artifact.hugging_face || !artifact.github);
}
void ModelCatalog::startDownload(const QString& modelId, bool huggingFace) {
    const QStringList files =
        huggingFace ? hugging_face_files_.value(modelId) : github_files_.value(modelId);
    if (files.isEmpty()) {
        if (huggingFace && artifacts_.value(modelId).github) {
            startDownload(modelId, false);
            return;
        }
        setStatus(QStringLiteral("Download manifest is not ready; refresh the catalog."));
        return;
    }
    const QString prefix = QStringLiteral("models/") + modelId + QLatin1Char('/');
    // Validate the entire manifest before issuing requests or creating staging files.
    for (const QString& path : files) {
        if (!path.startsWith(prefix) || !safeRelativePath(path.mid(prefix.size()))) {
            setStatus(QStringLiteral("Unsafe source path rejected."));
            return;
        }
    }
    download_id_ = modelId;
    download_from_hugging_face_ = huggingFace;
    download_destination_ = destinationFor(modelId);
    if (download_destination_.isEmpty()) {
        setStatus(QStringLiteral("Unsafe model path rejected."));
        return;
    }
    download_staging_ = download_destination_ + QStringLiteral(".partial-") +
                        QUuid::createUuid().toString(QUuid::WithoutBraces);
    if (!QDir().mkpath(download_staging_)) {
        setStatus(QStringLiteral("Could not create download staging directory."));
        return;
    }
    download_pending_ = files.size();
    download_error_.clear();
    beginWork();
    setStatus(QStringLiteral("Downloading %1 from %2…")
                  .arg(modelLabel(modelId),
                       huggingFace ? QStringLiteral("Hugging Face") : QStringLiteral("GitHub")));
    for (const QString& path : files) {
        const QUrl url(
            huggingFace
                ? QStringLiteral("https://huggingface.co/buckets/%1/resolve/%2?download=true")
                      .arg(kHuggingFaceDataset, path)
                : QStringLiteral("https://raw.githubusercontent.com/%1/main/%2")
                      .arg(kRepository, path));
        requestDownloadFile(path.mid(prefix.size()), url);
    }
}
void ModelCatalog::requestDownloadFile(const QString& relativePath, const QUrl& url) {
    QNetworkRequest request(url);
    request.setRawHeader("User-Agent", "AlphaDiamond");
    QNetworkReply* reply = network_->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, relativePath] {
        if (reply->error() == QNetworkReply::NoError) {
            const QByteArray data = reply->readAll();
            const QString target = QDir(download_staging_).filePath(relativePath);
            QSaveFile file(target);
            // Qt checks HTTP framing and decompresses responses. Content-Length may
            // be absent or describe compressed bytes; artifact SHA256 validates content.
            if (!QDir().mkpath(QFileInfo(target).absolutePath()) ||
                     !file.open(QIODevice::WriteOnly) || file.write(data) != data.size() ||
                     !file.commit())
                download_error_ = QStringLiteral("Could not save %1.").arg(relativePath);
        } else if (download_error_.isEmpty())
            download_error_ = reply->errorString();
        reply->deleteLater();
        if (--download_pending_ == 0) {
            completeDownload(download_error_.isEmpty(), download_error_);
            endWork();
        }
    });
}
bool ModelCatalog::validateDownloadedModel(const QString& path, QString* error) const {
    LocalModel model;
    if (!readLocalModel(path, &model) || model.id != download_id_) {
        *error = QStringLiteral("Downloaded model metadata validation failed.");
        return false;
    }
    const auto expected = artifacts_.value(download_id_);
    if (model.runtime_digest != expected.runtime_digest || model.model_digest != expected.model_digest) {
        *error = QStringLiteral("Downloaded model digest mismatch.");
        return false;
    }
#ifdef DIAMOND_QT_HAS_SOO
    try {
        diamond_model::validate_deployment_artifact(std::filesystem::path(path.toStdString()));
    } catch (const std::exception& exception) {
        *error =
            QStringLiteral("Downloaded model digest validation failed: %1").arg(exception.what());
        return false;
    }
#endif
    return true;
}
void ModelCatalog::completeDownload(bool success, const QString& error) {
    QString validationError;
    if (success && !validateDownloadedModel(download_staging_, &validationError)) {
        success = false;
        download_error_ = validationError;
    }
    if (success) {
        QDir().mkpath(QFileInfo(download_destination_).absolutePath());
        if (!QDir().rename(download_staging_, download_destination_)) {
            success = false;
            download_error_ = QStringLiteral("Could not atomically install downloaded model.");
        }
    }
    if (!success) {
        QDir(download_staging_).removeRecursively();
        if (download_from_hugging_face_ && artifacts_.value(download_id_).github) {
            setStatus(QStringLiteral("Hugging Face download failed; trying GitHub…"));
            startDownload(download_id_, false);
            return;
        }
        setStatus(error.isEmpty() ? download_error_ : error);
        return;
    }
    scanLocal();
    rebuildRows();
    setStatus(QStringLiteral("%1 downloaded. Select it for next game when ready.")
                  .arg(modelLabel(download_id_)));
}
