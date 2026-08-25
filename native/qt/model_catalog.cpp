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

#include <filesystem>
#include <stdexcept>

#ifdef DIAMOND_QT_HAS_SOO
#include "diamond_model/deployment_artifact.hpp"
#endif

namespace {
constexpr auto kRepository = "ubraket513/alphadiamond";
constexpr auto kHuggingFaceDataset = "ubraket513/AlphaDiamond";
const QRegularExpression kSemVer(QStringLiteral(
    "^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(?:-(?:0|[1-9][0-9]*|["
    "0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)(?:\\.(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-"
    "Za-z-][0-9A-Za-z-]*))*)?(?:\\+[0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*)?$"));
QString modelLabel(const QString &id) {
  const QStringList parts = id.split('/');
  if (parts.size() != 2)
    return id;
  QString family = parts.at(0);
  if (!family.isEmpty())
    family[0] = family.at(0).toUpper();
  return QStringLiteral("%1 %2").arg(family, parts.at(1));
}
QByteArray readAll(const QString &path) {
  QFile file(path);
  return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray{};
}
bool safeRelativePath(const QString &path) {
  if (path.isEmpty() || QDir::isAbsolutePath(path) ||
      path.contains(QLatin1Char('\\')))
    return false;
  const QString clean = QDir::cleanPath(path);
  return clean != QStringLiteral("..") &&
         !clean.startsWith(QStringLiteral("../")) &&
         !clean.contains(QStringLiteral("/../"));
}
int metaInt(const QJsonObject &root, const QJsonObject &source,
            const char *key) {
  const QJsonValue value = source.value(QLatin1String(key));
  return value.isDouble() ? value.toInt()
                          : root.value(QLatin1String(key)).toInt();
}
QString metaText(const QJsonObject &root, const QJsonObject &source,
                 const char *key) {
  const QString value = source.value(QLatin1String(key)).toVariant().toString();
  return value.isEmpty() ? root.value(QLatin1String(key)).toVariant().toString()
                         : value;
}
} // namespace

ModelCatalog::ModelCatalog(QObject *parent)
    : QObject(parent), network_(new QNetworkAccessManager(this)),
      local_root_(QDir(QStandardPaths::writableLocation(
                           QStandardPaths::AppLocalDataLocation))
                      .filePath(QStringLiteral("models"))) {
  QDir().mkpath(local_root_);
  QSettings settings;
  selected_id_ = settings.value(QStringLiteral("models/selectedId")).toString();
  selected_path_ =
      settings.value(QStringLiteral("models/selectedPath")).toString();
  scanLocal();
  active_id_ = selected_id_;
  active_path_ = selected_path_;
  rebuildRows();
}
QString ModelCatalog::selectedModelLabel() const {
  return selected_id_.isEmpty() ? QStringLiteral("None")
                                : modelLabel(selected_id_);
}
QString ModelCatalog::activeModelLabel() const {
  return active_id_.isEmpty() ? QStringLiteral("None") : modelLabel(active_id_);
}
void ModelCatalog::setStatus(const QString &status) {
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
  Q_EMIT changed();
}

bool ModelCatalog::readLocalModel(const QString &path,
                                  LocalModel *model) const {
  const QJsonDocument document = QJsonDocument::fromJson(
      readAll(QDir(path).filePath(QStringLiteral("metadata.json"))));
  if (!document.isObject())
    return false;
  const QJsonObject root = document.object(),
                    source = root.value(QStringLiteral("source")).toObject();
  const QString family = root.value(QStringLiteral("model_family")).toString();
  const QString version =
      root.value(QStringLiteral("model_version")).toString();
  if (family.isEmpty() || !kSemVer.match(version).hasMatch())
    return false;
#ifdef DIAMOND_QT_HAS_SOO
  try {
    diamond_model::validate_deployment_artifact(
        std::filesystem::path(path.toStdString()), family.toStdString());
  } catch (const std::exception &) {
    return false;
  }
#endif
  model->id = family + QLatin1Char('/') + version;
  model->path = QDir::cleanPath(path);
  model->version = version;
  model->runtime_digest =
      root.value(QStringLiteral("runtime_sha256")).toString();
  model->training_step = metaInt(root, source, "training_step");
  model->training_simulations = metaInt(root, source, "training_simulations");
  model->latest_elo = metaText(root, source, "latest_elo");
  return true;
}
void ModelCatalog::scanLocal() {
  local_models_.clear();
  local_paths_.clear();
  QSet<QString> visited;
  const QStringList bases = {
      QDir(QCoreApplication::applicationDirPath())
          .filePath(QStringLiteral("models")),
      QDir(QDir::currentPath()).filePath(QStringLiteral("models")),
      local_root_};
  for (const QString &base : bases) {
    QDirIterator it(base, QStringList{QStringLiteral("metadata.json")},
                    QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
      const QString root = QFileInfo(it.next()).absolutePath();
      const QString canonical = QFileInfo(root).canonicalFilePath();
      const QString unique =
          canonical.isEmpty() ? QDir::cleanPath(root) : canonical;
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
  if (!local_paths_.contains(selected_id_) ||
      !QFileInfo::exists(selected_path_)) {
    selected_id_.clear();
    selected_path_.clear();
  }
}
void ModelCatalog::rebuildRows() {
  QVariantList rows;
  QSet<QString> known;
  for (auto it = artifacts_.cbegin(); it != artifacts_.cend(); ++it) {
    const Artifact &a = it.value();
    const bool installed = local_paths_.contains(a.id);
    rows.push_back(QVariantMap{{"id", a.id},
                               {"name", modelLabel(a.id)},
                               {"version", a.version},
                               {"trainingStep", a.training_step},
                               {"trainingSimulations", a.training_simulations},
                               {"latestElo", a.latest_elo},
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
  for (const LocalModel &a : local_models_)
    if (!known.contains(a.id))
      rows.push_back(
          QVariantMap{{"id", a.id},
                      {"name", modelLabel(a.id)},
                      {"version", a.version},
                      {"trainingStep", a.training_step},
                      {"trainingSimulations", a.training_simulations},
                      {"latestElo", a.latest_elo},
                      {"installed", true},
                      {"compatible", true},
                      {"selected", a.id == selected_id_},
                      {"active", a.id == active_id_},
                      {"github", false},
                      {"huggingFace", false},
                      {"githubUrl", QString()},
                      {"huggingFaceUrl", QString()}});
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
}
void ModelCatalog::fetchGitHub() {
  const auto request = [this](const QUrl &url, auto parser) {
    QNetworkRequest request(url);
    request.setRawHeader("Accept", "application/vnd.github+json");
    request.setRawHeader("User-Agent", "AlphaDiamond");
    beginWork();
    QNetworkReply *reply = network_->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, parser] {
      if (reply->error() == QNetworkReply::NoError)
        (this->*parser)(reply->readAll());
      else
        setStatus(QStringLiteral("GitHub refresh failed: %1")
                      .arg(reply->errorString()));
      reply->deleteLater();
      endWork();
    });
  };
  request(
      QUrl(QStringLiteral(
               "https://raw.githubusercontent.com/%1/main/models/index.json")
               .arg(kRepository)),
      &ModelCatalog::parseGitHubIndex);
  request(QUrl(QStringLiteral(
                   "https://api.github.com/repos/%1/git/trees/main?recursive=1")
                   .arg(kRepository)),
          &ModelCatalog::parseGitHubTree);
}
void ModelCatalog::fetchHuggingFace() {
  QNetworkRequest request(
      QUrl(QStringLiteral("https://huggingface.co/api/buckets/%1/"
                          "tree?recursive=true&expand=false")
               .arg(kHuggingFaceDataset)));
  request.setRawHeader("User-Agent", "AlphaDiamond");
  beginWork();
  QNetworkReply *reply = network_->get(request);
  connect(reply, &QNetworkReply::finished, this, [this, reply] {
    if (reply->error() == QNetworkReply::NoError)
      parseHuggingFaceTree(reply->readAll());
    else
      setStatus(QStringLiteral("Hugging Face refresh failed: %1")
                    .arg(reply->errorString()));
    reply->deleteLater();
    endWork();
  });
}
void ModelCatalog::parseGitHubIndex(const QByteArray &payload) {
  const QJsonArray models = QJsonDocument::fromJson(payload)
                                .object()
                                .value(QStringLiteral("models"))
                                .toArray();
  for (const QJsonValue &value : models) {
    const QJsonObject metadata = value.toObject();
    addArtifact(QStringLiteral("GitHub"), metadata,
                QStringLiteral("https://github.com/%1/tree/main/models/%2")
                    .arg(kRepository,
                         metadata.value(QStringLiteral("path")).toString()));
  }
  rebuildRows();
}
void ModelCatalog::parseGitHubTree(const QByteArray &payload) {
  const QJsonArray tree = QJsonDocument::fromJson(payload)
                              .object()
                              .value(QStringLiteral("tree"))
                              .toArray();
  for (const QJsonValue &value : tree) {
    const QJsonObject item = value.toObject();
    const QString path = item.value(QStringLiteral("path")).toString();
    const QStringList p = path.split('/');
    if (item.value(QStringLiteral("type")).toString() ==
            QStringLiteral("blob") &&
        p.size() >= 4 && p.at(0) == QStringLiteral("models") &&
        safeRelativePath(path))
      github_files_[p.at(1) + QLatin1Char('/') + p.at(2)].push_back(path);
  }
}
void ModelCatalog::parseHuggingFaceTree(const QByteArray &payload) {
  QSet<QString> ids;
  for (const QJsonValue &value : QJsonDocument::fromJson(payload).array()) {
    const QString path =
        value.toObject().value(QStringLiteral("path")).toString();
    const QStringList p = path.split('/');
    if (p.size() < 4 || p.at(0) != QStringLiteral("models") ||
        !safeRelativePath(path))
      continue;
    const QString id = p.at(1) + QLatin1Char('/') + p.at(2);
    hugging_face_files_[id].push_back(path);
    if (p.at(3) == QStringLiteral("metadata.json"))
      ids.insert(id);
  }
  for (const QString &id : ids)
    fetchHuggingFaceMetadata(id);
}
void ModelCatalog::fetchHuggingFaceMetadata(const QString &modelId) {
  QNetworkRequest request(
      QUrl(QStringLiteral("https://huggingface.co/buckets/%1/resolve/models/%2/"
                          "metadata.json?download=true")
               .arg(kHuggingFaceDataset, modelId)));
  request.setRawHeader("User-Agent", "AlphaDiamond");
  beginWork();
  QNetworkReply *reply = network_->get(request);
  connect(reply, &QNetworkReply::finished, this, [this, reply, modelId] {
    if (reply->error() == QNetworkReply::NoError) {
      addArtifact(
          QStringLiteral("Hugging Face"),
          QJsonDocument::fromJson(reply->readAll()).object(),
          QStringLiteral("https://huggingface.co/buckets/%1/tree/models/%2")
              .arg(kHuggingFaceDataset, modelId));
      rebuildRows();
    } else
      setStatus(QStringLiteral("Hugging Face metadata failed: %1")
                    .arg(reply->errorString()));
    reply->deleteLater();
    endWork();
  });
}

void ModelCatalog::addArtifact(const QString &source,
                               const QJsonObject &metadata,
                               const QString &webUrl) {
  const QString family =
      metadata.value(QStringLiteral("model_family"))
          .toString(metadata.value(QStringLiteral("family")).toString());
  const QString version =
      metadata.value(QStringLiteral("model_version"))
          .toString(metadata.value(QStringLiteral("version")).toString());
  const QString id = family + QLatin1Char('/') + version;
  const QString modelDigest =
      metadata.value(QStringLiteral("model_sha256")).toString();
  const QString digest =
      metadata.value(QStringLiteral("runtime_sha256")).toString();
  if (family.isEmpty() || !kSemVer.match(version).hasMatch() ||
      modelDigest.isEmpty() || digest.isEmpty())
    return;
  if (artifacts_.contains(id) &&
      (artifacts_.value(id).model_digest != modelDigest ||
       artifacts_.value(id).runtime_digest != digest)) {
    failDigestMismatch(id);
    return;
  }
  const QJsonObject provenance =
      metadata.value(QStringLiteral("source")).toObject();
  Artifact artifact = artifacts_.value(id);
  artifact.id = id;
  artifact.version = version;
  artifact.model_digest = modelDigest;
  artifact.runtime_digest = digest;
  artifact.training_step = metaInt(metadata, provenance, "training_step");
  artifact.training_simulations =
      metaInt(metadata, provenance, "training_simulations");
  artifact.latest_elo = metaText(metadata, provenance, "latest_elo");
  if (source == QStringLiteral("GitHub")) {
    artifact.github = true;
    artifact.github_url = webUrl;
  } else {
    artifact.hugging_face = true;
    artifact.hugging_face_url = webUrl;
  }
  artifacts_.insert(id, artifact);
}
void ModelCatalog::failDigestMismatch(const QString &modelId) {
  catalog_error_ =
      QStringLiteral("Source digest mismatch for %1. Downloads are disabled.")
          .arg(modelLabel(modelId));
  setStatus(catalog_error_);
}
QString ModelCatalog::destinationFor(const QString &modelId) const {
  return safeRelativePath(modelId) ? QDir(local_root_).filePath(modelId)
                                   : QString();
}
void ModelCatalog::selectModel(const QString &modelId) {
  const QString path = local_paths_.value(modelId);
  if (path.isEmpty()) {
    setStatus(QStringLiteral("Download %1 before selecting it.")
                  .arg(modelLabel(modelId)));
    return;
  }
  selected_id_ = modelId;
  selected_path_ = path;
  QSettings settings;
  settings.setValue(QStringLiteral("models/selectedId"), selected_id_);
  settings.setValue(QStringLiteral("models/selectedPath"), selected_path_);
  setStatus(
      QStringLiteral("%1 selected for next game.").arg(modelLabel(modelId)));
  rebuildRows();
}
bool ModelCatalog::activateSelected() {
  if (selected_id_.isEmpty() || !QFileInfo::exists(selected_path_))
    return false;
  const bool changed =
      active_id_ != selected_id_ || active_path_ != selected_path_;
  active_id_ = selected_id_;
  active_path_ = selected_path_;
  if (changed) {
    setStatus(QStringLiteral("%1 is active.").arg(modelLabel(active_id_)));
    rebuildRows();
  }
  return changed;
}
void ModelCatalog::downloadModel(const QString &modelId) {
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
void ModelCatalog::startDownload(const QString &modelId, bool huggingFace) {
  const QStringList files = huggingFace ? hugging_face_files_.value(modelId)
                                        : github_files_.value(modelId);
  if (files.isEmpty()) {
    if (huggingFace && artifacts_.value(modelId).github) {
      startDownload(modelId, false);
      return;
    }
    setStatus(
        QStringLiteral("Download manifest is not ready; refresh the catalog."));
    return;
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
                .arg(modelLabel(modelId), huggingFace
                                              ? QStringLiteral("Hugging Face")
                                              : QStringLiteral("GitHub")));
  const QString prefix = QStringLiteral("models/") + modelId + QLatin1Char('/');
  for (const QString &path : files) {
    if (!path.startsWith(prefix) ||
        !safeRelativePath(path.mid(prefix.size()))) {
      completeDownload(false, QStringLiteral("Unsafe source path rejected."));
      endWork();
      return;
    }
    const QUrl url(
        huggingFace
            ? QStringLiteral(
                  "https://huggingface.co/buckets/%1/resolve/%2?download=true")
                  .arg(kHuggingFaceDataset, path)
            : QStringLiteral("https://raw.githubusercontent.com/%1/main/%2")
                  .arg(kRepository, path));
    requestDownloadFile(path.mid(prefix.size()), url);
  }
}
void ModelCatalog::requestDownloadFile(const QString &relativePath,
                                       const QUrl &url) {
  QNetworkRequest request(url);
  request.setRawHeader("User-Agent", "AlphaDiamond");
  QNetworkReply *reply = network_->get(request);
  connect(reply, &QNetworkReply::finished, this, [this, reply, relativePath] {
    if (reply->error() == QNetworkReply::NoError) {
      const QByteArray data = reply->readAll();
      const qint64 expected =
          reply->header(QNetworkRequest::ContentLengthHeader).toLongLong();
      const QString target = QDir(download_staging_).filePath(relativePath);
      QSaveFile file(target);
      if (expected >= 0 && expected != data.size())
        download_error_ =
            QStringLiteral("Download size validation failed for %1.")
                .arg(relativePath);
      else if (!QDir().mkpath(QFileInfo(target).absolutePath()) ||
               !file.open(QIODevice::WriteOnly) ||
               file.write(data) != data.size() || !file.commit())
        download_error_ =
            QStringLiteral("Could not save %1.").arg(relativePath);
    } else if (download_error_.isEmpty())
      download_error_ = reply->errorString();
    reply->deleteLater();
    if (--download_pending_ == 0) {
      completeDownload(download_error_.isEmpty(), download_error_);
      endWork();
    }
  });
}
bool ModelCatalog::validateDownloadedModel(const QString &path,
                                           QString *error) const {
  LocalModel model;
  if (!readLocalModel(path, &model) || model.id != download_id_) {
    *error = QStringLiteral("Downloaded model metadata validation failed.");
    return false;
  }
  if (model.runtime_digest != artifacts_.value(download_id_).runtime_digest) {
    *error = QStringLiteral("Downloaded model digest mismatch.");
    return false;
  }
#ifdef DIAMOND_QT_HAS_SOO
  try {
    diamond_model::validate_deployment_artifact(
        std::filesystem::path(path.toStdString()));
  } catch (const std::exception &exception) {
    *error = QStringLiteral("Downloaded model digest validation failed: %1")
                 .arg(exception.what());
    return false;
  }
#endif
  return true;
}
void ModelCatalog::completeDownload(bool success, const QString &error) {
  QString validationError;
  if (success &&
      !validateDownloadedModel(download_staging_, &validationError)) {
    success = false;
    download_error_ = validationError;
  }
  if (success) {
    QDir().mkpath(QFileInfo(download_destination_).absolutePath());
    if (!QDir().rename(download_staging_, download_destination_)) {
      success = false;
      download_error_ =
          QStringLiteral("Could not atomically install downloaded model.");
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
