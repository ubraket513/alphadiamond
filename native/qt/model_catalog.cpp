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
#include <QProcess>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QSettings>
#include <QStandardPaths>
#include <QUrl>

#include <filesystem>
#include <stdexcept>

#ifdef DIAMOND_QT_HAS_SOO
#include "diamond_model/deployment_artifact.hpp"
#endif

namespace {

constexpr auto kRepository = "ubraket513/alphadiamond";
constexpr auto kBucket = "hf://buckets/ubraket513/AlphaDiamond/models";

QString model_label(const QString& id) {
    const QStringList parts = id.split('/');
    if (parts.size() != 2) return id;
    QString family = parts.at(0);
    if (!family.isEmpty()) family[0] = family[0].toUpper();
    return QStringLiteral("%1 %2").arg(family, parts.at(1));
}

QByteArray read_all(const QString& path) {
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray{};
}

}  // namespace

ModelCatalog::ModelCatalog(QObject* parent)
    : QObject(parent),
      network_(new QNetworkAccessManager(this)),
      local_root_(QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
                      .filePath(QStringLiteral("models"))) {
    QDir().mkpath(local_root_);
    QSettings settings;
    selected_id_ = settings.value(QStringLiteral("models/selectedId")).toString();
    selected_path_ = settings.value(QStringLiteral("models/selectedPath")).toString();
    scanLocal();
    active_id_ = selected_id_;
    active_path_ = selected_path_;
    rebuildRows();
}

QString ModelCatalog::selectedModelLabel() const {
    return selected_id_.isEmpty() ? QStringLiteral("None") : model_label(selected_id_);
}

QString ModelCatalog::activeModelLabel() const {
    return active_id_.isEmpty() ? QStringLiteral("None") : model_label(active_id_);
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
    if (busy_count_ > 0) --busy_count_;
    Q_EMIT changed();
}

bool ModelCatalog::readLocalModel(const QString& path, bool packaged, LocalModel* model) const {
    const QJsonDocument document = QJsonDocument::fromJson(
        read_all(QDir(path).filePath(QStringLiteral("metadata.json"))));
    if (!document.isObject()) return false;
    const QJsonObject root = document.object();
    const QString family = root.value(QStringLiteral("model_family")).toString();
    const QString version = root.value(QStringLiteral("model_version")).toString();
    if (family.isEmpty() || version.isEmpty()) return false;
#ifdef DIAMOND_QT_HAS_SOO
    try {
        diamond_model::validate_deployment_artifact(
            std::filesystem::path(path.toStdString()), family.toStdString());
    } catch (const std::exception&) {
        return false;
    }
#endif
    model->id = family + QLatin1Char('/') + version;
    model->path = QDir::cleanPath(path);
    model->version = version;
    model->training_step = root.value(QStringLiteral("source")).toObject()
                               .value(QStringLiteral("training_step")).toInt();
    model->packaged = packaged;
    return true;
}

void ModelCatalog::scanLocal() {
    local_models_.clear();
    local_paths_.clear();
    QSet<QString> visited;

    const QStringList bases = {
        QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("models")),
        QDir(QDir::currentPath()).filePath(QStringLiteral("models")),
        local_root_
    };
    for (int base_index = 0; base_index < bases.size(); ++base_index) {
        const QString base = bases.at(base_index);
        QDirIterator iterator(base, QStringList{QStringLiteral("metadata.json")},
                              QDir::Files, QDirIterator::Subdirectories);
        while (iterator.hasNext()) {
            const QString metadata = iterator.next();
            const QString root = QFileInfo(metadata).absolutePath();
            const QString canonical = QFileInfo(root).canonicalFilePath();
            const QString unique = canonical.isEmpty() ? QDir::cleanPath(root) : canonical;
            if (visited.contains(unique)) continue;
            visited.insert(unique);
            LocalModel model;
            if (!readLocalModel(root, base_index < 2, &model)) continue;
            local_models_.push_back(model);
            if (!local_paths_.contains(model.id) || !model.packaged)
                local_paths_.insert(model.id, model.path);
        }
    }

    if (!local_paths_.contains(selected_id_) || !QFileInfo::exists(selected_path_)) {
        if (!local_models_.isEmpty()) {
            const LocalModel& fallback = local_models_.constFirst();
            selected_id_ = fallback.id;
            selected_path_ = fallback.path;
            QSettings settings;
            settings.setValue(QStringLiteral("models/selectedId"), selected_id_);
            settings.setValue(QStringLiteral("models/selectedPath"), selected_path_);
        } else {
            selected_id_.clear();
            selected_path_.clear();
        }
    }
}

QVariantMap ModelCatalog::remoteRow(const QString& source, const QString& modelId,
                                    const QString& version, int trainingStep,
                                    bool compatible, const QString& webUrl,
                                    const QString& note) const {
    const bool installed = local_paths_.contains(modelId);
    return QVariantMap{
        {QStringLiteral("id"), modelId},
        {QStringLiteral("name"), model_label(modelId)},
        {QStringLiteral("source"), source},
        {QStringLiteral("version"), version},
        {QStringLiteral("trainingStep"), trainingStep},
        {QStringLiteral("installed"), installed},
        {QStringLiteral("compatible"), compatible},
        {QStringLiteral("selected"), modelId == selected_id_},
        {QStringLiteral("active"), modelId == active_id_},
        {QStringLiteral("webUrl"), webUrl},
        {QStringLiteral("note"), note}
    };
}

void ModelCatalog::rebuildRows() {
    QVariantList rows;
    for (const LocalModel& model : local_models_) {
        rows.push_back(QVariantMap{
            {QStringLiteral("id"), model.id},
            {QStringLiteral("name"), model_label(model.id)},
            {QStringLiteral("source"), model.packaged ? QStringLiteral("Bundled")
                                                       : QStringLiteral("Local")},
            {QStringLiteral("version"), model.version},
            {QStringLiteral("trainingStep"), model.training_step},
            {QStringLiteral("installed"), true},
            {QStringLiteral("compatible"), true},
            {QStringLiteral("selected"), model.id == selected_id_ && model.path == selected_path_},
            {QStringLiteral("active"), model.id == active_id_ && model.path == active_path_},
            {QStringLiteral("webUrl"), QString()},
            {QStringLiteral("note"), model.packaged
                ? QStringLiteral("Shipped with this build")
                : QStringLiteral("Downloaded model")}
        });
    }
    for (const QVariant& value : remote_models_) {
        QVariantMap row = value.toMap();
        const QString id = row.value(QStringLiteral("id")).toString();
        row.insert(QStringLiteral("installed"), local_paths_.contains(id));
        row.insert(QStringLiteral("selected"), id == selected_id_);
        row.insert(QStringLiteral("active"), id == active_id_);
        rows.push_back(row);
    }
    models_ = rows;
    Q_EMIT changed();
}

void ModelCatalog::refresh() {
    if (busy()) return;
    remote_models_.clear();
    scanLocal();
    rebuildRows();
    setStatus(QStringLiteral("Refreshing GitHub and Hugging Face…"));
    fetchGitHub();
    fetchHuggingFace();
}

void ModelCatalog::fetchGitHub() {
    const auto requestJson = [this](const QUrl& url, auto parser) {
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
    requestJson(QUrl(QStringLiteral(
        "https://raw.githubusercontent.com/%1/main/models/index.json").arg(kRepository)),
        &ModelCatalog::parseGitHubIndex);
    requestJson(QUrl(QStringLiteral(
        "https://api.github.com/repos/%1/git/trees/main?recursive=1").arg(kRepository)),
        &ModelCatalog::parseGitHubTree);
    requestJson(QUrl(QStringLiteral(
        "https://api.github.com/repos/%1/releases?per_page=30").arg(kRepository)),
        &ModelCatalog::parseGitHubReleases);
}

void ModelCatalog::parseGitHubIndex(const QByteArray& payload) {
    const QJsonArray models = QJsonDocument::fromJson(payload).object()
                                  .value(QStringLiteral("models")).toArray();
    for (const QJsonValue& value : models) {
        const QJsonObject model = value.toObject();
        const QString id = model.value(QStringLiteral("path")).toString();
        const QString version = model.value(QStringLiteral("version")).toString();
        const int step = model.value(QStringLiteral("source")).toObject()
                             .value(QStringLiteral("training_step")).toInt();
        remote_models_.push_back(remoteRow(
            QStringLiteral("GitHub"), id, version, step, true,
            QStringLiteral("https://github.com/%1/tree/main/models/%2").arg(kRepository, id),
            QStringLiteral("Native repository artifact")));
    }
    rebuildRows();
}

void ModelCatalog::parseGitHubTree(const QByteArray& payload) {
    github_files_.clear();
    const QJsonArray tree = QJsonDocument::fromJson(payload).object()
                                .value(QStringLiteral("tree")).toArray();
    for (const QJsonValue& value : tree) {
        const QJsonObject item = value.toObject();
        if (item.value(QStringLiteral("type")).toString() != QStringLiteral("blob")) continue;
        const QString path = item.value(QStringLiteral("path")).toString();
        if (!path.startsWith(QStringLiteral("models/"))) continue;
        const QStringList parts = path.split('/');
        if (parts.size() < 4 || parts.at(3).isEmpty()) continue;
        const QString id = parts.at(1) + QLatin1Char('/') + parts.at(2);
        github_files_[id].push_back(path);
    }
}

void ModelCatalog::parseGitHubReleases(const QByteArray& payload) {
    const QJsonArray releases = QJsonDocument::fromJson(payload).array();
    for (const QJsonValue& value : releases) {
        const QJsonObject release = value.toObject();
        const QString tag = release.value(QStringLiteral("tag_name")).toString();
        if (tag.isEmpty()) continue;
        int step = 0;
        const QRegularExpression match(QStringLiteral("step0*([0-9]+)"),
                                       QRegularExpression::CaseInsensitiveOption);
        const auto result = match.match(tag);
        if (result.hasMatch()) step = result.captured(1).toInt();
        remote_models_.push_back(remoteRow(
            QStringLiteral("GitHub release"), QStringLiteral("checkpoint/") + tag,
            tag, step, false, release.value(QStringLiteral("html_url")).toString(),
            QStringLiteral("Training checkpoint; native export required")));
    }
    rebuildRows();
}

QString ModelCatalog::hfExecutable() const {
    const QString configured = qEnvironmentVariable("DIAMOND_HF_CLI");
    if (!configured.isEmpty() && QFileInfo(configured).isExecutable()) return configured;
    const QString found = QStandardPaths::findExecutable(QStringLiteral("hf"));
    if (!found.isEmpty()) return found;
#ifdef Q_OS_WIN
    const QString fallback = QDir::home().filePath(QStringLiteral(".local/bin/hf.exe"));
#else
    const QString fallback = QDir::home().filePath(QStringLiteral(".local/bin/hf"));
#endif
    return QFileInfo(fallback).isExecutable() ? fallback : QString();
}

void ModelCatalog::fetchHuggingFace() {
    const QString executable = hfExecutable();
    if (executable.isEmpty()) {
        setStatus(QStringLiteral("Hugging Face CLI not found. Install it and run hf auth login."));
        return;
    }
    auto* process = new QProcess(this);
    beginWork();
    connect(process, &QProcess::finished, this,
            [this, process](int exitCode, QProcess::ExitStatus exitStatus) {
        if (exitStatus == QProcess::NormalExit && exitCode == 0) {
            parseHuggingFace(process->readAllStandardOutput());
            setStatus(QStringLiteral("Model catalog is up to date."));
        } else {
            const QString error = QString::fromUtf8(process->readAllStandardError()).trimmed();
            setStatus(error.isEmpty()
                ? QStringLiteral("Hugging Face refresh failed. Run hf auth login.")
                : error);
        }
        process->deleteLater();
        endWork();
    });
    process->start(executable, QStringList{
        QStringLiteral("buckets"), QStringLiteral("list"), QString::fromLatin1(kBucket),
        QStringLiteral("-R"), QStringLiteral("--format"), QStringLiteral("json")
    });
}

void ModelCatalog::parseHuggingFace(const QByteArray& payload) {
    QSet<QString> ids;
    const QJsonArray entries = QJsonDocument::fromJson(payload).array();
    for (const QJsonValue& value : entries) {
        const QString path = value.toObject().value(QStringLiteral("path")).toString();
        const QStringList parts = path.split('/');
        if (parts.size() == 4 && parts.at(0) == QStringLiteral("models") &&
            parts.at(3) == QStringLiteral("metadata.json")) {
            ids.insert(parts.at(1) + QLatin1Char('/') + parts.at(2));
        }
    }
    for (const QString& id : ids) {
        const QString version = id.section('/', 1, 1);
        remote_models_.push_back(remoteRow(
            QStringLiteral("Hugging Face"), id, version, 0, true,
            QStringLiteral("https://huggingface.co/buckets/ubraket513/AlphaDiamond"),
            QStringLiteral("Storage Bucket native artifact")));
    }
    rebuildRows();
}

QString ModelCatalog::destinationFor(const QString& modelId) const {
    return QDir(local_root_).filePath(modelId);
}

void ModelCatalog::selectModel(const QString& modelId) {
    const QString path = local_paths_.value(modelId);
    if (path.isEmpty()) {
        setStatus(QStringLiteral("Download %1 before selecting it.").arg(model_label(modelId)));
        return;
    }
    selected_id_ = modelId;
    selected_path_ = path;
    QSettings settings;
    settings.setValue(QStringLiteral("models/selectedId"), selected_id_);
    settings.setValue(QStringLiteral("models/selectedPath"), selected_path_);
    setStatus(QStringLiteral("%1 selected for the next game.").arg(model_label(modelId)));
    rebuildRows();
}

bool ModelCatalog::activateSelected() {
    if (selected_id_.isEmpty() || !QFileInfo::exists(selected_path_)) return false;
    const bool changed = active_id_ != selected_id_ || active_path_ != selected_path_;
    active_id_ = selected_id_;
    active_path_ = selected_path_;
    if (changed) {
        setStatus(QStringLiteral("%1 is active.").arg(model_label(active_id_)));
        rebuildRows();
    }
    return changed;
}

void ModelCatalog::downloadModel(const QString& source, const QString& modelId) {
    if (busy()) return;
    if (local_paths_.contains(modelId)) {
        selectModel(modelId);
        return;
    }
    if (source == QStringLiteral("Hugging Face"))
        startHfDownload(modelId);
    else if (source == QStringLiteral("GitHub"))
        startGitHubDownload(modelId);
    else
        setStatus(QStringLiteral("This checkpoint must be exported to the native format first."));
}

void ModelCatalog::startHfDownload(const QString& modelId) {
    const QString executable = hfExecutable();
    if (executable.isEmpty()) {
        setStatus(QStringLiteral("Hugging Face CLI not found. Install it and run hf auth login."));
        return;
    }
    download_source_ = QStringLiteral("Hugging Face");
    download_id_ = modelId;
    download_destination_ = destinationFor(modelId);
    download_staging_ = download_destination_ + QStringLiteral(".partial");
    QDir(download_staging_).removeRecursively();
    QDir().mkpath(QFileInfo(download_staging_).absolutePath());

    auto* process = new QProcess(this);
    beginWork();
    setStatus(QStringLiteral("Downloading %1 from Hugging Face…").arg(model_label(modelId)));
    connect(process, &QProcess::finished, this,
            [this, process](int exitCode, QProcess::ExitStatus exitStatus) {
        if (exitStatus == QProcess::NormalExit && exitCode == 0)
            completeDownload(true);
        else
            completeDownload(false,
                QString::fromUtf8(process->readAllStandardError()).trimmed());
        process->deleteLater();
        endWork();
    });
    process->start(executable, QStringList{
        QStringLiteral("sync"),
        QString::fromLatin1(kBucket) + QLatin1Char('/') + modelId,
        download_staging_
    });
}

void ModelCatalog::startGitHubDownload(const QString& modelId) {
    const QStringList files = github_files_.value(modelId);
    if (files.isEmpty()) {
        setStatus(QStringLiteral("GitHub file list is not ready; refresh the catalog."));
        return;
    }
    download_source_ = QStringLiteral("GitHub");
    download_id_ = modelId;
    download_destination_ = destinationFor(modelId);
    download_staging_ = download_destination_ + QStringLiteral(".partial");
    QDir(download_staging_).removeRecursively();
    QDir().mkpath(download_staging_);
    download_pending_ = files.size();
    download_error_.clear();
    beginWork();
    setStatus(QStringLiteral("Downloading %1 from GitHub…").arg(model_label(modelId)));

    const QString prefix = QStringLiteral("models/") + modelId + QLatin1Char('/');
    for (const QString& path : files) {
        QNetworkRequest request(QUrl(QStringLiteral(
            "https://raw.githubusercontent.com/%1/main/%2").arg(kRepository, path)));
        request.setRawHeader("User-Agent", "AlphaDiamond");
        QNetworkReply* reply = network_->get(request);
        connect(reply, &QNetworkReply::finished, this, [this, reply, path, prefix] {
            if (reply->error() == QNetworkReply::NoError) {
                const QString relative = path.mid(prefix.size());
                const QString destination = QDir(download_staging_).filePath(relative);
                QDir().mkpath(QFileInfo(destination).absolutePath());
                QSaveFile file(destination);
                if (!file.open(QIODevice::WriteOnly) ||
                    file.write(reply->readAll()) < 0 || !file.commit())
                    download_error_ = QStringLiteral("Could not save %1").arg(relative);
            } else if (download_error_.isEmpty()) {
                download_error_ = reply->errorString();
            }
            reply->deleteLater();
            if (--download_pending_ == 0) {
                completeDownload(download_error_.isEmpty(), download_error_);
                endWork();
            }
        });
    }
}

void ModelCatalog::completeDownload(bool success, const QString& error) {
    if (success) {
        LocalModel downloaded;
        if (!readLocalModel(download_staging_, false, &downloaded)) {
            success = false;
            download_error_ = QStringLiteral("Downloaded model failed native artifact validation.");
        }
    }
    if (success) {
        QDir destination(download_destination_);
        if (destination.exists()) destination.removeRecursively();
        QDir().mkpath(QFileInfo(download_destination_).absolutePath());
        if (!QDir().rename(download_staging_, download_destination_)) {
            success = false;
            download_error_ = QStringLiteral("Could not install the downloaded model.");
        }
    }
    if (!success) {
        QDir(download_staging_).removeRecursively();
        setStatus(error.isEmpty() ? download_error_ : error);
        return;
    }
    scanLocal();
    rebuildRows();
    selectModel(download_id_);
}
