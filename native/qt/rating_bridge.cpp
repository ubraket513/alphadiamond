#include "rating_bridge.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSettings>
#include <QStandardPaths>
#include <QUuid>

#include <array>
#include <filesystem>
#include <optional>

#include "diamond_orchestration/rating.hpp"
#include "diamond_orchestration/rating_store.hpp"

namespace {
constexpr auto kProtocolId = "soo-elo-v1";

std::filesystem::path default_root() {
    const QString local = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    return std::filesystem::path((local.isEmpty() ? QDir::tempPath() : local).toStdString()) / "ratings";
}

struct ModelMetadata final {
    QString family;
    QString version;
    QString model_digest;
    QString runtime_digest;
};

std::optional<ModelMetadata> model_metadata(const QString& modelPath, const QString& modelId) {
    const QStringList id = modelId.split(QLatin1Char('/'));
    if (id.size() != 2 || id[0].isEmpty() || id[1].isEmpty()) return std::nullopt;
    QFile file{QDir(modelPath).filePath(QStringLiteral("metadata.json"))};
    if (!file.open(QIODevice::ReadOnly)) return std::nullopt;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject()) return std::nullopt;
    const QJsonObject object = document.object();
    ModelMetadata result{id[0], id[1], object.value(QStringLiteral("model_sha256")).toString(),
                         object.value(QStringLiteral("runtime_sha256")).toString()};
    static const QRegularExpression digest{QStringLiteral("^[0-9a-fA-F]{64}$")};
    if (!digest.match(result.model_digest).hasMatch() || !digest.match(result.runtime_digest).hasMatch())
        return std::nullopt;
    return result;
}

QString sync_script() {
    const QString configured = qEnvironmentVariable("DIAMOND_RATING_SYNC_SCRIPT");
    if (!configured.isEmpty()) return configured;
    const QStringList candidates = {
        QDir::current().filePath(QStringLiteral("tools/sync_ratings.sh")),
        QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("sync_ratings.sh"))};
    for (const QString& candidate : candidates) if (QFileInfo{candidate}.isFile()) return candidate;
    return {};
}

QString sync_binary() {
    const QString configured = qEnvironmentVariable("DIAMOND_RATING_SYNC_BINARY");
    if (!configured.isEmpty()) return configured;
#ifdef Q_OS_WIN
    const QString name = QStringLiteral("alphadiamond-rating-sync.exe");
#else
    const QString name = QStringLiteral("alphadiamond-rating-sync");
#endif
    const QDir app_dir{QCoreApplication::applicationDirPath()};
    const QStringList candidates = {app_dir.filePath(name), app_dir.filePath(QStringLiteral("../") + name),
                                    QDir::current().filePath(name)};
    for (const QString& candidate : candidates) if (QFileInfo{candidate}.isFile()) return candidate;
    return candidates.constFirst();
}

QString shell_program() {
    const QString configured = qEnvironmentVariable("DIAMOND_RATING_SYNC_SHELL");
    if (!configured.isEmpty()) return configured;
    const QString found = QStandardPaths::findExecutable(QStringLiteral("bash"));
    if (!found.isEmpty()) return found;
#ifdef Q_OS_WIN
    const QString git_bash = QStringLiteral("C:/Program Files/Git/bin/bash.exe");
    if (QFileInfo{git_bash}.isExecutable()) return git_bash;
#endif
    return {};
}

QString normalized_uuid(const QString& value) {
    const QUuid parsed = QUuid::fromString(value.trimmed());
    return parsed.isNull() ? QString{} : parsed.toString(QUuid::WithoutBraces);
}
}  // namespace

RatingBridge::RatingBridge(std::filesystem::path root, QObject* parent)
    : QObject(parent), root_(root.empty() ? default_root() : std::move(root)) {}

QString RatingBridge::installationId() {
    if (!installation_id_.isEmpty()) return installation_id_;
    const QString root = QString::fromStdString(root_.string());
    const QString identity_path = QDir(root).filePath(QStringLiteral("installation-id"));
    QFile identity_file{identity_path};
    if (identity_file.open(QIODevice::ReadOnly))
        installation_id_ = normalized_uuid(QString::fromUtf8(identity_file.readAll()));

    QSettings settings;
    if (installation_id_.isEmpty()) {
        installation_id_ = normalized_uuid(
            settings.value(QStringLiteral("ratings/installation_id")).toString());
    }
    if (installation_id_.isEmpty())
        installation_id_ = QUuid::createUuid().toString(QUuid::WithoutBraces);

    QDir().mkpath(root);
    QSaveFile saved_identity{identity_path};
    if (saved_identity.open(QIODevice::WriteOnly)) {
        saved_identity.write(installation_id_.toUtf8());
        saved_identity.write("\n");
        (void)saved_identity.commit();
    }
    settings.setValue(QStringLiteral("ratings/installation_id"), installation_id_);
    settings.sync();
    return installation_id_;
}

bool RatingBridge::writeProtocol(const QString& modelId, const QString& modelLabel,
                                 const QString& modelDigest, const QString& runtimeDigest) {
    const QStringList id = modelId.split(QLatin1Char('/'));
    if (id.size() != 2 || id[0].isEmpty() || id[1].isEmpty() || modelLabel.isEmpty()) return false;
    using Json = diamond_support::JsonValue;
    const auto model = diamond_orchestration::make_participant_identity(
        Json{Json::Object{{"kind", Json{"soo_model"}},
                          {"model_family", Json{id[0].toStdString()}},
                          {"model_version", Json{id[1].toStdString()}},
                          {"model_sha256", Json{modelDigest.toStdString()}},
                          {"runtime_sha256", Json{runtimeDigest.toStdString()}}}},
        modelLabel.toStdString());
    const auto human = diamond_orchestration::make_participant_identity(
        Json{Json::Object{{"installation_id", Json{installationId().toStdString()}},
                          {"kind", Json{"local_human"}}}}, "Local human");
    try {
        std::filesystem::create_directories(root_);
        const auto protocol = root_ / "protocol-v2.json";
        auto registry = std::filesystem::exists(protocol)
            ? diamond_orchestration::load_rating_registry(protocol)
            : diamond_orchestration::RatingRegistry{kProtocolId};
        registry.add_participant(model);
        registry.add_participant(human);
        diamond_orchestration::save_rating_registry(protocol, registry);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool RatingBridge::recordTerminalSooMatch(const QString& gameId, const QString& modelPath,
                                          const QString& modelId, const QString& modelLabel,
                                          int aiSeat, int winnerSeat, const QVariantList& turnOrder) {
    if (gameId.isEmpty() || modelPath.isEmpty() || (aiSeat != 1 && aiSeat != 2) ||
        (winnerSeat != 1 && winnerSeat != 2) || turnOrder.size() != 2) {
        setSyncState(QStringLiteral("Rating not recorded: invalid match identity."));
        return false;
    }
    const int first = turnOrder.at(0).toInt(), second = turnOrder.at(1).toInt();
    if ((first != 1 && first != 2) || (second != 1 && second != 2) || first == second) {
        setSyncState(QStringLiteral("Rating not recorded: invalid turn order."));
        return false;
    }
    const auto metadata = model_metadata(modelPath, modelId);
    if (!metadata || !writeProtocol(modelId, modelLabel, metadata->model_digest, metadata->runtime_digest)) {
        setSyncState(QStringLiteral("Rating not recorded: model identity unavailable."));
        return false;
    }
    using Json = diamond_support::JsonValue;
    const auto model = diamond_orchestration::make_participant_identity(
        Json{Json::Object{{"kind", Json{"soo_model"}},
                          {"model_family", Json{metadata->family.toStdString()}},
                          {"model_version", Json{metadata->version.toStdString()}},
                          {"model_sha256", Json{metadata->model_digest.toStdString()}},
                          {"runtime_sha256", Json{metadata->runtime_digest.toStdString()}}}},
        modelLabel.toStdString());
    const auto human = diamond_orchestration::make_participant_identity(
        Json{Json::Object{{"installation_id", Json{installationId().toStdString()}},
                          {"kind", Json{"local_human"}}}}, "Local human");
    const std::array<std::string, 2> participants = aiSeat == 1
        ? std::array{model.participant_id, human.participant_id}
        : std::array{human.participant_id, model.participant_id};
    const int loser = winnerSeat == 1 ? 2 : 1;
    try {
        const auto event = diamond_orchestration::make_soo_rating_event(
            0, kProtocolId, participants, {1, 2}, {first, second}, "standard-2p-v1", true,
            participants[winnerSeat - 1], participants[loser - 1], gameId.toStdString(),
            aiSeat == 1 ? std::vector{model, human} : std::vector{human, model});
        diamond_orchestration::RatingEventOutbox outbox{root_ / "outbox"};
        const bool published = outbox.publish(event);
        setSyncState(published ? QStringLiteral("Rating queued for sync.")
                               : QStringLiteral("Rating already queued for sync."));
        syncPending();
        return published;
    } catch (const std::exception&) {
        setSyncState(QStringLiteral("Rating not recorded: durable outbox failed."));
        return false;
    }
}

int RatingBridge::pendingEventCount() const {
    try {
        return static_cast<int>(diamond_orchestration::RatingEventOutbox{root_ / "outbox"}.pending_events().size());
    } catch (const std::exception&) { return 0; }
}

bool RatingBridge::syncRunning() const { return sync_process_ && sync_process_->state() != QProcess::NotRunning; }

void RatingBridge::syncPending() {
    if (syncRunning() || pendingEventCount() == 0) return;
    if (qEnvironmentVariableIntValue("DIAMOND_RATING_SYNC_DISABLED") != 0) {
        setSyncState(QStringLiteral("Rating queued; sync disabled for this process."));
        return;
    }
    const QString script = sync_script(), binary = sync_binary(), shell = shell_program();
    if (script.isEmpty() || shell.isEmpty() || !QFileInfo{binary}.isFile()) {
        setSyncState(QStringLiteral("Rating queued; sync will retry when launcher is available."));
        return;
    }
    sync_process_ = std::make_unique<QProcess>();
    sync_process_->setProcessChannelMode(QProcess::SeparateChannels);
    connect(sync_process_.get(), &QProcess::finished, this, [this](int code, QProcess::ExitStatus status) {
        setSyncState(code == 0 && status == QProcess::NormalExit
                         ? QStringLiteral("Ratings synced.")
                         : QStringLiteral("Rating sync deferred; events remain queued."));
    });
    connect(sync_process_.get(), &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        setSyncState(QStringLiteral("Rating sync deferred; events remain queued."));
    });
    const QString bucket = qEnvironmentVariable("DIAMOND_RATING_BUCKET", QStringLiteral("ubraket513/AlphaDiamond"));
    const QString outbox = QString::fromStdString((root_ / "outbox").string());
    const QString protocol = QString::fromStdString((root_ / "protocol-v2.json").string());
    setSyncState(QStringLiteral("Syncing ratings in background."));
    sync_process_->start(shell, {script, "--bucket", bucket, "--outbox", outbox, "--protocol", protocol,
                                 "--binary", binary});
}

void RatingBridge::setSyncState(const QString& state) {
    if (sync_state_ == state) return;
    sync_state_ = state;
    Q_EMIT changed();
}
