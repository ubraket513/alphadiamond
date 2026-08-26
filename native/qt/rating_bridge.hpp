#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QVariantList>

#include <filesystem>
#include <memory>

// Qt boundary for local Soo results. The core outbox owns durability; this
// class only supplies local identities and starts a best-effort background sync.
class RatingBridge final : public QObject {
    Q_OBJECT
  public:
    explicit RatingBridge(std::filesystem::path root = {}, QObject* parent = nullptr);

    bool recordTerminalSooMatch(const QString& gameId, const QString& modelPath,
                                const QString& modelId, const QString& modelLabel, int aiSeat,
                                int winnerSeat, const QVariantList& turnOrder);
    void syncPending();

    int pendingEventCount() const;
    QString syncState() const {
        return sync_state_;
    }
    bool syncRunning() const;

  Q_SIGNALS:
    void changed();

  private:
    std::filesystem::path root_;
    std::unique_ptr<QProcess> sync_process_;
    QString sync_state_ = QStringLiteral("Idle");
    QString installation_id_;

    QString installationId();
    bool writeProtocol(const QString& modelId, const QString& modelLabel,
                       const QString& modelDigest, const QString& runtimeDigest);
    void setSyncState(const QString& state);
};
