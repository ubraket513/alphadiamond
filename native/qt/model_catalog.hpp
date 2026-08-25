#pragma once

#include <QObject>
#include <QHash>
#include <QStringList>
#include <QVariantList>

class QNetworkAccessManager;
class QNetworkReply;
class QProcess;

class ModelCatalog final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList models READ models NOTIFY changed)
    Q_PROPERTY(QString status READ status NOTIFY changed)
    Q_PROPERTY(bool busy READ busy NOTIFY changed)
    Q_PROPERTY(QString selectedModelId READ selectedModelId NOTIFY changed)
    Q_PROPERTY(QString activeModelId READ activeModelId NOTIFY changed)
    Q_PROPERTY(QString selectedModelLabel READ selectedModelLabel NOTIFY changed)
    Q_PROPERTY(QString activeModelLabel READ activeModelLabel NOTIFY changed)
    Q_PROPERTY(QString localRoot READ localRoot CONSTANT)

  public:
    explicit ModelCatalog(QObject* parent = nullptr);

    QVariantList models() const { return models_; }
    QString status() const { return status_; }
    bool busy() const { return busy_count_ > 0; }
    QString selectedModelId() const { return selected_id_; }
    QString activeModelId() const { return active_id_; }
    QString selectedModelLabel() const;
    QString activeModelLabel() const;
    QString localRoot() const { return local_root_; }
    QString selectedModelPath() const { return selected_path_; }
    QString activeModelPath() const { return active_path_; }

    Q_INVOKABLE void refresh();
    Q_INVOKABLE void selectModel(const QString& modelId);
    Q_INVOKABLE void downloadModel(const QString& source, const QString& modelId);

    // Commits the pending selection at the new-game boundary.
    bool activateSelected();

  Q_SIGNALS:
    void changed();

  private:
    struct LocalModel {
        QString id;
        QString path;
        QString version;
        int training_step = 0;
        bool packaged = false;
    };

    void scanLocal();
    void rebuildRows();
    void setStatus(const QString& status);
    void beginWork();
    void endWork();
    void fetchGitHub();
    void fetchHuggingFace();
    void parseGitHubIndex(const QByteArray& payload);
    void parseGitHubTree(const QByteArray& payload);
    void parseGitHubReleases(const QByteArray& payload);
    void parseHuggingFace(const QByteArray& payload);
    void startHfDownload(const QString& modelId);
    void startGitHubDownload(const QString& modelId);
    void completeDownload(bool success, const QString& error = {});
    bool readLocalModel(const QString& path, bool packaged, LocalModel* model) const;
    QString hfExecutable() const;
    QString destinationFor(const QString& modelId) const;
    QVariantMap remoteRow(const QString& source, const QString& modelId,
                          const QString& version, int trainingStep,
                          bool compatible, const QString& webUrl,
                          const QString& note = {}) const;

    QNetworkAccessManager* network_;
    QVariantList models_;
    QVariantList remote_models_;
    QList<LocalModel> local_models_;
    QHash<QString, QString> local_paths_;
    QHash<QString, QStringList> github_files_;
    QString local_root_;
    QString selected_id_;
    QString selected_path_;
    QString active_id_;
    QString active_path_;
    QString status_ = QStringLiteral("Local models ready.");
    int busy_count_ = 0;

    QString download_source_;
    QString download_id_;
    QString download_staging_;
    QString download_destination_;
    int download_pending_ = 0;
    QString download_error_;
};
