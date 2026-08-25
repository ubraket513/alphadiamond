#pragma once

#include <QByteArray>
#include <QHash>
#include <QObject>
#include <QStringList>
#include <QVariantList>

class QNetworkAccessManager;
class QJsonObject;
class QUrl;

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
  explicit ModelCatalog(QObject *parent = nullptr);
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
  Q_INVOKABLE void selectModel(const QString &modelId);
  Q_INVOKABLE void downloadModel(const QString &modelId);
  bool activateSelected(); // Commits pending selection at new-game boundary.
Q_SIGNALS:
  void changed();

private:
  friend struct ModelCatalogTestAccess;
  struct LocalModel {
    QString id, path, version, model_digest, runtime_digest;
    int training_step = 0, training_simulations = 0;
  };
  struct Artifact {
    QString id, version, model_digest, runtime_digest, github_url,
        hugging_face_url;
    int training_step = 0, training_simulations = 0;
    bool github = false, hugging_face = false;
  };
  void scanLocal();
  void rebuildRows();
  void setStatus(const QString &status);
  void beginWork();
  void endWork();
  void fetchGitHub();
  void fetchHuggingFace();
  void parseGitHubIndex(const QByteArray &payload);
  void parseGitHubTree(const QByteArray &payload);
  void parseHuggingFaceTree(const QByteArray &payload);
  void fetchHuggingFaceMetadata(const QString &modelId);
  void fetchHuggingFaceRatings();
  bool parseHuggingFaceRatings(const QByteArray &payload, bool persist);
  void loadCachedRatings();
  QString latestRating(const QString &id, const QString &modelDigest,
                       const QString &runtimeDigest) const;
  void addArtifact(const QString &source, const QJsonObject &metadata,
                   const QString &webUrl);
  void failDigestMismatch(const QString &modelId);
  void startDownload(const QString &modelId, bool huggingFace);
  void requestDownloadFile(const QString &relativePath, const QUrl &url);
  void completeDownload(bool success, const QString &error = {});
  bool readLocalModel(const QString &path, LocalModel *model) const;
  bool validateDownloadedModel(const QString &path, QString *error) const;
  QString destinationFor(const QString &modelId) const;
  QNetworkAccessManager *network_;
  QVariantList models_;
  QList<LocalModel> local_models_;
  QHash<QString, QString> local_paths_;
  QHash<QString, Artifact> artifacts_;
  QHash<QString, QStringList> github_files_, hugging_face_files_;
  QHash<QString, QString> ratings_by_identity_;
  QString local_root_, selected_id_, selected_path_, active_id_, active_path_;
  QString status_ = QStringLiteral("Local models ready."), catalog_error_;
  int busy_count_ = 0;
  QString download_id_, download_staging_, download_destination_,
      download_error_;
  int download_pending_ = 0;
  bool download_from_hugging_face_ = false;
};
