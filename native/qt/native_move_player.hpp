#pragma once

#include <QObject>
#include <QString>

class QAudioOutput;
class QMediaPlayer;

class NativeMovePlayer final : public QObject {
    Q_OBJECT

  public:
    explicit NativeMovePlayer(QObject* parent = nullptr);

    bool available() const;
    bool enabled() const { return available() && !muted_; }
    bool muted() const { return muted_; }
    double volume() const { return volume_; }
    QString status() const { return status_; }
    int playRequestCount() const { return play_request_count_; }

    void setMuted(bool muted);
    void setVolume(double volume);
    bool play();

  Q_SIGNALS:
    void changed();

  private:
    void setStatus(const QString& status);
    void startPlayback();

    QMediaPlayer* player_ = nullptr;
    QAudioOutput* output_ = nullptr;
    bool muted_ = false;
    bool loaded_ = false;
    bool pending_ = false;
    double volume_ = 0.6;
    QString status_;
    int play_request_count_ = 0;
};
