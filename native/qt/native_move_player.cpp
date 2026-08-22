#include "native_move_player.hpp"

#include <QAudioOutput>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QMediaPlayer>
#include <QUrl>

#include <algorithm>
#include <cmath>

NativeMovePlayer::NativeMovePlayer(QObject* parent) : QObject(parent) {
    setObjectName(QStringLiteral("movePlayer"));
    const QString path = QDir(QCoreApplication::applicationDirPath())
                             .filePath(QStringLiteral("assets/sounds/move.m4a"));
    if (!QFileInfo::exists(path)) {
        status_ = QStringLiteral("Sound file missing: %1").arg(path);
        return;
    }

    output_ = new QAudioOutput(this);
    output_->setVolume(static_cast<float>(volume_));
    player_ = new QMediaPlayer(this);
    player_->setAudioOutput(output_);
    connect(player_, &QMediaPlayer::mediaStatusChanged, this,
            [this](QMediaPlayer::MediaStatus media_status) {
                if (media_status == QMediaPlayer::InvalidMedia) {
                    setStatus(QStringLiteral("Cannot decode move.m4a; no compatible codec is available."));
                } else if (media_status == QMediaPlayer::LoadedMedia ||
                           media_status == QMediaPlayer::BufferedMedia) {
                    loaded_ = true;
                    Q_EMIT changed();
                    if (pending_) {
                        pending_ = false;
                        startPlayback();
                    }
                }
            });
    connect(player_, &QMediaPlayer::errorOccurred, this,
            [this](QMediaPlayer::Error error, const QString& message) {
                if (error == QMediaPlayer::NoError) return;
                setStatus(message.isEmpty() ? QStringLiteral("Move sound playback failed.") : message);
            });
    player_->setSource(QUrl::fromLocalFile(path));
}

bool NativeMovePlayer::available() const {
    return player_ != nullptr && output_ != nullptr && status_.isEmpty();
}

void NativeMovePlayer::setMuted(bool muted) {
    if (muted_ == muted) return;
    muted_ = muted;
    Q_EMIT changed();
}

void NativeMovePlayer::setVolume(double volume) {
    if (!std::isfinite(volume)) volume = 0.6;
    volume = std::clamp(volume, 0.0, 1.0);
    const bool should_unmute = volume > 0.0 && muted_;
    if (volume_ == volume && !should_unmute) return;
    volume_ = volume;
    if (output_) output_->setVolume(static_cast<float>(volume_));
    if (should_unmute) muted_ = false;
    Q_EMIT changed();
}

bool NativeMovePlayer::play() {
    ++play_request_count_;
    if (!enabled()) return false;
    if (!loaded_) {
        pending_ = true;
        return false;
    }
    startPlayback();
    return true;
}

void NativeMovePlayer::setStatus(const QString& status) {
    if (status_ == status) return;
    status_ = status;
    pending_ = false;
    Q_EMIT changed();
}

void NativeMovePlayer::startPlayback() {
    if (!player_) return;
    player_->stop();
    player_->setPosition(0);
    player_->play();
}
