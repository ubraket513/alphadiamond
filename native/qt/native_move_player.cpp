#include "native_move_player.hpp"

#include <QSoundEffect>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>

#include <QUrl>

#include <algorithm>
#include <cmath>

NativeMovePlayer::NativeMovePlayer(QObject* parent) : QObject(parent) {
    setObjectName(QStringLiteral("movePlayer"));
    const QString path = QDir(QCoreApplication::applicationDirPath())
                             .filePath(QStringLiteral("assets/sounds/move.wav"));
    if (!QFileInfo::exists(path)) {
        status_ = QStringLiteral("Sound file missing: %1").arg(path);
        return;
    }

    player_ = new QSoundEffect(this);
    player_->setVolume(volume_);
    connect(player_, &QSoundEffect::statusChanged, this, [this] {
        loaded_ = player_->status() == QSoundEffect::Ready;
        if (player_->status() == QSoundEffect::Error)
            setStatus(QStringLiteral("Cannot load move.wav."));
        if (loaded_ && pending_) {
            pending_ = false;
            startPlayback();
        }
        Q_EMIT changed();
    });
    player_->setSource(QUrl::fromLocalFile(path));
}

bool NativeMovePlayer::available() const {
    return player_ != nullptr && status_.isEmpty();
}

void NativeMovePlayer::setMuted(bool muted) {
    if (muted_ == muted) return;
    muted_ = muted;
    if (player_)
        player_->setMuted(muted_);
    Q_EMIT changed();
}

void NativeMovePlayer::setVolume(double volume) {
    if (!std::isfinite(volume)) volume = 0.6;
    volume = std::clamp(volume, 0.0, 1.0);
    const bool should_unmute = volume > 0.0 && muted_;
    if (volume_ == volume && !should_unmute) return;
    volume_ = volume;
    if (player_)
        player_->setVolume(static_cast<float>(volume_));
    if (should_unmute) {
        muted_ = false;
        if (player_)
            player_->setMuted(false);
    }
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
    player_->play();
}
