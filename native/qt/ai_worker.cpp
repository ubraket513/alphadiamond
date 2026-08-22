#include "ai_worker.hpp"

#include <QString>

NativeAiWorker::NativeAiWorker(QObject* parent) : QObject(parent) {}

NativeAiWorker::~NativeAiWorker() {
    cancel();
    if (thread_) {
        thread_->wait();
        delete thread_;
        thread_ = nullptr;
    }
}

void NativeAiWorker::start(quint64 generation, std::function<int()> task) {
    if (running_.exchange(true)) return;
    cancel_requested_.store(false);
    thread_ = QThread::create([this, generation, task = std::move(task)]() mutable {
        try {
            const int action = task();
            running_.store(false);
            if (cancel_requested_.load()) Q_EMIT cancelled(generation);
            else Q_EMIT resultReady(generation, action);
        } catch (const std::exception& error) {
            running_.store(false);
            Q_EMIT failed(generation, QString::fromUtf8(error.what()));
        }
        Q_EMIT becameIdle();
    });
    QThread* launched_thread = thread_;
    QObject::connect(thread_, &QThread::finished, this, [this, launched_thread] {
        if (thread_ == launched_thread) thread_ = nullptr;
        launched_thread->deleteLater();
    });
    thread_->start();
}

void NativeAiWorker::cancel() { cancel_requested_.store(true); }
