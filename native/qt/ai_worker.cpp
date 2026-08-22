#include "ai_worker.hpp"

#include <QString>

NativeAiWorker::NativeAiWorker(QObject* parent) : QObject(parent) {}

NativeAiWorker::~NativeAiWorker() {
    cancel();
    if (thread_) thread_->wait();
}

void NativeAiWorker::start(quint64 generation, std::function<int()> task) {
    if (running_.exchange(true)) return;
    cancel_requested_.store(false);
    thread_ = QThread::create([this, generation, task = std::move(task)]() mutable {
        try {
            const int action = task();
            if (cancel_requested_.load()) emit cancelled(generation);
            else emit resultReady(generation, action);
        } catch (const std::exception& error) {
            emit failed(generation, QString::fromUtf8(error.what()));
        }
        running_.store(false);
    });
    QObject::connect(thread_, &QThread::finished, this, [this] { thread_ = nullptr; });
    thread_->start();
}

void NativeAiWorker::cancel() { cancel_requested_.store(true); }
