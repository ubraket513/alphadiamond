#pragma once

#include <QThread>

#include <atomic>
#include <functional>

#include "search_telemetry.hpp"

class NativeAiWorker final : public QObject {
    Q_OBJECT
  public:
    explicit NativeAiWorker(QObject* parent = nullptr);
    ~NativeAiWorker() override;

    void start(quint64 generation, std::function<AiSearchResult()> task);
    void cancel();
    bool isRunning() const { return running_.load(); }

  Q_SIGNALS:
    void resultReady(quint64 generation, AiSearchResult result);
    void failed(quint64 generation, QString message);
    void cancelled(quint64 generation);
    void becameIdle();

  private:
    std::atomic_bool cancel_requested_{false};
    std::atomic_bool running_{false};
    QThread* thread_ = nullptr;
};
