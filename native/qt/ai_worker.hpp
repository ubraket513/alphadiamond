#pragma once

#include <QThread>

#include <atomic>
#include <functional>

class NativeAiWorker final : public QObject {
    Q_OBJECT
  public:
    explicit NativeAiWorker(QObject* parent = nullptr);
    ~NativeAiWorker() override;

    void start(quint64 generation, std::function<int()> task);
    void cancel();
    bool isRunning() const { return running_.load(); }

  signals:
    void resultReady(quint64 generation, int action);
    void failed(quint64 generation, QString message);
    void cancelled(quint64 generation);

  private:
    std::atomic_bool cancel_requested_{false};
    std::atomic_bool running_{false};
    QThread* thread_ = nullptr;
};
