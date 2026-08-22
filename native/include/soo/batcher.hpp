// Global inference batcher with tickets.
//
// Work-conserving: the timer bounds only how long we wait for *future*
// arrivals, never how long an already-queued request sits. Section 5 of
// docs/native_selfplay_phase0.md.
//
// There is no minimum batch size, and there must not be one. A lane has at
// most one request outstanding, so a floor above the number of lanes in flight
// would deadlock.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace soo {

class Batcher {
  public:
    Batcher(int max_batch, int max_wait_us) : max_batch_(max_batch), max_wait_us_(max_wait_us) {}

    // Called by a search worker. Never blocks on the evaluation; the lane is
    // handed to the batcher and the worker goes and finds other work.
    void submit(int lane_id) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(lane_id);
            ++submitted_;
        }
        arrival_.notify_one();
    }

    // Fill `batch` with up to max_batch lane ids. Returns false once stopped
    // and drained. Blocks until at least one request arrives.
    bool collect(std::vector<int>& batch);

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        arrival_.notify_all();
    }

    // Queue depth right now; sampled for the scheduler report.
    size_t depth() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    uint64_t wakeups() const { return wakeups_; }

  private:
    const int max_batch_;
    const int max_wait_us_;

    std::mutex mutex_;
    std::condition_variable arrival_;
    std::vector<int> queue_;
    bool stopped_ = false;
    uint64_t submitted_ = 0;
    uint64_t wakeups_ = 0;
};

}  // namespace soo
