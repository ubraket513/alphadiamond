#include "soo/batcher.hpp"

#include <algorithm>
#include <chrono>

namespace soo {

bool Batcher::collect(std::vector<int>& batch) {
    using Clock = std::chrono::steady_clock;
    batch.clear();
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait for the first arrival. Nothing is in flight yet, so there is no
    // window to hold open and no deadline to honour.
    while (queue_.empty() && !stopped_) {
        arrival_.wait(lock);
        ++wakeups_;
    }
    if (queue_.empty() && stopped_) return false;

    const auto deadline = Clock::now() + std::chrono::microseconds(max_wait_us_);
    for (;;) {
        // Drain everything already queued, up to the cap.
        const size_t take = std::min(queue_.size(), static_cast<size_t>(max_batch_) - batch.size());
        batch.insert(batch.end(), queue_.begin(), queue_.begin() + static_cast<long>(take));
        queue_.erase(queue_.begin(), queue_.begin() + static_cast<long>(take));

        if (batch.size() >= static_cast<size_t>(max_batch_)) break;
        if (stopped_) break;
        if (Clock::now() >= deadline) break;
        // Wait for the next arrival or the deadline, whichever comes first.
        arrival_.wait_until(lock, deadline);
        ++wakeups_;
    }
    return !batch.empty();
}

}  // namespace soo
