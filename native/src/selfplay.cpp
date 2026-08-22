#include "soo/selfplay.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "soo/batcher.hpp"
#include "soo/encoder.hpp"
#include "soo/evaluator.hpp"
#include "soo/rules.hpp"

namespace soo {
namespace {

using Clock = std::chrono::steady_clock;

inline double seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// One logical game. Its SearchSession holds the pending request, so the
// batcher reads the payload straight out of the lane -- no copy into a queue.
struct Lane {
    Lane(const Match& match, const MCTSConfig& config) : session(match, config) {}

    SearchSession session;
    State state;
    int move_count = 0;
    uint64_t salt = 0;
    EvalOutcome outcome;
    Clock::time_point submitted_at;
};

// Dummy evaluator: request-dependent, per-lane salted, with configurable
// artificial latency.
//
// The salt is not decoration. With temperature = 0 and a request-deterministic
// evaluator, every lane starting from the same opening would play the *same*
// game in lockstep, which would flatter batch formation badly and make the
// whole sweep meaningless. Salting makes the trajectories diverge the way real
// self-play does.
// The salt has to avalanche, not merely perturb.
//
// A first version did `hash ^= salt + K + (hash << 6) + (hash >> 2)`, which
// moves mostly low bits -- and the value is taken from `hash >> 11`, so those
// bits are discarded. Lane values then differed at the 11th decimal place,
// every lane picked the same moves, and 512 "independent" games were one game
// replayed 512 times. That would have made every batching number in this gate
// meaningless, in the flattering direction. splitmix64's finalizer spreads a
// single changed bit across the whole word.
uint64_t salted_hash(const Encoded& encoded, const std::vector<int32_t>& actions, uint64_t salt) {
    uint64_t hash = request_hash(encoded, actions) ^ (salt * 0x9e3779b97f4a7c15ULL);
    hash ^= hash >> 30;
    hash *= 0xbf58476d1ce4e5b9ULL;
    hash ^= hash >> 27;
    hash *= 0x94d049bb133111ebULL;
    hash ^= hash >> 31;
    return hash;
}

// Priors must VARY, not just the value.
//
// A first version returned a uniform prior and salted only the value. Every
// lane then played an identical game: with equal priors the exploration term
// dominates, each simulation expands a fresh root child, every root edge ends
// on one visit, and the temperature-0 tie-break picks the lowest action id --
// so the value never enters the decision at all. 16 lanes, 1 distinct
// trajectory. A sweep run on that would be measuring one game replayed N times.
void fill_outcome_impl(const Encoded& encoded, const std::vector<int32_t>& actions,
                       uint64_t salt, EvalOutcome& outcome) {
    const uint64_t hash = salted_hash(encoded, actions, salt);

    outcome.priors.clear();
    outcome.priors.reserve(actions.size());
    double total = 0.0;
    for (const int32_t action : actions) {
        uint64_t action_hash = hash;
        action_hash ^= static_cast<uint64_t>(action) * 0x9e3779b97f4a7c15ULL;
        action_hash *= 0x100000001b3ULL;
        const double weight =
            static_cast<double>(action_hash >> 11) / 9007199254740992.0 + 0.25;
        outcome.priors.push_back(weight);
        total += weight;
    }
    for (double& prior : outcome.priors) prior /= total;
    outcome.value = static_cast<double>(hash >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

// A queue of runnable lanes. Workers block here and nowhere else.
class ReadyQueue {
  public:
    void push(int lane_id) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(lane_id);
        }
        ready_.notify_one();
    }

    void push_many(const std::vector<int>& lanes) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.insert(queue_.end(), lanes.begin(), lanes.end());
        }
        ready_.notify_all();
    }

    bool pop(int& lane_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !stopped_) ready_.wait(lock);
        if (queue_.empty()) return false;
        lane_id = queue_.back();
        queue_.pop_back();
        return true;
    }

    size_t depth() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        ready_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<int> queue_;
    bool stopped_ = false;
};

}  // namespace

void DummyBatchEvaluator::evaluate(std::vector<BatchItem>& batch) {
    for (BatchItem& item : batch) {
        fill_outcome_impl(*item.encoded, *item.actions, item.salt, *item.outcome);
    }
    // Artificial inference cost, per batch, the way a GPU call bills.
    if (latency_ms_ > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(latency_ms_));
    }
}

SchedulerMetrics run_scheduler(const Match& match, const State& opening,
                               const SchedulerConfig& config, BatchEvaluator& batch_evaluator) {
    MCTSConfig search_config;
    search_config.simulations = config.simulations;
    search_config.dirichlet_epsilon = 0.0;

    std::vector<std::unique_ptr<Lane>> lanes;
    lanes.reserve(static_cast<size_t>(config.games));
    for (int i = 0; i < config.games; ++i) {
        auto lane = std::make_unique<Lane>(match, search_config);
        lane->state = opening;
        lane->salt = 0x1000003ULL * static_cast<uint64_t>(i + 1);
        lane->session.begin(lane->state, 0.0, false);
        lanes.push_back(std::move(lane));
    }

    Batcher batcher(config.max_batch, config.max_wait_us);
    ReadyQueue ready;
    SchedulerMetrics metrics;
    std::mutex metrics_mutex;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> moves{0};
    std::atomic<uint64_t> finished{0};
    std::atomic<int> waiting{0};
    std::vector<std::vector<int32_t>> lane_moves(static_cast<size_t>(config.games));
    std::atomic<int> lanes_done{0};

    // A Python callback that raises, or a rules violation on a worker, would
    // otherwise escape a std::thread and terminate the process. Capture the
    // first failure, tear the run down so nobody waits on a dead thread, and
    // rethrow it on the calling thread where the caller can see it.
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto fail = [&](std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::move(error);
        }
        stop.store(true, std::memory_order_relaxed);
        ready.stop();
        batcher.stop();
    };

    for (int i = 0; i < config.games; ++i) ready.push(i);

    const auto started = Clock::now();

    // --- search workers -----------------------------------------------------
    std::vector<std::thread> workers;
    std::vector<double> busy(static_cast<size_t>(config.threads), 0.0);
    for (int w = 0; w < config.threads; ++w) {
        workers.emplace_back([&, w] {
          try {
            int lane_id = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                if (!ready.pop(lane_id)) break;
                const auto work_start = Clock::now();
                Lane& lane = *lanes[static_cast<size_t>(lane_id)];

                // Drive this lane until it needs an evaluation or has a move.
                const SearchSession::Status status = lane.session.advance();
                if (status == SearchSession::Status::NeedsEvaluation) {
                    // Request-only work happens here, on an idle worker, rather
                    // than on the evaluator thread.
                    BatchItem item{&lane.session.pending_state(),
                                   &lane.session.pending_features(),
                                   &lane.session.pending_actions(), lane.salt, &lane.outcome};
                    batch_evaluator.prepare(item);
                    lane.submitted_at = Clock::now();
                    waiting.fetch_add(1, std::memory_order_relaxed);
                    busy[static_cast<size_t>(w)] += seconds_since(work_start);
                    // Hand the lane over and go find other work. Not blocking
                    // here is the entire point of the design.
                    batcher.submit(lane_id);
                    continue;
                }

                // A move is ready: play it and start the next search.
                const int32_t action = lane.session.result().selected_action;
                if (config.trace_moves) {
                    // Only this worker ever touches this lane right now: a lane
                    // is either on the ready queue, owned by one worker, or held
                    // by the batcher. Never two at once.
                    lane_moves[static_cast<size_t>(lane_id)].push_back(action);
                }
                lane.state = apply_action(lane.state, match,
                                          to_physical_action(action, match,
                                                             lane.state.current_player));
                ++lane.move_count;
                moves.fetch_add(1, std::memory_order_relaxed);

                if (config.stop_after_moves > 0 &&
                    lane.move_count >= config.stop_after_moves) {
                    // Deterministic stop: this lane is finished for good, so the
                    // comparison across thread counts is over the same work.
                    if (lanes_done.fetch_add(1, std::memory_order_acq_rel) + 1 >=
                        config.games) {
                        stop.store(true, std::memory_order_relaxed);
                        ready.stop();
                        batcher.stop();
                    }
                    busy[static_cast<size_t>(w)] += seconds_since(work_start);
                    continue;
                }

                if (lane.state.status == kFinished || lane.move_count >= config.max_moves) {
                    finished.fetch_add(1, std::memory_order_relaxed);
                    lane.state = opening;
                    lane.move_count = 0;
                    // A fresh salt so the replacement game is a new trajectory
                    // rather than a replay of the one that just ended.
                    lane.salt = lane.salt * 6364136223846793005ULL + 1442695040888963407ULL;
                }
                lane.session.begin(lane.state, 0.0, false);
                busy[static_cast<size_t>(w)] += seconds_since(work_start);
                ready.push(lane_id);
            }
          } catch (...) {
            fail(std::current_exception());
          }
        });
    }

    // --- batcher / evaluator thread ----------------------------------------
    std::thread evaluator([&] {
      try {
        std::vector<int> batch;
        std::vector<BatchItem> items;
        while (batcher.collect(batch)) {
            const auto dispatch_at = Clock::now();
            const int ready_depth = static_cast<int>(ready.depth());
            const int waiting_now = waiting.load(std::memory_order_relaxed);

            items.clear();
            items.reserve(batch.size());
            for (const int lane_id : batch) {
                Lane& lane = *lanes[static_cast<size_t>(lane_id)];
                // pending_state(), not lane.state: the request belongs to the
                // node being expanded, which is only the root on the first
                // expansion of each move.
                items.push_back(BatchItem{&lane.session.pending_state(),
                                          &lane.session.pending_features(),
                                          &lane.session.pending_actions(), lane.salt,
                                          &lane.outcome});
            }
            const auto eval_start = Clock::now();
            batch_evaluator.evaluate(items);
            const double eval_seconds = seconds_since(eval_start);

            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                metrics.batch_sizes.push_back(static_cast<uint32_t>(batch.size()));
                metrics.ready_depth.push_back(static_cast<uint32_t>(ready_depth));
                metrics.waiting.push_back(static_cast<uint32_t>(waiting_now));
                metrics.evaluations += batch.size();
                ++metrics.batches;
                metrics.evaluator_seconds += eval_seconds;
                for (const int lane_id : batch) {
                    metrics.wait_ns.push_back(static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            dispatch_at - lanes[static_cast<size_t>(lane_id)]->submitted_at)
                            .count()));
                }
            }

            for (const int lane_id : batch) {
                Lane& lane = *lanes[static_cast<size_t>(lane_id)];
                lane.session.supply(lane.outcome);
                waiting.fetch_sub(1, std::memory_order_relaxed);
            }
            ready.push_many(batch);
        }
      } catch (...) {
        fail(std::current_exception());
      }
    });

    if (config.stop_after_moves > 0) {
        // Bounded run: the workers stop themselves once every lane is done.
        for (std::thread& worker : workers) worker.join();
        stop.store(true, std::memory_order_relaxed);
        ready.stop();
        batcher.stop();
        evaluator.join();
        metrics.wall_seconds = seconds_since(started);
        metrics.moves = moves.load();
        metrics.games_finished = finished.load();
        metrics.batcher_wakeups = batcher.wakeups();
        for (const double value : busy) metrics.worker_busy_seconds += value;
        if (config.trace_moves) metrics.lane_moves = std::move(lane_moves);
        if (failure) std::rethrow_exception(failure);
        return metrics;
    }

    std::this_thread::sleep_for(std::chrono::duration<double>(config.seconds));
    stop.store(true, std::memory_order_relaxed);
    ready.stop();
    batcher.stop();
    for (std::thread& worker : workers) worker.join();
    evaluator.join();

    metrics.wall_seconds = seconds_since(started);
    metrics.moves = moves.load();
    metrics.games_finished = finished.load();
    metrics.batcher_wakeups = batcher.wakeups();
    for (const double value : busy) metrics.worker_busy_seconds += value;
    if (config.trace_moves) metrics.lane_moves = std::move(lane_moves);
    if (failure) std::rethrow_exception(failure);
    return metrics;
}

}  // namespace soo
