#include "soo/selfplay.hpp"

#include "soo/episode_search.hpp"

#include <algorithm>
#include <atomic>
#include <deque>
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
    // The lane's own game seed.  Python gives each game a derived seed and then
    // uses ``seed + move_count`` per move; a lane mirrors that, so a lane's
    // move is reproducible from (game seed, move number) alone.
    uint64_t game_seed = 0;
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
    search_config.dirichlet_alpha = config.dirichlet_alpha;
    search_config.dirichlet_epsilon = config.dirichlet_epsilon;

    // The temperature schedule, exactly SooSelfPlayRunner's: full temperature
    // for the first ``temperature_moves`` moves of a game, greedy after.
    const auto temperature_for = [&config](int move_count) {
        return move_count < config.temperature_moves ? config.temperature : 0.0;
    };

    std::vector<std::unique_ptr<Lane>> lanes;
    lanes.reserve(static_cast<size_t>(config.games));
    for (int i = 0; i < config.games; ++i) {
        auto lane = std::make_unique<Lane>(match, search_config);
        lane->state = opening;
        lane->salt = 0x1000003ULL * static_cast<uint64_t>(i + 1);
        // Consecutive lane indices must not give correlated streams; the
        // session's RNG runs its seed through splitmix64 for exactly this.
        lane->game_seed = config.seed + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(i + 1);
        lane->session.reseed(lane->game_seed);
        lane->session.begin(lane->state, temperature_for(0), false);
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
                    // rather than a replay of the one that just ended.  With a
                    // real evaluator the salt does nothing (pitfall 7.9) and the
                    // game seed is what carries the divergence, so advance both.
                    lane.salt = lane.salt * 6364136223846793005ULL + 1442695040888963407ULL;
                    lane.game_seed =
                        lane.game_seed * 6364136223846793005ULL + 1442695040888963407ULL;
                }
                lane.session.reseed(lane.game_seed + static_cast<uint64_t>(lane.move_count));
                lane.session.begin(lane.state, temperature_for(lane.move_count), false);
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

// --------------------------------------------------------------------------
// Episode production
// --------------------------------------------------------------------------
namespace {

// One game being played to completion.  Unlike the benchmark Lane this never
// recycles: when its game ends the lane retires and the run finishes once every
// lane has.
struct EpisodeLane {
    EpisodeLane(const Match& match, const MCTSConfig& config) : session(match, config) {}

    // Seat-agnostic: Soo's scalar search and Min's vector search behind one
    // handle, because everything else about a lane is the same for both.
    EpisodeSearch session;
    State state;
    int move_count = 0;
    uint64_t game_seed = 0;
    std::optional<Clock::time_point> deadline;
    bool done = false;
    EvalOutcome outcome;
    // Which job this lane is currently playing.  A lane outlives its game.
    size_t job_index = 0;
    // Dynamics keys of the recent plies, newest last, for the repetition
    // trigger.  Bounded by repeat_window, so this is a handful of integers.
    std::deque<uint64_t> recent;

    void remember(uint64_t key, size_t window) {
        recent.push_back(key);
        while (recent.size() > window) recent.pop_front();
    }

    bool seen_recently(uint64_t key) const {
        return std::find(recent.begin(), recent.end(), key) != recent.end();
    }
};

}  // namespace

std::vector<Episode> run_episodes(const Match& match, const std::vector<EpisodeJob>& jobs,
                                  const EpisodeConfig& config, BatchEvaluator& batch_evaluator,
                                  EpisodeMetrics& metrics) {
    if (jobs.empty()) return {};
    if (config.threads < 1) throw std::invalid_argument("threads must be positive");
    if (config.max_batch < 1) throw std::invalid_argument("max_batch must be positive");

    MCTSConfig search_config;
    search_config.simulations = config.simulations;
    search_config.dirichlet_alpha = config.dirichlet_alpha;
    search_config.dirichlet_epsilon = config.dirichlet_epsilon;

    // SooSelfPlayRunner's schedule exactly: full temperature for the first
    // ``temperature_moves`` moves, greedy after.
    const auto temperature_for = [&config](int move_count) {
        return move_count < config.temperature_moves ? config.temperature : 0.0;
    };
    // Repetition wins over lateness when both are configured: it is the more
    // specific signal, and the audit says it is the right one.
    const auto simulations_for = [&config](const EpisodeLane& lane, uint64_t key,
                                           int move_count) {
        if (config.simulations_late <= 0) return config.simulations;
        if (config.repeat_window > 0) {
            return lane.seen_recently(key) ? config.simulations_late : config.simulations;
        }
        return move_count >= config.late_move_threshold ? config.simulations_late
                                                        : config.simulations;
    };

    std::vector<Episode> episodes(jobs.size());
    std::atomic<size_t> next_job{0};

    const int derived_lanes =
        config.lanes > 0 ? config.lanes : std::min<int>(static_cast<int>(jobs.size()),
                                                        2 * config.max_batch);
    const int lane_count = std::min<int>(derived_lanes, static_cast<int>(jobs.size()));

    // Seat a lane on the next unstarted job.  Returns false when the queue is
    // empty, which is how a lane retires.
    const auto seat = [&](EpisodeLane& lane) {
        for (;;) {
            const size_t index = next_job.fetch_add(1, std::memory_order_acq_rel);
            if (index >= jobs.size()) return false;
            const EpisodeJob& job = jobs[index];
            lane.job_index = index;
            lane.state = job.initial_state;
            lane.game_seed = job.seed;
            lane.move_count = 0;
            if (config.max_game_duration &&
                *config.max_game_duration > Clock::duration::zero()) {
                lane.deadline = Clock::now() + *config.max_game_duration;
            } else {
                lane.deadline.reset();
            }
            if (lane.state.status == kFinished) {
                // A job handed a terminal position produces no moves rather
                // than throwing out of a worker thread.  Take the next one.
                episodes[index].completed = true;
                episodes[index].finish_order.assign(
                    lane.state.finish_order.begin(),
                    lane.state.finish_order.begin() + lane.state.finished_count);
                continue;
            }
            lane.recent.clear();
            const uint64_t key = dynamics_key(lane.state);
            lane.session.reseed(lane.game_seed);
            lane.session.set_simulations(simulations_for(lane, key, 0));
            if (config.repeat_window > 0) {
                lane.remember(key, static_cast<size_t>(config.repeat_window));
            }
            lane.session.begin(lane.state, temperature_for(0));
            return true;
        }
    };

    std::vector<std::unique_ptr<EpisodeLane>> lanes;
    lanes.reserve(static_cast<size_t>(lane_count));
    for (int i = 0; i < lane_count; ++i) {
        auto lane = std::make_unique<EpisodeLane>(match, search_config);
        lane->done = !seat(*lane);
        lanes.push_back(std::move(lane));
    }

    Batcher batcher(config.max_batch, config.max_wait_us);
    ReadyQueue ready;
    std::mutex metrics_mutex;
    std::atomic<bool> stop{false};
    std::atomic<int> retired{0};
    // How often the boosted search budget fired.  Reported rather than
    // inferred: the trigger only earns its complexity if this stays small.
    std::atomic<uint64_t> boosted{0};

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

    const auto retire = [&](EpisodeLane& lane) {
        lane.done = true;
        if (retired.fetch_add(1, std::memory_order_acq_rel) + 1 >= lane_count) {
            stop.store(true, std::memory_order_relaxed);
            ready.stop();
            batcher.stop();
        }
    };

    for (int i = 0; i < lane_count; ++i) {
        if (lanes[static_cast<size_t>(i)]->done) {
            retire(*lanes[static_cast<size_t>(i)]);
        } else {
            ready.push(i);
        }
    }

    const auto started = Clock::now();
    std::vector<double> busy(static_cast<size_t>(config.threads), 0.0);
    std::vector<std::thread> workers;
    for (int w = 0; w < config.threads; ++w) {
        workers.emplace_back([&, w] {
          try {
            int lane_id = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                if (!ready.pop(lane_id)) break;
                const auto work_start = Clock::now();
                EpisodeLane& lane = *lanes[static_cast<size_t>(lane_id)];

                const auto abort_for_deadline = [&] {
                    Episode& episode = episodes[lane.job_index];
                    episode.move_count = lane.move_count;
                    episode.max_game_seconds_exceeded = true;
                    if (!seat(lane)) {
                        busy[static_cast<size_t>(w)] += seconds_since(work_start);
                        retire(lane);
                    } else {
                        busy[static_cast<size_t>(w)] += seconds_since(work_start);
                        ready.push(lane_id);
                    }
                };
                if (lane.deadline && Clock::now() >= *lane.deadline) {
                    abort_for_deadline();
                    continue;
                }

                const EpisodeSearch::Status status = lane.session.advance();
                if (lane.deadline && Clock::now() >= *lane.deadline) {
                    abort_for_deadline();
                    continue;
                }
                if (status == EpisodeSearch::Status::NeedsEvaluation) {
                    BatchItem item{&lane.session.pending_state(),
                                   &lane.session.pending_features(),
                                   &lane.session.pending_actions(), 0, &lane.outcome,
                                   lane.session.value_width()};
                    batch_evaluator.prepare(item);
                    busy[static_cast<size_t>(w)] += seconds_since(work_start);
                    batcher.submit(lane_id);
                    continue;
                }

                // A move is ready.  Record it *before* applying it: a training
                // sample is the position that was searched, paired with the
                // visit distribution that search produced.
                Episode& episode = episodes[lane.job_index];
                EpisodeMove move;
                // root_features(), not pending_features(): see the comment on
                // SearchSession::root_features.
                move.features = lane.session.root_features();
                move.root_actions = lane.session.root_actions();
                move.visit_counts = lane.session.visit_counts();
                move.selected_action = lane.session.selected_action();
                episode.moves.push_back(std::move(move));

                lane.state = apply_action(
                    lane.state, match,
                    to_physical_action(move.selected_action, match, lane.state.current_player));
                ++lane.move_count;

                const bool finished = lane.state.status == kFinished;
                const bool out_of_moves = lane.move_count >= config.max_moves;
                if (finished || out_of_moves) {
                    episode.move_count = lane.move_count;
                    if (finished) {
                        episode.completed = true;
                        episode.finish_order.assign(
                            lane.state.finish_order.begin(),
                            lane.state.finish_order.begin() + lane.state.finished_count);
                    } else {
                        episode.move_limit_exceeded = true;
                    }
                    // The lane does not retire with its game: it takes the next
                    // queued job, so a straggler costs its own lane and no
                    // other.
                    if (!seat(lane)) {
                        busy[static_cast<size_t>(w)] += seconds_since(work_start);
                        retire(lane);
                        continue;
                    }
                    busy[static_cast<size_t>(w)] += seconds_since(work_start);
                    ready.push(lane_id);
                    continue;
                }

                const uint64_t key = dynamics_key(lane.state);
                const int budget = simulations_for(lane, key, lane.move_count);
                if (budget > config.simulations) {
                    boosted.fetch_add(1, std::memory_order_relaxed);
                }
                if (config.repeat_window > 0) {
                    lane.remember(key, static_cast<size_t>(config.repeat_window));
                }
                lane.session.reseed(lane.game_seed + static_cast<uint64_t>(lane.move_count));
                lane.session.set_simulations(budget);
                lane.session.begin(lane.state, temperature_for(lane.move_count));
                busy[static_cast<size_t>(w)] += seconds_since(work_start);
                ready.push(lane_id);
            }
          } catch (...) {
            fail(std::current_exception());
          }
        });
    }

    std::thread evaluator([&] {
      try {
        std::vector<int> batch;
        std::vector<BatchItem> items;
        while (batcher.collect(batch)) {
            items.clear();
            items.reserve(batch.size());
            for (const int lane_id : batch) {
                EpisodeLane& lane = *lanes[static_cast<size_t>(lane_id)];
                items.push_back(BatchItem{&lane.session.pending_state(),
                                          &lane.session.pending_features(),
                                          &lane.session.pending_actions(), 0, &lane.outcome,
                                          lane.session.value_width()});
            }
            const auto eval_start = Clock::now();
            batch_evaluator.evaluate(items);
            const double eval_seconds = seconds_since(eval_start);
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                metrics.batch_sizes.push_back(static_cast<uint32_t>(batch.size()));
                metrics.evaluations += batch.size();
                ++metrics.batches;
                metrics.evaluator_seconds += eval_seconds;
            }
            for (const int lane_id : batch) {
                EpisodeLane& lane = *lanes[static_cast<size_t>(lane_id)];
                const double* values = lane.session.value_width() == 1
                                           ? &lane.outcome.value
                                           : lane.outcome.values.data();
                lane.session.supply(lane.outcome.priors, values);
            }
            ready.push_many(batch);
        }
      } catch (...) {
        fail(std::current_exception());
      }
    });

    for (std::thread& worker : workers) worker.join();
    stop.store(true, std::memory_order_relaxed);
    ready.stop();
    batcher.stop();
    evaluator.join();

    metrics.wall_seconds = seconds_since(started);
    for (const double value : busy) metrics.worker_busy_seconds += value;
    if (failure) std::rethrow_exception(failure);

    for (const Episode& episode : episodes) {
        metrics.moves += static_cast<uint64_t>(episode.move_count);
    }
    metrics.boosted_moves = boosted.load();
    return episodes;
}

}  // namespace soo
