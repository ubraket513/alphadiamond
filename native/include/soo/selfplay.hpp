// Many logical games over a fixed search-worker pool, plus one global batcher.
//
// The architectural criterion this exists to satisfy: a CPU worker must never
// sleep alongside a game that is waiting on the evaluator. Synchronous MCTS
// allows one outstanding request per lane, so if a worker blocked on its own
// evaluation the achievable batch size would be capped by the thread count and
// "many logical games" would buy nothing. Workers therefore hand a suspended
// lane to the batcher and immediately look for another runnable one.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "soo/mcts.hpp"
#include "soo/state.hpp"

namespace soo {

struct SchedulerConfig {
    int games = 64;             // logical games in flight
    int threads = 4;            // search workers
    int max_batch = 32;
    int max_wait_us = 2000;
    int simulations = 64;
    int max_moves = 400;
    double eval_latency_ms = 0.0;   // artificial per-batch evaluator cost
    double seconds = 5.0;           // measurement window
    bool trace_moves = false;       // record each lane's move sequence
    int stop_after_moves = 0;       // per lane; 0 = run for the full window
};

struct SchedulerMetrics {
    uint64_t evaluations = 0;
    uint64_t batches = 0;
    uint64_t moves = 0;
    uint64_t games_finished = 0;
    uint64_t batcher_wakeups = 0;
    double wall_seconds = 0.0;
    double worker_busy_seconds = 0.0;   // summed across workers
    double evaluator_seconds = 0.0;

    std::vector<uint32_t> batch_sizes;
    std::vector<uint32_t> ready_depth;   // runnable games, sampled per dispatch
    std::vector<uint32_t> waiting;       // games waiting on the evaluator
    std::vector<uint64_t> wait_ns;       // per-lane submit -> result latency

    // Per-lane move sequences, when config.trace_moves is set.
    //
    // A lane's evaluations depend only on its own request and its own salt, so
    // its trajectory must not depend on how many workers ran it, how batches
    // happened to form, or in what order lanes were scheduled. Comparing these
    // across thread counts is how the scheduler is proven free of cross-lane
    // contamination.
    std::vector<std::vector<int32_t>> lane_moves;
};

// Runs the scheduler for config.seconds and returns what it measured.
// Native only: no Python, no GIL, no PyTorch.
SchedulerMetrics run_scheduler(const Match& match, const State& opening,
                               const SchedulerConfig& config);

}  // namespace soo
