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

#include "soo/encoder.hpp"
#include "soo/evaluator.hpp"
#include "soo/mcts.hpp"
#include "soo/state.hpp"

namespace soo {

// One lane's pending request, as the evaluator sees it.
//
// Pointers into lane storage, not copies: the lane owns the encoded features
// and the legal set, and both stay alive for the whole dispatch.
struct BatchItem {
    const State* state = nullptr;
    const Encoded* encoded = nullptr;
    const std::vector<int32_t>* actions = nullptr;
    uint64_t salt = 0;
    EvalOutcome* outcome = nullptr;   // to be filled
};

// Whatever answers a batch. The dummy is native and sleeps; the Gate D one
// crosses into Python. Called only on the evaluator thread, one batch at a
// time, so implementations need no locking of their own.
class BatchEvaluator {
  public:
    virtual ~BatchEvaluator() = default;

    // Called on a SEARCH WORKER, before the lane is handed to the batcher.
    //
    // Anything an evaluator can compute from the request alone belongs here,
    // not in evaluate(). The workers sit at a few percent utilisation while the
    // single evaluator thread is the serial resource that must never be
    // starved, so work left in evaluate() is charged to the scarce thread. The
    // native vacancy prior is 7.5 us/eval -- 240 us per batch of 32 on the
    // critical path if computed there.
    virtual void prepare(BatchItem& item) { (void)item; }

    // Called on the evaluator thread, one batch at a time.
    virtual void evaluate(std::vector<BatchItem>& batch) = 0;
};

// Gate C's evaluator: request-dependent, per-lane salted, artificial latency.
class DummyBatchEvaluator : public BatchEvaluator {
  public:
    explicit DummyBatchEvaluator(double latency_ms) : latency_ms_(latency_ms) {}
    void evaluate(std::vector<BatchItem>& batch) override;

  private:
    double latency_ms_;
};

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

    // Exploration.  All zero reproduces the deterministic Gate C behaviour
    // exactly, including drawing nothing, so every measurement taken before
    // stochastic MCTS existed remains reproducible.
    double temperature = 0.0;       // move sampling, for the first N moves
    int temperature_moves = 0;      // SelfPlayConfig.temperature_moves
    double dirichlet_alpha = 0.3;
    double dirichlet_epsilon = 0.0;
    uint64_t seed = 0;              // base seed; each lane derives its own
};

// --------------------------------------------------------------------------
// Episode production: the same scheduler, driven to produce training data.
//
// run_scheduler is a measurement harness -- it runs for a wall-clock window and
// recycles a lane the moment its game ends.  Training needs the opposite: a
// fixed set of games, each played to its own terminal state, with every move
// recorded.  Same batching architecture, different stop condition and an
// output that is samples rather than timings.
// --------------------------------------------------------------------------

struct EpisodeJob {
    State initial_state;
    uint64_t seed = 0;      // Python's derive_game_seed for this attempt
};

// One recorded move.  The features are the *request* the evaluator answered, so
// a sample is exactly what the network saw, not a re-encoding of it.
struct EpisodeMove {
    Encoded features;
    std::vector<int32_t> root_actions;    // canonical ids, authoritative order
    std::vector<uint32_t> visit_counts;   // aligned; the policy target
    int32_t selected_action = 0;          // canonical
};

struct Episode {
    std::vector<EpisodeMove> moves;
    std::vector<uint8_t> finish_order;
    int move_count = 0;
    bool completed = false;
    // Set when the game hit max_moves instead of a terminal state.  Python's
    // runner returns zero samples for an aborted episode; the moves are kept
    // here anyway so the caller can decide, and the caller does discard them.
    bool move_limit_exceeded = false;
};

struct EpisodeConfig {
    // Concurrent games.  0 means "derive it": min(jobs, 2 * max_batch), which
    // is pitfall 7.6's rule.
    //
    // Jobs beyond this many are *queued*, not run in parallel, and a lane takes
    // the next unstarted job the moment its own game ends.  That is not a
    // detail -- with one lane per job, batch occupancy collapses to 1.0 for the
    // second half of the run while a handful of long games straggle (measured:
    // 128 games, lengths 63 to 402, mean batch 64.0 over the first tenth of
    // dispatches and 1.0 over the last half).  The Python pool learned the same
    // thing and calls it Finding 3, worth 35 % samples/hour.
    int lanes = 0;
    int threads = 4;
    int max_batch = 32;
    int max_wait_us = 2000;
    int simulations = 64;
    int max_moves = 2000;
    double temperature = 1.0;
    int temperature_moves = 20;
    double dirichlet_alpha = 0.3;
    double dirichlet_epsilon = 0.25;
};

struct EpisodeMetrics {
    uint64_t evaluations = 0;
    uint64_t batches = 0;
    uint64_t moves = 0;
    double wall_seconds = 0.0;
    double evaluator_seconds = 0.0;
    double worker_busy_seconds = 0.0;
    std::vector<uint32_t> batch_sizes;
};

std::vector<Episode> run_episodes(const Match& match, const std::vector<EpisodeJob>& jobs,
                                  const EpisodeConfig& config, BatchEvaluator& batch_evaluator,
                                  EpisodeMetrics& metrics);

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
SchedulerMetrics run_scheduler(const Match& match, const State& opening,
                               const SchedulerConfig& config, BatchEvaluator& evaluator);

}  // namespace soo
