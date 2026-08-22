// Scalar two-player PUCT, suspendable at its evaluation points.
// Python authority: diamond.alphazero.mcts.search_2p.MCTS2P.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "soo/encoder.hpp"
#include "soo/evaluator.hpp"
#include "soo/random.hpp"
#include "soo/state.hpp"
#include "soo/tree.hpp"

namespace soo {

struct MCTSConfig {
    int simulations = 200;
    double c_puct = 1.5;
    double dirichlet_alpha = 0.3;
    double dirichlet_epsilon = 0.0;
    uint64_t seed = 0;
};

// One expansion, as observed from outside the search.  Comparing this sequence
// is what proves the two backends walked the same tree in the same order --
// matching root visit counts alone can hide a divergent traversal.
struct EvalRecord {
    uint64_t request_hash = 0;
    std::vector<int32_t> legal_actions;
};

struct SearchResult {
    int32_t selected_action = 0;
    std::vector<int32_t> root_actions;    // in expansion order, not sorted
    std::vector<uint32_t> visit_counts;   // aligned with root_actions
    std::vector<double> q_values;         // aligned with root_actions
    std::vector<double> policy;           // aligned with root_actions
    // The root edge priors *after* any Dirichlet mixing, aligned with
    // root_actions.  Exposed because the mixture is otherwise unobservable
    // from outside, and an unobservable stochastic path cannot be gated.
    std::vector<double> root_priors;
    uint32_t simulations_run = 0;
    uint32_t evaluator_calls = 0;
    uint32_t nodes_created = 0;
    std::vector<EvalRecord> trace;        // populated only when tracing
};

// One search, driven from outside instead of driving an evaluator itself.
//
// The search suspends at exactly the points it needs a value, and the caller
// decides what happens next.  That is the whole reason this is not a plain
// loop: a thread that blocks on its own evaluation cannot pick up another
// game, so lanes would be pinned to threads and the achievable batch size
// would be capped by the thread count.  Suspending instead lets one worker
// hold many logical games in flight.
//
// Semantics are unchanged from the synchronous version, and Gate B is the
// proof: the same bit-identical q values and the same evaluator request
// sequence.
class SearchSession {
  public:
    enum class Status : uint8_t {
        NeedsEvaluation,  // pending_features()/pending_actions() await supply()
        Ready,            // result() is final
    };

    SearchSession(const Match& match, const MCTSConfig& config);

    // ``temperature`` and ``config.dirichlet_epsilon`` may both be positive.
    // With both at 0 the search draws nothing and Gate B's bit-exact parity
    // with Python still holds; that is asserted, not assumed.
    void begin(const State& state, double temperature, bool trace);

    // Reseed the sampling stream.  Python builds a fresh ``MCTS2P`` per move
    // with ``seed = selfplay.seed + move_count``; a native lane reuses one
    // session across moves, so the per-move reseed has to be explicit.  Without
    // it every move of a lane would continue one stream -- still deterministic,
    // but no longer reproducible from a move index alone.
    void reseed(uint64_t seed) { rng_.reseed(seed); }

    // Change the search budget between moves.
    //
    // 64 simulations is this project's reference point, not a measured optimum,
    // and it is not enough everywhere: measured on the A0 actor, doubling to 128
    // takes the discarded-move fraction from 28.6 % to 12.2 %, while 256 adds
    // nothing (12.5 %).  The cost is not uniform though -- most games finish
    // comfortably at 64, and the failures are a short-cycle attractor in a small
    // tail.  Paying for 128 everywhere is therefore mostly waste; paying for it
    // only where a game is already going wrong is not.  Must be called before
    // ``begin``.
    void set_simulations(int simulations) {
        if (simulations <= 0) throw std::invalid_argument("simulations must be positive");
        config_.simulations = simulations;
    }

    Status advance();
    const Encoded& pending_features() const { return pending_encoded_; }
    const std::vector<int32_t>& pending_actions() const { return pending_actions_; }
    // The state of the node awaiting evaluation -- NOT the search root. An
    // evaluator that derives anything from position (the vacancy prior does)
    // must use this one.
    const State& pending_state() const { return pending_state_; }

    // The encoding of the search ROOT -- NOT the pending node.  This is the
    // mirror image of the trap above, and it bites whoever records training
    // data: a sample is the position that was searched, paired with the visit
    // distribution that search produced.  ``pending_features()`` after a
    // completed search is whichever leaf happened to be expanded last, so
    // recording it silently mislabels every sample's player-to-act -- and the
    // value target is derived from that, so half the labels come out inverted.
    // Caught by the Gate F episode comparison; nothing cheaper found it.
    const Encoded& root_features() const { return root_encoded_; }
    void supply(const EvalOutcome& outcome) { pending_outcome_ = outcome; }

    const SearchResult& result() const { return result_; }
    const State& root_state() const { return root_state_; }

  private:
    enum class Phase : uint8_t { Root, AwaitRoot, Descend, AwaitLeaf, Done };

    void prepare_expansion(uint32_t node_index);
    double complete_expansion(uint32_t node_index);
    void backup(double value);
    void apply_dirichlet_noise(uint32_t node_index);
    void finalize();
    uint32_t select(uint32_t node_index) const;

    const Match& match_;
    MCTSConfig config_;
    Arena arena_;
    SearchResult result_;

    Phase phase_ = Phase::Done;
    bool trace_ = false;
    double temperature_ = 0.0;
    Rng rng_;
    // Scratch for the Dirichlet draws and the temperature weights.  Both are
    // per-search and short; reusing one buffer keeps them off the allocator on
    // the search worker, which the scheduler cares about.
    std::vector<double> noise_;
    int simulation_ = 0;
    uint32_t root_ = 0;
    uint32_t node_ = 0;
    State root_state_;
    // (owning node, edge) pairs: backup needs the node to keep its cached
    // visit aggregate in step with the edge it increments.
    std::vector<std::pair<uint32_t, uint32_t>> path_;

    State pending_state_;
    Encoded pending_encoded_;
    Encoded root_encoded_;
    std::vector<int32_t> pending_actions_;
    EvalOutcome pending_outcome_;
};

// Synchronous driver: a SearchSession plus an evaluator it calls inline.
// This is the Gate B shape, and it stays the reference for the scheduler.
class MCTS2P {
  public:
    MCTS2P(const Match& match, Evaluator& evaluator, const MCTSConfig& config);

    SearchResult run(const State& state, double temperature, bool trace);

  private:
    Evaluator& evaluator_;
    SearchSession session_;
};

}  // namespace soo
