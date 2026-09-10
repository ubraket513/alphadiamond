// Vector PUCT for the three-seat game, suspendable at its evaluation points.
// Python authority: diamond.alphazero.mcts.search_3p.MCTS3P.
//
// The difference from the two-player search is not the tree, it is what a value
// *is*. Soo backs a scalar and negates it once per edge because a 2P game is
// zero-sum between the two seats. Min backs a utility vector -- one component
// per seat, placement-ordered (1, 0, -1) at a terminal -- through every
// ancestor unchanged. Nothing is negated and nothing is rotated: the vector is
// indexed by *global* player id, so a node reads its own component out of it.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "soo/encoder.hpp"
#include "soo/mcts.hpp"  // MCTSConfig
#include "soo/random.hpp"
#include "soo/state.hpp"

namespace soo {

// One seat's share of a leaf's utility, indexed by seat position in the match.
using ValueVector = std::array<double, kMaxPlayers>;

struct EvalOutcome3P {
    std::vector<double> priors;  // aligned with the request's legal actions
    ValueVector value{};         // indexed by the request's canonical player order
};

struct SearchResult3P {
    int32_t selected_action = 0;
    std::vector<int32_t> root_actions;    // expansion order, not sorted
    std::vector<uint32_t> visit_counts;   // aligned with root_actions
    std::vector<ValueVector> q_vectors;   // aligned; indexed by seat position
    std::vector<double> policy;           // aligned
    std::vector<double> root_priors;      // after any Dirichlet mixing
    uint32_t simulations_run = 0;
    uint32_t evaluator_calls = 0;
    uint32_t nodes_created = 0;
};

// Same suspend-and-resume contract as SearchSession: the caller answers the
// positions the search asks about, so one worker can hold many games in flight
// and an arena can alternate two different networks inside one game.
class SearchSession3P {
  public:
    enum class Status : uint8_t { NeedsEvaluation, Ready };

    SearchSession3P(const Match& match, const MCTSConfig& config);

    void begin(const State& state, double temperature);
    void reseed(uint64_t seed) { rng_.reseed(seed); }
    void set_simulations(int simulations) {
        if (simulations <= 0)
            throw std::invalid_argument("simulations must be positive");
        config_.simulations = simulations;
    }

    // See SearchSession::set_budget: seconds, <= 0 for unlimited, first
    // simulation always runs.
    void set_budget(double seconds) { budget_seconds_ = seconds; }

    Status advance();
    const Encoded& pending_features() const { return pending_encoded_; }
    const std::vector<int32_t>& pending_actions() const { return pending_actions_; }
    const State& pending_state() const { return pending_state_; }
    void supply(const EvalOutcome3P& outcome) { pending_outcome_ = outcome; }

    const SearchResult3P& result() const { return result_; }
    // Seat ids in match order: the index a value component belongs to.
    const std::vector<uint8_t>& player_ids() const { return player_ids_; }

  private:
    struct VectorEdge {
        int32_t action = 0;
        int32_t child = -1;
        double prior = 0.0;
        ValueVector value_sum{};
        uint32_t visits = 0;

        double q(int seat) const {
            return visits ? value_sum[static_cast<size_t>(seat)] / static_cast<double>(visits)
                          : 0.0;
        }
    };

    struct VectorNode {
        State state;
        uint32_t edge_begin = 0;
        uint16_t edge_count = 0;
        int seat = 0;          // whose component this node maximises
        bool expanded = false;
        bool terminal = false;
        uint32_t total_visits = 0;
    };

    enum class Phase : uint8_t { Root, AwaitRoot, Descend, AwaitLeaf, Done };

    int seat_of(const State& state) const;
    void prepare_expansion(uint32_t node_index);
    ValueVector complete_expansion(uint32_t node_index);
    ValueVector terminal_value(const State& state) const;
    void apply_dirichlet_noise(uint32_t node_index);
    void backup(const ValueVector& value);
    void finalize();
    uint32_t select(uint32_t node_index) const;

    const Match& match_;
    MCTSConfig config_;
    SearchResult3P result_;
    std::vector<VectorNode> nodes_;
    std::vector<VectorEdge> edges_;
    std::vector<uint8_t> player_ids_;

    Phase phase_ = Phase::Done;
    double temperature_ = 0.0;
    Rng rng_;
    double budget_seconds_ = 0.0;
    std::chrono::steady_clock::time_point started_at_{};
    std::vector<double> weights_;
    int simulation_ = 0;
    uint32_t root_ = 0;
    uint32_t node_ = 0;
    std::vector<std::pair<uint32_t, uint32_t>> path_;

    State pending_state_;
    Encoded pending_encoded_;
    std::vector<int32_t> pending_actions_;
    EvalOutcome3P pending_outcome_;
};

}  // namespace soo
