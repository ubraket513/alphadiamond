// Synchronous scalar two-player PUCT.
// Python authority: diamond.alphazero.mcts.search_2p.MCTS2P.
#pragma once

#include <cstdint>
#include <vector>

#include "soo/evaluator.hpp"
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
    uint32_t simulations_run = 0;
    uint32_t evaluator_calls = 0;
    uint32_t nodes_created = 0;
    std::vector<EvalRecord> trace;        // populated only when tracing
};

class MCTS2P {
  public:
    MCTS2P(const Match& match, Evaluator& evaluator, const MCTSConfig& config);

    // ``temperature`` and ``dirichlet_epsilon`` must both be 0 for now: the
    // stochastic path needs the RNG policy of section 9 and is not implemented.
    SearchResult run(const State& state, double temperature, bool trace);

  private:
    double expand(uint32_t node_index, SearchResult& result, bool trace);
    uint32_t select(uint32_t node_index) const;

    const Match& match_;
    Evaluator& evaluator_;
    MCTSConfig config_;
    Arena arena_;
};

}  // namespace soo
