// Gate C.1: per-stage native cost for one search, no threads and no Python.
//
// Stages mirror the Python worker-side profile in
// docs/native_selfplay_phase0.md section 0.1 so the two tables can be read
// side by side. Measured with a dummy evaluator, because the point is the cost
// of the *lane* work, not of inference.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "soo/mcts.hpp"
#include "soo/state.hpp"

namespace soo {

struct StageTiming {
    uint64_t legal_ns = 0;      // canonical legal action generation
    uint64_t prior_ns = 0;      // vacancy bootstrap prior
    uint64_t encode_ns = 0;     // canonical encoder
    uint64_t apply_ns = 0;      // state + apply_action
    uint64_t search_ns = 0;     // whole search, wall clock
    uint64_t evaluations = 0;
    uint64_t searches = 0;
    uint64_t nodes = 0;
};

// Time each stage in isolation over a corpus of states, at the call counts a
// real search produces. Isolation is deliberate: attributing time inside a
// running search needs instrumentation that costs more than some of the stages.
StageTiming profile_stages(const std::vector<State>& states, const Match& match,
                           int repeats);

// Whole-search cost with an inline dummy evaluator.
StageTiming profile_searches(const std::vector<State>& states, const Match& match,
                             const MCTSConfig& config, int repeats);

}  // namespace soo
