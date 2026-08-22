#include "soo/profile.hpp"

#include <chrono>

#include "soo/encoder.hpp"
#include "soo/evaluator.hpp"
#include "soo/prior.hpp"
#include "soo/rules.hpp"

namespace soo {
namespace {

using Clock = std::chrono::steady_clock;

inline uint64_t elapsed_ns(const Clock::time_point& start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

}  // namespace

StageTiming profile_stages(const std::vector<State>& states, const Match& match, int repeats) {
    StageTiming timing;
    std::vector<int32_t> legal;
    std::vector<double> priors;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (const State& state : states) {
            if (state.status == kFinished) continue;

            auto start = Clock::now();
            legal.clear();
            canonical_legal_action_ids(state, match, legal);
            timing.legal_ns += elapsed_ns(start);
            if (legal.empty()) continue;

            start = Clock::now();
            const Encoded encoded = soo::encode(state, match);
            timing.encode_ns += elapsed_ns(start);

            start = Clock::now();
            vacancy_prior(legal, canonical_self_occupancy(state, match), priors);
            timing.prior_ns += elapsed_ns(start);

            // One apply per evaluation, matching the measured 1.01 calls/eval.
            start = Clock::now();
            volatile auto next = apply_action(
                state, match, to_physical_action(legal[0], match, state.current_player));
            timing.apply_ns += elapsed_ns(start);
            (void)next;

            ++timing.evaluations;
        }
    }
    return timing;
}

StageTiming profile_searches(const std::vector<State>& states, const Match& match,
                             const MCTSConfig& config, int repeats) {
    StageTiming timing;
    DeterministicEvaluator evaluator;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (const State& state : states) {
            if (state.status == kFinished) continue;
            MCTS2P search(match, evaluator, config);
            const auto start = Clock::now();
            const SearchResult result = search.run(state, 0.0, false);
            timing.search_ns += elapsed_ns(start);
            timing.evaluations += result.evaluator_calls;
            timing.nodes += result.nodes_created;
            ++timing.searches;
        }
    }
    return timing;
}

}  // namespace soo
