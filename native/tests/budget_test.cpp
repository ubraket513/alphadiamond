// The wall-clock budget, which is the one search bound that is not a count.
//
// This was a Python test driving the extension. It is C++ behaviour --
// MCTS2P::set_budget and MCTS3P::set_budget decide it, and nothing about the
// bridge is involved -- so it belongs where the rest of the search's contract
// is: in CTest, with no interpreter.
//
// Three properties, and the first is the one that is easy to get wrong. A
// search that comes back empty-handed because its budget was already spent
// hands its caller no move at all, which is worse than a shallow move.
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "check.hpp"
#include "golden.hpp"
#include "soo/board.hpp"
#include "soo/evaluator.hpp"
#include "soo/mcts.hpp"
#include "soo/mcts3p.hpp"
#include "soo/state.hpp"
#include "vector_evaluator.hpp"

namespace {

soo::MCTSConfig config(int simulations) {
    soo::MCTSConfig out;
    out.simulations = simulations;
    out.c_puct = 1.5;
    out.dirichlet_epsilon = 0.0;
    out.seed = 0;
    return out;
}

uint32_t total_visits(const std::vector<uint32_t>& visits) {
    uint32_t sum = 0;
    for (const uint32_t value : visits) sum += value;
    return sum;
}

// The budget lives on SearchSession, not on the MCTS2P convenience wrapper,
// so the search is driven the way the self-play pool drives it.
const soo::SearchResult& run_bounded(soo::SearchSession& session, soo::Evaluator& evaluator,
                                     const soo::State& state, double budget) {
    session.set_budget(budget);
    session.begin(state, 0.0, true);
    while (session.advance() == soo::SearchSession::Status::NeedsEvaluation) {
        session.supply(evaluator.evaluate(session.pending_features(), session.pending_actions()));
    }
    return session.result();
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 3, "usage: budget_test <golden-dir> <topology-dir>");
    REQUIRE(soo::load_topology_from_dir(std::string(argv[2])),
            "could not load the golden topology tables");

    soo_test::Golden golden;
    std::string error;
    REQUIRE(soo_test::load_golden(std::string(argv[1]) + "/rules-v1.txt", golden, error),
            error.c_str());

    const soo::Match& match = golden.match(2);
    REQUIRE(match.count == 2, "golden file is missing the 2P match line");
    const soo::State* opening = nullptr;
    for (const soo_test::GoldenCase& entry : golden.cases) {
        if (entry.player_count == 2 && entry.tag == "opening") {
            opening = &entry.state;
            break;
        }
    }
    REQUIRE(opening != nullptr, "golden file has no 2P opening position");

    soo::DeterministicEvaluator evaluator;
    constexpr int kSimulations = 4096;
    // Large enough that the budget is what stops it: 4,096 simulations finish
    // inside 50 ms on this evaluator, so a 50 ms budget on 4,096 proves nothing
    // -- the first version of this test passed for that reason and would have
    // gone on passing with the budget ignored entirely.
    constexpr int kUnfinishable = 400000;

    // No budget is not a spent budget. Zero means unlimited, and the two must
    // never collide: the search runs every simulation it was asked for.
    {
        soo::SearchSession session(match, config(64));
        const soo::SearchResult& result = run_bounded(session, evaluator, *opening, 0.0);
        CHECK_EQ(static_cast<int>(total_visits(result.visit_counts)), 64);
    }

    // An already-spent budget still returns a usable move. The first simulation
    // always runs, so there is a visit distribution and a selected action in it.
    {
        soo::SearchSession session(match, config(kSimulations));
        const soo::SearchResult& result = run_bounded(session, evaluator, *opening, 1e-9);
        CHECK(total_visits(result.visit_counts) >= 1);
        bool selected_was_visited = false;
        for (std::size_t i = 0; i < result.root_actions.size(); ++i) {
            if (result.root_actions[i] == result.selected_action) {
                selected_was_visited = result.visit_counts[i] >= 1;
                break;
            }
        }
        CHECK(selected_was_visited);
        // And it stopped: a full 4096-simulation search would not be this short.
        CHECK(result.simulations_run < static_cast<uint32_t>(kSimulations));
    }

    // A live budget cuts the search short rather than being advisory.
    {
        soo::SearchSession session(match, config(kUnfinishable));
        const auto started = std::chrono::steady_clock::now();
        const soo::SearchResult& result = run_bounded(session, evaluator, *opening, 0.05);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        CHECK(elapsed < 5.0);
        CHECK(result.simulations_run < static_cast<uint32_t>(kUnfinishable));
        CHECK(total_visits(result.visit_counts) >= 1);
    }

    // The three-seat search carries the same contract; it is a separate tree.
    {
        const soo::Match& match3 = golden.match(3);
        REQUIRE(match3.count == 3, "golden file is missing the 3P match line");
        const soo::State* opening3 = nullptr;
        for (const soo_test::GoldenCase& entry : golden.cases) {
            if (entry.player_count == 3 && entry.tag == "opening") {
                opening3 = &entry.state;
                break;
            }
        }
        REQUIRE(opening3 != nullptr, "golden file has no 3P opening position");

        soo::SearchSession3P session(match3, config(kSimulations));
        session.set_budget(1e-9);
        session.begin(*opening3, 0.0);
        while (session.advance() == soo::SearchSession3P::Status::NeedsEvaluation) {
            session.supply(
                soo_test::evaluate_vector(session.pending_features(), session.pending_actions()));
        }
        const soo::SearchResult3P& result = session.result();
        CHECK(total_visits(result.visit_counts) >= 1);
        CHECK(result.simulations_run < static_cast<uint32_t>(kSimulations));
    }

    return soo_test::report("budget_test");
}
