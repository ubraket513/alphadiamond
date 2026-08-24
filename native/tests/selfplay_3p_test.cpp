// The Min episode path must preserve the three-seat feature/value contract
// while remaining reproducible from a fixed job list.
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "check.hpp"
#include "golden.hpp"
#include "soo/board.hpp"
#include "soo/selfplay.hpp"

namespace {

soo::EpisodeConfig episode_config() {
    soo::EpisodeConfig config;
    config.lanes = 2;
    config.threads = 2;
    config.max_batch = 4;
    config.max_wait_us = 200;
    config.simulations = 4;
    config.max_moves = 24;
    config.temperature = 0.0;
    config.temperature_moves = 0;
    config.dirichlet_epsilon = 0.0;
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 2, "usage: selfplay_3p_test <golden-dir>");
    const std::string golden_dir = argv[1];
    REQUIRE(soo::load_topology_from_dir(golden_dir + "/topology"),
            "could not load the golden topology tables");

    soo_test::Golden golden;
    std::string error;
    REQUIRE(soo_test::load_golden(golden_dir + "/rules-v1.txt", golden, error), error.c_str());
    const soo::Match& match = golden.match(3);
    REQUIRE(match.count == 3, "golden file is missing the 3P match line");

    const soo::State* opening = nullptr;
    for (const soo_test::GoldenCase& entry : golden.cases) {
        if (entry.tag == "opening" && entry.player_count == 3) {
            opening = &entry.state;
            break;
        }
    }
    REQUIRE(opening != nullptr, "golden file has no 3P opening position");

    std::vector<soo::EpisodeJob> jobs;
    for (uint64_t seed : {7ULL, 11ULL, 13ULL}) jobs.push_back({*opening, seed});

    soo::DummyBatchEvaluator evaluator(0.0);
    soo::EpisodeMetrics first_metrics;
    soo::EpisodeMetrics second_metrics;
    const auto first = soo::run_episodes(match, jobs, episode_config(), evaluator, first_metrics);
    const auto second = soo::run_episodes(match, jobs, episode_config(), evaluator, second_metrics);

    CHECK_EQ(first.size(), jobs.size());
    CHECK_EQ(second.size(), jobs.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
        const std::string where = "3P episode " + std::to_string(index) + ": ";
        const soo::Episode& a = first[index];
        const soo::Episode& b = second[index];
        CHECK(!a.moves.empty());
        CHECK_EQ(a.moves.size(), b.moves.size());
        CHECK_EQ(a.finish_order, b.finish_order);
        for (std::size_t move = 0; move < a.moves.size(); ++move) {
            if (a.moves[move].features.feature_count != 6) {
                soo_test::fail(__FILE__, __LINE__, where + "recorded non-Min feature width");
            }
            if (a.moves[move].features.canonical_player_ids.size() != 3U) {
                soo_test::fail(__FILE__, __LINE__, where + "recorded non-Min player order");
            }
            if (a.moves[move].selected_action != b.moves[move].selected_action) {
                soo_test::fail(__FILE__, __LINE__, where + "selected action is not reproducible");
            }
        }
    }

    std::fprintf(stderr, "3P episodes=%zu moves=%llu\n", first.size(),
                 static_cast<unsigned long long>(first_metrics.moves));
    return soo_test::report("selfplay_3p_test");
}
