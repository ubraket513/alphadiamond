// The scheduler's correctness claim, which is not about speed.
//
// A lane's evaluations depend only on its own request and its own salt, so its
// trajectory must not depend on how many workers ran it, how batches happened
// to form, or in what order lanes were scheduled. Comparing traced lane moves
// across thread counts is how the scheduler is proven free of cross-lane
// contamination -- and it is the property most easily lost to a shared buffer
// or a reused RNG.
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "check.hpp"
#include "golden.hpp"
#include "soo/board.hpp"
#include "soo/selfplay.hpp"

namespace {

soo::SchedulerConfig scheduler_config(int threads) {
    soo::SchedulerConfig config;
    config.games = 8;
    config.threads = threads;
    config.max_batch = 8;
    config.max_wait_us = 200;
    config.simulations = 8;
    config.max_moves = 40;
    config.seconds = 0.4;
    config.trace_moves = true;
    config.stop_after_moves = 6;
    config.seed = 12345;
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 2, "usage: selfplay_test <golden-dir>");
    const std::string golden_dir = argv[1];
    REQUIRE(soo::load_topology_from_dir(golden_dir + "/topology"),
            "could not load the golden topology tables");

    soo_test::Golden golden;
    std::string error;
    REQUIRE(soo_test::load_golden(golden_dir + "/rules-v1.txt", golden, error), error.c_str());
    const soo::Match& match = golden.match(2);
    REQUIRE(match.count == 2, "golden file is missing the 2P match line");

    const soo::State* opening = nullptr;
    for (const soo_test::GoldenCase& entry : golden.cases) {
        if (entry.tag == "opening" && entry.player_count == 2) {
            opening = &entry.state;
            break;
        }
    }
    REQUIRE(opening != nullptr, "golden file has no 2P opening position");

    soo::DummyBatchEvaluator evaluator(0.0);

    const soo::SchedulerMetrics single = soo::run_scheduler(match, *opening,
                                                            scheduler_config(1), evaluator);
    const soo::SchedulerMetrics parallel = soo::run_scheduler(match, *opening,
                                                              scheduler_config(4), evaluator);

    CHECK(single.moves > 0);
    CHECK_EQ(single.lane_moves.size(), parallel.lane_moves.size());
    // A comparison of empty trajectories proves nothing, so require content.
    std::size_t traced_moves = 0;
    for (const auto& lane : single.lane_moves) traced_moves += lane.size();
    CHECK(traced_moves >= single.lane_moves.size());
    for (std::size_t lane = 0; lane < single.lane_moves.size(); ++lane) {
        const std::string where = "lane " + std::to_string(lane) + ": ";
        if (single.lane_moves[lane] != parallel.lane_moves[lane]) {
            soo_test::fail(__FILE__, __LINE__,
                           where + "trajectory changed with the worker count");
        }
    }

    // No batch may exceed the configured cap, and none may be empty: an empty
    // dispatch is a woken evaluator thread with nothing to do.
    for (const uint32_t size : parallel.batch_sizes) {
        CHECK(size >= 1);
        CHECK(size <= 8);
    }
    CHECK_EQ(parallel.evaluations > 0, true);

    // Episode production: same jobs, same seeds, same episodes. Training data
    // that is not reproducible from a job list cannot be audited later.
    soo::EpisodeConfig episodes;
    episodes.lanes = 2;
    episodes.threads = 2;
    episodes.max_batch = 4;
    episodes.max_wait_us = 200;
    episodes.simulations = 4;
    episodes.max_moves = 12;
    episodes.temperature = 0.0;
    episodes.temperature_moves = 0;
    episodes.dirichlet_epsilon = 0.0;

    std::vector<soo::EpisodeJob> jobs;
    for (uint64_t seed : {7ULL, 11ULL, 13ULL}) jobs.push_back({*opening, seed});

    soo::EpisodeMetrics first_metrics;
    soo::EpisodeMetrics second_metrics;
    const auto first = soo::run_episodes(match, jobs, episodes, evaluator, first_metrics);
    const auto second = soo::run_episodes(match, jobs, episodes, evaluator, second_metrics);

    CHECK_EQ(first.size(), jobs.size());
    CHECK_EQ(second.size(), jobs.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
        const std::string where = "episode " + std::to_string(index) + ": ";
        const soo::Episode& a = first[index];
        const soo::Episode& b = second[index];
        if (a.move_count != b.move_count || a.completed != b.completed) {
            soo_test::fail(__FILE__, __LINE__, where + "episode shape is not reproducible");
            continue;
        }
        CHECK_EQ(a.moves.size(), static_cast<std::size_t>(a.move_count));
        for (std::size_t move = 0; move < a.moves.size(); ++move) {
            if (a.moves[move].selected_action != b.moves[move].selected_action ||
                a.moves[move].visit_counts != b.moves[move].visit_counts ||
                a.moves[move].root_actions != b.moves[move].root_actions) {
                soo_test::fail(__FILE__, __LINE__, where + "move " + std::to_string(move) +
                                                       " is not reproducible");
                break;
            }
            // The recorded features must be the request the evaluator answered
            // for THIS move -- a sample is the position that was searched.
            CHECK(!a.moves[move].features.node_features.empty());
            bool selected_is_legal = false;
            for (const int32_t action : a.moves[move].root_actions) {
                if (action == a.moves[move].selected_action) selected_is_legal = true;
            }
            CHECK(selected_is_legal);
        }
    }

    // Three jobs on two lanes makes this job wait for a lane to be reseated.
    // A preallocated, never-run Episode would also have zero moves, so require
    // the queued job to make progress before accepting the scheduler contract.
    CHECK(first[2].move_count > 0);
    CHECK_EQ(first[2].completed || first[2].move_limit_exceeded, true);

    // A positive monotonic deadline terminates an episode independently of
    // the move limit. A one-nanosecond budget expires before a worker can
    // advance the lane; this is deterministic and needs no wall-clock sleep.
    auto timed = episodes;
    timed.max_moves = 1000;
    timed.max_game_duration = std::chrono::nanoseconds(1);
    soo::EpisodeMetrics timeout_metrics;
    const auto timed_out =
        soo::run_episodes(match, {{*opening, 17}}, timed, evaluator, timeout_metrics);
    REQUIRE(timed_out.size() == 1, "deadline run did not return one episode");
    CHECK_EQ(timed_out[0].completed, false);
    CHECK_EQ(timed_out[0].max_game_seconds_exceeded, true);
    CHECK_EQ(timed_out[0].move_limit_exceeded, false);
    CHECK_EQ(timed_out[0].move_count, 0);

    auto deadline_disabled = episodes;
    deadline_disabled.max_moves = 1;
    deadline_disabled.max_game_duration = std::chrono::steady_clock::duration::zero();
    soo::EpisodeMetrics disabled_metrics;
    const auto zero_budget =
        soo::run_episodes(match, {{*opening, 23}}, deadline_disabled, evaluator, disabled_metrics);
    REQUIRE(zero_budget.size() == 1, "zero-deadline run did not return one episode");
    CHECK_EQ(zero_budget[0].max_game_seconds_exceeded, false);
    CHECK_EQ(zero_budget[0].move_limit_exceeded, true);

    // The bootstrap prior has to reach the search, not just the config.
    //
    // It previously did not: `self_play.bootstrap_prior` was validated by the
    // config and consumed by nothing, so every from-scratch run played the
    // network's own prior, finished no games, and produced no samples. The
    // symptom surfaced two stages later as "insufficient replay samples", which
    // is why the wiring is asserted on behaviour rather than on the flag alone.
    //
    // Deterministic by construction: temperature and dirichlet are both zero
    // above, so with the same evaluator, jobs and seeds the only thing that can
    // move a trajectory is the prior the search was handed.
    auto bootstrapped = episodes;
    bootstrapped.bootstrap_prior = true;
    soo::EpisodeMetrics bootstrap_metrics;
    const auto with_prior =
        soo::run_episodes(match, jobs, bootstrapped, evaluator, bootstrap_metrics);
    REQUIRE(with_prior.size() == first.size(), "bootstrap run returned a different job count");
    bool bootstrap_changed_play = false;
    for (std::size_t game = 0; game < with_prior.size(); ++game) {
        if (with_prior[game].move_count != first[game].move_count) {
            bootstrap_changed_play = true;
            break;
        }
        for (std::size_t move = 0; move < with_prior[game].moves.size(); ++move) {
            if (with_prior[game].moves[move].selected_action !=
                first[game].moves[move].selected_action) {
                bootstrap_changed_play = true;
                break;
            }
        }
        if (bootstrap_changed_play)
            break;
    }
    CHECK(bootstrap_changed_play);

    std::fprintf(stderr, "scheduler moves=%llu episodes=%zu\n",
                 static_cast<unsigned long long>(parallel.moves), first.size());
    return soo_test::report("selfplay_test");
}
