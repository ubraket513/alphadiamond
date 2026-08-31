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
#include <set>
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

    // Playing games together must not change them.
    //
    // The promotion arena used to run one game per scheduler call, which forced
    // batch = 1 and made the stage the cost of an iteration. It now plays the
    // games of an opening block that share a turn order in one call. That is
    // only sound if a game's result does not depend on which games were in
    // flight beside it, so the claim is asserted rather than argued: the same
    // jobs, played strictly one at a time, must reproduce the grouped run move
    // for move.
    auto solitary = episodes;
    solitary.lanes = 1;
    solitary.threads = 1;
    solitary.max_batch = 1;
    for (std::size_t index = 0; index < jobs.size(); ++index) {
        soo::EpisodeMetrics alone_metrics;
        const auto alone =
            soo::run_episodes(match, {jobs[index]}, solitary, evaluator, alone_metrics);
        REQUIRE(alone.size() == 1, "solitary run did not return one episode");
        const std::string where = "solitary episode " + std::to_string(index) + ": ";
        if (alone[0].move_count != first[index].move_count ||
            alone[0].completed != first[index].completed) {
            soo_test::fail(__FILE__, __LINE__, where + "grouping changed the episode shape");
            continue;
        }
        for (std::size_t move = 0; move < alone[0].moves.size(); ++move) {
            if (alone[0].moves[move].selected_action != first[index].moves[move].selected_action) {
                soo_test::fail(__FILE__, __LINE__,
                               where + "grouping changed move " + std::to_string(move));
                break;
            }
        }
    }

    // A batch item has to say which game it came from.
    //
    // The arena router sends the candidate's turns to one model and everyone
    // else's to another, and the candidate holds a different seat in every game
    // of a block -- so with several games batched together, routing needs the
    // job. It cannot be recovered from the lane or the outcome pointer: a lane
    // takes the next unstarted job when its own game ends, which is exactly
    // what three jobs on two lanes exercises here. A `job` wired from the lane
    // id would report two distinct values for three games.
    class JobObserver final : public soo::BatchEvaluator {
      public:
        JobObserver(soo::BatchEvaluator& inner, std::size_t jobs) : inner_(inner), jobs_(jobs) {}
        void evaluate(std::vector<soo::BatchItem>& batch) override {
            for (const auto& item : batch) {
                REQUIRE(item.job >= 0 && static_cast<std::size_t>(item.job) < jobs_,
                        "batch item reported an unknown job");
                seen.insert(item.job);
            }
            inner_.evaluate(batch);
        }
        std::set<int> seen;

      private:
        soo::BatchEvaluator& inner_;
        std::size_t jobs_;
    };
    JobObserver observer(evaluator, jobs.size());
    soo::EpisodeMetrics observed_metrics;
    const auto observed = soo::run_episodes(match, jobs, episodes, observer, observed_metrics);
    CHECK_EQ(observed.size(), jobs.size());
    CHECK_EQ(observer.seen.size(), jobs.size());

    std::fprintf(stderr, "scheduler moves=%llu episodes=%zu\n",
                 static_cast<unsigned long long>(parallel.moves), first.size());
    return soo_test::report("selfplay_test");
}
