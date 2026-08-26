#include "diamond_orchestration/arena.hpp"
#include "diamond_orchestration/schedule.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

template <class Function> void require_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const diamond_orchestration::ScheduleError&) {
        return;
    }
    throw std::runtime_error(message);
}
}

int main() {
    try {
        const auto soo = diamond_orchestration::schedule_soo_pair({"opening-a", "opening-b"}, {"soo-a", "soo-b"});
        const auto min = diamond_orchestration::schedule_min_triple({"opening-a", "opening-b"}, {"min-a", "min-b", "min-c"});
        require(soo.size() == 8, "Soo must schedule four assignments per opening");
        require(min.size() == 72, "Min must schedule 36 assignments per opening");
        diamond_orchestration::validate_soo_rated_batch(soo, {"opening-a", "opening-b"});
        diamond_orchestration::validate_min_rated_batch(min, {"opening-a", "opening-b"});

        const diamond_orchestration::OpeningSuiteConfig opening_config{
            .id = "arena-openings-v1", .version = 1, .seed = 7, .count = 2, .max_depth = 3};
        const auto suite = diamond_orchestration::materialize_opening_suite(opening_config);
        require(suite == diamond_orchestration::materialize_opening_suite(opening_config),
                "opening suite must be deterministic");
        require(suite.openings.size() == 2 &&
                    suite.openings[0].opening_id != suite.openings[1].opening_id,
                "opening suite IDs must be stable and distinct");
        auto corrupt_suite = suite;
        corrupt_suite.suite_sha256 = "sha256:corrupt";
        require_throws([&] { corrupt_suite.validate(); },
                       "opening suite digest mismatch must fail");

        const auto soo_blocks =
            diamond_orchestration::schedule_soo_opening_blocks(suite, {"soo-a", "soo-b"});
        const auto min_blocks =
            diamond_orchestration::schedule_min_opening_blocks(suite, {"min-a", "min-b", "min-c"});
        require(soo_blocks.size() == 2 && soo_blocks[0].matches.size() == 4,
                "Soo openings must retain complete unique blocks");
        require(min_blocks.size() == 2 && min_blocks[0].matches.size() == 36,
                "Min openings must retain complete unique blocks");

        std::vector<diamond_orchestration::SooOpeningBlockResult> soo_results;
        for (size_t block = 0; block < soo_blocks.size(); ++block) {
            diamond_orchestration::SooOpeningBlockResult result{
                .opening_id = soo_blocks[block].opening.opening_id};
            for (size_t game = 0; game < soo_blocks[block].matches.size(); ++game)
                result.results.push_back(
                    {soo_blocks[block].matches[game].match_id,
                     block == 1 && game == 0 ? std::nullopt : std::optional<bool>{block == 0}});
            soo_results.push_back(std::move(result));
        }
        const auto soo_summary =
            diamond_orchestration::summarize_soo_opening_blocks(soo_blocks, soo_results);
        require(soo_summary.complete_blocks == 1 && soo_summary.incomplete_blocks == 1 &&
                    soo_summary.completed_matches == 7 && soo_summary.aborted_matches == 1 &&
                    soo_summary.complete_block_scores == std::vector<double>{1.0},
                "Soo accounting must exclude incomplete opening blocks");
        auto duplicate_result = soo_results;
        duplicate_result[0].results[1].match_id = duplicate_result[0].results[0].match_id;
        require_throws(
            [&] {
                (void)diamond_orchestration::summarize_soo_opening_blocks(soo_blocks,
                                                                          duplicate_result);
            },
            "duplicate assignment must fail");
        auto missing_result = soo_results;
        missing_result[0].results.pop_back();
        require_throws(
            [&] {
                (void)diamond_orchestration::summarize_soo_opening_blocks(soo_blocks,
                                                                          missing_result);
            },
            "missing assignment must fail");

        const diamond_orchestration::PromotionStatisticsConfig stats_config{
            .confidence_level = 0.95, .bootstrap_replicates = 23, .seed = 19};
        const auto bootstrap =
            diamond_orchestration::bootstrap_opening_blocks(soo_summary, stats_config, 0.55);
        require(bootstrap.promoted && bootstrap.point_estimate == 1.0 &&
                    bootstrap.confidence_lower == 1.0 && bootstrap.confidence_upper == 1.0,
                "opening-block bootstrap must promote only complete block evidence");
        require(bootstrap == diamond_orchestration::bootstrap_opening_blocks(soo_summary,
                                                                             stats_config, 0.55),
                "opening-block bootstrap must be deterministic");

        std::vector<diamond_orchestration::MinOpeningBlockResult> min_results;
        for (const auto& block : min_blocks) {
            diamond_orchestration::MinOpeningBlockResult result{.opening_id =
                                                                    block.opening.opening_id};
            for (const auto& match : block.matches)
                result.results.push_back({match.match_id, 0});
            min_results.push_back(std::move(result));
        }
        const auto min_summary =
            diamond_orchestration::summarize_min_opening_blocks(min_blocks, min_results);
        require(min_summary.complete_blocks == 2 && min_summary.incomplete_blocks == 0 &&
                    min_summary.complete_block_scores == std::vector<double>{1.0, 1.0},
                "Min accounting must retain its whole three-player blocks");

        diamond_orchestration::ArenaConfig soo_config{.games = 4, .promotion_threshold = 0.5};
        const auto soo_result = diamond_orchestration::summarize_soo_arena({true, true, false, std::nullopt}, soo_config);
        require(soo_result.wins == 2 && soo_result.losses == 1 && soo_result.aborted_games == 1, "Soo arena accounting");
        require(soo_result.promoted, "Soo promotion threshold");

        diamond_orchestration::ArenaConfig min_config{.games = 18, .promotion_threshold = 0.25};
        std::vector<std::optional<int>> outcomes(18, 1);
        outcomes[0] = 0; outcomes[1] = 0; outcomes[2] = 2; outcomes[3] = std::nullopt;
        const auto min_result = diamond_orchestration::summarize_min_arena(outcomes, min_config);
        require(min_result.first_places == 2 && min_result.third_places == 1 && min_result.aborted_games == 1, "Min arena accounting");
    } catch (const std::exception& error) {
        std::cerr << "arena_schedule_test: " << error.what() << '\n'; return 1;
    }
}
