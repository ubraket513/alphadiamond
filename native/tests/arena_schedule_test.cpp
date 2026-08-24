#include "diamond_orchestration/arena.hpp"
#include "diamond_orchestration/schedule.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
}

int main() {
    try {
        const auto soo = diamond_orchestration::schedule_soo_pair({"opening-a", "opening-b"}, {"soo-a", "soo-b"});
        const auto min = diamond_orchestration::schedule_min_triple({"opening-a", "opening-b"}, {"min-a", "min-b", "min-c"});
        require(soo.size() == 8, "Soo must schedule four assignments per opening");
        require(min.size() == 72, "Min must schedule 36 assignments per opening");
        diamond_orchestration::validate_soo_rated_batch(soo, {"opening-a", "opening-b"});
        diamond_orchestration::validate_min_rated_batch(min, {"opening-a", "opening-b"});

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
