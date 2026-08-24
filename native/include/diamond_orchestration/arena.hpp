#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "diamond_orchestration/config.hpp"

namespace diamond_orchestration {

struct ArenaMatchup2 final {
    std::array<int, 2> turn_order;
    int candidate_player = 0;
};

struct ArenaMatchup3 final {
    std::array<int, 3> turn_order;
    int candidate_player = 0;
};

std::vector<ArenaMatchup2> balanced_soo_matchups();
std::vector<ArenaMatchup3> balanced_min_matchups();
void validate_soo_arena_games(const ArenaConfig& config);
void validate_min_arena_games(const ArenaConfig& config);

struct SooArenaResult final {
    int64_t wins = 0;
    int64_t losses = 0;
    int64_t aborted_games = 0;
    double win_rate = 0.0;
    bool promoted = false;
};

struct MinArenaResult final {
    int64_t first_places = 0;
    int64_t second_places = 0;
    int64_t third_places = 0;
    int64_t aborted_games = 0;
    double mean_utility = 0.0;
    bool promoted = false;
};

// A null result denotes a game stopped at its move budget.  Completed Soo
// outcomes are whether the candidate won; completed Min outcomes are the
// candidate's zero-based final placement.
SooArenaResult summarize_soo_arena(const std::vector<std::optional<bool>>& outcomes,
                                   const ArenaConfig& config);
MinArenaResult summarize_min_arena(const std::vector<std::optional<int>>& outcomes,
                                   const ArenaConfig& config);

// The executor owns gameplay/model selection; the arena owns the deterministic
// balanced schedule and accounting. A null outcome is a real aborted game.
using SooArenaExecutor = std::function<std::optional<bool>(const ArenaMatchup2&, std::size_t)>;
using MinArenaExecutor = std::function<std::optional<int>(const ArenaMatchup3&, std::size_t)>;
std::vector<std::optional<bool>> execute_soo_arena_games(const ArenaConfig& config,
                                                          const SooArenaExecutor& executor);
std::vector<std::optional<int>> execute_min_arena_games(const ArenaConfig& config,
                                                        const MinArenaExecutor& executor);

}  // namespace diamond_orchestration
