#include "diamond_orchestration/arena.hpp"

#include <algorithm>
#include <stdexcept>

namespace diamond_orchestration {

std::vector<ArenaMatchup2> balanced_soo_matchups() {
    std::vector<ArenaMatchup2> result;
    std::array<int, 2> order{1, 2};
    do for (int candidate : {1, 2}) result.push_back({order, candidate});
    while (std::next_permutation(order.begin(), order.end()));
    return result;
}

std::vector<ArenaMatchup3> balanced_min_matchups() {
    std::vector<ArenaMatchup3> result;
    std::array<int, 3> order{1, 2, 3};
    do for (int candidate : {1, 2, 3}) result.push_back({order, candidate});
    while (std::next_permutation(order.begin(), order.end()));
    return result;
}

void validate_soo_arena_games(const ArenaConfig& config) {
    config.validate();
    if (config.games % static_cast<int64_t>(balanced_soo_matchups().size()) != 0)
        throw std::invalid_argument("Soo arena games must be a multiple of 4");
}

void validate_min_arena_games(const ArenaConfig& config) {
    config.validate();
    if (config.games % static_cast<int64_t>(balanced_min_matchups().size()) != 0)
        throw std::invalid_argument("Min arena games must be a multiple of 18");
}

SooArenaResult summarize_soo_arena(const std::vector<std::optional<bool>>& outcomes,
                                   const ArenaConfig& config) {
    validate_soo_arena_games(config);
    if (outcomes.size() != static_cast<size_t>(config.games))
        throw std::invalid_argument("Soo arena outcome count must equal games");
    SooArenaResult result;
    for (const auto outcome : outcomes) {
        if (!outcome) ++result.aborted_games;
        else if (*outcome) ++result.wins;
        else ++result.losses;
    }
    const int64_t completed = result.wins + result.losses;
    result.win_rate = completed ? static_cast<double>(result.wins) / completed : 0.0;
    result.promoted = completed > 0 && result.win_rate >= config.promotion_threshold;
    return result;
}

MinArenaResult summarize_min_arena(const std::vector<std::optional<int>>& outcomes,
                                   const ArenaConfig& config) {
    validate_min_arena_games(config);
    if (outcomes.size() != static_cast<size_t>(config.games))
        throw std::invalid_argument("Min arena outcome count must equal games");
    MinArenaResult result;
    for (const auto outcome : outcomes) {
        if (!outcome) { ++result.aborted_games; continue; }
        switch (*outcome) {
            case 0: ++result.first_places; break;
            case 1: ++result.second_places; break;
            case 2: ++result.third_places; break;
            default: throw std::invalid_argument("Min candidate placement must be 0, 1, or 2");
        }
    }
    const int64_t completed = result.first_places + result.second_places + result.third_places;
    result.mean_utility = completed ? static_cast<double>(result.first_places - result.third_places) / completed : 0.0;
    result.promoted = completed > 0 && result.mean_utility >= config.promotion_threshold;
    return result;
}

std::vector<std::optional<bool>> execute_soo_arena_games(const ArenaConfig& config,
                                                          const SooArenaExecutor& executor) {
    validate_soo_arena_games(config);
    if (!executor) throw std::invalid_argument("Soo arena executor is required");
    const auto schedule = balanced_soo_matchups();
    std::vector<std::optional<bool>> outcomes;
    outcomes.reserve(static_cast<std::size_t>(config.games));
    for (std::size_t game = 0; game < static_cast<std::size_t>(config.games); ++game)
        outcomes.push_back(executor(schedule[game % schedule.size()], game));
    return outcomes;
}

std::vector<std::optional<int>> execute_min_arena_games(const ArenaConfig& config,
                                                        const MinArenaExecutor& executor) {
    validate_min_arena_games(config);
    if (!executor) throw std::invalid_argument("Min arena executor is required");
    const auto schedule = balanced_min_matchups();
    std::vector<std::optional<int>> outcomes;
    outcomes.reserve(static_cast<std::size_t>(config.games));
    for (std::size_t game = 0; game < static_cast<std::size_t>(config.games); ++game)
        outcomes.push_back(executor(schedule[game % schedule.size()], game));
    return outcomes;
}

}  // namespace diamond_orchestration
