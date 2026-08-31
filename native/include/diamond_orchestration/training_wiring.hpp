#pragma once

#include <cstddef>

#include "diamond_orchestration/config.hpp"
#include "soo/selfplay.hpp"

namespace diamond_orchestration {

struct TrainingIterationWiring final {
    soo::EpisodeConfig selfplay;
    std::size_t games_per_iteration = 0;
    std::size_t training_batch_size = 0;
    std::size_t training_steps = 0;
};

TrainingIterationWiring wire_training_iteration(const ProductionConfig& config);

// The arena's episode config for a group of `lanes` games played together.
//
// Separate from the iteration wiring because its lane count is not a setting:
// it is the size of the group of games sharing a turn order that the promotion
// arena is about to play, and the search threads and batch are bounded by it.
soo::EpisodeConfig wire_arena_episode(const ProductionConfig& config, std::size_t lanes);

} // namespace diamond_orchestration
