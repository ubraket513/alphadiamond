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

} // namespace diamond_orchestration
