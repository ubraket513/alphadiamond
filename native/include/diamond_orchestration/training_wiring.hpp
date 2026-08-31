#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "diamond_orchestration/config.hpp"
#include "soo/selfplay.hpp"

namespace diamond_orchestration {

struct TrainingIterationWiring final {
    soo::EpisodeConfig selfplay;
    std::size_t games_per_iteration = 0;
    std::size_t training_batch_size = 0;
    std::size_t training_steps = 0;
};

struct EvaluationPipelineWiring final {
    bool run_arena = true;
    bool record_rating = true;
    bool activate_candidate = false;
};

TrainingIterationWiring wire_training_iteration(const ProductionConfig& config);
EvaluationPipelineWiring wire_evaluation_pipeline(const ProductionConfig& config);

// The arena's episode config for a group of `lanes` games played together.
//
// Separate from the iteration wiring because its lane count is not a setting:
// it is the size of the group of games sharing a turn order that the promotion
// arena is about to play, and the search threads and batch are bounded by it.
soo::EpisodeConfig wire_arena_episode(const ProductionConfig& config, std::size_t lanes);

// Validate a checkpoint-boundary transition. Architecture, replay identity,
// optimizer semantics, MCTS budget, and protocol fields remain immutable;
// only measured throughput controls and the explicit bootstrap/Arena phase
// switches may change. Returns changed field names in stable audit order.
std::vector<std::string> validate_training_config_transition(const ProductionConfig& from,
                                                             const ProductionConfig& to);

} // namespace diamond_orchestration
