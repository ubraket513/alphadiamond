#include "diamond_orchestration/training_wiring.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

#include "diamond_orchestration/report.hpp"

namespace diamond_orchestration {
namespace {

int native_int(int64_t value, std::string_view field) {
    if (value > std::numeric_limits<int>::max()) {
        throw ConfigError(std::string(field) + " exceeds the native scheduler limit");
    }
    return static_cast<int>(value);
}

}  // namespace

TrainingIterationWiring wire_training_iteration(const ProductionConfig& config) {
    config.validate();
    return {
        .selfplay =
            {
                .lanes = native_int(config.workers.logical_lanes, "workers.logical_lanes"),
                .threads = native_int(config.workers.search_threads, "workers.search_threads"),
                .max_batch =
                    native_int(config.inference.max_batch_size, "inference.max_batch_size"),
                .max_wait_us = native_int(config.inference.max_wait_us, "inference.max_wait_us"),
                .simulations = native_int(config.mcts.simulations, "mcts.simulations"),
                .max_moves = native_int(config.self_play.max_moves, "self_play.max_moves"),
                .temperature = config.self_play.temperature,
                .temperature_moves = native_int(config.self_play.temperature_moves,
                                                "self_play.temperature_moves"),
                .dirichlet_alpha = config.mcts.dirichlet_alpha,
                .dirichlet_epsilon = config.mcts.dirichlet_epsilon,
            },
        .games_per_iteration =
            static_cast<std::size_t>(config.workers.games_per_iteration),
        .training_batch_size = static_cast<std::size_t>(config.training.batch_size),
        .training_steps =
            static_cast<std::size_t>(config.training.train_steps_per_iteration),
    };
}

void require_cpu_training_runtime(const ProductionConfig& config) {
    if (config.runtime.device != "cpu") {
        throw CommandArgumentError("runtime.device " + config.runtime.device +
                                   " requires CUDA support; this native training build supports "
                                   "cpu only");
    }
}

}  // namespace diamond_orchestration
