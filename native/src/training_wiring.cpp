#include "diamond_orchestration/training_wiring.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace diamond_orchestration {
namespace {

int native_int(int64_t value, std::string_view field) {
    if (value > std::numeric_limits<int>::max()) {
        throw ConfigError(std::string(field) + " exceeds the native scheduler limit");
    }
    return static_cast<int>(value);
}

} // namespace

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
                .temperature_moves =
                    native_int(config.self_play.temperature_moves, "self_play.temperature_moves"),
                .dirichlet_alpha = config.mcts.dirichlet_alpha,
                .dirichlet_epsilon = config.mcts.dirichlet_epsilon,
                .simulations_late =
                    native_int(config.mcts.simulations_late, "mcts.simulations_late"),
                .repeat_window = native_int(config.mcts.repeat_window, "mcts.repeat_window"),
                .bootstrap_prior = config.self_play.bootstrap_prior != kBootstrapPriorNone,
            },
        .games_per_iteration = static_cast<std::size_t>(config.workers.games_per_iteration),
        .training_batch_size = static_cast<std::size_t>(config.training.batch_size),
        .training_steps = static_cast<std::size_t>(config.training.train_steps_per_iteration),
    };
}

soo::EpisodeConfig wire_arena_episode(const ProductionConfig& config, std::size_t lanes) {
    config.validate();
    if (lanes == 0)
        throw ConfigError("an arena group must contain at least one game");
    // Threads and batch cannot usefully exceed the games in flight: synchronous
    // MCTS allows one outstanding request per lane, so a group of four games can
    // never present a fifth position to evaluate.
    const auto bounded = [lanes](int64_t configured, std::string_view field) {
        return std::min(native_int(configured, field), static_cast<int>(lanes));
    };
    return {
        .lanes = static_cast<int>(lanes),
        .threads = bounded(config.workers.search_threads, "workers.search_threads"),
        .max_batch = bounded(config.inference.max_batch_size, "inference.max_batch_size"),
        .max_wait_us = native_int(config.inference.max_wait_us, "inference.max_wait_us"),
        .simulations = native_int(config.mcts.simulations, "mcts.simulations"),
        .max_moves = native_int(config.arena.max_moves, "arena.max_moves"),
        .temperature = 0.0,
        .temperature_moves = 0,
        .dirichlet_alpha = config.mcts.dirichlet_alpha,
        .dirichlet_epsilon = 0.0,
        // The arena plays the phase the config declares. While a run is
        // bootstrapping, a network prior steers neither side into a camp, so
        // every arena game would reach `max_moves` and the stage would report
        // nothing but incomplete blocks -- having paid for a full-length game
        // each. Both sides take the same heuristic prior, so the comparison
        // stays symmetric and turns on the value head, which is what actually
        // differs between candidate and champion at that point. Removing the
        // prior from the config removes it from here.
        .bootstrap_prior = config.self_play.bootstrap_prior != kBootstrapPriorNone,
    };
}

} // namespace diamond_orchestration
