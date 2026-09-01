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
                .bootstrap_prior_weight = config.self_play.bootstrap_prior_weight,
            },
        .games_per_iteration = static_cast<std::size_t>(config.workers.games_per_iteration),
        .training_batch_size = static_cast<std::size_t>(config.training.batch_size),
        .training_steps = static_cast<std::size_t>(config.training.train_steps_per_iteration),
    };
}

EvaluationPipelineWiring wire_evaluation_pipeline(const ProductionConfig& config) {
    config.validate();
    return {
        .run_arena = config.arena.enabled,
        .record_rating = config.arena.enabled,
        .activate_candidate = !config.arena.enabled,
    };
}

std::vector<std::string>
validate_training_config_transition(const ProductionConfig& from, const ProductionConfig& to,
                                    std::string_view rollback_failed_gate) {
    from.validate();
    to.validate();
    std::vector<std::string> changed;
    const auto note = [&changed](bool differs, const char* field) {
        if (differs) changed.emplace_back(field);
    };
    note(from.runtime.precision != to.runtime.precision, "runtime.precision");
    note(from.workers.logical_lanes != to.workers.logical_lanes,
         "workers.logical_lanes");
    note(from.workers.search_threads != to.workers.search_threads,
         "workers.search_threads");
    note(from.workers.games_per_iteration != to.workers.games_per_iteration,
         "workers.games_per_iteration");
    note(from.inference.max_batch_size != to.inference.max_batch_size,
         "inference.max_batch_size");
    note(from.inference.max_wait_us != to.inference.max_wait_us,
         "inference.max_wait_us");
    note(from.inference.request_queue_capacity != to.inference.request_queue_capacity,
         "inference.request_queue_capacity");
    note(from.inference.response_timeout_s != to.inference.response_timeout_s,
         "inference.response_timeout_s");
    note(from.training.train_steps_per_iteration != to.training.train_steps_per_iteration,
         "training.train_steps_per_iteration");
    note(from.self_play.bootstrap_prior != to.self_play.bootstrap_prior,
         "self_play.bootstrap_prior");
    note(from.self_play.bootstrap_prior_weight != to.self_play.bootstrap_prior_weight,
         "self_play.bootstrap_prior_weight");
    note(from.arena.enabled != to.arena.enabled, "arena.enabled");

    auto immutable_projection = to;
    immutable_projection.runtime.precision = from.runtime.precision;
    immutable_projection.workers.logical_lanes = from.workers.logical_lanes;
    immutable_projection.workers.search_threads = from.workers.search_threads;
    immutable_projection.workers.games_per_iteration = from.workers.games_per_iteration;
    immutable_projection.inference.max_batch_size = from.inference.max_batch_size;
    immutable_projection.inference.max_wait_us = from.inference.max_wait_us;
    immutable_projection.inference.request_queue_capacity =
        from.inference.request_queue_capacity;
    immutable_projection.inference.response_timeout_s = from.inference.response_timeout_s;
    immutable_projection.training.train_steps_per_iteration =
        from.training.train_steps_per_iteration;
    immutable_projection.self_play.bootstrap_prior = from.self_play.bootstrap_prior;
    immutable_projection.self_play.bootstrap_prior_weight = from.self_play.bootstrap_prior_weight;
    immutable_projection.arena.enabled = from.arena.enabled;
    if (immutable_projection != from)
        throw ConfigError("training config transition changes an immutable field");
    if (to.self_play.bootstrap_prior_weight > from.self_play.bootstrap_prior_weight &&
        rollback_failed_gate.empty())
        throw ConfigError(
            "increasing self_play.bootstrap_prior_weight requires a named failed gate");
    if (changed.empty())
        throw ConfigError("training config transition does not change any field");
    return changed;
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
        // Promotion is greedy except when a physical position has already
        // occurred within eight plies. Seeded temperature sampling on that
        // move breaks the demonstrated cycle without adding search cost or
        // making ordinary arena play stochastic.
        .repeat_window = 8,
        .repetition_temperature = 1.0,
        // The arena plays the phase the config declares. While a run is
        // bootstrapping, a network prior steers neither side into a camp, so
        // every arena game would reach `max_moves` and the stage would report
        // nothing but incomplete blocks -- having paid for a full-length game
        // each. Both sides take the same heuristic prior, so the comparison
        // stays symmetric and turns on the value head, which is what actually
        // differs between candidate and champion at that point. Removing the
        // prior from the config removes it from here.
        .bootstrap_prior = config.self_play.bootstrap_prior != kBootstrapPriorNone,
        .bootstrap_prior_weight = config.self_play.bootstrap_prior_weight,
    };
}

} // namespace diamond_orchestration
