#include "diamond_orchestration/config.hpp"

#include <cmath>
#include <cctype>
#include <limits>
#include <set>
#include <utility>

namespace diamond_orchestration {
namespace {

using Json = diamond_support::JsonValue;
using Object = Json::Object;

const Object& object(const Json& value, std::string_view name) {
    if (const auto* result = std::get_if<Object>(&value.value)) return *result;
    throw ConfigError(std::string(name) + " must be a JSON object");
}

void require_exact_keys(const Object& value, std::initializer_list<std::string_view> expected,
                        std::string_view name) {
    std::set<std::string> actual;
    for (const auto& [key, ignored] : value) {
        (void)ignored;
        actual.insert(key);
    }
    std::set<std::string> required;
    for (std::string_view key : expected) required.emplace(key);
    if (actual == required) return;

    std::string details;
    for (const auto& key : required) {
        if (!actual.contains(key)) {
            if (!details.empty()) details += "; ";
            details += "missing: " + key;
        }
    }
    for (const auto& key : actual) {
        if (!required.contains(key)) {
            if (!details.empty()) details += "; ";
            details += "unexpected: " + key;
        }
    }
    throw ConfigError("invalid " + std::string(name) + " (" + details + ")");
}

const Json& field(const Object& value, std::string_view name, std::string_view object_name) {
    const auto found = value.find(std::string(name));
    if (found == value.end())
        throw ConfigError(std::string(object_name) + " missing " + std::string(name));
    return found->second;
}

int64_t integer(const Json& value, std::string_view name) {
    if (const auto* result = std::get_if<int64_t>(&value.value)) return *result;
    throw ConfigError(std::string(name) + " must be an integer");
}

uint64_t non_negative_seed(const Json& value, std::string_view name) {
    const int64_t result = integer(value, name);
    if (result < 0) throw ConfigError(std::string(name) + " must be a non-negative integer");
    return static_cast<uint64_t>(result);
}

double number(const Json& value, std::string_view name) {
    double result;
    if (const auto* integer_value = std::get_if<int64_t>(&value.value)) result = static_cast<double>(*integer_value);
    else if (const auto* double_value = std::get_if<double>(&value.value)) result = *double_value;
    else throw ConfigError(std::string(name) + " must be a number");
    if (!std::isfinite(result)) throw ConfigError(std::string(name) + " must be finite");
    return result;
}

const std::string& string(const Json& value, std::string_view name) {
    if (const auto* result = std::get_if<std::string>(&value.value)) return *result;
    throw ConfigError(std::string(name) + " must be a string");
}

int64_t json_integer(uint64_t value, std::string_view name) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw ConfigError(std::string(name) + " exceeds JSON integer range");
    return static_cast<int64_t>(value);
}

bool bootstrap_prior_is_valid(std::string_view value) {
    return value == kBootstrapPriorNone || value == kCanonicalTargetDistanceV1 ||
           value == kCanonicalTargetVacancyDistanceV2;
}

bool non_blank(std::string_view value) {
    for (unsigned char character : value)
        if (!std::isspace(character)) return true;
    return false;
}

}  // namespace

void NetworkConfig::validate() const {
    if (width <= 0 || residual_blocks <= 0)
        throw ConfigError("network dimensions must be positive");
}

Json NetworkConfig::to_json() const {
    validate();
    return Json{Object{{"residual_blocks", Json{residual_blocks}}, {"width", Json{width}}}};
}

NetworkConfig NetworkConfig::from_json(const Json& value) {
    const Object& input = object(value, "network");
    require_exact_keys(input, {"residual_blocks", "width"}, "network");
    NetworkConfig result{.width = integer(field(input, "width", "network"), "network.width"),
                         .residual_blocks = integer(field(input, "residual_blocks", "network"), "network.residual_blocks")};
    result.validate();
    return result;
}

void MCTSConfig::validate() const {
    if (simulations <= 0) throw ConfigError("mcts.simulations must be positive");
    if (!std::isfinite(c_puct) || c_puct <= 0) throw ConfigError("mcts.c_puct must be a positive finite number");
    if (!std::isfinite(dirichlet_alpha) || dirichlet_alpha <= 0)
        throw ConfigError("mcts.dirichlet_alpha must be a positive finite number");
    if (!std::isfinite(dirichlet_epsilon) || dirichlet_epsilon < 0 || dirichlet_epsilon > 1)
        throw ConfigError("mcts.dirichlet_epsilon must be a finite number in [0, 1]");
}

Json MCTSConfig::to_json() const {
    validate();
    return Json{Object{{"c_puct", Json{c_puct}},
                       {"dirichlet_alpha", Json{dirichlet_alpha}},
                       {"dirichlet_epsilon", Json{dirichlet_epsilon}},
                       {"seed", Json{json_integer(seed, "mcts.seed")}},
                       {"simulations", Json{simulations}}}};
}

MCTSConfig MCTSConfig::from_json(const Json& value) {
    const Object& input = object(value, "mcts");
    require_exact_keys(input, {"c_puct", "dirichlet_alpha", "dirichlet_epsilon", "seed", "simulations"}, "mcts");
    MCTSConfig result{.simulations = integer(field(input, "simulations", "mcts"), "mcts.simulations"),
                      .c_puct = number(field(input, "c_puct", "mcts"), "mcts.c_puct"),
                      .dirichlet_alpha = number(field(input, "dirichlet_alpha", "mcts"), "mcts.dirichlet_alpha"),
                      .dirichlet_epsilon = number(field(input, "dirichlet_epsilon", "mcts"), "mcts.dirichlet_epsilon"),
                      .seed = non_negative_seed(field(input, "seed", "mcts"), "mcts.seed")};
    result.validate();
    return result;
}

void SelfPlayConfig::validate() const {
    if (max_moves <= 0) throw ConfigError("self_play.max_moves must be positive");
    if (temperature_moves < 0) throw ConfigError("self_play.temperature_moves must be non-negative");
    if (!std::isfinite(temperature) || temperature < 0)
        throw ConfigError("self_play.temperature must be a non-negative finite number");
    if (!bootstrap_prior_is_valid(bootstrap_prior))
        throw ConfigError("self_play.bootstrap_prior must be a supported bootstrap prior");
    if (max_game_seconds && (!std::isfinite(*max_game_seconds) || *max_game_seconds <= 0))
        throw ConfigError("self_play.max_game_seconds must be a positive finite number or null");
}

Json SelfPlayConfig::to_json() const {
    validate();
    Json budget{nullptr};
    if (max_game_seconds) budget = Json{*max_game_seconds};
    return Json{Object{{"bootstrap_prior", Json{bootstrap_prior}},
                       {"max_game_seconds", std::move(budget)},
                       {"max_moves", Json{max_moves}},
                       {"seed", Json{json_integer(seed, "self_play.seed")}},
                       {"temperature", Json{temperature}},
                       {"temperature_moves", Json{temperature_moves}}}};
}

SelfPlayConfig SelfPlayConfig::from_json(const Json& value) {
    const Object& input = object(value, "self_play");
    require_exact_keys(input, {"bootstrap_prior", "max_game_seconds", "max_moves", "seed", "temperature", "temperature_moves"}, "self_play");
    std::optional<double> budget;
    const Json& budget_value = field(input, "max_game_seconds", "self_play");
    if (!std::holds_alternative<std::nullptr_t>(budget_value.value))
        budget = number(budget_value, "self_play.max_game_seconds");
    SelfPlayConfig result{.max_moves = integer(field(input, "max_moves", "self_play"), "self_play.max_moves"),
                          .temperature_moves = integer(field(input, "temperature_moves", "self_play"), "self_play.temperature_moves"),
                          .temperature = number(field(input, "temperature", "self_play"), "self_play.temperature"),
                          .seed = non_negative_seed(field(input, "seed", "self_play"), "self_play.seed"),
                          .bootstrap_prior = string(field(input, "bootstrap_prior", "self_play"), "self_play.bootstrap_prior"),
                          .max_game_seconds = budget};
    result.validate();
    return result;
}

void ReplayConfig::validate() const {
    if (capacity <= 0) throw ConfigError("replay.capacity must be positive");
}

Json ReplayConfig::to_json() const {
    validate();
    return Json{Object{{"capacity", Json{capacity}}, {"seed", Json{json_integer(seed, "replay.seed")}}}};
}

ReplayConfig ReplayConfig::from_json(const Json& value) {
    const Object& input = object(value, "replay");
    require_exact_keys(input, {"capacity", "seed"}, "replay");
    ReplayConfig result{.capacity = integer(field(input, "capacity", "replay"), "replay.capacity"),
                        .seed = non_negative_seed(field(input, "seed", "replay"), "replay.seed")};
    result.validate();
    return result;
}

void TrainingConfig::validate() const {
    if (batch_size <= 0) throw ConfigError("training.batch_size must be positive");
    if (!std::isfinite(learning_rate) || learning_rate <= 0)
        throw ConfigError("training.learning_rate must be a positive finite number");
    if (!std::isfinite(weight_decay) || weight_decay < 0)
        throw ConfigError("training.weight_decay must be a non-negative finite number");
    if (device.empty()) throw ConfigError("training.device must be a non-empty string");
}

Json TrainingConfig::to_json() const {
    validate();
    return Json{Object{{"batch_size", Json{batch_size}},
                       {"device", Json{device}},
                       {"learning_rate", Json{learning_rate}},
                       {"seed", Json{json_integer(seed, "training.seed")}},
                       {"weight_decay", Json{weight_decay}}}};
}

TrainingConfig TrainingConfig::from_json(const Json& value) {
    const Object& input = object(value, "training");
    require_exact_keys(input, {"batch_size", "device", "learning_rate", "seed", "weight_decay"}, "training");
    TrainingConfig result{.batch_size = integer(field(input, "batch_size", "training"), "training.batch_size"),
                          .learning_rate = number(field(input, "learning_rate", "training"), "training.learning_rate"),
                          .weight_decay = number(field(input, "weight_decay", "training"), "training.weight_decay"),
                          .device = string(field(input, "device", "training"), "training.device"),
                          .seed = non_negative_seed(field(input, "seed", "training"), "training.seed")};
    result.validate();
    return result;
}

void ArenaConfig::validate() const {
    if (games <= 0 || max_moves <= 0)
        throw ConfigError("arena.games and arena.max_moves must be positive");
    if (!std::isfinite(promotion_threshold) || promotion_threshold < 0 || promotion_threshold > 1)
        throw ConfigError("arena.promotion_threshold must be a finite number in [0, 1]");
}

Json ArenaConfig::to_json() const {
    validate();
    return Json{Object{{"games", Json{games}},
                       {"max_moves", Json{max_moves}},
                       {"promotion_threshold", Json{promotion_threshold}},
                       {"seed", Json{json_integer(seed, "arena.seed")}}}};
}

ArenaConfig ArenaConfig::from_json(const Json& value) {
    const Object& input = object(value, "arena");
    require_exact_keys(input, {"games", "max_moves", "promotion_threshold", "seed"}, "arena");
    ArenaConfig result{.games = integer(field(input, "games", "arena"), "arena.games"),
                       .seed = non_negative_seed(field(input, "seed", "arena"), "arena.seed"),
                       .max_moves = integer(field(input, "max_moves", "arena"), "arena.max_moves"),
                       .promotion_threshold = number(field(input, "promotion_threshold", "arena"), "arena.promotion_threshold")};
    result.validate();
    return result;
}

void WorkerConfig::validate() const {
    if (worker_count <= 0 || games_per_iteration <= 0)
        throw ConfigError("workers.worker_count and workers.games_per_iteration must be positive");
    if (!non_blank(retry_id)) throw ConfigError("workers.retry_id must be a non-empty string");
}

Json WorkerConfig::to_json() const {
    validate();
    return Json{Object{{"games_per_iteration", Json{games_per_iteration}},
                       {"retry_id", Json{retry_id}},
                       {"worker_count", Json{worker_count}}}};
}

WorkerConfig WorkerConfig::from_json(const Json& value) {
    const Object& input = object(value, "workers");
    require_exact_keys(input, {"games_per_iteration", "retry_id", "worker_count"}, "workers");
    WorkerConfig result{.worker_count = integer(field(input, "worker_count", "workers"), "workers.worker_count"),
                        .games_per_iteration = integer(field(input, "games_per_iteration", "workers"), "workers.games_per_iteration"),
                        .retry_id = string(field(input, "retry_id", "workers"), "workers.retry_id")};
    result.validate();
    return result;
}

void InferenceConfig::validate() const {
    if (max_batch_size <= 0 || max_wait_ms <= 0 || request_queue_capacity <= 0)
        throw ConfigError("inference batch, wait, and queue limits must be positive");
    if (!std::isfinite(response_timeout_s) || response_timeout_s <= 0)
        throw ConfigError("inference.response_timeout_s must be a positive finite number");
}

Json InferenceConfig::to_json() const {
    validate();
    return Json{Object{{"max_batch_size", Json{max_batch_size}},
                       {"max_wait_ms", Json{max_wait_ms}},
                       {"request_queue_capacity", Json{request_queue_capacity}},
                       {"response_timeout_s", Json{response_timeout_s}}}};
}

InferenceConfig InferenceConfig::from_json(const Json& value) {
    const Object& input = object(value, "inference");
    require_exact_keys(input, {"max_batch_size", "max_wait_ms", "request_queue_capacity", "response_timeout_s"}, "inference");
    InferenceConfig result{.max_batch_size = integer(field(input, "max_batch_size", "inference"), "inference.max_batch_size"),
                           .max_wait_ms = integer(field(input, "max_wait_ms", "inference"), "inference.max_wait_ms"),
                           .request_queue_capacity = integer(field(input, "request_queue_capacity", "inference"), "inference.request_queue_capacity"),
                           .response_timeout_s = number(field(input, "response_timeout_s", "inference"), "inference.response_timeout_s")};
    result.validate();
    return result;
}

void BenchmarkConfig::validate() const {
    if (opening_count <= 0) throw ConfigError("benchmark.opening_count must be positive");
    if (opening_max_depth < 0) throw ConfigError("benchmark.opening_max_depth must be non-negative");
    if (opening_count > 1 && opening_max_depth == 0)
        throw ConfigError("benchmark.opening_max_depth must be positive for multiple openings");
    if (!non_blank(opening_suite_version))
        throw ConfigError("benchmark.opening_suite_version must be a non-empty string");
}

Json BenchmarkConfig::to_json() const {
    validate();
    return Json{Object{{"opening_count", Json{opening_count}},
                       {"opening_max_depth", Json{opening_max_depth}},
                       {"opening_seed", Json{json_integer(opening_seed, "benchmark.opening_seed")}},
                       {"opening_suite_version", Json{opening_suite_version}}}};
}

BenchmarkConfig BenchmarkConfig::from_json(const Json& value) {
    const Object& input = object(value, "benchmark");
    require_exact_keys(input, {"opening_count", "opening_max_depth", "opening_seed", "opening_suite_version"}, "benchmark");
    BenchmarkConfig result{.opening_count = integer(field(input, "opening_count", "benchmark"), "benchmark.opening_count"),
                           .opening_max_depth = integer(field(input, "opening_max_depth", "benchmark"), "benchmark.opening_max_depth"),
                           .opening_seed = non_negative_seed(field(input, "opening_seed", "benchmark"), "benchmark.opening_seed"),
                           .opening_suite_version = string(field(input, "opening_suite_version", "benchmark"), "benchmark.opening_suite_version")};
    result.validate();
    return result;
}

void ProductionConfig::validate() const {
    if (schema_version != 1) throw ConfigError("unsupported production config version");
    if (model_name != "Soo" && model_name != "Min")
        throw ConfigError("model_name must be Soo or Min");
    if (!non_blank(model_version)) throw ConfigError("model_version must be a non-empty string");
    network.validate();
    mcts.validate();
    self_play.validate();
    replay.validate();
    training.validate();
    arena.validate();
    workers.validate();
    inference.validate();
    benchmark.validate();
    const int64_t balance_cycle = model_name == "Soo" ? 4 : 18;
    if (arena.games % balance_cycle != 0)
        throw ConfigError("arena.games must be a complete model balance cycle");
    if (training.batch_size > replay.capacity)
        throw ConfigError("training.batch_size exceeds replay.capacity");
}

Json ProductionConfig::to_json() const {
    validate();
    return Json{Object{{"arena", arena.to_json()},
                       {"benchmark", benchmark.to_json()},
                       {"inference", inference.to_json()},
                       {"mcts", mcts.to_json()},
                       {"model_name", Json{model_name}},
                       {"model_version", Json{model_version}},
                       {"network", network.to_json()},
                       {"replay", replay.to_json()},
                       {"run_seed", Json{json_integer(run_seed, "run_seed")}},
                       {"schema_version", Json{schema_version}},
                       {"self_play", self_play.to_json()},
                       {"training", training.to_json()},
                       {"workers", workers.to_json()}}};
}

ProductionConfig ProductionConfig::from_json(const Json& value) {
    const Object& input = object(value, "production config");
    require_exact_keys(input, {"arena", "benchmark", "inference", "mcts", "model_name", "model_version",
                               "network", "replay", "run_seed", "schema_version", "self_play", "training", "workers"},
                       "production config");
    ProductionConfig result{.schema_version = integer(field(input, "schema_version", "production config"), "schema_version"),
                            .model_name = string(field(input, "model_name", "production config"), "model_name"),
                            .model_version = string(field(input, "model_version", "production config"), "model_version"),
                            .network = NetworkConfig::from_json(field(input, "network", "production config")),
                            .mcts = MCTSConfig::from_json(field(input, "mcts", "production config")),
                            .self_play = SelfPlayConfig::from_json(field(input, "self_play", "production config")),
                            .replay = ReplayConfig::from_json(field(input, "replay", "production config")),
                            .training = TrainingConfig::from_json(field(input, "training", "production config")),
                            .arena = ArenaConfig::from_json(field(input, "arena", "production config")),
                            .workers = WorkerConfig::from_json(field(input, "workers", "production config")),
                            .inference = InferenceConfig::from_json(field(input, "inference", "production config")),
                            .benchmark = BenchmarkConfig::from_json(field(input, "benchmark", "production config")),
                            .run_seed = non_negative_seed(field(input, "run_seed", "production config"), "run_seed")};
    result.validate();
    return result;
}

}  // namespace diamond_orchestration
