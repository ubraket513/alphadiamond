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

constexpr std::string_view kV1MigrationError =
    "production config schema_version 1 is not supported; migrate max_wait_ms to max_wait_us, "
    "worker_count to logical_lanes and search_threads, and add runtime, run_budget, "
    "train_steps_per_iteration, opening_suite, and promotion_statistics";

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

// As require_exact_keys, but a named set of keys may also be absent. Strictness
// is preserved -- unknown keys are still rejected -- while a field added after a
// run started stays readable in that run's immutable resolved-config.json, which
// `resume` parses back through from_json.
void require_keys(const Object& value, std::initializer_list<std::string_view> expected,
                  std::initializer_list<std::string_view> optional, std::string_view name) {
    std::set<std::string> required;
    for (std::string_view key : expected) required.emplace(key);
    std::set<std::string> permitted = required;
    for (std::string_view key : optional) permitted.emplace(key);

    std::string details;
    std::set<std::string> actual;
    for (const auto& [key, ignored] : value) {
        (void)ignored;
        actual.insert(key);
    }
    for (const auto& key : required) {
        if (!actual.contains(key)) {
            if (!details.empty()) details += "; ";
            details += "missing: " + key;
        }
    }
    for (const auto& key : actual) {
        if (!permitted.contains(key)) {
            if (!details.empty()) details += "; ";
            details += "unexpected: " + key;
        }
    }
    if (details.empty()) return;
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
    if (const auto* integer_value = std::get_if<int64_t>(&value.value))
        result = static_cast<double>(*integer_value);
    else if (const auto* double_value = std::get_if<double>(&value.value))
        result = *double_value;
    else
        throw ConfigError(std::string(name) + " must be a number");
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

bool non_blank(std::string_view value) {
    for (unsigned char character : value)
        if (!std::isspace(character))
            return true;
    return false;
}

bool bootstrap_prior_is_valid(std::string_view value) {
    return value == kBootstrapPriorNone || value == kCanonicalTargetDistanceV1 ||
           value == kCanonicalTargetVacancyDistanceV2;
}

bool is_device_name(std::string_view value) {
    if (value == "cpu" || value == "cuda")
        return true;
    constexpr std::string_view prefix = "cuda:";
    if (!value.starts_with(prefix))
        return false;
    const std::string_view index = value.substr(prefix.size());
    if (index.empty())
        return false;
    for (unsigned char character : index)
        if (!std::isdigit(character))
            return false;
    return true;
}

std::optional<double> optional_number(const Json& value, std::string_view name) {
    if (std::holds_alternative<std::nullptr_t>(value.value))
        return std::nullopt;
    return number(value, name);
}

std::optional<int64_t> optional_integer(const Json& value, std::string_view name) {
    if (std::holds_alternative<std::nullptr_t>(value.value))
        return std::nullopt;
    return integer(value, name);
}

Json optional_json(const std::optional<double>& value) {
    return value ? Json{*value} : Json{nullptr};
}

Json optional_json(const std::optional<int64_t>& value) {
    return value ? Json{*value} : Json{nullptr};
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
                         .residual_blocks = integer(field(input, "residual_blocks", "network"),
                                                    "network.residual_blocks")};
    result.validate();
    return result;
}

void RuntimeConfig::validate() const {
    if (!is_device_name(device))
        throw ConfigError("runtime.device must be cpu, cuda, or cuda:N");
    if (precision != "fp32")
        throw ConfigError("runtime.precision must be fp32");
}

Json RuntimeConfig::to_json() const {
    validate();
    return Json{Object{{"device", Json{device}}, {"precision", Json{precision}}}};
}

RuntimeConfig RuntimeConfig::from_json(const Json& value) {
    const Object& input = object(value, "runtime");
    require_exact_keys(input, {"device", "precision"}, "runtime");
    RuntimeConfig result{.device = string(field(input, "device", "runtime"), "runtime.device"),
                         .precision =
                             string(field(input, "precision", "runtime"), "runtime.precision")};
    result.validate();
    return result;
}

void MCTSConfig::validate() const {
    if (simulations <= 0) throw ConfigError("mcts.simulations must be positive");
    if (!std::isfinite(c_puct) || c_puct <= 0)
        throw ConfigError("mcts.c_puct must be a positive finite number");
    if (!std::isfinite(dirichlet_alpha) || dirichlet_alpha <= 0)
        throw ConfigError("mcts.dirichlet_alpha must be a positive finite number");
    if (!std::isfinite(dirichlet_epsilon) || dirichlet_epsilon < 0 || dirichlet_epsilon > 1)
        throw ConfigError("mcts.dirichlet_epsilon must be a finite number in [0, 1]");
    if (simulations_late < 0) throw ConfigError("mcts.simulations_late must be non-negative");
    if (repeat_window < 0) throw ConfigError("mcts.repeat_window must be non-negative");
    // The two are one control and are meaningless apart: a window with no
    // boosted budget never changes the search, and a boosted budget with no
    // window can never fire. Rejecting the half-set pair keeps a silently
    // inert configuration from looking enabled.
    if ((simulations_late > 0) != (repeat_window > 0))
        throw ConfigError("mcts.simulations_late and mcts.repeat_window must both be set "
                          "to enable the repetition trigger, or both be zero");
}

Json MCTSConfig::to_json() const {
    validate();
    return Json{Object{{"c_puct", Json{c_puct}},
                       {"dirichlet_alpha", Json{dirichlet_alpha}},
                       {"dirichlet_epsilon", Json{dirichlet_epsilon}},
                       {"repeat_window", Json{repeat_window}},
                       {"simulations_late", Json{simulations_late}},
                       {"seed", Json{json_integer(seed, "mcts.seed")}},
                       {"simulations", Json{simulations}}}};
}

MCTSConfig MCTSConfig::from_json(const Json& value) {
    const Object& input = object(value, "mcts");
    require_keys(input, {"c_puct", "dirichlet_alpha", "dirichlet_epsilon", "seed", "simulations"},
                 {"repeat_window", "simulations_late"}, "mcts");
    MCTSConfig result{
        .simulations = integer(field(input, "simulations", "mcts"), "mcts.simulations"),
        .c_puct = number(field(input, "c_puct", "mcts"), "mcts.c_puct"),
        .dirichlet_alpha = number(field(input, "dirichlet_alpha", "mcts"), "mcts.dirichlet_alpha"),
        .dirichlet_epsilon =
            number(field(input, "dirichlet_epsilon", "mcts"), "mcts.dirichlet_epsilon"),
        .seed = non_negative_seed(field(input, "seed", "mcts"), "mcts.seed")};
    if (input.contains("simulations_late"))
        result.simulations_late =
            integer(field(input, "simulations_late", "mcts"), "mcts.simulations_late");
    if (input.contains("repeat_window"))
        result.repeat_window = integer(field(input, "repeat_window", "mcts"), "mcts.repeat_window");
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
    return Json{Object{{"bootstrap_prior", Json{bootstrap_prior}},
                       {"max_game_seconds", optional_json(max_game_seconds)},
                       {"max_moves", Json{max_moves}},
                       {"seed", Json{json_integer(seed, "self_play.seed")}},
                       {"temperature", Json{temperature}},
                       {"temperature_moves", Json{temperature_moves}}}};
}

SelfPlayConfig SelfPlayConfig::from_json(const Json& value) {
    const Object& input = object(value, "self_play");
    require_exact_keys(input,
                       {"bootstrap_prior", "max_game_seconds", "max_moves", "seed", "temperature",
                        "temperature_moves"},
                       "self_play");
    SelfPlayConfig result{
        .max_moves = integer(field(input, "max_moves", "self_play"), "self_play.max_moves"),
        .temperature_moves =
            integer(field(input, "temperature_moves", "self_play"), "self_play.temperature_moves"),
        .temperature = number(field(input, "temperature", "self_play"), "self_play.temperature"),
        .seed = non_negative_seed(field(input, "seed", "self_play"), "self_play.seed"),
        .bootstrap_prior =
            string(field(input, "bootstrap_prior", "self_play"), "self_play.bootstrap_prior"),
        .max_game_seconds = optional_number(field(input, "max_game_seconds", "self_play"),
                                            "self_play.max_game_seconds")};
    result.validate();
    return result;
}

void WorkerConfig::validate() const {
    if (logical_lanes <= 0 || search_threads <= 0 || games_per_iteration <= 0)
        throw ConfigError("workers.logical_lanes, workers.search_threads, and "
                          "workers.games_per_iteration must be positive");
    if (!non_blank(retry_id))
        throw ConfigError("workers.retry_id must be a non-empty string");
    // Games must *exceed* lanes, not merely equal them. With one game per lane
    // the job queue never engages: a lane cannot pick up fresh work when its
    // game ends, so the run finishes at the pace of its slowest game while the
    // rest idle. Batch occupancy collapses toward 1 over the second half of the
    // run and throughput is misread as a batching problem. This project has
    // paid for that twice -- once in the Python pool, once in a measurement
    // harness whose throughput column had to be retracted -- so it is a
    // configuration error rather than a note in a document.
    if (logical_lanes >= games_per_iteration)
        throw ConfigError("workers.games_per_iteration must exceed workers.logical_lanes, "
                          "otherwise the job queue never engages");
}

Json WorkerConfig::to_json() const {
    validate();
    return Json{Object{{"games_per_iteration", Json{games_per_iteration}},
                       {"logical_lanes", Json{logical_lanes}},
                       {"retry_id", Json{retry_id}},
                       {"search_threads", Json{search_threads}}}};
}

WorkerConfig WorkerConfig::from_json(const Json& value) {
    const Object& input = object(value, "workers");
    require_exact_keys(
        input, {"games_per_iteration", "logical_lanes", "retry_id", "search_threads"}, "workers");
    WorkerConfig result{
        .logical_lanes = integer(field(input, "logical_lanes", "workers"), "workers.logical_lanes"),
        .search_threads =
            integer(field(input, "search_threads", "workers"), "workers.search_threads"),
        .games_per_iteration =
            integer(field(input, "games_per_iteration", "workers"), "workers.games_per_iteration"),
        .retry_id = string(field(input, "retry_id", "workers"), "workers.retry_id")};
    result.validate();
    return result;
}

void InferenceConfig::validate() const {
    if (max_batch_size <= 0 || max_wait_us <= 0 || request_queue_capacity <= 0)
        throw ConfigError("inference batch, wait, and queue limits must be positive");
    if (!std::isfinite(response_timeout_s) || response_timeout_s <= 0)
        throw ConfigError("inference.response_timeout_s must be a positive finite number");
}

Json InferenceConfig::to_json() const {
    validate();
    return Json{Object{{"max_batch_size", Json{max_batch_size}},
                       {"max_wait_us", Json{max_wait_us}},
                       {"request_queue_capacity", Json{request_queue_capacity}},
                       {"response_timeout_s", Json{response_timeout_s}}}};
}

InferenceConfig InferenceConfig::from_json(const Json& value) {
    const Object& input = object(value, "inference");
    require_exact_keys(
        input, {"max_batch_size", "max_wait_us", "request_queue_capacity", "response_timeout_s"},
        "inference");
    InferenceConfig result{
        .max_batch_size =
            integer(field(input, "max_batch_size", "inference"), "inference.max_batch_size"),
        .max_wait_us = integer(field(input, "max_wait_us", "inference"), "inference.max_wait_us"),
        .request_queue_capacity = integer(field(input, "request_queue_capacity", "inference"),
                                          "inference.request_queue_capacity"),
        .response_timeout_s = number(field(input, "response_timeout_s", "inference"),
                                     "inference.response_timeout_s")};
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
    if (batch_size <= 0 || train_steps_per_iteration <= 0)
        throw ConfigError(
            "training.batch_size and training.train_steps_per_iteration must be positive");
    if (!std::isfinite(learning_rate) || learning_rate <= 0)
        throw ConfigError("training.learning_rate must be a positive finite number");
    if (!std::isfinite(weight_decay) || weight_decay < 0)
        throw ConfigError("training.weight_decay must be a non-negative finite number");
}

Json TrainingConfig::to_json() const {
    validate();
    return Json{Object{{"batch_size", Json{batch_size}},
                       {"learning_rate", Json{learning_rate}},
                       {"seed", Json{json_integer(seed, "training.seed")}},
                       {"train_steps_per_iteration", Json{train_steps_per_iteration}},
                       {"weight_decay", Json{weight_decay}}}};
}

TrainingConfig TrainingConfig::from_json(const Json& value) {
    const Object& input = object(value, "training");
    require_exact_keys(
        input, {"batch_size", "learning_rate", "seed", "train_steps_per_iteration", "weight_decay"},
        "training");
    TrainingConfig result{
        .batch_size = integer(field(input, "batch_size", "training"), "training.batch_size"),
        .train_steps_per_iteration = integer(field(input, "train_steps_per_iteration", "training"),
                                             "training.train_steps_per_iteration"),
        .learning_rate =
            number(field(input, "learning_rate", "training"), "training.learning_rate"),
        .weight_decay = number(field(input, "weight_decay", "training"), "training.weight_decay"),
        .seed = non_negative_seed(field(input, "seed", "training"), "training.seed")};
    result.validate();
    return result;
}

void RunBudgetConfig::validate() const {
    if (max_iterations && *max_iterations <= 0)
        throw ConfigError("run_budget.max_iterations must be a positive integer or null");
    if (max_wall_clock_seconds &&
        (!std::isfinite(*max_wall_clock_seconds) || *max_wall_clock_seconds <= 0))
        throw ConfigError(
            "run_budget.max_wall_clock_seconds must be a positive finite number or null");
    if (!max_iterations && !max_wall_clock_seconds)
        throw ConfigError("run_budget requires max_iterations or max_wall_clock_seconds");
    if (checkpoint_every_iterations <= 0)
        throw ConfigError("run_budget.checkpoint_every_iterations must be positive");
    if (max_iterations && checkpoint_every_iterations > *max_iterations)
        throw ConfigError(
            "run_budget.checkpoint_every_iterations must not exceed run_budget.max_iterations");
}

Json RunBudgetConfig::to_json() const {
    validate();
    return Json{Object{{"checkpoint_every_iterations", Json{checkpoint_every_iterations}},
                       {"max_iterations", optional_json(max_iterations)},
                       {"max_wall_clock_seconds", optional_json(max_wall_clock_seconds)}}};
}

RunBudgetConfig RunBudgetConfig::from_json(const Json& value) {
    const Object& input = object(value, "run_budget");
    require_exact_keys(input,
                       {"checkpoint_every_iterations", "max_iterations", "max_wall_clock_seconds"},
                       "run_budget");
    RunBudgetConfig result{.max_iterations =
                               optional_integer(field(input, "max_iterations", "run_budget"),
                                                "run_budget.max_iterations"),
                           .max_wall_clock_seconds =
                               optional_number(field(input, "max_wall_clock_seconds", "run_budget"),
                                               "run_budget.max_wall_clock_seconds"),
                           .checkpoint_every_iterations =
                               integer(field(input, "checkpoint_every_iterations", "run_budget"),
                                       "run_budget.checkpoint_every_iterations")};
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
                       .promotion_threshold = number(field(input, "promotion_threshold", "arena"),
                                                     "arena.promotion_threshold")};
    result.validate();
    return result;
}

void OpeningSuiteConfig::validate() const {
    if (!non_blank(id))
        throw ConfigError("opening_suite.id must be a non-empty string");
    if (version <= 0 || count <= 0)
        throw ConfigError("opening_suite.version and opening_suite.count must be positive");
    if (max_depth < 0)
        throw ConfigError("opening_suite.max_depth must be non-negative");
    if (count > 1 && max_depth == 0)
        throw ConfigError("opening_suite.max_depth must be positive for multiple openings");
}

Json OpeningSuiteConfig::to_json() const {
    validate();
    return Json{Object{{"count", Json{count}},
                       {"id", Json{id}},
                       {"max_depth", Json{max_depth}},
                       {"seed", Json{json_integer(seed, "opening_suite.seed")}},
                       {"version", Json{version}}}};
}

OpeningSuiteConfig OpeningSuiteConfig::from_json(const Json& value) {
    const Object& input = object(value, "opening_suite");
    require_exact_keys(input, {"count", "id", "max_depth", "seed", "version"}, "opening_suite");
    OpeningSuiteConfig result{
        .id = string(field(input, "id", "opening_suite"), "opening_suite.id"),
        .version = integer(field(input, "version", "opening_suite"), "opening_suite.version"),
        .seed = non_negative_seed(field(input, "seed", "opening_suite"), "opening_suite.seed"),
        .count = integer(field(input, "count", "opening_suite"), "opening_suite.count"),
        .max_depth =
            integer(field(input, "max_depth", "opening_suite"), "opening_suite.max_depth")};
    result.validate();
    return result;
}

void PromotionStatisticsConfig::validate() const {
    if (method != kOpeningBlockBootstrapV1)
        throw ConfigError("promotion_statistics.method must be opening-block-bootstrap-v1");
    if (resampling_unit != kOpeningBlockResamplingUnit)
        throw ConfigError("promotion_statistics.resampling_unit must be opening_block");
    if (!std::isfinite(confidence_level) || confidence_level <= 0 || confidence_level >= 1)
        throw ConfigError("promotion_statistics.confidence_level must be in (0, 1)");
    if (bootstrap_replicates <= 0)
        throw ConfigError("promotion_statistics.bootstrap_replicates must be positive");
}

Json PromotionStatisticsConfig::to_json() const {
    validate();
    return Json{Object{{"bootstrap_replicates", Json{bootstrap_replicates}},
                       {"confidence_level", Json{confidence_level}},
                       {"method", Json{method}},
                       {"resampling_unit", Json{resampling_unit}},
                       {"seed", Json{json_integer(seed, "promotion_statistics.seed")}}}};
}

PromotionStatisticsConfig PromotionStatisticsConfig::from_json(const Json& value) {
    const Object& input = object(value, "promotion_statistics");
    require_exact_keys(
        input, {"bootstrap_replicates", "confidence_level", "method", "resampling_unit", "seed"},
        "promotion_statistics");
    PromotionStatisticsConfig result{
        .method =
            string(field(input, "method", "promotion_statistics"), "promotion_statistics.method"),
        .resampling_unit = string(field(input, "resampling_unit", "promotion_statistics"),
                                  "promotion_statistics.resampling_unit"),
        .confidence_level = number(field(input, "confidence_level", "promotion_statistics"),
                                   "promotion_statistics.confidence_level"),
        .bootstrap_replicates =
            integer(field(input, "bootstrap_replicates", "promotion_statistics"),
                    "promotion_statistics.bootstrap_replicates"),
        .seed = non_negative_seed(field(input, "seed", "promotion_statistics"),
                                  "promotion_statistics.seed")};
    result.validate();
    return result;
}

void ProductionConfig::validate() const {
    if (schema_version != 2)
        throw ConfigError("unsupported production config schema_version");
    if (model_name != "Soo" && model_name != "Min")
        throw ConfigError("model_name must be Soo or Min");
    if (!non_blank(model_version)) throw ConfigError("model_version must be a non-empty string");
    network.validate();
    runtime.validate();
    mcts.validate();
    self_play.validate();
    workers.validate();
    inference.validate();
    replay.validate();
    training.validate();
    run_budget.validate();
    arena.validate();
    opening_suite.validate();
    promotion_statistics.validate();
    if (model_name == "Soo" && (network.width != 128 || network.residual_blocks != 6))
        throw ConfigError("Soo production config requires network width 128 and residual_blocks 6");
    const int64_t balance_cycle = model_name == "Soo" ? 4 : 18;
    if (arena.games % balance_cycle != 0)
        throw ConfigError("arena.games must be a complete model balance cycle");
    if (training.batch_size > replay.capacity)
        throw ConfigError("training.batch_size exceeds replay.capacity");
}

Json ProductionConfig::to_json() const {
    validate();
    return Json{Object{{"arena", arena.to_json()},
                       {"inference", inference.to_json()},
                       {"mcts", mcts.to_json()},
                       {"model_name", Json{model_name}},
                       {"model_version", Json{model_version}},
                       {"network", network.to_json()},
                       {"opening_suite", opening_suite.to_json()},
                       {"promotion_statistics", promotion_statistics.to_json()},
                       {"replay", replay.to_json()},
                       {"run_budget", run_budget.to_json()},
                       {"run_seed", Json{json_integer(run_seed, "run_seed")}},
                       {"runtime", runtime.to_json()},
                       {"schema_version", Json{schema_version}},
                       {"self_play", self_play.to_json()},
                       {"training", training.to_json()},
                       {"workers", workers.to_json()}}};
}

ProductionConfig ProductionConfig::from_json(const Json& value) {
    const Object& input = object(value, "production config");
    const int64_t schema_version =
        integer(field(input, "schema_version", "production config"), "schema_version");
    if (schema_version == 1)
        throw ConfigError(std::string(kV1MigrationError));
    require_exact_keys(input,
                       {"arena", "inference", "mcts", "model_name", "model_version", "network",
                        "opening_suite", "promotion_statistics", "replay", "run_budget", "run_seed",
                        "runtime", "schema_version", "self_play", "training", "workers"},
                       "production config");
    ProductionConfig result{
        .schema_version = schema_version,
        .model_name = string(field(input, "model_name", "production config"), "model_name"),
        .model_version =
            string(field(input, "model_version", "production config"), "model_version"),
        .network = NetworkConfig::from_json(field(input, "network", "production config")),
        .runtime = RuntimeConfig::from_json(field(input, "runtime", "production config")),
        .mcts = MCTSConfig::from_json(field(input, "mcts", "production config")),
        .self_play = SelfPlayConfig::from_json(field(input, "self_play", "production config")),
        .workers = WorkerConfig::from_json(field(input, "workers", "production config")),
        .inference = InferenceConfig::from_json(field(input, "inference", "production config")),
        .replay = ReplayConfig::from_json(field(input, "replay", "production config")),
        .training = TrainingConfig::from_json(field(input, "training", "production config")),
        .run_budget = RunBudgetConfig::from_json(field(input, "run_budget", "production config")),
        .arena = ArenaConfig::from_json(field(input, "arena", "production config")),
        .opening_suite =
            OpeningSuiteConfig::from_json(field(input, "opening_suite", "production config")),
        .promotion_statistics = PromotionStatisticsConfig::from_json(
            field(input, "promotion_statistics", "production config")),
        .run_seed = non_negative_seed(field(input, "run_seed", "production config"), "run_seed")};
    result.validate();
    return result;
}

}  // namespace diamond_orchestration
