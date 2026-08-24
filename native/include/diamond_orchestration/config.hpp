#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "diamond_support/json.hpp"

namespace diamond_orchestration {

class ConfigError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

inline constexpr std::string_view kBootstrapPriorNone = "none";
inline constexpr std::string_view kCanonicalTargetDistanceV1 =
    "canonical-target-distance-v1";
inline constexpr std::string_view kCanonicalTargetVacancyDistanceV2 =
    "canonical-target-vacancy-distance-v2";

// These types deliberately use only values representable by JsonValue.  Their
// JSON encoders require every member, while default construction is available
// for callers assembling an in-memory configuration.
struct NetworkConfig final {
    int64_t width = 128;
    int64_t residual_blocks = 6;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static NetworkConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const NetworkConfig&) const = default;
};

struct MCTSConfig final {
    int64_t simulations = 200;
    double c_puct = 1.5;
    double dirichlet_alpha = 0.3;
    double dirichlet_epsilon = 0.25;
    uint64_t seed = 0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static MCTSConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const MCTSConfig&) const = default;
};

struct SelfPlayConfig final {
    int64_t max_moves = 2000;
    int64_t temperature_moves = 20;
    double temperature = 1.0;
    uint64_t seed = 0;
    std::string bootstrap_prior = std::string(kBootstrapPriorNone);
    std::optional<double> max_game_seconds;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static SelfPlayConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const SelfPlayConfig&) const = default;
};

struct ReplayConfig final {
    int64_t capacity = 100000;
    uint64_t seed = 0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static ReplayConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const ReplayConfig&) const = default;
};

struct TrainingConfig final {
    int64_t batch_size = 128;
    double learning_rate = 1e-3;
    double weight_decay = 1e-4;
    std::string device = "cpu";
    uint64_t seed = 0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static TrainingConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const TrainingConfig&) const = default;
};

struct ArenaConfig final {
    int64_t games = 36;
    uint64_t seed = 0;
    int64_t max_moves = 2000;
    double promotion_threshold = 0.55;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static ArenaConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const ArenaConfig&) const = default;
};

struct WorkerConfig final {
    int64_t worker_count = 1;
    int64_t games_per_iteration = 1;
    std::string retry_id = "attempt-0";

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static WorkerConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const WorkerConfig&) const = default;
};

struct InferenceConfig final {
    int64_t max_batch_size = 1;
    int64_t max_wait_ms = 1;
    int64_t request_queue_capacity = 1;
    double response_timeout_s = 5.0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static InferenceConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const InferenceConfig&) const = default;
};

struct BenchmarkConfig final {
    int64_t opening_count = 1;
    int64_t opening_max_depth = 0;
    uint64_t opening_seed = 0;
    std::string opening_suite_version = "production-openings-v1";

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static BenchmarkConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const BenchmarkConfig&) const = default;
};

// The top-level document accepted by the current production JSON files.  It
// owns the exact schema boundary; callers never provide YAML or loosely typed
// key/value data to native orchestration.
struct ProductionConfig final {
    int64_t schema_version = 1;
    std::string model_name = "Soo";
    std::string model_version = "2.0.0";
    NetworkConfig network;
    MCTSConfig mcts;
    SelfPlayConfig self_play;
    ReplayConfig replay;
    TrainingConfig training;
    ArenaConfig arena{.games = 40};
    WorkerConfig workers;
    InferenceConfig inference;
    BenchmarkConfig benchmark;
    uint64_t run_seed = 0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static ProductionConfig from_json(const diamond_support::JsonValue& value);

    bool operator==(const ProductionConfig&) const = default;
};

}  // namespace diamond_orchestration
