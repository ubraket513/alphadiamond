#pragma once

#include <cstdint>
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
inline constexpr std::string_view kCanonicalTargetDistanceV1 = "canonical-target-distance-v1";
inline constexpr std::string_view kCanonicalTargetVacancyDistanceV2 =
    "canonical-target-vacancy-distance-v2";
inline constexpr std::string_view kOpeningBlockBootstrapV1 = "opening-block-bootstrap-v1";
inline constexpr std::string_view kOpeningBlockResamplingUnit = "opening_block";

struct NetworkConfig final {
    int64_t width = 128;
    int64_t residual_blocks = 6;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static NetworkConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const NetworkConfig&) const = default;
};

struct RuntimeConfig final {
    std::string device = "cpu";
    std::string precision = "fp32";

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static RuntimeConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const RuntimeConfig&) const = default;
};

struct MCTSConfig final {
    int64_t simulations = 200;
    double c_puct = 1.5;
    double dirichlet_alpha = 0.3;
    double dirichlet_epsilon = 0.25;
    uint64_t seed = 0;

    // Adaptive search budget for the repetition attractor. The aborted tail is a
    // short-cycle shuffle rather than slow progress -- median 31.6 % unique
    // positions, one position revisited 61 times, 68.4 % of moves returning
    // within 8 ply -- so the budget is spent where a game has demonstrably
    // looped. `simulations_late` applies instead of `simulations` when the
    // current position already occurred within `repeat_window` plies of the
    // game; both zero disables the trigger and every move uses `simulations`.
    // See docs/model-training/soo_scratch_training.md sections 6.2 and 6.6.
    int64_t simulations_late = 0;
    int64_t repeat_window = 0;

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
    double bootstrap_prior_weight = 1.0;
    std::optional<double> max_game_seconds;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static SelfPlayConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const SelfPlayConfig&) const = default;
};

struct WorkerConfig final {
    int64_t logical_lanes = 1;
    int64_t search_threads = 1;
    // Two, not one: games must exceed lanes or the job queue never engages, and
    // a default configuration has to be a valid one.
    int64_t games_per_iteration = 2;
    std::string retry_id = "attempt-0";

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static WorkerConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const WorkerConfig&) const = default;
};

struct InferenceConfig final {
    int64_t max_batch_size = 1;
    int64_t max_wait_us = 1;
    int64_t request_queue_capacity = 1;
    double response_timeout_s = 5.0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static InferenceConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const InferenceConfig&) const = default;
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
    int64_t train_steps_per_iteration = 1;
    double learning_rate = 1e-3;
    double weight_decay = 1e-4;
    uint64_t seed = 0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static TrainingConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const TrainingConfig&) const = default;
};

struct RunBudgetConfig final {
    std::optional<int64_t> max_iterations = 1;
    std::optional<double> max_wall_clock_seconds;
    int64_t checkpoint_every_iterations = 1;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static RunBudgetConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const RunBudgetConfig&) const = default;
};

struct ArenaConfig final {
    bool enabled = true;
    int64_t games = 36;
    uint64_t seed = 0;
    int64_t max_moves = 2000;
    double promotion_threshold = 0.55;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static ArenaConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const ArenaConfig&) const = default;
};

struct OpeningSuiteConfig final {
    std::string id = "production-openings-v1";
    int64_t version = 1;
    uint64_t seed = 0;
    int64_t count = 1;
    int64_t max_depth = 0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static OpeningSuiteConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const OpeningSuiteConfig&) const = default;
};

struct PromotionStatisticsConfig final {
    std::string method = std::string(kOpeningBlockBootstrapV1);
    std::string resampling_unit = std::string(kOpeningBlockResamplingUnit);
    double confidence_level = 0.95;
    int64_t bootstrap_replicates = 10000;
    uint64_t seed = 0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static PromotionStatisticsConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const PromotionStatisticsConfig&) const = default;
};

struct ProductionConfig final {
    int64_t schema_version = 2;
    std::string model_name = "Soo";
    std::string model_version = "2.0.0";
    NetworkConfig network;
    RuntimeConfig runtime;
    MCTSConfig mcts;
    SelfPlayConfig self_play;
    WorkerConfig workers;
    InferenceConfig inference;
    ReplayConfig replay;
    TrainingConfig training;
    RunBudgetConfig run_budget;
    ArenaConfig arena{.games = 40};
    OpeningSuiteConfig opening_suite;
    PromotionStatisticsConfig promotion_statistics;
    uint64_t run_seed = 0;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    static ProductionConfig from_json(const diamond_support::JsonValue& value);
    bool operator==(const ProductionConfig&) const = default;
};

} // namespace diamond_orchestration
