#include "diamond_orchestration/config.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

using diamond_orchestration::ConfigError;
using diamond_orchestration::InferenceConfig;
using diamond_orchestration::MCTSConfig;
using diamond_orchestration::NetworkConfig;
using diamond_orchestration::OpeningSuiteConfig;
using diamond_orchestration::ProductionConfig;
using diamond_orchestration::PromotionStatisticsConfig;
using diamond_orchestration::ReplayConfig;
using diamond_orchestration::RunBudgetConfig;
using diamond_orchestration::RuntimeConfig;
using diamond_orchestration::SelfPlayConfig;
using diamond_orchestration::TrainingConfig;
using diamond_orchestration::WorkerConfig;
using diamond_support::JsonValue;

namespace {

constexpr std::string_view kV1MigrationError =
    "production config schema_version 1 is not supported; migrate max_wait_ms to max_wait_us, "
    "worker_count to logical_lanes and search_threads, and add runtime, run_budget, "
    "train_steps_per_iteration, opening_suite, and promotion_statistics";

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()>& action, const char* message) {
    try {
        action();
    } catch (const ConfigError&) {
        return;
    }
    throw std::runtime_error(message);
}

void rejects_with_message(const std::function<void()>& action, std::string_view expected,
                          const char* message) {
    try {
        action();
    } catch (const ConfigError& error) {
        require(error.what() == expected, message);
        return;
    }
    throw std::runtime_error(message);
}

JsonValue runtime_json(std::string device, std::string precision = "fp32") {
    return JsonValue{JsonValue::Object{{"device", JsonValue{std::move(device)}},
                                       {"precision", JsonValue{std::move(precision)}}}};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(RuntimeConfig{} == RuntimeConfig{.device = "cpu", .precision = "fp32"},
                "runtime defaults changed");
        require(RunBudgetConfig{} == RunBudgetConfig{.max_iterations = 1,
                                                     .max_wall_clock_seconds = std::nullopt,
                                                     .checkpoint_every_iterations = 1},
                "run-budget defaults changed");
        require(OpeningSuiteConfig{} == OpeningSuiteConfig{.id = "production-openings-v1",
                                                           .version = 1,
                                                           .seed = 0,
                                                           .count = 1,
                                                           .max_depth = 0},
                "opening-suite defaults changed");
        require(PromotionStatisticsConfig{} ==
                    PromotionStatisticsConfig{.method = "opening-block-bootstrap-v1",
                                              .resampling_unit = "opening_block",
                                              .confidence_level = 0.95,
                                              .bootstrap_replicates = 10000,
                                              .seed = 0},
                "promotion-statistics defaults changed");
        require(TrainingConfig{} == TrainingConfig{.batch_size = 128,
                                                   .train_steps_per_iteration = 1,
                                                   .learning_rate = 1e-3,
                                                   .weight_decay = 1e-4,
                                                   .seed = 0},
                "training defaults changed");
        require(WorkerConfig{} == WorkerConfig{.logical_lanes = 1,
                                               .search_threads = 1,
                                               .games_per_iteration = 1,
                                               .retry_id = "attempt-0"},
                "worker defaults changed");
        require(InferenceConfig{} == InferenceConfig{.max_batch_size = 1,
                                                     .max_wait_us = 1,
                                                     .request_queue_capacity = 1,
                                                     .response_timeout_s = 5.0},
                "inference defaults changed");

        const JsonValue network{JsonValue::Object{{"residual_blocks", JsonValue{int64_t{1}}},
                                                  {"width", JsonValue{int64_t{8}}}}};
        require(diamond_support::canonical_json(NetworkConfig::from_json(network).to_json()) ==
                    diamond_support::canonical_json(network),
                "network JSON must round trip");

        const JsonValue mcts{JsonValue::Object{{"c_puct", JsonValue{1.5}},
                                               {"dirichlet_alpha", JsonValue{0.3}},
                                               {"dirichlet_epsilon", JsonValue{0.25}},
                                               {"repeat_window", JsonValue{int64_t{8}}},
                                               {"seed", JsonValue{int64_t{7}}},
                                               {"simulations", JsonValue{int64_t{10}}},
                                               {"simulations_late", JsonValue{int64_t{20}}}}};
        require(diamond_support::canonical_json(MCTSConfig::from_json(mcts).to_json()) ==
                    diamond_support::canonical_json(mcts),
                "mcts JSON must round trip");

        // The repetition trigger was added after runs were already on disk, and
        // `resume` parses a run's immutable resolved-config.json back through
        // from_json. A config written before the field existed must therefore
        // still load, with the trigger disabled.
        const JsonValue legacy_mcts{JsonValue::Object{{"c_puct", JsonValue{1.5}},
                                                      {"dirichlet_alpha", JsonValue{0.3}},
                                                      {"dirichlet_epsilon", JsonValue{0.25}},
                                                      {"seed", JsonValue{int64_t{7}}},
                                                      {"simulations", JsonValue{int64_t{10}}}}};
        const auto legacy = MCTSConfig::from_json(legacy_mcts);
        require(legacy.simulations_late == 0 && legacy.repeat_window == 0,
                "an mcts config predating the repetition trigger must load it disabled");

        // Unknown keys stay rejected: tolerating absence must not become
        // tolerating anything.
        JsonValue::Object unknown_mcts = std::get<JsonValue::Object>(legacy_mcts.value);
        unknown_mcts.emplace("simulations_lete", JsonValue{int64_t{20}});
        rejects([&] { (void)MCTSConfig::from_json(JsonValue{unknown_mcts}); },
                "an unknown mcts key must still be rejected");

        // Half a control is inert but looks enabled, so it is refused.
        for (const auto& half : {std::pair<std::string, int64_t>{"simulations_late", 20},
                                 std::pair<std::string, int64_t>{"repeat_window", 8}}) {
            JsonValue::Object partial = std::get<JsonValue::Object>(legacy_mcts.value);
            partial.emplace(half.first, JsonValue{half.second});
            rejects([&] { (void)MCTSConfig::from_json(JsonValue{partial}); },
                    "half of the repetition trigger must be rejected");
        }

        const JsonValue self_play{
            JsonValue::Object{{"bootstrap_prior", JsonValue{std::string("none")}},
                              {"max_game_seconds", JsonValue{nullptr}},
                              {"max_moves", JsonValue{int64_t{20}}},
                              {"seed", JsonValue{int64_t{7}}},
                              {"temperature", JsonValue{1.0}},
                              {"temperature_moves", JsonValue{int64_t{2}}}}};
        require(diamond_support::canonical_json(SelfPlayConfig::from_json(self_play).to_json()) ==
                    diamond_support::canonical_json(self_play),
                "self-play JSON must round trip");

        const JsonValue replay{JsonValue::Object{{"capacity", JsonValue{int64_t{32}}},
                                                 {"seed", JsonValue{int64_t{7}}}}};
        require(diamond_support::canonical_json(ReplayConfig::from_json(replay).to_json()) ==
                    diamond_support::canonical_json(replay),
                "replay JSON must round trip");

        const JsonValue training{
            JsonValue::Object{{"batch_size", JsonValue{int64_t{4}}},
                              {"learning_rate", JsonValue{0.001}},
                              {"seed", JsonValue{int64_t{7}}},
                              {"train_steps_per_iteration", JsonValue{int64_t{2}}},
                              {"weight_decay", JsonValue{0.0}}}};
        require(diamond_support::canonical_json(TrainingConfig::from_json(training).to_json()) ==
                    diamond_support::canonical_json(training),
                "training JSON must round trip");

        for (const char* device : {"cpu", "cuda", "cuda:0", "cuda:17"}) {
            const JsonValue value = runtime_json(device);
            require(diamond_support::canonical_json(RuntimeConfig::from_json(value).to_json()) ==
                        diamond_support::canonical_json(value),
                    "runtime JSON must round trip");
        }
        for (const char* device : {"CPU", "CUDA", "cuda:", "cuda:-1", "cuda:1x", "gpu"}) {
            rejects([&] { (void)RuntimeConfig::from_json(runtime_json(device)); },
                    "invalid runtime device must be rejected");
        }
        rejects([] { (void)RuntimeConfig::from_json(runtime_json("cpu", "fp16")); },
                "unsupported precision must be rejected");
        rejects(
            [] {
                (void)RuntimeConfig::from_json(
                    JsonValue{JsonValue::Object{{"device", JsonValue{"cpu"}},
                                                {"precision", JsonValue{"fp32"}},
                                                {"unknown", JsonValue{true}}}});
            },
            "unknown runtime key must be rejected");
        rejects(
            [] {
                (void)RuntimeConfig::from_json(
                    JsonValue{JsonValue::Object{{"device", JsonValue{"cpu"}}}});
            },
            "missing runtime key must be rejected");

        ProductionConfig production{
            .schema_version = 2,
            .model_name = "Soo",
            .model_version = "2.0.0",
            .network = NetworkConfig{.width = 128, .residual_blocks = 6},
            .runtime = RuntimeConfig{.device = "cuda:0", .precision = "fp32"},
            .mcts = MCTSConfig{.simulations = 400,
                               .c_puct = 1.5,
                               .dirichlet_alpha = 0.3,
                               .dirichlet_epsilon = 0.25,
                               .seed = 7},
            .self_play = SelfPlayConfig{.max_moves = 2000,
                                        .temperature_moves = 20,
                                        .temperature = 1.0,
                                        .seed = 7,
                                        .bootstrap_prior = "none"},
            .workers = WorkerConfig{.logical_lanes = 2,
                                    .search_threads = 3,
                                    .games_per_iteration = 2,
                                    .retry_id = "attempt-0"},
            .inference = InferenceConfig{.max_batch_size = 8,
                                         .max_wait_us = 50,
                                         .request_queue_capacity = 32,
                                         .response_timeout_s = 10.0},
            .replay = ReplayConfig{.capacity = 128, .seed = 7},
            .training = TrainingConfig{.batch_size = 4,
                                       .train_steps_per_iteration = 1,
                                       .learning_rate = 0.001,
                                       .weight_decay = 0.0,
                                       .seed = 7},
            .run_budget = RunBudgetConfig{.max_iterations = 1,
                                          .max_wall_clock_seconds = std::nullopt,
                                          .checkpoint_every_iterations = 1},
            .arena = {.games = 4, .seed = 7, .max_moves = 2000, .promotion_threshold = 0.55},
            .opening_suite = OpeningSuiteConfig{.id = "production-openings-v1",
                                                .version = 1,
                                                .seed = 7,
                                                .count = 1,
                                                .max_depth = 0},
            .promotion_statistics =
                PromotionStatisticsConfig{.method = "opening-block-bootstrap-v1",
                                          .resampling_unit = "opening_block",
                                          .confidence_level = 0.95,
                                          .bootstrap_replicates = 10000,
                                          .seed = 7},
            .run_seed = 7,
        };
        const JsonValue production_json = production.to_json();
        require(diamond_support::canonical_json(
                    ProductionConfig::from_json(production_json).to_json()) ==
                    diamond_support::canonical_json(production_json),
                "production v2 JSON must round trip");

        rejects_with_message(
            [] {
                (void)ProductionConfig::from_json(
                    JsonValue{JsonValue::Object{{"schema_version", JsonValue{int64_t{1}}}}});
            },
            kV1MigrationError, "v1 migration error must be exact");
        rejects(
            [&] {
                auto invalid = std::get<JsonValue::Object>(production_json.value);
                std::get<JsonValue::Object>(invalid.at("run_budget").value)["max_iterations"] =
                    JsonValue{int64_t{0}};
                (void)ProductionConfig::from_json(JsonValue{std::move(invalid)});
            },
            "invalid run budget must be rejected");
        rejects(
            [&] {
                auto invalid = std::get<JsonValue::Object>(production_json.value);
                std::get<JsonValue::Object>(
                    invalid.at("promotion_statistics").value)["resampling_unit"] =
                    JsonValue{"game"};
                (void)ProductionConfig::from_json(JsonValue{std::move(invalid)});
            },
            "invalid promotion statistics must be rejected");
        rejects(
            [] {
                (void)NetworkConfig::from_json(
                    JsonValue{JsonValue::Object{{"width", JsonValue{int64_t{8}}}}});
            },
            "missing nested key must be rejected");
        rejects(
            [] {
                (void)ReplayConfig::from_json(
                    JsonValue{JsonValue::Object{{"capacity", JsonValue{int64_t{8}}},
                                                {"seed", JsonValue{int64_t{0}}},
                                                {"unknown", JsonValue{true}}}});
            },
            "unknown nested key must be rejected");
        rejects(
            [] {
                (void)SelfPlayConfig::from_json(
                    JsonValue{JsonValue::Object{{"bootstrap_prior", JsonValue{std::string("none")}},
                                                {"max_game_seconds", JsonValue{0.0}},
                                                {"max_moves", JsonValue{int64_t{1}}},
                                                {"seed", JsonValue{int64_t{0}}},
                                                {"temperature", JsonValue{1.0}},
                                                {"temperature_moves", JsonValue{int64_t{0}}}}});
            },
            "zero game budget must be rejected");
        rejects(
            [] {
                (void)MCTSConfig::from_json(
                    JsonValue{JsonValue::Object{{"c_puct", JsonValue{1.0}},
                                                {"dirichlet_alpha", JsonValue{0.3}},
                                                {"dirichlet_epsilon", JsonValue{1.1}},
                                                {"seed", JsonValue{int64_t{0}}},
                                                {"simulations", JsonValue{int64_t{1}}}}});
            },
            "out-of-range dirichlet epsilon must be rejected");
        rejects([&] {
            auto invalid = std::get<JsonValue::Object>(production_json.value);
            invalid.emplace("unknown", JsonValue{true});
            (void)ProductionConfig::from_json(JsonValue{std::move(invalid)});
        }, "unknown production key must be rejected");

        if (argc == 2) {
            const std::filesystem::path root = argv[1];
            for (const char* name : {"soo-production.json", "soo-bootstrap.json",
                                     "min-production.json", "min-bootstrap.json"}) {
                std::ifstream input(root / name, std::ios::binary);
                require(static_cast<bool>(input), "cannot open reference production config");
                const std::string contents{std::istreambuf_iterator<char>(input), {}};
                const JsonValue reference = diamond_support::parse_json(contents);
                const ProductionConfig loaded = ProductionConfig::from_json(reference);
                require(diamond_support::canonical_json(loaded.to_json()) ==
                            diamond_support::canonical_json(reference),
                        "reference production config must round trip");
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "config_test: " << error.what() << '\n';
        return 1;
    }
}
