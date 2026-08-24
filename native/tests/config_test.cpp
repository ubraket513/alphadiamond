#include "diamond_orchestration/config.hpp"

#include <functional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using diamond_orchestration::ConfigError;
using diamond_orchestration::MCTSConfig;
using diamond_orchestration::NetworkConfig;
using diamond_orchestration::ProductionConfig;
using diamond_orchestration::ReplayConfig;
using diamond_orchestration::SelfPlayConfig;
using diamond_orchestration::TrainingConfig;
using diamond_support::JsonValue;

namespace {

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

JsonValue json_number(double value) { return JsonValue{value}; }

}  // namespace

int main(int argc, char** argv) {
    try {
        require(NetworkConfig{} == NetworkConfig{.width = 128, .residual_blocks = 6},
                "network defaults changed");
        require(MCTSConfig{} == MCTSConfig{.simulations = 200, .c_puct = 1.5,
                                             .dirichlet_alpha = 0.3, .dirichlet_epsilon = 0.25, .seed = 0},
                "mcts defaults changed");
        require(SelfPlayConfig{} == SelfPlayConfig{.max_moves = 2000, .temperature_moves = 20,
                                                    .temperature = 1.0, .seed = 0, .bootstrap_prior = "none"},
                "self-play defaults changed");
        require(ReplayConfig{} == ReplayConfig{.capacity = 100000, .seed = 0}, "replay defaults changed");
        require(TrainingConfig{} == TrainingConfig{.batch_size = 128, .learning_rate = 1e-3,
                                                    .weight_decay = 1e-4, .device = "cpu", .seed = 0},
                "training defaults changed");

        const JsonValue network{JsonValue::Object{{"residual_blocks", JsonValue{int64_t{1}}}, {"width", JsonValue{int64_t{8}}}}};
        require(diamond_support::canonical_json(NetworkConfig::from_json(network).to_json()) ==
                    diamond_support::canonical_json(network),
                "network JSON must round trip");

        const JsonValue mcts{JsonValue::Object{{"c_puct", json_number(1.5)},
                                                {"dirichlet_alpha", json_number(0.3)},
                                                {"dirichlet_epsilon", json_number(0.25)},
                                                {"seed", JsonValue{int64_t{7}}},
                                                {"simulations", JsonValue{int64_t{10}}}}};
        require(diamond_support::canonical_json(MCTSConfig::from_json(mcts).to_json()) ==
                    diamond_support::canonical_json(mcts),
                "mcts JSON must round trip");

        const JsonValue self_play{JsonValue::Object{{"bootstrap_prior", JsonValue{std::string("none")}},
                                                     {"max_game_seconds", JsonValue{nullptr}},
                                                     {"max_moves", JsonValue{int64_t{20}}},
                                                     {"seed", JsonValue{int64_t{7}}},
                                                     {"temperature", json_number(1.0)},
                                                     {"temperature_moves", JsonValue{int64_t{2}}}}};
        require(diamond_support::canonical_json(SelfPlayConfig::from_json(self_play).to_json()) ==
                    diamond_support::canonical_json(self_play),
                "self-play JSON must round trip");

        const JsonValue replay{JsonValue::Object{{"capacity", JsonValue{int64_t{32}}}, {"seed", JsonValue{int64_t{7}}}}};
        require(diamond_support::canonical_json(ReplayConfig::from_json(replay).to_json()) ==
                    diamond_support::canonical_json(replay),
                "replay JSON must round trip");

        const JsonValue training{JsonValue::Object{{"batch_size", JsonValue{int64_t{4}}},
                                                    {"device", JsonValue{std::string("cpu")}},
                                                    {"learning_rate", json_number(0.001)},
                                                    {"seed", JsonValue{int64_t{7}}},
                                                    {"weight_decay", json_number(0.0)}}};
        require(diamond_support::canonical_json(TrainingConfig::from_json(training).to_json()) ==
                    diamond_support::canonical_json(training),
                "training JSON must round trip");

        ProductionConfig production{
            .network = NetworkConfig{.width = 8, .residual_blocks = 1},
            .mcts = MCTSConfig{.simulations = 1, .c_puct = 1.5, .dirichlet_alpha = 0.3,
                               .dirichlet_epsilon = 0.25, .seed = 7},
            .self_play = SelfPlayConfig{.max_moves = 2000, .temperature_moves = 20,
                                        .temperature = 1.0, .seed = 7, .bootstrap_prior = "none"},
            .replay = ReplayConfig{.capacity = 128, .seed = 7},
            .training = TrainingConfig{.batch_size = 1, .learning_rate = 0.001,
                                       .weight_decay = 0.0, .device = "cpu", .seed = 7},
            .arena = {.games = 4, .seed = 7, .max_moves = 2000, .promotion_threshold = 0.55},
            .workers = {.worker_count = 2, .games_per_iteration = 2, .retry_id = "attempt-0"},
            .inference = {.max_batch_size = 8, .max_wait_ms = 2,
                          .request_queue_capacity = 32, .response_timeout_s = 10.0},
            .benchmark = {.opening_count = 1, .opening_max_depth = 0, .opening_seed = 7,
                          .opening_suite_version = "production-openings-v1"},
            .run_seed = 7,
        };
        const JsonValue production_json = production.to_json();
        require(diamond_support::canonical_json(ProductionConfig::from_json(production_json).to_json()) ==
                    diamond_support::canonical_json(production_json),
                "production JSON must round trip");

        rejects([] { NetworkConfig::from_json(JsonValue{JsonValue::Object{{"width", JsonValue{int64_t{8}}}}}); },
                "missing key must be rejected");
        rejects([] { ReplayConfig::from_json(JsonValue{JsonValue::Object{{"capacity", JsonValue{int64_t{8}}}, {"seed", JsonValue{int64_t{0}}}, {"unknown", JsonValue{true}}}}); },
                "unknown key must be rejected");
        rejects([] { SelfPlayConfig::from_json(JsonValue{JsonValue::Object{{"bootstrap_prior", JsonValue{std::string("none")}}, {"max_game_seconds", JsonValue{0.0}}, {"max_moves", JsonValue{int64_t{1}}}, {"seed", JsonValue{int64_t{0}}}, {"temperature", JsonValue{1.0}}, {"temperature_moves", JsonValue{int64_t{0}}}}}); },
                "zero game budget must be rejected");
        rejects([] { MCTSConfig::from_json(JsonValue{JsonValue::Object{{"c_puct", JsonValue{1.0}}, {"dirichlet_alpha", JsonValue{0.3}}, {"dirichlet_epsilon", JsonValue{1.1}}, {"seed", JsonValue{int64_t{0}}}, {"simulations", JsonValue{int64_t{1}}}}}); },
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
