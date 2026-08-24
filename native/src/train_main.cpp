#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "diamond_model/soo_model.hpp"
#include "diamond_orchestration/arena.hpp"
#include "diamond_orchestration/coordinator.hpp"
#include "diamond_orchestration/rating_store.hpp"
#include "diamond_orchestration/report.hpp"
#include "diamond_pipeline/pipeline.hpp"
#include "diamond_training/checkpoint.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/board.hpp"

namespace {
using Json = diamond_support::JsonValue;
using Object = Json::Object;
using diamond_orchestration::CommandArtifactError;
using diamond_orchestration::CommandRequest;
using diamond_orchestration::ProductionConfig;
using diamond_orchestration::RunStage;

std::filesystem::path root(const CommandRequest& request) {
    return request.runtime_dir / "runs" / (request.model_name == "Soo" ? "soo" : "min") / request.run_id;
}
void write_json(const std::filesystem::path& path, Json value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write native command artifact: " + path.string());
    output << diamond_support::canonical_json(value) << '\n';
    if (!output) throw std::runtime_error("cannot write native command artifact: " + path.string());
}
diamond_training::Compatibility wire(const ProductionConfig& config) {
    const diamond_training::NetworkConfig network{config.network.residual_blocks, config.network.width};
    return config.model_name == "Soo" ? diamond_training::Compatibility::soo(config.model_version, network)
                                      : diamond_training::Compatibility::min(config.model_version, network);
}
soo::Match game_match(const ProductionConfig& config) {
    soo::ensure_topology_configured();
    soo::Match match;
    if (config.model_name == "Soo") {
        match.count = 2; match.players[0] = {1, 0, 3}; match.players[1] = {2, 3, 0};
    } else {
        match.count = 3; match.players[0] = {1, 0, 3}; match.players[1] = {2, 2, 5}; match.players[2] = {3, 4, 1};
    }
    return match;
}
soo::State opening(const soo::Match& match) {
    soo::State state;
    for (std::size_t seat = 0; seat < match.count; ++seat)
        for (uint8_t position : soo::topology().camp_positions[match.players[seat].camp])
            state.occupancy[position] = match.players[seat].id;
    state.current_player = match.players[0].id;
    return state;
}
diamond_model::DiamondModel model(const ProductionConfig& config) {
    return diamond_model::DiamondModel(config.network.width, config.network.residual_blocks,
        config.model_name == "Soo" ? 4 : 6, config.model_name == "Soo" ? 1 : 3);
}
Json describe(RunStage stage, const diamond_orchestration::RunState& state) {
    return Json{Object{{"iteration", state.payload().at("iteration")},
                       {"stage", Json{static_cast<int64_t>(stage)}}}};
}
struct Result { std::size_t games = 0; uint64_t step = 0; };
Result iterate(const CommandRequest& request, const ProductionConfig& config, const std::string& operation) {
    auto native_model = model(config); const auto compatibility = wire(config);
    diamond_training::Trainer trainer(native_model, compatibility, {config.training.learning_rate, config.training.weight_decay});
    try { (void)diamond_training::load_checkpoint_v2(request.checkpoint_path, trainer); }
    catch (const diamond_training::CheckpointError& error) { throw CommandArtifactError(error.what()); }
    diamond_pipeline::ModelPool models(1);
    const diamond_pipeline::ModelKey key{config.model_name, config.model_version,
        diamond_support::sha256(request.checkpoint_path.lexically_normal().string())};
    models.install(key, native_model); models.activate(key);
    diamond_pipeline::ReplayStore replay(root(request) / "replay", compatibility,
        static_cast<std::size_t>(config.replay.capacity), config.replay.seed);
    diamond_pipeline::IterationRequest job;
    job.operation_id = operation; job.model_key = key; job.compatibility = compatibility; job.match = game_match(config);
    job.selfplay = {.lanes = static_cast<int>(config.workers.worker_count), .threads = static_cast<int>(config.workers.worker_count),
        .max_batch = static_cast<int>(config.inference.max_batch_size), .max_wait_us = static_cast<int>(config.inference.max_wait_ms * 1000),
        .simulations = static_cast<int>(config.mcts.simulations), .max_moves = static_cast<int>(config.self_play.max_moves),
        .temperature = config.self_play.temperature, .temperature_moves = static_cast<int>(config.self_play.temperature_moves),
        .dirichlet_alpha = config.mcts.dirichlet_alpha, .dirichlet_epsilon = config.mcts.dirichlet_epsilon};
    const auto start = opening(job.match);
    for (int64_t game = 0; game < config.workers.games_per_iteration; ++game)
        job.jobs.push_back({start, config.run_seed + static_cast<uint64_t>(game)});
    job.training_steps = 1; job.checkpoint_root = root(request) / "candidate-checkpoint";
    const auto result = diamond_pipeline::run_iteration(job, models, replay, trainer, {});
    write_json(root(request) / "iteration.json", Json{Object{{"operation_id", Json{result.operation_id}},
        {"completed_games", Json{static_cast<int64_t>(result.completed_games)}},
        {"aborted_games", Json{static_cast<int64_t>(result.aborted_games)}},
        {"training_step", Json{static_cast<int64_t>(result.training_step)}}}});
    return {result.completed_games + result.aborted_games, result.training_step};
}
Object train(const CommandRequest& request, const ProductionConfig& config, bool resume) {
    diamond_orchestration::RunStateStore store(request.runtime_dir / "runs");
    const auto compatibility = wire(config);
    const auto initial = resume ? store.load(request.model_name, request.run_id) :
        store.initialize(diamond_orchestration::RunState::initialize(request.run_id,
            Object{{"model_name", Json{compatibility.model_name}}, {"model_version", Json{compatibility.model_version}},
                   {"player_count", Json{static_cast<int64_t>(compatibility.player_count)}},
                   {"value_semantics_version", Json{compatibility.value_semantics_version}}},
            Object{{"pipeline", Json{"native-pipeline-v2"}},
                   {"rating", Json{request.model_name == "Soo" ? "soo-elo-v1" : "min-trueskill-v1"}}}, config.run_seed));
    Result result;
    diamond_orchestration::Coordinator coordinator(store, describe,
        [&](RunStage stage, const diamond_orchestration::RunState&, const std::string& operation) {
            write_json(root(request) / "stages" / (operation + ".json"),
                       Json{Object{{"operation_id", Json{operation}}, {"stage", Json{static_cast<int64_t>(stage)}}}});
            if (stage == RunStage::train) result = iterate(request, config, operation);
        });
    const auto complete = coordinator.run(initial);
    return {{"run_id", Json{complete.run_id()}}, {"stage", Json{"COMPLETE"}},
            {"completed_games", Json{static_cast<int64_t>(result.games)}}, {"training_step", Json{static_cast<int64_t>(result.step)}}};
}
Object evaluate(const CommandRequest&, const ProductionConfig&) {
    throw CommandArtifactError("candidate-versus-champion model loading is unavailable: checkpoint v2 stores no model weights");
}
Object report(const CommandRequest& request) {
    const auto path = root(request) / "rating-registry.json";
    const auto registry = diamond_orchestration::load_rating_registry(path);
    return {{"rating", registry.report_json()}, {"registry_path", Json{path.string()}}};
}
Object service(const CommandRequest& request, const ProductionConfig& config) {
    if (request.command == "train") return train(request, config, false);
    if (request.command == "resume") return train(request, config, true);
    if (request.command == "evaluate") return evaluate(request, config);
    return report(request);
}
}  // namespace

int main(int argc, char** argv) {
    return diamond_orchestration::dispatch_command(argc, argv, service, std::cout);
}
