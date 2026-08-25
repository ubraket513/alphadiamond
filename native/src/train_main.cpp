#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "diamond_model/soo_model.hpp"
#include "diamond_orchestration/arena.hpp"
#include "diamond_orchestration/coordinator.hpp"
#include "diamond_orchestration/rating_store.hpp"
#include "diamond_orchestration/report.hpp"
#include "diamond_orchestration/training_wiring.hpp"
#include "diamond_pipeline/pipeline.hpp"
#include "diamond_training/checkpoint.hpp"
#include "diamond_training/device.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/board.hpp"

namespace {
using Json = diamond_support::JsonValue;
using Array = Json::Array;
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
std::optional<std::chrono::steady_clock::duration> selfplay_deadline(
    const ProductionConfig& config) {
    if (!config.self_play.max_game_seconds || *config.self_play.max_game_seconds <= 0.0)
        return std::nullopt;
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(*config.self_play.max_game_seconds));
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
diamond_pipeline::ModelKey checkpoint_key(const ProductionConfig& config,
                                          const std::filesystem::path& checkpoint) {
    try {
        const auto info = diamond_training::inspect_checkpoint_v2(checkpoint);
        std::ifstream state(info.generation / "state.pt", std::ios::binary);
        if (!state) throw CommandArtifactError("cannot read checkpoint model state");
        const std::string bytes((std::istreambuf_iterator<char>(state)), {});
        if (bytes.empty()) throw CommandArtifactError("checkpoint model state is empty");
        return {config.model_name, config.model_version, diamond_support::sha256(bytes)};
    } catch (const CommandArtifactError&) {
        throw;
    } catch (const diamond_training::CheckpointError& error) {
        throw CommandArtifactError(error.what());
    }
}
soo::Match ordered_match(const ProductionConfig& config, const auto& turn_order) {
    auto match = game_match(config);
    const auto players = match.players;
    for (std::size_t seat = 0; seat < turn_order.size(); ++seat) {
        const auto found = std::find_if(players.begin(), players.begin() + match.count,
            [&](const soo::PlayerSpec& player) { return player.id == turn_order[seat]; });
        if (found == players.begin() + match.count)
            throw std::invalid_argument("arena turn order references an unknown player");
        match.players[seat] = *found;
    }
    return match;
}
soo::EpisodeConfig arena_episode_config(const ProductionConfig& config) {
    return {.lanes = 1,
            .threads = 1,
            .max_batch = 1,
            .max_wait_us = static_cast<int>(config.inference.max_wait_us),
            .simulations = static_cast<int>(config.mcts.simulations),
            .max_moves = static_cast<int>(config.arena.max_moves),
            .temperature = 0.0,
            .temperature_moves = 0,
            .dirichlet_alpha = config.mcts.dirichlet_alpha,
            .dirichlet_epsilon = 0.0};
}

class ArenaModelRouter final : public soo::BatchEvaluator {
  public:
    ArenaModelRouter(diamond_pipeline::ModelPool& candidate,
                     diamond_pipeline::ModelPool& champion, int candidate_player)
        : candidate_(candidate), champion_(champion), candidate_player_(candidate_player) {}

    void evaluate(std::vector<soo::BatchItem>& batch) override {
        std::vector<soo::BatchItem> candidate;
        std::vector<soo::BatchItem> champion;
        candidate.reserve(batch.size());
        champion.reserve(batch.size());
        for (const auto& item : batch) {
            if (!item.state) throw std::invalid_argument("arena evaluation requires a state");
            (item.state->current_player == candidate_player_ ? candidate : champion).push_back(item);
        }
        candidate_.evaluate(candidate);
        champion_.evaluate(champion);
    }

  private:
    diamond_pipeline::ModelPool& candidate_;
    diamond_pipeline::ModelPool& champion_;
    int candidate_player_;
};

soo::Episode play_arena_game(const ProductionConfig& config, const soo::Match& match,
                             uint64_t seed, int candidate_player,
                             diamond_pipeline::ModelPool& candidate,
                             diamond_pipeline::ModelPool& champion) {
    ArenaModelRouter evaluator(candidate, champion, candidate_player);
    soo::EpisodeMetrics metrics;
    auto episodes = soo::run_episodes(match, {{opening(match), seed}},
                                      arena_episode_config(config), evaluator, metrics);
    if (episodes.size() != 1) throw std::runtime_error("arena did not return exactly one game");
    return std::move(episodes.front());
}

diamond_orchestration::RatingRegistry rating_registry(
    const std::filesystem::path& path, const ProductionConfig& config) {
    if (std::filesystem::exists(path)) return diamond_orchestration::load_rating_registry(path);
    if (config.model_name == "Soo")
        return diamond_orchestration::RatingRegistry("soo-elo-v1");
    return diamond_orchestration::RatingRegistry("min-trueskill-v1",
                                                   diamond_orchestration::TrueSkillConfig{});
}
Json describe(RunStage stage, const diamond_orchestration::RunState& state) {
    return Json{Object{{"iteration", state.payload().at("iteration")},
                       {"stage", Json{static_cast<int64_t>(stage)}}}};
}
struct Result { std::size_t games = 0; uint64_t step = 0; };
Result iterate(const CommandRequest& request, const ProductionConfig& config, const std::string& operation) {
    const auto wiring = diamond_orchestration::wire_training_iteration(config);
    auto native_model = model(config); const auto compatibility = wire(config);
    diamond_training::Trainer trainer(native_model, compatibility, {config.training.learning_rate, config.training.weight_decay});
    try { (void)diamond_training::load_checkpoint_v2(request.checkpoint_path, trainer); }
    catch (const diamond_training::CheckpointError& error) { throw CommandArtifactError(error.what()); }
    diamond_pipeline::ModelPool models(1);
    const auto key = checkpoint_key(config, request.checkpoint_path);
    models.install(key, native_model); models.activate(key);
    diamond_pipeline::ReplayStore replay(root(request) / "replay", compatibility,
        static_cast<std::size_t>(config.replay.capacity), config.replay.seed);
    diamond_pipeline::IterationRequest job;
    job.operation_id = operation; job.model_key = key; job.compatibility = compatibility; job.match = game_match(config);
    job.selfplay = wiring.selfplay;
    job.selfplay.max_game_duration = selfplay_deadline(config);
    const auto start = opening(job.match);
    for (std::size_t game = 0; game < wiring.games_per_iteration; ++game)
        job.jobs.push_back({start, config.run_seed + static_cast<uint64_t>(game)});
    job.training_batch_size = wiring.training_batch_size;
    job.training_steps = wiring.training_steps;
    job.checkpoint_root = root(request) / "candidate-checkpoint";
    const auto result = diamond_pipeline::run_iteration(job, models, replay, trainer, {});
    Array training_batch_sizes;
    training_batch_sizes.reserve(result.training_batch_sizes.size());
    for (const auto batch_size : result.training_batch_sizes)
        training_batch_sizes.emplace_back(Json{static_cast<int64_t>(batch_size)});
    write_json(root(request) / "iteration.json", Json{Object{{"operation_id", Json{result.operation_id}},
        {"completed_games", Json{static_cast<int64_t>(result.completed_games)}},
        {"aborted_games", Json{static_cast<int64_t>(result.aborted_games)}},
        {"requested_training_steps", Json{static_cast<int64_t>(result.requested_training_steps)}},
        {"completed_training_steps", Json{static_cast<int64_t>(result.completed_training_steps)}},
        {"training_batch_sizes", Json{std::move(training_batch_sizes)}},
        {"replay_size", Json{static_cast<int64_t>(result.replay_size)}},
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
Object evaluate(const CommandRequest& request, const ProductionConfig& config) {
    const auto candidate_path = root(request) / "candidate-checkpoint";
    auto candidate_model = model(config);
    auto champion_model = model(config);
    diamond_pipeline::ModelPool candidate(1);
    diamond_pipeline::ModelPool champion(1);
    const auto candidate_key = checkpoint_key(config, candidate_path);
    const auto champion_key = checkpoint_key(config, request.checkpoint_path);
    try {
        candidate.install_checkpoint(candidate_key, candidate_path, candidate_model);
        champion.install_checkpoint(champion_key, request.checkpoint_path, champion_model);
        candidate.activate(candidate_key);
        champion.activate(champion_key);
    } catch (const std::exception& error) {
        throw CommandArtifactError(error.what());
    }

    const auto registry_path = root(request) / "rating-registry.json";
    auto registry = rating_registry(registry_path, config);
    const auto candidate_id = "candidate-" + candidate_key.checkpoint_sha256.substr(0, 16);
    const auto champion_id = "champion-" + champion_key.checkpoint_sha256.substr(0, 16);
    registry.add_participant(candidate_id, "Candidate");
    registry.add_participant(champion_id, "Champion");

    Object result;
    if (config.model_name == "Soo") {
        const auto outcomes = diamond_orchestration::execute_soo_arena_games(
            config.arena, [&](const diamond_orchestration::ArenaMatchup2& matchup,
                              std::size_t game) -> std::optional<bool> {
                const auto match = ordered_match(config, matchup.turn_order);
                const auto episode = play_arena_game(
                    config, match,
                    config.arena.seed + game * static_cast<uint64_t>(config.arena.max_moves),
                    matchup.candidate_player, candidate, champion);
                if (!episode.completed || episode.finish_order.empty()) return std::nullopt;
                return episode.finish_order.front() == matchup.candidate_player;
            });
        const auto summary = diamond_orchestration::summarize_soo_arena(outcomes, config.arena);
        const auto matchups = diamond_orchestration::balanced_soo_matchups();
        for (std::size_t game = 0; game < outcomes.size(); ++game) {
            if (!outcomes[game]) continue;
            const auto& matchup = matchups[game % matchups.size()];
            const std::array<std::string, 2> participants{candidate_id, champion_id};
            const std::array<int, 2> seats = matchup.candidate_player == 1
                ? std::array<int, 2>{1, 2} : std::array<int, 2>{2, 1};
            const auto& winner = *outcomes[game] ? candidate_id : champion_id;
            const auto& loser = *outcomes[game] ? champion_id : candidate_id;
            registry.record_event(diamond_orchestration::make_soo_rating_event(
                game, "soo-elo-v1", participants, seats, matchup.turn_order, "initial",
                true, winner, loser));
        }
        result = {{"aborted_games", Json{summary.aborted_games}},
                  {"losses", Json{summary.losses}},
                  {"promoted", Json{summary.promoted}},
                  {"rating_status", Json{"eligible"}},
                  {"win_rate", Json{summary.win_rate}},
                  {"wins", Json{summary.wins}}};
    } else {
        const auto outcomes = diamond_orchestration::execute_min_arena_games(
            config.arena, [&](const diamond_orchestration::ArenaMatchup3& matchup,
                              std::size_t game) -> std::optional<int> {
                const auto match = ordered_match(config, matchup.turn_order);
                const auto episode = play_arena_game(
                    config, match,
                    config.arena.seed + game * static_cast<uint64_t>(config.arena.max_moves),
                    matchup.candidate_player, candidate, champion);
                if (!episode.completed) return std::nullopt;
                const auto found = std::find(episode.finish_order.begin(), episode.finish_order.end(),
                                             matchup.candidate_player);
                if (found == episode.finish_order.end())
                    throw std::runtime_error("completed Min arena omitted the candidate");
                return static_cast<int>(std::distance(episode.finish_order.begin(), found));
            });
        const auto summary = diamond_orchestration::summarize_min_arena(outcomes, config.arena);
        result = {{"aborted_games", Json{summary.aborted_games}},
                  {"first_places", Json{summary.first_places}},
                  {"mean_utility", Json{summary.mean_utility}},
                  {"promoted", Json{summary.promoted}},
                  {"rating_status", Json{"insufficient_history"}},
                  {"second_places", Json{summary.second_places}},
                  {"third_places", Json{summary.third_places}}};
    }
    diamond_orchestration::save_rating_registry(registry_path, registry);
    result.emplace("candidate_checkpoint", Json{candidate_path.string()});
    result.emplace("candidate_sha256", Json{candidate_key.checkpoint_sha256});
    result.emplace("champion_checkpoint", Json{request.checkpoint_path.string()});
    result.emplace("champion_sha256", Json{champion_key.checkpoint_sha256});
    result.emplace("registry_path", Json{registry_path.string()});
    write_json(root(request) / "arena.json", Json{result});
    return result;
}
Object report(const CommandRequest& request) {
    const auto registry_path = root(request) / "rating-registry.json";
    const auto evaluation_path = root(request) / "arena.json";
    try {
        const auto registry = diamond_orchestration::load_rating_registry(registry_path);
        std::ifstream evaluation(evaluation_path, std::ios::binary);
        if (!evaluation) throw CommandArtifactError("cannot open evaluation report: " + evaluation_path.string());
        const std::string contents((std::istreambuf_iterator<char>(evaluation)), {});
        return {{"evaluation", diamond_support::parse_json(contents)},
                {"evaluation_path", Json{evaluation_path.string()}},
                {"rating", registry.report_json()},
                {"registry_path", Json{registry_path.string()}}};
    } catch (const CommandArtifactError&) {
        throw;
    } catch (const std::exception& error) {
        throw CommandArtifactError(error.what());
    }
}
Object service(const CommandRequest& request, const ProductionConfig& config) {
    // Reporting is read-only and must remain available when inspecting a run
    // created on a different device class than the current host.
    if (request.command == "report") return report(request);

    const auto resolved = [&] {
        try {
            return diamond_training::resolve_device(config.runtime.device);
        } catch (const diamond_training::DeviceResolutionError& error) {
            throw diamond_orchestration::CommandArgumentError(error.what());
        }
    }();

    // Task 1 establishes the preflight contract. The model, evaluator, and
    // trainer do not own CUDA tensors until the following CUDA-runtime tasks;
    // rejecting here prevents them from silently continuing on CPU.
    if (resolved.torch_device.is_cuda()) {
        throw diamond_orchestration::CommandArgumentError(
            "runtime.device " + resolved.canonical_name +
            " resolved successfully, but CUDA execution is not installed yet");
    }

    auto canonical_config = config;
    canonical_config.runtime.device = resolved.canonical_name;
    Object details;
    if (request.command == "train") details = train(request, canonical_config, false);
    else if (request.command == "resume") details = train(request, canonical_config, true);
    else details = evaluate(request, canonical_config);
    details.emplace("requested_device", Json{resolved.requested_name});
    details.emplace("canonical_device", Json{resolved.canonical_name});
    return details;
}
}  // namespace

int main(int argc, char** argv) {
    return diamond_orchestration::dispatch_command(argc, argv, service, std::cout);
}
