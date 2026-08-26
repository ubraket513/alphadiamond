#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_orchestration/arena.hpp"
#include "diamond_orchestration/coordinator.hpp"
#include "diamond_orchestration/rating_store.hpp"
#include "diamond_orchestration/report.hpp"
#include "diamond_orchestration/schedule.hpp"
#include "diamond_orchestration/training_wiring.hpp"
#include "diamond_pipeline/pipeline.hpp"
#include "diamond_training/checkpoint.hpp"
#include "diamond_training/device.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/board.hpp"
#include "soo/rules.hpp"
#include <torch/version.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
using Json = diamond_support::JsonValue;
using Array = Json::Array;
using Object = Json::Object;
using diamond_orchestration::CommandArtifactError;
using diamond_orchestration::CommandRequest;
using diamond_orchestration::ProductionConfig;
using diamond_orchestration::RunStage;
using diamond_orchestration::StageOutcome;

std::filesystem::path root(const CommandRequest& request) {
    return request.run_dir;
}
void write_json(const std::filesystem::path& path, Json value) {
    static std::atomic_uint64_t sequence{0};
    std::filesystem::create_directories(path.parent_path());
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto temporary =
        path.string() + ".tmp." + std::to_string(nonce) + "." + std::to_string(++sequence);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("cannot write native command artifact: " + path.string());
        output << diamond_support::canonical_json(value) << '\n';
        if (!output)
            throw std::runtime_error("cannot write native command artifact: " + path.string());
    }
#ifdef _WIN32
    if (!MoveFileExW(std::filesystem::path(temporary).wstring().c_str(), path.wstring().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot activate native command artifact: " + path.string());
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot activate native command artifact: " + path.string());
    }
#endif
}

Object read_object(const std::filesystem::path& path, const char* description) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw CommandArtifactError(std::string(description) + " is missing");
    try {
        const auto parsed =
            diamond_support::parse_json(std::string(std::istreambuf_iterator<char>(input), {}));
        return std::get<Object>(parsed.value);
    } catch (const std::exception& error) {
        throw CommandArtifactError(std::string(description) + " is invalid: " + error.what());
    }
}

std::string file_sha256(const std::filesystem::path& path, const char* description) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw CommandArtifactError(std::string(description) + " is missing");
    return diamond_support::sha256(std::string(std::istreambuf_iterator<char>(input), {}));
}

std::string source_commit() {
#ifdef DIAMOND_SOURCE_GIT_COMMIT
    return DIAMOND_SOURCE_GIT_COMMIT;
#else
    if (const char* value = std::getenv("GITHUB_SHA"); value && *value)
        return value;
    return "unavailable";
#endif
}

std::string creation_timestamp() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

Object protocol_ids(const ProductionConfig& config) {
    return {{"pipeline", Json{"native-pipeline-v2"}},
            {"rating", Json{config.model_name == "Soo" ? "soo-elo-v1" : "min-trueskill-v1"}},
            {"opening_suite", Json{config.opening_suite.id}}};
}

diamond_training::CheckpointProvenance
checkpoint_provenance(const ProductionConfig& config, const std::string& replay_manifest_sha256) {
#ifdef _WIN32
    constexpr const char* platform = "windows";
#elif defined(__APPLE__)
    constexpr const char* platform = "macos";
#else
    constexpr const char* platform = "linux";
#endif
#ifdef DIAMOND_SOURCE_DIRTY
    constexpr bool dirty = DIAMOND_SOURCE_DIRTY != 0;
#else
    constexpr bool dirty = false;
#endif
    const auto config_bytes = diamond_support::canonical_json(config.to_json());
    return {.source_git_commit = source_commit(),
            .resolved_config_bytes = config_bytes,
            .replay_manifest_sha256 = replay_manifest_sha256,
            .protocol_ids_json = diamond_support::canonical_json(Json{protocol_ids(config)}),
            .creation_timestamp = creation_timestamp(),
            .environment_json = diamond_support::canonical_json(Json{Object{
                {"dirty_tree", Json{dirty}},
                {"platform", Json{platform}},
                {"torch_threads", Json{static_cast<int64_t>(torch::get_num_threads())}},
                {"torch_version", Json{TORCH_VERSION}},
            }}),
            .rng_state_status = "gap_no_stable_libtorch_cpp_api",
            .rng_state_version = 0};
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
soo::State opening(const soo::Match& match,
                   const diamond_orchestration::MaterializedOpening& descriptor) {
    descriptor.validate();
    auto state = opening(match);
    std::mt19937_64 rng(descriptor.seed);
    for (int64_t ply = 0; ply < descriptor.depth; ++ply) {
        std::vector<int32_t> legal;
        soo::legal_action_ids(state, legal);
        if (legal.empty())
            throw CommandArtifactError("opening descriptor reached a terminal state early");
        std::uniform_int_distribution<std::size_t> choice(0, legal.size() - 1);
        state = soo::apply_action(state, match, legal[choice(rng)]);
    }
    return state;
}
uint64_t arena_seed(const diamond_orchestration::MaterializedOpening& opening,
                    const std::string& match_id) {
    const auto digest = diamond_support::sha256(diamond_support::canonical_json(Json{Object{
        {"match_id", Json{match_id}},
        {"opening_seed", Json{std::to_string(opening.seed)}},
    }}));
    return std::stoull(digest.substr(0, 16), nullptr, 16);
}
diamond_model::DiamondModel model(const ProductionConfig& config) {
    return diamond_model::DiamondModel(config.network.width, config.network.residual_blocks,
        config.model_name == "Soo" ? 4 : 6, config.model_name == "Soo" ? 1 : 3);
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
                             const soo::State& start, uint64_t seed, int candidate_player,
                             diamond_pipeline::ModelPool& candidate,
                             diamond_pipeline::ModelPool& champion) {
    ArenaModelRouter evaluator(candidate, champion, candidate_player);
    soo::EpisodeMetrics metrics;
    auto episodes =
        soo::run_episodes(match, {{start, seed}}, arena_episode_config(config), evaluator, metrics);
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
int64_t iteration_number(const diamond_orchestration::RunState& state) {
    return std::get<int64_t>(state.payload().at("iteration").value);
}

std::filesystem::path iteration_root(const CommandRequest& request, int64_t iteration) {
    return root(request) / "iterations" / std::to_string(iteration);
}

std::filesystem::path candidate_checkpoint(const CommandRequest& request, int64_t iteration) {
    return iteration_root(request, iteration) / "candidate-checkpoint";
}

const char* stage_label(RunStage stage) {
    switch (stage) {
    case RunStage::initialize:
        return "initialize";
    case RunStage::self_play:
        return "self-play";
    case RunStage::replay_ingest:
        return "replay-ingest";
    case RunStage::train:
        return "train";
    case RunStage::save_candidate:
        return "save-candidate";
    case RunStage::promotion_arena:
        return "promotion-arena";
    case RunStage::rating_benchmark:
        return "rating-benchmark";
    case RunStage::promote_or_reject:
        return "promote-or-reject";
    case RunStage::persist:
        return "persist";
    case RunStage::complete:
        return "complete";
    }
    throw CommandArtifactError("unknown pipeline stage");
}

std::filesystem::path stage_report_path(const CommandRequest& request, int64_t iteration,
                                        RunStage stage) {
    return iteration_root(request, iteration) / (std::string(stage_label(stage)) + ".json");
}

Array training_metrics_json(const std::vector<diamond_training::TrainingMetrics>& metrics) {
    Array values;
    values.reserve(metrics.size());
    for (const auto& metric : metrics)
        values.emplace_back(Json{Object{
            {"backward_seconds", Json{metric.backward_seconds}},
            {"collation_seconds", Json{metric.collation_seconds}},
            {"forward_seconds", Json{metric.forward_seconds}},
            {"h2d_seconds", Json{metric.h2d_seconds}},
            {"optimizer_seconds", Json{metric.optimizer_seconds}},
            {"peak_cuda_allocated_bytes",
             Json{static_cast<int64_t>(metric.peak_cuda_allocated_bytes)}},
            {"peak_cuda_memory_available", Json{metric.peak_cuda_memory_available}},
            {"peak_cuda_reserved_bytes",
             Json{static_cast<int64_t>(metric.peak_cuda_reserved_bytes)}},
            {"policy_loss", Json{metric.policy_loss}},
            {"samples_per_second", Json{metric.samples_per_second}},
            {"total_loss", Json{metric.total_loss}},
            {"total_step_seconds", Json{metric.total_step_seconds}},
            {"training_step", Json{static_cast<int64_t>(metric.training_step)}},
            {"value_loss", Json{metric.value_loss}},
        }});
    return values;
}

diamond_pipeline::IterationRequest iteration_job(const ProductionConfig& config,
                                                 const diamond_orchestration::RunState& state,
                                                 const std::string& operation,
                                                 const diamond_pipeline::ModelKey& key) {
    const auto iteration = iteration_number(state);
    const auto wiring = diamond_orchestration::wire_training_iteration(config);
    diamond_pipeline::IterationRequest job;
    job.operation_id = operation;
    job.model_key = key;
    job.compatibility = wire(config);
    job.match = game_match(config);
    job.selfplay = wiring.selfplay;
    job.selfplay.max_game_duration = selfplay_deadline(config);
    const auto start = opening(job.match);
    for (std::size_t game = 0; game < wiring.games_per_iteration; ++game) {
        const Array seed_identity{
            Json{std::string("SELF_PLAY")},   Json{iteration},
            Json{static_cast<int64_t>(game)}, Json{config.workers.retry_id},
            Json{config.opening_suite.id},
        };
        job.jobs.push_back({start, state.derive_seed(seed_identity)});
    }
    job.training_batch_size = wiring.training_batch_size;
    job.training_steps = wiring.training_steps;
    return job;
}

const char* initialization_name(diamond_orchestration::CommandInitialization mode) {
    using Mode = diamond_orchestration::CommandInitialization;
    switch (mode) {
    case Mode::scratch:
        return "scratch";
    case Mode::native_checkpoint:
        return "native_checkpoint";
    case Mode::warm_start:
        return "warm_start";
    case Mode::none:
        break;
    }
    throw CommandArtifactError("training initialization mode is missing");
}

void persist_initialization(const CommandRequest& request) {
    Json source{nullptr};
    if (request.initialization == diamond_orchestration::CommandInitialization::native_checkpoint)
        source =
            Json{std::filesystem::absolute(request.checkpoint_path).lexically_normal().string()};
    else if (request.initialization == diamond_orchestration::CommandInitialization::warm_start)
        source =
            Json{std::filesystem::absolute(request.warm_start_path).lexically_normal().string()};
    write_json(root(request) / "initialization.json",
               Json{Object{
                   {"mode", Json{initialization_name(request.initialization)}},
                   {"schema_version", Json{int64_t{1}}},
                   {"source_path", std::move(source)},
               }});
}

CommandRequest authoritative_request(const CommandRequest& request) {
    if (request.initialization != diamond_orchestration::CommandInitialization::none)
        return request;
    std::ifstream input(root(request) / "initialization.json", std::ios::binary);
    if (!input)
        throw CommandArtifactError("run initialization metadata is missing");
    try {
        const auto parsed =
            diamond_support::parse_json(std::string(std::istreambuf_iterator<char>(input), {}));
        const auto& object = std::get<Object>(parsed.value);
        if (object.size() != 3 || std::get<int64_t>(object.at("schema_version").value) != 1)
            throw CommandArtifactError("run initialization metadata is invalid");
        auto result = request;
        const auto mode = std::get<std::string>(object.at("mode").value);
        const auto* source = std::get_if<std::string>(&object.at("source_path").value);
        if (mode == "scratch" && !source)
            result.initialization = diamond_orchestration::CommandInitialization::scratch;
        else if (mode == "native_checkpoint" && source && !source->empty()) {
            result.initialization = diamond_orchestration::CommandInitialization::native_checkpoint;
            result.checkpoint_path = *source;
        } else if (mode == "warm_start" && source && !source->empty()) {
            result.initialization = diamond_orchestration::CommandInitialization::warm_start;
            result.warm_start_path = *source;
        } else {
            throw CommandArtifactError("run initialization metadata is inconsistent");
        }
        return result;
    } catch (const CommandArtifactError&) {
        throw;
    } catch (const std::exception& error) {
        throw CommandArtifactError("run initialization metadata is invalid: " +
                                   std::string(error.what()));
    }
}

std::optional<uint64_t> deployment_training_step(const std::filesystem::path& artifact) {
    std::ifstream input(artifact / "metadata.json", std::ios::binary);
    if (!input)
        throw CommandArtifactError("deployment metadata is missing");
    const auto parsed =
        diamond_support::parse_json(std::string(std::istreambuf_iterator<char>(input), {}));
    const auto& metadata = std::get<Object>(parsed.value);
    const auto& source = std::get<Object>(metadata.at("source").value);
    const auto& value = source.at("training_step").value;
    if (std::holds_alternative<std::nullptr_t>(value))
        return std::nullopt;
    const auto step = std::get<int64_t>(value);
    if (step < 0)
        throw CommandArtifactError("deployment source training_step is invalid");
    return static_cast<uint64_t>(step);
}

struct IterationSource {
    uint64_t training_step = 0;
    std::string model_digest;
    diamond_training::CheckpointInitializationMode mode =
        diamond_training::CheckpointInitializationMode::scratch;
    std::optional<uint64_t> source_training_step;
    bool optimizer_restored = false;
    bool optimizer_reset = false;
    std::optional<std::string> optimizer_reset_reason;
};

void validate_checkpoint_context(const diamond_training::CheckpointInfo& saved,
                                 const CommandRequest& request, const ProductionConfig& config,
                                 std::optional<uint64_t> expected_iteration = std::nullopt,
                                 std::optional<std::string_view> replay_sha256 = std::nullopt) {
    if (saved.format_version != 3 || !saved.lineage || !saved.provenance)
        throw CommandArtifactError("exact continuation requires a checkpoint v3 manifest");
    if (saved.lineage->run_id != request.run_id)
        throw CommandArtifactError("native checkpoint run identity does not match --run-dir");
    if (expected_iteration && saved.lineage->iteration != *expected_iteration)
        throw CommandArtifactError(
            "native checkpoint iteration does not match the durable run stage");
    const auto expected_config = diamond_support::canonical_json(config.to_json());
    const auto expected_protocols = diamond_support::canonical_json(Json{protocol_ids(config)});
    if (saved.provenance->resolved_config_bytes != expected_config ||
        saved.provenance->protocol_ids_json != expected_protocols) {
        throw CommandArtifactError("native checkpoint config or protocol provenance mismatch");
    }
    if (replay_sha256 && saved.provenance->replay_manifest_sha256 != *replay_sha256)
        throw CommandArtifactError("native checkpoint replay provenance mismatch");
}

IterationSource load_iteration_source(const CommandRequest& request, const ProductionConfig& config,
                                      int64_t iteration, diamond_training::Trainer& trainer,
                                      const diamond_training::ResolvedDevice& device) {
    try {
        if (iteration != 0) {
            const auto source = candidate_checkpoint(request, iteration - 1);
            if (!std::filesystem::exists(source / "CURRENT"))
                throw CommandArtifactError("previous iteration candidate checkpoint is missing");
            validate_checkpoint_context(diamond_training::inspect_checkpoint_v2(source), request,
                                        config);
            const auto saved = diamond_training::load_checkpoint_v3(
                source, trainer, device, diamond_training::CheckpointLoadIntent::exact_resume);
            validate_checkpoint_context(saved, request, config);
            return {.training_step = saved.training_step,
                    .model_digest = saved.model_digest,
                    .mode = diamond_training::CheckpointInitializationMode::resume,
                    .source_training_step = saved.training_step,
                    .optimizer_restored = true};
        }
        if (request.initialization == diamond_orchestration::CommandInitialization::scratch) {
            return {.training_step = 0,
                    .model_digest = diamond_training::canonical_model_digest(trainer.model()),
                    .mode = diamond_training::CheckpointInitializationMode::scratch};
        }
        if (request.initialization == diamond_orchestration::CommandInitialization::warm_start) {
            const auto expected_family = request.model_name == "Soo" ? "soo" : "min";
            const auto artifact = diamond_model::validate_deployment_artifact(
                request.warm_start_path, expected_family);
            if (artifact.model_version != trainer.compatibility().model_version ||
                artifact.width != trainer.model()->width() ||
                artifact.residual_blocks != trainer.model()->residual_blocks() ||
                artifact.input_features != trainer.model()->input_features() ||
                artifact.value_size != trainer.model()->value_size()) {
                throw CommandArtifactError(
                    "warm-start artifact is incompatible with resolved config");
            }
            trainer.model()->load_weights(artifact.weights);
            return {.training_step = 0,
                    .model_digest = artifact.runtime_sha256,
                    .mode = diamond_training::CheckpointInitializationMode::warm_start,
                    .source_training_step = deployment_training_step(request.warm_start_path),
                    .optimizer_reset = true,
                    .optimizer_reset_reason = "deployment_weight_warm_start"};
        }
        if (request.initialization ==
            diamond_orchestration::CommandInitialization::native_checkpoint) {
            validate_checkpoint_context(
                diamond_training::inspect_checkpoint_v2(request.checkpoint_path), request, config);
            const auto saved = diamond_training::load_checkpoint_v3(
                request.checkpoint_path, trainer, device,
                diamond_training::CheckpointLoadIntent::exact_resume);
            validate_checkpoint_context(saved, request, config);
            return {.training_step = saved.training_step,
                    .model_digest = saved.model_digest,
                    .mode = diamond_training::CheckpointInitializationMode::resume,
                    .source_training_step = saved.training_step,
                    .optimizer_restored = true};
        }
        throw CommandArtifactError("training initialization mode is missing");
    } catch (const diamond_training::CheckpointError& error) {
        throw CommandArtifactError(error.what());
    } catch (const CommandArtifactError&) {
        throw;
    } catch (const std::exception& error) {
        throw CommandArtifactError(error.what());
    }
}

diamond_training::CheckpointLineage checkpoint_lineage(const CommandRequest& request,
                                                       int64_t iteration, uint64_t model_step,
                                                       const IterationSource& source) {
    diamond_training::CheckpointLineage lineage{
        .initialization_mode = source.mode,
        .run_id = request.run_id,
        .iteration = static_cast<uint64_t>(iteration),
        .model_step = model_step,
        .source_training_step = source.source_training_step,
        .optimizer_restored = source.optimizer_restored,
        .optimizer_reset = source.optimizer_reset,
        .optimizer_reset_reason = source.optimizer_reset_reason,
    };
    if (source.mode == diamond_training::CheckpointInitializationMode::resume)
        lineage.parent_digest = source.model_digest;
    if (source.mode != diamond_training::CheckpointInitializationMode::scratch)
        lineage.source_digest = source.model_digest;
    return lineage;
}

diamond_training::CheckpointInfo
load_actor_checkpoint(const CommandRequest& request, const ProductionConfig& config,
                      const diamond_orchestration::RunState& state,
                      diamond_training::Trainer& trainer,
                      const diamond_training::ResolvedDevice& device) {
    const auto& value = state.payload().at("champion_checkpoint").value;
    const auto* path = std::get_if<std::string>(&value);
    if (!path || path->empty())
        throw CommandArtifactError("champion checkpoint is not active");
    try {
        validate_checkpoint_context(diamond_training::inspect_checkpoint_v2(*path), request,
                                    config);
        const auto saved = diamond_training::load_checkpoint_v3(
            *path, trainer, device, diamond_training::CheckpointLoadIntent::exact_resume);
        validate_checkpoint_context(saved, request, config);
        return saved;
    } catch (const diamond_training::CheckpointError& error) {
        throw CommandArtifactError(error.what());
    }
}

StageOutcome stage_report(const CommandRequest& request, int64_t iteration, RunStage stage,
                          const std::string& operation, Object details = {}, Object progress = {}) {
    details.emplace("operation_id", Json{operation});
    details.emplace("iteration", Json{iteration});
    details.emplace("stage", Json{stage_label(stage)});
    details.try_emplace("completed_games", Json{int64_t{0}});
    details.try_emplace("aborted_games", Json{int64_t{0}});
    details.try_emplace("requested_training_steps", Json{int64_t{0}});
    details.try_emplace("completed_training_steps", Json{int64_t{0}});
    details.try_emplace("training_metrics", Json{Array{}});
    details.emplace("run_progress", Json{progress});
    Json result{std::move(details)};
    write_json(stage_report_path(request, iteration, stage), result);
    return {.result = std::move(result), .progress = std::move(progress)};
}

StageOutcome load_stage_report(const CommandRequest& request, int64_t iteration, RunStage stage,
                               const std::string& operation) {
    std::ifstream input(stage_report_path(request, iteration, stage), std::ios::binary);
    if (!input)
        throw CommandArtifactError("persisted stage report is missing");
    try {
        auto result =
            diamond_support::parse_json(std::string(std::istreambuf_iterator<char>(input), {}));
        const auto& object = std::get<Object>(result.value);
        if (std::get<std::string>(object.at("operation_id").value) != operation ||
            std::get<int64_t>(object.at("iteration").value) != iteration ||
            std::get<std::string>(object.at("stage").value) != stage_label(stage)) {
            throw CommandArtifactError("persisted stage report identity mismatch");
        }
        const auto progress = std::get<Object>(object.at("run_progress").value);
        return {.result = std::move(result), .progress = progress};
    } catch (const CommandArtifactError&) {
        throw;
    } catch (const std::exception& error) {
        throw CommandArtifactError("persisted stage report is invalid: " +
                                   std::string(error.what()));
    }
}

Object evaluate(const CommandRequest& request, const ProductionConfig& config,
                const diamond_training::ResolvedDevice& device);

StageOutcome execute_stage(const CommandRequest& request, const ProductionConfig& config,
                           const diamond_orchestration::RunState& state, RunStage stage,
                           const std::string& operation,
                           const diamond_training::ResolvedDevice& device) {
    const auto iteration = iteration_number(state);
    const auto report_path = stage_report_path(request, iteration, stage);
    if (std::filesystem::exists(report_path))
        return load_stage_report(request, iteration, stage, operation);

    const auto compatibility = wire(config);
    torch::manual_seed(static_cast<int64_t>(config.training.seed));
    const auto per_iteration = iteration_root(request, iteration);
    const auto episodes_path = per_iteration / "selfplay.episodes";
    if (stage == RunStage::initialize) {
        auto native_model = model(config);
        diamond_training::Trainer trainer(
            native_model, compatibility,
            {config.training.learning_rate, config.training.weight_decay}, device);
        const auto source = load_iteration_source(request, config, iteration, trainer, device);
        std::filesystem::path champion;
        diamond_training::CheckpointInfo saved;
        if (iteration == 0 && request.initialization ==
                                  diamond_orchestration::CommandInitialization::native_checkpoint) {
            champion = request.checkpoint_path;
            saved = diamond_training::inspect_checkpoint_v2(champion);
        } else if (iteration > 0) {
            const auto* active =
                std::get_if<std::string>(&state.payload().at("champion_checkpoint").value);
            if (!active || active->empty())
                throw CommandArtifactError("previous champion checkpoint is missing");
            champion = *active;
            saved = diamond_training::inspect_checkpoint_v2(champion);
        } else {
            champion = root(request) / "initial-champion-checkpoint";
            if (std::filesystem::exists(champion / "CURRENT")) {
                saved = diamond_training::inspect_checkpoint_v2(champion);
            } else {
                saved = diamond_training::save_checkpoint_v3(
                    champion, trainer,
                    checkpoint_lineage(request, iteration, trainer.training_step(), source),
                    checkpoint_provenance(config, std::string(64, '0')));
            }
        }
        validate_checkpoint_context(saved, request, config);
        return stage_report(
            request, iteration, stage, operation,
            {
                {"champion_checkpoint", Json{champion.string()}},
                {"champion_sha256", Json{saved.model_digest}},
                {"initialization_mode", Json{initialization_name(request.initialization)}},
                {"status", Json{"completed"}},
            },
            {{"champion_checkpoint", Json{champion.string()}},
             {"champion_model_key", Json{compatibility.model_name + ":" +
                                         compatibility.model_version + ":" + saved.model_digest}}});
    }
    if (stage == RunStage::self_play) {
        if (!std::filesystem::exists(episodes_path)) {
            auto native_model = model(config);
            diamond_training::Trainer trainer(
                native_model, compatibility,
                {config.training.learning_rate, config.training.weight_decay}, device);
            (void)load_actor_checkpoint(request, config, state, trainer, device);
            diamond_pipeline::ModelPool models(1, device);
            const auto key = models.install(compatibility, trainer.learner());
            models.activate(key);
            const auto result = diamond_pipeline::run_self_play(
                iteration_job(config, state, operation, key), models, {});
            diamond_pipeline::save_episode_artifact(episodes_path, operation, result.episodes);
        }
        const auto episodes =
            diamond_pipeline::load_episode_artifact(episodes_path, operation, compatibility);
        std::size_t complete = 0, aborted = 0, samples = 0, total_moves = 0;
        Object abort_reason_distribution;
        Array completed_ids;
        for (const auto& episode : episodes) {
            complete += episode.completed;
            aborted += !episode.completed;
            samples += episode.samples.size();
            total_moves += static_cast<std::size_t>(episode.move_count);
            if (episode.completed)
                completed_ids.emplace_back(Json{episode.game_id});
            else {
                const auto reason = episode.aborted_reason.empty() ? std::string{"unspecified"}
                                                                   : episode.aborted_reason;
                auto [found, inserted] =
                    abort_reason_distribution.try_emplace(reason, Json{int64_t{0}});
                (void)inserted;
                ++std::get<int64_t>(found->second.value);
            }
        }
        return stage_report(
            request, iteration, stage, operation,
            {
                {"requested_games", Json{static_cast<int64_t>(episodes.size())}},
                {"completed_games", Json{static_cast<int64_t>(complete)}},
                {"aborted_games", Json{static_cast<int64_t>(aborted)}},
                {"abort_reason_distribution", Json{std::move(abort_reason_distribution)}},
                {"new_samples", Json{static_cast<int64_t>(samples)}},
                {"total_moves", Json{static_cast<int64_t>(total_moves)}},
            },
            {{"completed_game_ids", Json{std::move(completed_ids)}}});
    }
    if (stage == RunStage::replay_ingest) {
        const auto self_play_report = read_object(
            stage_report_path(request, iteration, RunStage::self_play), "self-play stage report");
        const auto self_play_operation =
            std::get<std::string>(self_play_report.at("operation_id").value);
        const auto episodes = diamond_pipeline::load_episode_artifact(
            episodes_path, self_play_operation, compatibility);
        diamond_pipeline::ReplayStore replay(root(request) / "replay", compatibility,
                                             static_cast<std::size_t>(config.replay.capacity),
                                             config.replay.seed);
        const auto ingested = diamond_pipeline::ingest_self_play(replay, episodes);
        std::size_t complete = 0, aborted = 0;
        for (const auto& episode : episodes) {
            complete += episode.completed;
            aborted += !episode.completed;
        }
        const auto replay_manifest = replay.manifest_path().string();
        const auto replay_digest = replay.manifest_digest();
        return stage_report(
            request, iteration, stage, operation,
            {
                {"requested_games", Json{static_cast<int64_t>(episodes.size())}},
                {"completed_games", Json{static_cast<int64_t>(complete)}},
                {"aborted_games", Json{static_cast<int64_t>(aborted)}},
                {"accepted_games", Json{static_cast<int64_t>(ingested.accepted_games)}},
                {"duplicate_games", Json{static_cast<int64_t>(ingested.duplicate_games)}},
                {"accepted_samples", Json{static_cast<int64_t>(ingested.accepted_samples)}},
                {"duplicate_samples", Json{static_cast<int64_t>(ingested.duplicate_samples)}},
                {"replay_size", Json{static_cast<int64_t>(replay.size())}},
                {"replay_manifest", Json{replay_manifest}},
                {"replay_manifest_sha256", Json{replay_digest}},
            },
            {{"replay_manifest", Json{replay_manifest}}});
    }
    if (stage == RunStage::train) {
        const auto staged = per_iteration / "trained-checkpoint";
        diamond_pipeline::TrainingResult trained;
        IterationSource source;
        auto native_model = model(config);
        diamond_training::Trainer trainer(
            native_model, compatibility,
            {config.training.learning_rate, config.training.weight_decay}, device);
        source = load_iteration_source(request, config, iteration, trainer, device);
        diamond_pipeline::ReplayStore replay(root(request) / "replay", compatibility,
                                             static_cast<std::size_t>(config.replay.capacity),
                                             config.replay.seed);
        if (!std::filesystem::exists(staged / "CURRENT")) {
            const auto key = diamond_pipeline::ModelKey{
                compatibility.model_name, compatibility.model_version,
                diamond_training::canonical_model_digest(trainer.learner())};
            trained = diamond_pipeline::train_replay(iteration_job(config, state, operation, key),
                                                     replay, trainer, {});
            try {
                (void)diamond_training::save_checkpoint_v3(
                    staged, trainer,
                    checkpoint_lineage(request, iteration, trainer.training_step(), source),
                    checkpoint_provenance(config, replay.manifest_digest()));
            } catch (const diamond_training::CheckpointError& error) {
                throw CommandArtifactError(error.what());
            }
        } else {
            try {
                diamond_pipeline::ReplayStore replay(
                    root(request) / "replay", compatibility,
                    static_cast<std::size_t>(config.replay.capacity), config.replay.seed);
                validate_checkpoint_context(diamond_training::inspect_checkpoint_v2(staged),
                                            request, config, static_cast<uint64_t>(iteration),
                                            replay.manifest_digest());
                const auto saved = diamond_training::load_checkpoint_v3(
                    staged, trainer, device, diamond_training::CheckpointLoadIntent::exact_resume);
                validate_checkpoint_context(saved, request, config,
                                            static_cast<uint64_t>(iteration),
                                            replay.manifest_digest());
                trained.operation_id = operation;
                trained.requested_training_steps =
                    static_cast<std::size_t>(config.training.train_steps_per_iteration);
                trained.completed_training_steps =
                    static_cast<std::size_t>(saved.training_step - source.training_step);
                trained.replay_size = replay.size();
                trained.training_step = saved.training_step;
            } catch (const diamond_training::CheckpointError& error) {
                throw CommandArtifactError(error.what());
            }
        }
        Array batch_sizes;
        for (const auto value : trained.training_batch_sizes)
            batch_sizes.emplace_back(Json{static_cast<int64_t>(value)});
        return stage_report(
            request, iteration, stage, operation,
            {
                {"requested_training_steps",
                 Json{static_cast<int64_t>(trained.requested_training_steps)}},
                {"completed_training_steps",
                 Json{static_cast<int64_t>(trained.completed_training_steps)}},
                {"training_batch_sizes", Json{std::move(batch_sizes)}},
                {"training_metrics", Json{training_metrics_json(trained.training_metrics)}},
                {"replay_size", Json{static_cast<int64_t>(trained.replay_size)}},
                {"training_step", Json{static_cast<int64_t>(trained.training_step)}},
                {"trained_checkpoint", Json{staged.string()}},
            },
            {{"training_step", Json{static_cast<int64_t>(trained.training_step)}}});
    }
    if (stage == RunStage::save_candidate) {
        const auto staged = per_iteration / "trained-checkpoint";
        auto native_model = model(config);
        diamond_training::Trainer trainer(
            native_model, compatibility,
            {config.training.learning_rate, config.training.weight_decay}, device);
        diamond_training::CheckpointInfo saved;
        try {
            diamond_pipeline::ReplayStore replay(root(request) / "replay", compatibility,
                                                 static_cast<std::size_t>(config.replay.capacity),
                                                 config.replay.seed);
            validate_checkpoint_context(diamond_training::inspect_checkpoint_v2(staged), request,
                                        config, static_cast<uint64_t>(iteration),
                                        replay.manifest_digest());
            saved = diamond_training::load_checkpoint_v3(
                staged, trainer, device, diamond_training::CheckpointLoadIntent::exact_resume);
            validate_checkpoint_context(saved, request, config, static_cast<uint64_t>(iteration),
                                        replay.manifest_digest());
        } catch (const diamond_training::CheckpointError& error) {
            throw CommandArtifactError(error.what());
        }
        if (!saved.lineage || !saved.provenance ||
            saved.lineage->iteration != static_cast<uint64_t>(iteration) ||
            saved.lineage->model_step != saved.training_step)
            throw CommandArtifactError("trained candidate checkpoint v3 lineage is invalid");
        const auto candidate = candidate_checkpoint(request, iteration);
        try {
            if (std::filesystem::exists(candidate / "CURRENT")) {
                const auto active = diamond_training::inspect_checkpoint_v2(candidate);
                if (active.format_version != 3 || !active.lineage ||
                    active.lineage->iteration != static_cast<uint64_t>(iteration) ||
                    active.training_step != saved.training_step) {
                    saved = diamond_training::save_checkpoint_v3(candidate, trainer, *saved.lineage,
                                                                 *saved.provenance);
                } else {
                    saved = active;
                }
            } else {
                saved = diamond_training::save_checkpoint_v3(candidate, trainer, *saved.lineage,
                                                             *saved.provenance);
            }
        } catch (const diamond_training::CheckpointError& error) {
            throw CommandArtifactError(error.what());
        }
        return stage_report(request, iteration, stage, operation,
                            {
                                {"candidate_checkpoint", Json{candidate.string()}},
                                {"candidate_sha256", Json{saved.model_digest}},
                                {"training_step", Json{static_cast<int64_t>(saved.training_step)}},
                                {"lineage_iteration", Json{iteration}},
                                {"lineage_valid", Json{true}},
                            },
                            {{"candidate_checkpoint", Json{candidate.string()}},
                             {"training_step", Json{static_cast<int64_t>(saved.training_step)}}});
    }
    if (stage == RunStage::promotion_arena) {
        const auto* candidate =
            std::get_if<std::string>(&state.payload().at("candidate_checkpoint").value);
        const auto* champion =
            std::get_if<std::string>(&state.payload().at("champion_checkpoint").value);
        if (!candidate || candidate->empty() || !champion || champion->empty())
            throw CommandArtifactError(
                "promotion arena requires candidate and champion checkpoints");
        auto evaluation_request = request;
        evaluation_request.candidate_path = *candidate;
        evaluation_request.champion_path = *champion;
        evaluation_request.opening_suite_id = config.opening_suite.id;
        auto result = evaluate(evaluation_request, config, device);
        result.emplace("arena_path", Json{(root(request) / "arena.json").string()});
        result.emplace("arena_sha256",
                       Json{file_sha256(root(request) / "arena.json", "promotion arena report")});
        result.emplace("status", Json{"completed"});
        return stage_report(request, iteration, stage, operation, std::move(result));
    }
    if (stage == RunStage::rating_benchmark) {
        const auto arena_path = stage_report_path(request, iteration, RunStage::promotion_arena);
        const auto arena = read_object(arena_path, "promotion arena stage report");
        const auto registry_path = root(request) / "rating-registry.json";
        const auto registry = diamond_orchestration::load_rating_registry(registry_path);
        auto records = std::get<Array>(state.payload().at("rating_records").value);
        Object record{
            {"arena_report_sha256", Json{file_sha256(arena_path, "promotion arena stage report")}},
            {"candidate_sha256", arena.at("candidate_sha256")},
            {"champion_sha256", arena.at("champion_sha256")},
            {"iteration", Json{iteration}},
            {"protocol_id", Json{config.model_name == "Soo" ? "soo-elo-v1" : "min-trueskill-v1"}},
            {"registry_path", Json{registry_path.string()}},
            {"registry_sha256", Json{file_sha256(registry_path, "rating registry")}},
            {"registry_summary", Json{registry.report_json()}},
        };
        if (const auto found = arena.find("candidate_runtime_sha256"); found != arena.end())
            record.emplace("candidate_runtime_sha256", found->second);
        if (const auto found = arena.find("champion_runtime_sha256"); found != arena.end())
            record.emplace("champion_runtime_sha256", found->second);
        records.emplace_back(Json{record});
        return stage_report(request, iteration, stage, operation,
                            {
                                {"rating_record", Json{record}},
                                {"status", Json{"completed"}},
                            },
                            {{"rating_records", Json{std::move(records)}}});
    }
    if (stage == RunStage::promote_or_reject) {
        const auto arena_path = stage_report_path(request, iteration, RunStage::promotion_arena);
        const auto arena = read_object(arena_path, "promotion arena stage report");
        const auto candidate_path =
            std::filesystem::path(std::get<std::string>(arena.at("candidate_checkpoint").value));
        const auto champion_path =
            std::filesystem::path(std::get<std::string>(arena.at("champion_checkpoint").value));
        const auto candidate = diamond_training::inspect_checkpoint_v2(candidate_path);
        const auto champion = diamond_training::inspect_checkpoint_v2(champion_path);
        const auto* durable_candidate =
            std::get_if<std::string>(&state.payload().at("candidate_checkpoint").value);
        const auto* durable_champion =
            std::get_if<std::string>(&state.payload().at("champion_checkpoint").value);
        if (!durable_candidate || !durable_champion ||
            candidate_path.lexically_normal() !=
                std::filesystem::path(*durable_candidate).lexically_normal() ||
            champion_path.lexically_normal() !=
                std::filesystem::path(*durable_champion).lexically_normal()) {
            throw CommandArtifactError("promotion checkpoint identity or lineage mismatch");
        }
        validate_checkpoint_context(candidate, request, config, static_cast<uint64_t>(iteration));
        validate_checkpoint_context(champion, request, config);

        const auto arena_report_sha256 = file_sha256(arena_path, "promotion arena stage report");
        const auto& ratings = std::get<Array>(state.payload().at("rating_records").value);
        if (ratings.empty())
            throw CommandArtifactError("promotion requires a durable rating record");
        const auto& rating = std::get<Object>(ratings.back().value);
        if (std::get<std::string>(rating.at("arena_report_sha256").value) != arena_report_sha256) {
            throw CommandArtifactError(
                "promotion arena report no longer matches its rating record");
        }
        const auto arena_result_path =
            std::filesystem::path(std::get<std::string>(arena.at("arena_path").value));
        if (file_sha256(arena_result_path, "promotion arena result") !=
            std::get<std::string>(arena.at("arena_sha256").value)) {
            throw CommandArtifactError("promotion arena result digest mismatch");
        }

        const auto arena_candidate_sha256 =
            std::get<std::string>(arena.at("candidate_sha256").value);
        const auto arena_champion_sha256 = std::get<std::string>(arena.at("champion_sha256").value);
        std::string candidate_runtime_sha256;
        std::string champion_runtime_sha256;
        std::string identity_schema = "checkpoint-model-and-runtime-v1";
        const auto candidate_runtime = arena.find("candidate_runtime_sha256");
        const auto champion_runtime = arena.find("champion_runtime_sha256");
        if (candidate_runtime != arena.end() && champion_runtime != arena.end()) {
            if (candidate.model_digest != arena_candidate_sha256 ||
                champion.model_digest != arena_champion_sha256) {
                throw CommandArtifactError("promotion checkpoint model digest mismatch");
            }
            candidate_runtime_sha256 = std::get<std::string>(candidate_runtime->second.value);
            champion_runtime_sha256 = std::get<std::string>(champion_runtime->second.value);
        } else if (candidate_runtime == arena.end() && champion_runtime == arena.end()) {
            // Early native reports used candidate_sha256/champion_sha256 for the
            // ModelPool runtime identity. Recompute those identities so an
            // already checksummed run can resume without weakening validation.
            auto candidate_model = model(config);
            auto champion_model = model(config);
            diamond_pipeline::ModelPool candidate_pool(1, device);
            diamond_pipeline::ModelPool champion_pool(1, device);
            try {
                candidate_runtime_sha256 =
                    candidate_pool
                        .install_checkpoint(compatibility, candidate_path, candidate_model)
                        .checkpoint_sha256;
                champion_runtime_sha256 =
                    champion_pool.install_checkpoint(compatibility, champion_path, champion_model)
                        .checkpoint_sha256;
            } catch (const std::exception& error) {
                throw CommandArtifactError(error.what());
            }
            if (candidate_runtime_sha256 != arena_candidate_sha256 ||
                champion_runtime_sha256 != arena_champion_sha256) {
                throw CommandArtifactError("legacy promotion runtime digest mismatch");
            }
            identity_schema = "legacy-runtime-v0";
        } else {
            throw CommandArtifactError("promotion arena identity fields are incomplete");
        }
        const bool promoted = std::get<bool>(arena.at("promoted").value);
        const auto incomplete_blocks = std::get<int64_t>(arena.at("incomplete_blocks").value);
        if (promoted && incomplete_blocks != 0)
            throw CommandArtifactError(
                "an incomplete paired-opening block cannot promote a candidate");
        const auto active_path = promoted ? candidate_path : champion_path;
        const auto& active = promoted ? candidate : champion;
        auto records = std::get<Array>(state.payload().at("promotion_records").value);
        const Object record{
            {"arena_report_sha256", Json{arena_report_sha256}},
            {"candidate_checkpoint", Json{candidate_path.string()}},
            {"candidate_runtime_sha256", Json{candidate_runtime_sha256}},
            {"candidate_sha256", Json{candidate.model_digest}},
            {"champion_checkpoint_before", Json{champion_path.string()}},
            {"champion_checkpoint_after", Json{active_path.string()}},
            {"champion_runtime_sha256_before", Json{champion_runtime_sha256}},
            {"champion_sha256_after", Json{active.model_digest}},
            {"decision", Json{promoted ? "promote" : "reject"}},
            {"identity_schema", Json{identity_schema}},
            {"incomplete_blocks", Json{incomplete_blocks}},
            {"iteration", Json{iteration}},
            {"promotion_reason", arena.at("promotion_reason")},
        };
        records.emplace_back(Json{record});
        const auto champion_key = compatibility.model_name + ":" + compatibility.model_version +
                                  ":" + active.model_digest;
        return stage_report(request, iteration, stage, operation,
                            {
                                {"champion_checkpoint", Json{active_path.string()}},
                                {"champion_model_key", Json{champion_key}},
                                {"decision", Json{promoted ? "promote" : "reject"}},
                                {"promotion_record", Json{record}},
                                {"status", Json{"completed"}},
                            },
                            {{"champion_checkpoint", Json{active_path.string()}},
                             {"champion_model_key", Json{champion_key}},
                             {"promotion_records", Json{std::move(records)}}});
    }
    if (stage == RunStage::persist) {
        const auto& promotions = std::get<Array>(state.payload().at("promotion_records").value);
        const auto& ratings = std::get<Array>(state.payload().at("rating_records").value);
        if (promotions.empty() || ratings.empty())
            throw CommandArtifactError("persist requires durable rating and promotion records");
        const auto* champion =
            std::get_if<std::string>(&state.payload().at("champion_checkpoint").value);
        const auto* champion_key =
            std::get_if<std::string>(&state.payload().at("champion_model_key").value);
        if (!champion || !champion_key || !std::filesystem::exists(*champion))
            throw CommandArtifactError("persist requires an active champion checkpoint");
        return stage_report(request, iteration, stage, operation,
                            {
                                {"champion_checkpoint", Json{*champion}},
                                {"champion_model_key", Json{*champion_key}},
                                {"promotion_record", promotions.back()},
                                {"rating_record", ratings.back()},
                                {"status", Json{"completed"}},
                            });
    }
    return stage_report(request, iteration, stage, operation, {{"status", Json{"completed"}}});
}

Result aggregate_result(const CommandRequest& request) {
    Result result;
    const auto iterations = root(request) / "iterations";
    std::error_code error;
    for (std::filesystem::directory_iterator it(iterations, error), end; !error && it != end;
         it.increment(error)) {
        if (!it->is_directory(error))
            continue;
        std::ifstream input(it->path() / "self-play.json", std::ios::binary);
        if (!input)
            continue;
        try {
            const auto json =
                diamond_support::parse_json(std::string(std::istreambuf_iterator<char>(input), {}));
            const auto& object = std::get<Object>(json.value);
            result.games +=
                static_cast<std::size_t>(std::get<int64_t>(object.at("completed_games").value) +
                                         std::get<int64_t>(object.at("aborted_games").value));
        } catch (const std::exception&) {
            throw CommandArtifactError("persisted self-play stage report is invalid");
        }
        std::ifstream train_report(it->path() / "train.json", std::ios::binary);
        if (!train_report)
            continue;
        try {
            const auto json = diamond_support::parse_json(
                std::string(std::istreambuf_iterator<char>(train_report), {}));
            const auto& object = std::get<Object>(json.value);
            result.step = std::max(result.step, static_cast<uint64_t>(std::get<int64_t>(
                                                    object.at("training_step").value)));
        } catch (const std::exception&) {
            throw CommandArtifactError("persisted train stage report is invalid");
        }
    }
    if (error)
        throw CommandArtifactError("cannot scan persisted stage reports");
    return result;
}

Object train(const CommandRequest& request, const ProductionConfig& config, bool resume,
             const diamond_training::ResolvedDevice& device) {
    const auto effective_request = resume ? authoritative_request(request) : request;
    diamond_orchestration::RunStateStore store(request.run_dir.parent_path().parent_path());
    const auto compatibility = wire(config);
    const auto canonical_config = config.to_json();
    const auto canonical_config_text = diamond_support::canonical_json(canonical_config);
    const auto config_sha256 = diamond_support::sha256(canonical_config_text);
    const auto config_path = root(request) / "resolved-config.json";
    if (resume) {
        std::ifstream input(config_path, std::ios::binary);
        if (!input) throw CommandArtifactError("resolved run config is missing");
        const std::string stored((std::istreambuf_iterator<char>(input)), {});
        try {
            if (diamond_support::canonical_json(diamond_support::parse_json(stored)) !=
                canonical_config_text) {
                throw CommandArtifactError(
                    "resume config does not match the immutable resolved run config");
            }
        } catch (const CommandArtifactError&) {
            throw;
        } catch (const std::exception&) {
            throw CommandArtifactError("resolved run config is invalid");
        }
    }
    const auto initial = resume ? store.load(request.model_name, request.run_id) :
        store.initialize(diamond_orchestration::RunState::initialize(request.run_id,
            Object{{"model_name", Json{compatibility.model_name}}, {"model_version", Json{compatibility.model_version}},
                   {"player_count", Json{static_cast<int64_t>(compatibility.player_count)}},
                   {"value_semantics_version", Json{compatibility.value_semantics_version}},
                   {"resolved_config_sha256", Json{config_sha256}}},
            Object{{"pipeline", Json{"native-pipeline-v2"}},
                   {"rating", Json{request.model_name == "Soo" ? "soo-elo-v1" : "min-trueskill-v1"}}}, config.run_seed));
    if (!resume) {
        write_json(config_path, canonical_config);
        persist_initialization(effective_request);
    }
    diamond_orchestration::Coordinator coordinator(
        store, describe,
        [&](RunStage stage, const diamond_orchestration::RunState& state,
            const std::string& operation) {
            return execute_stage(effective_request, config, state, stage, operation, device);
        });
    std::optional<uint64_t> max_iterations;
    if (config.run_budget.max_iterations)
        max_iterations = static_cast<uint64_t>(*config.run_budget.max_iterations);
    std::optional<std::chrono::steady_clock::time_point> deadline;
    if (config.run_budget.max_wall_clock_seconds) {
        deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(*config.run_budget.max_wall_clock_seconds));
    }
    const auto complete = coordinator.run_bounded(initial, max_iterations, deadline);
    const auto result = aggregate_result(effective_request);
    const auto completed_iterations =
        std::get<int64_t>(complete.payload().at("iteration").value) + 1;
    return {{"run_id", Json{complete.run_id()}}, {"stage", Json{"COMPLETE"}},
            {"completed_iterations", Json{completed_iterations}},
            {"completed_games", Json{static_cast<int64_t>(result.games)}},
            {"config_sha256", Json{config_sha256}},
            {"training_step", Json{static_cast<int64_t>(result.step)}}};
}
Object evaluate(const CommandRequest& request, const ProductionConfig& config,
                const diamond_training::ResolvedDevice& device) {
    if (request.opening_suite_id != config.opening_suite.id)
        throw diamond_orchestration::CommandArgumentError(
            "--opening-suite does not match the immutable resolved run config");
    const auto candidate_path = request.candidate_path;
    const auto champion_path = request.champion_path;
    const auto compatibility = wire(config);
    auto candidate_model = model(config);
    auto champion_model = model(config);
    diamond_pipeline::ModelPool candidate(1, device);
    diamond_pipeline::ModelPool champion(1, device);
    diamond_training::CheckpointInfo candidate_info;
    diamond_training::CheckpointInfo champion_info;
    diamond_pipeline::ModelKey candidate_key;
    diamond_pipeline::ModelKey champion_key;
    try {
        candidate_info = diamond_training::inspect_checkpoint_v2(candidate_path);
        champion_info = diamond_training::inspect_checkpoint_v2(champion_path);
        candidate_key = candidate.install_checkpoint(compatibility, candidate_path, candidate_model);
        champion_key = champion.install_checkpoint(compatibility, champion_path, champion_model);
        candidate.activate(candidate_key);
        champion.activate(champion_key);
    } catch (const std::exception& error) {
        throw CommandArtifactError(error.what());
    }

    const auto registry_path = root(request) / "rating-registry.json";
    auto registry = rating_registry(registry_path, config);
    const auto candidate_id = "candidate-" + candidate_info.model_digest.substr(0, 16);
    const auto champion_id = "champion-" + champion_info.model_digest.substr(0, 16);
    registry.add_participant(candidate_id, "Candidate");
    registry.add_participant(champion_id, "Champion");

    const auto suite = diamond_orchestration::materialize_opening_suite(config.opening_suite);
    write_json(root(request) / "opening-suite.json", Json{Object{
                                                         {"config", config.opening_suite.to_json()},
                                                         {"suite_sha256", Json{suite.suite_sha256}},
                                                     }});

    Object result;
    if (config.model_name == "Soo") {
        const std::array<std::string, 2> participants{candidate_id, champion_id};
        const auto blocks = diamond_orchestration::schedule_soo_opening_blocks(suite, participants);
        const std::size_t expected_games = blocks.size() * blocks.front().matches.size();
        if (static_cast<std::size_t>(config.arena.games) != expected_games)
            throw diamond_orchestration::CommandArgumentError(
                "arena.games must equal the complete Soo opening schedule size");
        std::vector<diamond_orchestration::SooOpeningBlockResult> outcomes;
        int64_t wins = 0;
        int64_t losses = 0;
        uint64_t sequence = 0;
        for (const auto& block : blocks) {
            diamond_orchestration::SooOpeningBlockResult block_result{.opening_id =
                                                                          block.opening.opening_id};
            for (const auto& cell : block.matches) {
                const auto match = ordered_match(config, cell.turn_order);
                const int candidate_player = cell.seat_assignment[0];
                const auto episode = play_arena_game(config, match, opening(match, block.opening),
                                                     arena_seed(block.opening, cell.match_id),
                                                     candidate_player, candidate, champion);
                std::optional<bool> candidate_won;
                if (episode.completed && !episode.finish_order.empty()) {
                    candidate_won = episode.finish_order.front() == candidate_player;
                    *candidate_won ? ++wins : ++losses;
                }
                block_result.results.push_back({cell.match_id, candidate_won});
                if (!candidate_won) {
                    ++sequence;
                    continue;
                }
                const auto& winner = *candidate_won ? candidate_id : champion_id;
                const auto& loser = *candidate_won ? champion_id : candidate_id;
                registry.record_event(diamond_orchestration::make_soo_rating_event(
                    sequence++, "soo-elo-v1", participants, cell.seat_assignment, cell.turn_order,
                    block.opening.opening_id, true, winner, loser, cell.match_id));
            }
            outcomes.push_back(std::move(block_result));
        }
        const auto summary = diamond_orchestration::summarize_soo_opening_blocks(blocks, outcomes);
        const auto statistics = diamond_orchestration::bootstrap_opening_blocks(
            summary, config.promotion_statistics, config.arena.promotion_threshold);
        const bool promoted = statistics.promoted && summary.incomplete_blocks == 0;
        result = {{"aborted_games", Json{summary.aborted_matches}},
                  {"complete_blocks", Json{summary.complete_blocks}},
                  {"confidence_lower", Json{statistics.confidence_lower}},
                  {"confidence_upper", Json{statistics.confidence_upper}},
                  {"incomplete_blocks", Json{summary.incomplete_blocks}},
                  {"losses", Json{losses}},
                  {"point_estimate", Json{statistics.point_estimate}},
                  {"promoted", Json{promoted}},
                  {"promotion_reason",
                   Json{promoted                         ? "confidence_threshold_met"
                        : summary.incomplete_blocks != 0 ? "incomplete_opening_blocks"
                        : summary.complete_blocks == 0   ? "no_complete_opening_blocks"
                                                         : "confidence_lower_below_threshold"}},
                  {"rating_status", Json{"eligible"}},
                  {"win_rate", Json{statistics.point_estimate}},
                  {"wins", Json{wins}}};
    } else {
        const std::array<std::string, 3> participants{candidate_id, champion_id + "-a",
                                                      champion_id + "-b"};
        registry.add_participant(participants[1], "Champion A");
        registry.add_participant(participants[2], "Champion B");
        const auto blocks = diamond_orchestration::schedule_min_opening_blocks(suite, participants);
        const std::size_t expected_games = blocks.size() * blocks.front().matches.size();
        if (static_cast<std::size_t>(config.arena.games) != expected_games)
            throw diamond_orchestration::CommandArgumentError(
                "arena.games must equal the complete Min opening schedule size");
        std::vector<diamond_orchestration::MinOpeningBlockResult> outcomes;
        int64_t placements[3]{};
        uint64_t sequence = 0;
        for (const auto& block : blocks) {
            diamond_orchestration::MinOpeningBlockResult block_result{.opening_id =
                                                                          block.opening.opening_id};
            for (const auto& cell : block.matches) {
                const auto match = ordered_match(config, cell.turn_order);
                const int candidate_player = cell.seat_assignment[0];
                const auto episode = play_arena_game(config, match, opening(match, block.opening),
                                                     arena_seed(block.opening, cell.match_id),
                                                     candidate_player, candidate, champion);
                std::optional<int> placement;
                if (episode.completed) {
                    const auto found = std::find(episode.finish_order.begin(),
                                                 episode.finish_order.end(), candidate_player);
                    if (found == episode.finish_order.end())
                        throw std::runtime_error("completed Min arena omitted the candidate");
                    placement =
                        static_cast<int>(std::distance(episode.finish_order.begin(), found));
                    ++placements[*placement];
                }
                block_result.results.push_back({cell.match_id, placement});
                if (!placement) {
                    ++sequence;
                    continue;
                }
                std::array<std::string, 3> ranking;
                for (std::size_t rank = 0; rank < ranking.size(); ++rank) {
                    const auto seat = episode.finish_order[rank];
                    const auto participant =
                        std::find(cell.seat_assignment.begin(), cell.seat_assignment.end(), seat);
                    if (participant == cell.seat_assignment.end())
                        throw std::runtime_error("Min arena ranking contains an unknown seat");
                    ranking[rank] = participants[static_cast<std::size_t>(
                        std::distance(cell.seat_assignment.begin(), participant))];
                }
                registry.record_event(diamond_orchestration::make_min_rating_event(
                    sequence++, "min-trueskill-v1", participants, cell.seat_assignment,
                    cell.turn_order, block.opening.opening_id, true, ranking, cell.match_id));
            }
            outcomes.push_back(std::move(block_result));
        }
        const auto summary = diamond_orchestration::summarize_min_opening_blocks(blocks, outcomes);
        const auto statistics = diamond_orchestration::bootstrap_opening_blocks(
            summary, config.promotion_statistics, config.arena.promotion_threshold);
        const bool promoted = statistics.promoted && summary.incomplete_blocks == 0;
        result = {{"aborted_games", Json{summary.aborted_matches}},
                  {"complete_blocks", Json{summary.complete_blocks}},
                  {"confidence_lower", Json{statistics.confidence_lower}},
                  {"confidence_upper", Json{statistics.confidence_upper}},
                  {"first_places", Json{placements[0]}},
                  {"incomplete_blocks", Json{summary.incomplete_blocks}},
                  {"mean_utility", Json{statistics.point_estimate}},
                  {"promoted", Json{promoted}},
                  {"promotion_reason",
                   Json{promoted                         ? "confidence_threshold_met"
                        : summary.incomplete_blocks != 0 ? "incomplete_opening_blocks"
                        : summary.complete_blocks == 0   ? "no_complete_opening_blocks"
                                                         : "confidence_lower_below_threshold"}},
                  {"rating_status", Json{"eligible"}},
                  {"second_places", Json{placements[1]}},
                  {"third_places", Json{placements[2]}}};
    }
    diamond_orchestration::save_rating_registry(registry_path, registry);
    result.emplace("candidate_checkpoint", Json{candidate_path.string()});
    result.emplace("candidate_runtime_sha256", Json{candidate_key.checkpoint_sha256});
    result.emplace("candidate_sha256", Json{candidate_info.model_digest});
    result.emplace("champion_checkpoint", Json{champion_path.string()});
    result.emplace("champion_runtime_sha256", Json{champion_key.checkpoint_sha256});
    result.emplace("champion_sha256", Json{champion_info.model_digest});
    result.emplace("opening_suite_sha256", Json{suite.suite_sha256});
    result.emplace("requested_games", Json{config.arena.games});
    result.emplace("registry_path", Json{registry_path.string()});
    write_json(root(request) / "arena.json", Json{result});
    return result;
}
Object report(const CommandRequest& request) {
    const auto registry_path = root(request) / "rating-registry.json";
    const auto evaluation_path = root(request) / "arena.json";
    const auto state_path = root(request) / "state.json";
    const auto config_path = root(request) / "resolved-config.json";
    const auto initialization_path = root(request) / "initialization.json";
    try {
        const auto state = read_object(state_path, "run state");
        const auto config = read_object(config_path, "resolved run config");
        const auto initialization = read_object(initialization_path, "run initialization");
        const auto iteration = std::get<int64_t>(state.at("iteration").value);

        Object stage_reports;
        constexpr std::array stages{RunStage::initialize,       RunStage::self_play,
                                    RunStage::replay_ingest,    RunStage::train,
                                    RunStage::save_candidate,   RunStage::promotion_arena,
                                    RunStage::rating_benchmark, RunStage::promote_or_reject,
                                    RunStage::persist};
        for (const auto stage : stages) {
            const auto path = stage_report_path(request, iteration, stage);
            if (!std::filesystem::exists(path))
                continue;
            stage_reports.emplace(stage_label(stage),
                                  Json{Object{
                                      {"path", Json{path.string()}},
                                      {"result", Json{read_object(path, "stage report")}},
                                      {"sha256", Json{file_sha256(path, "stage report")}},
                                  }});
        }

        Object result{
            {"evaluation", Json{nullptr}},
            {"evaluation_path", Json{evaluation_path.string()}},
            {"initialization", Json{initialization}},
            {"rating", Json{nullptr}},
            {"registry_path", Json{registry_path.string()}},
            {"resolved_config", Json{config}},
            {"resolved_config_file_sha256", Json{file_sha256(config_path, "resolved run config")}},
            {"run_state", Json{state}},
            {"run_state_path", Json{state_path.string()}},
            {"run_state_sha256", Json{file_sha256(state_path, "run state")}},
            {"stage_reports", Json{std::move(stage_reports)}},
        };
        if (std::filesystem::exists(evaluation_path))
            result.at("evaluation") = Json{read_object(evaluation_path, "evaluation report")};
        if (std::filesystem::exists(registry_path)) {
            const auto registry = diamond_orchestration::load_rating_registry(registry_path);
            result.at("rating") = registry.report_json();
        }
        return result;
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

    auto canonical_config = config;
    canonical_config.runtime.device = resolved.canonical_name;
    Object details;
    if (request.command == "train") details = train(request, canonical_config, false, resolved);
    else if (request.command == "resume") details = train(request, canonical_config, true, resolved);
    else details = evaluate(request, canonical_config, resolved);
    details.emplace("requested_device", Json{resolved.requested_name});
    details.emplace("canonical_device", Json{resolved.canonical_name});
    return details;
}
}  // namespace

int main(int argc, char** argv) {
    return diamond_orchestration::dispatch_command(argc, argv, service, std::cout);
}
