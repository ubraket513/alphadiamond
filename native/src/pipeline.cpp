#include "diamond_pipeline/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "diamond_training/checkpoint.hpp"

namespace diamond_pipeline {
namespace {

TrainingSample sample_from_move(const soo::EpisodeMove& move, const soo::Episode& episode,
                                const Compatibility& compatibility) {
    TrainingSample sample;
    sample.compatibility = compatibility;
    sample.node_features = move.features.node_features;
    sample.canonical_player_ids.assign(move.features.canonical_player_ids.begin(),
                                       move.features.canonical_player_ids.end());
    const auto visits = std::accumulate(move.visit_counts.begin(), move.visit_counts.end(), uint64_t{0});
    if (visits == 0 || move.root_actions.size() != move.visit_counts.size())
        throw PipelineError("self-play emitted an invalid sparse visit policy");
    for (std::size_t i = 0; i < move.root_actions.size(); ++i) {
        if (move.visit_counts[i] != 0)
            sample.sparse_policy.emplace_back(move.root_actions[i],
                static_cast<float>(static_cast<double>(move.visit_counts[i]) / visits));
    }
    if (compatibility.player_count == 2) {
        if (episode.finish_order.empty() || sample.canonical_player_ids.empty())
            throw PipelineError("completed Soo episode has no terminal order");
        sample.value_target = {episode.finish_order.front() == sample.canonical_player_ids.front() ? 1.0F : -1.0F};
    } else {
        for (int32_t player : sample.canonical_player_ids) {
            const auto found = std::find(episode.finish_order.begin(), episode.finish_order.end(), player);
            if (found == episode.finish_order.end()) throw PipelineError("completed Min episode has no terminal rank");
            const auto rank = static_cast<float>(std::distance(episode.finish_order.begin(), found));
            sample.value_target.push_back(1.0F - rank);
        }
    }
    return sample;
}

void write_u64(std::ostream& out, uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

uint64_t read_u64(std::istream& input) {
    uint64_t value = 0;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input)
        throw PipelineError("self-play artifact is truncated");
    return value;
}

template <class T> void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <class T> T read_value(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input)
        throw PipelineError("self-play artifact is truncated");
    return value;
}

void write_string(std::ostream& out, std::string_view value) {
    write_u64(out, value.size());
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string read_string(std::istream& input) {
    const auto size = read_u64(input);
    if (size > (1ULL << 32))
        throw PipelineError("self-play artifact string is too large");
    std::string value(static_cast<std::size_t>(size), '\0');
    input.read(value.data(), static_cast<std::streamsize>(value.size()));
    if (!input)
        throw PipelineError("self-play artifact is truncated");
    return value;
}

void write_sample(std::ostream& out, const TrainingSample& sample) {
    write_u64(out, sample.node_features.size());
    for (const auto value : sample.node_features)
        write_value(out, value);
    write_u64(out, sample.canonical_player_ids.size());
    for (const auto value : sample.canonical_player_ids)
        write_value(out, value);
    write_u64(out, sample.sparse_policy.size());
    for (const auto& [action, probability] : sample.sparse_policy) {
        write_value(out, action);
        write_value(out, probability);
    }
    write_u64(out, sample.value_target.size());
    for (const auto value : sample.value_target)
        write_value(out, value);
}

TrainingSample read_sample(std::istream& input, const Compatibility& compatibility) {
    TrainingSample sample;
    sample.compatibility = compatibility;
    const auto features = read_u64(input);
    if (features > (1ULL << 32))
        throw PipelineError("self-play artifact sample is too large");
    sample.node_features.resize(static_cast<std::size_t>(features));
    for (auto& value : sample.node_features)
        value = read_value<float>(input);
    const auto players = read_u64(input);
    if (players > 64)
        throw PipelineError("self-play artifact player list is invalid");
    sample.canonical_player_ids.resize(static_cast<std::size_t>(players));
    for (auto& value : sample.canonical_player_ids)
        value = read_value<int32_t>(input);
    const auto policy = read_u64(input);
    if (policy > (1ULL << 32))
        throw PipelineError("self-play artifact policy is too large");
    sample.sparse_policy.resize(static_cast<std::size_t>(policy));
    for (auto& [action, probability] : sample.sparse_policy) {
        action = read_value<int32_t>(input);
        probability = read_value<float>(input);
    }
    const auto targets = read_u64(input);
    if (targets > 64)
        throw PipelineError("self-play artifact value target is invalid");
    sample.value_target.resize(static_cast<std::size_t>(targets));
    for (auto& value : sample.value_target)
        value = read_value<float>(input);
    return sample;
}

}  // namespace

void save_episode_artifact(const std::filesystem::path& path, std::string_view operation_id,
                           std::span<const Episode> episodes) {
    if (operation_id.empty())
        throw PipelineError("self-play operation_id is required");
    if (std::filesystem::exists(path))
        throw PipelineError("self-play artifact already exists: " + path.string());
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.parent_path() / (path.filename().string() + ".tmp");
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out)
        throw PipelineError("cannot write self-play artifact: " + path.string());
    write_string(out, "diamond-selfplay-v1");
    write_string(out, operation_id);
    write_u64(out, episodes.size());
    for (const auto& episode : episodes) {
        write_string(out, episode.game_id);
        write_u64(out, episode.seed);
        write_string(out, episode.retry_id);
        write_value(out, episode.completed);
        write_string(out, episode.aborted_reason);
        write_u64(out, episode.move_count);
        write_u64(out, episode.final_order.size());
        for (const auto value : episode.final_order)
            write_value(out, value);
        write_u64(out, episode.samples.size());
        for (const auto& sample : episode.samples)
            write_sample(out, sample);
    }
    out.flush();
    if (!out) {
        out.close();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw PipelineError("cannot write self-play artifact: " + path.string());
    }
    out.close();
    std::filesystem::rename(temporary, path);
}

std::vector<Episode> load_episode_artifact(const std::filesystem::path& path,
                                           std::string_view operation_id,
                                           const Compatibility& compatibility) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw PipelineError("cannot read self-play artifact: " + path.string());
    if (read_string(input) != "diamond-selfplay-v1" || read_string(input) != operation_id)
        throw PipelineError("self-play artifact identity does not match operation");
    const auto count = read_u64(input);
    if (count > (1ULL << 20))
        throw PipelineError("self-play artifact has too many games");
    std::vector<Episode> episodes;
    episodes.reserve(static_cast<std::size_t>(count));
    for (uint64_t index = 0; index < count; ++index) {
        Episode episode;
        episode.game_id = read_string(input);
        episode.seed = read_u64(input);
        episode.retry_id = read_string(input);
        episode.completed = read_value<bool>(input);
        episode.aborted_reason = read_string(input);
        episode.move_count = read_u64(input);
        const auto order = read_u64(input);
        if (order > 64)
            throw PipelineError("self-play artifact finish order is invalid");
        episode.final_order.resize(static_cast<std::size_t>(order));
        for (auto& value : episode.final_order)
            value = read_value<int32_t>(input);
        const auto samples = read_u64(input);
        if (samples > (1ULL << 24))
            throw PipelineError("self-play artifact has too many samples");
        episode.samples.reserve(static_cast<std::size_t>(samples));
        for (uint64_t sample = 0; sample < samples; ++sample)
            episode.samples.push_back(read_sample(input, compatibility));
        episode.compatibility = compatibility;
        episode.model_key = {compatibility.model_name, compatibility.model_version,
                             "selfplay-artifact"};
        episodes.push_back(std::move(episode));
    }
    return episodes;
}

SelfPlayResult run_self_play(const IterationRequest& request, ModelPool& models,
                             std::stop_token stop) {
    if (request.operation_id.empty()) throw PipelineError("iteration operation_id is required");
    request.compatibility.validate();
    models.require_ready(stop, std::chrono::steady_clock::now() + std::chrono::hours(24));
    models.require_compatible(request.compatibility);
    if (!(models.active_key() == request.model_key))
        throw PipelineError("active model key does not match iteration request");

    SelfPlayResult result{.operation_id = request.operation_id};
    if (request.jobs.empty())
        return result;
    soo::EpisodeMetrics metrics;
    const auto episodes =
        soo::run_episodes(request.match, request.jobs, request.selfplay, models, metrics);
    std::vector<soo::VisitTargetObservation> all_targets;
    std::vector<soo::VisitTargetObservation> completed_targets;
    std::vector<soo::VisitTargetObservation> aborted_targets;
    for (const auto& episode : episodes) {
        auto& bucket = episode.completed ? completed_targets : aborted_targets;
        for (const auto& move : episode.moves) {
            const auto row = soo::inspect_visit_target(move.visit_counts);
            all_targets.push_back(row);
            bucket.push_back(row);
        }
    }
    result.metrics.all_targets = soo::summarize_visit_targets(all_targets);
    result.metrics.completed_targets = soo::summarize_visit_targets(completed_targets);
    result.metrics.aborted_targets = soo::summarize_visit_targets(aborted_targets);
    result.episodes.reserve(episodes.size());
    for (std::size_t index = 0; index < episodes.size(); ++index) {
        if (stop.stop_requested())
            throw CancelledError("native pipeline cancelled during self-play");
        const auto& episode = episodes[index];
        Episode record;
        record.game_id = request.operation_id + "-" + std::to_string(index);
        record.seed = request.jobs[index].seed;
        record.retry_id = "native-v2";
        record.model_key = request.model_key;
        record.compatibility = request.compatibility;
        record.move_count = static_cast<uint64_t>(episode.move_count);
        record.completed = episode.completed;
        record.final_order.assign(episode.finish_order.begin(), episode.finish_order.end());
        if (!episode.completed) {
            record.aborted_reason = episode.max_game_seconds_exceeded ? "max_game_seconds"
                                    : episode.move_limit_exceeded     ? "max_moves"
                                                                      : "interrupted";
            ++result.aborted_games;
        } else {
            ++result.completed_games;
            for (const auto& move : episode.moves)
                record.samples.push_back(sample_from_move(move, episode, request.compatibility));
            result.new_samples += record.samples.size();
        }
        if (!episode.completed) {
            AbortedGameDiagnostics aborted;
            aborted.game_id = record.game_id;
            aborted.seed = record.seed;
            aborted.move_count = record.move_count;
            aborted.abort_reason = record.aborted_reason;
            aborted.state = episode.diagnostics;
            result.aborted_diagnostics.push_back(std::move(aborted));
        }
        result.episodes.push_back(std::move(record));
    }

    {
        std::vector<uint64_t> completed_moves;
        completed_moves.reserve(result.completed_games);
        for (const auto& record : result.episodes)
            if (record.completed) completed_moves.push_back(record.move_count);
        if (!completed_moves.empty()) {
            std::sort(completed_moves.begin(), completed_moves.end());
            const auto quantile = [&completed_moves](double q) {
                const auto last = static_cast<double>(completed_moves.size() - 1);
                return completed_moves[static_cast<std::size_t>(q * last)];
            };
            result.metrics.completed_moves_p50 = quantile(0.50);
            result.metrics.completed_moves_p90 = quantile(0.90);
            result.metrics.completed_moves_p99 = quantile(0.99);
            result.metrics.completed_moves_max = completed_moves.back();
        }
    }

    for (std::size_t index = 0; index < episodes.size(); ++index) {
        uint64_t blocked_cells = 0;
        for (const auto& camp : episodes[index].diagnostics.camps)
            blocked_cells += camp.foreign_in_target;
        if (blocked_cells == 0) continue;
        if (episodes[index].completed) {
            ++result.metrics.completed_with_blocked_camp;
        } else {
            ++result.metrics.aborted_with_blocked_camp;
            result.metrics.aborted_blocked_cells_total += blocked_cells;
        }
    }

    result.metrics.evaluations = metrics.evaluations;
    result.metrics.batches = metrics.batches;
    result.metrics.moves = metrics.moves;
    result.metrics.boosted_moves = metrics.boosted_moves;
    if (metrics.moves > 0) {
        result.metrics.boosted_fraction =
            static_cast<double>(metrics.boosted_moves) / static_cast<double>(metrics.moves);
    }
    result.metrics.wall_seconds = metrics.wall_seconds;
    result.metrics.evaluator_seconds = metrics.evaluator_seconds;
    result.metrics.worker_busy_seconds = metrics.worker_busy_seconds;
    if (metrics.wall_seconds > 0.0)
        result.metrics.evaluator_busy_fraction = metrics.evaluator_seconds / metrics.wall_seconds;
    if (!metrics.batch_sizes.empty()) {
        auto sizes = metrics.batch_sizes;
        std::sort(sizes.begin(), sizes.end());
        double total = 0.0;
        for (const uint32_t size : sizes) total += size;
        result.metrics.batch_mean = total / static_cast<double>(sizes.size());
        const auto quantile = [&sizes](double q) {
            const auto last = static_cast<double>(sizes.size() - 1);
            return sizes[static_cast<std::size_t>(q * last)];
        };
        result.metrics.batch_p50 = quantile(0.50);
        result.metrics.batch_p90 = quantile(0.90);
        result.metrics.batch_max = sizes.back();
    }
    return result;
}

ReplayIngestReport ingest_self_play(ReplayStore& replay, std::span<const Episode> episodes) {
    return replay.ingest_iteration(episodes);
}

TrainingResult train_replay(const IterationRequest& request, ReplayStore& replay,
                            diamond_training::Trainer& trainer, std::stop_token stop) {
    if (request.operation_id.empty())
        throw PipelineError("iteration operation_id is required");
    request.compatibility.validate();
    if (!(request.compatibility == trainer.compatibility()))
        throw PipelineError("iteration compatibility does not match trainer");
    TrainingResult result{.operation_id = request.operation_id,
                          .requested_training_steps = request.training_steps,
                          .replay_size = replay.size(),
                          .training_step = trainer.training_step()};
    if (request.training_steps != 0 && request.training_batch_size == 0)
        throw PipelineError("training batch size must be positive");
    if (request.training_steps != 0 && request.training_batch_size > result.replay_size) {
        throw PipelineError("insufficient replay samples: requested " +
                            std::to_string(request.training_batch_size) + ", available " +
                            std::to_string(result.replay_size));
    }
    // Sampling is stateless: the seed is a pure function of the replay seed,
    // the iteration and the local step, so a TRAIN stage killed part-way and
    // re-run from step 0 draws exactly the same minibatch sequence, and the
    // replay store is never written during training.
    for (std::size_t step = 0; step < request.training_steps; ++step) {
        if (stop.stop_requested()) throw CancelledError("native pipeline cancelled during training");
        const auto drawn = std::chrono::steady_clock::now();
        auto samples = replay.sample(
            request.training_batch_size,
            replay_sampling_seed(replay.replay_seed(), request.iteration, step));
        const auto sample_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - drawn).count();
        result.replay_sample_seconds += sample_seconds;
        result.replay_sample_max_seconds =
            std::max(result.replay_sample_max_seconds, sample_seconds);
        result.training_metrics.push_back(trainer.train(samples));
        ++result.completed_training_steps;
        result.training_batch_sizes.push_back(samples.size());
    }
    result.training_step = trainer.training_step();
    return result;
}

IterationResult run_iteration(const IterationRequest& request, ModelPool& models,
                              ReplayStore& replay, diamond_training::Trainer& trainer,
                              std::stop_token stop) {
    const auto self_play = run_self_play(request, models, stop);
    const auto ingest = ingest_self_play(replay, self_play.episodes);
    const auto training = train_replay(request, replay, trainer, stop);
    if (request.checkpoint_root)
        (void)diamond_training::save_checkpoint_v2(*request.checkpoint_root, trainer);
    return {.operation_id = request.operation_id,
            .completed_games = self_play.completed_games,
            .aborted_games = self_play.aborted_games,
            .new_samples = self_play.new_samples,
            .requested_training_steps = training.requested_training_steps,
            .completed_training_steps = training.completed_training_steps,
            .training_batch_sizes = std::move(training.training_batch_sizes),
            .training_metrics = std::move(training.training_metrics),
            .replay_size = training.replay_size,
            .training_step = training.training_step};
}

}  // namespace diamond_pipeline
