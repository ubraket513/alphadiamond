#include "diamond_pipeline/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

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

}  // namespace

IterationResult run_iteration(const IterationRequest& request, ModelPool& models,
                              ReplayStore& replay, diamond_training::Trainer& trainer,
                              std::stop_token stop) {
    if (request.operation_id.empty()) throw PipelineError("iteration operation_id is required");
    request.compatibility.validate();
    if (!(request.compatibility == trainer.compatibility()))
        throw PipelineError("iteration compatibility does not match trainer");
    models.require_ready(stop, std::chrono::steady_clock::now() + std::chrono::hours(24));
    if (!(models.active_key() == request.model_key))
        throw PipelineError("active model key does not match iteration request");

    IterationResult result{.operation_id = request.operation_id,
                           .requested_training_steps = request.training_steps,
                           .replay_size = replay.size(),
                           .training_step = trainer.training_step()};
    if (request.jobs.empty() && request.training_steps == 0) return result;
    if (!request.jobs.empty()) {
        soo::EpisodeMetrics metrics;
        const auto episodes =
            soo::run_episodes(request.match, request.jobs, request.selfplay, models, metrics);
        std::vector<Episode> records;
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
                    : episode.move_limit_exceeded ? "max_moves" : "interrupted";
                record.samples.clear();
                ++result.aborted_games;
            } else {
                ++result.completed_games;
                for (const auto& move : episode.moves)
                    record.samples.push_back(
                        sample_from_move(move, episode, request.compatibility));
                result.new_samples += record.samples.size();
            }
            records.push_back(std::move(record));
        }
        replay.ingest(records);
        result.replay_size = replay.size();
    }
    if (request.training_steps != 0 && request.training_batch_size == 0)
        throw PipelineError("training batch size must be positive");
    if (request.training_steps != 0 && request.training_batch_size > result.replay_size) {
        throw PipelineError("insufficient replay samples: requested " +
                            std::to_string(request.training_batch_size) + ", available " +
                            std::to_string(result.replay_size));
    }
    for (std::size_t step = 0; step < request.training_steps; ++step) {
        if (stop.stop_requested()) throw CancelledError("native pipeline cancelled during training");
        auto samples = replay.sample(request.training_batch_size);
        (void)trainer.train(samples);
        ++result.completed_training_steps;
        result.training_batch_sizes.push_back(samples.size());
    }
    if (request.checkpoint_root) (void)diamond_training::save_checkpoint_v2(*request.checkpoint_root, trainer);
    result.training_step = trainer.training_step();
    return result;
}

}  // namespace diamond_pipeline
