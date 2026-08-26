#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "diamond_pipeline/model_pool.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/selfplay.hpp"

namespace diamond_pipeline {

struct IterationRequest {
    std::string operation_id;
    ModelKey model_key;
    Compatibility compatibility;
    soo::Match match;
    std::vector<soo::EpisodeJob> jobs;
    soo::EpisodeConfig selfplay;
    std::size_t training_batch_size = 1;
    std::size_t training_steps = 0;
    std::optional<std::filesystem::path> checkpoint_root;
};

struct IterationResult {
    std::string operation_id;
    std::size_t completed_games = 0;
    std::size_t aborted_games = 0;
    std::size_t new_samples = 0;
    std::size_t requested_training_steps = 0;
    std::size_t completed_training_steps = 0;
    std::vector<std::size_t> training_batch_sizes;
    std::vector<diamond_training::TrainingMetrics> training_metrics;
    std::size_t replay_size = 0;
    uint64_t training_step = 0;
};

// Engine-side counters for one self-play stage. These live only for the
// duration of the run -- the stage report is rebuilt from the reloaded episode
// artifact so that it survives `resume`, and episodes carry no scheduler state
// -- so they are written to a sidecar at the moment self-play executes. Without
// that, `boosted_moves` in particular is computed and dropped, leaving no way to
// tell a repetition trigger that was configured and never needed from one that
// was configured and silently did nothing.
struct SelfPlayMetrics {
    uint64_t evaluations = 0;
    uint64_t batches = 0;
    uint64_t moves = 0;
    uint64_t boosted_moves = 0;
    double boosted_fraction = 0.0;
    double wall_seconds = 0.0;
    double evaluator_seconds = 0.0;
    double worker_busy_seconds = 0.0;
    double evaluator_busy_fraction = 0.0;
    double batch_mean = 0.0;
    uint32_t batch_p50 = 0;
    uint32_t batch_p90 = 0;
    uint32_t batch_max = 0;
};

struct SelfPlayResult {
    std::string operation_id;
    std::size_t completed_games = 0;
    std::size_t aborted_games = 0;
    std::size_t new_samples = 0;
    SelfPlayMetrics metrics;
    std::vector<Episode> episodes;
};

struct TrainingResult {
    std::string operation_id;
    std::size_t requested_training_steps = 0;
    std::size_t completed_training_steps = 0;
    std::vector<std::size_t> training_batch_sizes;
    std::vector<diamond_training::TrainingMetrics> training_metrics;
    std::size_t replay_size = 0;
    uint64_t training_step = 0;
};

// Durable self-play hand-off used between the coordinator's SELF_PLAY and
// REPLAY_INGEST callbacks.  It contains only native pipeline records.
void save_episode_artifact(const std::filesystem::path& path, std::string_view operation_id,
                           std::span<const Episode> episodes);
std::vector<Episode> load_episode_artifact(const std::filesystem::path& path,
                                           std::string_view operation_id,
                                           const Compatibility& compatibility);

SelfPlayResult run_self_play(const IterationRequest& request, ModelPool& models,
                             std::stop_token stop = {});
ReplayIngestReport ingest_self_play(ReplayStore& replay, std::span<const Episode> episodes);
TrainingResult train_replay(const IterationRequest& request, ReplayStore& replay,
                            diamond_training::Trainer& trainer, std::stop_token stop = {});

IterationResult run_iteration(const IterationRequest& request, ModelPool& models,
                              ReplayStore& replay, diamond_training::Trainer& trainer,
                              std::stop_token stop);

}  // namespace diamond_pipeline
