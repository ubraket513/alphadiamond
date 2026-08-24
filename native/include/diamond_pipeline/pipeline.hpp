#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <stop_token>
#include <string>
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
    std::size_t training_steps = 0;
    std::optional<std::filesystem::path> checkpoint_root;
};

struct IterationResult {
    std::string operation_id;
    std::size_t completed_games = 0;
    std::size_t aborted_games = 0;
    std::size_t new_samples = 0;
    uint64_t training_step = 0;
};

IterationResult run_iteration(const IterationRequest& request, ModelPool& models,
                              ReplayStore& replay, diamond_training::Trainer& trainer,
                              std::stop_token stop);

}  // namespace diamond_pipeline
