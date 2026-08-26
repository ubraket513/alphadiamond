#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <vector>

#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/replay.hpp"
#include "diamond_training/device.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/selfplay.hpp"

namespace diamond_pipeline {

class PipelineError : public std::runtime_error { public: using std::runtime_error::runtime_error; };
class CancelledError final : public PipelineError { public: using PipelineError::PipelineError; };
class DeadlineExceededError final : public PipelineError { public: using PipelineError::PipelineError; };
class IncompatibleCheckpointError final : public PipelineError { public: using PipelineError::PipelineError; };

class ModelPool final : public soo::BatchEvaluator {
  public:
    struct EvaluationStats final {
        std::size_t forward_calls = 0;
        std::size_t h2d_transfers = 0;
        std::size_t d2h_transfers = 0;
        std::size_t batch_size = 0;
        std::size_t max_legal_actions = 0;

        bool operator==(const EvaluationStats&) const = default;
    };

    ModelPool(std::size_t capacity, diamond_training::ResolvedDevice device);

    ModelKey install(const Compatibility& compatibility, const diamond_model::DiamondModel& source);
    ModelKey install_checkpoint(const Compatibility& compatibility,
                                const std::filesystem::path& checkpoint_root,
                                diamond_model::DiamondModel staging);
    void activate(const ModelKey& key);
    const ModelKey& active_key() const;
    // Read-only observation of the currently active, immutable actor.
    const diamond_model::DiamondModel& active_model() const;
    std::size_t resident_count() const;
    void require_compatible(const Compatibility& expected) const;
    void require_ready(std::stop_token stop, std::chrono::steady_clock::time_point deadline) const;
    void evaluate(std::vector<soo::BatchItem>& batch) override;
    EvaluationStats last_evaluation_stats() const noexcept {
        return last_evaluation_stats_;
    }

  private:
    struct ResidentModel final {
        Compatibility compatibility;
        diamond_model::DiamondModel actor;
    };

    std::size_t capacity_;
    diamond_training::ResolvedDevice device_;
    std::map<ModelKey, ResidentModel> models_;
    std::optional<ModelKey> active_;
    EvaluationStats last_evaluation_stats_;
};

}  // namespace diamond_pipeline
