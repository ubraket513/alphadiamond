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

        // Where one batch actually goes. `evaluator_busy_fraction` counts all
        // of this as "the evaluator", so a high value says nothing about
        // whether the GPU is saturated: it may be CPU collation, three host to
        // device copies, the forward pass, the softmax, the copy back, or the
        // scatter into the waiting lanes. Splitting it is what makes the next
        // optimisation decidable. GPU-side spans are measured with CUDA events
        // because the work is enqueued asynchronously and a host clock would
        // attribute all of it to whichever call happens to synchronise;
        // on CPU the same fields fall back to a host clock.
        double collation_seconds = 0.0;
        double h2d_seconds = 0.0;
        double forward_seconds = 0.0;
        double policy_postprocess_seconds = 0.0;
        double d2h_seconds = 0.0;
        double scatter_seconds = 0.0;

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
    // Summed over every batch since the last reset. last_evaluation_stats()
    // describes one batch and is useless for a run-level breakdown.
    EvaluationStats accumulated_evaluation_stats() const noexcept {
        return accumulated_evaluation_stats_;
    }
    std::size_t evaluated_batches() const noexcept {
        return evaluated_batches_;
    }
    void reset_evaluation_stats() noexcept {
        accumulated_evaluation_stats_ = EvaluationStats{};
        evaluated_batches_ = 0;
    }

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
    EvaluationStats accumulated_evaluation_stats_;
    std::size_t evaluated_batches_ = 0;

    // Reused pinned staging buffers, grown on demand and never shrunk.
    //
    // Three fresh heap allocations and three pageable host-to-device copies per
    // batch cost a third of the evaluator's time, close to the forward pass
    // itself. Pageable memory cannot be copied asynchronously: the driver has
    // to stage it through an internal pinned buffer first. Pinning the source
    // once removes that staging copy and lets the transfer overlap.
    //
    // Reuse is safe because one evaluator thread drives one pool, and because
    // the device-to-host copy at the end of every evaluate() synchronises the
    // stream -- so the previous batch's transfers have completed before the
    // next one can overwrite these buffers.
    torch::Tensor staging_features_;
    torch::Tensor staging_legal_indices_;
    torch::Tensor staging_valid_mask_;
};

}  // namespace diamond_pipeline
