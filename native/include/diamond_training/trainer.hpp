#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <torch/torch.h>

#include "diamond_model/soo_model.hpp"
#include "diamond_training/training_sample.hpp"

namespace diamond_training {

struct TrainingConfig {
    double learning_rate;
    double weight_decay;
};

struct TrainingMetrics {
    double total_loss;
    double policy_loss;
    double value_loss;
    uint64_t training_step;
};

enum class ModelRole {
    actor,
    learner,
    candidate,
};

// Rebuilds a model from source architecture and copies every named parameter
// and buffer to target. The result owns distinct storage and has the requested
// runtime role applied.
diamond_model::DiamondModel snapshot_model(const diamond_model::DiamondModel& source,
                                           const Compatibility& compatibility,
                                           torch::Device target, ModelRole role);

// A versioned SHA-256 identity over sorted named parameters then named buffers.
// FP32 tensor data is serialized canonically as contiguous CPU little-endian bytes.
std::string canonical_model_digest(const diamond_model::DiamondModel& model);

class Trainer {
  public:
    Trainer(diamond_model::DiamondModel model, Compatibility compatibility,
            TrainingConfig config);

    TrainingMetrics train(std::span<const TrainingSample> samples);
    uint64_t training_step() const { return training_step_; }
    const Compatibility& compatibility() const { return compatibility_; }
    const TrainingConfig& config() const { return config_; }
    diamond_model::DiamondModel& model() { return model_; }
    const diamond_model::DiamondModel& model() const { return model_; }
    diamond_model::DiamondModel& learner() { return model_; }
    const diamond_model::DiamondModel& learner() const { return model_; }
    diamond_model::DiamondModel candidate_snapshot() const;
    torch::optim::AdamW& optimizer() { return optimizer_; }
    void restore_checkpoint_state(TrainingConfig config, uint64_t training_step) {
        config_ = config;
        training_step_ = training_step;
    }

  private:
    diamond_model::DiamondModel model_;
    Compatibility compatibility_;
    TrainingConfig config_;
    torch::optim::AdamW optimizer_;
    uint64_t training_step_ = 0;
};

}  // namespace diamond_training
