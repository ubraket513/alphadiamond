#pragma once

#include <cstdint>
#include <span>

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

class Trainer {
  public:
    Trainer(diamond_model::DiamondModel model, Compatibility compatibility,
            TrainingConfig config);

    TrainingMetrics train(std::span<const TrainingSample> samples);

  private:
    diamond_model::DiamondModel model_;
    Compatibility compatibility_;
    TrainingConfig config_;
    torch::optim::AdamW optimizer_;
    uint64_t training_step_ = 0;
};

}  // namespace diamond_training
