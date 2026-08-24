#include "diamond_training/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace diamond_training {
namespace {

constexpr int64_t kHoleCount = 73;
constexpr int64_t kActionCount = kHoleCount * kHoleCount;
constexpr double kPolicySumTolerance = 1e-5;

void require_finite(std::span<const float> values, const char* name) {
    if (std::any_of(values.begin(), values.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void validate_sample(const TrainingSample& sample, const Compatibility& compatibility,
                     int64_t input_features, int64_t value_size) {
    if (!(sample.compatibility == compatibility))
        throw std::invalid_argument("training sample compatibility mismatch");
    if (sample.node_features.size() != static_cast<size_t>(kHoleCount * input_features))
        throw std::invalid_argument("training sample feature width mismatch");
    if (sample.value_target.size() != static_cast<size_t>(value_size))
        throw std::invalid_argument("training sample value width mismatch");

    require_finite(sample.node_features, "training sample features");
    require_finite(sample.value_target, "training sample value target");

    double policy_sum = 0.0;
    for (const auto& [action, probability] : sample.sparse_policy) {
        if (action < 0 || action >= kActionCount)
            throw std::invalid_argument("training sample policy action is out of range");
        if (!std::isfinite(probability))
            throw std::invalid_argument("training sample policy probability must be finite");
        if (probability < 0.0F)
            throw std::invalid_argument("training sample policy probability is negative");
        policy_sum += probability;
    }
    if (std::fabs(policy_sum - 1.0) > kPolicySumTolerance)
        throw std::invalid_argument("training sample policy probabilities must sum to one");
}

}  // namespace

Trainer::Trainer(diamond_model::DiamondModel model, Compatibility compatibility,
                 TrainingConfig config)
    : model_(std::move(model)),
      compatibility_(std::move(compatibility)),
      config_(config),
      optimizer_(model_->parameters(), torch::optim::AdamWOptions(config.learning_rate)
                                      .weight_decay(config.weight_decay)) {
    if (!model_) throw std::invalid_argument("trainer model is required");
    compatibility_.validate();
    if (!std::isfinite(config_.learning_rate) || config_.learning_rate <= 0.0)
        throw std::invalid_argument("trainer learning rate must be positive and finite");
    if (!std::isfinite(config_.weight_decay) || config_.weight_decay < 0.0)
        throw std::invalid_argument("trainer weight decay must be finite and non-negative");
}

TrainingMetrics Trainer::train(std::span<const TrainingSample> samples) {
    if (samples.empty()) throw std::invalid_argument("training batch must not be empty");

    const int64_t batch_size = static_cast<int64_t>(samples.size());
    const int64_t input_features = model_->input_features();
    const int64_t value_size = model_->value_size();
    for (const TrainingSample& sample : samples) {
        validate_sample(sample, compatibility_, input_features, value_size);
    }

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto features = torch::empty({batch_size, kHoleCount, input_features}, options);
    auto policy_targets = torch::zeros({batch_size, kActionCount}, options);
    auto value_targets = torch::empty({batch_size, value_size}, options);
    float* feature_data = features.data_ptr<float>();
    float* value_data = value_targets.data_ptr<float>();

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const TrainingSample& sample = samples[static_cast<size_t>(batch)];
        const size_t feature_count = sample.node_features.size();
        const size_t value_count = sample.value_target.size();
        std::memcpy(feature_data + static_cast<size_t>(batch) * feature_count,
                    sample.node_features.data(), feature_count * sizeof(float));
        std::memcpy(value_data + static_cast<size_t>(batch) * value_count,
                    sample.value_target.data(), value_count * sizeof(float));
        for (const auto& [action, probability] : sample.sparse_policy) {
            policy_targets.index_put_({batch, action}, probability);
        }
    }

    optimizer_.zero_grad();
    auto [policy_logits, predicted_values] = model_->forward(features);
    auto policy_loss = -(policy_targets * torch::log_softmax(policy_logits, 1)).sum(1).mean();
    auto value_loss = torch::mse_loss(predicted_values, value_targets);
    auto total_loss = policy_loss + value_loss;
    if (!torch::isfinite(total_loss).item<bool>()) {
        throw std::invalid_argument("training produced a non-finite loss");
    }
    total_loss.backward();
    optimizer_.step();
    ++training_step_;

    return TrainingMetrics{total_loss.item<double>(), policy_loss.item<double>(),
                           value_loss.item<double>(), training_step_};
}

}  // namespace diamond_training
