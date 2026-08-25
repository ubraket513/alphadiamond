#include "diamond_training/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace diamond_training {
namespace {

constexpr int64_t kHoleCount = 73;
constexpr int64_t kActionCount = kHoleCount * kHoleCount;
constexpr double kPolicySumTolerance = 1e-5;

size_t checked_count(int64_t value, const char* name) {
    if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
    if (static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max())
        throw std::invalid_argument(std::string(name) + " is too large");
    return static_cast<size_t>(value);
}

size_t checked_product(size_t left, size_t right, const char* name) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::invalid_argument(std::string(name) + " is too large");
    return left * right;
}

void require_finite(std::span<const float> values, const char* name) {
    if (std::any_of(values.begin(), values.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void validate_sample(const TrainingSample& sample, const Compatibility& compatibility,
                     size_t feature_count, size_t value_count) {
    if (!(sample.compatibility == compatibility))
        throw std::invalid_argument("training sample compatibility mismatch");
    if (sample.node_features.size() != feature_count)
        throw std::invalid_argument("training sample feature width mismatch");
    if (sample.value_target.size() != value_count)
        throw std::invalid_argument("training sample value width mismatch");

    require_finite(sample.node_features, "training sample features");
    require_finite(sample.value_target, "training sample value target");

    double policy_sum = 0.0;
    std::unordered_set<int32_t> actions;
    actions.reserve(sample.sparse_policy.size());
    for (const auto& [action, probability] : sample.sparse_policy) {
        if (action < 0 || action >= kActionCount)
            throw std::invalid_argument("training sample policy action is out of range");
        if (!std::isfinite(probability))
            throw std::invalid_argument("training sample policy probability must be finite");
        if (probability < 0.0F)
            throw std::invalid_argument("training sample policy probability is negative");
        if (!actions.insert(action).second)
            throw std::invalid_argument("training sample policy action is duplicated");
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
    if (samples.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("training batch is too large");

    const int64_t batch_size = static_cast<int64_t>(samples.size());
    const int64_t input_features = model_->input_features();
    const int64_t value_size = model_->value_size();
    const size_t feature_count = checked_product(static_cast<size_t>(kHoleCount),
                                                 checked_count(input_features, "model input features"),
                                                 "training sample feature width");
    const size_t value_count = checked_count(value_size, "model value size");
    const size_t batch_count = static_cast<size_t>(batch_size);
    for (const TrainingSample& sample : samples) {
        validate_sample(sample, compatibility_, feature_count, value_count);
    }

    std::vector<float> feature_buffer(
        checked_product(batch_count, feature_count, "training feature buffer"));
    std::vector<float> policy_buffer(
        checked_product(batch_count, static_cast<size_t>(kActionCount), "training policy buffer"), 0.0F);
    std::vector<float> value_buffer(
        checked_product(batch_count, value_count, "training value buffer"));

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const TrainingSample& sample = samples[static_cast<size_t>(batch)];
        const size_t batch_offset = static_cast<size_t>(batch);
        std::memcpy(feature_buffer.data() + batch_offset * feature_count,
                    sample.node_features.data(), feature_count * sizeof(float));
        std::memcpy(value_buffer.data() + batch_offset * value_count,
                    sample.value_target.data(), value_count * sizeof(float));
        for (const auto& [action, probability] : sample.sparse_policy) {
            policy_buffer[batch_offset * static_cast<size_t>(kActionCount) +
                          static_cast<size_t>(action)] = probability;
        }
    }

    const auto host_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto device = model_->parameters().front().device();
    auto features = torch::from_blob(feature_buffer.data(),
                                     {batch_size, kHoleCount, input_features}, host_options)
                        .to(device);
    auto policy_targets = torch::from_blob(policy_buffer.data(), {batch_size, kActionCount}, host_options)
                              .to(device);
    auto value_targets = torch::from_blob(value_buffer.data(), {batch_size, value_size}, host_options)
                             .to(device);

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
