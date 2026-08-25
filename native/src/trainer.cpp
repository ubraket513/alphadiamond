#include "diamond_training/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "diamond_support/json.hpp"

namespace diamond_training {
namespace {

constexpr int64_t kHoleCount = 73;
constexpr int64_t kActionCount = kHoleCount * kHoleCount;
constexpr double kPolicySumTolerance = 1e-5;
constexpr int64_t kSooInputFeatures = 4;
constexpr int64_t kSooValueSize = 1;
constexpr int64_t kMinInputFeatures = 6;
constexpr int64_t kMinValueSize = 3;

using NamedTensors = std::map<std::string, torch::Tensor>;

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

void validate_config(const TrainingConfig& config) {
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0)
        throw std::invalid_argument("trainer learning rate must be positive and finite");
    if (!std::isfinite(config.weight_decay) || config.weight_decay < 0.0)
        throw std::invalid_argument("trainer weight decay must be finite and non-negative");
}

torch::optim::AdamWOptions adamw_options(const TrainingConfig& config) {
    validate_config(config);
    return torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay);
}

NamedTensors collect_named_tensors(const diamond_model::DiamondModel& model, bool parameters) {
    if (!model) throw std::invalid_argument("model snapshot source is empty");

    NamedTensors result;
    const auto named = parameters ? model->named_parameters() : model->named_buffers();
    for (const auto& entry : named) {
        if (!entry.value().defined())
            throw std::invalid_argument("model contains an undefined named tensor: " + entry.key());
        if (!result.emplace(entry.key(), entry.value()).second)
            throw std::invalid_argument("model contains a duplicated named tensor: " + entry.key());
    }
    return result;
}

torch::Device model_device(const diamond_model::DiamondModel& model) {
    const auto parameters = collect_named_tensors(model, true);
    if (parameters.empty()) throw std::invalid_argument("model snapshot source has no parameters");
    return parameters.begin()->second.device();
}

void require_expected_architecture(const diamond_model::DiamondModel& model,
                                  const Compatibility& compatibility) {
    if (!model) throw std::invalid_argument("model snapshot source is empty");
    compatibility.validate();
    if (model->width() != compatibility.network_config.width ||
        model->residual_blocks() != compatibility.network_config.residual_blocks) {
        throw std::invalid_argument("model architecture does not match compatibility network");
    }

    const bool is_soo = compatibility.family() == "soo";
    const int64_t expected_features = is_soo ? kSooInputFeatures : kMinInputFeatures;
    const int64_t expected_values = is_soo ? kSooValueSize : kMinValueSize;
    if (model->input_features() != expected_features || model->value_size() != expected_values) {
        throw std::invalid_argument("model architecture does not match compatibility family");
    }
    if (model->adjacency.sizes() != torch::IntArrayRef({6, 73, 73})) {
        throw std::invalid_argument("model adjacency must have shape [6,73,73]");
    }
}

void require_matching_named_tensors(const NamedTensors& source, const NamedTensors& destination,
                                   const torch::Device& source_device,
                                   const torch::Device& destination_device,
                                   const char* kind) {
    if (source.size() != destination.size())
        throw std::invalid_argument(std::string("model snapshot ") + kind + " names differ");

    for (const auto& [name, source_tensor] : source) {
        const auto destination_it = destination.find(name);
        if (destination_it == destination.end()) {
            throw std::invalid_argument(std::string("model snapshot is missing ") + kind + ": " + name);
        }
        const auto& destination_tensor = destination_it->second;
        if (source_tensor.device() != source_device) {
            throw std::invalid_argument(std::string("model snapshot source ") + kind +
                                        " device is inconsistent: " + name);
        }
        if (destination_tensor.device() != destination_device) {
            throw std::invalid_argument(std::string("model snapshot destination ") + kind +
                                        " device is inconsistent: " + name);
        }
        if (source_tensor.scalar_type() != destination_tensor.scalar_type()) {
            throw std::invalid_argument(std::string("model snapshot ") + kind +
                                        " dtype differs: " + name);
        }
        if (source_tensor.sizes() != destination_tensor.sizes()) {
            throw std::invalid_argument(std::string("model snapshot ") + kind +
                                        " shape differs: " + name);
        }
    }
}

void append_u64_le(std::string& stream, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        stream.push_back(static_cast<char>((value >> shift) & 0xffU));
    }
}

void append_string(std::string& stream, std::string_view value) {
    append_u64_le(stream, static_cast<uint64_t>(value.size()));
    stream.append(value.data(), value.size());
}

void append_tensor(std::string& stream, std::string_view kind, const std::string& name,
                   const torch::Tensor& tensor) {
    if (!tensor.defined()) throw std::invalid_argument("cannot digest an undefined tensor: " + name);
    if (tensor.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("canonical model tensor must be float32: " + name);
    }
    append_string(stream, kind);
    append_string(stream, name);
    // Use a semantic dtype tag rather than LibTorch's enum ordinal, whose
    // numeric value is not part of the cross-version identity contract.
    append_string(stream, "float32");
    append_u64_le(stream, static_cast<uint64_t>(tensor.dim()));
    for (const int64_t dimension : tensor.sizes()) {
        if (dimension < 0) throw std::invalid_argument("cannot digest a tensor with a negative dimension");
        append_u64_le(stream, static_cast<uint64_t>(dimension));
    }

    const auto cpu = tensor.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    if (!cpu.device().is_cpu() || cpu.scalar_type() != torch::kFloat32 || !cpu.is_contiguous()) {
        throw std::invalid_argument("cannot canonicalize model tensor: " + name);
    }
    if (!torch::isfinite(cpu).all().item<bool>()) {
        throw std::invalid_argument("canonical model tensor must be finite: " + name);
    }
    const int64_t element_count = cpu.numel();
    if (element_count < 0 ||
        static_cast<uint64_t>(element_count) > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        throw std::invalid_argument("model tensor is too large to digest: " + name);
    }
    append_u64_le(stream, static_cast<uint64_t>(element_count) * sizeof(float));
    const float* values = cpu.data_ptr<float>();  // cpu is explicit; never dereference CUDA storage.
    for (int64_t index = 0; index < element_count; ++index) {
        uint32_t bits = 0;
        std::memcpy(&bits, values + index, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            stream.push_back(static_cast<char>((bits >> shift) & 0xffU));
        }
    }
}

}  // namespace

diamond_model::DiamondModel snapshot_model(const diamond_model::DiamondModel& source,
                                           const Compatibility& compatibility,
                                           torch::Device target_device, ModelRole role) {
    require_expected_architecture(source, compatibility);
    const torch::Device source_device = model_device(source);
    auto snapshot = diamond_model::DiamondModel(source->width(), source->residual_blocks(),
                                                source->input_features(), source->value_size());
    {
        torch::NoGradGuard no_grad;
        snapshot->to(target_device);
        const auto source_parameters = collect_named_tensors(source, true);
        const auto source_buffers = collect_named_tensors(source, false);
        const auto destination_parameters = collect_named_tensors(snapshot, true);
        const auto destination_buffers = collect_named_tensors(snapshot, false);
        require_matching_named_tensors(source_parameters, destination_parameters, source_device,
                                      target_device, "parameter");
        require_matching_named_tensors(source_buffers, destination_buffers, source_device,
                                      target_device, "buffer");
        if (snapshot->adjacency.sizes() != torch::IntArrayRef({6, 73, 73})) {
            throw std::invalid_argument("snapshot adjacency must have shape [6,73,73]");
        }
        for (const auto& [name, source_tensor] : source_parameters) {
            destination_parameters.at(name).copy_(source_tensor);
        }
        for (const auto& [name, source_tensor] : source_buffers) {
            destination_buffers.at(name).copy_(source_tensor);
        }
    }

    bool trainable = false;
    switch (role) {
        case ModelRole::actor:
        case ModelRole::candidate:
            break;
        case ModelRole::learner:
            trainable = true;
            break;
        default:
            throw std::invalid_argument("unknown model snapshot role");
    }
    if (trainable) snapshot->train();
    else snapshot->eval();
    for (auto& parameter : snapshot->named_parameters()) {
        parameter.value().set_requires_grad(trainable);
    }
    return snapshot;
}

std::string canonical_model_digest(const diamond_model::DiamondModel& model) {
    std::string stream;
    append_string(stream, "alphadiamond.canonical-model-digest.v1");
    const auto parameters = collect_named_tensors(model, true);
    const auto buffers = collect_named_tensors(model, false);
    append_u64_le(stream, static_cast<uint64_t>(parameters.size()));
    for (const auto& [name, tensor] : parameters) append_tensor(stream, "parameter", name, tensor);
    append_u64_le(stream, static_cast<uint64_t>(buffers.size()));
    for (const auto& [name, tensor] : buffers) append_tensor(stream, "buffer", name, tensor);
    return diamond_support::sha256(stream);
}

Trainer::Trainer(diamond_model::DiamondModel model, Compatibility compatibility,
                 TrainingConfig config)
    : model_(snapshot_model(model, compatibility, model_device(model), ModelRole::learner)),
      compatibility_(std::move(compatibility)),
      config_(config),
      optimizer_(model_->parameters(), adamw_options(config)) {}

diamond_model::DiamondModel Trainer::candidate_snapshot() const {
    return snapshot_model(model_, compatibility_, model_device(model_), ModelRole::candidate);
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
