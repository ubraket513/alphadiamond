#include "diamond_training/trainer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <ATen/DeviceAccelerator.h>
#include <c10/core/Event.h>
#if defined(DIAMOND_TORCH_WITH_CUDA)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

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
using StepClock = std::chrono::steady_clock;

NamedTensors collect_named_tensors(const diamond_model::DiamondModel& model, bool parameters);

struct AcceleratorStepEvents {
    explicit AcceleratorStepEvents(c10::DeviceIndex device_index)
        : stream(at::accelerator::getCurrentStream(device_index)),
          h2d_start(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
          h2d_end(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
          forward_start(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
          forward_end(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
          backward_start(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
          backward_end(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
          optimizer_start(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
          optimizer_end(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT) {}

    c10::Stream stream;
    c10::Event h2d_start;
    c10::Event h2d_end;
    c10::Event forward_start;
    c10::Event forward_end;
    c10::Event backward_start;
    c10::Event backward_end;
    c10::Event optimizer_start;
    c10::Event optimizer_end;
};

double seconds_between(StepClock::time_point start, StepClock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

double event_seconds(const c10::Event& start, const c10::Event& end) {
    return start.elapsedTime(end) / 1000.0;
}

size_t checked_count(int64_t value, const char* name) {
    if (value <= 0)
        throw std::invalid_argument(std::string(name) + " must be positive");
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

void validate_resolved_device(const ResolvedDevice& device) {
    const DeviceRequest parsed = parse_device_request(device.requested_name);
    if (parsed.cuda_index != device.cuda_index) {
        throw std::invalid_argument("resolved trainer device metadata is inconsistent");
    }
    if (device.torch_device.str() != device.canonical_name) {
        throw std::invalid_argument("resolved trainer device canonical name is inconsistent");
    }
    if (device.torch_device.is_cpu()) {
        if (device.canonical_name != "cpu" || device.cuda_index.has_value()) {
            throw std::invalid_argument("resolved CPU trainer device metadata is inconsistent");
        }
        return;
    }
    if (!device.torch_device.is_cuda() || !device.cuda_index.has_value() ||
        !device.torch_device.has_index() || device.torch_device.index() != *device.cuda_index) {
        throw std::invalid_argument("trainer device must be resolved CPU or indexed CUDA");
    }
}

void require_tensor_device(const torch::Tensor& tensor, const torch::Device& expected,
                           const std::string& name) {
    if (!tensor.defined())
        throw std::invalid_argument(name + " is undefined");
    if (tensor.device() != expected) {
        throw std::invalid_argument(name + " is on " + tensor.device().str() + ", expected " +
                                    expected.str());
    }
}

void require_model_device(const diamond_model::DiamondModel& model, const torch::Device& expected) {
    if (!model || model->parameters().empty()) {
        throw std::invalid_argument("learner has no parameters");
    }
    for (const auto& parameter : model->named_parameters()) {
        require_tensor_device(parameter.value(), expected, "learner parameter " + parameter.key());
    }
    for (const auto& buffer : model->named_buffers()) {
        require_tensor_device(buffer.value(), expected, "learner buffer " + buffer.key());
    }
}

torch::Tensor finite_flag(std::span<const torch::Tensor> tensors, const torch::Device& expected,
                          const char* name) {
    if (tensors.empty())
        throw std::invalid_argument(std::string(name) + " is empty");
    torch::Tensor result;
    for (const torch::Tensor& tensor : tensors) {
        require_tensor_device(tensor, expected, name);
        const auto tensor_finite = torch::isfinite(tensor).all();
        result = result.defined() ? torch::logical_and(result, tensor_finite) : tensor_finite;
    }
    return result;
}

void require_finite_flag(const torch::Tensor& flag, const char* message) {
    if (!flag.item<bool>())
        throw std::invalid_argument(message);
}

void require_gradients(const diamond_model::DiamondModel& model, const torch::Device& expected) {
    for (const auto& parameter : model->named_parameters()) {
        const torch::Tensor gradient = parameter.value().grad();
        require_tensor_device(gradient, expected, "learner gradient " + parameter.key());
    }
}

void require_optimizer_state(const torch::optim::AdamW& optimizer, const torch::Device& expected) {
    if (optimizer.state().size() != optimizer.size()) {
        throw std::invalid_argument("AdamW state does not cover every learner parameter");
    }
    for (const auto& entry : optimizer.state()) {
        const auto* state = dynamic_cast<const torch::optim::AdamWParamState*>(entry.second.get());
        if (state == nullptr || state->step() <= 0) {
            throw std::invalid_argument("learner optimizer contains invalid AdamW state");
        }
        require_tensor_device(state->exp_avg(), expected, "AdamW exp_avg");
        require_tensor_device(state->exp_avg_sq(), expected, "AdamW exp_avg_sq");
        if (state->max_exp_avg_sq().defined()) {
            require_tensor_device(state->max_exp_avg_sq(), expected, "AdamW max_exp_avg_sq");
        }
    }
}

torch::optim::AdamWOptions adamw_options(const TrainingConfig& config) {
    validate_config(config);
    return torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay);
}

NamedTensors collect_named_tensors(const diamond_model::DiamondModel& model, bool parameters) {
    if (!model)
        throw std::invalid_argument("model snapshot source is empty");

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
    if (parameters.empty())
        throw std::invalid_argument("model snapshot source has no parameters");
    return parameters.begin()->second.device();
}

void require_expected_architecture(const diamond_model::DiamondModel& model,
                                   const Compatibility& compatibility) {
    if (!model)
        throw std::invalid_argument("model snapshot source is empty");
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
                                    const torch::Device& destination_device, const char* kind) {
    if (source.size() != destination.size())
        throw std::invalid_argument(std::string("model snapshot ") + kind + " names differ");

    for (const auto& [name, source_tensor] : source) {
        const auto destination_it = destination.find(name);
        if (destination_it == destination.end()) {
            throw std::invalid_argument(std::string("model snapshot is missing ") + kind + ": " +
                                        name);
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
    if (!tensor.defined())
        throw std::invalid_argument("cannot digest an undefined tensor: " + name);
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
        if (dimension < 0)
            throw std::invalid_argument("cannot digest a tensor with a negative dimension");
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
    if (element_count < 0 || static_cast<uint64_t>(element_count) >
                                 std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        throw std::invalid_argument("model tensor is too large to digest: " + name);
    }
    append_u64_le(stream, static_cast<uint64_t>(element_count) * sizeof(float));
    const float* values = cpu.data_ptr<float>(); // cpu is explicit; never dereference CUDA storage.
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
    if (trainable)
        snapshot->train();
    else
        snapshot->eval();
    for (auto& parameter : snapshot->named_parameters()) {
        parameter.value().set_requires_grad(trainable);
    }
    return snapshot;
}

void zero_value_head(const diamond_model::DiamondModel& model) {
    if (!model)
        throw std::invalid_argument("zero_value_head requires a model");
    torch::NoGradGuard no_grad;
    model->value_linear2->weight.zero_();
    if (model->value_linear2->bias.defined())
        model->value_linear2->bias.zero_();
}

std::string canonical_model_digest(const diamond_model::DiamondModel& model) {
    std::string stream;
    append_string(stream, "alphadiamond.canonical-model-digest.v1");
    const auto parameters = collect_named_tensors(model, true);
    const auto buffers = collect_named_tensors(model, false);
    append_u64_le(stream, static_cast<uint64_t>(parameters.size()));
    for (const auto& [name, tensor] : parameters)
        append_tensor(stream, "parameter", name, tensor);
    append_u64_le(stream, static_cast<uint64_t>(buffers.size()));
    for (const auto& [name, tensor] : buffers)
        append_tensor(stream, "buffer", name, tensor);
    return diamond_support::sha256(stream);
}

Trainer::Trainer(diamond_model::DiamondModel model, Compatibility compatibility,
                 TrainingConfig config, const ResolvedDevice& device)
    : model_(snapshot_model(model, compatibility, device.torch_device, ModelRole::learner)),
      compatibility_(std::move(compatibility)), config_(config), device_(device),
      optimizer_(model_->parameters(), adamw_options(config)) {
    validate_resolved_device(device_);
    require_model_device(model_, device_.torch_device);
}

diamond_model::DiamondModel Trainer::candidate_snapshot() const {
    return snapshot_model(model_, compatibility_, device_.torch_device, ModelRole::candidate);
}

TrainingMetrics Trainer::train(std::span<const TrainingSample> samples) {
    const auto total_start = StepClock::now();
    if (samples.empty()) throw std::invalid_argument("training batch must not be empty");
    if (samples.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("training batch is too large");

    const int64_t batch_size = static_cast<int64_t>(samples.size());
    const int64_t input_features = model_->input_features();
    const int64_t value_size = model_->value_size();
    const size_t feature_count = checked_product(
        static_cast<size_t>(kHoleCount), checked_count(input_features, "model input features"),
        "training sample feature width");
    const size_t value_count = checked_count(value_size, "model value size");
    const size_t batch_count = static_cast<size_t>(batch_size);
    for (const TrainingSample& sample : samples) {
        validate_sample(sample, compatibility_, feature_count, value_count);
    }

    std::vector<float> feature_buffer(
        checked_product(batch_count, feature_count, "training feature buffer"));
    std::vector<float> policy_buffer(
        checked_product(batch_count, static_cast<size_t>(kActionCount), "training policy buffer"),
        0.0F);
    std::vector<float> value_buffer(
        checked_product(batch_count, value_count, "training value buffer"));

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const TrainingSample& sample = samples[static_cast<size_t>(batch)];
        const size_t batch_offset = static_cast<size_t>(batch);
        std::memcpy(feature_buffer.data() + batch_offset * feature_count,
                    sample.node_features.data(), feature_count * sizeof(float));
        std::memcpy(value_buffer.data() + batch_offset * value_count, sample.value_target.data(),
                    value_count * sizeof(float));
        for (const auto& [action, probability] : sample.sparse_policy) {
            policy_buffer[batch_offset * static_cast<size_t>(kActionCount) +
                          static_cast<size_t>(action)] = probability;
        }
    }

    const auto host_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto host_features = torch::from_blob(
        feature_buffer.data(), {batch_size, kHoleCount, input_features}, host_options);
    const auto host_policy_targets =
        torch::from_blob(policy_buffer.data(), {batch_size, kActionCount}, host_options);
    const auto host_value_targets =
        torch::from_blob(value_buffer.data(), {batch_size, value_size}, host_options);
    if (!host_features.is_contiguous() || !host_policy_targets.is_contiguous() ||
        !host_value_targets.is_contiguous()) {
        throw std::invalid_argument("training host tensors must be contiguous");
    }
    const auto collation_end = StepClock::now();

    require_model_device(model_, device_.torch_device);
    const bool is_cuda = device_.torch_device.is_cuda();
    const c10::DeviceIndex device_index = device_.torch_device.index();
    bool peak_cuda_memory_available = false;
#if defined(DIAMOND_TORCH_WITH_CUDA)
    if (is_cuda) {
        try {
            c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);
            peak_cuda_memory_available = true;
        } catch (const c10::Error&) {
            peak_cuda_memory_available = false;
        }
    }
#endif

    std::optional<AcceleratorStepEvents> events;
    if (is_cuda) {
        events.emplace(device_index);
        events->h2d_start.record(events->stream);
    }
    const auto h2d_start = StepClock::now();
    auto features = host_features.to(device_.torch_device);
    auto policy_targets = host_policy_targets.to(device_.torch_device);
    auto value_targets = host_value_targets.to(device_.torch_device);
    const auto h2d_end = StepClock::now();
    if (events)
        events->h2d_end.record(events->stream);
    require_tensor_device(features, device_.torch_device, "training features");
    require_tensor_device(policy_targets, device_.torch_device, "training policy targets");
    require_tensor_device(value_targets, device_.torch_device, "training value targets");

    optimizer_.zero_grad();
    if (events)
        events->forward_start.record(events->stream);
    const auto forward_start = StepClock::now();
    auto [policy_logits, predicted_values] = model_->forward(features);
    require_tensor_device(policy_logits, device_.torch_device, "learner policy logits");
    require_tensor_device(predicted_values, device_.torch_device, "learner predicted values");
    if (policy_logits.sizes() != torch::IntArrayRef({batch_size, kActionCount}) ||
        predicted_values.sizes() != torch::IntArrayRef({batch_size, value_size})) {
        throw std::invalid_argument("learner output shape mismatch");
    }
    auto policy_loss = -(policy_targets * torch::log_softmax(policy_logits, 1)).sum(1).mean();
    auto value_loss = torch::mse_loss(predicted_values, value_targets);
    auto total_loss = policy_loss + value_loss;
    const std::array forward_tensors{policy_loss, value_loss, total_loss};
    const auto forward_finite =
        finite_flag(forward_tensors, device_.torch_device, "learner forward tensor");
    const auto forward_end = StepClock::now();
    if (events)
        events->forward_end.record(events->stream);
    require_finite_flag(forward_finite, "training produced a non-finite forward result");

    if (events)
        events->backward_start.record(events->stream);
    const auto backward_start = StepClock::now();
    total_loss.backward();
    require_gradients(model_, device_.torch_device);
    const auto backward_end = StepClock::now();
    if (events)
        events->backward_end.record(events->stream);

    if (events)
        events->optimizer_start.record(events->stream);
    const auto optimizer_start = StepClock::now();
    optimizer_.step();
    require_model_device(model_, device_.torch_device);
    require_optimizer_state(optimizer_, device_.torch_device);
    const auto optimizer_end = StepClock::now();
    if (events) {
        events->optimizer_end.record(events->stream);
        events->optimizer_end.synchronize();
    }
    ++training_step_;

    uint64_t peak_cuda_allocated_bytes = 0;
    uint64_t peak_cuda_reserved_bytes = 0;
#if defined(DIAMOND_TORCH_WITH_CUDA)
    if (peak_cuda_memory_available) {
        try {
            const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
            constexpr size_t aggregate =
                static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
            if (stats.allocated_bytes[aggregate].peak < 0 ||
                stats.reserved_bytes[aggregate].peak < 0) {
                throw std::invalid_argument("CUDA allocator returned negative peak memory");
            }
            peak_cuda_allocated_bytes =
                static_cast<uint64_t>(stats.allocated_bytes[aggregate].peak);
            peak_cuda_reserved_bytes = static_cast<uint64_t>(stats.reserved_bytes[aggregate].peak);
        } catch (const c10::Error&) {
            peak_cuda_memory_available = false;
        }
    }
#endif

    const auto total_end = StepClock::now();
    const double total_step_seconds = seconds_between(total_start, total_end);
    if (!std::isfinite(total_step_seconds) || total_step_seconds <= 0.0) {
        throw std::invalid_argument("training step duration is not positive and finite");
    }

    const double h2d_seconds = events ? event_seconds(events->h2d_start, events->h2d_end)
                                      : seconds_between(h2d_start, h2d_end);
    const double forward_seconds = events
                                       ? event_seconds(events->forward_start, events->forward_end)
                                       : seconds_between(forward_start, forward_end);
    const double backward_seconds =
        events ? event_seconds(events->backward_start, events->backward_end)
               : seconds_between(backward_start, backward_end);
    const double optimizer_seconds =
        events ? event_seconds(events->optimizer_start, events->optimizer_end)
               : seconds_between(optimizer_start, optimizer_end);

    return TrainingMetrics{
        .total_loss = total_loss.item<double>(),
        .policy_loss = policy_loss.item<double>(),
        .value_loss = value_loss.item<double>(),
        .training_step = training_step_,
        .collation_seconds = seconds_between(total_start, collation_end),
        .h2d_seconds = h2d_seconds,
        .forward_seconds = forward_seconds,
        .backward_seconds = backward_seconds,
        .optimizer_seconds = optimizer_seconds,
        .total_step_seconds = total_step_seconds,
        .samples_per_second = static_cast<double>(batch_size) / total_step_seconds,
        .peak_cuda_allocated_bytes = peak_cuda_allocated_bytes,
        .peak_cuda_reserved_bytes = peak_cuda_reserved_bytes,
        .peak_cuda_memory_available = peak_cuda_memory_available,
    };
}

}  // namespace diamond_training
