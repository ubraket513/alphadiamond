#include "diamond_pipeline/model_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "diamond_training/checkpoint.hpp"

namespace diamond_pipeline {

ModelPool::ModelPool(std::size_t capacity, diamond_training::ResolvedDevice device)
    : capacity_(capacity), device_(std::move(device)) {
    if (capacity == 0) throw std::invalid_argument("model pool capacity must be positive");
}

ModelKey ModelPool::install(const Compatibility& compatibility,
                            const diamond_model::DiamondModel& source) {
    compatibility.validate();
    if (!source) throw std::invalid_argument("model pool requires a model");
    auto actor = diamond_training::snapshot_model(source, compatibility, device_.torch_device,
                                                  diamond_training::ModelRole::actor);
    ModelKey key{compatibility.model_name, compatibility.model_version,
                 diamond_training::canonical_model_digest(actor)};
    if (!models_.contains(key) && models_.size() == capacity_)
        throw PipelineError("model pool residency capacity is exhausted");
    models_.insert_or_assign(key, ResidentModel{compatibility, std::move(actor)});
    return key;
}

ModelKey ModelPool::install_checkpoint(const Compatibility& compatibility,
                                       const std::filesystem::path& checkpoint_root,
                                       diamond_model::DiamondModel staging) {
    if (!staging) throw std::invalid_argument("model pool requires a checkpoint staging model");
    try {
        (void)diamond_training::load_checkpoint_v2_weights(checkpoint_root, staging, device_);
    } catch (const diamond_training::CheckpointError& error) {
        throw IncompatibleCheckpointError(error.what());
    }
    return install(compatibility, staging);
}

void ModelPool::activate(const ModelKey& key) {
    if (!models_.contains(key)) throw PipelineError("requested model key is not resident");
    active_ = key;
}

const ModelKey& ModelPool::active_key() const {
    if (!active_) throw PipelineError("no active model key");
    return *active_;
}

const diamond_model::DiamondModel& ModelPool::active_model() const {
    return models_.at(active_key()).actor;
}

std::size_t ModelPool::resident_count() const { return models_.size(); }

void ModelPool::require_compatible(const Compatibility& expected) const {
    expected.validate();
    if (!(models_.at(active_key()).compatibility == expected))
        throw IncompatibleCheckpointError("active model compatibility does not match request");
}

void ModelPool::require_ready(std::stop_token stop,
                              std::chrono::steady_clock::time_point deadline) const {
    if (stop.stop_requested()) throw CancelledError("native pipeline cancelled");
    if (std::chrono::steady_clock::now() >= deadline)
        throw DeadlineExceededError("native pipeline deadline exceeded");
    (void)active_key();
}

void ModelPool::evaluate(std::vector<soo::BatchItem>& batch) {
    if (batch.empty()) {
        last_evaluation_stats_ = {};
        return;
    }

    constexpr int64_t kBoardNodes = 73;
    constexpr int64_t kActionSpace = kBoardNodes * kBoardNodes;
    constexpr int64_t kOutcomeValueCapacity = 3;

    auto& resident = models_.at(active_key());
    auto& model = resident.actor;
    const int64_t features_per_node = model->input_features();
    const int64_t value_width = model->value_size();
    const int64_t family_value_width = resident.compatibility.family() == "soo" ? 1 : 3;
    if (value_width != family_value_width || value_width <= 0 ||
        value_width > kOutcomeValueCapacity) {
        throw PipelineError("active model value width is incompatible with its output family");
    }

    std::size_t max_legal_actions = 0;
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(kActionSpace), 0);
    for (const soo::BatchItem& item : batch) {
        if (!item.encoded || !item.actions || !item.outcome)
            throw PipelineError("malformed native inference batch item");
        if (item.encoded->feature_count != features_per_node ||
            item.encoded->node_features.size() !=
                static_cast<std::size_t>(kBoardNodes * features_per_node)) {
            throw PipelineError("native inference feature shape is incompatible with the active model");
        }
        if (!std::all_of(item.encoded->node_features.begin(), item.encoded->node_features.end(),
                         [](float value) { return std::isfinite(value); })) {
            throw PipelineError("native inference features must be finite");
        }
        if (item.actions->empty())
            throw PipelineError("native inference requires at least one legal action");
        if (item.value_width != value_width)
            throw PipelineError("model value width is incompatible with self-play match");

        std::fill(seen.begin(), seen.end(), std::uint8_t{0});
        for (const int32_t action : *item.actions) {
            if (action < 0 || action >= kActionSpace)
                throw PipelineError("native inference legal action is out of range");
            auto& was_seen = seen[static_cast<std::size_t>(action)];
            if (was_seen != 0)
                throw PipelineError("native inference legal actions must be unique");
            was_seen = 1;
        }
        max_legal_actions = std::max(max_legal_actions, item.actions->size());
    }

    const auto batch_size = static_cast<int64_t>(batch.size());
    const auto max_legal = static_cast<int64_t>(max_legal_actions);
    const auto feature_row_size = static_cast<std::size_t>(kBoardNodes * features_per_node);
    std::vector<float> feature_buffer(batch.size() * feature_row_size);
    std::vector<int64_t> legal_index_buffer(batch.size() * max_legal_actions, 0);
    std::vector<std::uint8_t> valid_mask_buffer(batch.size() * max_legal_actions, 0);

    for (std::size_t row = 0; row < batch.size(); ++row) {
        const auto& item = batch[row];
        std::copy(item.encoded->node_features.begin(), item.encoded->node_features.end(),
                  feature_buffer.begin() + static_cast<std::ptrdiff_t>(row * feature_row_size));
        const std::size_t legal_offset = row * max_legal_actions;
        for (std::size_t column = 0; column < item.actions->size(); ++column) {
            legal_index_buffer[legal_offset + column] = (*item.actions)[column];
            valid_mask_buffer[legal_offset + column] = 1;
        }
    }

    const auto cpu_options = torch::TensorOptions().device(torch::kCPU);
    const auto host_features = torch::from_blob(
        feature_buffer.data(), {batch_size, kBoardNodes, features_per_node},
        cpu_options.dtype(torch::kFloat32));
    const auto host_legal_indices = torch::from_blob(
        legal_index_buffer.data(), {batch_size, max_legal}, cpu_options.dtype(torch::kInt64));
    const auto host_valid_mask = torch::from_blob(
        valid_mask_buffer.data(), {batch_size, max_legal}, cpu_options.dtype(torch::kUInt8));

    EvaluationStats stats{.batch_size = batch.size(),
                          .max_legal_actions = max_legal_actions};
    auto device_features = host_features;
    auto device_legal_indices = host_legal_indices;
    auto device_valid_mask = host_valid_mask;
    if (device_.torch_device.is_cuda()) {
        device_features = host_features.to(device_.torch_device);
        ++stats.h2d_transfers;
        device_legal_indices = host_legal_indices.to(device_.torch_device);
        ++stats.h2d_transfers;
        device_valid_mask = host_valid_mask.to(device_.torch_device);
        ++stats.h2d_transfers;
    }

    torch::NoGradGuard no_grad;
    ++stats.forward_calls;
    const auto [logits, values] = model->forward(device_features);
    if (logits.dim() != 2 || logits.size(0) != batch_size ||
        logits.size(1) != kActionSpace || values.dim() != 2 ||
        values.size(0) != batch_size || values.size(1) != value_width ||
        logits.scalar_type() != torch::kFloat32 || values.scalar_type() != torch::kFloat32 ||
        logits.device() != device_.torch_device || values.device() != device_.torch_device) {
        throw PipelineError("active model produced an incompatible inference output");
    }

    const auto legal_logits = logits.gather(1, device_legal_indices);
    const auto padded_priors = torch::softmax(
        legal_logits.masked_fill(device_valid_mask.eq(0),
                                 -std::numeric_limits<float>::infinity()),
        1);
    const auto finite_rows = torch::logical_and(
        torch::isfinite(padded_priors).all(1), torch::isfinite(values).all(1));
    const auto compact_device =
        torch::cat({padded_priors, values, finite_rows.unsqueeze(1).to(torch::kFloat32)}, 1)
            .contiguous();

    auto compact_cpu = compact_device;
    if (device_.torch_device.is_cuda()) {
        compact_cpu = compact_device.to(torch::kCPU);
        ++stats.d2h_transfers;
    }
    if (!compact_cpu.is_contiguous() || compact_cpu.scalar_type() != torch::kFloat32 ||
        !compact_cpu.device().is_cpu()) {
        throw PipelineError("native inference compact output is not a contiguous CPU float tensor");
    }

    const auto compact_width = max_legal + value_width + 1;
    const float* compact = compact_cpu.data_ptr<float>();
    for (std::size_t row = 0; row < batch.size(); ++row) {
        const std::size_t offset = row * static_cast<std::size_t>(compact_width);
        if (compact[offset + static_cast<std::size_t>(max_legal + value_width)] < 0.5F)
            throw PipelineError("native model produced a non-finite inference row");
    }

    std::vector<soo::EvalOutcome> staged_outcomes;
    staged_outcomes.reserve(batch.size());
    for (std::size_t row = 0; row < batch.size(); ++row) {
        const auto& item = batch[row];
        const std::size_t offset = row * static_cast<std::size_t>(compact_width);
        soo::EvalOutcome staged = *item.outcome;
        staged.priors.resize(item.actions->size());
        for (std::size_t column = 0; column < item.actions->size(); ++column) {
            const float prior = compact[offset + column];
            if (!std::isfinite(prior))
                throw PipelineError("native model produced a non-finite policy prior");
            staged.priors[column] = prior;
        }
        for (int64_t value = 0; value < value_width; ++value) {
            const float component = compact[offset + max_legal_actions +
                                            static_cast<std::size_t>(value)];
            if (!std::isfinite(component))
                throw PipelineError("native model produced a non-finite value");
            staged.values[static_cast<std::size_t>(value)] = component;
        }
        staged.value = staged.values[0];
        staged_outcomes.push_back(std::move(staged));
    }

    for (std::size_t row = 0; row < batch.size(); ++row)
        *batch[row].outcome = std::move(staged_outcomes[row]);
    last_evaluation_stats_ = stats;
}

}  // namespace diamond_pipeline
