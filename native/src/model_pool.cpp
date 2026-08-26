#include "diamond_pipeline/model_pool.hpp"

#include <algorithm>
#include <optional>
#include <chrono>
#include <cstdlib>
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
    if (!source)
        throw std::invalid_argument("model pool requires a model");
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
    if (!staging)
        throw std::invalid_argument("model pool requires a checkpoint staging model");
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
            throw PipelineError(
                "native inference feature shape is incompatible with the active model");
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

    using EvalClock = std::chrono::steady_clock;
    const auto host_seconds = [](EvalClock::time_point a, EvalClock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    // One event pair per stage. Constructed only on CUDA; on CPU every stage
    // falls back to the host clock, which is exact there because nothing is
    // enqueued asynchronously.
    struct StageEvents {
        explicit StageEvents(const c10::Stream& s)
            : stream(s),
              h2d_start(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
              h2d_end(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
              forward_start(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
              forward_end(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
              post_start(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
              post_end(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
              d2h_start(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT),
              d2h_end(c10::kCUDA, c10::EventFlag::BACKEND_DEFAULT) {}
        c10::Stream stream;
        c10::Event h2d_start, h2d_end;
        c10::Event forward_start, forward_end;
        c10::Event post_start, post_end;
        c10::Event d2h_start, d2h_end;
    };
    // Opt-in. Reading a CUDA event requires synchronising on it, which blocks
    // the evaluator thread until the GPU drains and serialises what is
    // otherwise a pipelined stream of batches. Measured cost of leaving it on:
    // evaluations/s fell from 77.7k to 49.9k, a third of throughput. A
    // measurement that changes the thing it measures has to be something you
    // ask for, so production runs pay nothing and the split is available when
    // DIAMOND_EVAL_STAGE_TIMING is set.
    static const bool stage_timing = [] {
        const char* flag = std::getenv("DIAMOND_EVAL_STAGE_TIMING");
        return flag != nullptr && *flag != '\0' && *flag != '0';
    }();
    const bool on_cuda = device_.torch_device.is_cuda() && stage_timing;
    std::optional<StageEvents> events;
    if (on_cuda)
        events.emplace(c10::impl::getDeviceGuardImpl(c10::kCUDA)->getStream(device_.torch_device));
    const auto event_seconds = [](const c10::Event& a, c10::Event& b) {
        b.synchronize();
        return a.elapsedTime(b) / 1000.0;
    };
    const auto collation_start = EvalClock::now();

    const auto batch_size = static_cast<int64_t>(batch.size());
    const auto max_legal = static_cast<int64_t>(max_legal_actions);
    const auto feature_row_size = static_cast<std::size_t>(kBoardNodes * features_per_node);
    const bool cuda_destination = device_.torch_device.is_cuda();
    // Grow the reused staging buffers if this batch needs more than the last.
    // Pinning is only meaningful when the destination is CUDA.
    const auto staging_options =
        torch::TensorOptions().device(torch::kCPU).pinned_memory(cuda_destination);
    const auto ensure = [&staging_options](torch::Tensor& buffer, int64_t rows, int64_t columns,
                                           torch::ScalarType type) {
        if (!buffer.defined() || buffer.size(0) < rows || buffer.size(1) < columns ||
            buffer.scalar_type() != type) {
            buffer = torch::empty({rows, columns}, staging_options.dtype(type));
        }
    };
    const auto feature_columns = static_cast<int64_t>(feature_row_size);
    const auto legal_columns = std::max<int64_t>(max_legal, 1);
    ensure(staging_features_, batch_size, feature_columns, torch::kFloat32);
    ensure(staging_legal_indices_, batch_size, legal_columns, torch::kInt64);
    ensure(staging_valid_mask_, batch_size, legal_columns, torch::kUInt8);

    float* feature_buffer = staging_features_.data_ptr<float>();
    int64_t* legal_index_buffer = staging_legal_indices_.data_ptr<int64_t>();
    std::uint8_t* valid_mask_buffer = staging_valid_mask_.data_ptr<std::uint8_t>();
    const auto feature_stride = static_cast<std::size_t>(staging_features_.size(1));
    const auto legal_stride = static_cast<std::size_t>(staging_legal_indices_.size(1));

    for (std::size_t row = 0; row < batch.size(); ++row) {
        const auto& item = batch[row];
        std::copy(item.encoded->node_features.begin(), item.encoded->node_features.end(),
                  feature_buffer + row * feature_stride);
        const std::size_t legal_offset = row * legal_stride;
        // The buffers outlive a batch, so padding must be cleared rather than
        // assumed zero the way a fresh allocation could be.
        std::fill_n(legal_index_buffer + legal_offset, max_legal_actions, int64_t{0});
        std::fill_n(valid_mask_buffer + legal_offset, max_legal_actions, std::uint8_t{0});
        for (std::size_t column = 0; column < item.actions->size(); ++column) {
            legal_index_buffer[legal_offset + column] = (*item.actions)[column];
            valid_mask_buffer[legal_offset + column] = 1;
        }
    }

    const auto collation_end = EvalClock::now();

    using torch::indexing::Slice;
    const auto host_features =
        staging_features_.index({Slice(0, batch_size), Slice(0, feature_columns)})
            .contiguous()
            .view({batch_size, kBoardNodes, features_per_node});
    const auto host_legal_indices =
        staging_legal_indices_.index({Slice(0, batch_size), Slice(0, max_legal)}).contiguous();
    const auto host_valid_mask =
        staging_valid_mask_.index({Slice(0, batch_size), Slice(0, max_legal)}).contiguous();

    EvaluationStats stats{.batch_size = batch.size(), .max_legal_actions = max_legal_actions};
    auto device_features = host_features;
    auto device_legal_indices = host_legal_indices;
    auto device_valid_mask = host_valid_mask;
    const auto h2d_host_start = EvalClock::now();
    if (device_.torch_device.is_cuda()) {
        if (on_cuda) events->h2d_start.record(events->stream);
        // Non-blocking is only honoured from pinned memory, which is what the
        // staging buffers are for; the forward pass is enqueued on the same
        // stream and therefore stays ordered after these copies.
        device_features = host_features.to(device_.torch_device, /*non_blocking=*/true);
        ++stats.h2d_transfers;
        device_legal_indices = host_legal_indices.to(device_.torch_device, /*non_blocking=*/true);
        ++stats.h2d_transfers;
        device_valid_mask = host_valid_mask.to(device_.torch_device, /*non_blocking=*/true);
        ++stats.h2d_transfers;
        if (on_cuda) events->h2d_end.record(events->stream);
    }
    const auto h2d_host_end = EvalClock::now();

    torch::NoGradGuard no_grad;
    ++stats.forward_calls;
    const auto forward_host_start = EvalClock::now();
    if (on_cuda) events->forward_start.record(events->stream);
    const auto [logits, values] = model->forward(device_features);
    if (on_cuda) events->forward_end.record(events->stream);
    const auto forward_host_end = EvalClock::now();
    if (logits.dim() != 2 || logits.size(0) != batch_size || logits.size(1) != kActionSpace ||
        values.dim() != 2 || values.size(0) != batch_size || values.size(1) != value_width ||
        logits.scalar_type() != torch::kFloat32 || values.scalar_type() != torch::kFloat32 ||
        logits.device() != device_.torch_device || values.device() != device_.torch_device) {
        throw PipelineError("active model produced an incompatible inference output");
    }

    const auto post_host_start = EvalClock::now();
    if (on_cuda) events->post_start.record(events->stream);
    const auto legal_logits = logits.gather(1, device_legal_indices);
    const auto padded_priors = torch::softmax(
        legal_logits.masked_fill(device_valid_mask.eq(0), -std::numeric_limits<float>::infinity()),
        1);
    const auto finite_rows =
        torch::logical_and(torch::isfinite(padded_priors).all(1), torch::isfinite(values).all(1));
    const auto compact_device =
        torch::cat({padded_priors, values, finite_rows.unsqueeze(1).to(torch::kFloat32)}, 1)
            .contiguous();

    if (on_cuda) events->post_end.record(events->stream);
    const auto post_host_end = EvalClock::now();

    const auto d2h_host_start = EvalClock::now();
    auto compact_cpu = compact_device;
    if (device_.torch_device.is_cuda()) {
        if (on_cuda) events->d2h_start.record(events->stream);
        compact_cpu = compact_device.to(torch::kCPU);
        ++stats.d2h_transfers;
        if (on_cuda) events->d2h_end.record(events->stream);
    }
    const auto d2h_host_end = EvalClock::now();
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

    const auto scatter_start = EvalClock::now();
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
            const float component =
                compact[offset + max_legal_actions + static_cast<std::size_t>(value)];
            if (!std::isfinite(component))
                throw PipelineError("native model produced a non-finite value");
            staged.values[static_cast<std::size_t>(value)] = component;
        }
        staged.value = staged.values[0];
        staged_outcomes.push_back(std::move(staged));
    }

    for (std::size_t row = 0; row < batch.size(); ++row)
        *batch[row].outcome = std::move(staged_outcomes[row]);
    const auto scatter_end = EvalClock::now();

    stats.collation_seconds = host_seconds(collation_start, collation_end);
    stats.scatter_seconds = host_seconds(scatter_start, scatter_end);
    if (on_cuda) {
        stats.h2d_seconds = event_seconds(events->h2d_start, events->h2d_end);
        stats.forward_seconds = event_seconds(events->forward_start, events->forward_end);
        stats.policy_postprocess_seconds = event_seconds(events->post_start, events->post_end);
        stats.d2h_seconds = event_seconds(events->d2h_start, events->d2h_end);
    } else {
        stats.h2d_seconds = host_seconds(h2d_host_start, h2d_host_end);
        stats.forward_seconds = host_seconds(forward_host_start, forward_host_end);
        stats.policy_postprocess_seconds = host_seconds(post_host_start, post_host_end);
        stats.d2h_seconds = host_seconds(d2h_host_start, d2h_host_end);
    }
    last_evaluation_stats_ = stats;
    auto& total = accumulated_evaluation_stats_;
    total.forward_calls += stats.forward_calls;
    total.h2d_transfers += stats.h2d_transfers;
    total.d2h_transfers += stats.d2h_transfers;
    total.batch_size += stats.batch_size;
    total.max_legal_actions = std::max(total.max_legal_actions, stats.max_legal_actions);
    total.collation_seconds += stats.collation_seconds;
    total.h2d_seconds += stats.h2d_seconds;
    total.forward_seconds += stats.forward_seconds;
    total.policy_postprocess_seconds += stats.policy_postprocess_seconds;
    total.d2h_seconds += stats.d2h_seconds;
    total.scatter_seconds += stats.scatter_seconds;
    ++evaluated_batches_;
}

}  // namespace diamond_pipeline
