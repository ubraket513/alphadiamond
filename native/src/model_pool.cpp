#include "diamond_pipeline/model_pool.hpp"

#include <algorithm>
#include <cmath>

#include "diamond_training/checkpoint.hpp"

namespace diamond_pipeline {

ModelPool::ModelPool(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) throw std::invalid_argument("model pool capacity must be positive");
}

void ModelPool::install(ModelKey key, diamond_model::DiamondModel model) {
    if (!model) throw std::invalid_argument("model pool requires a model");
    if (key.model_name.empty() || key.model_version.empty() || key.checkpoint_sha256.size() != 64)
        throw IncompatibleCheckpointError("model key must contain a full v2 checkpoint identity");
    if (!models_.contains(key) && models_.size() == capacity_)
        throw PipelineError("model pool residency capacity is exhausted");
    model->eval();
    models_.insert_or_assign(std::move(key), std::move(model));
}

void ModelPool::install_checkpoint(ModelKey key, const std::filesystem::path& checkpoint_root,
                                   diamond_model::DiamondModel model) {
    try {
        (void)diamond_training::inspect_checkpoint_v2(checkpoint_root);
    } catch (const diamond_training::CheckpointError& error) {
        throw IncompatibleCheckpointError(error.what());
    }
    install(std::move(key), std::move(model));
}

void ModelPool::activate(const ModelKey& key) {
    if (!models_.contains(key)) throw PipelineError("requested model key is not resident");
    active_ = key;
}

const ModelKey& ModelPool::active_key() const {
    if (!active_) throw PipelineError("no active model key");
    return *active_;
}

std::size_t ModelPool::resident_count() const { return models_.size(); }

void ModelPool::require_ready(std::stop_token stop,
                              std::chrono::steady_clock::time_point deadline) const {
    if (stop.stop_requested()) throw CancelledError("native pipeline cancelled");
    if (std::chrono::steady_clock::now() >= deadline)
        throw DeadlineExceededError("native pipeline deadline exceeded");
    (void)active_key();
}

void ModelPool::evaluate(std::vector<soo::BatchItem>& batch) {
    if (batch.empty()) return;
    auto& model = models_.at(active_key());
    const int64_t features_per_node = model->input_features();
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> rows;
    rows.reserve(batch.size());
    for (const soo::BatchItem& item : batch) {
        if (!item.encoded || !item.actions || !item.outcome ||
            item.encoded->feature_count != features_per_node || item.actions->empty())
            throw PipelineError("malformed native inference batch item");
        rows.push_back(torch::from_blob(const_cast<float*>(item.encoded->node_features.data()),
                                        {73, features_per_node}, torch::kFloat32).clone());
    }
    const auto input = torch::stack(rows);
    const auto [logits, values] = model->forward(input);
    for (std::size_t row = 0; row < batch.size(); ++row) {
        auto& item = batch[row];
        const auto indices = torch::tensor(*item.actions, torch::TensorOptions().dtype(torch::kLong));
        const auto policy = torch::softmax(logits[row].index_select(0, indices), 0).contiguous();
        const float* priors = policy.data_ptr<float>();
        item.outcome->priors.assign(priors, priors + item.actions->size());
        if (item.value_width != values.size(1))
            throw PipelineError("model value width is incompatible with self-play match");
        const auto row_values = values[row].contiguous();
        const float* native_values = row_values.data_ptr<float>();
        item.outcome->value = native_values[0];
        for (int64_t value = 0; value < values.size(1); ++value)
            item.outcome->values[static_cast<std::size_t>(value)] = native_values[value];
        if (!std::isfinite(item.outcome->value)) throw PipelineError("native model produced non-finite value");
    }
}

}  // namespace diamond_pipeline
