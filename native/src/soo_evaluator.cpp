#include "diamond_model/soo_evaluator.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace diamond_model {

SooEvaluator::SooEvaluator(SooModel model) : model_(std::move(model)) {
    if (!model_) throw std::invalid_argument("SooEvaluator requires a model");
    model_->eval();
}

soo::EvalOutcome SooEvaluator::evaluate(const soo::Encoded& encoded,
                                        const std::vector<int32_t>& legal_actions) {
    if (encoded.feature_count != 4 || encoded.node_features.size() != 73U * 4U) {
        throw std::invalid_argument("Soo evaluator requires one [73,4] encoded state");
    }
    if (legal_actions.empty()) throw std::invalid_argument("Soo evaluator requires legal actions");
    for (const auto action : legal_actions) {
        if (action < 0 || action >= 73 * 73) {
            throw std::invalid_argument("legal action is outside the Soo policy space");
        }
    }

    torch::NoGradGuard no_grad;
    auto features = torch::from_blob(
                         const_cast<float*>(encoded.node_features.data()), {1, 73, 4}, torch::kFloat32)
                         .clone();
    const auto [policy, value] = model_->forward(features);
    const auto action_index = torch::tensor(legal_actions, torch::TensorOptions().dtype(torch::kLong));
    const auto legal_logits = policy.index_select(1, action_index);
    const auto probabilities = torch::softmax(legal_logits, 1).contiguous().view(-1);
    const auto values = probabilities.data_ptr<float>();
    soo::EvalOutcome outcome;
    outcome.priors.reserve(legal_actions.size());
    for (size_t i = 0; i < legal_actions.size(); ++i) {
        if (!std::isfinite(values[i])) throw std::runtime_error("Soo produced a non-finite prior");
        outcome.priors.push_back(static_cast<double>(values[i]));
    }
    outcome.value = static_cast<double>(value.item<float>());
    if (!std::isfinite(outcome.value)) throw std::runtime_error("Soo produced a non-finite value");
    return outcome;
}

}  // namespace diamond_model
