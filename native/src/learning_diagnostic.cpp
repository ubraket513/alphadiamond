#include "diamond_pipeline/learning_diagnostic.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "soo/board.hpp"

namespace diamond_pipeline {
namespace {
struct Evaluation {
    HeldOutMetrics metrics;
    torch::Tensor logits;
    torch::Tensor values;
};
Evaluation evaluate(diamond_training::Trainer& trainer,
                    const std::vector<diamond_training::TrainingSample>& samples,
                    std::size_t batch_size) {
    const int64_t rows = static_cast<int64_t>(samples.size());
    const int64_t features = trainer.model()->input_features();
    const int64_t values = trainer.model()->value_size();
    std::vector<float> feature_data, policy_data(rows * soo::kActionSize, 0.0F), value_data;
    feature_data.reserve(rows * soo::kBoardSize * features);
    value_data.reserve(rows * values);
    double entropy = 0.0;
    for (int64_t row = 0; row < rows; ++row) {
        const auto& sample = samples[static_cast<std::size_t>(row)];
        feature_data.insert(feature_data.end(), sample.node_features.begin(),
                            sample.node_features.end());
        value_data.insert(value_data.end(), sample.value_target.begin(), sample.value_target.end());
        for (const auto& [action, probability] : sample.sparse_policy) {
            policy_data[static_cast<std::size_t>(row) * soo::kActionSize + action] = probability;
            if (probability > 0.0F)
                entropy -= probability * std::log(probability);
        }
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto x =
        torch::from_blob(feature_data.data(), {rows, soo::kBoardSize, features}, options).clone();
    auto targets = torch::from_blob(policy_data.data(), {rows, soo::kActionSize}, options).clone();
    auto value_targets = torch::from_blob(value_data.data(), {rows, values}, options).clone();
    torch::NoGradGuard guard;
    trainer.model()->eval();
    std::vector<torch::Tensor> logits_batches, value_batches;
    for (int64_t begin = 0; begin < rows; begin += static_cast<int64_t>(batch_size)) {
        const auto end = std::min(rows, begin + static_cast<int64_t>(batch_size));
        auto [batch_logits, batch_values] =
            trainer.model()->forward(x.slice(0, begin, end).to(trainer.device().torch_device));
        logits_batches.push_back(batch_logits.detach().to(torch::kCPU));
        value_batches.push_back(batch_values.detach().to(torch::kCPU));
    }
    auto logits = torch::cat(logits_batches, 0);
    auto predicted = torch::cat(value_batches, 0);
    auto log_probs = torch::log_softmax(logits, 1);
    const double cross_entropy = -(targets * log_probs).sum(1).mean().item<double>();
    const double agreement =
        logits.argmax(1).eq(targets.argmax(1)).to(torch::kFloat32).mean().item<double>();
    return {.metrics = {.target_entropy = entropy / rows,
                        .full_cross_entropy = cross_entropy,
                        .full_kl = std::max(0.0, cross_entropy - entropy / rows),
                        .top1_agreement = agreement,
                        .value_mse = torch::mse_loss(predicted, value_targets).item<double>()},
            .logits = logits.detach().clone(),
            .values = predicted.detach().clone()};
}
} // namespace

LearningDiagnosticResult run_min_learning_diagnostic(diamond_training::Trainer& trainer,
                                                     const ReplayStore& replay,
                                                     const LearningDiagnosticConfig& config) {
    if (trainer.compatibility().model_name != "Min")
        throw std::invalid_argument("learning diagnostic requires Min");
    if (!config.steps || !config.batch_size || !config.evaluation_samples ||
        !config.evaluation_batch)
        throw std::invalid_argument("learning diagnostic counts must be non-zero");
    const auto held_out = replay.sample(config.evaluation_samples, config.seed ^ 0xd1a69057ULL);
    auto start = evaluate(trainer, held_out, config.evaluation_batch);
    LearningDiagnosticResult result;
    result.initial = start.metrics;
    const int64_t blocks = trainer.compatibility().network_config.residual_blocks;
    for (std::size_t step = 0; step < config.steps; ++step) {
        std::unordered_map<std::string, torch::Tensor> before;
        for (const auto& parameter : trainer.model()->named_parameters())
            before.emplace(parameter.key(), parameter.value().detach().clone());
        auto samples = replay.sample(
            config.batch_size, replay_sampling_seed(replay.replay_seed(), config.iteration, step));
        const auto losses = trainer.train(samples);
        std::map<diamond_training::ParameterGroup, std::array<double, 3>> squares;
        for (const auto group : {diamond_training::ParameterGroup::input_projection,
                                 diamond_training::ParameterGroup::residual_trunk,
                                 diamond_training::ParameterGroup::last_residual_block,
                                 diamond_training::ParameterGroup::output_norm,
                                 diamond_training::ParameterGroup::policy_source,
                                 diamond_training::ParameterGroup::policy_destination,
                                 diamond_training::ParameterGroup::value_hidden,
                                 diamond_training::ParameterGroup::value_output})
            squares.emplace(group, std::array<double, 3>{});
        for (const auto& parameter : trainer.model()->named_parameters()) {
            if (!parameter.value().grad().defined())
                throw std::runtime_error("undefined diagnostic gradient");
            const auto group = diamond_training::classify_parameter(parameter.key(), blocks);
            auto& sums = squares[group];
            sums[0] += parameter.value().detach().to(torch::kFloat64).pow(2).sum().item<double>();
            sums[1] +=
                parameter.value().grad().detach().to(torch::kFloat64).pow(2).sum().item<double>();
            sums[2] += (parameter.value().detach() - before.at(parameter.key()))
                           .to(torch::kFloat64)
                           .pow(2)
                           .sum()
                           .item<double>();
        }
        const bool retain = step < 2 || step + 1 == config.steps ||
                            (config.log_every && (step + 1) % config.log_every == 0);
        if (retain) {
            LearningStepDiagnostic row{.local_step = step + 1, .losses = losses};
            for (const auto& [group, sums] : squares) {
                auto& norms = row.groups[group];
                norms.parameter_l2 = std::sqrt(sums[0]);
                norms.gradient_l2 = std::sqrt(sums[1]);
                norms.update_l2 = std::sqrt(sums[2]);
                norms.relative_update =
                    norms.parameter_l2 > 0 ? norms.update_l2 / norms.parameter_l2 : 0.0;
                if (!std::isfinite(norms.parameter_l2 + norms.gradient_l2 + norms.update_l2 +
                                   norms.relative_update))
                    throw std::runtime_error("non-finite diagnostic norm");
            }
            if (row.groups.size() != 8)
                throw std::runtime_error("missing diagnostic parameter group");
            result.steps.push_back(std::move(row));
        }
    }
    auto finish = evaluate(trainer, held_out, config.evaluation_batch);
    result.final = finish.metrics;
    auto start_probs = torch::softmax(start.logits, 1);
    result.drift.policy_kl =
        (start_probs * (torch::log_softmax(start.logits, 1) - torch::log_softmax(finish.logits, 1)))
            .sum(1)
            .mean()
            .item<double>();
    result.drift.logit_rms_delta =
        torch::sqrt(torch::mean(torch::pow(finish.logits - start.logits, 2))).item<double>();
    result.drift.value_rms_delta =
        torch::sqrt(torch::mean(torch::pow(finish.values - start.values, 2))).item<double>();
    result.drift.top1_agreement_delta = result.final.top1_agreement - result.initial.top1_agreement;
    result.drift.full_cross_entropy_delta =
        result.final.full_cross_entropy - result.initial.full_cross_entropy;
    result.drift.value_mse_delta = result.final.value_mse - result.initial.value_mse;
    return result;
}
} // namespace diamond_pipeline
