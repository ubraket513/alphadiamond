#include "diamond_pipeline/vacancy_distillation.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "soo/action.hpp"
#include "soo/board.hpp"
#include "soo/prior.hpp"

namespace diamond_pipeline {

namespace {

struct Evaluation {
    VacancyMetrics metrics;
    std::vector<std::vector<double>> probabilities;
};

torch::Tensor feature_tensor(std::span<const diamond_training::TrainingSample> samples,
                             torch::Device device) {
    std::vector<float> values;
    values.reserve(samples.size() * soo::kBoardSize * 6);
    for (const auto& sample : samples) {
        if (sample.node_features.size() != soo::kBoardSize * 6)
            throw std::invalid_argument("vacancy distillation requires Min feature width 6");
        values.insert(values.end(), sample.node_features.begin(), sample.node_features.end());
    }
    return torch::from_blob(values.data(),
                            {static_cast<int64_t>(samples.size()), soo::kBoardSize, 6},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

std::pair<torch::Tensor, torch::Tensor>
teacher_tensors(std::span<const diamond_training::TrainingSample> samples, torch::Device device) {
    std::vector<float> probabilities(samples.size() * soo::kActionSize, 0.0F);
    std::vector<uint8_t> legal(samples.size() * soo::kActionSize, false);
    for (std::size_t row = 0; row < samples.size(); ++row) {
        const auto target = vacancy_target(samples[row]);
        for (std::size_t index = 0; index < target.actions.size(); ++index) {
            const auto offset =
                row * soo::kActionSize + static_cast<std::size_t>(target.actions[index]);
            probabilities[offset] = static_cast<float>(target.probabilities[index]);
            legal[offset] = true;
        }
    }
    auto targets = torch::from_blob(probabilities.data(),
                                    {static_cast<int64_t>(samples.size()), soo::kActionSize},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .clone()
                       .to(device);
    auto mask =
        torch::from_blob(legal.data(), {static_cast<int64_t>(samples.size()), soo::kActionSize},
                         torch::TensorOptions().dtype(torch::kBool))
            .clone()
            .to(device);
    return {targets, mask};
}

torch::Tensor legal_log_probs(const torch::Tensor& logits, const torch::Tensor& mask) {
    return torch::log_softmax(
               logits.masked_fill(mask.logical_not(), -std::numeric_limits<float>::infinity()), 1)
        .masked_fill(mask.logical_not(), 0.0);
}

Evaluation evaluate(diamond_training::Trainer& trainer,
                    std::span<const diamond_training::TrainingSample> samples,
                    std::size_t batch_size) {
    torch::NoGradGuard guard;
    trainer.model()->eval();
    std::vector<torch::Tensor> rows;
    for (std::size_t begin = 0; begin < samples.size(); begin += batch_size) {
        const auto count = std::min(batch_size, samples.size() - begin);
        const auto batch = samples.subspan(begin, count);
        auto [logits, values] =
            trainer.model()->forward(feature_tensor(batch, trainer.device().torch_device));
        (void)values;
        rows.push_back(logits.detach().to(torch::kCPU));
    }
    const auto logits = torch::cat(rows, 0);
    Evaluation result;
    result.probabilities.resize(samples.size());
    double cross_entropy = 0.0;
    double entropy = 0.0;
    double expected = 0.0;
    double teacher_expected = 0.0;
    double agreement = 0.0;
    double top3_mass = 0.0;
    for (std::size_t row = 0; row < samples.size(); ++row) {
        const auto target = vacancy_target(samples[row]);
        std::vector<double> network(target.actions.size());
        double highest = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < target.actions.size(); ++index) {
            network[index] =
                logits[static_cast<int64_t>(row)][target.actions[index]].item<double>();
            highest = std::max(highest, network[index]);
        }
        double total = 0.0;
        for (double& value : network) {
            value = std::exp(value - highest);
            total += value;
        }
        for (double& value : network)
            value /= total;
        result.probabilities[row] = network;

        std::size_t network_top = 0;
        std::size_t teacher_top = 0;
        std::vector<std::size_t> order(network.size());
        for (std::size_t index = 0; index < network.size(); ++index) {
            order[index] = index;
            if (network[index] > network[network_top])
                network_top = index;
            if (target.probabilities[index] > target.probabilities[teacher_top])
                teacher_top = index;
            cross_entropy -= target.probabilities[index] * std::log(network[index]);
            entropy -= target.probabilities[index] * std::log(target.probabilities[index]);
            expected += network[index] * target.progress[index];
            teacher_expected += target.probabilities[index] * target.progress[index];
        }
        agreement += network_top == teacher_top ? 1.0 : 0.0;
        std::ranges::sort(order, [&](std::size_t left, std::size_t right) {
            return network[left] > network[right];
        });
        for (std::size_t index = 0; index < std::min<std::size_t>(3, order.size()); ++index)
            top3_mass += target.probabilities[order[index]];
    }
    const double count = static_cast<double>(samples.size());
    result.metrics.legal_kl = std::max(0.0, (cross_entropy - entropy) / count);
    result.metrics.expected_progress = expected / count;
    result.metrics.teacher_expected_progress = teacher_expected / count;
    result.metrics.expected_progress_ratio =
        std::abs(teacher_expected) > 1e-12 ? expected / teacher_expected : 0.0;
    result.metrics.top1_agreement = agreement / count;
    result.metrics.top3_teacher_mass = top3_mass / count;
    return result;
}

} // namespace

VacancyTarget vacancy_target(const diamond_training::TrainingSample& sample) {
    if (sample.compatibility.model_name != "Min" || sample.compatibility.player_count != 3)
        throw std::invalid_argument("vacancy distillation requires a Min sample");
    constexpr std::size_t feature_width = 6;
    if (sample.node_features.size() != soo::kBoardSize * feature_width)
        throw std::invalid_argument("vacancy distillation requires Min feature width 6");
    if (sample.sparse_policy.empty())
        throw std::invalid_argument("vacancy distillation requires legal actions");

    soo::PieceSet occupied;
    for (std::size_t position = 0; position < soo::kBoardSize; ++position) {
        const float value = sample.node_features[position * feature_width];
        if (value != 0.0F && value != 1.0F)
            throw std::invalid_argument("canonical self occupancy must be binary");
        if (value == 1.0F)
            occupied.set(position);
    }

    VacancyTarget target;
    target.actions.reserve(sample.sparse_policy.size());
    std::unordered_set<int32_t> unique;
    unique.reserve(sample.sparse_policy.size());
    for (const auto& [action, probability] : sample.sparse_policy) {
        (void)probability;
        if (action < 0 || action >= soo::kActionSize)
            throw std::invalid_argument("vacancy distillation action is out of range");
        if (!unique.insert(action).second)
            throw std::invalid_argument("vacancy distillation action is duplicated");
        target.actions.push_back(action);
    }

    soo::vacancy_prior(target.actions, occupied, target.probabilities);
    const auto& camp = soo::target_camp_set();
    const double before = soo::vacancy_potential(occupied, camp);
    target.progress.reserve(target.actions.size());
    for (const int32_t action : target.actions) {
        int source = 0;
        int destination = 0;
        soo::decode_action(action, source, destination);
        auto moved = occupied;
        moved.reset(source);
        moved.set(destination);
        target.progress.push_back(before - soo::vacancy_potential(moved, camp));
    }
    return target;
}

VacancyDistillationResult
run_vacancy_distillation(diamond_training::Trainer& trainer,
                         std::span<const diamond_training::TrainingSample> training_samples,
                         std::span<const diamond_training::TrainingSample> held_out_samples,
                         const VacancyDistillationConfig& config) {
    if (trainer.compatibility().model_name != "Min")
        throw std::invalid_argument("vacancy distillation requires Min");
    if (training_samples.empty() || held_out_samples.empty() || config.steps == 0 ||
        config.batch_size == 0 || config.evaluation_batch == 0)
        throw std::invalid_argument("vacancy distillation counts must be non-zero");

    for (const auto& parameter : trainer.model()->named_parameters()) {
        const bool policy = parameter.key().starts_with("policy_source.") ||
                            parameter.key().starts_with("policy_destination.");
        const bool value = parameter.key().starts_with("value_linear");
        const bool trainable = config.arm == DistillationArm::policy_head ? policy : !value;
        parameter.value().set_requires_grad(trainable);
    }
    std::unordered_map<std::string, torch::Tensor> trainable_before;
    for (const auto& parameter : trainer.model()->named_parameters())
        if (parameter.value().requires_grad())
            trainable_before.emplace(parameter.key(), parameter.value().detach().clone());

    auto initial = evaluate(trainer, held_out_samples, config.evaluation_batch);
    trainer.model()->train();
    double gradient_squared = 0.0;
    for (std::size_t step = 0; step < config.steps; ++step) {
        const std::size_t begin = (step * config.batch_size) % training_samples.size();
        std::vector<diamond_training::TrainingSample> batch;
        batch.reserve(config.batch_size);
        for (std::size_t index = 0; index < config.batch_size; ++index)
            batch.push_back(training_samples[(begin + index) % training_samples.size()]);
        const auto span = std::span<const diamond_training::TrainingSample>(batch);
        auto features = feature_tensor(span, trainer.device().torch_device);
        auto [targets, legal] = teacher_tensors(span, trainer.device().torch_device);
        trainer.optimizer().zero_grad();
        auto [logits, values] = trainer.model()->forward(features);
        (void)values;
        const auto loss = -(targets * legal_log_probs(logits, legal)).sum(1).mean();
        if (!torch::isfinite(loss).item<bool>())
            throw std::runtime_error("vacancy distillation produced non-finite loss");
        loss.backward();
        gradient_squared = 0.0;
        for (const auto& parameter : trainer.model()->named_parameters())
            if (parameter.value().requires_grad() && parameter.value().grad().defined())
                gradient_squared += parameter.value()
                                        .grad()
                                        .detach()
                                        .to(torch::kFloat64)
                                        .pow(2)
                                        .sum()
                                        .item<double>();
        trainer.optimizer().step();
        trainer.record_external_optimizer_step();
    }
    auto final = evaluate(trainer, held_out_samples, config.evaluation_batch);
    double policy_kl = 0.0;
    for (std::size_t row = 0; row < initial.probabilities.size(); ++row)
        for (std::size_t index = 0; index < initial.probabilities[row].size(); ++index) {
            const double before = initial.probabilities[row][index];
            policy_kl += before * std::log(before / final.probabilities[row][index]);
        }
    policy_kl /= static_cast<double>(initial.probabilities.size());
    double update_squared = 0.0;
    for (const auto& parameter : trainer.model()->named_parameters()) {
        const auto found = trainable_before.find(parameter.key());
        if (found != trainable_before.end())
            update_squared += (parameter.value().detach() - found->second)
                                  .to(torch::kFloat64)
                                  .pow(2)
                                  .sum()
                                  .item<double>();
    }
    return {.initial = initial.metrics,
            .final = final.metrics,
            .policy_kl = policy_kl,
            .trainable_update_l2 = std::sqrt(update_squared),
            .trainable_gradient_l2 = std::sqrt(gradient_squared)};
}

} // namespace diamond_pipeline
