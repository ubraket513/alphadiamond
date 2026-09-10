#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <vector>

#include "diamond_pipeline/vacancy_distillation.hpp"
#include "diamond_training/training_sample.hpp"
#include "diamond_training/device.hpp"
#include "diamond_training/trainer.hpp"
#include "diamond_model/soo_model.hpp"
#include "soo/action.hpp"
#include "soo/board.hpp"

#ifdef CHECK
#undef CHECK
#endif
#include "check.hpp"

namespace {

diamond_training::TrainingSample sample() {
    diamond_training::TrainingSample value;
    value.compatibility =
        diamond_training::Compatibility::min("2.0.0", {.residual_blocks = 1, .width = 8});
    value.node_features.assign(soo::kBoardSize * 6, 0.0F);
    for (int position = 0; position < 10; ++position)
        value.node_features[static_cast<std::size_t>(position) * 6] = 1.0F;
    value.node_features[70 * 6 + 3] = 1.0F;
    value.node_features[69 * 6 + 4] = 1.0F;
    value.node_features[68 * 6 + 5] = 1.0F;
    value.canonical_player_ids = {0, 1, 2};
    value.sparse_policy = {{soo::encode_action(0, 70), 1.0F},
                           {soo::encode_action(1, 69), 0.0F},
                           {soo::encode_action(2, 68), 0.0F}};
    value.value_target = {1.0F, 0.0F, -1.0F};
    return value;
}

bool throws_invalid(const std::function<void()>& action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

diamond_training::Trainer trainer() {
    torch::manual_seed(17);
    const auto compatibility =
        diamond_training::Compatibility::min("2.0.0", {.residual_blocks = 1, .width = 8});
    auto model = diamond_model::DiamondModel(8, 1, 6, 3);
    model->set_adjacency(torch::zeros({6, soo::kBoardSize, soo::kBoardSize}));
    return diamond_training::Trainer(model, compatibility,
                                     {.learning_rate = 1e-2, .weight_decay = 0.0},
                                     diamond_training::resolve_device("cpu"));
}

std::unordered_map<std::string, torch::Tensor> snapshot(diamond_training::Trainer& value) {
    std::unordered_map<std::string, torch::Tensor> result;
    for (const auto& parameter : value.model()->named_parameters())
        result.emplace(parameter.key(), parameter.value().detach().clone());
    return result;
}

} // namespace

int main() {
    soo::ensure_topology_configured();
    const auto input = sample();
    const auto target = diamond_pipeline::vacancy_target(input);
    CHECK_EQ(target.actions.size(), std::size_t{3});
    CHECK_EQ(target.probabilities.size(), std::size_t{3});
    CHECK_EQ(target.progress.size(), std::size_t{3});
    for (std::size_t index = 0; index < target.actions.size(); ++index)
        CHECK_EQ(target.actions[index], input.sparse_policy[index].first);

    double sum = 0.0;
    for (const double probability : target.probabilities) {
        CHECK(std::isfinite(probability));
        CHECK(probability > 0.0);
        sum += probability;
    }
    CHECK(std::abs(sum - 1.0) < 1e-12);

    auto empty = sample();
    empty.sparse_policy.clear();
    CHECK(throws_invalid([&] { (void)diamond_pipeline::vacancy_target(empty); }));

    auto short_features = sample();
    short_features.node_features.pop_back();
    CHECK(throws_invalid([&] { (void)diamond_pipeline::vacancy_target(short_features); }));

    auto non_binary = sample();
    non_binary.node_features[0] = 0.5F;
    CHECK(throws_invalid([&] { (void)diamond_pipeline::vacancy_target(non_binary); }));

    auto duplicate = sample();
    duplicate.sparse_policy.push_back(duplicate.sparse_policy.front());
    CHECK(throws_invalid([&] { (void)diamond_pipeline::vacancy_target(duplicate); }));

    auto out_of_range = sample();
    out_of_range.sparse_policy.front().first = soo::kActionSize;
    CHECK(throws_invalid([&] { (void)diamond_pipeline::vacancy_target(out_of_range); }));

    std::vector<diamond_training::TrainingSample> samples(8, sample());
    {
        auto learner = trainer();
        const auto before = snapshot(learner);
        const auto result = diamond_pipeline::run_vacancy_distillation(
            learner, samples, samples,
            {.arm = diamond_pipeline::DistillationArm::policy_head,
             .steps = 10,
             .batch_size = 4,
             .evaluation_batch = 4});
        CHECK_EQ(learner.training_step(), uint64_t{10});
        CHECK(result.trainable_update_l2 > 0.0);
        CHECK(result.trainable_gradient_l2 > 0.0);
        CHECK(result.final.legal_kl < result.initial.legal_kl);
        for (const auto& parameter : learner.model()->named_parameters()) {
            const bool policy = parameter.key().starts_with("policy_source.") ||
                                parameter.key().starts_with("policy_destination.");
            CHECK(torch::equal(before.at(parameter.key()), parameter.value()) == !policy);
        }
    }
    {
        auto learner = trainer();
        const auto before = snapshot(learner);
        (void)diamond_pipeline::run_vacancy_distillation(
            learner, samples, samples,
            {.arm = diamond_pipeline::DistillationArm::trunk_policy,
             .steps = 2,
             .batch_size = 4,
             .evaluation_batch = 4});
        bool trunk_changed = false;
        bool policy_changed = false;
        for (const auto& parameter : learner.model()->named_parameters()) {
            const bool value = parameter.key().starts_with("value_linear");
            const bool changed = !torch::equal(before.at(parameter.key()), parameter.value());
            if (parameter.key().starts_with("blocks.") ||
                parameter.key().starts_with("input_projection."))
                trunk_changed = trunk_changed || changed;
            if (parameter.key().starts_with("policy_"))
                policy_changed = policy_changed || changed;
            if (value)
                CHECK(!changed);
        }
        CHECK(trunk_changed);
        CHECK(policy_changed);
    }

    return soo_test::report("vacancy_distillation_test");
}
