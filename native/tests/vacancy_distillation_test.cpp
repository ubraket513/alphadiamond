#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "diamond_pipeline/vacancy_distillation.hpp"
#include "diamond_training/training_sample.hpp"
#include "soo/action.hpp"
#include "soo/board.hpp"

namespace {

diamond_training::TrainingSample sample() {
    diamond_training::TrainingSample value;
    value.compatibility =
        diamond_training::Compatibility::min("2.0.0", {.residual_blocks = 1, .width = 8});
    value.node_features.assign(soo::kBoardSize * 6, 0.0F);
    for (int position = 0; position < 10; ++position)
        value.node_features[static_cast<std::size_t>(position) * 6] = 1.0F;
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

    return soo_test::report("vacancy_distillation_test");
}
