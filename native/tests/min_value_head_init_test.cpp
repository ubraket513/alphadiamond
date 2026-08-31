// Min's scratch network starts from a neutral value, and can still learn one.
//
// A fresh Min value head answers every leaf with an arbitrary three-vector.
// During the bootstrap phase that noise is the only thing competing with the
// vacancy heuristic's sense of direction, so the scratch model zeroes the final
// value layer. Two claims have to hold for that to be a sound change rather
// than a plausible one:
//
//   1. It is *only* that layer. The zeroing happens after construction, so both
//      arms of an A/B consume the same RNG draws and every other tensor is
//      bit-identical. If that failed, a measured difference between them would
//      not be attributable to the value head.
//
//   2. The zero does not trap the head. value_linear2 = 0 sends no gradient to
//      value_linear1 on the first backward, so the layers below start moving
//      only on the second step. Min production runs one train step per
//      iteration, so this is checked across an iteration boundary too -- a
//      network that only learned on the second step of a run that never takes
//      one would silently never learn a value at all.
#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#ifdef CHECK
#undef CHECK
#endif
#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_training/trainer.hpp"

namespace {

constexpr int64_t kBoardNodes = 73;
constexpr int64_t kMinFeatures = 6;
constexpr int64_t kValueSize = 3;
constexpr int64_t kWidth = 16;
constexpr int64_t kBlocks = 2;

diamond_training::Compatibility min_compatibility() {
    return diamond_training::Compatibility::min("2.0.0",
                                                {.residual_blocks = kBlocks, .width = kWidth});
}

// Both arms are built from the same seed, so the draws are identical.
diamond_model::DiamondModel min_model(uint64_t seed) {
    torch::manual_seed(static_cast<int64_t>(seed));
    auto model = diamond_model::DiamondModel(kWidth, kBlocks, kMinFeatures, kValueSize);
    auto adjacency = torch::zeros({6, kBoardNodes, kBoardNodes});
    // A ring, so every hole has a neighbour in one direction and the trunk has
    // something to gather. The topology is not what this test is about.
    for (int64_t node = 0; node < kBoardNodes; ++node)
        adjacency[0][node][(node + 1) % kBoardNodes] = 1.0F;
    model->set_adjacency(adjacency);
    return model;
}

std::vector<diamond_training::TrainingSample> samples(std::size_t count) {
    std::vector<diamond_training::TrainingSample> out;
    for (std::size_t index = 0; index < count; ++index) {
        diamond_training::TrainingSample sample;
        sample.compatibility = min_compatibility();
        sample.node_features.assign(static_cast<std::size_t>(kBoardNodes * kMinFeatures), 0.0F);
        // A distinct occupancy per sample, so the batch is not one row repeated.
        sample.node_features[(index % static_cast<std::size_t>(kBoardNodes)) *
                             static_cast<std::size_t>(kMinFeatures)] = 1.0F;
        sample.canonical_player_ids = {1, 2, 3};
        sample.sparse_policy = {{static_cast<int32_t>(index % 64), 1.0F}};
        // A target the head cannot reach by staying at zero.
        sample.value_target = {1.0F, 0.0F, -1.0F};
        out.push_back(std::move(sample));
    }
    return out;
}

double max_abs(const torch::Tensor& tensor) {
    return tensor.detach().abs().max().template item<float>();
}

double max_abs_difference(const torch::Tensor& left, const torch::Tensor& right) {
    return (left.detach() - right.detach()).abs().max().template item<float>();
}

bool tensors_identical(const torch::Tensor& left, const torch::Tensor& right) {
    if (left.sizes() != right.sizes())
        return false;
    return torch::equal(left.detach().cpu(), right.detach().cpu());
}

} // namespace

int main() {
    const auto device = diamond_training::resolve_device("cpu");

    // (1) One layer apart, everything else bit-identical.
    auto random_arm = min_model(4242);
    auto zero_arm = min_model(4242);
    diamond_training::zero_value_head(zero_arm);

    const auto random_named = random_arm->named_parameters();
    const auto zero_named = zero_arm->named_parameters();
    CHECK_EQ(random_named.size(), zero_named.size());
    std::size_t compared = 0;
    for (const auto& entry : random_named) {
        const auto* other = zero_named.find(entry.key());
        REQUIRE(other != nullptr, ("missing parameter: " + entry.key()).c_str());
        const bool is_value_head = entry.key().rfind("value_linear2", 0) == 0;
        if (is_value_head) {
            CHECK(max_abs(*other) == 0.0);
            // The arms must actually differ there, or the A/B is comparing a
            // model with itself and would "pass" for the wrong reason.
            CHECK(max_abs(entry.value()) > 0.0);
        } else if (!tensors_identical(entry.value(), *other)) {
            soo_test::fail(__FILE__, __LINE__,
                           "same-seed arms differ outside the value head: " + entry.key());
        }
        ++compared;
    }
    CHECK(compared > 0);

    // (2) The zeroed head answers exactly [0,0,0], on arbitrary states.
    {
        torch::NoGradGuard no_grad;
        zero_arm->eval();
        auto features = torch::rand({5, kBoardNodes, kMinFeatures});
        const auto [policy, value] = zero_arm->forward(features);
        (void)policy;
        CHECK_EQ(value.size(0), int64_t{5});
        CHECK_EQ(value.size(1), kValueSize);
        // Exactly zero, not merely small: tanh(0) is 0 and the bias is zeroed,
        // so any nonzero here means the head was not neutral after all.
        CHECK(max_abs(value) == 0.0);
        zero_arm->train();
    }

    // (3) Gradient flow. The final layer learns on the first step; the layer
    // below it cannot, because a zero weight passes no gradient back.
    diamond_training::Trainer trainer(zero_arm, min_compatibility(), {1e-2, 0.0}, device);
    auto& learner = trainer.learner();
    const auto linear1_before = learner->value_linear1->weight.detach().clone();
    const auto batch = samples(8);

    trainer.train(batch);
    CHECK(max_abs(learner->value_linear2->weight) > 0.0);
    const double linear1_after_first =
        max_abs_difference(learner->value_linear1->weight, linear1_before);
    if (linear1_after_first != 0.0) {
        // Not a failure of the model -- a failure of this test's premise, which
        // the rest of the reasoning depends on. Say so rather than passing.
        soo_test::fail(__FILE__, __LINE__,
                       "value_linear1 moved on the first step, so the zero head did not block "
                       "the gradient as assumed");
    }

    trainer.train(batch);
    CHECK(max_abs_difference(learner->value_linear1->weight, linear1_before) > 0.0);

    // (4) Across an iteration boundary. Min production takes one train step per
    // iteration, so the second step is a *later iteration*, reached through the
    // checkpoint restore that carries the training step forward. The optimizer
    // and the weights persist; only the sample batch is new. If learning below
    // the head depended on two steps within one call, this is where it breaks.
    auto fresh = min_model(99);
    diamond_training::zero_value_head(fresh);
    diamond_training::Trainer per_iteration(fresh, min_compatibility(), {1e-2, 0.0}, device);
    auto& boundary_learner = per_iteration.learner();
    const auto boundary_before = boundary_learner->value_linear1->weight.detach().clone();

    per_iteration.train(samples(4)); // iteration 0: one step
    per_iteration.restore_checkpoint_state({1e-2, 0.0}, per_iteration.training_step());
    per_iteration.train(samples(4)); // iteration 1: one step
    CHECK(max_abs_difference(boundary_learner->value_linear1->weight, boundary_before) > 0.0);
    CHECK_EQ(per_iteration.training_step(), uint64_t{2});

    return soo_test::report("min_value_head_init_test");
}
