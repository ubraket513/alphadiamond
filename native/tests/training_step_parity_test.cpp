#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#ifdef CHECK
#undef CHECK
#endif
#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_training/trainer.hpp"

namespace {

constexpr float kCpuRtol = 1e-5F;
constexpr float kCpuAtol = 1e-6F;

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    REQUIRE(static_cast<bool>(file), path.string().c_str());
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    REQUIRE(bytes >= 0 && bytes % static_cast<std::streamoff>(sizeof(float)) == 0,
            "invalid float32 fixture");
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    return values;
}

bool close(float actual, float expected) {
    return std::fabs(actual - expected) <= kCpuAtol + kCpuRtol * std::fabs(expected);
}

void check_close(float actual, float expected, const std::string& name) {
    if (!close(actual, expected)) {
        soo_test::fail(__FILE__, __LINE__, name + " differs: actual=" +
                       std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

void check_selected_values(const torch::Tensor& actual, const std::vector<float>& expected,
                           const std::string& name) {
    const auto flattened = actual.detach().contiguous().view(-1);
    REQUIRE(static_cast<size_t>(flattened.numel()) == expected.size(), name.c_str());
    const std::array<size_t, 3> selected = {0, expected.size() / 2, expected.size() - 1};
    for (const size_t index : selected) {
        check_close(flattened[static_cast<int64_t>(index)].item<float>(), expected[index],
                    name + "[" + std::to_string(index) + "]");
    }
}

diamond_training::Compatibility compatibility_for(const std::string& family) {
    const diamond_training::NetworkConfig network{.residual_blocks = 1, .width = 8};
    return family == "soo" ? diamond_training::Compatibility::soo("1.0.0", network)
                           : diamond_training::Compatibility::min("1.0.0", network);
}

diamond_training::TrainingConfig parity_config() {
    return diamond_training::TrainingConfig{.learning_rate = 1e-3, .weight_decay = 1e-4};
}

diamond_model::DiamondModel load_model(const std::filesystem::path& root,
                                       const std::string& family) {
    const int64_t input_features = family == "soo" ? 4 : 6;
    const int64_t value_size = family == "soo" ? 1 : 3;
    diamond_model::DiamondModel model(8, 1, input_features, value_size);
    model->load_weights(root / family / "initial_parameters");
    return model;
}

std::vector<diamond_training::TrainingSample> samples_for(const std::filesystem::path& root,
                                                          const std::string& family) {
    const int64_t input_features = family == "soo" ? 4 : 6;
    const int64_t value_size = family == "soo" ? 1 : 3;
    const auto inputs = read_f32(root / family / "inputs.f32");
    const auto policies = read_f32(root / family / "policy_targets.f32");
    const auto values = read_f32(root / family / "value_targets.f32");
    REQUIRE(inputs.size() == static_cast<size_t>(2 * 73 * input_features), "input shape");
    REQUIRE(policies.size() == static_cast<size_t>(2 * 5329), "policy shape");
    REQUIRE(values.size() == static_cast<size_t>(2 * value_size), "value shape");

    std::vector<diamond_training::TrainingSample> samples;
    for (int64_t batch = 0; batch < 2; ++batch) {
        diamond_training::TrainingSample sample;
        sample.compatibility = compatibility_for(family);
        const auto feature_offset = static_cast<size_t>(batch * 73 * input_features);
        sample.node_features.assign(inputs.begin() + static_cast<std::ptrdiff_t>(feature_offset),
                                    inputs.begin() + static_cast<std::ptrdiff_t>(feature_offset + 73 * input_features));
        for (int32_t action = 0; action < 5329; ++action) {
            const float probability = policies[static_cast<size_t>(batch * 5329 + action)];
            if (probability != 0.0F) sample.sparse_policy.emplace_back(action, probability);
        }
        const auto value_offset = static_cast<size_t>(batch * value_size);
        sample.value_target.assign(values.begin() + static_cast<std::ptrdiff_t>(value_offset),
                                   values.begin() + static_cast<std::ptrdiff_t>(value_offset + value_size));
        samples.push_back(std::move(sample));
    }
    return samples;
}

template <class F>
void check_invalid_argument(F&& operation, const std::string& name) {
    try {
        operation();
        soo_test::fail(__FILE__, __LINE__, name + " did not reject invalid input");
    } catch (const std::invalid_argument&) {
    }
}

void check_validation(const std::filesystem::path& root) {
    auto model = load_model(root, "soo");
    diamond_training::Trainer trainer(model, compatibility_for("soo"), parity_config());
    const std::span<const diamond_training::TrainingSample> empty;
    check_invalid_argument([&] { (void)trainer.train(empty); }, "empty batch");

    auto sample = samples_for(root, "soo").front();
    sample.compatibility = compatibility_for("min");
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "compatibility mismatch");

    sample = samples_for(root, "soo").front();
    sample.node_features.front() = std::numeric_limits<float>::quiet_NaN();
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "non-finite feature");

    sample = samples_for(root, "soo").front();
    sample.value_target.front() = std::numeric_limits<float>::infinity();
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "non-finite value target");

    sample = samples_for(root, "soo").front();
    sample.sparse_policy.front().second = -0.25F;
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "negative policy probability");

    sample = samples_for(root, "soo").front();
    sample.sparse_policy.front().second = std::numeric_limits<float>::quiet_NaN();
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "non-finite policy target");

    sample = samples_for(root, "soo").front();
    sample.sparse_policy.front().second = 0.20F;
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "policy sum outside tolerance");

    sample = samples_for(root, "soo").front();
    sample.sparse_policy.front().first = 5329;
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "action outside policy space");

    sample = samples_for(root, "soo").front();
    sample.sparse_policy.emplace_back(sample.sparse_policy.front().first, 0.0F);
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "duplicate policy action");

    sample = samples_for(root, "soo").front();
    sample.value_target.push_back(0.0F);
    check_invalid_argument([&] { (void)trainer.train(std::span(&sample, 1)); },
                           "Soo value width");

    auto min_model = load_model(root, "min");
    diamond_training::Trainer min_trainer(min_model, compatibility_for("min"), parity_config());
    auto min_sample = samples_for(root, "min").front();
    min_sample.value_target.pop_back();
    check_invalid_argument([&] { (void)min_trainer.train(std::span(&min_sample, 1)); },
                           "Min value width");
}

void check_parity(const std::filesystem::path& root, const std::string& family) {
    auto model = load_model(root, family);
    auto samples = samples_for(root, family);
    const auto expected_logits = read_f32(root / family / "initial_logits.f32");
    const auto expected_values = read_f32(root / family / "initial_values.f32");
    const auto expected_losses = read_f32(root / family / "losses.f32");
    const auto expected_after = read_f32(
        root / family / "after_step_parameters/policy_head__source__weight.f32");
    const auto expected_gradient = read_f32(
        root / family / "gradients/policy_head__source__weight.f32");
    const auto expected_resumed_losses = read_f32(root / family / "resumed_losses.f32");
    const auto expected_resumed = read_f32(
        root / family / "resumed_parameters/policy_head__source__weight.f32");

    model->eval();
    {
        torch::NoGradGuard no_grad;
        const auto features = torch::from_blob(samples.front().node_features.data(),
                                               {1, 73, model->input_features()}, torch::kFloat32)
                                  .clone();
        const auto [logits, values] = model->forward(features);
        check_close(logits.flatten()[0].item<float>(), expected_logits.front(),
                    family + " initial logit");
        check_close(values.flatten()[0].item<float>(), expected_values.front(),
                    family + " initial value");
    }

    diamond_training::Trainer trainer(model, compatibility_for(family), parity_config());
    const auto metrics = trainer.train(samples);
    REQUIRE(expected_losses.size() == 3, "loss fixture");
    check_close(static_cast<float>(metrics.total_loss), expected_losses[0], family + " total loss");
    check_close(static_cast<float>(metrics.policy_loss), expected_losses[1], family + " policy loss");
    check_close(static_cast<float>(metrics.value_loss), expected_losses[2], family + " value loss");
    CHECK_EQ(metrics.training_step, uint64_t{1});
    CHECK(std::isfinite(metrics.total_loss));
    CHECK(std::isfinite(metrics.policy_loss));
    CHECK(std::isfinite(metrics.value_loss));

    const auto source = model->policy_source->weight;
    CHECK(source.grad().defined());
    CHECK(source.grad().abs().sum().item<float>() > 0.0F);
    check_selected_values(source.grad(), expected_gradient, family + " policy-source gradient");
    check_selected_values(source, expected_after, family + " AdamW update");

    const auto resumed = trainer.train(samples);
    REQUIRE(expected_resumed_losses.size() == 3, "resumed loss fixture");
    check_close(static_cast<float>(resumed.total_loss), expected_resumed_losses[0],
                family + " resumed total loss");
    CHECK_EQ(resumed.training_step, uint64_t{2});
    check_selected_values(source, expected_resumed, family + " AdamW stateful update");
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: training_step_parity_test <fixture-dir>");
    const auto root = std::filesystem::path(argv[1]);
    check_validation(root);
    check_parity(root, "soo");
    check_parity(root, "min");
    return soo_test::report("training_step_parity_test");
}
