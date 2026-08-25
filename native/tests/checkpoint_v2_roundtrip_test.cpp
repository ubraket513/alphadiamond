#include <cmath>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <torch/torch.h>
#ifdef DIAMOND_TEST_WITH_CUDA
#include <torch/cuda.h>
#endif

#include "check.hpp"
#include "diamond_training/checkpoint.hpp"

namespace {

constexpr diamond_training::TrainingConfig kConfig{
    .learning_rate = 1e-3,
    .weight_decay = 1e-4,
};

diamond_training::Compatibility compatibility() {
    return diamond_training::Compatibility::soo(
        "1.0.0", {.residual_blocks = 1, .width = 8});
}

diamond_training::TrainingSample sample_for(
    const diamond_training::Compatibility& sample_compatibility) {
    diamond_training::TrainingSample sample;
    sample.compatibility = sample_compatibility;
    sample.node_features.assign(73U * 4U, 0.0F);
    sample.node_features[7] = 0.25F;
    sample.sparse_policy.emplace_back(0, 1.0F);
    sample.value_target = {0.125F};
    return sample;
}

void check_tensor_equal(const torch::Tensor& actual, const torch::Tensor& expected,
                        const std::string& name) {
    if (actual.defined() != expected.defined()) {
        soo_test::fail(__FILE__, __LINE__, name + " definedness differs");
        return;
    }
    if (actual.defined() && !torch::equal(actual.detach().to(torch::kCPU),
                                           expected.detach().to(torch::kCPU))) {
        soo_test::fail(__FILE__, __LINE__, name + " differs");
    }
}

void check_adamw_equal(const torch::optim::AdamW& actual,
                       const torch::optim::AdamW& expected) {
    const auto actual_parameters = actual.param_groups().front().params();
    const auto expected_parameters = expected.param_groups().front().params();
    REQUIRE(actual_parameters.size() == expected_parameters.size(), "AdamW parameter count");
    CHECK_EQ(actual.state().size(), expected.state().size());
    CHECK_EQ(actual.state().size(), actual_parameters.size());

    for (size_t index = 0; index < actual_parameters.size(); ++index) {
        const auto actual_entry = actual.state().find(actual_parameters[index].unsafeGetTensorImpl());
        const auto expected_entry = expected.state().find(expected_parameters[index].unsafeGetTensorImpl());
        REQUIRE(actual_entry != actual.state().end(), "missing restored AdamW state");
        REQUIRE(expected_entry != expected.state().end(), "missing saved AdamW state");
        const auto* actual_state =
            dynamic_cast<const torch::optim::AdamWParamState*>(actual_entry->second.get());
        const auto* expected_state =
            dynamic_cast<const torch::optim::AdamWParamState*>(expected_entry->second.get());
        REQUIRE(actual_state != nullptr && expected_state != nullptr, "AdamW state type");
        CHECK_EQ(actual_state->step(), expected_state->step());
        check_tensor_equal(actual_state->exp_avg(), expected_state->exp_avg(),
                           "AdamW exp_avg[" + std::to_string(index) + "]");
        check_tensor_equal(actual_state->exp_avg_sq(), expected_state->exp_avg_sq(),
                           "AdamW exp_avg_sq[" + std::to_string(index) + "]");
        check_tensor_equal(actual_state->max_exp_avg_sq(), expected_state->max_exp_avg_sq(),
                           "AdamW max_exp_avg_sq[" + std::to_string(index) + "]");
    }
}

void check_tensor_device(const torch::Tensor& tensor, const torch::Device& device,
                         const std::string& name) {
    REQUIRE(tensor.defined(), name.c_str());
    CHECK(tensor.device() == device);
    CHECK(torch::isfinite(tensor).all().item<bool>());
}

void check_trainer_device_state(diamond_training::Trainer& trainer,
                                const diamond_training::ResolvedDevice& device) {
    const auto parameters = trainer.model()->parameters();
    for (const auto& parameter : trainer.model()->named_parameters()) {
        check_tensor_device(parameter.value(), device.torch_device,
                            "model parameter " + parameter.key());
    }
    for (const auto& buffer : trainer.model()->named_buffers()) {
        check_tensor_device(buffer.value(), device.torch_device,
                            "model buffer " + buffer.key());
    }

    const auto& groups = trainer.optimizer().param_groups();
    REQUIRE(groups.size() == 1, "AdamW parameter group count");
    REQUIRE(groups.front().params().size() == parameters.size(), "AdamW parameter count");
    CHECK_EQ(trainer.optimizer().state().size(), parameters.size());
    for (const auto& parameter : parameters) {
        const auto entry = trainer.optimizer().state().find(parameter.unsafeGetTensorImpl());
        REQUIRE(entry != trainer.optimizer().state().end(), "missing AdamW state");
        const auto* state =
            dynamic_cast<const torch::optim::AdamWParamState*>(entry->second.get());
        REQUIRE(state != nullptr, "AdamW state type");
        CHECK_EQ(state->step(), static_cast<int64_t>(trainer.training_step()));
        check_tensor_device(state->exp_avg(), device.torch_device, "AdamW exp_avg");
        check_tensor_device(state->exp_avg_sq(), device.torch_device, "AdamW exp_avg_sq");
        if (state->max_exp_avg_sq().defined()) {
            check_tensor_device(state->max_exp_avg_sq(), device.torch_device,
                                "AdamW max_exp_avg_sq");
        }
    }
}

#ifdef DIAMOND_TEST_WITH_CUDA
void check_cuda_roundtrips(const std::filesystem::path& scratch,
                           const diamond_training::Compatibility& checkpoint_compatibility,
                           const std::vector<diamond_training::TrainingSample>& samples) {
    REQUIRE(torch::cuda::is_available(),
            "CUDA checkpoint coverage requires an available CUDA device");
    const auto cpu = diamond_training::resolve_device("cpu");
    const auto cuda = diamond_training::resolve_device("cuda");

    auto check_restored = [&](const std::filesystem::path& root,
                              const diamond_training::ResolvedDevice& target,
                              const std::string& expected_digest, uint64_t expected_step) {
        diamond_training::Trainer destination(
            diamond_model::DiamondModel(8, 1, 4, 1), checkpoint_compatibility, kConfig, target);
        const auto loaded = diamond_training::load_checkpoint_v2(root, destination, target);
        CHECK_EQ(loaded.training_step, expected_step);
        CHECK_EQ(destination.training_step(), expected_step);
        CHECK_EQ(diamond_training::canonical_model_digest(destination.model()), expected_digest);
        check_trainer_device_state(destination, target);
        CHECK_EQ(destination.train(samples).training_step, expected_step + 1);
        check_trainer_device_state(destination, target);
    };

    diamond_training::Trainer cpu_source(
        diamond_model::DiamondModel(8, 1, 4, 1), checkpoint_compatibility, kConfig, cpu);
    CHECK_EQ(cpu_source.train(samples).training_step, uint64_t{1});
    const auto cpu_digest = diamond_training::canonical_model_digest(cpu_source.model());
    const auto cpu_checkpoint = scratch / "cpu-to-cuda";
    (void)diamond_training::save_checkpoint_v2(cpu_checkpoint, cpu_source);
    check_restored(cpu_checkpoint, cuda, cpu_digest, cpu_source.training_step());

    diamond_training::Trainer cuda_source(
        diamond_model::DiamondModel(8, 1, 4, 1), checkpoint_compatibility, kConfig, cuda);
    CHECK_EQ(cuda_source.train(samples).training_step, uint64_t{1});
    const auto cuda_digest = diamond_training::canonical_model_digest(cuda_source.model());
    const auto cuda_checkpoint = scratch / "cuda-source";
    (void)diamond_training::save_checkpoint_v2(cuda_checkpoint, cuda_source);
    check_restored(cuda_checkpoint, cuda, cuda_digest, cuda_source.training_step());
    check_restored(cuda_checkpoint, cpu, cuda_digest, cuda_source.training_step());
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const bool run_cuda = argc == 3 && std::string_view(argv[2]) == "--cuda";
    REQUIRE(argc == 2 || run_cuda,
            "usage: checkpoint_v2_roundtrip_test <scratch> [--cuda]");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);

    const auto device = diamond_training::resolve_device("cpu");
    const auto checkpoint_compatibility = compatibility();
    const std::vector<diamond_training::TrainingSample> samples{
        sample_for(checkpoint_compatibility)};
    if (run_cuda) {
#ifdef DIAMOND_TEST_WITH_CUDA
        check_cuda_roundtrips(scratch, checkpoint_compatibility, samples);
        std::filesystem::remove_all(scratch);
        return soo_test::report("checkpoint_v2_roundtrip_test_cuda");
#else
        REQUIRE(false, "CUDA checkpoint coverage was not compiled for this build");
#endif
    }

    diamond_training::Trainer saved(
        diamond_model::DiamondModel(8, 1, 4, 1), checkpoint_compatibility, kConfig, device);
    CHECK_EQ(saved.train(samples).training_step, uint64_t{1});
    REQUIRE(!saved.optimizer().state().empty(), "training must create AdamW state");
    const std::string saved_digest = diamond_training::canonical_model_digest(saved.model());

    const auto written = diamond_training::save_checkpoint_v2(scratch, saved);
    CHECK(std::filesystem::exists(written.generation / "state.pt"));
    CHECK(std::filesystem::exists(written.generation / "optimizer.pt"));
    CHECK_EQ(written.training_step, uint64_t{1});

    diamond_training::Trainer restored(
        diamond_model::DiamondModel(8, 1, 4, 1), checkpoint_compatibility, kConfig, device);
    const auto loaded = diamond_training::load_checkpoint_v2(scratch, restored, device);
    CHECK_EQ(loaded.training_step, saved.training_step());
    CHECK_EQ(restored.training_step(), saved.training_step());
    CHECK_EQ(diamond_training::canonical_model_digest(restored.model()), saved_digest);
    check_adamw_equal(restored.optimizer(), saved.optimizer());
    check_trainer_device_state(restored, device);

    // Identical model and optimizer state must yield an identical next update.
    const auto saved_next = saved.train(samples);
    const auto restored_next = restored.train(samples);
    CHECK_EQ(saved_next.training_step, restored_next.training_step);
    CHECK(std::fabs(saved_next.total_loss - restored_next.total_loss) < 1e-12);
    CHECK(std::fabs(saved_next.policy_loss - restored_next.policy_loss) < 1e-12);
    CHECK(std::fabs(saved_next.value_loss - restored_next.value_loss) < 1e-12);
    CHECK_EQ(diamond_training::canonical_model_digest(restored.model()),
             diamond_training::canonical_model_digest(saved.model()));
    check_adamw_equal(restored.optimizer(), saved.optimizer());
    check_trainer_device_state(restored, device);

    std::filesystem::remove_all(scratch);
    return soo_test::report("checkpoint_v2_roundtrip_test");
}
