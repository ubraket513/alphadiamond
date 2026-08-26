#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
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

template <typename Operation> bool rejects(Operation&& operation) {
    try {
        operation();
        return false;
    } catch (const diamond_training::CheckpointError&) {
        return true;
    }
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

    const diamond_training::CheckpointProvenance provenance{
        .source_git_commit = "0123456789abcdef0123456789abcdef01234567",
        .resolved_config_bytes = "{}",
        .replay_manifest_sha256 =
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        .protocol_ids_json = "{}",
        .creation_timestamp = "2026-08-26T00:00:00Z",
        .environment_json = "{}",
    };

    // v3 makes a warm start explicit: source weights transfer, but neither
    // source optimizer state nor source training step can become a resume.
    diamond_training::Trainer warm_seed(diamond_model::DiamondModel(8, 1, 4, 1),
                                        checkpoint_compatibility, kConfig, device);
    (void)diamond_training::load_checkpoint_v2_weights(scratch, warm_seed.model(), device);
    const diamond_training::CheckpointLineage warm_lineage{
        .initialization_mode = diamond_training::CheckpointInitializationMode::warm_start,
        .run_id = "warm-start-run",
        .iteration = 0,
        .model_step = 0,
        .source_digest = saved_digest,
        .source_training_step = saved.training_step(),
        .optimizer_restored = false,
        .optimizer_reset = true,
        .optimizer_reset_reason = "warm_start",
    };
    const auto warm_root = scratch / "warm-start";
    const auto warm_info =
        diamond_training::save_checkpoint_v3(warm_root, warm_seed, warm_lineage, provenance);
    CHECK_EQ(warm_info.format_version, uint64_t{3});
    CHECK(warm_info.lineage.has_value());
    CHECK_EQ(warm_info.lineage->run_id, std::string("warm-start-run"));
    CHECK_EQ(warm_info.lineage->model_step, uint64_t{0});
    CHECK(warm_info.lineage->optimizer_reset);
    CHECK(!warm_info.lineage->optimizer_restored);
    const auto inspected_warm = diamond_training::inspect_checkpoint_v2(warm_root);
    CHECK_EQ(inspected_warm.format_version, uint64_t{3});
    CHECK(inspected_warm.lineage.has_value());
    CHECK(inspected_warm.provenance.has_value());
    CHECK_EQ(inspected_warm.provenance->source_git_commit, provenance.source_git_commit);
    CHECK_EQ(inspected_warm.provenance->resolved_config_bytes, provenance.resolved_config_bytes);
    // Exact resume restores the native step-zero snapshot written after the
    // warm start; it never reads the external source optimizer or source step.
    diamond_training::Trainer warm_exact(diamond_model::DiamondModel(8, 1, 4, 1),
                                         checkpoint_compatibility, kConfig, device);
    (void)diamond_training::load_checkpoint_v3(
        warm_root, warm_exact, device, diamond_training::CheckpointLoadIntent::exact_resume);
    CHECK_EQ(warm_exact.training_step(), uint64_t{0});
    CHECK(warm_exact.optimizer().state().empty());
    CHECK_EQ(diamond_training::canonical_model_digest(warm_exact.model()), saved_digest);

    diamond_training::Trainer warm_destination(diamond_model::DiamondModel(8, 1, 4, 1),
                                               checkpoint_compatibility, kConfig, device);
    CHECK_EQ(warm_destination.train(samples).training_step, uint64_t{1});
    (void)diamond_training::load_checkpoint_v3(warm_root, warm_destination, device,
                                               diamond_training::CheckpointLoadIntent::warm_start);
    CHECK_EQ(warm_destination.training_step(), uint64_t{0});
    CHECK(warm_destination.optimizer().state().empty());
    CHECK_EQ(diamond_training::canonical_model_digest(warm_destination.model()), saved_digest);

    CHECK_EQ(warm_destination.train(samples).training_step, uint64_t{1});
    const diamond_training::CheckpointLineage resume_lineage{
        .initialization_mode = diamond_training::CheckpointInitializationMode::resume,
        .run_id = "warm-start-run",
        .iteration = 1,
        .model_step = warm_destination.training_step(),
        .parent_digest = warm_info.model_digest,
        .source_digest = warm_info.model_digest,
        .source_training_step = warm_info.training_step,
        .optimizer_restored = true,
        .optimizer_reset = false,
    };
    const auto resume_root = scratch / "resume";
    const auto resume_info = diamond_training::save_checkpoint_v3(resume_root, warm_destination,
                                                                  resume_lineage, provenance);
    diamond_training::Trainer exact_destination(diamond_model::DiamondModel(8, 1, 4, 1),
                                                checkpoint_compatibility, kConfig, device);
    const auto destination_before_tamper =
        diamond_training::canonical_model_digest(exact_destination.model());
    const auto provenance_manifest = resume_info.generation / "manifest.json";
    const std::string original_manifest = [&] {
        std::ifstream input(provenance_manifest, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }();
    {
        std::string tampered = original_manifest;
        const auto position = tampered.find(provenance.source_git_commit);
        REQUIRE(position != std::string::npos, "v3 provenance git commit");
        tampered.replace(position, provenance.source_git_commit.size(), "tampered");
        std::ofstream output(provenance_manifest, std::ios::binary | std::ios::trunc);
        output << tampered;
        REQUIRE(static_cast<bool>(output), "cannot tamper v3 provenance");
    }
    CHECK(rejects([&] {
        (void)diamond_training::load_checkpoint_v3(
            resume_root, exact_destination, device,
            diamond_training::CheckpointLoadIntent::exact_resume);
    }));
    CHECK_EQ(diamond_training::canonical_model_digest(exact_destination.model()),
             destination_before_tamper);
    {
        std::ofstream output(provenance_manifest, std::ios::binary | std::ios::trunc);
        output << original_manifest;
        REQUIRE(static_cast<bool>(output), "cannot restore v3 provenance");
    }
    (void)diamond_training::load_checkpoint_v3(
        resume_root, exact_destination, device,
        diamond_training::CheckpointLoadIntent::exact_resume);
    CHECK_EQ(exact_destination.training_step(), resume_info.training_step);
    check_adamw_equal(exact_destination.optimizer(), warm_destination.optimizer());
    {
        const auto manifest_path = resume_info.generation / "manifest.json";
        std::ifstream input(manifest_path, std::ios::binary);
        const std::string manifest{std::istreambuf_iterator<char>(input),
                                   std::istreambuf_iterator<char>()};
        const auto position = manifest.find(resume_info.model_digest);
        REQUIRE(position != std::string::npos, "v3 manifest model digest");
        std::string corrupt = manifest;
        corrupt.replace(position, resume_info.model_digest.size(), 64, '0');
        std::ofstream output(manifest_path, std::ios::binary | std::ios::trunc);
        output << corrupt;
        REQUIRE(static_cast<bool>(output), "cannot corrupt v3 manifest");
    }
    CHECK(rejects([&] { (void)diamond_training::inspect_checkpoint_v2(resume_root); }));

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
