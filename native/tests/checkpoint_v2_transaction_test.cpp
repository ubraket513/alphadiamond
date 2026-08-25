#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <torch/torch.h>

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

std::vector<diamond_training::TrainingSample> samples_for(
    const diamond_training::Compatibility& sample_compatibility) {
    diamond_training::TrainingSample sample;
    sample.compatibility = sample_compatibility;
    sample.node_features.assign(73U * 4U, 0.0F);
    sample.node_features[11] = -0.5F;
    sample.sparse_policy.emplace_back(0, 1.0F);
    sample.value_target = {0.25F};
    return {sample};
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
    for (size_t index = 0; index < actual_parameters.size(); ++index) {
        const auto actual_entry = actual.state().find(actual_parameters[index].unsafeGetTensorImpl());
        const auto expected_entry = expected.state().find(expected_parameters[index].unsafeGetTensorImpl());
        REQUIRE(actual_entry != actual.state().end(), "missing actual AdamW state");
        REQUIRE(expected_entry != expected.state().end(), "missing expected AdamW state");
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

struct TrainerSnapshot {
    std::string digest;
    diamond_training::TrainingConfig config;
    uint64_t training_step;
};

TrainerSnapshot snapshot(const diamond_training::Trainer& trainer) {
    return {
        .digest = diamond_training::canonical_model_digest(trainer.model()),
        .config = trainer.config(),
        .training_step = trainer.training_step(),
    };
}

void check_snapshot(const diamond_training::Trainer& trainer, const TrainerSnapshot& expected) {
    CHECK_EQ(diamond_training::canonical_model_digest(trainer.model()), expected.digest);
    CHECK_EQ(trainer.config().learning_rate, expected.config.learning_rate);
    CHECK_EQ(trainer.config().weight_decay, expected.config.weight_decay);
    CHECK_EQ(trainer.training_step(), expected.training_step);
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
        return false;
    } catch (const diamond_training::CheckpointError&) {
        return true;
    }
}

void write_text(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(static_cast<bool>(output), "cannot corrupt checkpoint archive");
    output << contents;
    REQUIRE(static_cast<bool>(output), "cannot corrupt checkpoint archive");
}

diamond_training::Trainer make_trained(const diamond_training::ResolvedDevice& device,
                                       const diamond_training::Compatibility& checkpoint_compatibility,
                                       const std::vector<diamond_training::TrainingSample>& samples) {
    diamond_training::Trainer trainer(
        diamond_model::DiamondModel(8, 1, 4, 1), checkpoint_compatibility, kConfig, device);
    CHECK_EQ(trainer.train(samples).training_step, uint64_t{1});
    REQUIRE(!trainer.optimizer().state().empty(), "training must create AdamW state");
    return trainer;
}

diamond_training::Trainer clone_trainer(const std::filesystem::path& root,
                                        diamond_training::Trainer& source,
                                        const diamond_training::ResolvedDevice& device,
                                        const diamond_training::Compatibility& checkpoint_compatibility) {
    (void)diamond_training::save_checkpoint_v2(root, source);
    diamond_training::Trainer clone(
        diamond_model::DiamondModel(8, 1, 4, 1), checkpoint_compatibility, kConfig, device);
    (void)diamond_training::load_checkpoint_v2(root, clone, device);
    return clone;
}

void check_rejected_load_is_unchanged(
    const std::filesystem::path& corrupt_root, diamond_training::Trainer& destination,
    diamond_training::Trainer& control,
    const std::vector<diamond_training::TrainingSample>& samples,
    const diamond_training::ResolvedDevice& device) {
    const auto before = snapshot(destination);
    check_adamw_equal(destination.optimizer(), control.optimizer());
    CHECK(rejects([&] { (void)diamond_training::load_checkpoint_v2(corrupt_root, destination, device); }));
    check_snapshot(destination, before);
    check_adamw_equal(destination.optimizer(), control.optimizer());

    // A failed staged restore must not perturb the next live optimizer update.
    const auto destination_next = destination.train(samples);
    const auto control_next = control.train(samples);
    CHECK_EQ(destination_next.training_step, control_next.training_step);
    CHECK(std::fabs(destination_next.total_loss - control_next.total_loss) < 1e-12);
    CHECK_EQ(diamond_training::canonical_model_digest(destination.model()),
             diamond_training::canonical_model_digest(control.model()));
    check_adamw_equal(destination.optimizer(), control.optimizer());
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: checkpoint_v2_transaction_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);
    const auto checkpoint_compatibility = compatibility();
    const auto device = diamond_training::resolve_device("cpu");
    const auto samples = samples_for(checkpoint_compatibility);
    auto trainer = make_trained(device, checkpoint_compatibility, samples);
    const auto first = diamond_training::save_checkpoint_v2(scratch / "activation", trainer);
    const auto current_before = [&] { std::ifstream in(scratch / "activation" / "CURRENT"); std::string value; std::getline(in, value); return value; }();
    _putenv_s("DIAMOND_CHECKPOINT_FAIL_ACTIVATE", "1");
    bool failed = false; try { (void)diamond_training::save_checkpoint_v2(scratch / "activation", trainer); } catch (const diamond_training::CheckpointError&) { failed = true; }
    _putenv_s("DIAMOND_CHECKPOINT_FAIL_ACTIVATE", "");
    CHECK(failed);
    const auto current_after = [&] { std::ifstream in(scratch / "activation" / "CURRENT"); std::string value; std::getline(in, value); return value; }();
    CHECK_EQ(current_after, current_before); CHECK(std::filesystem::exists(first.generation));

    auto source = make_trained(device, checkpoint_compatibility, samples);
    const auto corrupt_optimizer = scratch / "corrupt-optimizer";
    const auto corrupt_info = diamond_training::save_checkpoint_v2(corrupt_optimizer, source);
    write_text(corrupt_info.generation / "optimizer.pt", "not a LibTorch archive");
    auto optimizer_destination = make_trained(device, checkpoint_compatibility, samples);
    auto optimizer_control = clone_trainer(scratch / "optimizer-control", optimizer_destination,
                                           device, checkpoint_compatibility);
    check_rejected_load_is_unchanged(corrupt_optimizer, optimizer_destination, optimizer_control,
                                     samples, device);

    // This archive parses successfully, but staged validation must reject its NaN AdamW state.
    const auto nan_optimizer = scratch / "nan-optimizer";
    const auto nan_info = diamond_training::save_checkpoint_v2(nan_optimizer, source);
    auto nan_source = make_trained(device, checkpoint_compatibility, samples);
    auto first_state = nan_source.optimizer().state().begin();
    auto* state = dynamic_cast<torch::optim::AdamWParamState*>(first_state->second.get());
    REQUIRE(state != nullptr, "AdamW state type");
    state->exp_avg().fill_(std::numeric_limits<float>::quiet_NaN());
    torch::serialize::OutputArchive archive;
    nan_source.optimizer().save(archive);
    archive.save_to((nan_info.generation / "optimizer.pt").string());
    auto nan_destination = make_trained(device, checkpoint_compatibility, samples);
    auto nan_control = clone_trainer(scratch / "nan-control", nan_destination,
                                     device, checkpoint_compatibility);
    check_rejected_load_is_unchanged(nan_optimizer, nan_destination, nan_control, samples, device);

    const auto corrupt_weights = scratch / "corrupt-weights";
    const auto weight_info = diamond_training::save_checkpoint_v2(corrupt_weights, source);
    write_text(weight_info.generation / "state.pt", "not a LibTorch archive");
    auto weight_destination = make_trained(device, checkpoint_compatibility, samples);
    const auto weight_before = snapshot(weight_destination);
    auto optimizer_before = clone_trainer(scratch / "weights-control", weight_destination,
                                          device, checkpoint_compatibility);
    CHECK(rejects([&] {
        (void)diamond_training::load_checkpoint_v2_weights(corrupt_weights, weight_destination.model(), device);
    }));
    check_snapshot(weight_destination, weight_before);
    check_adamw_equal(weight_destination.optimizer(), optimizer_before.optimizer());
    const auto weights_next = weight_destination.train(samples);
    const auto weights_control_next = optimizer_before.train(samples);
    CHECK_EQ(weights_next.training_step, weights_control_next.training_step);
    CHECK_EQ(diamond_training::canonical_model_digest(weight_destination.model()),
             diamond_training::canonical_model_digest(optimizer_before.model()));
    check_adamw_equal(weight_destination.optimizer(), optimizer_before.optimizer());

    std::filesystem::remove_all(scratch);
    return soo_test::report("checkpoint_v2_transaction_test");
}
