#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/serialize.h>

#ifdef CHECK
#undef CHECK
#endif
#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_training/trainer.hpp"

namespace {

constexpr float kCpuRtol = 1e-5F;
constexpr float kCpuAtol = 1e-6F;
constexpr float kCudaRtol = 5e-3F;
constexpr float kCudaAtol = 5e-4F;

diamond_training::ResolvedDevice cpu_device() {
    return diamond_training::resolve_device("cpu");
}

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

void check_metrics(const diamond_training::TrainingMetrics& metrics, size_t batch_size,
                   bool cuda) {
    CHECK(std::isfinite(metrics.total_loss));
    CHECK(std::isfinite(metrics.policy_loss));
    CHECK(std::isfinite(metrics.value_loss));
    for (const double duration : {metrics.collation_seconds, metrics.h2d_seconds,
                                  metrics.forward_seconds, metrics.backward_seconds,
                                  metrics.optimizer_seconds}) {
        CHECK(std::isfinite(duration));
        CHECK(duration >= 0.0);
    }
    CHECK(std::isfinite(metrics.total_step_seconds));
    CHECK(metrics.total_step_seconds > 0.0);
    CHECK(std::isfinite(metrics.samples_per_second));
    CHECK(metrics.samples_per_second > 0.0);
    check_close(static_cast<float>(metrics.samples_per_second * metrics.total_step_seconds),
                static_cast<float>(batch_size), "training throughput identity");
    if (cuda) {
        CHECK(metrics.peak_cuda_memory_available);
        if (metrics.peak_cuda_memory_available) {
            CHECK(metrics.peak_cuda_allocated_bytes > 0);
            CHECK(metrics.peak_cuda_reserved_bytes >= metrics.peak_cuda_allocated_bytes);
        }
    } else {
        CHECK_EQ(metrics.peak_cuda_allocated_bytes, uint64_t{0});
        CHECK_EQ(metrics.peak_cuda_reserved_bytes, uint64_t{0});
        CHECK(!metrics.peak_cuda_memory_available);
    }
}

void check_model_device(const diamond_model::DiamondModel& model, const torch::Device& device,
                        bool gradients) {
    for (const auto& parameter : model->named_parameters()) {
        CHECK(parameter.value().device() == device);
        CHECK(torch::isfinite(parameter.value()).all().item<bool>());
        if (gradients) {
            REQUIRE(parameter.value().grad().defined(), parameter.key().c_str());
            CHECK(parameter.value().grad().device() == device);
            CHECK(torch::isfinite(parameter.value().grad()).all().item<bool>());
        }
    }
    for (const auto& buffer : model->named_buffers()) {
        CHECK(buffer.value().device() == device);
        CHECK(torch::isfinite(buffer.value()).all().item<bool>());
    }
}

void check_optimizer_device(const torch::optim::AdamW& optimizer, const torch::Device& device) {
    REQUIRE(optimizer.state().size() == optimizer.size(), "AdamW state coverage");
    for (const auto& entry : optimizer.state()) {
        const auto* state = dynamic_cast<const torch::optim::AdamWParamState*>(entry.second.get());
        REQUIRE(state != nullptr, "AdamW state type");
        CHECK(state->step() > 0);
        for (const auto& tensor : {state->exp_avg(), state->exp_avg_sq(),
                                  state->max_exp_avg_sq()}) {
            if (!tensor.defined()) continue;
            CHECK(tensor.device() == device);
            CHECK(torch::isfinite(tensor).all().item<bool>());
        }
    }
}

void check_named_close(const diamond_model::DiamondModel& actual,
                       const diamond_model::DiamondModel& expected, bool gradients,
                       float rtol, float atol, const char* name) {
    const auto actual_parameters = actual->named_parameters();
    const auto expected_parameters = expected->named_parameters();
    REQUIRE(actual_parameters.size() == expected_parameters.size(), name);
    for (const auto& expected_parameter : expected_parameters) {
        const auto* actual_parameter = actual_parameters.find(expected_parameter.key());
        REQUIRE(actual_parameter != nullptr, expected_parameter.key().c_str());
        const auto actual_tensor =
            (gradients ? actual_parameter->grad() : *actual_parameter).detach().to(torch::kCPU);
        const auto expected_tensor =
            (gradients ? expected_parameter.value().grad() : expected_parameter.value())
                .detach()
                .to(torch::kCPU);
        if (!torch::allclose(actual_tensor, expected_tensor, rtol, atol)) {
            const float max_difference =
                (actual_tensor - expected_tensor).abs().max().item<float>();
            soo_test::fail(__FILE__, __LINE__, std::string(name) + " " +
                           expected_parameter.key() + " max difference=" +
                           std::to_string(max_difference));
        }
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

std::string serialized_optimizer(const torch::optim::AdamW& optimizer) {
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    torch::serialize::OutputArchive output;
    optimizer.save(output);
    output.save_to(bytes);
    return bytes.str();
}

void check_validation(const std::filesystem::path& root) {
    auto model = load_model(root, "soo");
    diamond_training::Trainer trainer(model, compatibility_for("soo"), parity_config(),
                                      cpu_device());
    const std::string initial_digest = diamond_training::canonical_model_digest(trainer.model());
    CHECK_EQ(trainer.training_step(), uint64_t{0});
    CHECK(trainer.optimizer().state().empty());
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

    CHECK_EQ(trainer.training_step(), uint64_t{0});
    CHECK(trainer.optimizer().state().empty());
    CHECK(diamond_training::canonical_model_digest(trainer.model()) == initial_digest);
    for (const torch::Tensor& parameter : trainer.model()->parameters()) {
        CHECK(!parameter.grad().defined());
    }

    auto min_model = load_model(root, "min");
    diamond_training::Trainer min_trainer(min_model, compatibility_for("min"), parity_config(),
                                          cpu_device());
    auto min_sample = samples_for(root, "min").front();
    min_sample.value_target.pop_back();
    check_invalid_argument([&] { (void)min_trainer.train(std::span(&min_sample, 1)); },
                           "Min value width");
}

void check_invalid_is_transactional(const std::filesystem::path& root) {
    auto model = load_model(root, "soo");
    diamond_training::Trainer trainer(model, compatibility_for("soo"), parity_config(),
                                      cpu_device());
    const auto valid = samples_for(root, "soo");
    (void)trainer.train(valid);

    const std::string digest_before = diamond_training::canonical_model_digest(trainer.model());
    const std::string optimizer_before = serialized_optimizer(trainer.optimizer());
    const uint64_t step_before = trainer.training_step();
    std::vector<torch::Tensor> gradients_before;
    for (const torch::Tensor& parameter : trainer.model()->parameters()) {
        gradients_before.push_back(parameter.grad().detach().clone());
    }

    auto invalid = valid;
    invalid.back().node_features.back() = std::numeric_limits<float>::quiet_NaN();
    check_invalid_argument([&] { (void)trainer.train(invalid); },
                           "invalid batch after populated optimizer");

    CHECK_EQ(trainer.training_step(), step_before);
    CHECK(diamond_training::canonical_model_digest(trainer.model()) == digest_before);
    CHECK(serialized_optimizer(trainer.optimizer()) == optimizer_before);
    const auto parameters = trainer.model()->parameters();
    REQUIRE(parameters.size() == gradients_before.size(), "gradient count after invalid batch");
    for (size_t index = 0; index < parameters.size(); ++index) {
        CHECK(torch::equal(parameters[index].grad(), gradients_before[index]));
    }
}

void check_role(const diamond_model::DiamondModel& model, bool trainable, const char* name) {
    REQUIRE(static_cast<bool>(model), name);
    CHECK(model->is_training() == trainable);
    for (const auto& parameter : model->named_parameters()) {
        CHECK(parameter.value().requires_grad() == trainable);
    }
}

void check_deep_storage(const diamond_model::DiamondModel& original,
                        const diamond_model::DiamondModel& snapshot, bool parameters,
                        const char* name) {
    const auto original_tensors = parameters ? original->named_parameters() : original->named_buffers();
    const auto snapshot_tensors = parameters ? snapshot->named_parameters() : snapshot->named_buffers();
    REQUIRE(original_tensors.size() == snapshot_tensors.size(), name);
    for (const auto& original_tensor : original_tensors) {
        const auto* copied_tensor = snapshot_tensors.find(original_tensor.key());
        REQUIRE(copied_tensor != nullptr, name);
        CHECK(original_tensor.value().data_ptr() != copied_tensor->data_ptr());
        CHECK(torch::equal(original_tensor.value(), *copied_tensor));
    }
}

void check_snapshot_isolation(const std::filesystem::path& root) {
    const auto compatibility = compatibility_for("soo");
    auto source = load_model(root, "soo");
    auto actor = diamond_training::snapshot_model(source, compatibility, torch::Device(torch::kCPU),
                                                  diamond_training::ModelRole::actor);
    const auto actor_holder_alias = actor;
    CHECK(actor_holder_alias->parameters().front().data_ptr() ==
          actor->parameters().front().data_ptr());
    check_role(actor, false, "actor role");

    const std::string actor_digest = diamond_training::canonical_model_digest(actor);
    auto nonfinite = diamond_training::snapshot_model(
        actor, compatibility, torch::Device(torch::kCPU), diamond_training::ModelRole::candidate);
    {
        torch::NoGradGuard no_grad;
        nonfinite->parameters().front().flatten()[0].fill_(
            std::numeric_limits<float>::quiet_NaN());
    }
    check_invalid_argument(
        [&] { (void)diamond_training::canonical_model_digest(nonfinite); },
        "non-finite model digest");
    auto wrong_dtype = diamond_training::snapshot_model(
        actor, compatibility, torch::Device(torch::kCPU), diamond_training::ModelRole::candidate);
    wrong_dtype->to(torch::kFloat64);
    check_invalid_argument(
        [&] { (void)diamond_training::canonical_model_digest(wrong_dtype); },
        "non-FP32 model digest");
    const auto actor_first_parameter = actor->parameters().front().detach().clone();
    const auto actor_adjacency = actor->adjacency.detach().clone();
    diamond_training::Trainer trainer(actor, compatibility, parity_config(), cpu_device());
    CHECK_EQ(trainer.device().canonical_name, std::string("cpu"));
    check_role(trainer.model(), true, "learner role");
    check_model_device(trainer.model(), trainer.device().torch_device, false);
    check_deep_storage(actor, trainer.model(), true, "learner parameter storage");
    check_deep_storage(actor, trainer.model(), false, "learner buffer storage");

    const auto samples = samples_for(root, "soo");
    const auto initial_candidate = trainer.candidate_snapshot();
    check_role(initial_candidate, false, "candidate role");
    check_model_device(initial_candidate, trainer.device().torch_device, false);
    CHECK(diamond_training::canonical_model_digest(initial_candidate) ==
          diamond_training::canonical_model_digest(trainer.model()));
    check_deep_storage(trainer.model(), initial_candidate, true, "candidate parameter storage");
    check_deep_storage(trainer.model(), initial_candidate, false, "candidate buffer storage");

    const auto metrics = trainer.train(samples);
    check_metrics(metrics, samples.size(), false);
    check_model_device(trainer.model(), trainer.device().torch_device, true);
    check_optimizer_device(trainer.optimizer(), trainer.device().torch_device);
    CHECK(torch::equal(actor->parameters().front(), actor_first_parameter));
    CHECK(torch::equal(actor->adjacency, actor_adjacency));
    CHECK(diamond_training::canonical_model_digest(actor) == actor_digest);
    check_role(actor, false, "actor remains inference-only");

    const auto older_candidate = trainer.candidate_snapshot();
    const std::string older_digest = diamond_training::canonical_model_digest(older_candidate);
    CHECK(older_digest == diamond_training::canonical_model_digest(trainer.model()));
    std::stringstream serialized(std::ios::in | std::ios::out | std::ios::binary);
    torch::serialize::OutputArchive output;
    older_candidate->save(output);
    output.save_to(serialized);
    serialized.seekg(0);
    torch::serialize::InputArchive input;
    input.load_from(serialized);
    auto reloaded = diamond_model::DiamondModel(8, 1, 4, 1);
    reloaded->load(input);
    CHECK(diamond_training::canonical_model_digest(reloaded) == older_digest);

    (void)trainer.train(samples);
    CHECK(diamond_training::canonical_model_digest(older_candidate) == older_digest);
    const auto newer_candidate = trainer.candidate_snapshot();
    CHECK(diamond_training::canonical_model_digest(newer_candidate) != older_digest);
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

    diamond_training::Trainer trainer(model, compatibility_for(family), parity_config(),
                                      cpu_device());
    CHECK(trainer.device().torch_device.is_cpu());
    check_model_device(trainer.model(), trainer.device().torch_device, false);
    const auto metrics = trainer.train(samples);
    REQUIRE(expected_losses.size() == 3, "loss fixture");
    check_close(static_cast<float>(metrics.total_loss), expected_losses[0], family + " total loss");
    check_close(static_cast<float>(metrics.policy_loss), expected_losses[1], family + " policy loss");
    check_close(static_cast<float>(metrics.value_loss), expected_losses[2], family + " value loss");
    CHECK_EQ(metrics.training_step, uint64_t{1});
    check_metrics(metrics, samples.size(), false);
    check_model_device(trainer.model(), trainer.device().torch_device, true);
    check_optimizer_device(trainer.optimizer(), trainer.device().torch_device);

    const auto source = trainer.model()->policy_source->weight;
    CHECK(source.grad().defined());
    CHECK(source.grad().abs().sum().item<float>() > 0.0F);
    check_selected_values(source.grad(), expected_gradient, family + " policy-source gradient");
    check_selected_values(source, expected_after, family + " AdamW update");

    const auto resumed = trainer.train(samples);
    REQUIRE(expected_resumed_losses.size() == 3, "resumed loss fixture");
    check_close(static_cast<float>(resumed.total_loss), expected_resumed_losses[0],
                family + " resumed total loss");
    CHECK_EQ(resumed.training_step, uint64_t{2});
    check_metrics(resumed, samples.size(), false);
    check_selected_values(source, expected_resumed, family + " AdamW stateful update");
}

std::vector<diamond_training::TrainingSample> repeated_samples(
    const std::vector<diamond_training::TrainingSample>& source, size_t count) {
    REQUIRE(!source.empty(), "repeated training source");
    std::vector<diamond_training::TrainingSample> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        result.push_back(source[index % source.size()]);
    }
    return result;
}

void check_cuda_close(double actual, double expected, const char* name) {
    if (std::fabs(actual - expected) >
        static_cast<double>(kCudaAtol) + static_cast<double>(kCudaRtol) * std::fabs(expected)) {
        soo_test::fail(__FILE__, __LINE__, std::string(name) + " differs: actual=" +
                       std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

void check_cuda_path(const std::filesystem::path& root) {
    REQUIRE(torch::cuda::is_available(),
            "CUDA-labelled trainer test requires an available CUDA device");
    const auto cuda = diamond_training::resolve_device("cuda:0");
    REQUIRE(cuda.torch_device.is_cuda(), "resolved CUDA trainer device");
    const auto source_samples = samples_for(root, "soo");
    const std::string initial_digest =
        diamond_training::canonical_model_digest(load_model(root, "soo"));

    for (const size_t batch_size : {size_t{1}, size_t{17}, size_t{64}, size_t{256}}) {
        auto source = load_model(root, "soo");
        diamond_training::Trainer trainer(source, compatibility_for("soo"), parity_config(), cuda);
        CHECK_EQ(trainer.device().canonical_name, std::string("cuda:0"));
        check_model_device(trainer.model(), cuda.torch_device, false);
        const auto batch = repeated_samples(source_samples, batch_size);
        const auto metrics = trainer.train(batch);
        CHECK_EQ(metrics.training_step, uint64_t{1});
        check_metrics(metrics, batch_size, true);
        check_model_device(trainer.model(), cuda.torch_device, true);
        check_optimizer_device(trainer.optimizer(), cuda.torch_device);
        const auto candidate = trainer.candidate_snapshot();
        check_role(candidate, false, "CUDA candidate role");
        check_model_device(candidate, cuda.torch_device, false);
        CHECK(diamond_training::canonical_model_digest(candidate) != initial_digest);
    }

    auto parity_source = load_model(root, "soo");
    const auto parity_batch = repeated_samples(source_samples, 2);
    diamond_training::Trainer cpu_trainer(parity_source, compatibility_for("soo"), parity_config(),
                                          cpu_device());
    diamond_training::Trainer cuda_trainer(parity_source, compatibility_for("soo"), parity_config(),
                                           cuda);
    const auto cpu_metrics = cpu_trainer.train(parity_batch);
    const auto cuda_metrics = cuda_trainer.train(parity_batch);
    check_cuda_close(cuda_metrics.total_loss, cpu_metrics.total_loss, "CUDA total loss");
    check_cuda_close(cuda_metrics.policy_loss, cpu_metrics.policy_loss, "CUDA policy loss");
    check_cuda_close(cuda_metrics.value_loss, cpu_metrics.value_loss, "CUDA value loss");
    check_named_close(cuda_trainer.model(), cpu_trainer.model(), false, kCudaRtol, kCudaAtol,
                      "CUDA AdamW update");
    check_named_close(cuda_trainer.model(), cpu_trainer.model(), true, kCudaRtol, kCudaAtol,
                      "CUDA gradients");
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--cuda"),
            "usage: training_step_parity_test <fixture-dir> [--cuda]");
    const auto root = std::filesystem::path(argv[1]);
    if (argc == 3) {
        check_cuda_path(root);
        return soo_test::report("training_step_parity_test_cuda");
    }
    check_validation(root);
    check_invalid_is_transactional(root);
    check_snapshot_isolation(root);
    check_parity(root, "soo");
    check_parity(root, "min");
    return soo_test::report("training_step_parity_test");
}
