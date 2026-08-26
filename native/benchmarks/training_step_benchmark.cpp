#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_support/build_provenance.hpp"
#include "diamond_training/trainer.hpp"

namespace {

struct Args {
    std::filesystem::path artifact = "models/soo/2.0.0";
    std::string device = "cpu";
    std::uint64_t batch_size = 256;
    std::uint64_t warmups = 1;
    std::uint64_t repetitions = 1;
    int threads = 0;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (i + 1 >= argc)
            throw std::invalid_argument("usage: training_step_benchmark [--artifact DIR] [--device cpu|cuda|cuda:N] [--batch-size N] [--warmups N] [--repetitions N] [--threads N]");
        const std::string value = argv[++i];
        try {
            if (option == "--artifact") args.artifact = value;
            else if (option == "--device") args.device = value;
            else if (option == "--batch-size") args.batch_size = std::stoull(value);
            else if (option == "--warmups") args.warmups = std::stoull(value);
            else if (option == "--repetitions") args.repetitions = std::stoull(value);
            else if (option == "--threads") args.threads = std::stoi(value);
            else throw std::invalid_argument("unknown option");
        } catch (...) {
            throw std::invalid_argument("benchmark arguments must be valid positive values");
        }
    }
    if (args.batch_size == 0 || args.repetitions == 0 || args.threads < 0)
        throw std::invalid_argument("batch size, repetitions, and threads must be positive");
    return args;
}

diamond_training::TrainingSample sample(const diamond_training::Compatibility& compatibility,
                                        int64_t input_features) {
    diamond_training::TrainingSample value;
    value.compatibility = compatibility;
    value.node_features.resize(static_cast<std::size_t>(73 * input_features));
    for (std::size_t i = 0; i < value.node_features.size(); ++i)
        value.node_features[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0F;
    value.canonical_player_ids = {1, 2};
    value.sparse_policy = {{0, 1.0F}};
    value.value_target = {1.0F};
    return value;
}

int run(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    torch::manual_seed(7);
    if (args.threads > 0) torch::set_num_threads(args.threads);
    torch::set_num_interop_threads(1);
    const auto artifact = diamond_model::validate_deployment_artifact(args.artifact, "soo");
    auto compatibility = diamond_training::Compatibility::soo(
        artifact.model_version, {.residual_blocks = artifact.residual_blocks, .width = artifact.width});
    auto model = diamond_model::DiamondModel(artifact.width, artifact.residual_blocks,
                                             artifact.input_features, artifact.value_size);
    model->load_weights(artifact.weights);
    const auto device = diamond_training::resolve_device(args.device);
    diamond_training::Trainer trainer(model, compatibility,
                                      {.learning_rate = 1e-3, .weight_decay = 1e-4}, device);
    const std::vector<diamond_training::TrainingSample> samples(
        static_cast<std::size_t>(args.batch_size), sample(compatibility, artifact.input_features));
    for (std::uint64_t i = 0; i < args.warmups; ++i) (void)trainer.train(samples);
    diamond_training::TrainingMetrics metrics{};
    std::vector<double> seconds;
    seconds.reserve(args.repetitions);
    for (std::uint64_t i = 0; i < args.repetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        metrics = trainer.train(samples);
        seconds.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    auto sorted = seconds;
    std::sort(sorted.begin(), sorted.end());
    std::cout
        << "{\"schema_version\":1,\"benchmark\":\"training_step\",\"workload\":{\"repetitions\":"
        << args.repetitions << ",\"warmups\":" << args.warmups
        << ",\"batch_size\":" << args.batch_size << ",\"threads\":" << torch::get_num_threads()
        << "},\"environment\":{\"requested_device\":\"" << device.requested_name
        << "\",\"canonical_device\":\"" << device.canonical_name
        << "\",\"precision\":\"float32\"},\"provenance\":"
        << diamond_support::build_provenance_json() << ",\"samples_seconds\":[";
    for (std::size_t i = 0; i < seconds.size(); ++i)
        std::cout << (i ? "," : "") << seconds[i];
    std::cout << "],\"summary_seconds\":{\"min\":" << sorted.front()
              << ",\"median\":" << sorted[sorted.size() / 2] << ",\"max\":" << sorted.back()
              << ",\"range\":" << sorted.back() - sorted.front()
              << "},\"domain\":{\"model_version\":\"" << artifact.model_version
              << "\",\"model_sha256\":\"" << artifact.model_sha256 << "\",\"runtime_sha256\":\""
              << artifact.runtime_sha256 << "\",\"width\":" << artifact.width
              << ",\"residual_blocks\":" << artifact.residual_blocks
              << ",\"total_loss\":" << metrics.total_loss
              << ",\"policy_loss\":" << metrics.policy_loss
              << ",\"value_loss\":" << metrics.value_loss
              << ",\"collation_seconds\":" << metrics.collation_seconds
              << ",\"h2d_seconds\":" << metrics.h2d_seconds
              << ",\"forward_seconds\":" << metrics.forward_seconds
              << ",\"backward_seconds\":" << metrics.backward_seconds
              << ",\"optimizer_seconds\":" << metrics.optimizer_seconds
              << ",\"total_step_seconds\":" << metrics.total_step_seconds
              << ",\"samples_per_second\":" << metrics.samples_per_second
              << ",\"peak_cuda_memory_available\":"
              << (metrics.peak_cuda_memory_available ? "true" : "false")
              << ",\"peak_cuda_allocated_bytes\":" << metrics.peak_cuda_allocated_bytes
              << ",\"peak_cuda_reserved_bytes\":" << metrics.peak_cuda_reserved_bytes
              << ",\"training_step\":" << metrics.training_step << "}}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "training_step_benchmark: " << error.what() << '\n'; return 1; }
}
