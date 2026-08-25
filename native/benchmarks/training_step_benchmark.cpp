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
#include "diamond_training/trainer.hpp"

namespace {

struct Args {
    std::filesystem::path artifact = "models/soo/2.0.0";
    std::uint64_t batch_size = 256;
    std::uint64_t repetitions = 1;
    int threads = 0;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (i + 1 >= argc)
            throw std::invalid_argument("usage: training_step_benchmark [--artifact DIR] [--batch-size N] [--repetitions N] [--threads N]");
        const std::string value = argv[++i];
        try {
            if (option == "--artifact") args.artifact = value;
            else if (option == "--batch-size") args.batch_size = std::stoull(value);
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
    diamond_training::Trainer trainer(model, compatibility,
                                      {.learning_rate = 1e-3, .weight_decay = 1e-4});
    const std::vector<diamond_training::TrainingSample> samples(
        static_cast<std::size_t>(args.batch_size), sample(compatibility, artifact.input_features));
    (void)trainer.train(samples);
    const auto start = std::chrono::steady_clock::now();
    diamond_training::TrainingMetrics metrics{};
    for (std::uint64_t i = 0; i < args.repetitions; ++i) metrics = trainer.train(samples);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"name\":\"training_step\",\"repetitions\":" << args.repetitions
              << ",\"batch_size\":" << args.batch_size
              << ",\"width\":" << artifact.width
              << ",\"residual_blocks\":" << artifact.residual_blocks
              << ",\"torch_threads\":" << torch::get_num_threads()
              << ",\"elapsed_seconds\":" << elapsed
              << ",\"seconds_per_repetition\":" << elapsed / args.repetitions
              << ",\"loss\":" << metrics.total_loss
              << ",\"training_step\":" << metrics.training_step << "}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "training_step_benchmark: " << error.what() << '\n'; return 1; }
}
