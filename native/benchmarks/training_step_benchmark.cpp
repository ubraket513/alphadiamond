#include <chrono>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "diamond_model/soo_model.hpp"
#include "diamond_training/trainer.hpp"

namespace {

struct Args { std::uint64_t repetitions = 1; };

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option != "--repetitions" || i + 1 >= argc)
            throw std::invalid_argument("usage: training_step_benchmark [--repetitions N]");
        try { args.repetitions = std::stoull(argv[++i]); }
        catch (...) { throw std::invalid_argument("repetitions must be a positive integer"); }
        if (args.repetitions == 0) throw std::invalid_argument("repetitions must be positive");
    }
    return args;
}

diamond_training::TrainingSample sample() {
    diamond_training::TrainingSample value;
    value.compatibility = diamond_training::Compatibility::soo("benchmark-1", {.residual_blocks = 1, .width = 8});
    value.node_features.resize(73 * 4);
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
    auto compatibility = diamond_training::Compatibility::soo("benchmark-1", {.residual_blocks = 1, .width = 8});
    diamond_training::Trainer trainer(diamond_model::DiamondModel(8, 1, 4, 1), compatibility,
                                      {.learning_rate = 1e-3, .weight_decay = 1e-4});
    const std::vector<diamond_training::TrainingSample> samples{sample()};
    (void)trainer.train(samples);
    const auto start = std::chrono::steady_clock::now();
    diamond_training::TrainingMetrics metrics{};
    for (std::uint64_t i = 0; i < args.repetitions; ++i) metrics = trainer.train(samples);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"name\":\"training_step\",\"repetitions\":" << args.repetitions
              << ",\"elapsed_seconds\":" << elapsed
              << ",\"seconds_per_repetition\":" << elapsed / args.repetitions
              << ",\"training_step\":" << metrics.training_step << "}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "training_step_benchmark: " << error.what() << '\n'; return 1; }
}
