#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "diamond_model/soo_model.hpp"
#include "diamond_support/build_provenance.hpp"
#include "diamond_training/checkpoint.hpp"

namespace {

struct Args { std::uint64_t repetitions = 1; std::filesystem::path scratch; };
Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if ((option != "--repetitions" && option != "--scratch") || i + 1 >= argc)
            throw std::invalid_argument("usage: checkpoint_benchmark --scratch PATH [--repetitions N]");
        const std::string value = argv[++i];
        if (option == "--scratch") args.scratch = value;
        else { try { args.repetitions = std::stoull(value); } catch (...) { throw std::invalid_argument("repetitions must be a positive integer"); } }
    }
    if (args.scratch.empty() || args.repetitions == 0) throw std::invalid_argument("scratch and positive repetitions are required");
    return args;
}

int run(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    std::filesystem::remove_all(args.scratch);
    auto compatibility =
        diamond_training::Compatibility::soo("benchmark-1", {.residual_blocks = 1, .width = 8});
    const auto device = diamond_training::resolve_device("cpu");
    diamond_training::Trainer trainer(diamond_model::DiamondModel(8, 1, 4, 1), compatibility,
                                      {.learning_rate = 1e-3, .weight_decay = 1e-4}, device);
    diamond_training::CheckpointInfo checkpoint;
    auto operation = [&] {
        checkpoint = diamond_training::save_checkpoint_v3(
            args.scratch, trainer,
            {.initialization_mode = diamond_training::CheckpointInitializationMode::scratch,
             .run_id = "checkpoint-benchmark",
             .iteration = 0,
             .model_step = trainer.training_step()});
        checkpoint = diamond_training::validate_checkpoint_v2(args.scratch);
    };
    operation();
    std::vector<double> seconds;
    for (std::uint64_t i = 0; i < args.repetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        operation();
        seconds.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    auto sorted = seconds;
    std::sort(sorted.begin(), sorted.end());
    std::cout << "{\"schema_version\":1,\"benchmark\":\"checkpoint\",\"workload\":{\"repetitions\":"
              << args.repetitions
              << ",\"warmups\":1},\"environment\":{\"requested_device\":\"cpu\",\"canonical_"
                 "device\":\"cpu\",\"precision\":\"float32\"},\"provenance\":"
              << diamond_support::build_provenance_json() << ",\"samples_seconds\":[";
    for (std::size_t i = 0; i < seconds.size(); ++i)
        std::cout << (i ? "," : "") << seconds[i];
    std::cout << "],\"summary_seconds\":{\"min\":" << sorted.front()
              << ",\"median\":" << sorted[sorted.size() / 2] << ",\"max\":" << sorted.back()
              << ",\"range\":" << sorted.back() - sorted.front()
              << "},\"domain\":{\"operation\":\"save_validate_v3\",\"model_sha256\":\""
              << checkpoint.model_digest << "\",\"optimizer_sha256\":\""
              << checkpoint.optimizer_digest << "\",\"training_step\":" << checkpoint.training_step
              << "}}\n";
    return 0;
}
} // namespace
int main(int argc, char** argv) { try { return run(argc, argv); } catch (const std::exception& error) { std::cerr << "checkpoint_benchmark: " << error.what() << '\n'; return 1; } }
