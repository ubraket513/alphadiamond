#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "diamond_model/soo_model.hpp"
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
    auto compatibility = diamond_training::Compatibility::soo("benchmark-1", {.residual_blocks = 1, .width = 8});
    diamond_training::Trainer trainer(diamond_model::DiamondModel(8, 1, 4, 1), compatibility,
                                      {.learning_rate = 1e-3, .weight_decay = 1e-4});
    auto operation = [&] { (void)diamond_training::save_checkpoint_v2(args.scratch, trainer); (void)diamond_training::validate_checkpoint_v2(args.scratch); };
    operation();
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < args.repetitions; ++i) operation();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"name\":\"checkpoint\",\"repetitions\":" << args.repetitions
              << ",\"elapsed_seconds\":" << elapsed
              << ",\"seconds_per_repetition\":" << elapsed / args.repetitions << "}\n";
    return 0;
}
} // namespace
int main(int argc, char** argv) { try { return run(argc, argv); } catch (const std::exception& error) { std::cerr << "checkpoint_benchmark: " << error.what() << '\n'; return 1; } }
