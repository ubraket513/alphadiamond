#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "diamond_pipeline/replay_store.hpp"

namespace {
struct Args { std::uint64_t repetitions = 1; std::filesystem::path scratch; };
Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if ((option != "--repetitions" && option != "--scratch") || i + 1 >= argc)
            throw std::invalid_argument("usage: replay_benchmark --scratch PATH [--repetitions N]");
        const std::string value = argv[++i];
        if (option == "--scratch") args.scratch = value;
        else { try { args.repetitions = std::stoull(value); } catch (...) { throw std::invalid_argument("repetitions must be a positive integer"); } }
    }
    if (args.scratch.empty() || args.repetitions == 0) throw std::invalid_argument("scratch and positive repetitions are required");
    return args;
}
diamond_pipeline::Episode episode(const diamond_pipeline::Compatibility& compatibility, std::uint64_t id) {
    diamond_pipeline::Episode value;
    value.game_id = "benchmark-game-" + std::to_string(id);
    value.seed = id; value.compatibility = compatibility;
    diamond_training::TrainingSample sample;
    sample.compatibility = compatibility; sample.node_features.assign(73 * 4, 0.125F);
    sample.canonical_player_ids = {1, 2}; sample.sparse_policy = {{0, 1.0F}}; sample.value_target = {1.0F};
    value.samples.push_back(std::move(sample));
    return value;
}
int run(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    std::filesystem::remove_all(args.scratch);
    const auto compatibility = diamond_pipeline::Compatibility::soo("benchmark-1", {.residual_blocks = 1, .width = 8});
    diamond_pipeline::ReplayStore store(args.scratch, compatibility, args.repetitions + 1, 17);
    auto operation = [&](std::uint64_t id) { auto item = episode(compatibility, id); (void)store.ingest(std::span<const diamond_pipeline::Episode>(&item, 1)); (void)store.sample(1); };
    operation(0);
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < args.repetitions; ++i) operation(i + 1);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"name\":\"replay\",\"repetitions\":" << args.repetitions
              << ",\"elapsed_seconds\":" << elapsed
              << ",\"seconds_per_repetition\":" << elapsed / args.repetitions << "}\n";
    return 0;
}
} // namespace
int main(int argc, char** argv) { try { return run(argc, argv); } catch (const std::exception& error) { std::cerr << "replay_benchmark: " << error.what() << '\n'; return 1; } }
