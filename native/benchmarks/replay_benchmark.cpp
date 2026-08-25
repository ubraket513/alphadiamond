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
struct Args {
    std::uint64_t repetitions = 1;
    std::uint64_t pool_size = 128;
    std::uint64_t batch_size = 4;
    std::filesystem::path scratch;
};
Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if ((option != "--repetitions" && option != "--pool-size" && option != "--batch-size" && option != "--scratch") || i + 1 >= argc)
            throw std::invalid_argument("usage: replay_benchmark --scratch PATH [--repetitions N] [--pool-size N] [--batch-size N]");
        const std::string value = argv[++i];
        if (option == "--scratch") args.scratch = value;
        else {
            try {
                const auto parsed = std::stoull(value);
                if (option == "--repetitions") args.repetitions = parsed;
                else if (option == "--pool-size") args.pool_size = parsed;
                else args.batch_size = parsed;
            } catch (...) { throw std::invalid_argument("benchmark counts must be positive integers"); }
        }
    }
    if (args.scratch.empty() || args.repetitions == 0 || args.pool_size == 0 || args.batch_size == 0 || args.batch_size > args.pool_size)
        throw std::invalid_argument("scratch and positive pool/repetition counts with batch <= pool are required");
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
    diamond_pipeline::ReplayStore store(args.scratch, compatibility, args.pool_size, 17);
    std::vector<diamond_pipeline::Episode> pool;
    pool.reserve(args.pool_size);
    for (std::uint64_t id = 0; id < args.pool_size; ++id) pool.push_back(episode(compatibility, id));
    (void)store.ingest(pool);
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < args.repetitions; ++i) (void)store.sample(args.batch_size);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto stats = store.last_sampling_stats();
    std::cout << "{\"name\":\"replay\",\"batch_size\":" << args.batch_size
              << ",\"pool_size\":" << args.pool_size
              << ",\"repetitions\":" << args.repetitions
              << ",\"selection_slots\":" << stats.selection_slots
              << ",\"elapsed_seconds\":" << elapsed
              << ",\"seconds_per_repetition\":" << elapsed / args.repetitions << "}\n";
    return 0;
}
} // namespace
int main(int argc, char** argv) { try { return run(argc, argv); } catch (const std::exception& error) { std::cerr << "replay_benchmark: " << error.what() << '\n'; return 1; } }
