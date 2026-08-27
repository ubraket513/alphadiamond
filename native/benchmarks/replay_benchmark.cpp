#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <utility>

#include "diamond_pipeline/replay_store.hpp"
#include "diamond_support/build_provenance.hpp"

namespace {
struct Args {
    std::uint64_t repetitions = 1;
    std::uint64_t pool_size = 128;
    std::uint64_t batch_size = 4;
    std::filesystem::path scratch;
    bool reopen_only = false;
};
Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--reopen-only") {
            args.reopen_only = true;
            continue;
        }
        if ((option != "--repetitions" && option != "--pool-size" && option != "--batch-size" &&
             option != "--scratch") ||
            i + 1 >= argc)
            throw std::invalid_argument("usage: replay_benchmark --scratch PATH [--repetitions N] "
                                        "[--pool-size N] [--batch-size N]");
        const std::string value = argv[++i];
        if (option == "--scratch") args.scratch = value;
        else {
            try {
                const auto parsed = std::stoull(value);
                if (option == "--repetitions")
                    args.repetitions = parsed;
                else if (option == "--pool-size")
                    args.pool_size = parsed;
                else
                    args.batch_size = parsed;
            } catch (...) {
                throw std::invalid_argument("benchmark counts must be positive integers");
            }
        }
    }
    if (args.reopen_only)
        return args.scratch.empty() ? throw std::invalid_argument("--scratch is required") : args;
    if (args.scratch.empty() || args.repetitions == 0 || args.pool_size == 0 ||
        args.batch_size == 0 || args.batch_size > args.pool_size)
        throw std::invalid_argument(
            "scratch and positive pool/repetition counts with batch <= pool are required");
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
// Time opening an existing store, full versus metadata-only.  This is the cost
// a training iteration pays three times over, and at a 1M capacity it dominated
// everything else the iteration did.
int reopen(const std::filesystem::path& root, const diamond_pipeline::Compatibility& compatibility,
           std::uint64_t capacity) {
    const auto measure = [&](diamond_pipeline::ReplayContents contents) {
        const auto start = std::chrono::steady_clock::now();
        diamond_pipeline::ReplayStore store(root, compatibility, capacity, 17, contents);
        const auto seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return std::pair{seconds, store.size()};
    };
    const auto [meta_s, meta_n] = measure(diamond_pipeline::ReplayContents::metadata_only);
    const auto [full_s, full_n] = measure(diamond_pipeline::ReplayContents::full);
    std::cout << "{\"schema_version\":1,\"benchmark\":\"replay-reopen\",\"metadata_only_seconds\":"
              << meta_s << ",\"full_seconds\":" << full_s << ",\"metadata_only_size\":" << meta_n
              << ",\"full_size\":" << full_n
              << ",\"speedup\":" << (meta_s > 0 ? full_s / meta_s : 0) << "}\n";
    return 0;
}

int run(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    if (args.reopen_only) {
        return reopen(
            args.scratch,
            diamond_pipeline::Compatibility::soo("2.0.0", {.residual_blocks = 6, .width = 128}),
            args.pool_size);
    }
    std::filesystem::remove_all(args.scratch);
    const auto compatibility = diamond_pipeline::Compatibility::soo("benchmark-1", {.residual_blocks = 1, .width = 8});
    diamond_pipeline::ReplayStore store(args.scratch, compatibility, args.pool_size, 17);
    std::vector<diamond_pipeline::Episode> pool;
    pool.reserve(args.pool_size);
    for (std::uint64_t id = 0; id < args.pool_size; ++id)
        pool.push_back(episode(compatibility, id));
    (void)store.ingest(pool);
    std::vector<double> seconds;
    for (std::uint64_t i = 0; i < args.repetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        (void)store.sample(args.batch_size,
                           diamond_pipeline::replay_sampling_seed(17, 0, i));
        seconds.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    auto sorted = seconds;
    std::sort(sorted.begin(), sorted.end());
    const auto stats = store.last_sampling_stats();
    std::cout << "{\"schema_version\":1,\"benchmark\":\"replay\",\"workload\":{\"repetitions\":"
              << args.repetitions << ",\"warmups\":0,\"batch_size\":" << args.batch_size
              << ",\"pool_size\":" << args.pool_size
              << "},\"environment\":{\"requested_device\":\"cpu\",\"canonical_device\":\"cpu\","
                 "\"precision\":\"float32\"},\"provenance\":"
              << diamond_support::build_provenance_json() << ",\"samples_seconds\":[";
    for (std::size_t i = 0; i < seconds.size(); ++i)
        std::cout << (i ? "," : "") << seconds[i];
    std::cout << "],\"summary_seconds\":{\"min\":" << sorted.front()
              << ",\"median\":" << sorted[sorted.size() / 2] << ",\"max\":" << sorted.back()
              << ",\"range\":" << sorted.back() - sorted.front()
              << "},\"domain\":{\"selection_slots\":" << stats.selection_slots << "}}\n";
    return 0;
}
} // namespace
int main(int argc, char** argv) { try { return run(argc, argv); } catch (const std::exception& error) { std::cerr << "replay_benchmark: " << error.what() << '\n'; return 1; } }
