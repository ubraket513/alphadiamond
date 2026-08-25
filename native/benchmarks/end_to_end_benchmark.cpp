#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <torch/torch.h>

#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/model_pool.hpp"
#include "diamond_pipeline/pipeline.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/action.hpp"
#include "soo/board.hpp"
#include "soo/rules.hpp"

namespace {

struct Options {
    std::size_t repetitions = 1;
    std::optional<std::filesystem::path> scratch;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") {
            std::cout << "usage: end_to_end_benchmark [--repetitions N] [--scratch PATH]\n";
            std::exit(0);
        }
        if (argument == "--repetitions") {
            if (++index == argc) throw std::invalid_argument("--repetitions requires a value");
            const std::string_view value = argv[index];
            unsigned long long parsed = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (error != std::errc{} || end != value.data() + value.size() || parsed == 0) {
                throw std::invalid_argument("--repetitions must be a positive integer");
            }
            options.repetitions = static_cast<std::size_t>(parsed);
            continue;
        }
        if (argument == "--scratch") {
            if (++index == argc) throw std::invalid_argument("--scratch requires a path");
            options.scratch = std::filesystem::path(argv[index]);
            continue;
        }
        throw std::invalid_argument("unknown argument: " + std::string(argument));
    }
    return options;
}

soo::Match make_match() {
    soo::Match match;
    match.count = 2;
    match.players[0] = {1, 0, 3};
    match.players[1] = {2, 3, 0};
    return match;
}

// Build a legal position with exactly one action.  Its sole move fills player
// one's target camp, so every iteration emits one real training sample without
// relying on a long, timing-sensitive self-play game.
soo::State one_move_finish(const soo::Match& match) {
    const auto& target = soo::topology().camp_positions[match.players[0].target_camp];
    for (const uint8_t hole : target) {
        for (int direction = 0; direction < soo::kDirections; ++direction) {
            const int8_t source = soo::topology().neighbour[hole][direction];
            if (source < 0) continue;

            bool source_is_target = false;
            for (const uint8_t position : target) source_is_target |= position == source;
            if (source_is_target) continue;

            soo::State state;
            state.occupancy.fill(match.players[1].id);
            for (const uint8_t position : target) {
                if (position != hole) state.occupancy[position] = match.players[0].id;
            }
            state.occupancy[static_cast<uint8_t>(source)] = match.players[0].id;
            state.occupancy[hole] = soo::kEmpty;
            state.current_player = match.players[0].id;

            std::vector<int32_t> actions;
            soo::legal_action_ids(state, actions);
            const int32_t expected = soo::encode_action(source, hole);
            if (actions.size() != 1 || actions.front() != expected) continue;

            const soo::State finished = soo::apply_action(state, match, expected);
            if (finished.status == soo::kFinished && finished.finished_count == match.count) {
                return state;
            }
        }
    }
    throw std::runtime_error("could not construct a one-move terminal Soo position");
}

diamond_pipeline::IterationResult run_once(const std::filesystem::path& scratch,
                                           std::size_t iteration) {
    const auto compatibility = diamond_training::Compatibility::soo(
        "benchmark-v1", {.residual_blocks = 1, .width = 8});
    const diamond_pipeline::ModelKey key{"Soo", "benchmark-v1", std::string(64, 'b')};
    const soo::Match match = make_match();

    torch::manual_seed(123456);
    auto model = diamond_model::DiamondModel(8, 1, 4, 1);
    diamond_training::Trainer trainer(model, compatibility,
                                      {.learning_rate = 1e-3, .weight_decay = 1e-4});
    diamond_pipeline::ModelPool models(1);
    models.install(key, model);
    models.activate(key);

    const auto replay_root = scratch / ("iteration-" + std::to_string(iteration)) / "replay";
    diamond_pipeline::IterationResult result;
    {
        diamond_pipeline::ReplayStore replay(replay_root, compatibility, 8, 7);
        diamond_pipeline::IterationRequest request;
        request.operation_id = "benchmark-" + std::to_string(iteration);
        request.model_key = key;
        request.compatibility = compatibility;
        request.match = match;
        request.jobs = {{one_move_finish(match), 41}};
        request.selfplay = {.lanes = 1,
                            .threads = 1,
                            .max_batch = 1,
                            .max_wait_us = 200,
                            .simulations = 1,
                            .max_moves = 1,
                            .temperature = 0.0,
                            .temperature_moves = 0,
                            .dirichlet_alpha = 0.3,
                            .dirichlet_epsilon = 0.0};
        request.training_steps = 1;
        result = diamond_pipeline::run_iteration(request, models, replay, trainer, {});
    }

    // run_iteration has no operation-resume API. Reopening the persisted
    // replay store is the closest existing native resume behavior and catches
    // persistence failures without inventing a production entry point.
    [[maybe_unused]] diamond_pipeline::ReplayStore reopened(replay_root, compatibility, 8, 7);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        soo::ensure_topology_configured();
        const std::filesystem::path scratch = options.scratch.value_or(
            std::filesystem::temp_directory_path() / "alphadiamond-native-benchmarks");
        std::filesystem::create_directories(scratch);

        (void)run_once(scratch, 0);  // warm-up
        std::size_t completed = 0;
        std::size_t aborted = 0;
        std::size_t samples = 0;
        uint64_t training_steps = 0;
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
            const auto result = run_once(scratch, repetition + 1);
            completed += result.completed_games;
            aborted += result.aborted_games;
            samples += result.new_samples;
            training_steps += result.training_step;
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

        std::cout << "{\"benchmark\":\"end_to_end\",\"repetitions\":" << options.repetitions
                  << ",\"warmup_runs\":1,\"completed_games\":" << completed
                  << ",\"aborted_games\":" << aborted << ",\"new_samples\":" << samples
                  << ",\"training_steps\":" << training_steps << ",\"elapsed_ms\":" << elapsed_ms
                  << ",\"resume\":\"replay_store_reopen\",\"missing_resume_api\":\"diamond_pipeline::run_iteration has no operation-resume API\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "end_to_end_benchmark: " << error.what() << '\n';
        return 2;
    }
}
