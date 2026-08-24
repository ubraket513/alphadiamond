#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "soo/board.hpp"
#include "soo/selfplay.hpp"

namespace {

struct Options {
    std::size_t repetitions = 1;
    std::optional<std::string> scratch;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") {
            std::cout << "usage: selfplay_benchmark [--repetitions N] [--scratch PATH]\n";
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
            options.scratch = argv[index];
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

soo::State opening(const soo::Match& match) {
    soo::State state;
    for (std::size_t seat = 0; seat < match.count; ++seat) {
        for (const uint8_t position : soo::topology().camp_positions[match.players[seat].camp]) {
            state.occupancy[position] = match.players[seat].id;
        }
    }
    state.current_player = match.players[0].id;
    return state;
}

struct Totals {
    std::size_t completed = 0;
    std::size_t aborted = 0;
    uint64_t moves = 0;
};

Totals run_once(const soo::Match& match, const soo::State& initial) {
    const soo::EpisodeConfig config{
        .lanes = 2,
        .threads = 2,
        .max_batch = 2,
        .max_wait_us = 200,
        .simulations = 4,
        .max_moves = 8,
        .temperature = 0.0,
        .temperature_moves = 0,
        .dirichlet_alpha = 0.3,
        .dirichlet_epsilon = 0.0,
    };
    const std::vector<soo::EpisodeJob> jobs{{initial, 17}, {initial, 29}};
    soo::DummyBatchEvaluator evaluator(0.0);
    soo::EpisodeMetrics metrics;
    const auto episodes = soo::run_episodes(match, jobs, config, evaluator, metrics);

    Totals totals;
    for (const soo::Episode& episode : episodes) {
        totals.completed += episode.completed ? 1 : 0;
        totals.aborted += episode.completed ? 0 : 1;
        totals.moves += static_cast<uint64_t>(episode.move_count);
    }
    return totals;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        soo::ensure_topology_configured();
        const soo::Match match = make_match();
        const soo::State initial = opening(match);

        (void)run_once(match, initial);  // warm-up
        Totals totals;
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
            const Totals current = run_once(match, initial);
            totals.completed += current.completed;
            totals.aborted += current.aborted;
            totals.moves += current.moves;
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

        std::cout << "{\"benchmark\":\"selfplay\",\"repetitions\":" << options.repetitions
                  << ",\"warmup_runs\":1,\"completed_episodes\":" << totals.completed
                  << ",\"aborted_episodes\":" << totals.aborted
                  << ",\"moves\":" << totals.moves << ",\"elapsed_ms\":" << elapsed_ms << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "selfplay_benchmark: " << error.what() << '\n';
        return 2;
    }
}
