#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <torch/torch.h>

#include "diamond_model/deployment_artifact.hpp"
#include "diamond_pipeline/model_pool.hpp"
#include "diamond_support/build_provenance.hpp"
#include "diamond_training/device.hpp"
#include "soo/board.hpp"
#include "soo/selfplay.hpp"

namespace {

struct Options {
    std::filesystem::path artifact = "models/soo/2.0.0";
    std::string device = "cpu";
    std::size_t repetitions = 1;
    std::size_t warmups = 1;
    std::size_t lanes = 2;
    std::size_t threads = 1;
    std::size_t max_batch = 2;
    std::size_t simulations = 4;
    std::size_t max_moves = 8;
    uint64_t max_wait_us = 200;
    std::optional<std::filesystem::path> scratch;
};

std::size_t parse_count(std::string_view value, std::string_view option, bool allow_zero = false) {
    unsigned long long parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size() ||
        (!allow_zero && parsed == 0) || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(option) + (allow_zero
                                                               ? " must be a non-negative integer"
                                                               : " must be a positive integer"));
    }
    return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--help") {
            std::cout << "usage: selfplay_benchmark [--artifact DIR] [--device cpu|cuda|cuda:N] "
                         "[--lanes N] [--threads N] [--max-batch N] [--max-wait-us N] "
                         "[--simulations N] [--max-moves N] [--warmups N] [--repetitions N] "
                         "[--scratch PATH]\n";
            std::exit(0);
        }
        if (++index == argc)
            throw std::invalid_argument(std::string(option) + " requires a value");
        const std::string_view value = argv[index];
        if (option == "--artifact")
            options.artifact = value;
        else if (option == "--device")
            options.device = value;
        else if (option == "--lanes")
            options.lanes = parse_count(value, option);
        else if (option == "--threads")
            options.threads = parse_count(value, option);
        else if (option == "--max-batch")
            options.max_batch = parse_count(value, option);
        else if (option == "--max-wait-us")
            options.max_wait_us = parse_count(value, option, true);
        else if (option == "--simulations")
            options.simulations = parse_count(value, option);
        else if (option == "--max-moves")
            options.max_moves = parse_count(value, option);
        else if (option == "--warmups")
            options.warmups = parse_count(value, option, true);
        else if (option == "--repetitions")
            options.repetitions = parse_count(value, option);
        else if (option == "--scratch")
            options.scratch = value;
        else
            throw std::invalid_argument("unknown argument: " + std::string(option));
    }
    if (options.threads > options.lanes)
        throw std::invalid_argument("threads must not exceed lanes");
    const auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (options.lanes > int_max || options.threads > int_max || options.max_batch > int_max ||
        options.max_wait_us > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        options.simulations > int_max || options.max_moves > int_max) {
        throw std::invalid_argument("self-play workload counts must fit in a signed integer");
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
    std::size_t attempted = 0;
    std::size_t completed = 0;
    std::size_t aborted = 0;
    std::size_t completed_samples = 0;
    std::size_t max_move_aborts = 0;
    std::size_t deadline_aborts = 0;
    std::size_t other_aborts = 0;
    uint64_t evaluations = 0;
    uint64_t batches = 0;
    uint64_t moves = 0;
    double wall_seconds = 0.0;
    double evaluator_seconds = 0.0;
    double worker_busy_seconds = 0.0;
    std::vector<uint32_t> batch_sizes;
    std::vector<uint32_t> move_counts;
};

Totals run_once(const Options& options, const soo::Match& match, const soo::State& initial,
                diamond_pipeline::ModelPool& evaluator) {
    const soo::EpisodeConfig config{
        .lanes = static_cast<int>(options.lanes),
        .threads = static_cast<int>(options.threads),
        .max_batch = static_cast<int>(options.max_batch),
        .max_wait_us = static_cast<int>(options.max_wait_us),
        .simulations = static_cast<int>(options.simulations),
        .max_moves = static_cast<int>(options.max_moves),
        .temperature = 0.0,
        .temperature_moves = 0,
        .dirichlet_alpha = 0.3,
        .dirichlet_epsilon = 0.0,
    };
    std::vector<soo::EpisodeJob> jobs;
    jobs.reserve(options.lanes);
    for (std::size_t lane = 0; lane < options.lanes; ++lane)
        jobs.push_back({initial, static_cast<uint64_t>(17 + lane * 12)});

    soo::EpisodeMetrics metrics;
    const auto episodes = soo::run_episodes(match, jobs, config, evaluator, metrics);

    Totals totals;
    totals.attempted = episodes.size();
    totals.evaluations = metrics.evaluations;
    totals.batches = metrics.batches;
    totals.moves = metrics.moves;
    totals.wall_seconds = metrics.wall_seconds;
    totals.evaluator_seconds = metrics.evaluator_seconds;
    totals.worker_busy_seconds = metrics.worker_busy_seconds;
    totals.batch_sizes = std::move(metrics.batch_sizes);
    for (const soo::Episode& episode : episodes) {
        totals.completed += episode.completed ? 1 : 0;
        totals.aborted += episode.completed ? 0 : 1;
        totals.completed_samples += episode.completed ? episode.moves.size() : 0;
        totals.move_counts.push_back(static_cast<uint32_t>(episode.move_count));
        if (!episode.completed) {
            if (episode.move_limit_exceeded)
                ++totals.max_move_aborts;
            else if (episode.max_game_seconds_exceeded)
                ++totals.deadline_aborts;
            else
                ++totals.other_aborts;
        }
    }
    return totals;
}

void accumulate(Totals& destination, Totals source) {
    destination.attempted += source.attempted;
    destination.completed += source.completed;
    destination.aborted += source.aborted;
    destination.completed_samples += source.completed_samples;
    destination.max_move_aborts += source.max_move_aborts;
    destination.deadline_aborts += source.deadline_aborts;
    destination.other_aborts += source.other_aborts;
    destination.evaluations += source.evaluations;
    destination.batches += source.batches;
    destination.moves += source.moves;
    destination.wall_seconds += source.wall_seconds;
    destination.evaluator_seconds += source.evaluator_seconds;
    destination.worker_busy_seconds += source.worker_busy_seconds;
    destination.batch_sizes.insert(destination.batch_sizes.end(), source.batch_sizes.begin(),
                                   source.batch_sizes.end());
    destination.move_counts.insert(destination.move_counts.end(), source.move_counts.begin(),
                                   source.move_counts.end());
}

uint32_t percentile(std::vector<uint32_t> values, double fraction) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

double ratio(double numerator, double denominator) {
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        soo::ensure_topology_configured();
        torch::set_num_threads(static_cast<int>(options.threads));
        torch::set_num_interop_threads(1);

        const auto artifact = diamond_model::validate_deployment_artifact(options.artifact, "soo");
        const auto compatibility = diamond_training::Compatibility::soo(
            artifact.model_version,
            {.residual_blocks = artifact.residual_blocks, .width = artifact.width});
        auto model = diamond_model::DiamondModel(artifact.width, artifact.residual_blocks,
                                                 artifact.input_features, artifact.value_size);
        model->load_weights(artifact.weights);
        const auto device = diamond_training::resolve_device(options.device);
        diamond_pipeline::ModelPool evaluator(1, device);
        const auto model_key = evaluator.install(compatibility, model);
        evaluator.activate(model_key);

        const soo::Match match = make_match();
        const soo::State initial = opening(match);
        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
            (void)run_once(options, match, initial, evaluator);

        Totals totals;
        std::vector<double> seconds;
        seconds.reserve(options.repetitions);
        for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
            const auto started = std::chrono::steady_clock::now();
            Totals current = run_once(options, match, initial, evaluator);
            seconds.push_back(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
            accumulate(totals, std::move(current));
        }

        auto sorted_seconds = seconds;
        std::sort(sorted_seconds.begin(), sorted_seconds.end());
        const auto max_batch =
            totals.batch_sizes.empty()
                ? 0U
                : *std::max_element(totals.batch_sizes.begin(), totals.batch_sizes.end());
        uint64_t total_batch_rows = 0;
        for (const uint32_t size : totals.batch_sizes)
            total_batch_rows += size;
        double measured_seconds = 0.0;
        for (const double value : seconds)
            measured_seconds += value;

        std::cout
            << "{\"schema_version\":1,\"benchmark\":\"selfplay\",\"workload\":{\"repetitions\":"
            << options.repetitions << ",\"warmups\":" << options.warmups
            << ",\"lanes\":" << options.lanes << ",\"threads\":" << options.threads
            << ",\"max_batch\":" << options.max_batch << ",\"max_wait_us\":" << options.max_wait_us
            << ",\"simulations\":" << options.simulations << ",\"max_moves\":" << options.max_moves
            << "},\"environment\":{\"requested_device\":\"" << device.requested_name
            << "\",\"canonical_device\":\"" << device.canonical_name
            << "\",\"precision\":\"float32\",\"torch_threads\":" << torch::get_num_threads()
            << "},\"provenance\":" << diamond_support::build_provenance_json()
            << ",\"samples_seconds\":[";
        for (std::size_t index = 0; index < seconds.size(); ++index)
            std::cout << (index ? "," : "") << seconds[index];
        std::cout << "],\"summary_seconds\":{\"min\":" << sorted_seconds.front()
                  << ",\"median\":" << sorted_seconds[sorted_seconds.size() / 2]
                  << ",\"max\":" << sorted_seconds.back()
                  << ",\"range\":" << sorted_seconds.back() - sorted_seconds.front()
                  << "},\"domain\":{\"model_sha256\":\"" << artifact.model_sha256
                  << "\",\"runtime_sha256\":\"" << artifact.runtime_sha256
                  << "\",\"attempted_episodes\":" << totals.attempted
                  << ",\"completed_episodes\":" << totals.completed
                  << ",\"aborted_episodes\":" << totals.aborted
                  << ",\"abort_reasons\":{\"max_moves\":" << totals.max_move_aborts
                  << ",\"max_game_seconds\":" << totals.deadline_aborts
                  << ",\"other\":" << totals.other_aborts
                  << "},\"completed_samples\":" << totals.completed_samples
                  << ",\"moves\":" << totals.moves
                  << ",\"moves_p50\":" << percentile(totals.move_counts, 0.50)
                  << ",\"moves_p90\":" << percentile(totals.move_counts, 0.90)
                  << ",\"moves_p99\":" << percentile(totals.move_counts, 0.99)
                  << ",\"evaluations\":" << totals.evaluations << ",\"batches\":" << totals.batches
                  << ",\"evaluations_per_second\":"
                  << ratio(static_cast<double>(totals.evaluations), totals.wall_seconds)
                  << ",\"batches_per_second\":"
                  << ratio(static_cast<double>(totals.batches), totals.wall_seconds)
                  << ",\"batch_mean\":"
                  << ratio(static_cast<double>(total_batch_rows),
                           static_cast<double>(totals.batch_sizes.size()))
                  << ",\"batch_p50\":" << percentile(totals.batch_sizes, 0.50)
                  << ",\"batch_p90\":" << percentile(totals.batch_sizes, 0.90)
                  << ",\"batch_max\":" << max_batch << ",\"evaluator_busy_fraction\":"
                  << ratio(totals.evaluator_seconds, totals.wall_seconds)
                  << ",\"search_worker_busy_fraction\":"
                  << ratio(totals.worker_busy_seconds,
                           totals.wall_seconds * static_cast<double>(options.threads))
                  << ",\"samples_per_hour\":"
                  << ratio(static_cast<double>(totals.completed_samples) * 3600.0, measured_seconds)
                  << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "selfplay_benchmark: " << error.what() << '\n';
        return 2;
    }
}
