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

#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/model_pool.hpp"
#include "diamond_pipeline/pipeline.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_support/build_provenance.hpp"
#include "diamond_training/checkpoint.hpp"
#include "diamond_training/device.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/action.hpp"
#include "soo/board.hpp"
#include "soo/rules.hpp"

namespace {

struct Options {
    std::string device = "cpu";
    std::size_t repetitions = 1;
    std::size_t warmups = 1;
    std::size_t games = 4;
    std::size_t lanes = 2;
    std::size_t threads = 1;
    std::size_t batch_size = 2;
    std::size_t training_steps = 2;
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
            std::cout << "usage: end_to_end_benchmark [--device cpu|cuda|cuda:N] [--games N] "
                         "[--lanes N] [--threads N] [--batch-size N] [--training-steps N] "
                         "[--warmups N] [--repetitions N] [--scratch PATH]\n";
            std::exit(0);
        }
        if (++index == argc)
            throw std::invalid_argument(std::string(option) + " requires a value");
        const std::string_view value = argv[index];
        if (option == "--device")
            options.device = value;
        else if (option == "--games")
            options.games = parse_count(value, option);
        else if (option == "--lanes")
            options.lanes = parse_count(value, option);
        else if (option == "--threads")
            options.threads = parse_count(value, option);
        else if (option == "--batch-size")
            options.batch_size = parse_count(value, option);
        else if (option == "--training-steps")
            options.training_steps = parse_count(value, option);
        else if (option == "--warmups")
            options.warmups = parse_count(value, option, true);
        else if (option == "--repetitions")
            options.repetitions = parse_count(value, option);
        else if (option == "--scratch")
            options.scratch = std::filesystem::path{std::string(value)};
        else
            throw std::invalid_argument("unknown argument: " + std::string(option));
    }
    if (!(options.games > options.lanes && options.lanes > options.threads))
        throw std::invalid_argument("end-to-end workload requires games > lanes > threads");
    if (options.batch_size > options.games)
        throw std::invalid_argument("batch size must not exceed games per iteration");
    const auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (options.games > int_max || options.lanes > int_max || options.threads > int_max ||
        options.batch_size > int_max || options.training_steps > int_max) {
        throw std::invalid_argument("end-to-end workload counts must fit in a signed integer");
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

// Build a deterministic one-action terminal fixture. Player two is already on
// the podium, so player one's sole move closes the match and emits one real
// training sample without relying on a long, timing-sensitive self-play game.
soo::State one_move_finish(const soo::Match& match) {
    for (int source = 0; source < soo::kBoardSize; ++source) {
        for (int direction = 0; direction < soo::kDirections; ++direction) {
            const int8_t hole = soo::topology().neighbour[source][direction];
            if (hole < 0)
                continue;

            soo::State state;
            state.occupancy.fill(match.players[1].id);
            state.occupancy[static_cast<uint8_t>(source)] = match.players[0].id;
            state.occupancy[static_cast<uint8_t>(hole)] = soo::kEmpty;
            state.current_player = match.players[0].id;
            state.finished_count = 1;
            state.finish_order[0] = match.players[1].id;

            std::vector<int32_t> actions;
            soo::legal_action_ids(state, actions);
            const int32_t expected = soo::encode_action(source, hole);
            if (actions.size() != 1 || actions.front() != expected) continue;

            const soo::State finished = soo::apply_action(state, match, expected);
            if (finished.status == soo::kFinished && finished.finished_count == match.count)
                return state;
        }
    }
    throw std::runtime_error("could not construct a one-move terminal Soo position");
}

diamond_pipeline::IterationRequest
make_request(const Options& options, std::string operation_id, std::size_t iteration,
             const diamond_pipeline::ModelKey& model_key,
             const diamond_training::Compatibility& compatibility, const soo::Match& match,
             const soo::State& opening) {
    diamond_pipeline::IterationRequest request;
    request.operation_id = std::move(operation_id);
    request.model_key = model_key;
    request.compatibility = compatibility;
    request.match = match;
    request.jobs.reserve(options.games);
    for (std::size_t game = 0; game < options.games; ++game) {
        request.jobs.push_back({opening, static_cast<uint64_t>(41 + iteration * 1000 + game)});
    }
    request.selfplay = {.lanes = static_cast<int>(options.lanes),
                        .threads = static_cast<int>(options.threads),
                        .max_batch = static_cast<int>(options.lanes),
                        .max_wait_us = 200,
                        .simulations = 1,
                        .max_moves = 1,
                        .temperature = 0.0,
                        .temperature_moves = 0,
                        .dirichlet_alpha = 0.3,
                        .dirichlet_epsilon = 0.0};
    request.training_batch_size = options.batch_size;
    request.training_steps = options.training_steps;
    return request;
}

void require_iteration(const diamond_pipeline::IterationResult& result, const Options& options,
                       uint64_t expected_step) {
    if (result.completed_games != options.games || result.aborted_games != 0 ||
        result.new_samples != options.games ||
        result.requested_training_steps != options.training_steps ||
        result.completed_training_steps != options.training_steps ||
        result.training_step != expected_step ||
        result.training_batch_sizes.size() != options.training_steps ||
        !std::all_of(result.training_batch_sizes.begin(), result.training_batch_sizes.end(),
                     [&](std::size_t size) { return size == options.batch_size; })) {
        throw std::runtime_error("end-to-end iteration accounting mismatch");
    }
}

struct RunResult {
    std::size_t completed_games = 0;
    std::size_t aborted_games = 0;
    std::size_t new_samples = 0;
    std::size_t replay_size = 0;
    std::size_t requested_training_steps = 0;
    std::size_t completed_training_steps = 0;
    uint64_t resume_step = 0;
    uint64_t final_step = 0;
    std::string first_checkpoint_model_digest;
    std::string final_checkpoint_model_digest;
    std::string replay_manifest_digest;
    std::vector<std::size_t> training_batch_sizes;
    std::vector<diamond_training::TrainingMetrics> training_metrics;
};

std::string source_git_commit() {
#ifdef DIAMOND_SOURCE_GIT_COMMIT
    return DIAMOND_SOURCE_GIT_COMMIT;
#else
    return "unavailable";
#endif
}

diamond_training::CheckpointProvenance
checkpoint_provenance(const Options& options, const diamond_training::ResolvedDevice& device,
                      std::string replay_manifest_digest) {
    return {.source_git_commit = source_git_commit(),
            .resolved_config_bytes =
                "{\"batch_size\":" + std::to_string(options.batch_size) +
                ",\"benchmark\":\"end_to_end\",\"games\":" + std::to_string(options.games) +
                ",\"lanes\":" + std::to_string(options.lanes) +
                ",\"training_steps\":" + std::to_string(options.training_steps) + "}",
            .replay_manifest_sha256 = std::move(replay_manifest_digest),
            .protocol_ids_json =
                "{\"benchmark\":\"end-to-end-v2\",\"checkpoint\":\"native-checkpoint-v3\"}",
            .creation_timestamp = "benchmark-run",
            .environment_json = "{\"canonical_device\":\"" + device.canonical_name +
                                "\",\"torch_threads\":" + std::to_string(torch::get_num_threads()) +
                                "}"};
}

RunResult run_once(const Options& options, const std::filesystem::path& root,
                   const diamond_training::ResolvedDevice& device) {
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    if (cleanup_error)
        throw std::runtime_error("cannot reset end-to-end scratch: " + cleanup_error.message());
    std::filesystem::create_directories(root);

    const auto compatibility = diamond_training::Compatibility::soo(
        "benchmark-v1", {.residual_blocks = 1, .width = 8});
    const soo::Match match = make_match();
    const soo::State opening = one_move_finish(match);
    const auto replay_root = root / "replay";
    const auto first_checkpoint = root / "checkpoint-iteration-0";
    const auto final_checkpoint = root / "checkpoint-iteration-1";
    const std::size_t replay_capacity = options.games * 2;

    diamond_pipeline::IterationResult first;
    diamond_training::CheckpointInfo first_info;
    torch::manual_seed(123456);
    {
        auto model = diamond_model::DiamondModel(8, 1, 4, 1);
        diamond_training::Trainer trainer(model, compatibility,
                                          {.learning_rate = 1e-3, .weight_decay = 1e-4}, device);
        diamond_pipeline::ModelPool models(1, device);
        const auto key = models.install(compatibility, model);
        models.activate(key);
        diamond_pipeline::ReplayStore replay(replay_root, compatibility, replay_capacity, 7);
        const auto request =
            make_request(options, "benchmark-iteration-0", 0, key, compatibility, match, opening);
        first = diamond_pipeline::run_iteration(request, models, replay, trainer, {});
        first_info = diamond_training::save_checkpoint_v3(
            first_checkpoint, trainer,
            {.initialization_mode = diamond_training::CheckpointInitializationMode::scratch,
             .run_id = "end-to-end-benchmark",
             .iteration = 0,
             .model_step = trainer.training_step()},
            checkpoint_provenance(options, device, replay.manifest_digest()));
    }
    require_iteration(first, options, options.training_steps);
    first_info = diamond_training::validate_checkpoint_v2(first_checkpoint);
    if (first_info.training_step != options.training_steps)
        throw std::runtime_error("first checkpoint step mismatch");

    diamond_pipeline::IterationResult second;
    diamond_training::CheckpointInfo final_info;
    uint64_t resume_step = 0;
    {
        auto model = diamond_model::DiamondModel(8, 1, 4, 1);
        diamond_training::Trainer trainer(model, compatibility,
                                          {.learning_rate = 1e-3, .weight_decay = 1e-4}, device);
        const auto resumed = diamond_training::load_checkpoint_v3(
            first_checkpoint, trainer, device,
            diamond_training::CheckpointLoadIntent::exact_resume);
        resume_step = resumed.training_step;
        diamond_pipeline::ModelPool models(1, device);
        const auto key = models.install(compatibility, trainer.model());
        models.activate(key);
        diamond_pipeline::ReplayStore replay(replay_root, compatibility, replay_capacity, 7);
        const auto request =
            make_request(options, "benchmark-iteration-1", 1, key, compatibility, match, opening);
        second = diamond_pipeline::run_iteration(request, models, replay, trainer, {});
        final_info = diamond_training::save_checkpoint_v3(
            final_checkpoint, trainer,
            {.initialization_mode = diamond_training::CheckpointInitializationMode::resume,
             .run_id = "end-to-end-benchmark",
             .iteration = 1,
             .model_step = trainer.training_step(),
             .parent_digest = first_info.model_digest,
             .source_digest = first_info.model_digest,
             .source_training_step = resume_step,
             .optimizer_restored = true},
            checkpoint_provenance(options, device, replay.manifest_digest()));
    }
    require_iteration(second, options, options.training_steps * 2);
    final_info = diamond_training::validate_checkpoint_v2(final_checkpoint);
    if (resume_step != options.training_steps ||
        final_info.training_step != options.training_steps * 2)
        throw std::runtime_error("checkpoint resume step mismatch");

    diamond_pipeline::ReplayStore reopened(replay_root, compatibility, replay_capacity, 7);
    if (reopened.size() != options.games * 2)
        throw std::runtime_error("replay exact-work accounting mismatch after resume");

    RunResult result;
    result.completed_games = first.completed_games + second.completed_games;
    result.aborted_games = first.aborted_games + second.aborted_games;
    result.new_samples = first.new_samples + second.new_samples;
    result.replay_size = reopened.size();
    result.requested_training_steps =
        first.requested_training_steps + second.requested_training_steps;
    result.completed_training_steps =
        first.completed_training_steps + second.completed_training_steps;
    result.resume_step = resume_step;
    result.final_step = second.training_step;
    result.first_checkpoint_model_digest = first_info.model_digest;
    result.final_checkpoint_model_digest = final_info.model_digest;
    result.replay_manifest_digest = reopened.manifest_digest();
    result.training_batch_sizes = std::move(first.training_batch_sizes);
    result.training_batch_sizes.insert(result.training_batch_sizes.end(),
                                       second.training_batch_sizes.begin(),
                                       second.training_batch_sizes.end());
    result.training_metrics = std::move(first.training_metrics);
    result.training_metrics.insert(result.training_metrics.end(), second.training_metrics.begin(),
                                   second.training_metrics.end());
    return result;
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
        const auto device = diamond_training::resolve_device(options.device);
        const std::filesystem::path scratch = options.scratch.value_or(
            std::filesystem::temp_directory_path() / "alphadiamond-native-benchmarks");
        std::filesystem::create_directories(scratch);

        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
            (void)run_once(options, scratch / ("warmup-" + std::to_string(warmup)), device);

        RunResult totals;
        std::vector<double> seconds;
        seconds.reserve(options.repetitions);
        for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
            const auto started = std::chrono::steady_clock::now();
            RunResult current =
                run_once(options, scratch / ("repetition-" + std::to_string(repetition)), device);
            seconds.push_back(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
            totals.completed_games += current.completed_games;
            totals.aborted_games += current.aborted_games;
            totals.new_samples += current.new_samples;
            totals.replay_size += current.replay_size;
            totals.requested_training_steps += current.requested_training_steps;
            totals.completed_training_steps += current.completed_training_steps;
            totals.resume_step = current.resume_step;
            totals.final_step = current.final_step;
            totals.first_checkpoint_model_digest = current.first_checkpoint_model_digest;
            totals.final_checkpoint_model_digest = current.final_checkpoint_model_digest;
            totals.replay_manifest_digest = current.replay_manifest_digest;
            totals.training_batch_sizes.insert(totals.training_batch_sizes.end(),
                                               current.training_batch_sizes.begin(),
                                               current.training_batch_sizes.end());
            totals.training_metrics.insert(totals.training_metrics.end(),
                                           current.training_metrics.begin(),
                                           current.training_metrics.end());
        }

        auto sorted = seconds;
        std::sort(sorted.begin(), sorted.end());
        double measured_seconds = 0.0;
        double training_step_seconds = 0.0;
        for (const double value : seconds)
            measured_seconds += value;
        for (const auto& metrics : totals.training_metrics)
            training_step_seconds += metrics.total_step_seconds;

        std::cout
            << "{\"schema_version\":1,\"benchmark\":\"end_to_end\",\"workload\":{\"repetitions\":"
            << options.repetitions << ",\"warmups\":" << options.warmups
            << ",\"iterations_per_repetition\":2,\"games_per_iteration\":" << options.games
            << ",\"lanes\":" << options.lanes << ",\"threads\":" << options.threads
            << ",\"batch_size\":" << options.batch_size
            << ",\"training_steps_per_iteration\":" << options.training_steps
            << ",\"opening\":\"synthetic-one-move-terminal\"},\"environment\":{\"requested_"
               "device\":\""
            << device.requested_name << "\",\"canonical_device\":\"" << device.canonical_name
            << "\",\"precision\":\"float32\",\"torch_threads\":" << torch::get_num_threads()
            << "},\"provenance\":" << diamond_support::build_provenance_json()
            << ",\"samples_seconds\":[";
        for (std::size_t index = 0; index < seconds.size(); ++index)
            std::cout << (index ? "," : "") << seconds[index];
        std::cout << "],\"summary_seconds\":{\"min\":" << sorted.front()
                  << ",\"median\":" << sorted[sorted.size() / 2] << ",\"max\":" << sorted.back()
                  << ",\"range\":" << sorted.back() - sorted.front()
                  << "},\"domain\":{\"completed_games\":" << totals.completed_games
                  << ",\"aborted_games\":" << totals.aborted_games
                  << ",\"new_samples\":" << totals.new_samples
                  << ",\"replay_size_sum\":" << totals.replay_size
                  << ",\"requested_training_steps\":" << totals.requested_training_steps
                  << ",\"completed_training_steps\":" << totals.completed_training_steps
                  << ",\"resume_step\":" << totals.resume_step
                  << ",\"final_step\":" << totals.final_step << ",\"training_batch_sizes\":[";
        for (std::size_t index = 0; index < totals.training_batch_sizes.size(); ++index)
            std::cout << (index ? "," : "") << totals.training_batch_sizes[index];
        std::cout << "],\"training_step_seconds\":" << training_step_seconds
                  << ",\"completed_samples_per_hour\":"
                  << ratio(static_cast<double>(totals.new_samples) * 3600.0, measured_seconds)
                  << ",\"checkpoint_after_first_iteration\":true,"
                     "\"boundary_restart_resume\":true,\"duplicate_durable_work\":false,"
                     "\"first_checkpoint_model_digest\":\""
                  << totals.first_checkpoint_model_digest
                  << "\",\"final_checkpoint_model_digest\":\""
                  << totals.final_checkpoint_model_digest << "\",\"replay_manifest_digest\":\""
                  << totals.replay_manifest_digest << "\"}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "end_to_end_benchmark: " << error.what() << '\n';
        return 2;
    }
}
