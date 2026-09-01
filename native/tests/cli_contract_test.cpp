#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "diamond_orchestration/report.hpp"
#include "diamond_orchestration/training_resources.hpp"
#include "diamond_orchestration/training_wiring.hpp"

namespace {

using diamond_orchestration::CommandRequest;
using diamond_orchestration::CommandService;
using diamond_orchestration::ProductionConfig;
using diamond_support::JsonValue;

std::filesystem::path write_config(const std::filesystem::path& scratch) {
    const auto path = scratch / "soo-production.json";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << diamond_support::canonical_json(ProductionConfig{}.to_json());
    return path;
}

std::vector<std::string> command(std::string verb, const std::filesystem::path& scratch,
                                 const std::filesystem::path& config) {
    const auto run_dir = scratch / "runtime" / "runs" / "soo" / "native-cli";
    if (verb == "train")
        return {std::move(verb), "--run-dir",     run_dir.string(),
                "--config",      config.string(), "--scratch"};
    if (verb == "resume" || verb == "report")
        return {std::move(verb), "--run-dir", run_dir.string()};
    return {std::move(verb),
            "--run-dir",
            run_dir.string(),
            "--candidate",
            (scratch / "candidate-checkpoint").string(),
            "--champion",
            (scratch / "champion-checkpoint").string(),
            "--opening-suite",
            "production-openings-v1"};
}

JsonValue::Object report_object(const std::ostringstream& output) {
    const auto parsed = diamond_support::parse_json(output.str());
    return std::get<JsonValue::Object>(parsed.value);
}

JsonValue::Object report_object(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(static_cast<bool>(input), "cannot open native CLI report");
    const std::string contents{std::istreambuf_iterator<char>(input), {}};
    const auto parsed = diamond_support::parse_json(contents);
    return std::get<JsonValue::Object>(parsed.value);
}

} // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 3, "usage: cli_contract_test <scratch> <alphadiamond-train>");
    const auto scratch = std::filesystem::path(argv[1]);
    const auto train_binary = std::filesystem::path(argv[2]);
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    const auto config = write_config(scratch);
    const auto run_dir = scratch / "runtime" / "runs" / "soo" / "native-cli";
    std::filesystem::create_directories(run_dir);
    std::filesystem::copy_file(config, run_dir / "resolved-config.json",
                               std::filesystem::copy_options::overwrite_existing);

    std::vector<std::string> calls;
    const CommandService service = [&](const CommandRequest& request,
                                       const ProductionConfig& parsed) {
        CHECK_EQ(parsed.model_name, std::string("Soo"));
        CHECK_EQ(request.run_dir, run_dir);
        calls.push_back(request.command);
        return JsonValue::Object{{"service", JsonValue{std::string("injected")}}};
    };
    for (const char* verb : {"train", "resume", "evaluate", "report"}) {
        std::ostringstream output;
        CHECK_EQ(diamond_orchestration::dispatch_command(command(verb, scratch, config), service,
                                                         output),
                 diamond_orchestration::kExitOk);
        const auto& report = report_object(output);
        CHECK_EQ(std::get<std::string>(report.at("command").value), std::string(verb));
        CHECK_EQ(std::get<std::string>(report.at("status").value), std::string("ok"));
        CHECK_EQ(std::get<std::string>(report.at("service").value), std::string("injected"));
    }
    CHECK_EQ(calls.size(), std::size_t{4});

    auto active_config = ProductionConfig{};
    active_config.model_version = "active-transition";
    {
        std::ofstream output(run_dir / "active-config.json", std::ios::binary | std::ios::trunc);
        output << diamond_support::canonical_json(active_config.to_json());
    }
    const CommandService active_evaluation = [&](const CommandRequest& request,
                                                  const ProductionConfig& parsed) {
        CHECK_EQ(request.command, std::string("evaluate"));
        CHECK_EQ(parsed.model_version, std::string("active-transition"));
        return JsonValue::Object{{"service", JsonValue{std::string("active")}}};
    };
    std::ostringstream active_evaluation_output;
    CHECK_EQ(diamond_orchestration::dispatch_command(command("evaluate", scratch, config),
                                                     active_evaluation,
                                                     active_evaluation_output),
             diamond_orchestration::kExitOk);

    auto conflicting_selectors = command("train", scratch, config);
    conflicting_selectors.emplace_back("--warm-start");
    conflicting_selectors.emplace_back((scratch / "deployment").string());
    std::ostringstream conflict_output;
    CHECK_EQ(
        diamond_orchestration::dispatch_command(conflicting_selectors, service, conflict_output),
        diamond_orchestration::kExitArgumentError);

    auto missing_selector = command("train", scratch, config);
    missing_selector.pop_back();
    std::ostringstream selector_output;
    CHECK_EQ(diamond_orchestration::dispatch_command(missing_selector, service, selector_output),
             diamond_orchestration::kExitArgumentError);

    auto transition_resume = command("resume", scratch, config);
    transition_resume.emplace_back("--config");
    transition_resume.emplace_back(config.string());
    std::ostringstream resume_argument_output;
    CHECK_EQ(
        diamond_orchestration::dispatch_command(transition_resume, service, resume_argument_output),
        diamond_orchestration::kExitOk);

    auto incomplete_evaluate = command("evaluate", scratch, config);
    incomplete_evaluate.erase(incomplete_evaluate.begin() + 5, incomplete_evaluate.begin() + 7);
    std::ostringstream evaluate_argument_output;
    CHECK_EQ(diamond_orchestration::dispatch_command(incomplete_evaluate, service,
                                                     evaluate_argument_output),
             diamond_orchestration::kExitArgumentError);

    std::ostringstream argument_output;
    CHECK_EQ(diamond_orchestration::dispatch_command({"train"}, service, argument_output),
             diamond_orchestration::kExitArgumentError);
    CHECK_EQ(std::get<std::string>(report_object(argument_output).at("status").value),
             std::string("error"));

    std::ostringstream runtime_output;
    const CommandService incompatible = [](const CommandRequest&,
                                           const ProductionConfig&) -> JsonValue::Object {
        throw diamond_orchestration::CommandArtifactError("incompatible checkpoint");
    };
    CHECK_EQ(diamond_orchestration::dispatch_command(command("train", scratch, config),
                                                     incompatible, runtime_output),
             diamond_orchestration::kExitArtifactError);

    std::ostringstream interrupted_output;
    const CommandService interrupted = [](const CommandRequest&,
                                          const ProductionConfig&) -> JsonValue::Object {
        throw diamond_orchestration::CommandInterruptedError("unfinished stage");
    };
    CHECK_EQ(diamond_orchestration::dispatch_command(command("resume", scratch, config),
                                                     interrupted, interrupted_output),
             diamond_orchestration::kExitInterrupted);

    std::ostringstream internal_output;
    const CommandService broken = [](const CommandRequest&,
                                     const ProductionConfig&) -> JsonValue::Object {
        throw std::runtime_error("broken");
    };
    CHECK_EQ(diamond_orchestration::dispatch_command(command("train", scratch, config), broken,
                                                     internal_output),
             diamond_orchestration::kExitInternalError);

    ProductionConfig wiring_config;
    wiring_config.workers.logical_lanes = 7;
    wiring_config.workers.search_threads = 2;
    wiring_config.workers.games_per_iteration = 9;
    wiring_config.inference.max_batch_size = 11;
    wiring_config.inference.max_wait_us = 50;
    wiring_config.training.batch_size = 3;
    wiring_config.training.train_steps_per_iteration = 2;
    const auto wiring = diamond_orchestration::wire_training_iteration(wiring_config);
    CHECK_EQ(wiring.selfplay.lanes, 7);
    CHECK_EQ(wiring.selfplay.threads, 2);
    CHECK_EQ(wiring.selfplay.max_batch, 11);
    CHECK_EQ(wiring.selfplay.max_wait_us, 50);
    CHECK_EQ(wiring.games_per_iteration, std::size_t{9});
    CHECK_EQ(wiring.training_batch_size, std::size_t{3});
    CHECK_EQ(wiring.training_steps, std::size_t{2});

    // The repetition trigger reached self-play only if it is forwarded here.
    // It was implemented in the engine and measured as the best search-budget
    // policy, then shipped unreachable: no config parsed it and this wiring did
    // not carry it, so every run silently used a flat budget. A field the
    // engine supports but the wiring drops fails invisibly -- the search stays
    // correct and only the abort rate moves -- so it is asserted rather than
    // assumed. See docs/model-training/repetition_trigger_config_gap.md.
    CHECK_EQ(wiring.selfplay.simulations_late, 0);
    CHECK_EQ(wiring.selfplay.repeat_window, 0);
    auto trigger_config = wiring_config;
    trigger_config.mcts.simulations_late = 256;
    trigger_config.mcts.repeat_window = 8;
    const auto triggered = diamond_orchestration::wire_training_iteration(trigger_config);
    CHECK_EQ(triggered.selfplay.simulations_late, 256);
    CHECK_EQ(triggered.selfplay.repeat_window, 8);

    // The same failure, one field over. `self_play.bootstrap_prior` was parsed
    // and validated by the config and read by nothing: vacancy_prior() had a
    // single caller, a timing harness. So the heuristic never reached search,
    // every from-scratch game ran to max_moves, and the iteration died on
    // "insufficient replay samples" -- which reads like a replay bug and is
    // not one. Soo hid it by warm-starting from a shipped artifact; Min has
    // none, so for Min this was the difference between training and not.
    // Asserted here because a prior that is silently dropped leaves the search
    // correct and only moves the completion rate.
    CHECK(!wiring.selfplay.bootstrap_prior);
    auto bootstrap_config = wiring_config;
    bootstrap_config.self_play.bootstrap_prior = "canonical-target-vacancy-distance-v2";
    bootstrap_config.self_play.bootstrap_prior_weight = 0.5;
    const auto bootstrap_wiring =
        diamond_orchestration::wire_training_iteration(bootstrap_config).selfplay;
    CHECK(bootstrap_wiring.bootstrap_prior);
    CHECK_EQ(bootstrap_wiring.bootstrap_prior_weight, 0.5);
    auto explicit_none = wiring_config;
    explicit_none.self_play.bootstrap_prior =
        std::string(diamond_orchestration::kBootstrapPriorNone);
    CHECK(!diamond_orchestration::wire_training_iteration(explicit_none).selfplay.bootstrap_prior);

    // The arena runs the same phase as self-play.
    //
    // It did not: `arena_episode_config` never set the prior, so during
    // bootstrap both sides searched on a network prior that steers nobody into
    // a camp, every game reached `arena.max_moves`, and the stage reported only
    // incomplete blocks -- after paying for a full-length game each, played one
    // at a time. Dropped here the failure is quiet in the same way self-play's
    // was: the arena stays correct and only its completion rate and its cost
    // move. Both sides take the same prior, so the comparison stays symmetric.
    CHECK(!diamond_orchestration::wire_arena_episode(wiring_config, 4).bootstrap_prior);
    const auto bootstrap_arena = diamond_orchestration::wire_arena_episode(bootstrap_config, 4);
    CHECK(bootstrap_arena.bootstrap_prior);
    CHECK_EQ(bootstrap_arena.bootstrap_prior_weight, 0.5);

    // Lanes are the size of the group of games being played together, and
    // threads and batch cannot usefully exceed it: synchronous MCTS allows one
    // outstanding request per lane, so a group of two games can never present a
    // third position to evaluate. Above the group size both saturate.
    const auto small_group = diamond_orchestration::wire_arena_episode(wiring_config, 2);
    CHECK_EQ(small_group.lanes, 2);
    CHECK_EQ(small_group.threads, 2);
    CHECK_EQ(small_group.max_batch, 2);
    const auto large_group = diamond_orchestration::wire_arena_episode(wiring_config, 32);
    CHECK_EQ(large_group.lanes, 32);
    CHECK_EQ(large_group.threads, 2);
    CHECK_EQ(large_group.max_batch, 11);

    // The arena is a measurement, not a source of training data: its search is
    // greedy from the first move and takes no root exploration noise, and its
    // move budget is the arena's own rather than self-play's. Only a repeated
    // physical state activates the seeded escape below.
    CHECK_EQ(large_group.temperature, 0.0);
    CHECK_EQ(large_group.temperature_moves, 0);
    CHECK_EQ(large_group.dirichlet_epsilon, 0.0);
    CHECK_EQ(large_group.max_moves, static_cast<int>(wiring_config.arena.max_moves));
    CHECK_EQ(large_group.repeat_window, 8);
    CHECK_EQ(large_group.repetition_temperature, 1.0);

    // Repetition escape is an evaluation rule, not a training-data policy.
    // Self-play keeps its configured repetition search trigger and never
    // samples a move merely because a position repeated.
    CHECK_EQ(wiring.selfplay.repetition_temperature, 0.0);

    // Long-running self-play can deliberately bypass comparative evaluation.
    // In that mode the newly trained candidate must become the next actor;
    // keeping the old champion would silently generate every iteration from
    // stale weights and defeat continuous training.
    auto continuous_config = wiring_config;
    continuous_config.arena.enabled = false;
    const auto continuous = diamond_orchestration::wire_evaluation_pipeline(continuous_config);
    CHECK(!continuous.run_arena);
    CHECK(!continuous.record_rating);
    CHECK(continuous.activate_candidate);

    const auto evaluated = diamond_orchestration::wire_evaluation_pipeline(wiring_config);
    CHECK(evaluated.run_arena);
    CHECK(evaluated.record_rating);
    CHECK(!evaluated.activate_candidate);

    // A live training invocation keeps one materialized replay across stages
    // and iterations. Reconstructing it here would parse and hash every JSON
    // chunk again; at the 1M production capacity that was the dominant
    // iteration cost even after metadata-only opens were introduced.
    diamond_orchestration::TrainingRunResources resources(
        scratch / "cached-replay",
        diamond_training::Compatibility::soo(
            "2.0.0", diamond_training::NetworkConfig{.residual_blocks = 6, .width = 128}),
        1000000, 7);
    CHECK(!resources.replay_loaded());
    auto& first_replay = resources.full_replay();
    CHECK(resources.replay_loaded());
    auto& next_iteration_replay = resources.full_replay();
    CHECK_EQ(&first_replay, &next_iteration_replay);

    auto cuda_config = wiring_config;
    cuda_config.runtime.device = "cuda";
    const auto cuda_config_path = scratch / "cuda-config.json";
    {
        std::ofstream output(cuda_config_path, std::ios::binary | std::ios::trunc);
        output << diamond_support::canonical_json(cuda_config.to_json());
    }
    const auto cuda_run_dir = scratch / "cuda-runtime" / "runs" / "soo" / "cuda-preflight";
    const CommandService cuda_preflight = [](const CommandRequest&,
                                             const ProductionConfig& parsed) -> JsonValue::Object {
        if (parsed.runtime.device != "cuda")
            throw std::runtime_error("CUDA preflight test received the wrong device");
        throw diamond_orchestration::CommandArgumentError(
            "runtime.device cuda requires CUDA, but no CUDA device is available");
    };
    std::ostringstream cuda_output;
    CHECK_EQ(diamond_orchestration::dispatch_command({"train", "--run-dir", cuda_run_dir.string(),
                                                      "--config", cuda_config_path.string(),
                                                      "--scratch"},
                                                     cuda_preflight, cuda_output),
             diamond_orchestration::kExitArgumentError);
    CHECK(!std::filesystem::exists(cuda_run_dir));
    const auto cuda_report = report_object(cuda_output);
    CHECK_EQ(std::get<std::string>(cuda_report.at("status").value), std::string("error"));
    CHECK_EQ(std::get<std::string>(cuda_report.at("error").value),
             std::string("runtime.device cuda requires CUDA, but no CUDA device is available"));

    auto smoke_config = ProductionConfig{};
    smoke_config.runtime.device = "cpu";
    smoke_config.workers.games_per_iteration = 2;
    smoke_config.workers.logical_lanes = 1;
    smoke_config.workers.search_threads = 1;
    smoke_config.inference.max_batch_size = 1;
    smoke_config.inference.max_wait_us = 1;
    smoke_config.mcts.simulations = 1;
    smoke_config.self_play.bootstrap_prior = "canonical-target-vacancy-distance-v2";
    smoke_config.self_play.max_moves = 500;
    smoke_config.training.batch_size = 1;
    smoke_config.training.train_steps_per_iteration = 1;
    smoke_config.run_budget.max_iterations = 1;
    smoke_config.arena.enabled = false;
    const auto smoke_config_path = scratch / "sidecar-smoke.json";
    {
        std::ofstream output(smoke_config_path, std::ios::binary | std::ios::trunc);
        output << diamond_support::canonical_json(smoke_config.to_json());
    }
    const auto smoke_run = scratch / "soo" / "sidecar-run";
    const std::string smoke_command = "\"" + train_binary.string() + "\" train --run-dir \"" +
                                      smoke_run.string() + "\" --config \"" +
                                      smoke_config_path.string() + "\" --scratch";
    CHECK_EQ(std::system(smoke_command.c_str()), 0);
    const auto sidecar = report_object(smoke_run / "iterations" / "0" / "selfplay.metrics.json");
    CHECK_EQ(std::get<int64_t>(sidecar.at("schema_version").value), int64_t{2});
    const auto& search_targets = std::get<JsonValue::Object>(sidecar.at("search_targets").value);
    const auto& all_targets = std::get<JsonValue::Object>(search_targets.at("all").value);
    CHECK(std::get<int64_t>(all_targets.at("rows").value) > 0);
    return soo_test::report("cli_contract_test");
}
