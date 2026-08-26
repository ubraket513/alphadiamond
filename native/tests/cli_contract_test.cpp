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
    (void)argv[2];
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    const auto config = write_config(scratch);
    const auto run_dir = scratch / "runtime" / "runs" / "soo" / "native-cli";
    std::filesystem::create_directories(run_dir);
    std::filesystem::copy_file(config, run_dir / "resolved-config.json",
                               std::filesystem::copy_options::overwrite_existing);

    std::vector<std::string> calls;
    const CommandService service = [&](const CommandRequest& request, const ProductionConfig& parsed) {
        CHECK_EQ(parsed.model_name, std::string("Soo"));
        CHECK_EQ(request.run_dir, run_dir);
        calls.push_back(request.command);
        return JsonValue::Object{{"service", JsonValue{std::string("injected")}}};
    };
    for (const char* verb : {"train", "resume", "evaluate", "report"}) {
        std::ostringstream output;
        CHECK_EQ(diamond_orchestration::dispatch_command(command(verb, scratch, config), service, output),
                 diamond_orchestration::kExitOk);
        const auto& report = report_object(output);
        CHECK_EQ(std::get<std::string>(report.at("command").value), std::string(verb));
        CHECK_EQ(std::get<std::string>(report.at("status").value), std::string("ok"));
        CHECK_EQ(std::get<std::string>(report.at("service").value), std::string("injected"));
    }
    CHECK_EQ(calls.size(), std::size_t{4});

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

    auto invalid_resume = command("resume", scratch, config);
    invalid_resume.emplace_back("--config");
    invalid_resume.emplace_back(config.string());
    std::ostringstream resume_argument_output;
    CHECK_EQ(
        diamond_orchestration::dispatch_command(invalid_resume, service, resume_argument_output),
        diamond_orchestration::kExitArgumentError);

    auto incomplete_evaluate = command("evaluate", scratch, config);
    incomplete_evaluate.erase(incomplete_evaluate.begin() + 5, incomplete_evaluate.begin() + 7);
    std::ostringstream evaluate_argument_output;
    CHECK_EQ(diamond_orchestration::dispatch_command(incomplete_evaluate, service,
                                                     evaluate_argument_output),
             diamond_orchestration::kExitArgumentError);

    std::ostringstream argument_output;
    CHECK_EQ(diamond_orchestration::dispatch_command({"train"}, service, argument_output),
             diamond_orchestration::kExitArgumentError);
    CHECK_EQ(std::get<std::string>(report_object(argument_output).at("status").value), std::string("error"));

    std::ostringstream runtime_output;
    const CommandService incompatible = [](const CommandRequest&, const ProductionConfig&)
        -> JsonValue::Object { throw diamond_orchestration::CommandArtifactError("incompatible checkpoint"); };
    CHECK_EQ(diamond_orchestration::dispatch_command(command("train", scratch, config), incompatible, runtime_output),
             diamond_orchestration::kExitArtifactError);

    std::ostringstream interrupted_output;
    const CommandService interrupted = [](const CommandRequest&, const ProductionConfig&)
        -> JsonValue::Object { throw diamond_orchestration::CommandInterruptedError("unfinished stage"); };
    CHECK_EQ(diamond_orchestration::dispatch_command(command("resume", scratch, config), interrupted, interrupted_output),
             diamond_orchestration::kExitInterrupted);

    std::ostringstream internal_output;
    const CommandService broken = [](const CommandRequest&, const ProductionConfig&)
        -> JsonValue::Object { throw std::runtime_error("broken"); };
    CHECK_EQ(diamond_orchestration::dispatch_command(command("train", scratch, config), broken, internal_output),
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
    return soo_test::report("cli_contract_test");
}
