#include "diamond_orchestration/report.hpp"

#include <cctype>
#include <fstream>
#include <ostream>
#include <string_view>
#include <typeinfo>
#include <utility>

#include "diamond_orchestration/run_state.hpp"
#include "diamond_support/json.hpp"

namespace diamond_orchestration {
namespace {

using Json = diamond_support::JsonValue;
using Object = Json::Object;

bool is_verb(std::string_view value) {
    return value == "train" || value == "resume" || value == "evaluate" ||
           value == "report";
}

bool safe_run_id(std::string_view value) {
    if (value.empty() || !std::isalnum(static_cast<unsigned char>(value.front()))) return false;
    for (unsigned char character : value)
        if (!std::isalnum(character) && character != '.' && character != '_' && character != '-') return false;
    return value != "." && value != "..";
}

std::string required_value(const std::vector<std::string>& arguments, std::size_t& index,
                           std::string_view option) {
    if (++index >= arguments.size() || arguments[index].starts_with("--"))
        throw CommandArgumentError("missing required argument " + std::string(option));
    return arguments[index];
}

CommandRequest parse_request(const std::vector<std::string>& arguments) {
    if (arguments.empty()) throw CommandArgumentError("the following arguments are required: command");
    if (!is_verb(arguments.front())) throw CommandArgumentError("unknown command: " + arguments.front());

    CommandRequest request{.command = arguments.front()};
    bool have_run_dir = false, have_runtime = false, have_model = false, have_run_id = false;
    bool have_config = false, have_checkpoint = false, have_warm_start = false;
    bool have_scratch = false, have_candidate = false, have_champion = false;
    bool have_opening_suite = false;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const auto& option = arguments[index];
        if (option == "--run-dir") {
            if (have_run_dir)
                throw CommandArgumentError("duplicate argument --run-dir");
            request.run_dir = required_value(arguments, index, option);
            have_run_dir = true;
        } else if (option == "--runtime-dir") {
            if (have_runtime) throw CommandArgumentError("duplicate argument --runtime-dir");
            request.runtime_dir = required_value(arguments, index, option);
            have_runtime = true;
        } else if (option == "--model") {
            if (have_model) throw CommandArgumentError("duplicate argument --model");
            request.model_name = required_value(arguments, index, option);
            have_model = true;
        } else if (option == "--run-id") {
            if (have_run_id) throw CommandArgumentError("duplicate argument --run-id");
            request.run_id = required_value(arguments, index, option);
            have_run_id = true;
        } else if (option == "--config") {
            if (have_config) throw CommandArgumentError("duplicate argument --config");
            request.config_path = required_value(arguments, index, option);
            have_config = true;
        } else if (option == "--checkpoint") {
            if (have_checkpoint) throw CommandArgumentError("duplicate argument --checkpoint");
            request.checkpoint_path = required_value(arguments, index, option);
            have_checkpoint = true;
        } else if (option == "--warm-start") {
            if (have_warm_start)
                throw CommandArgumentError("duplicate argument --warm-start");
            request.warm_start_path = required_value(arguments, index, option);
            have_warm_start = true;
        } else if (option == "--scratch") {
            if (have_scratch)
                throw CommandArgumentError("duplicate argument --scratch");
            have_scratch = true;
        } else if (option == "--candidate") {
            if (have_candidate)
                throw CommandArgumentError("duplicate argument --candidate");
            request.candidate_path = required_value(arguments, index, option);
            have_candidate = true;
        } else if (option == "--champion") {
            if (have_champion)
                throw CommandArgumentError("duplicate argument --champion");
            request.champion_path = required_value(arguments, index, option);
            have_champion = true;
        } else if (option == "--opening-suite") {
            if (have_opening_suite)
                throw CommandArgumentError("duplicate argument --opening-suite");
            request.opening_suite_id = required_value(arguments, index, option);
            have_opening_suite = true;
        } else {
            throw CommandArgumentError("unrecognized argument: " + option);
        }
    }

    if (!have_run_dir) {
        if (!have_runtime || !have_model || !have_run_id)
            throw CommandArgumentError("missing required argument --run-dir");
        if (request.model_name != "Soo" && request.model_name != "Min")
            throw CommandArgumentError("--model must be Soo or Min");
        request.run_dir = request.runtime_dir / "runs" /
                          (request.model_name == "Soo" ? "soo" : "min") / request.run_id;
    } else {
        const auto family = request.run_dir.parent_path().filename().string();
        const auto derived_model = family == "soo" ? "Soo" : family == "min" ? "Min" : "";
        if (derived_model[0] == '\0')
            throw CommandArgumentError("--run-dir parent must be soo or min");
        const auto derived_run_id = request.run_dir.filename().string();
        if (have_model && request.model_name != derived_model)
            throw CommandArgumentError("--model does not match --run-dir");
        if (have_run_id && request.run_id != derived_run_id)
            throw CommandArgumentError("--run-id does not match --run-dir");
        request.model_name = derived_model;
        request.run_id = derived_run_id;
        request.runtime_dir = request.run_dir.parent_path().parent_path().parent_path();
    }
    if (!safe_run_id(request.run_id)) throw CommandArgumentError("run_id contains unsafe path characters");

    const int initialization_count = static_cast<int>(have_scratch) +
                                     static_cast<int>(have_checkpoint) +
                                     static_cast<int>(have_warm_start);
    if (request.command == "train") {
        if (!have_config)
            throw CommandArgumentError("missing required argument --config");
        if (initialization_count != 1)
            throw CommandArgumentError(
                "train requires exactly one of --scratch, --checkpoint, or --warm-start");
        if (have_candidate || have_champion || have_opening_suite)
            throw CommandArgumentError("train received an evaluation-only argument");
        request.initialization = have_scratch      ? CommandInitialization::scratch
                                 : have_warm_start ? CommandInitialization::warm_start
                                                   : CommandInitialization::native_checkpoint;
    } else if (request.command == "resume" || request.command == "report") {
        if ((request.command == "report" && have_config) || initialization_count != 0 ||
            have_candidate || have_champion || have_opening_suite || have_runtime || have_model ||
            have_run_id) {
            throw CommandArgumentError(request.command +
                                       (request.command == "resume"
                                            ? " accepts only --run-dir and optional --config"
                                            : " accepts only --run-dir"));
        }
    } else {
        if (have_config || initialization_count != 0 || !have_candidate || !have_champion ||
            !have_opening_suite || have_runtime || have_model || have_run_id) {
            throw CommandArgumentError(
                "evaluate requires only --run-dir, --candidate, --champion, and --opening-suite");
        }
    }
    return request;
}

ProductionConfig load_config(const CommandRequest& request) {
    try {
        auto path = request.command == "train" ||
                            (request.command == "resume" && !request.config_path.empty())
                        ? request.config_path
                        : request.run_dir / "resolved-config.json";
        if ((request.command == "resume" && request.config_path.empty()) ||
            request.command == "evaluate") {
            const auto active = request.run_dir / "active-config.json";
            if (std::filesystem::exists(active)) path = active;
        }
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            if (request.command == "train")
                throw CommandArgumentError("cannot open config: " + path.string());
            throw CommandArtifactError("resolved run config is missing: " + path.string());
        }
        const std::string contents((std::istreambuf_iterator<char>(input)), {});
        const auto parsed = diamond_support::parse_json(contents);
        const auto config = ProductionConfig::from_json(parsed);
        if (config.model_name != request.model_name)
            throw CommandArgumentError("--model does not match config model_name");
        return config;
    } catch (const CommandArgumentError&) {
        throw;
    } catch (const CommandArtifactError&) {
        throw;
    } catch (const ConfigError& error) {
        throw CommandArgumentError(error.what());
    } catch (const std::exception& error) {
        throw CommandArgumentError("invalid config: " + std::string(error.what()));
    }
}

void emit_error(const std::string& command, const std::string& error, std::ostream& output) {
    write_command_report(make_command_report(command, "error", {{"error", Json{error}}}), output);
}

}  // namespace

Json make_command_report(const std::string& command, const std::string& status, Object details) {
    if (status != "ok" && status != "error") throw std::invalid_argument("invalid command report status");
    if (details.contains("command") || details.contains("status"))
        throw std::invalid_argument("command report details cannot replace reserved fields");
    details.emplace("command", command.empty() ? Json{nullptr} : Json{command});
    details.emplace("status", Json{status});
    return Json{std::move(details)};
}

void write_command_report(const Json& report, std::ostream& output) {
    output << diamond_support::canonical_json(report) << '\n';
}

int dispatch_command(const std::vector<std::string>& arguments, const CommandService& service,
                     std::ostream& output) {
    const std::string command_hint = arguments.empty() ? "" : arguments.front();
    try {
        if (arguments.size() == 1 && (arguments.front() == "--help" || arguments.front() == "-h")) {
            write_command_report(
                make_command_report(
                    "", "ok",
                    {{"usage", Json{"train --run-dir DIR --config FILE (--scratch|--checkpoint "
                                    "DIR|--warm-start ARTIFACT); "
                                    "resume --run-dir DIR [--config FILE]; evaluate --run-dir DIR --candidate DIR "
                                    "--champion DIR "
                                    "--opening-suite ID; report --run-dir DIR"}}}),
                output);
            return kExitOk;
        }
        const auto request = parse_request(arguments);
        if (!service) throw std::runtime_error("native command service is required");
        const auto config = load_config(request);
        auto details = service(request, config);
        if (details.contains("error")) throw std::runtime_error("successful command report cannot contain error");
        write_command_report(make_command_report(request.command, "ok", std::move(details)), output);
        return kExitOk;
    } catch (const CommandArgumentError& error) {
        emit_error(command_hint, error.what(), output);
        return kExitArgumentError;
    } catch (const CommandArtifactError& error) {
        emit_error(command_hint, error.what(), output);
        return kExitArtifactError;
    } catch (const CommandInterruptedError& error) {
        emit_error(command_hint, error.what(), output);
        return kExitInterrupted;
    } catch (const RunStateError& error) {
        emit_error(command_hint, error.what(), output);
        return kExitArtifactError;
    } catch (const std::exception& error) {
        emit_error(command_hint, std::string(typeid(error).name()) + ": " + error.what(), output);
        return kExitInternalError;
    } catch (...) {
        emit_error(command_hint, "unknown native command failure", output);
        return kExitInternalError;
    }
}

int dispatch_command(int argc, char** argv, const CommandService& service, std::ostream& output) {
    std::vector<std::string> arguments;
    for (int index = 1; index < argc; ++index) arguments.emplace_back(argv[index]);
    return dispatch_command(arguments, service, output);
}

}  // namespace diamond_orchestration
