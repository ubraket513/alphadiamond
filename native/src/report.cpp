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
    bool have_runtime = false, have_model = false, have_run_id = false;
    bool have_config = false, have_checkpoint = false;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const auto& option = arguments[index];
        if (option == "--runtime-dir") {
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
        } else {
            throw CommandArgumentError("unrecognized argument: " + option);
        }
    }
    if (!have_runtime) throw CommandArgumentError("missing required argument --runtime-dir");
    if (!have_model) throw CommandArgumentError("missing required argument --model");
    if (!have_config) throw CommandArgumentError("missing required argument --config");
    if (!have_checkpoint) throw CommandArgumentError("missing required argument --checkpoint");
    if (request.model_name != "Soo" && request.model_name != "Min")
        throw CommandArgumentError("--model must be Soo or Min");
    if (!have_run_id) throw CommandArgumentError("missing required argument --run-id");
    if (!safe_run_id(request.run_id)) throw CommandArgumentError("run_id contains unsafe path characters");
    return request;
}

ProductionConfig load_config(const CommandRequest& request) {
    try {
        std::ifstream input(request.config_path, std::ios::binary);
        if (!input) throw CommandArgumentError("cannot open config: " + request.config_path.string());
        const std::string contents((std::istreambuf_iterator<char>(input)), {});
        const auto parsed = diamond_support::parse_json(contents);
        const auto config = ProductionConfig::from_json(parsed);
        if (config.model_name != request.model_name)
            throw CommandArgumentError("--model does not match config model_name");
        return config;
    } catch (const CommandArgumentError&) {
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
            write_command_report(make_command_report("", "ok", {{"usage", Json{"alphadiamond-train <train|resume|evaluate|report> [options]"}}}), output);
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
