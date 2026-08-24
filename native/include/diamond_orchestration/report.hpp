#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

#include "diamond_orchestration/config.hpp"

namespace diamond_orchestration {

inline constexpr int kExitOk = 0;
inline constexpr int kExitArgumentError = 2;
inline constexpr int kExitArtifactError = 3;
inline constexpr int kExitInterrupted = 4;
inline constexpr int kExitInternalError = 5;

class CommandArgumentError : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
};

class CommandArtifactError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class CommandInterruptedError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct CommandRequest final {
    std::string command;
    std::filesystem::path runtime_dir;
    std::string model_name;
    std::string run_id;
    std::filesystem::path config_path;
    std::filesystem::path checkpoint_path;
};

// Production services own all side effects.  The command boundary validates
// input and config, then supplies the parsed immutable request and config.
using CommandService = std::function<diamond_support::JsonValue::Object(
    const CommandRequest&, const ProductionConfig&)>;

diamond_support::JsonValue make_command_report(
    const std::string& command, const std::string& status,
    diamond_support::JsonValue::Object details = {});
void write_command_report(const diamond_support::JsonValue& report, std::ostream& output);

// Dispatches one command and always emits one canonical JSON report.
int dispatch_command(const std::vector<std::string>& arguments,
                     const CommandService& service, std::ostream& output);
int dispatch_command(int argc, char** argv, const CommandService& service,
                     std::ostream& output);

}  // namespace diamond_orchestration
