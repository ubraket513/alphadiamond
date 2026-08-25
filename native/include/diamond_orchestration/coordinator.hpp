#pragma once

#include <chrono>
#include <cstdint>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <optional>
#include <stdexcept>
#include <string>

#include "diamond_orchestration/run_state.hpp"

namespace diamond_orchestration {

class CoordinatorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// The callback must make the supplied operation durable before returning.  Its
// operation ID is stable across retries of the same unfinished stage.
using StageCallback = std::function<void(RunStage stage, const RunState& state,
                                         const std::string& operation_id)>;
using StagePayload = std::function<diamond_support::JsonValue(
    RunStage stage, const RunState& state)>;

class Coordinator final {
  public:
    Coordinator(RunStateStore& store, StagePayload describe_stage,
                StageCallback execute_stage);

    // Advance every unfinished stage in an authoritative state snapshot.
    RunState run(const RunState& state);

    // Complete the current iteration, then start fresh iterations until the
    // total run budget or wall-clock deadline is reached. At least one bound
    // is required. A returned state is always at a durable stage boundary.
    RunState run_bounded(
        const RunState& state, std::optional<uint64_t> max_iterations,
        std::optional<std::chrono::steady_clock::time_point> deadline);

    // Load an interrupted run and advance only the stage at its saved cursor
    // and later stages.
    RunState resume(const std::string& model_name, const std::string& run_id);

    // Returns the durable operation identity for state.stage().
    static std::string operation_id(const RunState& state,
                                    const diamond_support::JsonValue& stage_payload);

  private:
    RunStateStore& store_;
    StagePayload describe_stage_;
    StageCallback execute_stage_;
};

}  // namespace diamond_orchestration
