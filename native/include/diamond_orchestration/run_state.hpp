#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

#include "diamond_support/json.hpp"

namespace diamond_orchestration {

class RunStateError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class RunStage {
    initialize, self_play, replay_ingest, train, save_candidate,
    promotion_arena, rating_benchmark, promote_or_reject, persist, complete,
};

class RunState final {
  public:
    static RunState initialize(std::string run_id,
                               diamond_support::JsonValue::Object compatibility,
                               diamond_support::JsonValue::Object protocol_ids,
                               uint64_t run_seed);
    static RunState initialize(std::string run_id, std::string model_name,
                               std::string compatibility_namespace,
                               std::string protocol_id, uint64_t run_seed);
    static RunState from_json(diamond_support::JsonValue::Object payload);

    const diamond_support::JsonValue::Object& payload() const noexcept { return payload_; }
    const std::string& run_id() const noexcept { return run_id_; }
    const std::string& model_name() const noexcept { return model_name_; }
    RunStage stage() const noexcept { return stage_; }
    uint64_t generation() const noexcept { return generation_; }
    uint64_t derive_seed(const diamond_support::JsonValue::Array& identities) const;
    RunState with_progress(diamond_support::JsonValue::Object changes) const;

  private:
    explicit RunState(diamond_support::JsonValue::Object payload);

    diamond_support::JsonValue::Object payload_;
    std::string run_id_;
    std::string model_name_;
    RunStage stage_ = RunStage::initialize;
    uint64_t generation_ = 0;

    friend class RunStateStore;
};

class RunStateStore final {
  public:
    explicit RunStateStore(std::filesystem::path root) : root_(std::move(root)) {}

    RunState initialize(const RunState& state);
    RunState load(const std::string& model_name, const std::string& run_id) const;
    RunState save(const RunState& state);
    RunState transition(const RunState& state, RunStage next, std::string completion_marker,
                        diamond_support::JsonValue::Object progress = {});
    RunState start_next_iteration(const RunState& state);

    // Operation results are immutable. A matching result is reused after a
    // crash between durable work completion and the state transition.
    std::optional<diamond_support::JsonValue>
    load_operation_result(const RunState& state, const std::string& operation_id) const;
    void commit_operation_result(const RunState& state, const std::string& operation_id,
                                 diamond_support::JsonValue result);

  private:
    std::filesystem::path state_path(const std::string& model_name, const std::string& run_id) const;
    std::filesystem::path operation_result_path(const RunState& state,
                                                const std::string& operation_id) const;
    std::filesystem::path root_;
};

}  // namespace diamond_orchestration
