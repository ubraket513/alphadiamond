#include "diamond_orchestration/coordinator.hpp"

#include <array>
#include <utility>

#include "diamond_support/json.hpp"

namespace diamond_orchestration {
namespace {

using Json = diamond_support::JsonValue;
using Object = Json::Object;

constexpr std::array<RunStage, 10> kStages = {
    RunStage::initialize, RunStage::self_play, RunStage::replay_ingest,
    RunStage::train, RunStage::save_candidate, RunStage::promotion_arena,
    RunStage::rating_benchmark, RunStage::promote_or_reject,
    RunStage::persist, RunStage::complete,
};
constexpr std::array<std::string_view, 10> kStageNames = {
    "INITIALIZE", "SELF_PLAY", "REPLAY_INGEST", "TRAIN", "SAVE_CANDIDATE",
    "PROMOTION_ARENA", "RATING_BENCHMARK", "PROMOTE_OR_REJECT", "PERSIST", "COMPLETE",
};

std::string_view stage_name(RunStage stage) {
    for (std::size_t index = 0; index < kStages.size(); ++index)
        if (kStages[index] == stage) return kStageNames[index];
    throw CoordinatorError("unknown run stage");
}

RunStage next_stage(RunStage stage) {
    for (std::size_t index = 0; index + 1 < kStages.size(); ++index)
        if (kStages[index] == stage) return kStages[index + 1];
    throw CoordinatorError("COMPLETE is terminal");
}

const std::string& string_field(const Object& object, std::string_view name) {
    const auto found = object.find(std::string(name));
    if (found == object.end()) throw CoordinatorError("run state missing " + std::string(name));
    const auto* value = std::get_if<std::string>(&found->second.value);
    if (!value || value->empty()) throw CoordinatorError("run state has invalid " + std::string(name));
    return *value;
}

int64_t integer_field(const Object& object, std::string_view name) {
    const auto found = object.find(std::string(name));
    if (found == object.end()) throw CoordinatorError("run state missing " + std::string(name));
    const auto* value = std::get_if<int64_t>(&found->second.value);
    if (!value || *value < 0) throw CoordinatorError("run state has invalid " + std::string(name));
    return *value;
}

const Json& field(const Object& object, std::string_view name) {
    const auto found = object.find(std::string(name));
    if (found == object.end()) throw CoordinatorError("run state missing " + std::string(name));
    return found->second;
}

StageOutcome decode_outcome(const Json& value) {
    const auto* object = std::get_if<Object>(&value.value);
    if (!object || object->size() != 2 || !object->contains("result") ||
        !object->contains("progress")) {
        throw CoordinatorError("durable stage outcome is malformed");
    }
    const auto* progress = std::get_if<Object>(&object->at("progress").value);
    if (!progress)
        throw CoordinatorError("durable stage progress is malformed");
    return {.result = object->at("result"), .progress = *progress};
}

void require_authoritative(RunStateStore& store, const RunState& state) {
    const auto authoritative = store.load(state.model_name(), state.run_id());
    if (diamond_support::canonical_json(Json{authoritative.payload()}) !=
        diamond_support::canonical_json(Json{state.payload()})) {
        throw RunStateError("supplied training run state is stale");
    }
}

}  // namespace

Coordinator::Coordinator(RunStateStore& store, StagePayload describe_stage,
                         StageCallback execute_stage)
    : store_(store), describe_stage_(std::move(describe_stage)),
      execute_stage_(std::move(execute_stage)) {
    if (!describe_stage_ || !execute_stage_)
        throw CoordinatorError("stage description and callback are required");
}

RunState Coordinator::run(const RunState& state) {
    require_authoritative(store_, state);
    auto current = state;
    while (current.stage() != RunStage::complete) {
        const auto payload = describe_stage_(current.stage(), current);
        const auto id = operation_id(current, payload);
        const auto durable = store_.load_operation_result(current, id);
        StageOutcome outcome;
        if (durable) {
            outcome = decode_outcome(*durable);
            store_.commit_operation_result(current, id, *durable);
        } else {
            outcome = execute_stage_(current.stage(), current, id);
            store_.commit_operation_result(current, id, durable_result(outcome));
        }
        current = store_.transition(current, next_stage(current.stage()), id,
                                    std::move(outcome.progress));
    }
    return current;
}

RunState Coordinator::run_bounded(
    const RunState& state, std::optional<uint64_t> max_iterations,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
    if ((!max_iterations && !deadline) || (max_iterations && *max_iterations == 0))
        throw CoordinatorError("bounded run requires a positive iteration or wall-clock budget");

    auto current = state;
    if (current.stage() != RunStage::complete) current = run(current);
    while ((!max_iterations ||
            static_cast<uint64_t>(integer_field(current.payload(), "iteration")) + 1 <
                *max_iterations) &&
           (!deadline || std::chrono::steady_clock::now() < *deadline)) {
        current = run(store_.start_next_iteration(current));
    }
    return current;
}

RunState Coordinator::resume(const std::string& model_name, const std::string& run_id) {
    return run(store_.load(model_name, run_id));
}

std::string Coordinator::operation_id(const RunState& state, const Json& stage_payload) {
    const auto& payload = state.payload();
    Object identity = {
        {"compatibility_namespace", Json{string_field(payload, "compatibility_namespace")}},
        {"iteration", Json{integer_field(payload, "iteration")}},
        {"model_name", Json{state.model_name()}},
        {"protocol_ids", field(payload, "protocol_ids")},
        {"run_id", Json{state.run_id()}},
        {"run_seed", Json{integer_field(payload, "run_seed")}},
        {"stage", Json{std::string(stage_name(state.stage()))}},
        {"stage_payload", stage_payload},
    };
    return "sha256:" + diamond_support::sha256(diamond_support::canonical_json(Json{std::move(identity)}));
}

Json Coordinator::durable_result(const StageOutcome& outcome) {
    return Json{Object{{"progress", Json{outcome.progress}}, {"result", outcome.result}}};
}

}  // namespace diamond_orchestration
