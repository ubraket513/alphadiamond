#include "diamond_orchestration/run_state.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <fstream>
#include <set>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace diamond_orchestration {
namespace {

using Json = diamond_support::JsonValue;
using Object = Json::Object;
using Array = Json::Array;

const std::set<std::string> kExactFields = {
    "schema_version", "generation", "run_id", "model_identity", "compatibility",
    "compatibility_namespace", "protocol_ids", "run_seed", "stage",
    "champion_checkpoint", "champion_model_key", "candidate_checkpoint", "iteration",
    "training_step", "replay_manifest", "completed_game_ids", "promotion_records",
    "rating_records", "stage_completions",
};
const std::set<std::string> kProgressFields = {
    "champion_checkpoint", "champion_model_key", "candidate_checkpoint", "iteration",
    "training_step", "replay_manifest", "completed_game_ids", "promotion_records", "rating_records",
};

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

bool safe_id(std::string_view value) {
    if (value.empty() || value == "." || value == "..") return false;
    for (unsigned char c : value)
        if (!std::isalnum(c) && c != '.' && c != '_' && c != '-') return false;
    return true;
}

const std::string& string_field(const Object& object, std::string_view field) {
    const auto it = object.find(std::string(field));
    if (it == object.end()) throw RunStateError("run state missing " + std::string(field));
    const auto* value = std::get_if<std::string>(&it->second.value);
    if (!value || value->empty()) throw RunStateError("run state has invalid " + std::string(field));
    return *value;
}

uint64_t uint_field(const Object& object, std::string_view field) {
    const auto it = object.find(std::string(field));
    if (it == object.end()) throw RunStateError("run state missing " + std::string(field));
    const auto* value = std::get_if<int64_t>(&it->second.value);
    if (!value || *value < 0) throw RunStateError("run state has invalid " + std::string(field));
    return static_cast<uint64_t>(*value);
}

const Object& object_field(const Object& object, std::string_view field) {
    const auto it = object.find(std::string(field));
    if (it == object.end()) throw RunStateError("run state missing " + std::string(field));
    const auto* value = std::get_if<Object>(&it->second.value);
    if (!value) throw RunStateError("run state has invalid " + std::string(field));
    return *value;
}

const Array& array_field(const Object& object, std::string_view field) {
    const auto it = object.find(std::string(field));
    if (it == object.end()) throw RunStateError("run state missing " + std::string(field));
    const auto* value = std::get_if<Array>(&it->second.value);
    if (!value) throw RunStateError("run state has invalid " + std::string(field));
    return *value;
}

void optional_string_field(const Object& object, std::string_view field) {
    const auto it = object.find(std::string(field));
    if (it == object.end()) throw RunStateError("run state missing " + std::string(field));
    if (std::holds_alternative<std::nullptr_t>(it->second.value)) return;
    const auto* value = std::get_if<std::string>(&it->second.value);
    if (!value || value->empty()) throw RunStateError("run state has invalid " + std::string(field));
}

RunStage stage_from_name(std::string_view name) {
    for (std::size_t index = 0; index < kStageNames.size(); ++index)
        if (kStageNames[index] == name) return kStages[index];
    throw RunStateError("run state has invalid stage");
}

std::string_view stage_name(RunStage stage) {
    for (std::size_t index = 0; index < kStages.size(); ++index)
        if (kStages[index] == stage) return kStageNames[index];
    throw RunStateError("unknown run stage");
}

RunStage next_stage(RunStage stage) {
    for (std::size_t index = 0; index + 1 < kStages.size(); ++index)
        if (kStages[index] == stage) return kStages[index + 1];
    throw RunStateError("COMPLETE is terminal");
}

void atomic_write(const std::filesystem::path& path, const std::string& contents) {
    static std::atomic_uint64_t sequence{0};
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp." + std::to_string(++sequence);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw RunStateError("cannot write run state transaction");
        output << contents;
        if (!output) throw RunStateError("cannot write run state transaction");
    }
#ifdef _WIN32
    if (!MoveFileExW(std::filesystem::path(temporary).wstring().c_str(), path.wstring().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::filesystem::remove(temporary);
        throw RunStateError("cannot activate run state transaction");
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) { std::filesystem::remove(temporary); throw RunStateError("cannot activate run state transaction"); }
#endif
}

}  // namespace

RunState::RunState(Object payload) : payload_(std::move(payload)) {
    std::set<std::string> actual;
    for (const auto& [key, unused] : payload_) { (void)unused; actual.insert(key); }
    if (actual != kExactFields) throw RunStateError("malformed training run state");
    const auto schema = uint_field(payload_, "schema_version");
    if (schema != 2) throw RunStateError("unsupported training run schema version");
    run_id_ = string_field(payload_, "run_id");
    if (!safe_id(run_id_)) throw RunStateError("run_id contains unsafe path characters");
    const auto& model = object_field(payload_, "model_identity");
    model_name_ = string_field(model, "model_name");
    if (model_name_ != "Soo" && model_name_ != "Min") throw RunStateError("run state has invalid model name");
    generation_ = uint_field(payload_, "generation");
    stage_ = stage_from_name(string_field(payload_, "stage"));
    (void)string_field(payload_, "compatibility_namespace");
    const auto& protocols = object_field(payload_, "protocol_ids");
    if (protocols.empty()) throw RunStateError("protocol_ids must not be empty");
    for (const auto& [key, value] : protocols) {
        (void)key;
        const auto* text = std::get_if<std::string>(&value.value);
        if (!text || text->empty()) throw RunStateError("protocol_ids values must be non-empty strings");
    }
    (void)uint_field(payload_, "run_seed");
    (void)object_field(payload_, "compatibility");
    (void)uint_field(payload_, "iteration");
    (void)uint_field(payload_, "training_step");
    optional_string_field(payload_, "champion_checkpoint");
    optional_string_field(payload_, "candidate_checkpoint");
    optional_string_field(payload_, "replay_manifest");
    (void)array_field(payload_, "completed_game_ids");
    (void)array_field(payload_, "promotion_records");
    (void)array_field(payload_, "rating_records");
    const auto& completions = object_field(payload_, "stage_completions");
    const auto stage_index = static_cast<std::size_t>(std::find(kStages.begin(), kStages.end(), stage_) - kStages.begin());
    if (completions.size() != stage_index) throw RunStateError("stage_completions do not match stage");
    for (std::size_t index = 0; index < stage_index; ++index) {
        const auto marker = completions.find(std::string(kStageNames[index]));
        if (marker == completions.end()) throw RunStateError("stage_completions do not match stage");
        const auto* text = std::get_if<std::string>(&marker->second.value);
        if (!text || text->empty()) throw RunStateError("stage completion marker is invalid");
    }
}

RunState RunState::initialize(std::string run_id, Object compatibility,
                              Object protocol_ids, uint64_t run_seed) {
    const auto model_name = string_field(compatibility, "model_name");
    Object identity;
    for (const auto field : {"model_name", "model_version", "player_count", "value_semantics_version"}) {
        const auto found = compatibility.find(field);
        if (found == compatibility.end()) throw RunStateError("compatibility missing identity field");
        identity.emplace(field, found->second);
    }
    const auto compatibility_json = diamond_support::canonical_json(Json{compatibility});
    Object payload{{"candidate_checkpoint", Json{nullptr}},
                   {"champion_checkpoint", Json{nullptr}},
                   {"champion_model_key", Json{nullptr}},
                   {"compatibility", Json{std::move(compatibility)}},
                   {"compatibility_namespace", Json{"sha256:" + diamond_support::sha256(compatibility_json)}},
                   {"completed_game_ids", Json{Array{}}},
                   {"generation", Json{int64_t{0}}},
                   {"iteration", Json{int64_t{0}}},
                   {"model_identity", Json{std::move(identity)}},
                   {"promotion_records", Json{Array{}}},
                   {"protocol_ids", Json{std::move(protocol_ids)}},
                   {"rating_records", Json{Array{}}},
                   {"replay_manifest", Json{nullptr}},
                   {"run_id", Json{std::move(run_id)}},
                   {"run_seed", Json{static_cast<int64_t>(run_seed)}},
                   {"schema_version", Json{int64_t{2}}},
                   {"stage", Json{std::string(stage_name(RunStage::initialize))}},
                   {"stage_completions", Json{Object{}}},
                   {"training_step", Json{int64_t{0}}}};
    (void)model_name;
    return RunState(std::move(payload));
}

RunState RunState::initialize(std::string run_id, std::string model_name,
                              std::string compatibility_namespace,
                              std::string protocol_id, uint64_t run_seed) {
    Object compatibility{{"model_name", Json{model_name}},
                         {"model_version", Json{std::string("1.0.0")}},
                         {"player_count", Json{int64_t{model_name == "Soo" ? 2 : 3}}},
                         {"value_semantics_version", Json{std::string(model_name == "Soo" ? "current-player-scalar-winloss-v1" : "canonical-player-placement-vector-v1")}}};
    Object identity{{"model_name", Json{model_name}},
                    {"model_version", Json{std::string("1.0.0")}},
                    {"player_count", Json{int64_t{model_name == "Soo" ? 2 : 3}}},
                    {"value_semantics_version", Json{std::string(model_name == "Soo" ? "current-player-scalar-winloss-v1" : "canonical-player-placement-vector-v1")}}};
    Object payload{{"candidate_checkpoint", Json{nullptr}},
                   {"champion_checkpoint", Json{nullptr}},
                   {"champion_model_key", Json{nullptr}},
                   {"compatibility", Json{std::move(compatibility)}},
                   {"compatibility_namespace", Json{std::move(compatibility_namespace)}},
                   {"completed_game_ids", Json{Array{}}},
                   {"generation", Json{int64_t{0}}},
                   {"iteration", Json{int64_t{0}}},
                   {"model_identity", Json{std::move(identity)}},
                   {"promotion_records", Json{Array{}}},
                   {"protocol_ids", Json{Object{{"pipeline", Json{std::move(protocol_id)}}}}},
                   {"rating_records", Json{Array{}}},
                   {"replay_manifest", Json{nullptr}},
                   {"run_id", Json{std::move(run_id)}},
                   {"run_seed", Json{static_cast<int64_t>(run_seed)}},
                   {"schema_version", Json{int64_t{2}}},
                   {"stage", Json{std::string(stage_name(RunStage::initialize))}},
                   {"stage_completions", Json{Object{}}},
                   {"training_step", Json{int64_t{0}}}};
    return RunState(std::move(payload));
}

RunState RunState::from_json(Object payload) { return RunState(std::move(payload)); }

uint64_t RunState::derive_seed(const Array& identities) const {
    if (identities.empty()) throw RunStateError("seed derivation requires identities");
    Object seed{{"compatibility_namespace", payload_.at("compatibility_namespace")},
                {"identities", Json{identities}}, {"model_identity", payload_.at("model_identity")},
                {"namespace", Json{std::string("alphadiamond-training-run-seed-v1")}},
                {"protocol_ids", payload_.at("protocol_ids")}, {"run_id", Json{run_id_}},
                {"run_seed", payload_.at("run_seed")}};
    const auto digest = diamond_support::sha256(diamond_support::canonical_json(Json{std::move(seed)}));
    return std::stoull(digest.substr(0, 16), nullptr, 16) & ((uint64_t{1} << 63) - 1);
}

RunState RunState::with_progress(Object changes) const {
    auto payload = payload_;
    for (auto& [field, value] : changes) {
        if (!kProgressFields.contains(field)) throw RunStateError("unsupported run progress field");
        payload[field] = std::move(value);
    }
    return RunState(std::move(payload));
}

std::filesystem::path RunStateStore::state_path(const std::string& model_name, const std::string& run_id) const {
    if (!safe_id(run_id) || (model_name != "Soo" && model_name != "Min"))
        throw RunStateError("invalid run state identity");
    std::string lower = model_name;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return root_ / lower / run_id / "state.json";
}

RunState RunStateStore::initialize(const RunState& state) {
    const auto path = state_path(state.model_name(), state.run_id());
    if (std::filesystem::exists(path)) throw RunStateError("training run already exists: " + state.run_id());
    atomic_write(path, diamond_support::canonical_json(Json{state.payload_}));
    return state;
}

RunState RunStateStore::load(const std::string& model_name, const std::string& run_id) const {
    const auto path = state_path(model_name, run_id);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw RunStateError("training run does not exist: " + run_id);
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    try {
        auto value = diamond_support::parse_json(text);
        auto* object = std::get_if<Object>(&value.value);
        if (!object) throw RunStateError("run state must be a JSON object");
        auto state = RunState::from_json(std::move(*object));
        if (state.model_name() != model_name || state.run_id() != run_id)
            throw RunStateError("training run path does not match its identity");
        return state;
    } catch (const RunStateError&) { throw; }
      catch (const std::exception& error) { throw RunStateError("corrupt training run state: " + std::string(error.what())); }
}

RunState RunStateStore::save(const RunState& state) {
    const auto current = load(state.model_name(), state.run_id());
    if (current.generation() != state.generation()) throw RunStateError("stale training run generation");
    for (const auto field : {"run_id", "model_identity", "compatibility", "compatibility_namespace",
                             "protocol_ids", "run_seed", "schema_version"}) {
        if (diamond_support::canonical_json(state.payload_.at(field)) !=
            diamond_support::canonical_json(current.payload_.at(field)))
            throw RunStateError("immutable training run identity changed");
    }
    if (state.stage() != current.stage() ||
        diamond_support::canonical_json(state.payload_.at("stage_completions")) !=
            diamond_support::canonical_json(current.payload_.at("stage_completions")))
        throw RunStateError("stage changes must use transition");
    auto payload = state.payload_;
    payload["generation"] = Json{static_cast<int64_t>(state.generation() + 1)};
    auto result = RunState::from_json(std::move(payload));
    atomic_write(state_path(result.model_name(), result.run_id()), diamond_support::canonical_json(Json{result.payload_}));
    return result;
}

RunState RunStateStore::transition(const RunState& state, RunStage next, std::string completion_marker,
                                   Object progress) {
    if (completion_marker.empty()) throw RunStateError("completion marker is required");
    const auto current = load(state.model_name(), state.run_id());
    if (current.generation() != state.generation()) throw RunStateError("stale training run generation");
    if (current.stage() != state.stage()) throw RunStateError("stale training run stage");
    if (next != next_stage(current.stage())) throw RunStateError("invalid run stage transition");
    auto payload = current.payload_;
    for (auto& [field, value] : progress) {
        if (!kProgressFields.contains(field)) throw RunStateError("unsupported transition progress field");
        payload[field] = std::move(value);
    }
    auto& completions = std::get<Object>(payload.at("stage_completions").value);
    completions.emplace(std::string(stage_name(current.stage())), Json{std::move(completion_marker)});
    payload["generation"] = Json{static_cast<int64_t>(current.generation() + 1)};
    payload["stage"] = Json{std::string(stage_name(next))};
    auto result = RunState::from_json(std::move(payload));
    atomic_write(state_path(result.model_name(), result.run_id()), diamond_support::canonical_json(Json{result.payload_}));
    return result;
}

RunState RunStateStore::start_next_iteration(const RunState& state) {
    const auto current = load(state.model_name(), state.run_id());
    if (current.generation() != state.generation()) throw RunStateError("stale training run generation");
    if (current.stage() != RunStage::complete) throw RunStateError("only COMPLETE can start next iteration");
    auto payload = current.payload_;
    payload["candidate_checkpoint"] = Json{nullptr};
    payload["completed_game_ids"] = Json{Array{}};
    payload["generation"] = Json{static_cast<int64_t>(current.generation() + 1)};
    payload["iteration"] = Json{static_cast<int64_t>(uint_field(payload, "iteration") + 1)};
    payload["stage"] = Json{std::string(stage_name(RunStage::initialize))};
    payload["stage_completions"] = Json{Object{}};
    auto result = RunState::from_json(std::move(payload));
    atomic_write(state_path(result.model_name(), result.run_id()), diamond_support::canonical_json(Json{result.payload_}));
    return result;
}

}  // namespace diamond_orchestration
