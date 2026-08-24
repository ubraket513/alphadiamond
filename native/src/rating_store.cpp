#include "diamond_orchestration/rating_store.hpp"

#include <array>
#include <atomic>
#include <fstream>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace diamond_orchestration {
namespace {
using Json = diamond_support::JsonValue;
using Object = Json::Object;
using Array = Json::Array;

const Object& object(const Json& value, const char* field) {
    const auto* result = std::get_if<Object>(&value.value);
    if (!result) throw RatingError(std::string(field) + " must be an object");
    return *result;
}
const Json& field(const Object& object, const char* name) {
    const auto found = object.find(name);
    if (found == object.end()) throw RatingError(std::string("rating registry is missing ") + name);
    return found->second;
}
const std::string& text(const Json& value, const char* name) {
    const auto* result = std::get_if<std::string>(&value.value);
    if (!result || result->empty()) throw RatingError(std::string("rating registry has invalid ") + name);
    return *result;
}
int integer(const Json& value, const char* name) {
    const auto* result = std::get_if<int64_t>(&value.value);
    if (!result) throw RatingError(std::string("rating registry has invalid ") + name);
    return static_cast<int>(*result);
}
uint64_t index(const Json& value) {
    const auto* result = std::get_if<int64_t>(&value.value);
    if (!result || *result < 0) throw RatingError("rating registry has invalid sequence_index");
    return static_cast<uint64_t>(*result);
}
bool completed(const Json& value) {
    const auto* result = std::get_if<bool>(&value.value);
    if (!result) throw RatingError("rating registry has invalid completed");
    return *result;
}
const Array& array(const Json& value, const char* name) {
    const auto* result = std::get_if<Array>(&value.value);
    if (!result) throw RatingError(std::string("rating registry has invalid ") + name);
    return *result;
}
template <typename T, std::size_t Count>
std::array<T, Count> fixed(const Json& value, const char* name) {
    const auto& source = array(value, name);
    if (source.size() != Count) throw RatingError(std::string("rating registry has invalid ") + name);
    std::array<T, Count> result{};
    for (std::size_t i = 0; i < Count; ++i) {
        if constexpr (std::is_same_v<T, std::string>) result[i] = text(source[i], name);
        else result[i] = integer(source[i], name);
    }
    return result;
}
std::string nullable_text(const Json& value, const char* name) {
    if (std::holds_alternative<std::nullptr_t>(value.value)) return {};
    return text(value, name);
}
void atomic_write(const std::filesystem::path& path, const std::string& contents) {
    static std::atomic_uint64_t sequence{0};
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp." + std::to_string(++sequence);
    { std::ofstream output(temporary, std::ios::binary | std::ios::trunc); if (!output) throw RatingError("cannot write rating registry"); output << contents; if (!output) throw RatingError("cannot write rating registry"); }
#ifdef _WIN32
    if (!MoveFileExW(std::filesystem::path(temporary).wstring().c_str(), path.wstring().c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { std::filesystem::remove(temporary); throw RatingError("cannot activate rating registry"); }
#else
    std::error_code error; std::filesystem::rename(temporary, path, error); if (error) { std::filesystem::remove(temporary); throw RatingError("cannot activate rating registry"); }
#endif
}
}  // namespace

void save_rating_registry(const std::filesystem::path& path, const RatingRegistry& registry) {
    Array events;
    for (const auto& event : registry.events()) std::visit([&](const auto& value) { events.emplace_back(value.to_json()); }, event);
    Object payload{{"events", Json{std::move(events)}}, {"registry", registry.report_json()}, {"schema_version", Json{int64_t{1}}}};
    atomic_write(path, diamond_support::canonical_json(Json{std::move(payload)}));
}

RatingRegistry load_rating_registry(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw RatingError("cannot open rating registry: " + path.string());
    const std::string contents((std::istreambuf_iterator<char>(input)), {});
    const auto root = object(diamond_support::parse_json(contents), "rating registry");
    const auto* version = std::get_if<int64_t>(&field(root, "schema_version").value);
    if (!version || *version != 1) throw RatingError("unsupported rating registry schema version");
    const auto& report = object(field(root, "registry"), "registry");
    const auto& family = text(field(report, "family"), "family");
    RatingRegistry registry = family == "soo" ? RatingRegistry(text(field(report, "protocol_id"), "protocol_id"))
                                                : RatingRegistry(text(field(report, "protocol_id"), "protocol_id"), TrueSkillConfig{});
    if (family != "soo" && family != "min") throw RatingError("rating registry has invalid family");
    for (const auto& row : array(field(report, "leaderboard"), "leaderboard")) {
        const auto& participant = object(row, "leaderboard entry");
        registry.add_participant(text(field(participant, "participant_id"), "participant_id"), text(field(participant, "display_name"), "display_name"));
    }
    for (const auto& encoded : array(field(root, "events"), "events")) {
        const auto& event = object(encoded, "event");
        const auto& type = text(field(event, "event_type"), "event_type");
        if (type == "soo") {
            const bool done = completed(field(event, "completed"));
            auto value = make_soo_rating_event(index(field(event, "sequence_index")), text(field(event, "protocol_id"), "protocol_id"), fixed<std::string, 2>(field(event, "participant_ids"), "participant_ids"), fixed<int, 2>(field(event, "seat_assignment"), "seat_assignment"), fixed<int, 2>(field(event, "turn_order"), "turn_order"), text(field(event, "opening_id"), "opening_id"), done, nullable_text(field(event, "winner_id"), "winner_id"), nullable_text(field(event, "loser_id"), "loser_id"));
            if (value.event_id != text(field(event, "event_id"), "event_id")) throw RatingError("Soo rating event ID is corrupt");
            registry.record_event(value);
        } else if (type == "min") {
            const bool done = completed(field(event, "completed"));
            std::array<std::string, 3> ranking{};
            if (done) ranking = fixed<std::string, 3>(field(event, "final_ranking"), "final_ranking");
            auto value = make_min_rating_event(index(field(event, "sequence_index")), text(field(event, "protocol_id"), "protocol_id"), fixed<std::string, 3>(field(event, "participant_ids"), "participant_ids"), fixed<int, 3>(field(event, "seat_assignment"), "seat_assignment"), fixed<int, 3>(field(event, "turn_order"), "turn_order"), text(field(event, "opening_id"), "opening_id"), done, ranking);
            if (value.event_id != text(field(event, "event_id"), "event_id")) throw RatingError("Min rating event ID is corrupt");
            registry.record_event(value);
        } else throw RatingError("rating registry has unknown event type");
    }
    return registry;
}

}  // namespace diamond_orchestration
