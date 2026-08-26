#include "diamond_orchestration/rating_store.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <fstream>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
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
const Json* optional_field(const Object& object, const char* name) {
    const auto found = object.find(name);
    return found == object.end() ? nullptr : &found->second;
}
const std::string& text(const Json& value, const char* name) {
    const auto* result = std::get_if<std::string>(&value.value);
    if (!result || result->empty()) throw RatingError(std::string("rating registry has invalid ") + name);
    return *result;
}
std::string optional_text(const Object& object, const char* name) {
    const auto* value = optional_field(object, name);
    return value ? text(*value, name) : std::string{};
}
double number(const Json& value, const char* name) {
    if (const auto* result = std::get_if<double>(&value.value))
        return *result;
    if (const auto* result = std::get_if<int64_t>(&value.value))
        return static_cast<double>(*result);
    throw RatingError(std::string("rating registry has invalid ") + name);
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
std::vector<ParticipantIdentity> embedded_identities(const Object& event) {
    const auto* encoded = optional_field(event, "participant_identities");
    if (!encoded)
        return {};
    std::vector<ParticipantIdentity> result;
    for (const auto& value : array(*encoded, "participant_identities")) {
        const auto& identity = object(value, "participant identity");
        ParticipantIdentity participant{text(field(identity, "participant_id"), "participant_id"),
                                        text(field(identity, "display_name"), "display_name"),
                                        field(identity, "full_identity")};
        participant.validate();
        result.push_back(std::move(participant));
    }
    return result;
}
std::string nullable_text(const Json& value, const char* name) {
    if (std::holds_alternative<std::nullptr_t>(value.value)) return {};
    return text(value, name);
}
std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw RatingError("cannot open rating store artifact: " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}
void flush_file(const std::filesystem::path& path) {
#ifdef _WIN32
    const HANDLE handle =
        CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE || !FlushFileBuffers(handle)) {
        if (handle != INVALID_HANDLE_VALUE)
            CloseHandle(handle);
        throw RatingError("cannot flush rating store artifact");
    }
    CloseHandle(handle);
#else
    const int handle = ::open(path.c_str(), O_RDONLY);
    if (handle < 0 || ::fsync(handle) != 0) {
        if (handle >= 0)
            ::close(handle);
        throw RatingError("cannot flush rating store artifact");
    }
    ::close(handle);
#endif
}
void flush_directory(const std::filesystem::path& path) {
#ifndef _WIN32
    const int handle = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (handle >= 0) {
        (void)::fsync(handle);
        ::close(handle);
    }
#else
    (void)path;
#endif
}
void atomic_write(const std::filesystem::path& path, const std::string& contents) {
    static std::atomic_uint64_t sequence{0};
    std::filesystem::create_directories(path.parent_path());
    const auto process =
#ifdef _WIN32
        static_cast<unsigned long long>(GetCurrentProcessId());
#else
        static_cast<unsigned long long>(::getpid());
#endif
    const auto temporary = path.parent_path() / (".rating-tmp-" + std::to_string(process) + "-" +
                                                 std::to_string(++sequence));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            throw RatingError("cannot write rating registry");
        output << contents;
        output.flush();
        if (!output)
            throw RatingError("cannot write rating registry");
    }
    try {
        flush_file(temporary);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
#ifdef _WIN32
    if (!MoveFileExW(temporary.wstring().c_str(), path.wstring().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::filesystem::remove(temporary);
        throw RatingError("cannot activate rating registry");
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw RatingError("cannot activate rating registry");
    }
    flush_directory(path.parent_path());
#endif
}
std::function<void(RatingStoreFailpoint)> failpoint;
void trigger(RatingStoreFailpoint point) {
    if (failpoint)
        failpoint(point);
}
std::string event_file_name(const std::string& event_id) {
    constexpr std::string_view prefix = "sha256:";
    if (!event_id.starts_with(prefix) || event_id.size() != prefix.size() + 64 ||
        !std::all_of(event_id.begin() + static_cast<std::ptrdiff_t>(prefix.size()), event_id.end(),
                     [](unsigned char c) { return std::isxdigit(c) != 0; }))
        throw RatingError("rating event ID is not a SHA-256 identity");
    return event_id.substr(prefix.size()) + ".json";
}
void write_new_file(const std::filesystem::path& path, const std::string& contents) {
#ifdef _WIN32
    const HANDLE handle = CreateFileW(path.wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        if (GetLastError() == ERROR_FILE_EXISTS || GetLastError() == ERROR_ALREADY_EXISTS)
            throw RatingError("rating store artifact already exists");
        throw RatingError("cannot create rating store artifact");
    }
    DWORD written = 0;
    const bool ok = WriteFile(handle, contents.data(), static_cast<DWORD>(contents.size()),
                              &written, nullptr) &&
                    written == contents.size() && FlushFileBuffers(handle);
    CloseHandle(handle);
    if (!ok) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        throw RatingError("cannot write rating store artifact");
    }
#else
    const int handle = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (handle < 0) {
        if (errno == EEXIST)
            throw RatingError("rating store artifact already exists");
        throw RatingError("cannot create rating store artifact");
    }
    const auto written = ::write(handle, contents.data(), contents.size());
    const bool ok = written == static_cast<ssize_t>(contents.size()) && ::fsync(handle) == 0;
    ::close(handle);
    if (!ok) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        throw RatingError("cannot write rating store artifact");
    }
#endif
}
EloConfig elo_config(const Object& config) {
    const auto& value = object(field(config, "elo"), "Elo protocol config");
    EloConfig result{number(field(value, "initial_rating"), "initial_rating"),
                     number(field(value, "k_factor"), "k_factor"),
                     number(field(value, "logistic_scale"), "logistic_scale"),
                     text(field(value, "rating_system_version"), "rating_system_version")};
    result.validate();
    return result;
}
TrueSkillConfig trueskill_config(const Object& config) {
    const auto& value = object(field(config, "trueskill"), "TrueSkill protocol config");
    TrueSkillConfig result{number(field(value, "mu"), "mu"),
                           number(field(value, "sigma"), "sigma"),
                           number(field(value, "beta"), "beta"),
                           number(field(value, "tau"), "tau"),
                           number(field(value, "draw_probability"), "draw_probability"),
                           text(field(value, "rating_system_version"), "rating_system_version")};
    result.validate();
    return result;
}
}  // namespace

void save_rating_registry(const std::filesystem::path& path, const RatingRegistry& registry) {
    Array events;
    for (const auto& event : registry.events()) std::visit([&](const auto& value) { events.emplace_back(value.to_json()); }, event);
    Object payload{{"events", Json{std::move(events)}},
                   {"registry", registry.report_json()},
                   {"schema_version", Json{int64_t{2}}}};
    atomic_write(path, diamond_support::canonical_json(Json{std::move(payload)}));
}

RatingRegistry load_rating_registry(const std::filesystem::path& path) {
    const auto root = object(diamond_support::parse_json(read_bytes(path)), "rating registry");
    const auto* version = std::get_if<int64_t>(&field(root, "schema_version").value);
    if (!version || (*version != 1 && *version != 2))
        throw RatingError("unsupported rating registry schema version");
    const auto& report = object(field(root, "registry"), "registry");
    const auto& family = text(field(report, "family"), "family");
    if (family != "soo" && family != "min") throw RatingError("rating registry has invalid family");
    const auto& protocol = text(field(report, "protocol_id"), "protocol_id");
    RatingRegistry registry = [&]() {
        if (*version == 1)
            return family == "soo" ? RatingRegistry(protocol)
                                   : RatingRegistry(protocol, TrueSkillConfig{});
        const auto& config = object(field(report, "protocol_config"), "protocol config");
        const auto* config_version = std::get_if<int64_t>(&field(config, "schema_version").value);
        if (!config_version || *config_version != 2)
            throw RatingError("unsupported rating protocol config");
        return family == "soo" ? RatingRegistry(protocol, elo_config(config))
                               : RatingRegistry(protocol, trueskill_config(config));
    }();
    const Json* participant_rows = *version == 2 ? optional_field(report, "participants") : nullptr;
    const auto& rows = participant_rows ? array(*participant_rows, "participants")
                                        : array(field(report, "leaderboard"), "leaderboard");
    for (const auto& row : rows) {
        const auto& participant = object(row, "participant");
        const auto& id = text(field(participant, "participant_id"), "participant_id");
        const auto& name = text(field(participant, "display_name"), "display_name");
        const auto* identity = optional_field(participant, "full_identity");
        if (identity && !std::holds_alternative<std::nullptr_t>(identity->value))
            registry.add_participant(ParticipantIdentity{id, name, *identity});
        else
            registry.add_participant(id, name);
    }
    for (const auto& encoded : array(field(root, "events"), "events")) {
        const auto& event = object(encoded, "event");
        const auto& type = text(field(event, "event_type"), "event_type");
        const auto game_id = optional_text(event, "game_id");
        const auto sequence = game_id.empty() ? index(field(event, "sequence_index")) : uint64_t{0};
        auto identities = embedded_identities(event);
        for (const auto& identity : identities)
            registry.add_participant(identity);
        if (type == "soo") {
            const bool done = completed(field(event, "completed"));
            auto value = make_soo_rating_event(
                sequence, text(field(event, "protocol_id"), "protocol_id"),
                fixed<std::string, 2>(field(event, "participant_ids"), "participant_ids"),
                fixed<int, 2>(field(event, "seat_assignment"), "seat_assignment"),
                fixed<int, 2>(field(event, "turn_order"), "turn_order"),
                text(field(event, "opening_id"), "opening_id"), done,
                nullable_text(field(event, "winner_id"), "winner_id"),
                nullable_text(field(event, "loser_id"), "loser_id"), game_id,
                std::move(identities));
            if (value.event_id != text(field(event, "event_id"), "event_id")) throw RatingError("Soo rating event ID is corrupt");
            registry.record_event(value);
        } else if (type == "min") {
            const bool done = completed(field(event, "completed"));
            std::array<std::string, 3> ranking{};
            if (done) ranking = fixed<std::string, 3>(field(event, "final_ranking"), "final_ranking");
            auto value = make_min_rating_event(
                sequence, text(field(event, "protocol_id"), "protocol_id"),
                fixed<std::string, 3>(field(event, "participant_ids"), "participant_ids"),
                fixed<int, 3>(field(event, "seat_assignment"), "seat_assignment"),
                fixed<int, 3>(field(event, "turn_order"), "turn_order"),
                text(field(event, "opening_id"), "opening_id"), done, ranking, game_id,
                std::move(identities));
            if (value.event_id != text(field(event, "event_id"), "event_id")) throw RatingError("Min rating event ID is corrupt");
            registry.record_event(value);
        } else throw RatingError("rating registry has unknown event type");
    }
    return registry;
}

void set_rating_store_failpoint_for_testing(std::function<void(RatingStoreFailpoint)> value) {
    failpoint = std::move(value);
}

RatingEventOutbox::RatingEventOutbox(std::filesystem::path root) : root_(std::move(root)) {
    if (root_.empty())
        throw RatingError("rating outbox root must be non-empty");
}

namespace {
bool publish_event(const std::filesystem::path& root, const Json& payload) {
    const auto& event = object(payload, "rating event");
    const auto& event_id = text(field(event, "event_id"), "event_id");
    const auto path = root / "events" / event_file_name(event_id);
    const auto bytes = diamond_support::canonical_json(payload);
    std::filesystem::create_directories(path.parent_path());
    if (std::filesystem::exists(path)) {
        if (read_bytes(path) == bytes)
            return false;
        throw RatingError("rating event ID conflicts with different immutable bytes");
    }
    trigger(RatingStoreFailpoint::before_event_commit);
    try {
        write_new_file(path, bytes);
        flush_directory(path.parent_path());
        return true;
    } catch (const RatingError& error) {
        if (std::string(error.what()) != "rating store artifact already exists")
            throw;
        if (read_bytes(path) == bytes)
            return false;
        throw RatingError("rating event ID conflicts with different immutable bytes");
    }
}
} // namespace

bool RatingEventOutbox::publish(const SooRatingEvent& event) {
    return publish_event(root_, event.to_json());
}
bool RatingEventOutbox::publish(const MinRatingEvent& event) {
    return publish_event(root_, event.to_json());
}

void RatingEventOutbox::write_receipt(const std::string& event_id, Json receipt) {
    trigger(RatingStoreFailpoint::before_receipt_commit);
    atomic_write(root_ / "receipts" / event_file_name(event_id),
                 diamond_support::canonical_json(std::move(receipt)));
}

std::vector<Json> RatingEventOutbox::pending_events() const {
    const auto events = root_ / "events";
    const auto receipts = root_ / "receipts";
    std::vector<std::filesystem::path> paths;
    if (std::filesystem::exists(events))
        for (const auto& item : std::filesystem::directory_iterator(events))
            if (item.is_regular_file() && item.path().extension() == ".json" &&
                !std::filesystem::exists(receipts / item.path().filename()))
                paths.push_back(item.path());
    std::sort(paths.begin(), paths.end());
    std::vector<Json> result;
    for (const auto& path : paths)
        result.emplace_back(diamond_support::parse_json(read_bytes(path)));
    return result;
}

}  // namespace diamond_orchestration
