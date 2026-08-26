#include "diamond_orchestration/rating_store.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json = diamond_support::JsonValue;
using Object = Json::Object;
using Array = Json::Array;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("rating-sync: " + message);
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        fail("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        fail("cannot write " + path.string());
    output << text;
    output.flush();
    if (!output)
        fail("cannot write " + path.string());
}

const Object& object(const Json& value, const char* what) {
    const auto* result = std::get_if<Object>(&value.value);
    if (!result)
        fail(std::string(what) + " must be an object");
    return *result;
}

const Array& array(const Json& value, const char* what) {
    const auto* result = std::get_if<Array>(&value.value);
    if (!result)
        fail(std::string(what) + " must be an array");
    return *result;
}

const Json& required(const Object& value, const char* name) {
    const auto found = value.find(name);
    if (found == value.end())
        fail(std::string("missing ") + name);
    return found->second;
}

const std::string& text(const Json& value, const char* what) {
    const auto* result = std::get_if<std::string>(&value.value);
    if (!result || result->empty())
        fail(std::string(what) + " must be a non-empty string");
    return *result;
}

bool is_event_file_name(const std::filesystem::path& path, const std::string& event_id) {
    constexpr std::string_view prefix = "sha256:";
    return event_id.starts_with(prefix) && event_id.size() == prefix.size() + 64 &&
           path.filename() == event_id.substr(prefix.size()) + ".json";
}

std::string temporary_name(std::string_view stem) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return "." + std::string(stem) + "." + std::to_string(stamp) + ".json";
}

struct EventSet final {
    Array events;
    Array ids;
};

EventSet load_events(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory))
        fail("events directory does not exist: " + directory.string());
    std::vector<std::filesystem::path> paths;
    for (const auto& item : std::filesystem::directory_iterator(directory)) {
        if (!item.is_regular_file())
            continue;
        if (item.path().extension() != ".json")
            fail("events directory contains a non-JSON file");
        paths.push_back(item.path());
    }
    std::sort(paths.begin(), paths.end());

    std::map<std::string, std::string> bytes_by_id;
    std::map<std::string, Json> event_by_id;
    for (const auto& path : paths) {
        const Json event = diamond_support::parse_json(read_text(path));
        const auto& row = object(event, "event file");
        const std::string id = text(required(row, "event_id"), "event_id");
        if (!is_event_file_name(path, id))
            fail("event filename must be its SHA-256 event ID");
        const std::string bytes = diamond_support::canonical_json(event);
        const auto [existing, inserted] = bytes_by_id.emplace(id, bytes);
        if (!inserted && existing->second != bytes)
            fail("conflicting duplicate event ID: " + id);
        if (!inserted)
            fail("duplicate event ID: " + id);
        event_by_id.emplace(id, event);
    }

    EventSet result;
    for (const auto& [id, event] : event_by_id) {
        result.ids.emplace_back(id);
        result.events.emplace_back(event);
    }
    return result;
}

Json make_output(const diamond_orchestration::RatingRegistry& registry, const EventSet& events) {
    const Json report = registry.report_json();
    const auto& report_object = object(report, "rating report");
    const std::string& family = text(required(report_object, "family"), "family");
    const auto& participants = array(required(report_object, "participants"), "participants");
    const auto& leaderboard = array(required(report_object, "leaderboard"), "leaderboard");
    std::map<std::string, const Object*> ratings;
    for (const auto& row : leaderboard) {
        const auto& rating = object(row, "leaderboard entry");
        ratings.emplace(text(required(rating, "participant_id"), "participant_id"), &rating);
    }

    Array rows;
    for (const auto& row : participants) {
        const auto& participant = object(row, "participant");
        const std::string& id = text(required(participant, "participant_id"), "participant_id");
        const auto found = ratings.find(id);
        if (found == ratings.end())
            fail("participant is absent from leaderboard");
        Object merged{{"display_name", required(participant, "display_name")},
                      {"full_identity", required(participant, "full_identity")},
                      {"participant_id", required(participant, "participant_id")}};
        if (family == "soo") {
            merged.emplace("elo", required(*found->second, "rating"));
            merged.emplace("games", required(*found->second, "games"));
        } else if (family == "min") {
            for (const char* name : {"exposure", "mu", "rated_games", "sigma"})
                merged.emplace(name, required(*found->second, name));
        } else
            fail("registry reported an unknown family");
        rows.emplace_back(std::move(merged));
    }

    const std::string digest =
        "sha256:" + diamond_support::sha256(diamond_support::canonical_json(Json{events.ids}));
    Object protocol{{"config", required(report_object, "protocol_config")},
                    {"family", required(report_object, "family")},
                    {"id", required(report_object, "protocol_id")}};
    return Json{Object{
        {"event_set", Json{Object{{"count", Json{static_cast<int64_t>(events.events.size())}},
                                  {"sha256", Json{digest}}}}},
        {"protocol", Json{std::move(protocol)}},
        {"ratings", Json{std::move(rows)}},
        {"schema_version", Json{int64_t{2}}}}};
}

void promote_atomically(const std::filesystem::path& temporary,
                        const std::filesystem::path& destination) {
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error)
        fail("cannot atomically promote " + destination.string() + ": " + error.message());
}

void usage() {
    std::cerr << "usage: alphadiamond-rating-sync --protocol <protocol-v2.json> --events-dir "
                 "<events> --output <ratings.json>\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path protocol;
        std::filesystem::path events_directory;
        std::filesystem::path output;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if ((argument == "--protocol" || argument == "--events-dir" ||
                 argument == "--output") &&
                index + 1 < argc) {
                const std::filesystem::path value = argv[++index];
                if (argument == "--protocol")
                    protocol = value;
                else if (argument == "--events-dir")
                    events_directory = value;
                else
                    output = value;
            } else {
                usage();
                return 2;
            }
        }
        if (protocol.empty() || events_directory.empty() || output.empty()) {
            usage();
            return 2;
        }
        if (output.parent_path().empty())
            output = std::filesystem::current_path() / output;
        std::filesystem::create_directories(output.parent_path());

        const Json protocol_document = diamond_support::parse_json(read_text(protocol));
        const auto& protocol_root = object(protocol_document, "protocol");
        const auto version = std::get_if<int64_t>(&required(protocol_root, "schema_version").value);
        if (!version || *version != 2)
            fail("protocol schema_version must be 2");
        const Object& protocol_object =
            protocol_root.contains("registry")
                ? object(required(protocol_root, "registry"), "protocol registry")
                : protocol_root;
        (void)text(required(protocol_object, "family"), "family");
        (void)text(required(protocol_object, "protocol_id"), "protocol_id");
        (void)object(required(protocol_object, "protocol_config"), "protocol_config");
        (void)array(required(protocol_object, "participants"), "participants");
        const EventSet events = load_events(events_directory);

        const auto snapshot = output.parent_path() / temporary_name("rating-sync-input");
        const auto staged_output = output.parent_path() / temporary_name("rating-sync-output");
        try {
            write_text(snapshot, diamond_support::canonical_json(
                                     Json{Object{{"events", Json{events.events}},
                                                 {"registry", Json{protocol_object}},
                                                 {"schema_version", Json{int64_t{2}}}}}));
            auto registry = diamond_orchestration::load_rating_registry(snapshot);
            registry.rebuild();
            write_text(staged_output,
                       diamond_support::canonical_json(make_output(registry, events)) + "\n");
            std::filesystem::remove(snapshot);
            promote_atomically(staged_output, output);
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(snapshot, ignored);
            std::filesystem::remove(staged_output, ignored);
            throw;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
