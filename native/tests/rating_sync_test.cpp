#include "diamond_orchestration/rating_store.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <process.h>
#endif

namespace {
using Json = diamond_support::JsonValue;
using Object = Json::Object;
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
void write(const std::filesystem::path& path, const std::string& value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
    if (!output)
        throw std::runtime_error("cannot write fixture");
}
int run(const std::filesystem::path& executable, const std::filesystem::path& protocol,
        const std::filesystem::path& events, const std::filesystem::path& output) {
#ifdef _WIN32
    const std::string executable_text = executable.string();
    const std::string protocol_text = protocol.string();
    const std::string events_text = events.string();
    const std::string output_text = output.string();
    const char* arguments[] = {
        executable_text.c_str(), "--protocol", protocol_text.c_str(), "--events-dir",
        events_text.c_str(),     "--output",   output_text.c_str(),   nullptr};
    return static_cast<int>(_spawnv(_P_WAIT, executable_text.c_str(), arguments));
#else
    return std::system(("\"" + executable.string() + "\" --protocol \"" + protocol.string() +
                        "\" --events-dir \"" + events.string() + "\" --output \"" +
                        output.string() + "\"")
                           .c_str());
#endif
}
const Object& object(const Json& value) {
    return std::get<Object>(value.value);
}
const Json& field(const Object& value, const char* name) {
    return value.at(name);
}
} // namespace

int main(int argc, char** argv) {
    const auto root = std::filesystem::temp_directory_path() /
                      ("alphadiamond-rating-sync-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        require(argc == 2, "rating_sync_test needs sync executable path");
        using namespace diamond_orchestration;
        const auto candidate =
            make_participant_identity(Json{Object{{"checkpoint", Json{"candidate"}}}}, "Candidate");
        const auto baseline =
            make_participant_identity(Json{Object{{"checkpoint", Json{"baseline"}}}}, "Baseline");
        RatingRegistry protocol_registry{"sha256:rating-sync",
                                         EloConfig{1200.0, 24.0, 400.0, "soo-elo-v1"}};
        protocol_registry.add_participant(candidate);
        save_rating_registry(root / "protocol.json", protocol_registry);
        const auto event = make_soo_rating_event(
            12, "sha256:rating-sync", {candidate.participant_id, baseline.participant_id}, {1, 2},
            {1, 2}, "opening", true, candidate.participant_id, baseline.participant_id,
            "stable-game", {candidate, baseline});
        write(root / "events" / (event.event_id.substr(7) + ".json"),
              diamond_support::canonical_json(event.to_json()));
        require(run(argv[1], root / "protocol.json", root / "events", root / "ratings.json") == 0,
                "sync CLI accepts canonical v2 event set");
        std::ifstream input(root / "ratings.json", std::ios::binary);
        const std::string output_text{std::istreambuf_iterator<char>(input), {}};
        const Json output = diamond_support::parse_json(output_text);
        const auto& root_object = object(output);
        require(std::get<int64_t>(field(root_object, "schema_version").value) == 2,
                "output schema v2");
        require(std::get<int64_t>(field(object(field(root_object, "event_set")), "count").value) ==
                    1,
                "output event count");
        const auto& ratings = std::get<Json::Array>(field(root_object, "ratings").value);
        require(ratings.size() == 2, "output has per-model rating rows");
        require(object(ratings.front()).contains("full_identity"), "output retains full identity");
        input.close();

        write(root / "events" / "wrong-name.json",
              diamond_support::canonical_json(event.to_json()));
        require(run(argv[1], root / "protocol.json", root / "events", root / "bad.json") != 0,
                "sync CLI rejects duplicate event under an invalid immutable name");
        std::filesystem::remove_all(root);
    } catch (const std::exception& error) {
        std::cerr << "rating_sync_test: " << error.what() << '\n';
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        return 1;
    }
}
