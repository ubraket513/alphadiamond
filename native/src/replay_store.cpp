#include "diamond_pipeline/replay_store.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
// windows.h defines min/max as macros, which mangles std::min at its call site.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "diamond_support/json.hpp"
#include "diamond_pipeline/replay_segment.hpp"

namespace diamond_pipeline {
namespace {
using diamond_support::JsonValue;
using Object = JsonValue::Object;
using Array = JsonValue::Array;
enum class ReplayChunkEncoding { json_v1, binary_v1 };

const char* encoding_name(ReplayChunkEncoding encoding) {
    return encoding == ReplayChunkEncoding::binary_v1 ? "binary-v1" : "json-v1";
}

ReplayChunkEncoding encoding_from_string(const std::string& value) {
    if (value == "json-v1") return ReplayChunkEncoding::json_v1;
    if (value == "binary-v1") return ReplayChunkEncoding::binary_v1;
    throw std::runtime_error("replay chunk encoding " + value + " is unsupported");
}

std::string_view to_string_view(const std::vector<std::byte>& bytes) {
    return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::filesystem::path stream_path(const std::filesystem::path& path);

std::vector<std::byte> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream source(stream_path(path), std::ios::binary | std::ios::ate);
    if (!source)
        throw std::runtime_error("missing replay chunk");
    const auto offset = source.tellg();
    if (offset < 0)
        throw std::runtime_error("cannot read replay chunk");
    std::vector<std::byte> payload(static_cast<size_t>(offset));
    source.seekg(0, std::ios::beg);
    source.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!source)
        throw std::runtime_error("cannot read replay chunk");
    return payload;
}

JsonValue json_string_array(const std::vector<int32_t>& values) { Array out; for (auto value : values) out.emplace_back(JsonValue{int64_t(value)}); return JsonValue{std::move(out)}; }
JsonValue sample_json(const TrainingSample& sample) {
    Array features; const size_t width = sample.node_features.empty() ? 0 : sample.node_features.size() / 73;
    if (width == 0 || width * 73 != sample.node_features.size()) throw std::invalid_argument("replay sample feature shape is invalid");
    for (size_t row=0; row<73; ++row) { Array values; for(size_t col=0; col<width; ++col) values.emplace_back(JsonValue{double(sample.node_features[row*width+col])}); features.emplace_back(JsonValue{std::move(values)}); }
    Array policy; for (const auto& [action, probability] : sample.sparse_policy) policy.emplace_back(JsonValue{Array{JsonValue{int64_t(action)}, JsonValue{double(probability)}}});
    Array targets; for (float value : sample.value_target) targets.emplace_back(JsonValue{double(value)});
    return JsonValue{Object{{"canonical_player_ids",json_string_array(sample.canonical_player_ids)}, {"node_features",JsonValue{std::move(features)}}, {"schema_version",JsonValue{int64_t(1)}}, {"sparse_policy",JsonValue{std::move(policy)}}, {"value_target",JsonValue{std::move(targets)}}}};
}
JsonValue compatibility_json(const Compatibility& compatibility) {
    compatibility.validate();
    return JsonValue{Object{
        {"action_space_version", JsonValue{compatibility.action_space_version}},
        {"board_topology_version", JsonValue{compatibility.board_topology_version}},
        {"encoder_version", JsonValue{compatibility.encoder_version}},
        {"model_name", JsonValue{compatibility.model_name}},
        {"model_version", JsonValue{compatibility.model_version}},
        {"network_config", JsonValue{Object{{"residual_blocks", JsonValue{compatibility.network_config.residual_blocks}}, {"width", JsonValue{compatibility.network_config.width}}}}},
        {"player_count", JsonValue{compatibility.player_count}},
        {"ruleset_fingerprint", JsonValue{compatibility.ruleset_fingerprint}},
        {"ruleset_version", JsonValue{compatibility.ruleset_version}},
        {"seat_layout_version", JsonValue{compatibility.seat_layout_version}},
        {"value_semantics_version", JsonValue{compatibility.value_semantics_version}},
    }};
}
const Object& object(const JsonValue& value, const char* what) { if (const auto* found=std::get_if<Object>(&value.value)) return *found; throw std::runtime_error(std::string(what)+" must be an object"); }
const Array& array(const JsonValue& value, const char* what) { if (const auto* found=std::get_if<Array>(&value.value)) return *found; throw std::runtime_error(std::string(what)+" must be an array"); }
const JsonValue& field(const Object& value, const char* name) { const auto found=value.find(name); if(found==value.end()) throw std::runtime_error(std::string("missing ")+name); return found->second; }
std::string string(const JsonValue& value, const char* what) { if(const auto* found=std::get_if<std::string>(&value.value)) return *found; throw std::runtime_error(std::string(what)+" must be string"); }
int64_t integer(const JsonValue& value, const char* what) { if(const auto* found=std::get_if<int64_t>(&value.value)) return *found; throw std::runtime_error(std::string(what)+" must be integer"); }
double number(const JsonValue& value, const char* what) { if(const auto* i=std::get_if<int64_t>(&value.value)) return double(*i); if(const auto* d=std::get_if<double>(&value.value)) return *d; throw std::runtime_error(std::string(what)+" must be number"); }
Compatibility compatibility_from_json(const JsonValue& value) {
    const auto& row = object(value, "compatibility");
    static const std::array<std::string, 11> keys = {
        "action_space_version", "board_topology_version", "encoder_version", "model_name",
        "model_version", "network_config", "player_count", "ruleset_fingerprint",
        "ruleset_version", "seat_layout_version", "value_semantics_version"};
    if (row.size() != keys.size()) throw std::runtime_error("replay compatibility key set is invalid");
    for (const auto& key : keys) (void)field(row, key.c_str());
    const auto& network = object(field(row, "network_config"), "network config");
    if (network.size() != 2) throw std::runtime_error("replay network config key set is invalid");
    Compatibility out{
        string(field(row, "model_name"), "model name"),
        string(field(row, "model_version"), "model version"),
        integer(field(row, "player_count"), "player count"),
        string(field(row, "ruleset_version"), "ruleset version"),
        string(field(row, "board_topology_version"), "topology version"),
        string(field(row, "ruleset_fingerprint"), "ruleset fingerprint"),
        string(field(row, "encoder_version"), "encoder version"),
        string(field(row, "action_space_version"), "action space version"),
        string(field(row, "seat_layout_version"), "seat layout version"),
        string(field(row, "value_semantics_version"), "value semantics version"),
        {integer(field(network, "residual_blocks"), "network residual blocks"),
         integer(field(network, "width"), "network width")}};
    try { out.validate(); } catch (const std::invalid_argument& error) { throw std::runtime_error(error.what()); }
    return out;
}
TrainingSample legacy_sample(const JsonValue& value, const Compatibility& compatibility) { const auto& row=object(value,"sample"); TrainingSample sample; sample.compatibility=compatibility; for(const auto& feature:array(field(row,"node_features"),"features")) for(const auto& item:array(feature,"feature row")) sample.node_features.push_back(float(number(item,"feature"))); for(const auto& player:array(field(row,"canonical_player_ids"),"players")) sample.canonical_player_ids.push_back(int32_t(integer(player,"player"))); for(const auto& pair:array(field(row,"sparse_policy"),"policy")){const auto& values=array(pair,"policy row"); if(values.size()!=2) throw std::runtime_error("policy row width");sample.sparse_policy.emplace_back(int32_t(integer(values[0],"action")),float(number(values[1],"probability")));} for(const auto& target:array(field(row,"value_target"),"targets")) sample.value_target.push_back(float(number(target,"target"))); return sample; }
uint64_t next_splitmix(uint64_t& state) { state += 0x9e3779b97f4a7c15ULL; uint64_t value=state; value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL; value=(value^(value>>27))*0x94d049bb133111ebULL; return value^(value>>31); }
bool failure_injected(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0)
        return false;
    const bool present = value != nullptr && *value != '\0';
    std::free(value);
    return present;
#else
    const auto* value = std::getenv(name);
    return value != nullptr && *value != '\0';
#endif
}
// What the store needs to remember about an ingested episode once its samples
// are in the sampling pool and its chunk is on disk: identity, the duplicate
// -detection fields, and the sample count the manifest reports.  Deliberately
// not the samples themselves -- retaining a second copy of every sample
// alongside the pool doubled the store's resident cost for data no caller
// reads back.
struct EpisodeRecord {
    std::string game_id;
    std::string retry_id;
    std::string aborted_reason;
    uint64_t seed = 0;
    uint64_t move_count = 0;
    size_t sample_count = 0;
    bool completed = true;
};

JsonValue chunk_body(const Episode& episode, const Compatibility& compatibility) { Array rows; for (const auto& sample : episode.samples) rows.push_back(sample_json(sample)); return JsonValue{Object{{"compatibility",compatibility_json(compatibility)}, {"episode",JsonValue{Object{{"completed",JsonValue{episode.completed}}, {"game_id",JsonValue{episode.game_id}}, {"move_count",JsonValue{int64_t(episode.move_count)}}, {"retry_id",JsonValue{episode.retry_id}}, {"seed",JsonValue{int64_t(episode.seed)}}}}}, {"samples",JsonValue{std::move(rows)}}, {"schema_version",JsonValue{int64_t(1)}}}}; }
#ifdef _WIN32
std::wstring extended_windows_path(const std::filesystem::path& path) {
    std::wstring value = std::filesystem::absolute(path).wstring();
    if (value.rfind(L"\\\\?\\", 0) == 0)
        return value;
    if (value.rfind(L"\\\\", 0) == 0)
        return L"\\\\?\\UNC\\" + value.substr(2);
    return L"\\\\?\\" + value;
}

void remove_file(const std::filesystem::path& path, std::error_code& error) {
    if (DeleteFileW(extended_windows_path(path).c_str())) {
        error.clear();
        return;
    }
    const DWORD code = GetLastError();
    if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND) {
        error.clear();
        return;
    }
    error = std::error_code(static_cast<int>(code), std::system_category());
}
#else
void remove_file(const std::filesystem::path& path, std::error_code& error) {
    std::filesystem::remove(path, error);
}
#endif
std::filesystem::path stream_path(const std::filesystem::path& path) {
#ifdef _WIN32
    return std::filesystem::path(extended_windows_path(path));
#else
    return path;
#endif
}
void atomic_write(const std::filesystem::path& path, const std::string& contents) {
    static std::atomic_uint64_t sequence{0};
    std::filesystem::create_directories(path.parent_path());
    const auto process_id =
#ifdef _WIN32
        static_cast<unsigned long long>(GetCurrentProcessId());
#else
        static_cast<unsigned long long>(::getpid());
#endif
    // Keep the transaction leaf short.  Appending a suffix to a maximum-length
    // destination leaf can exceed Windows' MAX_PATH before the write starts.
    const std::filesystem::path temporary=path.parent_path()/(".replay-tmp-"+std::to_string(process_id)+"-"+std::to_string(++sequence));
    {
        std::ofstream file(stream_path(temporary), std::ios::binary | std::ios::trunc);
        if (!file)
            throw std::runtime_error("cannot write replay transaction");
        file << contents;
        file.flush();
        if (!file)
            throw std::runtime_error("cannot write replay transaction");
    }
#ifdef _WIN32
    const std::wstring temporary_native = extended_windows_path(temporary);
    const HANDLE temporary_handle =
        CreateFileW(temporary_native.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (temporary_handle == INVALID_HANDLE_VALUE || !FlushFileBuffers(temporary_handle)) {
        if (temporary_handle != INVALID_HANDLE_VALUE)
            CloseHandle(temporary_handle);
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot flush replay transaction");
    }
    CloseHandle(temporary_handle);
#else
    const int temporary_handle = ::open(temporary.c_str(), O_RDONLY);
    if (temporary_handle < 0 || ::fsync(temporary_handle) != 0) {
        if (temporary_handle >= 0)
            ::close(temporary_handle);
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot flush replay transaction");
    }
    ::close(temporary_handle);
#endif
    if (failure_injected("DIAMOND_REPLAY_FAIL_ACTIVATE")) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("injected replay activation failure");
    }
#ifdef _WIN32
    const std::wstring destination_native = extended_windows_path(path);
    if (!MoveFileExW(temporary_native.c_str(), destination_native.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot activate replay transaction");
    }
#else
    std::error_code error; std::filesystem::rename(temporary,path,error); if(error){std::filesystem::remove(temporary); throw std::runtime_error("cannot activate replay transaction");}
#endif
}
}  // namespace

struct ReplayStore::Impl {
    std::filesystem::path namespace_path;
    std::filesystem::path manifest_path;
    Compatibility compatibility;
    size_t capacity;
    uint64_t replay_seed;
    ReplayContents contents = ReplayContents::full;
    JsonValue authoritative_compatibility;
    std::vector<EpisodeRecord> episodes;
    std::vector<TrainingSample> samples;
    Array aborted_records;
    // Per-episode chunk digest, positionally aligned with `episodes`; empty for
    // an aborted episode, which has no chunk.  This used to hold the whole
    // parsed chunk body per episode, which cost ~30 KB per sample and was
    // retained for the store's lifetime -- at a 1M capacity that alone is tens
    // of gigabytes.  Every use was ultimately after the digest, so only the
    // digest is kept.
    std::vector<std::string> chunk_digests;
    std::vector<ReplayChunkEncoding> chunk_encodings;

    // Every completed episode gets its digest recorded at the same moment it
    // enters `episodes`, on both the load and the ingest path, so a missing one
    // is a broken invariant rather than a cache miss to recompute -- the
    // samples needed to recompute it are no longer retained.
    const std::string& chunk_digest(size_t index) const {
        if (index >= chunk_digests.size() || chunk_digests[index].empty())
            throw std::runtime_error("replay episode has no chunk digest");
        return chunk_digests[index];
    }

    ReplayChunkEncoding chunk_encoding(size_t index) const {
        if (index >= chunk_encodings.size())
            throw std::runtime_error("replay episode has no chunk encoding");
        return chunk_encodings[index];
    }

    std::filesystem::path chunk_path(size_t index) const {
        const auto& digest = chunk_digest(index);
        return namespace_path / "chunks" /
               (digest + std::string(chunk_encoding(index) == ReplayChunkEncoding::binary_v1
                               ? ".bin"
                               : ".json"));
    }

    std::string chunk_path(const std::string& digest, ReplayChunkEncoding encoding) const {
        return digest + std::string(encoding == ReplayChunkEncoding::binary_v1 ? ".bin" : ".json");
    }
    // Diagnostics for the last sample() call.  Sampling is pure, so this is
    // reporting only: nothing reads it back and no manifest records it.
    mutable ReplaySamplingStats sampling_stats;

    void write_manifest(size_t first = 0) {
        Array chunks, game_ids;
        bool has_binary_chunks = false;
        for (size_t i = first; i < episodes.size(); ++i)
            if (episodes[i].completed) {
                const auto& episode = episodes[i];
                const auto digest = chunk_digest(i);
                const auto encoding = chunk_encoding(i);
                if (encoding == ReplayChunkEncoding::binary_v1)
                    has_binary_chunks = true;
                chunks.emplace_back(
                    JsonValue{Object{{"game_id", JsonValue{episode.game_id}},
                                     {"sample_count", JsonValue{int64_t(episode.sample_count)}},
                                     {"sha256", JsonValue{digest}},
                                     {"encoding", JsonValue{encoding_name(encoding)}}}});
                game_ids.emplace_back(JsonValue{episode.game_id});
            }
        const auto compat = std::holds_alternative<Object>(authoritative_compatibility.value) ? authoritative_compatibility : compatibility_json(compatibility);
        // Contents identity only.  The manifest answers "which samples does
        // this store hold, under which compatibility, at which capacity" and
        // nothing else: no sampler state, no transaction records.  Training
        // therefore cannot change the manifest, and its digest is stable
        // across a TRAIN stage.
        Object final{{"aborted", JsonValue{aborted_records}},
                     {"capacity", JsonValue{int64_t(capacity)}},
                     {"chunks", JsonValue{std::move(chunks)}},
                     {"compatibility", std::move(compat)},
                     {"game_ids", JsonValue{std::move(game_ids)}},
                     {"schema_version", JsonValue{int64_t(has_binary_chunks ? 5 : 4)}}};
        atomic_write(manifest_path, diamond_support::canonical_json(JsonValue{std::move(final)}));
    }

    size_t retained_episode_start() const {
        size_t retained_samples = 0;
        size_t first = 0;
        for (size_t i = episodes.size(); i-- > 0;) {
            if (episodes[i].completed)
                retained_samples += episodes[i].sample_count;
            first = i;
            if (retained_samples >= capacity)
                break;
        }
        return first;
    }

    void discard_episode_prefix(size_t first) {
        if (first == 0)
            return;
        episodes.erase(episodes.begin(),
                       episodes.begin() + static_cast<std::ptrdiff_t>(first));
        chunk_digests.erase(chunk_digests.begin(),
                            chunk_digests.begin() + static_cast<std::ptrdiff_t>(first));
        chunk_encodings.erase(chunk_encodings.begin(),
                              chunk_encodings.begin() + static_cast<std::ptrdiff_t>(first));
    }

    void cleanup_unreachable_chunks() {
        if (failure_injected("DIAMOND_REPLAY_FAIL_BEFORE_CHUNK_CLEANUP"))
            throw std::runtime_error("injected replay pre-cleanup failure");
        std::unordered_set<std::string> referenced;
        for (size_t i = 0; i < episodes.size(); ++i)
            if (episodes[i].completed)
                referenced.insert(chunk_path(i).filename().string());
        std::error_code error;
        const auto chunks = namespace_path / "chunks";
        const bool chunks_exist = std::filesystem::exists(chunks, error);
        if (error)
            throw std::runtime_error("cannot inspect replay chunk directory: " + error.message());
        if (!chunks_exist) {
            if (!referenced.empty())
                throw std::runtime_error("replay manifest references a missing chunk directory");
            return;
        }
        for (std::filesystem::directory_iterator it(chunks, error), end; !error && it != end;
             it.increment(error)) {
            if (!it->is_regular_file(error) || error ||
                referenced.contains(it->path().filename().string()))
                continue;
            remove_file(it->path(), error);
            if (!error && failure_injected("DIAMOND_REPLAY_FAIL_AFTER_CHUNK_DELETE"))
                throw std::runtime_error("injected replay post-cleanup failure");
        }
        if (error)
            throw std::runtime_error("cannot clean unreachable replay chunk: " + error.message());
    }
};

ReplayStore::ReplayStore(std::filesystem::path root, Compatibility compatibility, size_t capacity,
                         uint64_t seed, ReplayContents contents)
    : impl_(std::make_unique<Impl>()) {
    if (capacity == 0) throw std::invalid_argument("replay capacity must be positive");
    impl_->compatibility = std::move(compatibility);
    impl_->capacity = capacity;
    impl_->replay_seed = seed;
    impl_->contents = contents;
    impl_->compatibility.validate();
    const auto compatibility_digest = diamond_support::sha256(
        diamond_support::canonical_json(compatibility_json(impl_->compatibility)));
    impl_->namespace_path = std::move(root) / "persistent-replay-v2" /
                            impl_->compatibility.family() / compatibility_digest;
    impl_->manifest_path = impl_->namespace_path / "manifest.json";
    if (!std::filesystem::exists(impl_->manifest_path)) {
        impl_->write_manifest();
        return;
    }

    std::ifstream source_manifest(stream_path(impl_->manifest_path), std::ios::binary);
    const std::string manifest_text{std::istreambuf_iterator<char>(source_manifest),
                                    std::istreambuf_iterator<char>()};
    const auto parsed = diamond_support::parse_json(manifest_text);
    const auto& manifest = object(parsed, "manifest");
    const auto schema_version = integer(field(manifest, "schema_version"), "schema");
    if (schema_version != 4 && schema_version != 5)
        throw std::runtime_error("unsupported replay manifest");
    const auto& chunks = array(field(manifest, "chunks"), "chunks");
    const auto& ids = array(field(manifest, "game_ids"), "game ids");
    if (chunks.size() != ids.size())
        throw std::runtime_error("manifest game_ids do not match ordered chunks");
    const auto persisted_capacity = integer(field(manifest, "capacity"), "capacity");
    if (persisted_capacity <= 0)
        throw std::runtime_error("invalid replay capacity");
    impl_->capacity = static_cast<size_t>(persisted_capacity);
    impl_->authoritative_compatibility = field(manifest, "compatibility");
    if (const auto found = manifest.find("aborted"); found != manifest.end())
        impl_->aborted_records = array(found->second, "aborted");

    // One pass over the descriptors.  Metadata-only stops here: the manifest
    // already carries game_id, sample_count and sha256 for every episode, which
    // is everything except the samples themselves -- so a stage that only needs
    // the manifest digest or the episode index never touches a chunk file.
    impl_->episodes.reserve(chunks.size());
    impl_->chunk_digests.reserve(chunks.size());
    impl_->chunk_encodings.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& descriptor = object(chunks[i], "chunk descriptor");
        const auto id = string(field(descriptor, "game_id"), "game id");
        if (id != string(ids[i], "game id"))
            throw std::runtime_error("manifest game_ids do not match ordered chunks");
        const auto digest = string(field(descriptor, "sha256"), "digest");
        const auto encoding = schema_version == 5 ?
                                (descriptor.find("encoding") == descriptor.end()
                                     ? ReplayChunkEncoding::json_v1
                                     : encoding_from_string(string(descriptor.find("encoding")->second,
                                                                  "chunk encoding")))
                                : ReplayChunkEncoding::json_v1;
        EpisodeRecord episode;
        episode.game_id = id;
        episode.completed = true;
        episode.sample_count =
            static_cast<size_t>(integer(field(descriptor, "sample_count"), "sample count"));
        impl_->episodes.push_back(std::move(episode));
        impl_->chunk_digests.push_back(digest);
        impl_->chunk_encodings.push_back(encoding);
        if (contents == ReplayContents::metadata_only)
            continue;

        // Full open: read and verify each chunk exactly once, materialising its
        // samples straight into the pool.  This used to parse every chunk
        // twice -- once to count, then again after clearing -- which at a 1M
        // capacity meant parsing ~2 GB of JSON per store construction, and a
        // training iteration constructs a store in three separate stages.
        const auto chunk_name = impl_->chunk_path(digest, encoding);
        if (encoding == ReplayChunkEncoding::json_v1) {
            const auto chunk_bytes = read_file_bytes(impl_->namespace_path / "chunks" / chunk_name);
            const std::string chunk_text(to_string_view(chunk_bytes));
            auto payload = object(diamond_support::parse_json(chunk_text), "chunk");
            const auto stored = string(field(payload, "sha256"), "digest");
            payload.erase("sha256");
            if (stored != digest ||
                diamond_support::sha256(diamond_support::canonical_json(JsonValue{payload})) != digest)
                throw std::runtime_error("corrupt replay chunk hash");
            const auto chunk_compatibility = compatibility_from_json(field(payload, "compatibility"));
            const auto samples_start = impl_->samples.size();
            for (const auto& row : array(field(payload, "samples"), "samples"))
                impl_->samples.push_back(legacy_sample(row, chunk_compatibility));
            if (impl_->samples.size() - samples_start != episode.sample_count)
                throw std::runtime_error("replay chunk sample count mismatch");
            continue;
        }

        const auto chunk_bytes = read_file_bytes(impl_->namespace_path / "chunks" / chunk_name);
        if (diamond_support::sha256(to_string_view(chunk_bytes)) != digest)
            throw std::runtime_error("corrupt replay chunk hash");
        const auto chunk = decode_replay_segment(std::span<const std::byte>(chunk_bytes),
                                                impl_->compatibility);
        if (chunk.compatibility != impl_->compatibility)
            throw std::runtime_error("replay chunk compatibility mismatch");
        if (chunk.samples.size() != episode.sample_count)
            throw std::runtime_error("replay chunk sample count mismatch");
        impl_->samples.insert(impl_->samples.end(), chunk.samples.begin(), chunk.samples.end());
    }
    if (impl_->samples.size() > impl_->capacity)
        impl_->samples.erase(impl_->samples.begin(),
                             impl_->samples.end() - static_cast<std::ptrdiff_t>(impl_->capacity));
    impl_->cleanup_unreachable_chunks();
}
ReplayStore::~ReplayStore() = default;
ReplayStore::ReplayStore(ReplayStore&&) noexcept = default;
ReplayStore& ReplayStore::operator=(ReplayStore&&) noexcept = default;

ReplayIngestReport ReplayStore::ingest_iteration(std::span<const Episode> episodes) {
    std::unordered_set<std::string> known; for(const auto& episode:impl_->episodes) known.insert(episode.game_id);
    for (const auto& row : impl_->aborted_records) { const auto& aborted = object(row, "aborted"); if (const auto value = aborted.find("game_id"); value != aborted.end()) known.insert(string(value->second, "aborted game id")); }
    std::vector<const Episode*> accepted;
    ReplayIngestReport report;
    for(const Episode& episode:episodes) {
        if(episode.game_id.empty()) throw std::invalid_argument("replay game_id is empty");
        if(episode.compatibility != impl_->compatibility) throw std::invalid_argument("replay compatibility mismatch");
        if (!known.insert(episode.game_id).second) {
            bool identical = false;
            for (size_t i = 0; i < impl_->episodes.size(); ++i) if (impl_->episodes[i].game_id == episode.game_id) {
                if (impl_->episodes[i].completed != episode.completed)
                    break;
                if (episode.completed && i < impl_->chunk_digests.size() &&
                    !impl_->chunk_digests[i].empty()) {
                    std::string encoded;
                    if (impl_->chunk_encoding(i) == ReplayChunkEncoding::binary_v1)
                        encoded =
                            diamond_support::sha256(to_string_view(encode_replay_segment(
                                episode, impl_->compatibility)));
                    else
                        encoded = diamond_support::sha256(
                            diamond_support::canonical_json(chunk_body(episode, impl_->compatibility)));
                    identical = encoded == impl_->chunk_digests[i];
                } else
                    identical = !episode.completed && impl_->episodes[i].aborted_reason ==
                                       episode.aborted_reason &&
                               impl_->episodes[i].move_count == episode.move_count &&
                               impl_->episodes[i].seed == episode.seed &&
                               impl_->episodes[i].retry_id == episode.retry_id;
                break;
            }
            if (!episode.completed) for (const auto& row : impl_->aborted_records) {
                const auto& aborted = object(row, "aborted");
                if (string(field(aborted, "game_id"), "aborted game id") == episode.game_id)
                    identical = string(field(aborted, "aborted_reason"), "aborted reason") == episode.aborted_reason && integer(field(aborted, "move_count"), "move count") == static_cast<int64_t>(episode.move_count) && integer(field(aborted, "seed"), "seed") == static_cast<int64_t>(episode.seed) && string(field(aborted, "retry_id"), "retry id") == episode.retry_id;
            }
            if (!identical) throw std::invalid_argument("conflicting duplicate game_id");
            ++report.duplicate_games;
            if (episode.completed)
                report.duplicate_samples += episode.samples.size();
            continue;
        }
        accepted.push_back(&episode);
        if (episode.completed)
            report.accepted_samples += episode.samples.size();
    }
    // Digests are captured here, where each body is serialised once anyway, so
    // the commit below never has to rebuild a chunk body.  Empty means aborted.
    std::vector<std::string> accepted_digests(accepted.size());
    std::vector<ReplayChunkEncoding> accepted_encodings(accepted.size(),
                                                       ReplayChunkEncoding::json_v1);
    for (size_t i = 0; i < accepted.size(); ++i) if (accepted[i]->completed) {
        const auto bytes = encode_replay_segment(*accepted[i], impl_->compatibility);
        const auto hash = diamond_support::sha256(to_string_view(bytes));
        accepted_digests[i] = hash;
        accepted_encodings[i] = ReplayChunkEncoding::binary_v1;
        atomic_write(impl_->namespace_path / "chunks" / (hash + ".bin"), std::string(to_string_view(bytes)));
    }
    if (!accepted.empty() && failure_injected("DIAMOND_REPLAY_FAIL_AFTER_CHUNK_ACTIVATE"))
        throw std::runtime_error("injected replay failure after chunk activation");
    const auto old_episode_count = impl_->episodes.size();
    const auto old_sample_count = impl_->samples.size();
    const auto old_aborted_count = impl_->aborted_records.size();
    const auto old_chunk_count = impl_->chunk_digests.size();
    const auto old_chunk_encoding_count = impl_->chunk_encodings.size();
    try {
        for (size_t i = 0; i < accepted.size(); ++i) {
            const auto& episode = *accepted[i];
            impl_->episodes.push_back(EpisodeRecord{.game_id = episode.game_id,
                                                    .retry_id = episode.retry_id,
                                                    .aborted_reason = episode.aborted_reason,
                                                    .seed = episode.seed,
                                                    .move_count = episode.move_count,
                                                    .sample_count = episode.samples.size(),
                                                    .completed = episode.completed});
            impl_->chunk_digests.push_back(accepted_digests[i]);
            impl_->chunk_encodings.push_back(accepted_encodings[i]);
            if (episode.completed) {
                // A metadata-only store has no pool to extend; size() reports
                // from the records instead.
                if (impl_->contents == ReplayContents::full)
                    impl_->samples.insert(impl_->samples.end(), episode.samples.begin(),
                                          episode.samples.end());
            } else {
                impl_->aborted_records.emplace_back(
                    JsonValue{Object{{"game_id", JsonValue{episode.game_id}},
                                     {"aborted_reason", JsonValue{episode.aborted_reason}},
                                     {"move_count", JsonValue{int64_t(episode.move_count)}},
                                     {"retry_id", JsonValue{episode.retry_id}},
                                     {"seed", JsonValue{int64_t(episode.seed)}}}});
            }
        }
    if (!accepted.empty() && failure_injected("DIAMOND_REPLAY_FAIL_BEFORE_MANIFEST_COMMIT"))
        throw std::runtime_error("injected replay failure before manifest commit");
    const auto retained_start = impl_->retained_episode_start();
    if (!accepted.empty())
        impl_->write_manifest(retained_start);
    } catch (...) {
        impl_->episodes.resize(old_episode_count);
        impl_->samples.resize(old_sample_count);
        impl_->aborted_records.resize(old_aborted_count);
        impl_->chunk_digests.resize(old_chunk_count);
        impl_->chunk_encodings.resize(old_chunk_encoding_count);
        throw;
    }
    if (!accepted.empty())
        impl_->discard_episode_prefix(impl_->retained_episode_start());
    if(impl_->samples.size()>impl_->capacity) impl_->samples.erase(impl_->samples.begin(),impl_->samples.end()-static_cast<std::ptrdiff_t>(impl_->capacity));
    if (!accepted.empty())
        impl_->cleanup_unreachable_chunks();
    report.accepted_games = accepted.size();
    return report;
}
size_t ReplayStore::ingest(std::span<const Episode> episodes) {
    return ingest_iteration(episodes).accepted_games;
}
size_t ReplayStore::size() const noexcept {
    if (!impl_)
        return 0;
    // Metadata-only holds no samples, so the pool size comes from the episode
    // records the manifest carried -- the same number a full open would report.
    if (impl_->contents == ReplayContents::metadata_only) {
        size_t total = 0;
        for (const auto& episode : impl_->episodes)
            if (episode.completed)
                total += episode.sample_count;
        return std::min(total, impl_->capacity);
    }
    return impl_->samples.size();
}
std::filesystem::path ReplayStore::manifest_path() const {
    if (!impl_ || !std::filesystem::is_regular_file(impl_->manifest_path))
        throw std::runtime_error("replay manifest is unavailable");
    return impl_->manifest_path;
}
std::string ReplayStore::manifest_digest() const {
    const auto path = manifest_path();
    std::ifstream input(stream_path(path), std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read replay manifest");
    const std::string contents((std::istreambuf_iterator<char>(input)), {});
    return diamond_support::sha256(contents);
}
uint64_t ReplayStore::replay_seed() const noexcept { return impl_ ? impl_->replay_seed : 0; }
uint64_t replay_sampling_seed(uint64_t replay_seed, uint64_t iteration, uint64_t training_step) {
    uint64_t state = replay_seed;
    state ^= next_splitmix(state) ^ (iteration + 0x2545f4914f6cdd1dULL);
    state ^= next_splitmix(state) ^ (training_step + 0x9e3779b97f4a7c15ULL);
    return next_splitmix(state);
}
std::vector<TrainingSample> ReplayStore::sample(size_t count, uint64_t seed) const {
    if (impl_->contents == ReplayContents::metadata_only)
        throw std::logic_error("replay store was opened metadata-only and holds no samples");
    if (count == 0)
        return {};
    if (count > impl_->samples.size())
        throw std::invalid_argument("replay sample count exceeds available samples");
    // Partial Fisher-Yates over a sparse swap map: `count` draws, no shuffle of
    // the whole pool and no allocation proportional to it.  The stream is
    // seeded per call, so this reads memory and copies rows -- nothing else.
    uint64_t stream = seed;
    std::unordered_map<size_t, size_t> swaps;
    swaps.reserve(count * 2);
    std::vector<size_t> selected;
    selected.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t remaining = impl_->samples.size() - i;
        const size_t index = i + next_splitmix(stream) % remaining;
        const auto selected_it = swaps.find(index);
        selected.push_back(selected_it == swaps.end() ? index : selected_it->second);
        const auto replacement_it = swaps.find(i);
        swaps[index] = replacement_it == swaps.end() ? i : replacement_it->second;
    }
    std::vector<TrainingSample> out;
    out.reserve(count);
    for (const auto index : selected)
        out.push_back(impl_->samples[index]);
    impl_->sampling_stats = {.selection_slots = swaps.size(), .copied_samples = out.size()};
    return out;
}
ReplaySamplingStats ReplayStore::last_sampling_stats() const noexcept {
    return impl_ ? impl_->sampling_stats : ReplaySamplingStats{};
}
void ReplayStore::prune() {
    const auto first = impl_->retained_episode_start();
    if (first == 0) return;
    impl_->write_manifest(first);
    impl_->discard_episode_prefix(first);
    if (impl_->samples.size() > impl_->capacity)
        impl_->samples.erase(impl_->samples.begin(),
                             impl_->samples.end() -
                                 static_cast<std::ptrdiff_t>(impl_->capacity));
    impl_->cleanup_unreachable_chunks();
}
}  // namespace diamond_pipeline
