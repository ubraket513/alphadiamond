#include "diamond_pipeline/replay_store.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "diamond_support/json.hpp"

namespace diamond_pipeline {
namespace {
using diamond_support::JsonValue;
using Object = JsonValue::Object;
using Array = JsonValue::Array;

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
bool same_json(const JsonValue& a, const JsonValue& b) {
    return diamond_support::canonical_json(a) == diamond_support::canonical_json(b);
}
TrainingSample legacy_sample(const JsonValue& value, const Compatibility& compatibility) { const auto& row=object(value,"sample"); TrainingSample sample; sample.compatibility=compatibility; for(const auto& feature:array(field(row,"node_features"),"features")) for(const auto& item:array(feature,"feature row")) sample.node_features.push_back(float(number(item,"feature"))); for(const auto& player:array(field(row,"canonical_player_ids"),"players")) sample.canonical_player_ids.push_back(int32_t(integer(player,"player"))); for(const auto& pair:array(field(row,"sparse_policy"),"policy")){const auto& values=array(pair,"policy row"); if(values.size()!=2) throw std::runtime_error("policy row width");sample.sparse_policy.emplace_back(int32_t(integer(values[0],"action")),float(number(values[1],"probability")));} for(const auto& target:array(field(row,"value_target"),"targets")) sample.value_target.push_back(float(number(target,"target"))); return sample; }
uint32_t mt_next(std::vector<uint32_t>& state, size_t& index) { if(index>=624){for(size_t i=0;i<624;++i){const uint32_t y=(state[i]&0x80000000U)|(state[(i+1)%624]&0x7fffffffU);state[i]=state[(i+397)%624]^(y>>1)^((y&1U)?0x9908b0dfU:0U);}index=0;} uint32_t y=state[index++];y^=y>>11;y^=(y<<7)&0x9d2c5680U;y^=(y<<15)&0xefc60000U;return y^(y>>18); }
size_t mt_below(std::vector<uint32_t>& state, size_t& index, size_t n) { const int bits=std::bit_width(n); uint32_t value; do { value=mt_next(state,index)>>(32-bits); } while(value>=n); return value; }
uint64_t next_splitmix(uint64_t& state) { state += 0x9e3779b97f4a7c15ULL; uint64_t value=state; value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL; value=(value^(value>>27))*0x94d049bb133111ebULL; return value^(value>>31); }
bool failure_injected(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0) return false;
    const bool present = value != nullptr && *value != '\0';
    std::free(value);
    return present;
#else
    const auto* value = std::getenv(name);
    return value != nullptr && *value != '\0';
#endif
}
JsonValue chunk_body(const Episode& episode, const Compatibility& compatibility) { Array rows; for (const auto& sample : episode.samples) rows.push_back(sample_json(sample)); return JsonValue{Object{{"compatibility",compatibility_json(compatibility)}, {"episode",JsonValue{Object{{"completed",JsonValue{episode.completed}}, {"game_id",JsonValue{episode.game_id}}, {"move_count",JsonValue{int64_t(episode.move_count)}}, {"retry_id",JsonValue{episode.retry_id}}, {"seed",JsonValue{int64_t(episode.seed)}}}}}, {"samples",JsonValue{std::move(rows)}}, {"schema_version",JsonValue{int64_t(1)}}}}; }
#ifdef _WIN32
std::wstring extended_windows_path(const std::filesystem::path& path) {
    std::wstring value = std::filesystem::absolute(path).wstring();
    if (value.rfind(L"\\\\?\\", 0) == 0) return value;
    if (value.rfind(L"\\\\", 0) == 0) return L"\\\\?\\UNC\\" + value.substr(2);
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
    { std::ofstream file(temporary, std::ios::binary|std::ios::trunc); if(!file) throw std::runtime_error("cannot write replay transaction"); file<<contents; file.flush(); if(!file) throw std::runtime_error("cannot write replay transaction"); }
#ifdef _WIN32
    const std::wstring temporary_native = extended_windows_path(temporary);
    const HANDLE temporary_handle = CreateFileW(
        temporary_native.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (temporary_handle == INVALID_HANDLE_VALUE || !FlushFileBuffers(temporary_handle)) {
        if (temporary_handle != INVALID_HANDLE_VALUE) CloseHandle(temporary_handle);
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot flush replay transaction");
    }
    CloseHandle(temporary_handle);
#else
    const int temporary_handle = ::open(temporary.c_str(), O_RDONLY);
    if (temporary_handle < 0 || ::fsync(temporary_handle) != 0) {
        if (temporary_handle >= 0) ::close(temporary_handle);
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot flush replay transaction");
    }
    ::close(temporary_handle);
#endif
    if (failure_injected("DIAMOND_REPLAY_FAIL_ACTIVATE")) { std::filesystem::remove(temporary); throw std::runtime_error("injected replay activation failure"); }
#ifdef _WIN32
    const std::wstring destination_native = extended_windows_path(path);
    if (!MoveFileExW(temporary_native.c_str(), destination_native.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { std::filesystem::remove(temporary); throw std::runtime_error("cannot activate replay transaction"); }
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
    uint64_t rng_state;
    bool legacy = false;
    JsonValue legacy_manifest;
    std::vector<uint32_t> mt_state;
    size_t mt_index = 0;
    std::optional<double> mt_gauss_next;
    JsonValue authoritative_compatibility;
    std::vector<Episode> episodes;
    std::vector<TrainingSample> samples;
    Array aborted_records;
    std::vector<JsonValue> chunk_payloads;
    std::string rng_algorithm = "splitmix64";
    JsonValue selection_transaction{nullptr};
    JsonValue ingest_transaction{nullptr};
    ReplaySamplingStats sampling_stats;

    JsonValue rng_json() const {
        Object rng{{"algorithm", JsonValue{rng_algorithm}}};
        if (rng_algorithm == "python-mt19937") {
            Array words; for (auto word : mt_state) words.emplace_back(JsonValue{int64_t(word)});
            words.emplace_back(JsonValue{int64_t(mt_index)});
            JsonValue gauss = mt_gauss_next ? JsonValue{*mt_gauss_next} : JsonValue{nullptr};
            rng["state"] = JsonValue{Array{JsonValue{int64_t(3)}, JsonValue{std::move(words)}, std::move(gauss)}};
        } else rng["state"] = JsonValue{std::to_string(rng_state)};
        return JsonValue{std::move(rng)};
    }

    void write_manifest() {
        if (legacy) {
            auto& manifest = std::get<Object>(legacy_manifest.value);
            auto& state = std::get<Array>(manifest.at("rng_state").value);
            auto& words = std::get<Array>(state.at(1).value);
            for (size_t i = 0; i < mt_state.size(); ++i) words[i] = JsonValue{int64_t(mt_state[i])};
            words[624] = JsonValue{int64_t(mt_index)};
            atomic_write(manifest_path, diamond_support::canonical_json(legacy_manifest));
            return;
        }
        Array chunks, game_ids; for (size_t i = 0; i < episodes.size(); ++i) if (episodes[i].completed) { const auto& episode = episodes[i]; const auto body = (i < chunk_payloads.size() && std::holds_alternative<Object>(chunk_payloads[i].value)) ? chunk_payloads[i] : chunk_body(episode, compatibility); const auto digest=diamond_support::sha256(diamond_support::canonical_json(body)); chunks.emplace_back(JsonValue{Object{{"game_id",JsonValue{episode.game_id}}, {"sample_count",JsonValue{int64_t(episode.samples.size())}}, {"sha256",JsonValue{digest}}}}); game_ids.emplace_back(JsonValue{episode.game_id}); }
        const auto compat = std::holds_alternative<Object>(authoritative_compatibility.value) ? authoritative_compatibility : compatibility_json(compatibility);
        Object final{{"aborted",JsonValue{aborted_records}}, {"capacity",JsonValue{int64_t(capacity)}}, {"chunks",JsonValue{std::move(chunks)}}, {"compatibility",std::move(compat)}, {"game_ids",JsonValue{std::move(game_ids)}}, {"ingest_transaction",ingest_transaction}, {"rng",rng_json()}, {"schema_version",JsonValue{int64_t(3)}}, {"selection_transaction",selection_transaction}};
        atomic_write(manifest_path, diamond_support::canonical_json(JsonValue{std::move(final)}));
    }

    void cleanup_unreachable_chunks() {
        if (failure_injected("DIAMOND_REPLAY_FAIL_BEFORE_CHUNK_CLEANUP"))
            throw std::runtime_error("injected replay pre-cleanup failure");
        std::unordered_set<std::string> referenced;
        for (size_t i = 0; i < episodes.size(); ++i) if (episodes[i].completed) {
            const auto body = i < chunk_payloads.size() && std::holds_alternative<Object>(chunk_payloads[i].value)
                ? chunk_payloads[i] : chunk_body(episodes[i], compatibility);
            referenced.insert(diamond_support::sha256(diamond_support::canonical_json(body)) + ".json");
        }
        std::error_code error;
        const auto chunks = namespace_path / "chunks";
        for (std::filesystem::directory_iterator it(chunks, error), end; !error && it != end; it.increment(error)) {
            if (!it->is_regular_file(error) || error || referenced.contains(it->path().filename().string())) continue;
            remove_file(it->path(), error);
            if (!error && failure_injected("DIAMOND_REPLAY_FAIL_AFTER_CHUNK_DELETE"))
                throw std::runtime_error("injected replay post-cleanup failure");
        }
        if (error) throw std::runtime_error("cannot clean unreachable replay chunk: " + error.message());
    }
};

ReplayStore::ReplayStore(std::filesystem::path root, Compatibility compatibility, size_t capacity, uint64_t seed) : impl_(std::make_unique<Impl>()) {
    if (capacity == 0) throw std::invalid_argument("replay capacity must be positive");
    impl_->compatibility=std::move(compatibility); impl_->capacity=capacity; impl_->rng_state=seed;
    const auto original_root = root;
    const auto compatibility_digest=diamond_support::sha256(diamond_support::canonical_json(compatibility_json(impl_->compatibility)));
    impl_->compatibility.validate();
    impl_->namespace_path=std::move(root)/"persistent-replay-v2"/impl_->compatibility.family()/compatibility_digest; impl_->manifest_path=impl_->namespace_path/"manifest.json";
    std::error_code root_error;
    const bool root_exists = std::filesystem::exists(original_root, root_error) && !root_error;
    if (!std::filesystem::exists(impl_->manifest_path) && root_exists) {
        // A migrated store is authoritative.  Resolve it by the complete
        // canonical compatibility object, never by model name/version alone.
        std::filesystem::path match;
        const auto v2_family = original_root / "persistent-replay-v2" / impl_->compatibility.family();
        if (std::filesystem::exists(v2_family)) for (const auto& entry : std::filesystem::directory_iterator(v2_family, std::filesystem::directory_options::skip_permission_denied)) {
            if (entry.path().filename() == "manifest.json" || !std::filesystem::is_directory(entry.path())) continue;
            const auto manifest_path = entry.path() / "manifest.json";
            try {
                std::ifstream source(manifest_path, std::ios::binary);
                const auto parsed = diamond_support::parse_json(std::string{std::istreambuf_iterator<char>(source), {}});
                const auto& manifest = object(parsed, "manifest");
                const auto schema = integer(field(manifest, "schema_version"), "schema");
                if ((schema == 2 || schema == 3) && same_json(field(manifest, "compatibility"), compatibility_json(impl_->compatibility))) {
                    if (!match.empty()) throw std::runtime_error("multiple replay stores match compatibility");
                    match = manifest_path;
                }
            } catch (const std::runtime_error&) { throw; } catch (...) {}
        }
        if (!match.empty()) { impl_->manifest_path = match; impl_->namespace_path = match.parent_path(); }
    }
    if (!std::filesystem::exists(impl_->manifest_path) && root_exists) {
        const auto v1_family = original_root / "persistent-replay-v1";
        if (std::filesystem::exists(v1_family)) for (const auto& family : std::filesystem::directory_iterator(v1_family, std::filesystem::directory_options::skip_permission_denied)) {
            if (!std::filesystem::is_directory(family.path())) continue;
            for (const auto& digest_dir : std::filesystem::directory_iterator(family.path(), std::filesystem::directory_options::skip_permission_denied)) {
            if (!std::filesystem::is_directory(digest_dir.path())) continue;
            const auto manifest_path = digest_dir.path() / "manifest.json";
            try {
                std::ifstream candidate(manifest_path, std::ios::binary); const std::string text((std::istreambuf_iterator<char>(candidate)),{});
                const auto parsed=diamond_support::parse_json(text); const auto& manifest=object(parsed,"manifest");
                const auto& legacy=object(field(manifest,"compatibility"),"compatibility");
                const bool compatibility_match = same_json(JsonValue{legacy}, compatibility_json(impl_->compatibility));
                if (compatibility_match) { impl_->manifest_path=manifest_path; impl_->namespace_path=manifest_path.parent_path(); impl_->legacy=true; break; }
            } catch (...) {}
            }
            if (impl_->legacy) break;
        }
    }
    if (!impl_->legacy) {
        if (!std::filesystem::exists(impl_->manifest_path)) { impl_->write_manifest(); return; }
        std::ifstream source_manifest(impl_->manifest_path, std::ios::binary); const std::string manifest_text{std::istreambuf_iterator<char>(source_manifest), std::istreambuf_iterator<char>()}; const auto parsed=diamond_support::parse_json(manifest_text); const auto& manifest=object(parsed,"manifest"); const auto schema=integer(field(manifest,"schema_version"),"schema"); if(schema!=2 && schema!=3) throw std::runtime_error("unsupported replay manifest"); const auto& chunks=array(field(manifest,"chunks"),"chunks"); const auto& ids=array(field(manifest,"game_ids"),"game ids"); if(chunks.size()!=ids.size()) throw std::runtime_error("manifest game_ids do not match ordered chunks"); for(size_t i=0;i<chunks.size();++i){const auto& descriptor=object(chunks[i],"chunk descriptor"); const auto id=string(field(descriptor,"game_id"),"game id"); if(id!=string(ids[i],"game id")) throw std::runtime_error("manifest game_ids do not match ordered chunks"); const auto digest=string(field(descriptor,"sha256"),"digest"); std::ifstream chunk_file(impl_->namespace_path/"chunks"/(digest+".json"),std::ios::binary); if(!chunk_file) throw std::runtime_error("missing replay chunk"); const std::string chunk_text{std::istreambuf_iterator<char>(chunk_file),std::istreambuf_iterator<char>()}; auto chunk=diamond_support::parse_json(chunk_text); auto payload=object(chunk,"chunk"); const auto stored=string(field(payload,"sha256"),"digest"); payload.erase("sha256"); if(stored!=digest||diamond_support::sha256(diamond_support::canonical_json(JsonValue{payload}))!=digest) throw std::runtime_error("corrupt replay chunk hash"); for(const auto& row:array(field(payload,"samples"),"samples")) impl_->samples.push_back(legacy_sample(row,impl_->compatibility)); }
        const auto persisted_capacity = integer(field(manifest, "capacity"), "capacity");
        if (persisted_capacity <= 0) throw std::runtime_error("invalid replay capacity");
        impl_->capacity = static_cast<size_t>(persisted_capacity);
        impl_->samples.clear();
        // Rebuild live episodes from the ordered descriptors so a subsequent
        // sample/ingest cannot rewrite the manifest with empty chunks.
        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto& descriptor = object(chunks[i], "chunk descriptor");
            const auto id = string(field(descriptor, "game_id"), "game id");
            const auto digest = string(field(descriptor, "sha256"), "digest");
            std::ifstream chunk_file(impl_->namespace_path / "chunks" / (digest + ".json"), std::ios::binary);
            const std::string text{std::istreambuf_iterator<char>(chunk_file), std::istreambuf_iterator<char>()};
            auto payload = object(diamond_support::parse_json(text), "chunk");
            payload.erase("sha256");
            impl_->chunk_payloads.emplace_back(payload);
            Episode episode; episode.game_id = id; episode.completed = true; episode.compatibility = compatibility_from_json(field(payload, "compatibility"));
            for (const auto& row : array(field(payload, "samples"), "samples")) { auto sample = legacy_sample(row, episode.compatibility); episode.samples.push_back(sample); impl_->samples.push_back(std::move(sample)); }
            impl_->episodes.push_back(std::move(episode));
        }
        if (impl_->samples.size() > impl_->capacity) impl_->samples.erase(impl_->samples.begin(), impl_->samples.end() - static_cast<std::ptrdiff_t>(impl_->capacity));
        if (const auto found = manifest.find("aborted"); found != manifest.end()) impl_->aborted_records = array(found->second, "aborted");
        if (const auto found = manifest.find("selection_transaction"); found != manifest.end()) impl_->selection_transaction = found->second;
        if (const auto found = manifest.find("ingest_transaction"); found != manifest.end()) impl_->ingest_transaction = found->second;
        impl_->authoritative_compatibility = field(manifest, "compatibility");
        const auto& rng=object(field(manifest,"rng"),"rng"); impl_->rng_algorithm=string(field(rng,"algorithm"),"rng algorithm"); if (impl_->rng_algorithm=="python-mt19937") { const auto& state=array(field(rng,"state"),"rng state"); const auto& words=array(state.at(1),"mt state"); if(words.size()!=625 || state.size()<2) throw std::runtime_error("invalid CPython MT19937 state"); for(size_t i=0;i<624;++i) impl_->mt_state.push_back(uint32_t(integer(words.at(i),"mt word"))); impl_->mt_index=size_t(integer(words.at(624),"mt index")); if(state.size()>2 && !std::holds_alternative<std::nullptr_t>(state.at(2).value)) impl_->mt_gauss_next=number(state.at(2),"mt gauss_next"); } else impl_->rng_state=std::stoull(string(field(rng,"state"),"rng state"));
        impl_->cleanup_unreachable_chunks();
        return;
    }
    std::ifstream manifest_file(impl_->manifest_path,std::ios::binary); std::string manifest_text((std::istreambuf_iterator<char>(manifest_file)),{}); impl_->legacy_manifest=diamond_support::parse_json(manifest_text); const auto& manifest=object(impl_->legacy_manifest,"manifest");
    if (integer(field(manifest, "schema_version"), "schema") != 1) throw std::runtime_error("unsupported replay manifest");
    impl_->authoritative_compatibility = field(manifest, "compatibility");
    if (!same_json(field(manifest, "compatibility"), compatibility_json(impl_->compatibility))) throw std::runtime_error("replay manifest compatibility mismatch");
    const auto& chunks=array(field(manifest,"chunks"),"chunks"); const auto& ids=array(field(manifest,"game_ids"),"game ids"); if(chunks.size()!=ids.size()) throw std::runtime_error("manifest game_ids do not match ordered chunks");
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = object(chunks[i], "chunk");
        const auto id = string(field(chunk, "game_id"), "game id");
        if (id != string(ids[i], "game id")) throw std::runtime_error("manifest game_ids do not match ordered chunks");
        std::ifstream source(impl_->namespace_path / "chunks" / (diamond_support::sha256(id) + ".json"), std::ios::binary);
        if (!source) throw std::runtime_error("missing replay chunk: " + (impl_->namespace_path / "chunks" / (diamond_support::sha256(id) + ".json")).string());
        const std::string text{std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
        auto parsed = diamond_support::parse_json(text);
        auto payload = object(parsed, "chunk");
        const auto expected = string(field(payload, "sha256"), "digest");
        payload.erase("sha256");
        if (expected != string(field(chunk, "sha256"), "digest") || diamond_support::sha256(diamond_support::canonical_json(JsonValue{payload})) != expected) throw std::runtime_error("corrupt replay chunk hash");
        impl_->chunk_payloads.emplace_back(payload);
        Episode episode; episode.game_id = id; episode.completed = true; episode.compatibility = compatibility_from_json(field(payload, "compatibility"));
        const auto& episode_data = object(field(payload, "episode"), "episode");
        if (const auto found = episode_data.find("seed"); found != episode_data.end()) episode.seed = uint64_t(integer(found->second, "seed"));
        if (const auto found = episode_data.find("move_count"); found != episode_data.end()) episode.move_count = uint64_t(integer(found->second, "move count"));
        if (const auto found = episode_data.find("retry_id"); found != episode_data.end()) episode.retry_id = string(found->second, "retry id");
        for (const auto& row : array(field(payload, "samples"), "samples")) {
            auto sample = legacy_sample(row, episode.compatibility);
            episode.samples.push_back(sample);
            impl_->samples.push_back(std::move(sample));
        }
        impl_->episodes.push_back(std::move(episode));
    }
    if (const auto found = manifest.find("aborted"); found != manifest.end()) impl_->aborted_records = array(found->second, "aborted");
    if(impl_->samples.size()>capacity) impl_->samples.erase(impl_->samples.begin(),impl_->samples.end()-static_cast<std::ptrdiff_t>(capacity)); const auto& state=array(field(manifest,"rng_state"),"rng state"); const auto& words=array(state.at(1),"mt state"); if(integer(state.at(0),"rng version")!=3||words.size()!=625)throw std::runtime_error("invalid CPython MT19937 state"); for(size_t i=0;i<624;++i)impl_->mt_state.push_back(uint32_t(integer(words[i],"mt word"))); impl_->mt_index=size_t(integer(words[624],"mt index")); if(state.size()>2&&!std::holds_alternative<std::nullptr_t>(state.at(2).value)) impl_->mt_gauss_next=number(state.at(2),"mt gauss_next"); impl_->rng_algorithm="python-mt19937";
    // Legacy stores are migrated transactionally before the object becomes usable.
    impl_->legacy = false;
    const auto v2_digest = diamond_support::sha256(diamond_support::canonical_json(impl_->authoritative_compatibility));
    impl_->namespace_path = original_root / "persistent-replay-v2" / impl_->compatibility.family() / v2_digest;
    impl_->manifest_path = impl_->namespace_path / "manifest.json";
    for (size_t i = 0; i < impl_->episodes.size(); ++i) if (impl_->episodes[i].completed) {
        auto payload = impl_->chunk_payloads[i];
        const auto digest = diamond_support::sha256(diamond_support::canonical_json(payload));
        auto stored = payload;
        std::get<Object>(stored.value)["sha256"] = JsonValue{digest};
        atomic_write(impl_->namespace_path / "chunks" / (digest + ".json"), diamond_support::canonical_json(stored));
    }
    impl_->write_manifest();
}
ReplayStore::~ReplayStore() = default;
ReplayStore::ReplayStore(ReplayStore&&) noexcept = default;
ReplayStore& ReplayStore::operator=(ReplayStore&&) noexcept = default;

ReplayIngestReport ReplayStore::ingest_iteration(std::span<const Episode> episodes) {
    std::unordered_set<std::string> known; for(const auto& episode:impl_->episodes) known.insert(episode.game_id);
    for (const auto& row : impl_->aborted_records) { const auto& aborted = object(row, "aborted"); if (const auto value = aborted.find("game_id"); value != aborted.end()) known.insert(string(value->second, "aborted game id")); }
    std::vector<Episode> accepted;
    ReplayIngestReport report;
    for(const Episode& episode:episodes) {
        if(episode.game_id.empty()) throw std::invalid_argument("replay game_id is empty");
        if(episode.compatibility != impl_->compatibility) throw std::invalid_argument("replay compatibility mismatch");
        if (!known.insert(episode.game_id).second) {
            bool identical = false;
            for (size_t i = 0; i < impl_->episodes.size(); ++i) if (impl_->episodes[i].game_id == episode.game_id) {
                if (impl_->episodes[i].completed != episode.completed) break;
                if (episode.completed && i < impl_->chunk_payloads.size() && std::holds_alternative<Object>(impl_->chunk_payloads[i].value)) identical = diamond_support::sha256(diamond_support::canonical_json(chunk_body(episode, impl_->compatibility))) == diamond_support::sha256(diamond_support::canonical_json(impl_->chunk_payloads[i]));
                else identical = !episode.completed && impl_->episodes[i].aborted_reason == episode.aborted_reason && impl_->episodes[i].move_count == episode.move_count && impl_->episodes[i].seed == episode.seed && impl_->episodes[i].retry_id == episode.retry_id;
                break;
            }
            if (!episode.completed) for (const auto& row : impl_->aborted_records) {
                const auto& aborted = object(row, "aborted");
                if (string(field(aborted, "game_id"), "aborted game id") == episode.game_id)
                    identical = string(field(aborted, "aborted_reason"), "aborted reason") == episode.aborted_reason && integer(field(aborted, "move_count"), "move count") == static_cast<int64_t>(episode.move_count) && integer(field(aborted, "seed"), "seed") == static_cast<int64_t>(episode.seed) && string(field(aborted, "retry_id"), "retry id") == episode.retry_id;
            }
            if (!identical) throw std::invalid_argument("conflicting duplicate game_id");
            ++report.duplicate_games;
            if (episode.completed) report.duplicate_samples += episode.samples.size();
            continue;
        }
        accepted.push_back(episode);
        if (episode.completed) report.accepted_samples += episode.samples.size();
    }
    for(const Episode& episode:accepted) if(episode.completed) { auto body=chunk_body(episode, impl_->compatibility); const auto hash=diamond_support::sha256(diamond_support::canonical_json(body)); std::get<Object>(body.value).emplace("sha256",JsonValue{hash}); atomic_write(impl_->namespace_path/"chunks"/(hash+".json"), diamond_support::canonical_json(std::move(body))); }
    if (!accepted.empty() && failure_injected("DIAMOND_REPLAY_FAIL_AFTER_CHUNK_ACTIVATE"))
        throw std::runtime_error("injected replay failure after chunk activation");
    const auto old_episodes = impl_->episodes; const auto old_samples = impl_->samples; const auto old_aborted = impl_->aborted_records; const auto old_chunks = impl_->chunk_payloads;
    try {
      for(const auto& episode:accepted) { impl_->episodes.push_back(episode); if(episode.completed) { impl_->samples.insert(impl_->samples.end(),episode.samples.begin(),episode.samples.end()); impl_->chunk_payloads.emplace_back(chunk_body(episode, impl_->compatibility)); } else { impl_->chunk_payloads.emplace_back(JsonValue{nullptr}); impl_->aborted_records.emplace_back(JsonValue{Object{{"game_id",JsonValue{episode.game_id}}, {"aborted_reason",JsonValue{episode.aborted_reason}}, {"move_count",JsonValue{int64_t(episode.move_count)}}, {"retry_id",JsonValue{episode.retry_id}}, {"seed",JsonValue{int64_t(episode.seed)}}}}); } }
    if(impl_->samples.size()>impl_->capacity) impl_->samples.erase(impl_->samples.begin(),impl_->samples.end()-static_cast<std::ptrdiff_t>(impl_->capacity));
      if(!accepted.empty()) {
          const auto transaction_id = diamond_support::sha256(diamond_support::canonical_json(JsonValue{Object{{"accepted_games", JsonValue{int64_t(accepted.size())}}, {"duplicate_games", JsonValue{int64_t(report.duplicate_games)}}, {"rng", impl_->rng_json()}}}));
          impl_->ingest_transaction = JsonValue{Object{{"accepted_games", JsonValue{int64_t(accepted.size())}}, {"accepted_samples", JsonValue{int64_t(report.accepted_samples)}}, {"duplicate_games", JsonValue{int64_t(report.duplicate_games)}}, {"duplicate_samples", JsonValue{int64_t(report.duplicate_samples)}}, {"state", JsonValue{"committed"}}, {"transaction_id", JsonValue{transaction_id}}}};
          impl_->write_manifest();
      }
    } catch (...) { impl_->episodes=old_episodes; impl_->samples=old_samples; impl_->aborted_records=old_aborted; impl_->chunk_payloads=old_chunks; throw; }
    report.accepted_games = accepted.size();
    return report;
}
size_t ReplayStore::ingest(std::span<const Episode> episodes) { return ingest_iteration(episodes).accepted_games; }
size_t ReplayStore::size() const noexcept { return impl_ ? impl_->samples.size() : 0; }
std::vector<TrainingSample> ReplayStore::sample(size_t count) {
    if (count == 0) return {};
    if (count > impl_->samples.size()) throw std::invalid_argument("replay sample count exceeds available samples");
    const auto old_rng = impl_->rng_state; const auto old_mt = impl_->mt_state;
    const auto old_index = impl_->mt_index; const auto old_gauss = impl_->mt_gauss_next;
    const auto before_rng = impl_->rng_json();
    bool activated = false;
    try {
        std::unordered_map<size_t, size_t> swaps;
        swaps.reserve(count * 2);
        std::vector<size_t> selected;
        selected.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const size_t remaining = impl_->samples.size() - i;
            const size_t offset = impl_->rng_algorithm == "python-mt19937"
                ? mt_below(impl_->mt_state, impl_->mt_index, remaining)
                : next_splitmix(impl_->rng_state) % remaining;
            const size_t index = i + offset;
            const auto selected_it = swaps.find(index);
            selected.push_back(selected_it == swaps.end() ? index : selected_it->second);
            const auto replacement_it = swaps.find(i);
            swaps[index] = replacement_it == swaps.end() ? i : replacement_it->second;
        }
        std::vector<TrainingSample> out;
        out.reserve(count);
        Array selected_ids;
        for (const auto index : selected) {
            out.push_back(impl_->samples[index]);
            selected_ids.emplace_back(JsonValue{"sample-index:" + std::to_string(index)});
        }
        impl_->sampling_stats = {.selection_slots = swaps.size(), .copied_samples = out.size()};
        const auto after_rng = impl_->rng_json();
        const auto transaction_id = diamond_support::sha256(diamond_support::canonical_json(JsonValue{Object{{"before_rng", before_rng}, {"selected_ids", JsonValue{selected_ids}}}}));
        impl_->selection_transaction = JsonValue{Object{{"after_rng", after_rng}, {"before_rng", before_rng}, {"selected_ids", JsonValue{std::move(selected_ids)}}, {"state", JsonValue{"committed"}}, {"transaction_id", JsonValue{transaction_id}}}};
        impl_->write_manifest();
        activated = true;
        if (failure_injected("DIAMOND_REPLAY_FAIL_AFTER_SELECTION_ACTIVATE"))
            throw std::runtime_error("injected replay selection post-activation failure");
        return out;
    } catch (...) {
        if (!activated) {
            impl_->rng_state=old_rng; impl_->mt_state=old_mt; impl_->mt_index=old_index; impl_->mt_gauss_next=old_gauss;
        }
        throw;
    }
}
ReplaySamplingStats ReplayStore::last_sampling_stats() const noexcept { return impl_ ? impl_->sampling_stats : ReplaySamplingStats{}; }
void ReplayStore::prune() {
    size_t total = 0, first = 0;
    for (size_t i = impl_->episodes.size(); i-- > 0;) {
        if (impl_->episodes[i].completed) total += impl_->episodes[i].samples.size();
        first = i;
        if (total >= impl_->capacity) break;
    }
    if (first == 0) return;
    const auto old_episodes=impl_->episodes; const auto old_samples=impl_->samples; const auto old_chunks=impl_->chunk_payloads; const auto old_aborted=impl_->aborted_records;
    try {
        impl_->episodes.erase(impl_->episodes.begin(), impl_->episodes.begin()+static_cast<std::ptrdiff_t>(first));
        if (impl_->chunk_payloads.size() >= first) impl_->chunk_payloads.erase(impl_->chunk_payloads.begin(), impl_->chunk_payloads.begin()+static_cast<std::ptrdiff_t>(first));
        if (impl_->samples.size()>impl_->capacity) impl_->samples.erase(impl_->samples.begin(),impl_->samples.end()-static_cast<std::ptrdiff_t>(impl_->capacity));
        impl_->write_manifest();
    } catch (...) { impl_->episodes=old_episodes; impl_->samples=old_samples; impl_->chunk_payloads=old_chunks; impl_->aborted_records=old_aborted; throw; }
    impl_->cleanup_unreachable_chunks();
}
void ReplayStore::restore_manifest(const std::filesystem::path& snapshot) {
    std::ifstream file(snapshot,std::ios::binary); if(!file) throw std::runtime_error("cannot read replay manifest snapshot");
    const std::string value((std::istreambuf_iterator<char>(file)),{}); const auto parsed=diamond_support::parse_json(value); const auto& manifest=object(parsed,"manifest");
    if (integer(field(manifest,"schema_version"),"schema") != 1) throw std::runtime_error("unsupported replay restore snapshot");
    // Reopen through the normal v1 loader in an isolated scratch namespace.  It
    // validates every descriptor/chunk and rebuilds episodes, samples, aborted
    // records, and RNG state before touching the live store.
    static std::atomic_uint64_t restore_sequence{0};
    const auto temp = std::filesystem::temp_directory_path() / ("alphadiamond-replay-restore-" + std::to_string(++restore_sequence));
    const auto legacy_dir = temp / "persistent-replay-v1" / "Soo" / snapshot.parent_path().filename();
    std::filesystem::create_directories(legacy_dir / "chunks");
    atomic_write(legacy_dir / "manifest.json", value);
    const auto source_chunks = snapshot.parent_path() / "chunks";
    if (std::filesystem::exists(source_chunks)) std::filesystem::copy(source_chunks, legacy_dir / "chunks", std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
    ReplayStore reopened(temp, impl_->compatibility, impl_->capacity, impl_->rng_state);
    const auto root = impl_->namespace_path.parent_path().parent_path().parent_path();
    const auto digest = diamond_support::sha256(diamond_support::canonical_json(reopened.impl_->authoritative_compatibility));
    reopened.impl_->namespace_path = root / "persistent-replay-v2" / reopened.impl_->compatibility.family() / digest;
    reopened.impl_->manifest_path = reopened.impl_->namespace_path / "manifest.json";
    reopened.impl_->write_manifest();
    impl_ = std::move(reopened.impl_);
    std::error_code ignored; std::filesystem::remove_all(temp, ignored);
}

}  // namespace diamond_pipeline
