#include "diamond_pipeline/replay_store.hpp"

#include <algorithm>
#include <bit>
#include <atomic>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

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
JsonValue compatibility_json(const Compatibility& compatibility) { return JsonValue{Object{{"family", JsonValue{compatibility.family}}, {"model_version", JsonValue{compatibility.model_version}}}}; }
const Object& object(const JsonValue& value, const char* what) { if (const auto* found=std::get_if<Object>(&value.value)) return *found; throw std::runtime_error(std::string(what)+" must be an object"); }
const Array& array(const JsonValue& value, const char* what) { if (const auto* found=std::get_if<Array>(&value.value)) return *found; throw std::runtime_error(std::string(what)+" must be an array"); }
const JsonValue& field(const Object& value, const char* name) { const auto found=value.find(name); if(found==value.end()) throw std::runtime_error(std::string("missing ")+name); return found->second; }
std::string string(const JsonValue& value, const char* what) { if(const auto* found=std::get_if<std::string>(&value.value)) return *found; throw std::runtime_error(std::string(what)+" must be string"); }
int64_t integer(const JsonValue& value, const char* what) { if(const auto* found=std::get_if<int64_t>(&value.value)) return *found; throw std::runtime_error(std::string(what)+" must be integer"); }
double number(const JsonValue& value, const char* what) { if(const auto* i=std::get_if<int64_t>(&value.value)) return double(*i); if(const auto* d=std::get_if<double>(&value.value)) return *d; throw std::runtime_error(std::string(what)+" must be number"); }
TrainingSample legacy_sample(const JsonValue& value, const Compatibility& compatibility) { const auto& row=object(value,"sample"); TrainingSample sample; sample.compatibility=compatibility; for(const auto& feature:array(field(row,"node_features"),"features")) for(const auto& item:array(feature,"feature row")) sample.node_features.push_back(float(number(item,"feature"))); for(const auto& player:array(field(row,"canonical_player_ids"),"players")) sample.canonical_player_ids.push_back(int32_t(integer(player,"player"))); for(const auto& pair:array(field(row,"sparse_policy"),"policy")){const auto& values=array(pair,"policy row"); if(values.size()!=2) throw std::runtime_error("policy row width");sample.sparse_policy.emplace_back(int32_t(integer(values[0],"action")),float(number(values[1],"probability")));} for(const auto& target:array(field(row,"value_target"),"targets")) sample.value_target.push_back(float(number(target,"target"))); return sample; }
uint32_t mt_next(std::vector<uint32_t>& state, size_t& index) { if(index>=624){for(size_t i=0;i<624;++i){const uint32_t y=(state[i]&0x80000000U)|(state[(i+1)%624]&0x7fffffffU);state[i]=state[(i+397)%624]^(y>>1)^((y&1U)?0x9908b0dfU:0U);}index=0;} uint32_t y=state[index++];y^=y>>11;y^=(y<<7)&0x9d2c5680U;y^=(y<<15)&0xefc60000U;return y^(y>>18); }
size_t mt_below(std::vector<uint32_t>& state, size_t& index, size_t n) { const int bits=std::bit_width(n); uint32_t value; do { value=mt_next(state,index)>>(32-bits); } while(value>=n); return value; }
uint64_t next_splitmix(uint64_t& state) { state += 0x9e3779b97f4a7c15ULL; uint64_t value=state; value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL; value=(value^(value>>27))*0x94d049bb133111ebULL; return value^(value>>31); }
JsonValue chunk_body(const Episode& episode, const Compatibility& compatibility) { Array rows; for (const auto& sample : episode.samples) rows.push_back(sample_json(sample)); return JsonValue{Object{{"compatibility",compatibility_json(compatibility)}, {"episode",JsonValue{Object{{"completed",JsonValue{episode.completed}}, {"game_id",JsonValue{episode.game_id}}, {"move_count",JsonValue{int64_t(episode.move_count)}}, {"retry_id",JsonValue{episode.retry_id}}, {"seed",JsonValue{int64_t(episode.seed)}}}}}, {"samples",JsonValue{std::move(rows)}}, {"schema_version",JsonValue{int64_t(1)}}}}; }
void atomic_write(const std::filesystem::path& path, const std::string& contents) { static std::atomic_uint64_t sequence{0}; std::filesystem::create_directories(path.parent_path()); const auto temporary=path.string()+".tmp."+std::to_string(++sequence); { std::ofstream file(temporary, std::ios::binary|std::ios::trunc); if(!file) throw std::runtime_error("cannot write replay transaction"); file<<contents; if(!file) throw std::runtime_error("cannot write replay transaction"); } std::error_code error; std::filesystem::rename(temporary,path,error); if(error){std::filesystem::remove(path,error); error.clear(); std::filesystem::rename(temporary,path,error);} if(error){std::filesystem::remove(temporary); throw std::runtime_error("cannot activate replay transaction");} }
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
    std::vector<Episode> episodes;
    std::vector<TrainingSample> samples;

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
        Array chunks, game_ids; for (const Episode& episode : episodes) if (episode.completed) { const auto body=chunk_body(episode, compatibility); const auto digest=diamond_support::sha256(diamond_support::canonical_json(body)); chunks.emplace_back(JsonValue{Object{{"game_id",JsonValue{episode.game_id}}, {"sample_count",JsonValue{int64_t(episode.samples.size())}}, {"sha256",JsonValue{digest}}}}); game_ids.emplace_back(JsonValue{episode.game_id}); }
        Object rng{{"algorithm",JsonValue{"splitmix64"}}, {"state",JsonValue{std::to_string(rng_state)}}};
        Object manifest{{"aborted",JsonValue{Array{}}}, {"capacity",JsonValue{int64_t(capacity)}}, {"chunks",JsonValue{std::move(chunks)}}, {"compatibility",compatibility_json(compatibility)}, {"game_ids",JsonValue{std::move(game_ids)}}, {"rng",JsonValue{std::move(rng)}}, {"schema_version",JsonValue{int64_t(2)}}};
        atomic_write(manifest_path, diamond_support::canonical_json(JsonValue{std::move(manifest)}));
    }
};

ReplayStore::ReplayStore(std::filesystem::path root, Compatibility compatibility, size_t capacity, uint64_t seed) : impl_(std::make_unique<Impl>()) {
    if (capacity == 0) throw std::invalid_argument("replay capacity must be positive");
    impl_->compatibility=std::move(compatibility); impl_->capacity=capacity; impl_->rng_state=seed;
    const auto digest=diamond_support::sha256(diamond_support::canonical_json(compatibility_json(impl_->compatibility)));
    impl_->namespace_path=std::move(root)/"persistent-replay-v2"/impl_->compatibility.family/digest; impl_->manifest_path=impl_->namespace_path/"manifest.json";
    if (!std::filesystem::exists(impl_->manifest_path)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied)) if (entry.path().filename()=="manifest.json" && entry.path().parent_path().parent_path().filename()=="Soo" && entry.path().parent_path().parent_path().parent_path().filename()=="persistent-replay-v1") { std::ifstream candidate(entry.path(), std::ios::binary); std::string text((std::istreambuf_iterator<char>(candidate)),{}); try { const auto parsed=diamond_support::parse_json(text); const auto& manifest=object(parsed,"manifest"); const auto& compat=object(field(manifest,"compatibility"),"compatibility"); if (string(field(compat,"model_version"),"model version") == impl_->compatibility.model_version) { impl_->manifest_path=entry.path(); impl_->namespace_path=entry.path().parent_path(); impl_->legacy=true; break; } } catch (...) {} }
    }
    if (!impl_->legacy) {
        if (!std::filesystem::exists(impl_->manifest_path)) { impl_->write_manifest(); return; }
        std::ifstream source_manifest(impl_->manifest_path, std::ios::binary); const std::string manifest_text{std::istreambuf_iterator<char>(source_manifest), std::istreambuf_iterator<char>()}; const auto parsed=diamond_support::parse_json(manifest_text); const auto& manifest=object(parsed,"manifest"); if(integer(field(manifest,"schema_version"),"schema")!=2) throw std::runtime_error("unsupported replay manifest"); const auto& chunks=array(field(manifest,"chunks"),"chunks"); const auto& ids=array(field(manifest,"game_ids"),"game ids"); if(chunks.size()!=ids.size()) throw std::runtime_error("manifest game_ids do not match ordered chunks"); for(size_t i=0;i<chunks.size();++i){const auto& descriptor=object(chunks[i],"chunk descriptor"); const auto id=string(field(descriptor,"game_id"),"game id"); if(id!=string(ids[i],"game id")) throw std::runtime_error("manifest game_ids do not match ordered chunks"); const auto digest=string(field(descriptor,"sha256"),"digest"); std::ifstream chunk_file(impl_->namespace_path/"chunks"/(digest+".json"),std::ios::binary); if(!chunk_file) throw std::runtime_error("missing replay chunk"); const std::string chunk_text{std::istreambuf_iterator<char>(chunk_file),std::istreambuf_iterator<char>()}; auto chunk=diamond_support::parse_json(chunk_text); auto payload=object(chunk,"chunk"); const auto stored=string(field(payload,"sha256"),"digest"); payload.erase("sha256"); if(stored!=digest||diamond_support::sha256(diamond_support::canonical_json(JsonValue{payload}))!=digest) throw std::runtime_error("corrupt replay chunk hash"); for(const auto& row:array(field(payload,"samples"),"samples")) impl_->samples.push_back(legacy_sample(row,impl_->compatibility)); }
        const auto& rng=object(field(manifest,"rng"),"rng"); impl_->rng_state=std::stoull(string(field(rng,"state"),"rng state")); return;
    }
    std::ifstream manifest_file(impl_->manifest_path,std::ios::binary); std::string manifest_text((std::istreambuf_iterator<char>(manifest_file)),{}); impl_->legacy_manifest=diamond_support::parse_json(manifest_text); const auto& manifest=object(impl_->legacy_manifest,"manifest");
    if (integer(field(manifest, "schema_version"), "schema") != 1) throw std::runtime_error("unsupported replay manifest");
    const auto& compat = object(field(manifest, "compatibility"), "compatibility");
    const auto model = string(field(compat, "model_name"), "model name");
    if ((model != "Soo" && model != "Min") || string(field(compat, "model_version"), "model version") != impl_->compatibility.model_version) throw std::runtime_error("replay manifest compatibility mismatch");
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
        Episode episode; episode.game_id = id; episode.completed = true;
        for (const auto& row : array(field(payload, "samples"), "samples")) {
            auto sample = legacy_sample(row, impl_->compatibility);
            episode.samples.push_back(sample);
            impl_->samples.push_back(std::move(sample));
        }
        impl_->episodes.push_back(std::move(episode));
    }
    if(impl_->samples.size()>capacity) impl_->samples.erase(impl_->samples.begin(),impl_->samples.end()-static_cast<std::ptrdiff_t>(capacity)); const auto& state=array(field(manifest,"rng_state"),"rng state"); const auto& words=array(state.at(1),"mt state"); if(integer(state.at(0),"rng version")!=3||words.size()!=625)throw std::runtime_error("invalid CPython MT19937 state"); for(size_t i=0;i<624;++i)impl_->mt_state.push_back(uint32_t(integer(words[i],"mt word"))); impl_->mt_index=size_t(integer(words[624],"mt index"));
}
ReplayStore::~ReplayStore() = default;
ReplayStore::ReplayStore(ReplayStore&&) noexcept = default;
ReplayStore& ReplayStore::operator=(ReplayStore&&) noexcept = default;

size_t ReplayStore::ingest(std::span<const Episode> episodes) {
    std::unordered_set<std::string> known; for(const auto& episode:impl_->episodes) known.insert(episode.game_id);
    std::vector<Episode> accepted; for(const Episode& episode:episodes) { if(episode.game_id.empty()) throw std::invalid_argument("replay game_id is empty"); if(episode.compatibility != impl_->compatibility) throw std::invalid_argument("replay compatibility mismatch"); if(!known.insert(episode.game_id).second) throw std::invalid_argument("conflicting duplicate game_id"); accepted.push_back(episode); }
    for(const Episode& episode:accepted) if(episode.completed) { auto body=chunk_body(episode, impl_->compatibility); const auto hash=diamond_support::sha256(diamond_support::canonical_json(body)); std::get<Object>(body.value).emplace("sha256",JsonValue{hash}); atomic_write(impl_->namespace_path/"chunks"/(hash+".json"), diamond_support::canonical_json(std::move(body))); }
    for(const auto& episode:accepted) { impl_->episodes.push_back(episode); if(episode.completed) impl_->samples.insert(impl_->samples.end(),episode.samples.begin(),episode.samples.end()); }
    if(impl_->samples.size()>impl_->capacity) impl_->samples.erase(impl_->samples.begin(),impl_->samples.end()-static_cast<std::ptrdiff_t>(impl_->capacity));
    if(!accepted.empty()) impl_->write_manifest(); return accepted.size();
}
std::vector<TrainingSample> ReplayStore::sample(size_t count) { if(count==0) return {}; if(count>impl_->samples.size()) throw std::invalid_argument("replay sample count exceeds available samples"); std::vector<TrainingSample> pool=impl_->samples, out; out.reserve(count); for(size_t i=0;i<count;++i){const size_t index=impl_->legacy?mt_below(impl_->mt_state,impl_->mt_index,pool.size()):next_splitmix(impl_->rng_state)%pool.size();out.push_back(pool[index]);pool[index]=pool.back();pool.pop_back();} impl_->write_manifest();return out; }
void ReplayStore::prune() { if(impl_->episodes.size()<=impl_->capacity)return; impl_->episodes.erase(impl_->episodes.begin(),impl_->episodes.end()-static_cast<std::ptrdiff_t>(impl_->capacity)); impl_->write_manifest(); }
void ReplayStore::restore_manifest(const std::filesystem::path& snapshot) { std::ifstream file(snapshot,std::ios::binary); if(!file) throw std::runtime_error("cannot read replay manifest snapshot"); std::string value((std::istreambuf_iterator<char>(file)),{}); (void)diamond_support::parse_json(value); atomic_write(impl_->manifest_path,value); }

}  // namespace diamond_pipeline
