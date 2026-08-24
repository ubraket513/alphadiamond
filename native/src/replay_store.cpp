#include "diamond_pipeline/replay_store.hpp"

#include <algorithm>
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
uint64_t next_splitmix(uint64_t& state) { state += 0x9e3779b97f4a7c15ULL; uint64_t value=state; value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL; value=(value^(value>>27))*0x94d049bb133111ebULL; return value^(value>>31); }
void atomic_write(const std::filesystem::path& path, const std::string& contents) { std::filesystem::create_directories(path.parent_path()); const auto temporary=path.string()+".tmp"; { std::ofstream file(temporary, std::ios::binary|std::ios::trunc); if(!file) throw std::runtime_error("cannot write replay transaction"); file<<contents; if(!file) throw std::runtime_error("cannot write replay transaction"); } std::filesystem::rename(temporary,path); }
}  // namespace

struct ReplayStore::Impl {
    std::filesystem::path namespace_path;
    std::filesystem::path manifest_path;
    Compatibility compatibility;
    size_t capacity;
    uint64_t rng_state;
    std::vector<Episode> episodes;
    std::vector<TrainingSample> samples;

    void write_manifest() const {
        Array chunks, game_ids; for (const Episode& episode : episodes) if (episode.completed) { const auto digest=diamond_support::sha256(episode.game_id); chunks.emplace_back(JsonValue{Object{{"game_id",JsonValue{episode.game_id}}, {"sample_count",JsonValue{int64_t(episode.samples.size())}}, {"sha256",JsonValue{digest}}}}); game_ids.emplace_back(JsonValue{episode.game_id}); }
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
    if (!std::filesystem::exists(impl_->manifest_path)) impl_->write_manifest();
}
ReplayStore::~ReplayStore() = default;
ReplayStore::ReplayStore(ReplayStore&&) noexcept = default;
ReplayStore& ReplayStore::operator=(ReplayStore&&) noexcept = default;

size_t ReplayStore::ingest(std::span<const Episode> episodes) {
    std::unordered_set<std::string> known; for(const auto& episode:impl_->episodes) known.insert(episode.game_id);
    std::vector<Episode> accepted; for(const Episode& episode:episodes) { if(episode.game_id.empty()) throw std::invalid_argument("replay game_id is empty"); if(episode.compatibility != impl_->compatibility) throw std::invalid_argument("replay compatibility mismatch"); if(!known.insert(episode.game_id).second) throw std::invalid_argument("conflicting duplicate game_id"); accepted.push_back(episode); }
    for(const Episode& episode:accepted) if(episode.completed) { Array rows; for(const auto& sample:episode.samples) rows.push_back(sample_json(sample)); Object body{{"compatibility",compatibility_json(impl_->compatibility)}, {"episode",JsonValue{Object{{"completed",JsonValue{true}}, {"game_id",JsonValue{episode.game_id}}, {"move_count",JsonValue{int64_t(episode.move_count)}}, {"retry_id",JsonValue{episode.retry_id}}, {"seed",JsonValue{int64_t(episode.seed)}}}}}, {"samples",JsonValue{std::move(rows)}}, {"schema_version",JsonValue{int64_t(1)}}}; const auto hash=diamond_support::sha256(diamond_support::canonical_json(JsonValue{body})); body.emplace("sha256",JsonValue{hash}); atomic_write(impl_->namespace_path/"chunks"/(diamond_support::sha256(episode.game_id)+".json"), diamond_support::canonical_json(JsonValue{std::move(body)})); }
    for(const auto& episode:accepted) { impl_->episodes.push_back(episode); if(episode.completed) impl_->samples.insert(impl_->samples.end(),episode.samples.begin(),episode.samples.end()); }
    if(impl_->samples.size()>impl_->capacity) impl_->samples.erase(impl_->samples.begin(),impl_->samples.end()-static_cast<std::ptrdiff_t>(impl_->capacity));
    if(!accepted.empty()) impl_->write_manifest(); return accepted.size();
}
std::vector<TrainingSample> ReplayStore::sample(size_t count) { if(count==0) return {}; if(count>impl_->samples.size()) throw std::invalid_argument("replay sample count exceeds available samples"); std::vector<TrainingSample> pool=impl_->samples, out; out.reserve(count); for(size_t i=0;i<count;++i){const size_t index=next_splitmix(impl_->rng_state)%(pool.size());out.push_back(pool[index]);pool[index]=pool.back();pool.pop_back();} impl_->write_manifest();return out; }
void ReplayStore::prune() { if(impl_->episodes.size()<=impl_->capacity)return; impl_->episodes.erase(impl_->episodes.begin(),impl_->episodes.end()-static_cast<std::ptrdiff_t>(impl_->capacity)); impl_->write_manifest(); }
void ReplayStore::restore_manifest(const std::filesystem::path& snapshot) { std::ifstream file(snapshot,std::ios::binary); if(!file) throw std::runtime_error("cannot read replay manifest snapshot"); std::string value((std::istreambuf_iterator<char>(file)),{}); (void)diamond_support::parse_json(value); atomic_write(impl_->manifest_path,value); }

}  // namespace diamond_pipeline
