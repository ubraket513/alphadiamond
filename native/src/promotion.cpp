#include "diamond_release/promotion.hpp"

#include <atomic>
#include <fstream>
#include <iterator>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "diamond_model/deployment_artifact.hpp"
#include "diamond_training/checkpoint.hpp"

namespace diamond_release {
namespace {

using Json = diamond_support::JsonValue;
using Object = Json::Object;

const Json& field(const Object& object, const char* name) {
    const auto found = object.find(name);
    if (found == object.end()) throw ReleaseError(std::string("promotion manifest is missing ") + name);
    return found->second;
}

const std::string& text(const Json& value, const char* name) {
    const auto* result = std::get_if<std::string>(&value.value);
    if (!result || result->empty())
        throw ReleaseError(std::string("promotion manifest has invalid ") + name);
    return *result;
}

Object read_object(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw ReleaseError("cannot open release manifest: " + path.string());
    const std::string contents((std::istreambuf_iterator<char>(input)), {});
    auto parsed = diamond_support::parse_json(contents);
    auto* object = std::get_if<Object>(&parsed.value);
    if (!object) throw ReleaseError("release manifest must be a JSON object");
    return std::move(*object);
}

std::string file_sha256(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw ReleaseError("cannot open immutable checkpoint: " + path.string());
    const std::string bytes((std::istreambuf_iterator<char>(input)), {});
    return diamond_support::sha256(bytes);
}

void atomic_write(const std::filesystem::path& path, const std::string& contents) {
    static std::atomic_uint64_t sequence{0};
    const auto temporary = std::filesystem::path(path.string() + ".tmp." +
                                                  std::to_string(++sequence));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw ReleaseError("cannot write release manifest");
        output << contents;
        if (!output) throw ReleaseError("cannot write release manifest");
    }
#ifdef _WIN32
    if (!MoveFileExW(temporary.wstring().c_str(), path.wstring().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const auto error = GetLastError();
        std::filesystem::remove(temporary);
        throw ReleaseError("cannot activate release manifest: Windows error " +
                           std::to_string(error));
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw ReleaseError("cannot activate release manifest: " + error.message());
    }
#endif
}

PromotionManifest decoded(Object payload) {
    const auto* version = std::get_if<int64_t>(&field(payload, "manifest_version").value);
    if (!version || *version != 2) throw ReleaseError("unsupported promotion manifest version");
    PromotionManifest result;
    result.state = parse_promotion_state(text(field(payload, "state"), "state"));
    result.model_family = text(field(payload, "model_family"), "model_family");
    if (result.model_family != "soo" && result.model_family != "min")
        throw ReleaseError("promotion manifest has invalid model_family");
    result.checkpoint_generation =
        text(field(payload, "checkpoint_generation"), "checkpoint_generation");
    result.checkpoint_sha256 = text(field(payload, "checkpoint_sha256"), "checkpoint_sha256");
    if (result.checkpoint_sha256.size() != 64)
        throw ReleaseError("promotion manifest has invalid checkpoint_sha256");
    result.payload = std::move(payload);
    return result;
}

int ordinal(PromotionState state) { return static_cast<int>(state); }

}  // namespace

std::string promotion_state_name(PromotionState state) {
    switch (state) {
        case PromotionState::archival: return "archival";
        case PromotionState::candidate: return "candidate";
        case PromotionState::promoted: return "promoted";
    }
    throw ReleaseError("unknown promotion state");
}

PromotionState parse_promotion_state(const std::string& value) {
    if (value == "archival") return PromotionState::archival;
    if (value == "candidate") return PromotionState::candidate;
    if (value == "promoted") return PromotionState::promoted;
    throw ReleaseError("unknown promotion state: " + value);
}

PromotionManifest initialize_promotion(const std::filesystem::path& checkpoint_dir,
                                       const std::string& model_family) {
    if (model_family != "soo" && model_family != "min")
        throw ReleaseError("model family must be soo or min");
    const auto manifest_path = checkpoint_dir / "promotion.json";
    if (std::filesystem::exists(manifest_path))
        throw ReleaseError("promotion manifest already exists");
    const auto checkpoint = diamond_training::validate_checkpoint_v2(checkpoint_dir);
    Object payload{
        {"checkpoint_generation", Json{checkpoint.generation.filename().string()}},
        {"checkpoint_sha256", Json{file_sha256(checkpoint.generation / "state.pt")}},
        {"manifest_version", Json{int64_t{2}}},
        {"model_family", Json{model_family}},
        {"state", Json{std::string{"archival"}}},
        {"training_step", Json{static_cast<int64_t>(checkpoint.training_step)}},
    };
    atomic_write(manifest_path,
                 diamond_support::canonical_json(Json{payload}) + "\n");
    return decoded(std::move(payload));
}

PromotionManifest load_promotion_manifest(const std::filesystem::path& checkpoint_dir) {
    auto result = decoded(read_object(checkpoint_dir / "promotion.json"));
    const auto checkpoint = diamond_training::validate_checkpoint_v2(checkpoint_dir);
    if (checkpoint.generation.filename().string() != result.checkpoint_generation)
        throw ReleaseError("active checkpoint generation changed after release initialization");
    const auto actual = file_sha256(checkpoint.generation / "state.pt");
    if (actual != result.checkpoint_sha256)
        throw ReleaseError("immutable checkpoint digest does not match its manifest");
    return result;
}

PromotionManifest promote_checkpoint(
    const std::filesystem::path& checkpoint_dir, PromotionState target,
    const std::optional<std::filesystem::path>& deployment_artifact) {
    auto manifest = load_promotion_manifest(checkpoint_dir);
    if (target != manifest.state && ordinal(target) != ordinal(manifest.state) + 1)
        throw ReleaseError("promotion states move forward exactly one step");

    if (target == PromotionState::promoted) {
        if (!deployment_artifact)
            throw ReleaseError("promoted state requires a validated deployment artifact");
        const auto artifact = diamond_model::validate_deployment_artifact(
            *deployment_artifact, manifest.model_family);
        manifest.payload["deployment"] = Json{Object{
            {"artifact_path", Json{deployment_artifact->generic_string()}},
            {"model_sha256", Json{artifact.model_sha256}},
            {"runtime_sha256", Json{artifact.runtime_sha256}},
        }};
    }
    manifest.state = target;
    manifest.payload["state"] = Json{promotion_state_name(target)};
    atomic_write(checkpoint_dir / "promotion.json",
                 diamond_support::canonical_json(Json{manifest.payload}) + "\n");
    return decoded(manifest.payload);
}

diamond_model::ModelIndex stage_release_models(
    const std::filesystem::path& output,
    const std::vector<std::filesystem::path>& deployment_artifacts,
    const std::vector<std::string>& default_families) {
    if (deployment_artifacts.empty()) throw ReleaseError("release requires at least one artifact");
    if (std::filesystem::exists(output)) throw ReleaseError("release output already exists");
    const auto staging = std::filesystem::path(output.string() + ".staging");
    if (std::filesystem::exists(staging)) throw ReleaseError("release staging path already exists");

    std::vector<std::filesystem::path> staged_artifacts;
    try {
        std::filesystem::create_directories(staging);
        for (const auto& source : deployment_artifacts) {
            const auto artifact = diamond_model::validate_deployment_artifact(source);
            const auto destination = staging / artifact.model_family / artifact.model_version;
            (void)diamond_model::write_runtime_deployment_artifact(source, destination);
            staged_artifacts.push_back(destination);
        }
        (void)diamond_model::write_model_index(staging, staged_artifacts,
                                               default_families);
        std::filesystem::rename(staging, output);
        return diamond_model::load_model_index(output);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
        throw;
    }
}

}  // namespace diamond_release
