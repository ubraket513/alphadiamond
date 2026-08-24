#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "diamond_model/model_index.hpp"
#include "diamond_support/json.hpp"

namespace diamond_release {

class ReleaseError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class PromotionState { archival, candidate, promoted };

struct PromotionManifest final {
    PromotionState state = PromotionState::archival;
    std::string model_family;
    std::string checkpoint_generation;
    std::string checkpoint_sha256;
    diamond_support::JsonValue::Object payload;
};

std::string promotion_state_name(PromotionState state);
PromotionState parse_promotion_state(const std::string& value);
PromotionManifest initialize_promotion(const std::filesystem::path& checkpoint_dir,
                                       const std::string& model_family);
PromotionManifest load_promotion_manifest(const std::filesystem::path& checkpoint_dir);
PromotionManifest promote_checkpoint(
    const std::filesystem::path& checkpoint_dir, PromotionState target,
    const std::optional<std::filesystem::path>& deployment_artifact = std::nullopt);

// Build a fresh runtime-only models tree and atomically activate it at output.
// Full development artifacts are validated at input but only their runtime
// metadata, topology, and weights are staged.
diamond_model::ModelIndex stage_release_models(
    const std::filesystem::path& output,
    const std::vector<std::filesystem::path>& deployment_artifacts,
    const std::vector<std::string>& default_families = {});

}  // namespace diamond_release
