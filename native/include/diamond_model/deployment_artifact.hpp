#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace diamond_model {

struct SooDeploymentArtifact {
    std::filesystem::path root;
    std::filesystem::path weights;
    std::string model_version;
    int64_t width = 0;
    int64_t residual_blocks = 0;
};

// Strictly validates the versioned Python-exported deployment bundle before
// any raw tensor is loaded by the production runtime.
SooDeploymentArtifact validate_soo_deployment_artifact(
    const std::filesystem::path& root);

}  // namespace diamond_model
