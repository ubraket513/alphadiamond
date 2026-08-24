#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace diamond_model {

// A validated deployment bundle, described by its own metadata rather than by
// constants compiled into this binary. Format 3 declares the family and the
// architecture; the loader checks the weights against what was declared, which
// is what lets a second model family ship without a format change.
struct DeploymentArtifact {
    std::filesystem::path root;
    std::filesystem::path weights;
    std::string model_family;       // "soo" or "min"
    std::string model_version;
    std::string model_sha256;
    std::string runtime_sha256;
    int64_t width = 0;
    int64_t residual_blocks = 0;
    int64_t input_features = 0;     // per hole
    int64_t value_size = 0;         // value-head outputs
};

// Strictly validates the versioned Python-exported deployment bundle before
// any raw tensor is loaded by the production runtime.
DeploymentArtifact validate_deployment_artifact(const std::filesystem::path& root);

// Same, and additionally requires the declared family.
DeploymentArtifact validate_deployment_artifact(const std::filesystem::path& root,
                                                const std::string& expected_family);

// Writes the release representation of a validated format-v3 artifact.  The
// destination must not already exist.  Only metadata.json, the topology
// tables, and weights/ are copied: that is precisely the runtime_sha256
// manifest.  The destination is validated before it is returned.
DeploymentArtifact write_runtime_deployment_artifact(
    const std::filesystem::path& source_root,
    const std::filesystem::path& destination_root);

// Soo-specific spelling kept for the existing probes and the Qt runtime.
inline DeploymentArtifact validate_soo_deployment_artifact(
    const std::filesystem::path& root) {
    return validate_deployment_artifact(root, "soo");
}

}  // namespace diamond_model
