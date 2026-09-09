#pragma once
#include "diamond_model/deployment_artifact.hpp"

namespace diamond_training {
// Export a native v3 checkpoint without changing it. Verifies canonical topology,
// finite weights, strict artifact validation and exact CPU inference parity.
// Raw-only artifacts identify the model by its canonical native tensor digest.
diamond_model::DeploymentArtifact export_checkpoint(
    const std::filesystem::path& checkpoint, const std::filesystem::path& destination,
    const std::string& version);
}
