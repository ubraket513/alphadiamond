#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace diamond_model {

// One model shipped beside the executable.
struct ModelIndexEntry {
    std::string family;
    std::string version;
    std::filesystem::path root;   // absolute: models/<family>/<version>
    std::string model_sha256;
    std::string runtime_sha256;
};

// The contents of models/index.json, as the application sees it.
//
// Models ship beside the binary rather than compiled into it, so a release can
// carry a different default without relinking, and a model can be replaced
// without shipping a new executable.
struct ModelIndex {
    std::vector<ModelIndexEntry> models;

    // The default entry for a family, or nullptr when the release does not
    // bundle one. Callers must handle nullptr: an application that ships
    // without a model still has to start.
    const ModelIndexEntry* default_for(const std::string& family) const;

  private:
    friend ModelIndex load_model_index(const std::filesystem::path&);
    std::vector<std::pair<std::string, std::string>> defaults_;  // family -> "<family>/<version>"
};

// Reads and validates models/index.json under `models_dir`. Throws when the
// file is missing or malformed; an entry pointing at an artifact is not itself
// validated here -- call validate_deployment_artifact before loading weights.
ModelIndex load_model_index(const std::filesystem::path& models_dir);

// Writes a canonical index.json for already-staged runtime artifacts.  Every
// root must be exactly models_dir/<family>/<version>; this keeps index paths
// package-relative and makes the emitted path unambiguous.  `index.json` must
// not already exist, and every default family must name exactly one artifact.
ModelIndex write_model_index(const std::filesystem::path& models_dir,
                             const std::vector<std::filesystem::path>& artifact_roots,
                             const std::vector<std::string>& default_families);

}  // namespace diamond_model
