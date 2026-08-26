#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

#include "diamond_training/trainer.hpp"

namespace diamond_training {

class CheckpointError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class CheckpointInitializationMode {
    scratch,
    warm_start,
    resume,
    audited_legacy_import,
};

enum class CheckpointLoadIntent {
    exact_resume,
    warm_start,
    audited_legacy_import,
};

// v3 records the origin of a run. Digests are lowercase SHA-256 hex strings.
// `model_step` is the native model/training step represented by this generation.
struct CheckpointLineage {
    CheckpointInitializationMode initialization_mode = CheckpointInitializationMode::scratch;
    std::string run_id;
    uint64_t iteration = 0;
    uint64_t model_step = 0;
    std::optional<std::string> parent_digest;
    std::optional<std::string> source_digest;
    std::optional<uint64_t> source_training_step;
    bool optimizer_restored = false;
    bool optimizer_reset = false;
    std::optional<std::string> optimizer_reset_reason;
};

// Section-F provenance. Callers should provide immutable source/config/replay
// facts; trainer/device/architecture facts are recorded by save_checkpoint_v3.
struct CheckpointProvenance {
    std::string source_git_commit = "unavailable";
    std::string resolved_config_bytes = "{}";
    std::string replay_manifest_sha256 =
        "0000000000000000000000000000000000000000000000000000000000000000";
    std::string protocol_ids_json = "{}";
    std::string creation_timestamp = "unavailable";
    std::string environment_json = "{}";
    // LibTorch has no documented stable C++ CPU RNG state archive API in the
    // supported build. This records that verified limitation explicitly.
    std::string rng_state_status = "gap_no_stable_libtorch_cpp_api";
    uint64_t rng_state_version = 0;
};

struct CheckpointInfo {
    std::filesystem::path generation;
    uint64_t training_step = 0;
    uint64_t format_version = 2;
    std::optional<CheckpointLineage> lineage;
    std::string model_digest;
    std::string optimizer_digest;
    std::optional<CheckpointProvenance> provenance;
};

// A checkpoint root uses CURRENT to one fully-written immutable v2 or v3
// generation. Each generation contains state.pt, optimizer.pt, and training_step.
// A legacy .pt file is deliberately not a valid root.
CheckpointInfo inspect_checkpoint_v2(const std::filesystem::path& root);
// Verify the active generation is structurally complete and both LibTorch
// archives can be read. This does not load weights into a model.
CheckpointInfo validate_checkpoint_v2(const std::filesystem::path& root);
// Copy the active v2 generation into a new checkpoint root without changing
// its payload. The destination must not already exist.
CheckpointInfo migrate_checkpoint_v2(const std::filesystem::path& source,
                                     const std::filesystem::path& destination);
CheckpointInfo save_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer);
// Writes the same immutable generation/CURRENT transaction as v2, with a
// strict v3 manifest carrying explicit initialization lineage.
CheckpointInfo save_checkpoint_v3(const std::filesystem::path& root, Trainer& trainer,
                                  const CheckpointLineage& lineage,
                                  const CheckpointProvenance& provenance = {});
CheckpointInfo load_checkpoint_v2_weights(const std::filesystem::path& root,
                                          diamond_model::DiamondModel& model,
                                          const ResolvedDevice& target);
CheckpointInfo load_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer,
                                  const ResolvedDevice& target);
// `exact_resume` restores model and AdamW state transactionally. `warm_start`
// restores weights into a fresh optimizer at step zero. Legacy import is
// deliberately gated: this native checkpoint reader cannot claim its parity.
CheckpointInfo load_checkpoint_v3(const std::filesystem::path& root, Trainer& trainer,
                                  const ResolvedDevice& target, CheckpointLoadIntent intent);

}  // namespace diamond_training
