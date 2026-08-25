#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>

#include "diamond_training/trainer.hpp"

namespace diamond_training {

class CheckpointError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct CheckpointInfo {
    std::filesystem::path generation;
    uint64_t training_step = 0;
};

// A checkpoint root is v2-only: CURRENT points to one fully-written immutable
// generation. Each generation contains state.pt (model parameters), optimizer.pt,
// and training_step. A legacy .pt file is deliberately not a valid root.
CheckpointInfo inspect_checkpoint_v2(const std::filesystem::path& root);
// Verify the active generation is structurally complete and both LibTorch
// archives can be read. This does not load weights into a model.
CheckpointInfo validate_checkpoint_v2(const std::filesystem::path& root);
// Copy the active v2 generation into a new checkpoint root without changing
// its payload. The destination must not already exist.
CheckpointInfo migrate_checkpoint_v2(const std::filesystem::path& source,
                                     const std::filesystem::path& destination);
CheckpointInfo save_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer);
CheckpointInfo load_checkpoint_v2_weights(const std::filesystem::path& root,
                                          diamond_model::DiamondModel& model,
                                          const ResolvedDevice& target);
CheckpointInfo load_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer,
                                  const ResolvedDevice& target);

}  // namespace diamond_training
