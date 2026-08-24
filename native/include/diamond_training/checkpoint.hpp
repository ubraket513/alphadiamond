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
// generation.  A legacy .pt file is deliberately not a valid root.
CheckpointInfo inspect_checkpoint_v2(const std::filesystem::path& root);
CheckpointInfo save_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer);
CheckpointInfo load_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer);

}  // namespace diamond_training
