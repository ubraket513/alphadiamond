#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include "diamond_pipeline/replay_store.hpp"
#include "diamond_training/training_sample.hpp"

namespace diamond_orchestration {

class TrainingRunResources final {
  public:
    TrainingRunResources(std::filesystem::path replay_root,
                         diamond_training::Compatibility compatibility, std::size_t replay_capacity,
                         uint64_t replay_seed);
    ~TrainingRunResources();

    TrainingRunResources(const TrainingRunResources&) = delete;
    TrainingRunResources& operator=(const TrainingRunResources&) = delete;

    bool replay_loaded() const noexcept;
    diamond_pipeline::ReplayStore& full_replay();

  private:
    std::filesystem::path replay_root_;
    diamond_training::Compatibility compatibility_;
    std::size_t replay_capacity_;
    uint64_t replay_seed_;
    std::unique_ptr<diamond_pipeline::ReplayStore> replay_;
};

} // namespace diamond_orchestration
