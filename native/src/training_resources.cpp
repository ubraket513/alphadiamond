#include "diamond_orchestration/training_resources.hpp"

#include <utility>

namespace diamond_orchestration {

TrainingRunResources::TrainingRunResources(std::filesystem::path replay_root,
                                           diamond_training::Compatibility compatibility,
                                           std::size_t replay_capacity, uint64_t replay_seed)
    : replay_root_(std::move(replay_root)), compatibility_(std::move(compatibility)),
      replay_capacity_(replay_capacity), replay_seed_(replay_seed) {}

TrainingRunResources::~TrainingRunResources() = default;

bool TrainingRunResources::replay_loaded() const noexcept {
    return replay_ != nullptr;
}

diamond_pipeline::ReplayStore& TrainingRunResources::full_replay() {
    if (!replay_) {
        replay_ = std::make_unique<diamond_pipeline::ReplayStore>(
            replay_root_, compatibility_, replay_capacity_, replay_seed_,
            diamond_pipeline::ReplayContents::full);
    }
    return *replay_;
}

} // namespace diamond_orchestration
