#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "diamond_pipeline/replay.hpp"

namespace diamond_pipeline {

class ReplayStore {
  public:
    ReplayStore(std::filesystem::path root, Compatibility compatibility,
                std::size_t capacity, uint64_t seed);
    std::size_t ingest(std::span<const Episode> episodes);
    std::vector<TrainingSample> sample(std::size_t count);
    void prune();
    void restore_manifest(const std::filesystem::path& snapshot);

    ~ReplayStore();
    ReplayStore(ReplayStore&&) noexcept;
    ReplayStore& operator=(ReplayStore&&) noexcept;
    ReplayStore(const ReplayStore&) = delete;
    ReplayStore& operator=(const ReplayStore&) = delete;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace diamond_pipeline
