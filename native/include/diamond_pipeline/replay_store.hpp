#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "diamond_pipeline/replay.hpp"

namespace diamond_pipeline {

struct ReplayIngestReport {
    std::size_t accepted_games = 0;
    std::size_t duplicate_games = 0;
    std::size_t accepted_samples = 0;
    std::size_t duplicate_samples = 0;
};

struct ReplaySamplingStats {
    std::size_t selection_slots = 0;
    std::size_t copied_samples = 0;
};

class ReplayStore {
  public:
    ReplayStore(std::filesystem::path root, Compatibility compatibility,
                std::size_t capacity, uint64_t seed);
    std::size_t ingest(std::span<const Episode> episodes);
    ReplayIngestReport ingest_iteration(std::span<const Episode> episodes);
    std::size_t size() const noexcept;
    std::filesystem::path manifest_path() const;
    std::string manifest_digest() const;
    std::vector<TrainingSample> sample(std::size_t count);
    ReplaySamplingStats last_sampling_stats() const noexcept;
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
