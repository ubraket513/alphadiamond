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

// The training seed for one minibatch.  Sampling is a pure function of the
// replay contents and this seed, so the same replay contents replayed at the
// same iteration and local step always yield the same minibatch -- which is
// what makes a killed TRAIN stage restartable from its beginning without any
// durable sampler state.
uint64_t replay_sampling_seed(uint64_t replay_seed, uint64_t iteration, uint64_t training_step);

// How much of the store a caller needs.  `metadata_only` reads the manifest and
// nothing else -- no chunk file is opened and no sample is materialised -- which
// is all a stage needs when it only wants the manifest digest or the episode
// index. Reading 1M samples back out of JSON to hash one file was costing ~100 s
// per stage, three times per training iteration.
enum class ReplayContents { full, metadata_only };
enum class ReplayOpenMode { create_if_missing, must_exist };

class ReplayStore {
  public:
    ReplayStore(std::filesystem::path root, Compatibility compatibility, std::size_t capacity,
                uint64_t seed, ReplayContents contents = ReplayContents::full,
                ReplayOpenMode open_mode = ReplayOpenMode::create_if_missing);
    std::size_t ingest(std::span<const Episode> episodes);
    ReplayIngestReport ingest_iteration(std::span<const Episode> episodes);
    std::size_t size() const noexcept;
    std::filesystem::path manifest_path() const;
    std::string manifest_digest() const;
    uint64_t replay_seed() const noexcept;
    // Pure: reads memory and copies rows.  It touches no file and mutates no
    // sampler state, so it may be called any number of times in any order.
    std::vector<TrainingSample> sample(std::size_t count, uint64_t seed) const;
    ReplaySamplingStats last_sampling_stats() const noexcept;
    void prune();

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
