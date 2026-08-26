#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "diamond_orchestration/config.hpp"

namespace diamond_orchestration {

class ScheduleError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct SooRatedMatch final {
    std::array<std::string, 2> participant_ids;
    std::array<int, 2> seat_assignment;
    std::array<int, 2> turn_order;
    std::string opening_id;
    std::string match_id;

    void validate() const;
    bool operator==(const SooRatedMatch&) const = default;
};

struct MinRatedMatch final {
    std::array<std::string, 3> participant_ids;
    std::array<int, 3> seat_assignment;
    std::array<int, 3> turn_order;
    std::string opening_id;
    std::string match_id;

    void validate() const;
    bool operator==(const MinRatedMatch&) const = default;
};

// This is a canonical opening-generation payload, not a game-engine state.
// train_main owns applying its seed/depth to the model-specific Match.
struct MaterializedOpening final {
    std::string opening_id;
    std::string serialized_state;
    std::string state_sha256;
    uint64_t seed = 0;
    int64_t depth = 0;

    void validate() const;
    bool operator==(const MaterializedOpening&) const = default;
};

struct MaterializedOpeningSuite final {
    OpeningSuiteConfig config;
    std::vector<MaterializedOpening> openings;
    std::string suite_sha256;

    void validate() const;
    bool operator==(const MaterializedOpeningSuite&) const = default;
};

MaterializedOpeningSuite materialize_opening_suite(const OpeningSuiteConfig& config);
std::vector<std::string> opening_ids(const MaterializedOpeningSuite& suite);

std::vector<SooRatedMatch> schedule_soo_pair(
    const std::vector<std::string>& opening_ids,
    const std::array<std::string, 2>& participant_ids);
std::vector<MinRatedMatch> schedule_min_triple(
    const std::vector<std::string>& opening_ids,
    const std::array<std::string, 3>& participant_ids);
std::vector<SooRatedMatch> schedule_soo_pair(const MaterializedOpeningSuite& suite,
                                             const std::array<std::string, 2>& participant_ids);
std::vector<MinRatedMatch> schedule_min_triple(const MaterializedOpeningSuite& suite,
                                               const std::array<std::string, 3>& participant_ids);

void validate_soo_rated_batch(const std::vector<SooRatedMatch>& matches,
                               const std::vector<std::string>& opening_ids);
void validate_min_rated_batch(const std::vector<MinRatedMatch>& matches,
                               const std::vector<std::string>& opening_ids);

struct SooOpeningBlock final {
    MaterializedOpening opening;
    std::vector<SooRatedMatch> matches;

    void validate() const;
    bool operator==(const SooOpeningBlock&) const = default;
};

struct MinOpeningBlock final {
    MaterializedOpening opening;
    std::vector<MinRatedMatch> matches;

    void validate() const;
    bool operator==(const MinOpeningBlock&) const = default;
};

std::vector<SooOpeningBlock>
schedule_soo_opening_blocks(const MaterializedOpeningSuite& suite,
                            const std::array<std::string, 2>& participant_ids);
std::vector<MinOpeningBlock>
schedule_min_opening_blocks(const MaterializedOpeningSuite& suite,
                            const std::array<std::string, 3>& participant_ids);

struct SooRatedResult final {
    std::string match_id;
    std::optional<bool> candidate_won;
};

struct MinRatedResult final {
    std::string match_id;
    std::optional<int> candidate_placement;
};

struct SooOpeningBlockResult final {
    std::string opening_id;
    std::vector<SooRatedResult> results;
};

struct MinOpeningBlockResult final {
    std::string opening_id;
    std::vector<MinRatedResult> results;
};

struct OpeningBlockSummary final {
    int64_t completed_matches = 0;
    int64_t aborted_matches = 0;
    int64_t complete_blocks = 0;
    int64_t incomplete_blocks = 0;
    std::vector<double> complete_block_scores;
};

OpeningBlockSummary summarize_soo_opening_blocks(const std::vector<SooOpeningBlock>& blocks,
                                                 const std::vector<SooOpeningBlockResult>& results);
OpeningBlockSummary summarize_min_opening_blocks(const std::vector<MinOpeningBlock>& blocks,
                                                 const std::vector<MinOpeningBlockResult>& results);

struct OpeningBlockBootstrap final {
    int64_t complete_blocks = 0;
    int64_t incomplete_blocks = 0;
    double point_estimate = 0.0;
    double confidence_lower = 0.0;
    double confidence_upper = 0.0;
    bool promoted = false;

    bool operator==(const OpeningBlockBootstrap&) const = default;
};

OpeningBlockBootstrap bootstrap_opening_blocks(const OpeningBlockSummary& summary,
                                               const PromotionStatisticsConfig& config,
                                               double promotion_threshold);

}  // namespace diamond_orchestration
