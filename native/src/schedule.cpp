#include "diamond_orchestration/schedule.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include "diamond_support/json.hpp"

namespace diamond_orchestration {
namespace {

using Json = diamond_support::JsonValue;

std::string digest(const Json& value) {
    return "sha256:" + diamond_support::sha256(diamond_support::canonical_json(value));
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

Json opening_state_payload(const OpeningSuiteConfig& config, int64_t index, uint64_t seed,
                           int64_t depth) {
    return Json{Json::Object{{"depth", Json{depth}},
                             {"opening_index", Json{index}},
                             {"opening_seed", Json{std::to_string(seed)}},
                             {"protocol", Json{std::string{"opening-suite-v1"}}},
                             {"suite", config.to_json()}}};
}

Json suite_payload(const MaterializedOpeningSuite& suite) {
    Json::Array openings;
    openings.reserve(suite.openings.size());
    for (const auto& opening : suite.openings)
        openings.emplace_back(Json{Json::Object{{"depth", Json{opening.depth}},
                                                {"opening_id", Json{opening.opening_id}},
                                                {"seed", Json{std::to_string(opening.seed)}},
                                                {"state_sha256", Json{opening.state_sha256}}}});
    return Json{Json::Object{{"openings", Json{std::move(openings)}},
                             {"protocol", Json{std::string{"opening-suite-v1"}}},
                             {"suite", suite.config.to_json()}}};
}

template <size_t Count>
void validate_participants(const std::array<std::string, Count>& ids) {
    std::set<std::string> distinct;
    for (const auto& id : ids) {
        if (id.empty()) throw ScheduleError("participant IDs must be non-empty");
        distinct.insert(id);
    }
    if (distinct.size() != Count) throw ScheduleError("participant IDs must be distinct");
}

template <size_t Count>
void validate_permutation(const std::array<int, Count>& values, const char* name) {
    std::array<int, Count> expected{};
    for (size_t index = 0; index < Count; ++index) expected[index] = static_cast<int>(index + 1);
    auto actual = values;
    std::sort(actual.begin(), actual.end());
    if (actual != expected) throw ScheduleError(std::string(name) + " must be a physical-seat permutation");
}

template <size_t Count>
Json string_array(const std::array<std::string, Count>& values) {
    Json::Array result;
    for (const auto& value : values) result.emplace_back(value);
    return Json{std::move(result)};
}

template <size_t Count>
Json int_array(const std::array<int, Count>& values) {
    Json::Array result;
    for (int value : values) result.emplace_back(int64_t{value});
    return Json{std::move(result)};
}

template <size_t Count>
std::string match_id(const char* type, const std::array<std::string, Count>& participants,
                     const std::array<int, Count>& seats, const std::array<int, Count>& order,
                     const std::string& opening) {
    Json value{Json::Object{{"match_type", Json{std::string(type)}},
                            {"opening_id", Json{opening}},
                            {"participant_ids", string_array(participants)},
                            {"seat_assignment", int_array(seats)},
                            {"turn_order", int_array(order)}}};
    return "sha256:" + diamond_support::sha256(diamond_support::canonical_json(value));
}

template <size_t Count>
void validate_openings(const std::vector<std::string>& openings) {
    if (openings.empty()) throw ScheduleError("opening IDs must be non-empty");
    std::set<std::string> distinct;
    for (const auto& opening : openings) {
        if (opening.empty()) throw ScheduleError("opening IDs must be non-empty");
        distinct.insert(opening);
    }
    if (distinct.size() != openings.size()) throw ScheduleError("opening IDs must be distinct");
}

template <size_t Count, class Match>
void validate_batch(const std::vector<Match>& matches, const std::vector<std::string>& openings,
                    std::vector<Match> (*build)(const std::vector<std::string>&,
                                                 const std::array<std::string, Count>&)) {
    validate_openings<Count>(openings);
    if (matches.empty()) throw ScheduleError("rated batch must be non-empty");
    for (const auto& match : matches) match.validate();
    const auto expected = build(openings, matches.front().participant_ids);
    if (matches.size() != expected.size()) throw ScheduleError("rated batch must contain a complete balance cycle");
    std::set<std::string> ids;
    for (const auto& match : matches) ids.insert(match.match_id);
    if (ids.size() != matches.size()) throw ScheduleError("rated batch contains duplicate matches");
    if (matches != expected) throw ScheduleError("rated batch is incompatible with its complete balance schedule");
}

template <class Block, class BlockResult, class Result, class Match, class Score>
OpeningBlockSummary summarize_blocks(const std::vector<Block>& blocks,
                                     const std::vector<BlockResult>& results, Score score) {
    if (blocks.empty())
        throw ScheduleError("opening blocks must be non-empty");
    if (blocks.size() != results.size())
        throw ScheduleError("opening block results are missing blocks");
    std::map<std::string, const Block*> expected_blocks;
    for (const auto& block : blocks) {
        block.validate();
        if (!expected_blocks.emplace(block.opening.opening_id, &block).second)
            throw ScheduleError("opening blocks contain duplicate openings");
    }

    OpeningBlockSummary summary;
    std::set<std::string> seen_blocks;
    for (const auto& block_result : results) {
        const auto found = expected_blocks.find(block_result.opening_id);
        if (block_result.opening_id.empty() || found == expected_blocks.end())
            throw ScheduleError("opening block result has an unknown opening");
        if (!seen_blocks.insert(block_result.opening_id).second)
            throw ScheduleError("opening block results contain duplicate openings");

        std::set<std::string> expected_ids;
        for (const Match& match : found->second->matches)
            expected_ids.insert(match.match_id);
        std::set<std::string> seen_ids;
        double total = 0.0;
        bool complete = true;
        for (const Result& result : block_result.results) {
            if (result.match_id.empty() || !expected_ids.contains(result.match_id))
                throw ScheduleError("opening block result has an unknown assignment");
            if (!seen_ids.insert(result.match_id).second)
                throw ScheduleError("opening block result has a duplicate assignment");
            if (!score(result, total)) {
                complete = false;
                ++summary.aborted_matches;
            } else {
                ++summary.completed_matches;
            }
        }
        if (seen_ids.size() != expected_ids.size())
            throw ScheduleError("opening block result is missing assignments");
        if (complete) {
            ++summary.complete_blocks;
            summary.complete_block_scores.push_back(total /
                                                    static_cast<double>(expected_ids.size()));
        } else {
            ++summary.incomplete_blocks;
        }
    }
    if (seen_blocks.size() != expected_blocks.size())
        throw ScheduleError("opening block results are missing openings");
    return summary;
}

double mean(const std::vector<double>& values) {
    return values.empty() ? 0.0
                          : std::accumulate(values.begin(), values.end(), 0.0) /
                                static_cast<double>(values.size());
}

double quantile(const std::vector<double>& sorted, double probability) {
    const double position = probability * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<size_t>(std::floor(position));
    const auto upper = static_cast<size_t>(std::ceil(position));
    return sorted[lower] +
           (sorted[upper] - sorted[lower]) * (position - static_cast<double>(lower));
}

}  // namespace

void SooRatedMatch::validate() const {
    validate_participants(participant_ids);
    validate_permutation(seat_assignment, "seat_assignment");
    validate_permutation(turn_order, "turn_order");
    if (opening_id.empty()) throw ScheduleError("opening_id must be non-empty");
    if (match_id != ::diamond_orchestration::match_id("soo", participant_ids, seat_assignment, turn_order, opening_id))
        throw ScheduleError("Soo match_id does not match match dimensions");
}

void MinRatedMatch::validate() const {
    validate_participants(participant_ids);
    validate_permutation(seat_assignment, "seat_assignment");
    validate_permutation(turn_order, "turn_order");
    if (opening_id.empty()) throw ScheduleError("opening_id must be non-empty");
    if (match_id != ::diamond_orchestration::match_id("min", participant_ids, seat_assignment, turn_order, opening_id))
        throw ScheduleError("Min match_id does not match match dimensions");
}

std::vector<SooRatedMatch> schedule_soo_pair(const std::vector<std::string>& openings,
                                              const std::array<std::string, 2>& participants) {
    validate_openings<2>(openings);
    validate_participants(participants);
    std::vector<SooRatedMatch> result;
    std::array<int, 2> seats{1, 2};
    for (const auto& opening : openings) {
        auto assignment = seats;
        do {
            auto order = seats;
            do {
                result.push_back({participants, assignment, order, opening,
                                  match_id("soo", participants, assignment, order, opening)});
            } while (std::next_permutation(order.begin(), order.end()));
        } while (std::next_permutation(assignment.begin(), assignment.end()));
    }
    return result;
}

std::vector<MinRatedMatch> schedule_min_triple(const std::vector<std::string>& openings,
                                                const std::array<std::string, 3>& participants) {
    validate_openings<3>(openings);
    validate_participants(participants);
    std::vector<MinRatedMatch> result;
    std::array<int, 3> seats{1, 2, 3};
    for (const auto& opening : openings) {
        auto assignment = seats;
        do {
            auto order = seats;
            do {
                result.push_back({participants, assignment, order, opening,
                                  match_id("min", participants, assignment, order, opening)});
            } while (std::next_permutation(order.begin(), order.end()));
        } while (std::next_permutation(assignment.begin(), assignment.end()));
    }
    return result;
}

void validate_soo_rated_batch(const std::vector<SooRatedMatch>& matches,
                               const std::vector<std::string>& openings) {
    validate_batch<2>(matches, openings, schedule_soo_pair);
}

void validate_min_rated_batch(const std::vector<MinRatedMatch>& matches,
                               const std::vector<std::string>& openings) {
    validate_batch<3>(matches, openings, schedule_min_triple);
}

void MaterializedOpening::validate() const {
    if (opening_id.empty() || serialized_state.empty() || state_sha256.empty())
        throw ScheduleError("materialized opening fields must be non-empty");
    if (depth < 0)
        throw ScheduleError("materialized opening depth must be non-negative");
    if (state_sha256 != digest(diamond_support::parse_json(serialized_state)))
        throw ScheduleError("materialized opening state digest does not match serialized state");
}

void MaterializedOpeningSuite::validate() const {
    config.validate();
    if (*this != materialize_opening_suite(config))
        throw ScheduleError("materialized opening suite is incompatible with its configuration");
}

MaterializedOpeningSuite materialize_opening_suite(const OpeningSuiteConfig& config) {
    config.validate();
    MaterializedOpeningSuite result;
    result.config = config;
    result.openings.reserve(static_cast<size_t>(config.count));
    for (int64_t index = 0; index < config.count; ++index) {
        const uint64_t seed = splitmix64(config.seed ^ static_cast<uint64_t>(index));
        const int64_t depth =
            config.max_depth == 0
                ? 0
                : static_cast<int64_t>(seed % static_cast<uint64_t>(config.max_depth)) + 1;
        const Json state = opening_state_payload(config, index, seed, depth);
        const std::string serialized_state = diamond_support::canonical_json(state);
        const std::string state_sha256 = digest(state);
        const std::string opening_id =
            digest(Json{Json::Object{{"opening_index", Json{index}},
                                     {"state_sha256", Json{state_sha256}},
                                     {"suite", config.to_json()}}});
        result.openings.push_back({opening_id, serialized_state, state_sha256, seed, depth});
    }
    result.suite_sha256 = digest(suite_payload(result));
    return result;
}

std::vector<std::string> opening_ids(const MaterializedOpeningSuite& suite) {
    suite.validate();
    std::vector<std::string> result;
    result.reserve(suite.openings.size());
    for (const auto& opening : suite.openings)
        result.push_back(opening.opening_id);
    return result;
}

std::vector<SooRatedMatch> schedule_soo_pair(const MaterializedOpeningSuite& suite,
                                             const std::array<std::string, 2>& participants) {
    return schedule_soo_pair(opening_ids(suite), participants);
}

std::vector<MinRatedMatch> schedule_min_triple(const MaterializedOpeningSuite& suite,
                                               const std::array<std::string, 3>& participants) {
    return schedule_min_triple(opening_ids(suite), participants);
}

void SooOpeningBlock::validate() const {
    opening.validate();
    validate_soo_rated_batch(matches, {opening.opening_id});
}

void MinOpeningBlock::validate() const {
    opening.validate();
    validate_min_rated_batch(matches, {opening.opening_id});
}

std::vector<SooOpeningBlock>
schedule_soo_opening_blocks(const MaterializedOpeningSuite& suite,
                            const std::array<std::string, 2>& participants) {
    suite.validate();
    std::vector<SooOpeningBlock> result;
    result.reserve(suite.openings.size());
    for (const auto& opening : suite.openings)
        result.push_back({opening, schedule_soo_pair(std::vector<std::string>{opening.opening_id},
                                                     participants)});
    return result;
}

std::vector<MinOpeningBlock>
schedule_min_opening_blocks(const MaterializedOpeningSuite& suite,
                            const std::array<std::string, 3>& participants) {
    suite.validate();
    std::vector<MinOpeningBlock> result;
    result.reserve(suite.openings.size());
    for (const auto& opening : suite.openings)
        result.push_back({opening, schedule_min_triple(std::vector<std::string>{opening.opening_id},
                                                       participants)});
    return result;
}

OpeningBlockSummary
summarize_soo_opening_blocks(const std::vector<SooOpeningBlock>& blocks,
                             const std::vector<SooOpeningBlockResult>& results) {
    return summarize_blocks<SooOpeningBlock, SooOpeningBlockResult, SooRatedResult, SooRatedMatch>(
        blocks, results, [](const SooRatedResult& result, double& total) {
            if (!result.candidate_won)
                return false;
            total += *result.candidate_won ? 1.0 : 0.0;
            return true;
        });
}

OpeningBlockSummary
summarize_min_opening_blocks(const std::vector<MinOpeningBlock>& blocks,
                             const std::vector<MinOpeningBlockResult>& results) {
    return summarize_blocks<MinOpeningBlock, MinOpeningBlockResult, MinRatedResult, MinRatedMatch>(
        blocks, results, [](const MinRatedResult& result, double& total) {
            if (!result.candidate_placement)
                return false;
            switch (*result.candidate_placement) {
            case 0:
                total += 1.0;
                return true;
            case 1:
                return true;
            case 2:
                total -= 1.0;
                return true;
            default:
                throw ScheduleError("Min candidate placement must be 0, 1, or 2");
            }
        });
}

OpeningBlockBootstrap bootstrap_opening_blocks(const OpeningBlockSummary& summary,
                                               const PromotionStatisticsConfig& config,
                                               double promotion_threshold) {
    config.validate();
    if (!std::isfinite(promotion_threshold))
        throw ScheduleError("promotion threshold must be finite");
    if (summary.complete_blocks != static_cast<int64_t>(summary.complete_block_scores.size()) ||
        summary.incomplete_blocks < 0 || summary.completed_matches < 0 ||
        summary.aborted_matches < 0)
        throw ScheduleError("opening block summary is inconsistent");

    OpeningBlockBootstrap result{.complete_blocks = summary.complete_blocks,
                                 .incomplete_blocks = summary.incomplete_blocks};
    if (summary.complete_block_scores.empty())
        return result;
    result.point_estimate = mean(summary.complete_block_scores);

    std::mt19937_64 rng(config.seed);
    std::vector<double> replicates;
    replicates.reserve(static_cast<size_t>(config.bootstrap_replicates));
    for (int64_t replicate = 0; replicate < config.bootstrap_replicates; ++replicate) {
        double total = 0.0;
        for (size_t sample = 0; sample < summary.complete_block_scores.size(); ++sample)
            total += summary.complete_block_scores[rng() % summary.complete_block_scores.size()];
        replicates.push_back(total / static_cast<double>(summary.complete_block_scores.size()));
    }
    std::sort(replicates.begin(), replicates.end());
    const double tail = (1.0 - config.confidence_level) / 2.0;
    result.confidence_lower = quantile(replicates, tail);
    result.confidence_upper = quantile(replicates, 1.0 - tail);
    result.promoted = result.confidence_lower >= promotion_threshold;
    return result;
}

}  // namespace diamond_orchestration
