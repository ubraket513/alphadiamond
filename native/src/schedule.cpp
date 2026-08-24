#include "diamond_orchestration/schedule.hpp"

#include <algorithm>
#include <set>

#include "diamond_support/json.hpp"

namespace diamond_orchestration {
namespace {

using Json = diamond_support::JsonValue;

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

}  // namespace diamond_orchestration
