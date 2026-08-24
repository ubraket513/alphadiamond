#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

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

std::vector<SooRatedMatch> schedule_soo_pair(
    const std::vector<std::string>& opening_ids,
    const std::array<std::string, 2>& participant_ids);
std::vector<MinRatedMatch> schedule_min_triple(
    const std::vector<std::string>& opening_ids,
    const std::array<std::string, 3>& participant_ids);

void validate_soo_rated_batch(const std::vector<SooRatedMatch>& matches,
                               const std::vector<std::string>& opening_ids);
void validate_min_rated_batch(const std::vector<MinRatedMatch>& matches,
                               const std::vector<std::string>& opening_ids);

}  // namespace diamond_orchestration
