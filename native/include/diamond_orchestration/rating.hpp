#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "diamond_support/json.hpp"

namespace diamond_orchestration {

class RatingError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct EloConfig final {
    double initial_rating = 1000.0;
    double k_factor = 32.0;
    double logistic_scale = 400.0;
    std::string rating_system_version = "soo-elo-v1";
    void validate() const;
};

struct TrueSkillConfig final {
    double mu = 25.0;
    double sigma = 25.0 / 3.0;
    double beta = 25.0 / 6.0;
    double tau = 0.0;
    double draw_probability = 0.0;
    std::string rating_system_version = "min-trueskill-v1";
    void validate() const;
};

// A canonical artifact identity. Display text is intentionally not hashed.
struct ParticipantIdentity final {
    std::string participant_id;
    std::string display_name;
    diamond_support::JsonValue full_identity;

    void validate() const;
    bool operator==(const ParticipantIdentity&) const;
};

std::string canonical_participant_id(const diamond_support::JsonValue& full_identity);
ParticipantIdentity make_participant_identity(diamond_support::JsonValue full_identity,
                                              std::string display_name);

struct SooRatingEvent final {
    uint64_t sequence_index = 0;
    std::string protocol_id;
    std::array<std::string, 2> participant_ids;
    std::array<int, 2> seat_assignment;
    std::array<int, 2> turn_order;
    std::string opening_id;
    bool completed = false;
    std::string winner_id;
    std::string loser_id;
    std::string event_id;
    // v2 only: stable, lexicographically sortable game identity; event_id then
    // excludes sequence_index so independently assigned local indexes converge.
    std::string game_id;
    // v2 distributed events may carry self-authenticating participant
    // registrations so a materializer does not depend on one device's catalog.
    std::vector<ParticipantIdentity> participant_identities;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    bool operator==(const SooRatingEvent&) const = default;
};

struct MinRatingEvent final {
    uint64_t sequence_index = 0;
    std::string protocol_id;
    std::array<std::string, 3> participant_ids;
    std::array<int, 3> seat_assignment;
    std::array<int, 3> turn_order;
    std::string opening_id;
    bool completed = false;
    std::array<std::string, 3> final_ranking;
    std::string event_id;
    std::string game_id;
    std::vector<ParticipantIdentity> participant_identities;

    void validate() const;
    diamond_support::JsonValue to_json() const;
    bool operator==(const MinRatingEvent&) const = default;
};

SooRatingEvent make_soo_rating_event(uint64_t sequence_index, std::string protocol_id,
                                     std::array<std::string, 2> participant_ids,
                                     std::array<int, 2> seat_assignment,
                                     std::array<int, 2> turn_order, std::string opening_id,
                                     bool completed, std::string winner_id = {},
                                     std::string loser_id = {}, std::string game_id = {},
                                     std::vector<ParticipantIdentity> participant_identities = {});
MinRatingEvent make_min_rating_event(uint64_t sequence_index, std::string protocol_id,
                                     std::array<std::string, 3> participant_ids,
                                     std::array<int, 3> seat_assignment,
                                     std::array<int, 3> turn_order, std::string opening_id,
                                     bool completed, std::array<std::string, 3> final_ranking = {},
                                     std::string game_id = {},
                                     std::vector<ParticipantIdentity> participant_identities = {});

double expected_elo_score(double rating_a, double rating_b, const EloConfig& config);
std::array<double, 2> rate_soo_match(double winner_rating, double loser_rating,
                                     bool completed, const EloConfig& config);

struct MinRating final {
    double mu = 25.0;
    double sigma = 25.0 / 3.0;
    double exposure = 0.0;
    uint64_t rated_games = 0;
};

struct SooLeaderboardEntry final {
    std::string participant_id;
    std::string display_name;
    double rating = 0.0;
    uint64_t games = 0;
};

struct MinLeaderboardEntry final {
    std::string participant_id;
    std::string display_name;
    MinRating rating;
};

class RatingRegistry final {
  public:
    explicit RatingRegistry(std::string protocol_id, EloConfig config = {});
    explicit RatingRegistry(std::string protocol_id, TrueSkillConfig config);

    void add_participant(std::string participant_id, std::string display_name);
    void add_participant(ParticipantIdentity identity);
    bool record_event(const SooRatingEvent& event);
    bool record_event(const MinRatingEvent& event);
    void merge(const RatingRegistry& other);
    void rebuild();

    const std::vector<std::variant<SooRatingEvent, MinRatingEvent>>& events() const noexcept { return events_; }
    const EloConfig& elo_config() const noexcept {
        return elo_;
    }
    const TrueSkillConfig& trueskill_config() const noexcept {
        return trueskill_;
    }
    std::vector<SooLeaderboardEntry> soo_leaderboard() const;
    std::vector<MinLeaderboardEntry> min_leaderboard() const;
    diamond_support::JsonValue report_json() const;

  private:
    enum class Family { soo, min };
    Family family_;
    std::string protocol_id_;
    EloConfig elo_;
    TrueSkillConfig trueskill_;
    std::map<std::string, std::string> participants_;
    std::map<std::string, diamond_support::JsonValue> identities_;
    std::vector<std::variant<SooRatingEvent, MinRatingEvent>> events_;
    std::map<std::string, double> soo_ratings_;
    std::map<std::string, MinRating> min_ratings_;

    void normalize_events();
};

}  // namespace diamond_orchestration
