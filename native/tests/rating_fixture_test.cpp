#include "diamond_orchestration/rating.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace { void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); } }

int main() {
    try {
        using namespace diamond_orchestration;
        const EloConfig elo;
        require(expected_elo_score(1000.0, 1000.0, elo) == 0.5, "equal Elo ratings");
        const auto updated = rate_soo_match(1200.0, 1000.0, true, elo);
        require(std::abs(updated[0] - 1207.6880983472654) < 1e-9, "Elo winner update");
        const auto aborted = rate_soo_match(1200.0, 1000.0, false, elo);
        require(aborted[0] == 1200.0 && aborted[1] == 1000.0, "aborted Elo no-op");

        const auto first = make_soo_rating_event(7, "sha256:protocol", {"soo-a", "soo-b"}, {1, 2}, {2, 1}, "opening-a", true, "soo-a", "soo-b");
        const auto same = make_soo_rating_event(7, "sha256:protocol", {"soo-a", "soo-b"}, {1, 2}, {2, 1}, "opening-a", true, "soo-a", "soo-b");
        require(first.event_id == same.event_id && first.event_id.starts_with("sha256:"), "Soo deterministic event ID");
        const auto min = make_min_rating_event(1, "sha256:min", {"min-a", "min-b", "min-c"}, {1, 2, 3}, {1, 3, 2}, "opening-b", true, {"min-a", "min-b", "min-c"});
        require(min.to_json().value.index() == 6, "Min event JSON object");

        RatingRegistry registry{"sha256:min", TrueSkillConfig{}};
        registry.add_participant("min-a", "Min A");
        registry.add_participant("min-b", "Min B");
        registry.add_participant("min-c", "Min C");
        const auto rating_for = [&registry](const std::string& id) {
            const auto rows = registry.min_leaderboard();
            for (const auto& row : rows) if (row.participant_id == id) return row.rating;
            throw std::runtime_error("missing Min rating");
        };
        const auto require_rating = [&rating_for](const std::string& id, double mu, double sigma,
                                                  double exposure, const char* message) {
            const auto rating = rating_for(id);
            constexpr double tolerance = 1e-6;
            if (std::abs(rating.mu - mu) > tolerance || std::abs(rating.sigma - sigma) > tolerance ||
                std::abs(rating.exposure - exposure) > tolerance) {
                std::ostringstream values;
                values << std::setprecision(17) << rating.mu << ", " << rating.sigma << ", "
                       << rating.exposure;
                throw std::runtime_error(std::string(message) + " (got " + values.str() + ")");
            }
        };
        require(registry.record_event(min), "first Min event is accepted");
        require_rating("min-a", 31.311358328508284, 6.698818641677955, 11.214902403474419,
                       "first-event first-place TrueSkill");
        require_rating("min-b", 25.000000000005922, 6.238469786776002, 6.284590639677916,
                       "first-event second-place TrueSkill");
        require_rating("min-c", 18.688641671485797, 6.698818641679213, -1.407814253551841,
                       "first-event third-place TrueSkill");

        const auto reverse = make_min_rating_event(2, "sha256:min", {"min-a", "min-b", "min-c"},
            {1, 2, 3}, {1, 3, 2}, "opening-c", true, {"min-c", "min-b", "min-a"});
        require(registry.record_event(reverse), "reverse-rank Min event is accepted");
        require_rating("min-a", 23.382898077083986, 5.238946639206399, 7.666058159464789,
                       "second-event third-place TrueSkill");
        require_rating("min-b", 24.999999999964484, 4.833244361187931, 10.50026691640069,
                       "second-event second-place TrueSkill");
        require_rating("min-c", 26.617101922960845, 5.238946639219459, 10.90026200530247,
                       "second-event first-place TrueSkill");
    } catch (const std::exception& error) {
        std::cerr << "rating_fixture_test: " << error.what() << '\n'; return 1;
    }
}
