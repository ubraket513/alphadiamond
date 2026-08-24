#include "diamond_orchestration/rating.hpp"

#include <iostream>
#include <stdexcept>

namespace { void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); } }

int main() {
    try {
        using namespace diamond_orchestration;
        RatingRegistry soo{"sha256:soo"};
        soo.add_participant("soo-a", "Soo A"); soo.add_participant("soo-b", "Soo B");
        const auto event = make_soo_rating_event(0, "sha256:soo", {"soo-a", "soo-b"}, {1, 2}, {1, 2}, "opening-a", true, "soo-a", "soo-b");
        require(soo.record_event(event), "first Soo event is accepted");
        require(!soo.record_event(event), "duplicate Soo event is idempotent");
        require(soo.soo_leaderboard().front().participant_id == "soo-a", "Soo leaderboard order");

        RatingRegistry min{"sha256:min", TrueSkillConfig{}};
        min.add_participant("min-a", "Min A"); min.add_participant("min-b", "Min B"); min.add_participant("min-c", "Min C");
        const auto min_event = make_min_rating_event(0, "sha256:min", {"min-a", "min-b", "min-c"}, {1, 2, 3}, {1, 2, 3}, "opening-a", true, {"min-a", "min-b", "min-c"});
        require(min.record_event(min_event), "first Min event is accepted");
        require(min.min_leaderboard().front().participant_id == "min-a", "Min leaderboard order");
        require(min.report_json().value.index() == 6, "registry report JSON object");
    } catch (const std::exception& error) {
        std::cerr << "rating_registry_test: " << error.what() << '\n'; return 1;
    }
}
