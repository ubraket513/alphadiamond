#include "diamond_orchestration/rating.hpp"

#include <iostream>
#include <stdexcept>
#include <variant>

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

        const auto identity = make_participant_identity(
            diamond_support::JsonValue{diamond_support::JsonValue::Object{
                {"checkpoint_sha256", diamond_support::JsonValue{"abc"}},
                {"model", diamond_support::JsonValue{"candidate"}}}}, "Canonical candidate");
        require(identity.participant_id.starts_with("sha256:"), "full participant identity is canonical");

        RatingRegistry first{"sha256:soo-v2"}, second{"sha256:soo-v2"};
        for (auto* registry : {&first, &second}) {
            registry->add_participant("a", "A"); registry->add_participant("b", "B");
        }
        const auto one = make_soo_rating_event(91, "sha256:soo-v2", {"a", "b"}, {1, 2}, {1, 2},
                                               "opening-1", true, "a", "b", "game-1");
        const auto one_reassigned = make_soo_rating_event(3, "sha256:soo-v2", {"a", "b"}, {1, 2}, {1, 2},
                                                          "opening-1", true, "a", "b", "game-1");
        const auto two = make_soo_rating_event(44, "sha256:soo-v2", {"a", "b"}, {1, 2}, {1, 2},
                                               "opening-2", true, "b", "a", "game-2");
        require(one.event_id == one_reassigned.event_id, "v2 event ID ignores assigned sequence");
        require(one.event_id != two.event_id, "distinct stable game IDs do not collapse");
        require(first.record_event(two) && first.record_event(one), "v2 events accepted in reverse order");
        require(!first.record_event(one_reassigned), "same v2 event is idempotent across sequence assignment");
        require(second.record_event(one) && second.record_event(two), "v2 events accepted in forward order");
        require(std::get<SooRatingEvent>(first.events()[0]).game_id == "game-1" &&
                std::get<SooRatingEvent>(first.events()[1]).game_id == "game-2",
                "v2 replay order follows the stable sortable game identity");
        require(first.soo_leaderboard()[0].rating == second.soo_leaderboard()[0].rating,
                "replay is independent of insertion order");
        require(first.events().size() == 2, "semantic union retained two games");
        for (std::size_t i = 0; i < first.events().size(); ++i)
            require(std::get<SooRatingEvent>(first.events()[i]).sequence_index == i,
                    "v2 authoritative sequences are contiguous");
        auto corrupt = two;
        corrupt.opening_id = "different-payload";
        bool conflict_rejected = false;
        try { first.record_event(corrupt); } catch (const RatingError&) { conflict_rejected = true; }
        require(conflict_rejected, "same event ID with different payload is rejected");
        const auto aborted_event = make_soo_rating_event(5, "sha256:soo-v2", {"a", "b"}, {1, 2}, {1, 2},
                                                         "opening-aborted", false, {}, {}, "game-aborted");
        require(first.record_event(aborted_event), "aborted event remains audit record");
        for (const auto& row : first.soo_leaderboard()) require(row.games == 2, "aborted event is unrated");
        first.merge(second);
        require(first.events().size() == 3, "merge is semantic union");

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
