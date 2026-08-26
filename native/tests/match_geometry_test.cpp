// The seat geometry is a contract, and it drifted.
//
// A seat is (player id, starting camp, target camp). The trainer, both
// benchmarks and the pipeline smoke test each built the Soo match by writing
// the triples out by hand, and the trainer's copy placed the two seats head-on:
// player 1 from camp 0 to camp 3, player 2 from camp 3 to camp 0, so each
// player's target camp was the other's starting camp. Nothing detected it. The
// rules stayed self-consistent, every golden test passed, and the Qt
// application -- which happened to hold the correct triples -- played the right
// game while training played a different one, at 77 % self-play completion
// instead of 97 %, for as long as that lasted.
//
// tests/golden/rules-v1.txt is normative and frozen, so it is the authority
// this pins against.
#include <array>
#include <string>

#include "check.hpp"
#include "golden.hpp"
#include "soo/board.hpp"
#include "soo/rules.hpp"
#include "soo/state.hpp"

namespace {

void check_match(const soo::Match& actual, const soo::Match& golden, const std::string& what) {
    CHECK_EQ(static_cast<int>(actual.count), static_cast<int>(golden.count));
    for (uint8_t seat = 0; seat < golden.count; ++seat) {
        const std::string where = what + " seat " + std::to_string(seat) + ": ";
        if (actual.players[seat].id != golden.players[seat].id ||
            actual.players[seat].camp != golden.players[seat].camp ||
            actual.players[seat].target_camp != golden.players[seat].target_camp) {
            soo_test::fail(__FILE__, __LINE__, where + "does not match the golden fixture");
        }
    }
}

// No player may target a camp another player starts in. This is the property
// whose violation shipped: it makes the target camp unfillable until the
// opponent evacuates it entirely, and the last piece out is easily sealed in.
void check_no_target_start_overlap(const soo::Match& match, const std::string& what) {
    for (uint8_t a = 0; a < match.count; ++a) {
        for (uint8_t b = 0; b < match.count; ++b) {
            if (a == b) continue;
            if (match.players[a].target_camp == match.players[b].camp) {
                soo_test::fail(__FILE__, __LINE__,
                               what + ": seat " + std::to_string(a) + " targets the camp seat " +
                                   std::to_string(b) + " starts in");
            }
        }
    }
}

// The opening must place each player's pieces in its own camp, and a target
// camp may only be occupied at move zero through a *shared corner*.
//
// Adjacent camps deliberately share one corner hole -- camp 0 and camp 5 share
// hole 12, camp 2 and camp 3 share hole 60, and so on for all six corners -- so
// one enemy piece sitting in a target camp at move zero is by design, and is
// why even correct geometry aborts a small fraction of games. What is not by
// design is a target camp that *is* another player's camp: that occupies ten
// cells rather than one, and the game cannot end until they are all vacated.
// The distinction is exactly what this checks.
void check_opening(const soo::Match& match, const std::string& what) {
    std::array<int, soo::kBoardSize> camp_membership{};
    for (int camp = 0; camp < soo::kCamps; ++camp)
        for (const uint8_t position : soo::topology().camp_positions[camp])
            ++camp_membership[position];

    soo::State state;
    for (uint8_t seat = 0; seat < match.count; ++seat)
        for (const uint8_t position : soo::topology().camp_positions[match.players[seat].camp])
            state.occupancy[position] = match.players[seat].id;

    for (uint8_t seat = 0; seat < match.count; ++seat) {
        const auto& player = match.players[seat];
        for (const uint8_t position : soo::topology().camp_positions[player.target_camp]) {
            if (state.occupancy[position] == 0) continue;
            if (state.occupancy[position] == player.id) {
                soo_test::fail(__FILE__, __LINE__,
                               what + ": seat " + std::to_string(seat) +
                                   " starts inside its own target camp");
            }
            if (camp_membership[position] < 2) {
                soo_test::fail(__FILE__, __LINE__,
                               what + ": target camp of seat " + std::to_string(seat) +
                                   " is occupied at move zero at a hole that is not a shared "
                                   "corner");
            }
        }
        // Nobody may already have finished, and nobody's camp may be fillable
        // only by evacuating an opponent's whole camp.
        CHECK_EQ(soo::has_finished(state, player), false);
    }
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 2, "usage: match_geometry_test <golden-dir>");
    const std::string golden_dir = argv[1];
    REQUIRE(soo::load_topology_from_dir(golden_dir + "/topology"),
            "could not load the golden topology tables");

    soo_test::Golden golden;
    std::string error;
    REQUIRE(soo_test::load_golden(golden_dir + "/rules-v1.txt", golden, error), error.c_str());

    check_match(soo::standard_soo_match(), golden.match(2), "Soo");
    check_match(soo::standard_min_match(), golden.match(3), "Min");

    check_no_target_start_overlap(soo::standard_soo_match(), "Soo");
    check_no_target_start_overlap(soo::standard_min_match(), "Min");

    check_opening(soo::standard_soo_match(), "Soo");
    check_opening(soo::standard_min_match(), "Min");

    // Each player targets the camp opposite its own; the board has six camps.
    for (const auto& [what, match] :
         {std::pair{std::string("Soo"), soo::standard_soo_match()},
          std::pair{std::string("Min"), soo::standard_min_match()}}) {
        for (uint8_t seat = 0; seat < match.count; ++seat) {
            const auto& player = match.players[seat];
            if ((player.camp + 3) % 6 != player.target_camp) {
                soo_test::fail(__FILE__, __LINE__,
                               what + " seat " + std::to_string(seat) +
                                   " does not target the camp opposite its own");
            }
        }
    }

    return soo_test::report("match_geometry_test");
}
