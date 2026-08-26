// POD game state; mirrors diamond.game.state.GameState.
#pragma once

#include <array>
#include <cstdint>

#include "soo/board.hpp"

namespace soo {

inline constexpr uint8_t kEmpty = 0;
inline constexpr uint8_t kInProgress = 0;
inline constexpr uint8_t kFinished = 1;

struct State {
    std::array<uint8_t, kBoardSize> occupancy{};
    uint8_t current_player = 0;
    uint8_t status = kInProgress;
    uint16_t turn_number = 1;
    std::array<uint8_t, kMaxPlayers> finish_order{};
    uint8_t finished_count = 0;

    bool has_placed(uint8_t player) const {
        for (uint8_t i = 0; i < finished_count; ++i) {
            if (finish_order[i] == player) return true;
        }
        return false;
    }

    void place(uint8_t player) {
        if (has_placed(player)) return;
        finish_order[finished_count++] = player;
    }

    bool operator==(const State& other) const {
        if (occupancy != other.occupancy) return false;
        if (current_player != other.current_player) return false;
        if (status != other.status) return false;
        if (turn_number != other.turn_number) return false;
        if (finished_count != other.finished_count) return false;
        for (uint8_t i = 0; i < finished_count; ++i) {
            if (finish_order[i] != other.finish_order[i]) return false;
        }
        return true;
    }
};

// Identity for *repetition*, which is not the same thing as equality.
//
// ``operator==`` includes ``turn_number``, so by that test a position can never
// repeat -- the ply counter always differs.  Repetition is about the dynamics:
// two positions are the same if every legal continuation from them is the same,
// which means occupancy, the side to move, and who has already finished.
// ``turn_number`` is bookkeeping and is deliberately excluded.
//
// Deliberately NOT the encoded features either.  The encoder canonicalises --
// it rotates the acting player's camp to a fixed orientation and reorders the
// player channels -- so symmetric images collide and the same position with the
// other side to move does not.  That is a reasonable notion for asking "did the
// network see this input before" and the wrong one for changing search
// behaviour.
inline uint64_t dynamics_key(const State& state) {
    // FNV-1a: no dependencies, and collisions here would only mis-trigger a
    // search-budget change rather than corrupt a game.
    uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](uint8_t byte) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    };
    for (const uint8_t cell : state.occupancy) mix(cell);
    mix(state.current_player);
    mix(state.status);
    mix(state.finished_count);
    for (uint8_t i = 0; i < state.finished_count; ++i) mix(state.finish_order[i]);
    return hash;
}

// One seat: mirrors diamond.game.state.PlayerSpec, reduced to what rules need.
struct PlayerSpec {
    uint8_t id = 0;
    uint8_t camp = 0;
    uint8_t target_camp = 0;
};

// The seats of a match, in turn order.
struct Match {
    std::array<PlayerSpec, kMaxPlayers> players{};
    uint8_t count = 0;

    const PlayerSpec& by_id(uint8_t player_id) const;
    int seat_of(uint8_t player_id) const;
};

// The seat geometry, in one place.
//
// A seat is (player id, starting camp, target camp) and the three are not
// independent: a player targets the camp *opposite* its own, and no player's
// target may be another player's start. Get that wrong and the game still runs
// -- the rules stay self-consistent -- but every game begins with the opponent
// already occupying the camp you must fill to win, which is unwinnable until
// they vacate it. That shipped: the trainer and both benchmarks placed the two
// Soo seats head-on while the Qt application used the correct geometry, and
// self-play completion sat at 77 % instead of 97 % for as long as it lasted.
//
// These are the values in tests/golden/rules-v1.txt, which is normative and
// frozen. Construct a standard match through these functions and never by
// writing the triples at a call site.
Match standard_soo_match();   // 2 players: {1, camp 2, target 5}, {2, camp 0, target 3}
Match standard_min_match();   // 3 players: {1,2,5}, {2,1,4}, {3,0,3}

}  // namespace soo
