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

}  // namespace soo
