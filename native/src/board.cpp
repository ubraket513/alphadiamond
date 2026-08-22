#include "soo/board.hpp"
#include "soo/state.hpp"

#include <stdexcept>

namespace soo {

Topology& mutable_topology() {
    static Topology instance;
    return instance;
}

const PlayerSpec& Match::by_id(uint8_t player_id) const {
    for (uint8_t i = 0; i < count; ++i) {
        if (players[i].id == player_id) return players[i];
    }
    throw std::invalid_argument("unknown player id");
}

int Match::seat_of(uint8_t player_id) const {
    for (uint8_t i = 0; i < count; ++i) {
        if (players[i].id == player_id) return i;
    }
    throw std::invalid_argument("unknown player id");
}

}  // namespace soo
