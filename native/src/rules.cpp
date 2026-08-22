#include "soo/rules.hpp"

#include <array>
#include <bitset>
#include <stdexcept>

#include "soo/action.hpp"
#include "soo/board.hpp"

namespace soo {

int moves_from(const State& state, int source, uint8_t* dest_out, uint8_t* kind_out) {
    const Topology& topo = topology();
    const uint8_t owner = state.occupancy[source];
    if (owner == kEmpty) return 0;

    std::bitset<kBoardSize> is_destination;
    int count = 0;

    // Single steps first: a step is always the shortest path to its target.
    for (int direction = 0; direction < kDirections; ++direction) {
        const int8_t adjacent = topo.neighbour[source][direction];
        if (adjacent < 0 || state.occupancy[adjacent] != kEmpty) continue;
        if (is_destination.test(adjacent)) continue;
        is_destination.set(adjacent);
        dest_out[count] = static_cast<uint8_t>(adjacent);
        if (kind_out) kind_out[count] = kStep;
        ++count;
    }

    // Then chained jumps, breadth-first so the first path to a hole is shortest.
    // ``visited`` is deliberately separate from ``is_destination``: a landing
    // already reachable by a step still extends the BFS frontier but does not
    // re-enter the move ordering.  That is exactly what the Python dict does.
    std::bitset<kBoardSize> visited;
    visited.set(source);
    std::array<uint8_t, kBoardSize> queue{};
    int head = 0;
    int tail = 0;
    queue[tail++] = static_cast<uint8_t>(source);

    while (head < tail) {
        const int current = queue[head++];
        for (int direction = 0; direction < kDirections; ++direction) {
            const int8_t over = topo.neighbour[current][direction];
            if (over < 0 || over == source || state.occupancy[over] == kEmpty) continue;
            const int8_t landing = topo.neighbour[over][direction];
            if (landing < 0 || visited.test(landing) || state.occupancy[landing] != kEmpty) {
                continue;
            }
            visited.set(landing);
            queue[tail++] = static_cast<uint8_t>(landing);
            // Mirrors Python's ``if landing not in moves``.  Provably always
            // true: every jump chain lands on source + 2v for an integer cube
            // vector v, whose lattice distance is 2*dist(v) -- always even, so
            // never the distance 1 of a step destination.  Kept because it
            // mirrors the oracle, not because it can fire.
            if (is_destination.test(landing)) continue;
            is_destination.set(landing);
            dest_out[count] = static_cast<uint8_t>(landing);
            if (kind_out) kind_out[count] = kJump;
            ++count;
        }
    }
    return count;
}

void legal_action_ids(const State& state, std::vector<int32_t>& out) {
    const uint8_t player = state.current_player;
    std::array<uint8_t, kBoardSize> destinations{};
    for (int source = 0; source < kBoardSize; ++source) {
        if (state.occupancy[source] != player) continue;
        const int count = moves_from(state, source, destinations.data(), nullptr);
        for (int i = 0; i < count; ++i) {
            out.push_back(encode_action(source, destinations[i]));
        }
    }
}

bool has_finished(const State& state, const PlayerSpec& spec) {
    const Topology& topo = topology();
    const auto& target = topo.camp_positions[spec.target_camp];
    for (int i = 0; i < kCampSize; ++i) {
        if (state.occupancy[target[i]] != spec.id) return false;
    }
    return true;
}

namespace {

// diamond.game.rules.update_ranking
void update_ranking(State& state, const Match& match) {
    for (uint8_t seat = 0; seat < match.count; ++seat) {
        const PlayerSpec& spec = match.players[seat];
        if (!state.has_placed(spec.id) && has_finished(state, spec)) {
            state.place(spec.id);
        }
    }
    const bool over = state.finished_count >= match.count - 1;
    if (over && state.status != kFinished) {
        // Whoever is left never finished; they take the last place implicitly.
        for (uint8_t seat = 0; seat < match.count; ++seat) {
            if (!state.has_placed(match.players[seat].id)) {
                state.place(match.players[seat].id);
            }
        }
        state.status = kFinished;
    }
}

// diamond.game.state.next_player_id
uint8_t next_player_id(const State& state, const Match& match, uint8_t current) {
    const int start = match.seat_of(current);
    for (int offset = 1; offset <= match.count; ++offset) {
        const uint8_t candidate = match.players[(start + offset) % match.count].id;
        if (!state.has_placed(candidate)) return candidate;
    }
    return current;
}

}  // namespace

State apply_action(const State& state, const Match& match, int32_t physical_action) {
    if (state.status == kFinished) {
        throw std::runtime_error("the game is already over");
    }
    int source = 0;
    int destination = 0;
    decode_action(physical_action, source, destination);

    const uint8_t mover = state.current_player;
    if (state.occupancy[source] != mover) {
        throw std::invalid_argument("action source is not occupied by the player to act");
    }
    std::array<uint8_t, kBoardSize> destinations{};
    const int count = moves_from(state, source, destinations.data(), nullptr);
    bool legal = false;
    for (int i = 0; i < count; ++i) {
        if (destinations[i] == destination) {
            legal = true;
            break;
        }
    }
    if (!legal) throw std::invalid_argument("action is not a legal move");

    State next = state;
    next.occupancy[source] = kEmpty;
    next.occupancy[destination] = mover;
    next.turn_number = static_cast<uint16_t>(state.turn_number + 1);
    // Rank first, then hand over: a player who just got home drops out of the
    // rotation, so the next seat must be chosen against the new podium.
    update_ranking(next, match);
    next.current_player = next_player_id(next, match, mover);
    return next;
}

uint8_t search_current_player(const State& state, const Match& match) {
    if (match.count == 2 && state.status == kFinished && state.finished_count == 2) {
        // Soo's terminal leaf needs the would-be opponent perspective so scalar
        // backup flips exactly once across the edge.
        return state.finish_order[1];
    }
    return state.current_player;
}

}  // namespace soo
