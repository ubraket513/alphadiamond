// Move generation, application, ranking.  The Python authority is
// diamond.game.rules / diamond.game.session.GameSession.commit.
#pragma once

#include <cstdint>
#include <vector>

#include "soo/state.hpp"

namespace soo {

inline constexpr uint8_t kStep = 0;
inline constexpr uint8_t kJump = 1;

// Destinations reachable from ``source``, in Python's ``moves_from`` dict
// order: single steps in direction order, then new jump landings in BFS
// discovery order.  ``kind_out`` may be null.  Returns the destination count.
int moves_from(const State& state, int source, uint8_t* dest_out, uint8_t* kind_out);

// Physical action ids for the player to act, in Python's ``legal_moves`` order
// (ascending source, then ``moves_from`` order).  Appends to ``out``.
void legal_action_ids(const State& state, std::vector<int32_t>& out);

// True when every hole of ``spec``'s target camp holds ``spec.id``.
bool has_finished(const State& state, const PlayerSpec& spec);

// GameSession.commit: validate, apply, rank, hand over.  Throws on an illegal
// action or a finished match.
State apply_action(const State& state, const Match& match, int32_t physical_action);

// Search-adapter view of whose perspective a state belongs to
// (DiamondSearchAdapter.current_player_id).
uint8_t search_current_player(const State& state, const Match& match);

}  // namespace soo
