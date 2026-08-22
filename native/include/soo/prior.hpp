// canonical-target-vacancy-distance-v2; mirrors
// diamond.alphazero.bootstrap.heuristic.CanonicalTargetVacancyDistancePrior.
#pragma once

#include <bitset>
#include <cstdint>
#include <vector>

#include "soo/board.hpp"
#include "soo/state.hpp"

namespace soo {

using PieceSet = std::bitset<kBoardSize>;

inline constexpr double kPriorTemperature = 1.0;

// Phi: sum over the acting player's pieces outside the target camp of the
// distance to the nearest target hole the player does not already hold.
double vacancy_potential(const PieceSet& occupied, const PieceSet& target);

// The acting player's pieces, in canonical position ids.
PieceSet canonical_self_occupancy(const State& state, const Match& match);

// Softmax over Phi(before) - Phi(after), aligned with ``canonical_actions``.
void vacancy_prior(const std::vector<int32_t>& canonical_actions,
                   const PieceSet& occupied,
                   std::vector<double>& out);

// The canonical target camp (z-) as a bitset.
const PieceSet& target_camp_set();

}  // namespace soo
