// Canonical player-relative encoding; mirrors diamond.alphazero.encoder.
#pragma once

#include <cstdint>
#include <vector>

#include "soo/state.hpp"

namespace soo {

struct Encoded {
    // Row-major [73][feature_count]; rows are indexed by canonical position id.
    std::vector<float> node_features;
    std::vector<uint8_t> canonical_player_ids;
    int feature_count = 0;
};

Encoded encode(const State& state, const Match& match);

// Turn-order seat ids rotated to start at ``state.current_player``.
void canonical_player_ids(const State& state, const Match& match, std::vector<uint8_t>& out);

int32_t to_canonical_action(int32_t physical_action, const Match& match, uint8_t current_player);
int32_t to_physical_action(int32_t canonical_action, const Match& match, uint8_t current_player);

// Legal actions in canonical space, in the same order as legal_action_ids.
void canonical_legal_action_ids(const State& state, const Match& match, std::vector<int32_t>& out);

}  // namespace soo
