#include "soo/encoder.hpp"

#include <stdexcept>

#include "soo/action.hpp"
#include "soo/board.hpp"
#include "soo/rules.hpp"

namespace soo {

void canonical_player_ids(const State& state, const Match& match, std::vector<uint8_t>& out) {
    const int start = match.seat_of(state.current_player);
    out.clear();
    for (int offset = 0; offset < match.count; ++offset) {
        out.push_back(match.players[(start + offset) % match.count].id);
    }
}

Encoded encode(const State& state, const Match& match) {
    const Topology& topo = topology();
    Encoded encoded;
    canonical_player_ids(state, match, encoded.canonical_player_ids);

    const int players = match.count;
    const int features = 2 * players;  // occupancy channels, then finished flags
    encoded.feature_count = features;
    encoded.node_features.assign(static_cast<size_t>(kBoardSize) * features, 0.0f);

    // channel_by_player, keyed by seat id (ids are 1-based and small).
    std::array<int, 256> channel_by_player{};
    channel_by_player.fill(-1);
    for (int channel = 0; channel < players; ++channel) {
        channel_by_player[encoded.canonical_player_ids[channel]] = channel;
    }

    const uint8_t home_camp = match.by_id(state.current_player).camp;
    const auto& mapping = topo.physical_to_canonical[home_camp];

    // Finished flags are per canonical channel, identical on every row.
    std::array<float, kMaxPlayers> finished{};
    for (int channel = 0; channel < players; ++channel) {
        finished[channel] = state.has_placed(encoded.canonical_player_ids[channel]) ? 1.0f : 0.0f;
    }

    for (int physical = 0; physical < kBoardSize; ++physical) {
        const int canonical = mapping[physical];
        float* row = encoded.node_features.data() + static_cast<size_t>(canonical) * features;
        const uint8_t occupant = state.occupancy[physical];
        if (occupant != kEmpty) {
            const int channel = channel_by_player[occupant];
            if (channel < 0) throw std::invalid_argument("occupancy contains unknown player id");
            row[channel] = 1.0f;
        }
        for (int channel = 0; channel < players; ++channel) {
            row[players + channel] = finished[channel];
        }
    }
    return encoded;
}

int32_t to_canonical_action(int32_t physical_action, const Match& match, uint8_t current_player) {
    int source = 0;
    int destination = 0;
    decode_action(physical_action, source, destination);
    const auto& mapping = topology().physical_to_canonical[match.by_id(current_player).camp];
    return encode_action(mapping[source], mapping[destination]);
}

int32_t to_physical_action(int32_t canonical_action, const Match& match, uint8_t current_player) {
    int source = 0;
    int destination = 0;
    decode_action(canonical_action, source, destination);
    const auto& mapping = topology().canonical_to_physical[match.by_id(current_player).camp];
    return encode_action(mapping[source], mapping[destination]);
}

void canonical_legal_action_ids(const State& state, const Match& match,
                                std::vector<int32_t>& out) {
    std::vector<int32_t> physical;
    physical.reserve(64);
    legal_action_ids(state, physical);
    const auto& mapping = topology().physical_to_canonical[match.by_id(state.current_player).camp];
    out.reserve(out.size() + physical.size());
    for (const int32_t action : physical) {
        out.push_back(encode_action(mapping[action / kBoardSize], mapping[action % kBoardSize]));
    }
}

}  // namespace soo
