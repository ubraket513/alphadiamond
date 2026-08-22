#include "soo/prior.hpp"

#include <cmath>
#include <stdexcept>

#include "soo/action.hpp"
#include "soo/encoder.hpp"

namespace soo {

const PieceSet& target_camp_set() {
    // Canonicalisation rotates the acting player's home camp to z+, so their
    // target camp is always canonical z-.  Rebuilt per call site is cheap, but
    // the set is fixed once topology is configured.
    static PieceSet cached;
    static bool built = false;
    if (!built) {
        const Topology& topo = topology();
        for (int i = 0; i < kCampSize; ++i) cached.set(topo.camp_positions[kCampZNeg][i]);
        built = true;
    }
    return cached;
}

double vacancy_potential(const PieceSet& occupied, const PieceSet& target) {
    const PieceSet vacancies = target & ~occupied;
    if (vacancies.none()) return 0.0;
    const Topology& topo = topology();

    // Vacancy list first: the inner min runs once per outside piece.
    uint8_t vacancy_ids[kCampSize];
    int vacancy_count = 0;
    for (int position = 0; position < kBoardSize; ++position) {
        if (vacancies.test(position)) vacancy_ids[vacancy_count++] = static_cast<uint8_t>(position);
    }

    const PieceSet outside = occupied & ~target;
    long total = 0;
    for (int piece = 0; piece < kBoardSize; ++piece) {
        if (!outside.test(piece)) continue;
        const auto& row = topo.pairwise[piece];
        int best = row[vacancy_ids[0]];
        for (int i = 1; i < vacancy_count; ++i) {
            const int candidate = row[vacancy_ids[i]];
            if (candidate < best) best = candidate;
        }
        total += best;
    }
    return static_cast<double>(total);
}

PieceSet canonical_self_occupancy(const State& state, const Match& match) {
    // Channel 0 of the encoded features is the acting player's occupancy; the
    // mapped physical positions are the same set without building features.
    const auto& mapping = topology().physical_to_canonical[match.by_id(state.current_player).camp];
    PieceSet occupied;
    for (int physical = 0; physical < kBoardSize; ++physical) {
        if (state.occupancy[physical] == state.current_player) occupied.set(mapping[physical]);
    }
    return occupied;
}

void vacancy_prior(const std::vector<int32_t>& canonical_actions,
                   const PieceSet& occupied,
                   std::vector<double>& out) {
    if (canonical_actions.empty()) {
        throw std::invalid_argument("evaluation requires at least one legal action");
    }
    const PieceSet& target = target_camp_set();
    const double before = vacancy_potential(occupied, target);

    out.assign(canonical_actions.size(), 0.0);
    double highest = -1e308;
    for (size_t i = 0; i < canonical_actions.size(); ++i) {
        int source = 0;
        int destination = 0;
        decode_action(canonical_actions[i], source, destination);
        PieceSet moved = occupied;
        moved.reset(source);
        moved.set(destination);
        const double score = before - vacancy_potential(moved, target);
        out[i] = score;
        if (score > highest) highest = score;
    }

    // Shift by the max before exponentiating: identical probabilities, no
    // overflow, and the result is independent of action ordering.
    double total = 0.0;
    for (double& value : out) {
        value = std::exp((value - highest) / kPriorTemperature);
        total += value;
    }
    for (double& value : out) value /= total;
}

}  // namespace soo
