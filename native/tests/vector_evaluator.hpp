// The Gate B reference evaluator, widened to a three-seat value vector.
//
// Specified in integer arithmetic -- FNV-1a, then an exact 53-bit mantissa
// division -- so this and its Python twin in tools/build_golden.py agree bit
// for bit rather than to a tolerance. Without that the golden file would be
// comparing two different functions and calling any difference a port bug.
//
// The three components are deliberately *different* from each other. A value
// vector whose components agree hides the two mistakes that matter in a 3P
// search: components assigned to the wrong seats, and a node maximising
// somebody else's component. Both leave a symmetric vector looking correct.
#pragma once

#include <cstdint>
#include <vector>

#include "soo/encoder.hpp"
#include "soo/evaluator.hpp"
#include "soo/mcts3p.hpp"

namespace soo_test {

constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr double kMantissaScale = 9007199254740992.0;  // 2^53

inline uint64_t mix(uint64_t hash, uint8_t byte) {
    return (hash ^ byte) * kFnvPrime;
}

inline uint64_t mix_u32(uint64_t hash, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        hash = mix(hash, static_cast<uint8_t>((value >> shift) & 0xFF));
    }
    return hash;
}

inline uint64_t mix_u64(uint64_t hash, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash = mix(hash, static_cast<uint8_t>((value >> shift) & 0xFF));
    }
    return hash;
}

// The top 53 bits mapped into [0, 1) without rounding: below 2^53 an integer
// converts to double exactly, so both languages land on the same value.
inline double unit(uint64_t hash) {
    return static_cast<double>(hash >> 11) / kMantissaScale;
}

// Answer one 3P request as a pure function of it.
inline soo::EvalOutcome3P evaluate_vector(const soo::Encoded& encoded,
                                          const std::vector<int32_t>& actions) {
    const uint64_t digest = soo::request_hash(encoded, actions);

    soo::EvalOutcome3P outcome;
    outcome.priors.reserve(actions.size());
    double total = 0.0;
    for (const int32_t action : actions) {
        const uint64_t action_hash =
            mix_u32(mix_u64(kFnvOffset, digest), static_cast<uint32_t>(action));
        // Shifted into [0.5, 1.5): strictly positive, and spread widely enough
        // to order the PUCT keys.
        const double weight = unit(action_hash) + 0.5;
        outcome.priors.push_back(weight);
        total += weight;
    }
    for (double& prior : outcome.priors) prior /= total;

    // One component per seat, in the request's canonical order, and pairwise
    // distinct by construction.
    for (int seat = 0; seat < 3; ++seat) {
        const uint64_t seat_hash =
            mix_u32(mix_u64(kFnvOffset, digest), static_cast<uint32_t>(0x5EA70000 + seat));
        outcome.value[static_cast<size_t>(seat)] = unit(seat_hash) * 2.0 - 1.0;
    }
    return outcome;
}

}  // namespace soo_test
