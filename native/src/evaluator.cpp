#include "soo/evaluator.hpp"

#include <cstddef>

namespace soo {
namespace {

constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

// 2^53: every value below is exactly representable as a double, so the
// integer -> double step introduces no rounding and both languages agree.
constexpr double kMantissaScale = 9007199254740992.0;

inline void mix(uint64_t& hash, uint8_t byte) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= kFnvPrime;
}

inline void mix_u32(uint64_t& hash, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) mix(hash, static_cast<uint8_t>(value >> shift));
}

inline void mix_u64(uint64_t& hash, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) mix(hash, static_cast<uint8_t>(value >> shift));
}

// Map the top 53 bits of a hash into [0, 1) without rounding.
inline double unit(uint64_t hash) {
    return static_cast<double>(hash >> 11) / kMantissaScale;
}

}  // namespace

uint64_t request_hash(const Encoded& encoded, const std::vector<int32_t>& legal_actions) {
    uint64_t hash = kFnvOffset;
    // Features are exactly 0.0 or 1.0 (Gate A proves it), so one byte each.
    for (const float feature : encoded.node_features) {
        mix(hash, feature == 1.0f ? 1 : 0);
    }
    mix(hash, 0xFF);
    for (const int32_t action : legal_actions) mix_u32(hash, static_cast<uint32_t>(action));
    mix(hash, 0xFE);
    for (const uint8_t player : encoded.canonical_player_ids) mix(hash, player);
    return hash;
}

EvalOutcome DeterministicEvaluator::evaluate(const Encoded& encoded,
                                             const std::vector<int32_t>& legal_actions) {
    const uint64_t hash = request_hash(encoded, legal_actions);

    EvalOutcome outcome;
    outcome.value = unit(hash) * 2.0 - 1.0;  // in [-1, 1)

    outcome.priors.reserve(legal_actions.size());
    // Sequential left-to-right accumulation, matching Python's builtin sum().
    double total = 0.0;
    for (const int32_t action : legal_actions) {
        uint64_t action_hash = kFnvOffset;
        mix_u64(action_hash, hash);
        mix_u32(action_hash, static_cast<uint32_t>(action));
        // Shifted into [0.5, 1.5): every legal action keeps a strictly positive
        // prior, and the spread is wide enough to order the PUCT keys.
        const double weight = unit(action_hash) + 0.5;
        outcome.priors.push_back(weight);
        total += weight;
    }
    for (double& prior : outcome.priors) prior /= total;
    return outcome;
}

EvalOutcome UniformPriorEvaluator::evaluate(const Encoded& encoded,
                                            const std::vector<int32_t>& legal_actions) {
    const uint64_t hash = request_hash(encoded, legal_actions);
    EvalOutcome outcome;
    outcome.value = unit(hash) * 2.0 - 1.0;
    // 1.0 / n for every action: Python computes the same quotient from the same
    // integer, so the priors are bit-identical and the PUCT keys tie exactly.
    const double prior = 1.0 / static_cast<double>(legal_actions.size());
    outcome.priors.assign(legal_actions.size(), prior);
    return outcome;
}

}  // namespace soo
