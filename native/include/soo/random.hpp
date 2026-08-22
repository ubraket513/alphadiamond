// Deterministic sampling for stochastic MCTS.
//
// Python authority: diamond.alphazero.mcts.puct.add_dirichlet_noise and
// select_from_visits, which draw from `random.Random`.
//
// Section 9 of docs/native_selfplay_phase0.md settles what parity means here:
// cross-backend bit-exact RNG parity is **not** required.  CPython's
// `gammavariate` uses a rejection algorithm whose draw count varies per sample,
// so reproducing its stream would mean reimplementing that algorithm on
// MT19937 -- reproducibility no experiment needs.  What is required is that the
// *distribution and semantics* match and that each backend is deterministic for
// a given seed.  Both are tested; the draw sequence deliberately is not.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace soo {

// xoshiro256**, seeded through splitmix64 so that neighbouring seeds (lane
// index, move number) produce unrelated streams rather than correlated ones.
// This matters more than it looks: lanes are seeded from consecutive integers,
// and a generator that does not decorrelate them recreates the Gate C salt bug
// in a new place -- see pitfall 7.1.
class Rng {
  public:
    explicit Rng(uint64_t seed = 0) { reseed(seed); }

    void reseed(uint64_t seed) {
        uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
        for (uint64_t& part : state_) {
            uint64_t x = (z += 0x9e3779b97f4a7c15ULL);
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            part = x ^ (x >> 31);
        }
    }

    uint64_t next() {
        const uint64_t result = rotl(state_[1] * 5, 7) * 9;
        const uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    // Half-open [0, 1), 53 significant bits -- the same resolution as CPython's
    // random().
    double uniform01() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

    // Open (0, 1).  The gamma sampler takes logs and divides by the draw, so a
    // literal zero would produce -inf / a division by zero once in 2^53 draws.
    // Rare enough never to be seen in testing and fatal when it happens.
    double uniform01_open() {
        for (;;) {
            const double u = uniform01();
            if (u > 0.0) return u;
        }
    }

    // Standard normal, Marsaglia polar.  Only the gamma sampler uses it, so the
    // second variate is discarded rather than cached; caching it would make the
    // stream depend on call history for no benefit.
    double normal() {
        for (;;) {
            const double u = 2.0 * uniform01() - 1.0;
            const double v = 2.0 * uniform01() - 1.0;
            const double s = u * u + v * v;
            if (s > 0.0 && s < 1.0) return u * std::sqrt(-2.0 * std::log(s) / s);
        }
    }

    // ``random.Random.gammavariate(alpha, 1.0)``.
    //
    // Marsaglia-Tsang, which is exact for alpha >= 1.  Production uses
    // alpha = 0.3, so the alpha < 1 branch is the one that actually runs: draw
    // from Gamma(alpha + 1) and scale by U^(1/alpha), the standard boost.
    // Getting that boost wrong is invisible in the mean and obvious in the
    // variance, which is why the tests check both moments.
    double gammavariate(double alpha) {
        if (alpha <= 0.0) throw std::invalid_argument("gamma alpha must be positive");
        if (alpha < 1.0) {
            const double boosted = gammavariate(alpha + 1.0);
            return boosted * std::pow(uniform01_open(), 1.0 / alpha);
        }
        const double d = alpha - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            double x, v;
            do {
                x = normal();
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            const double u = uniform01_open();
            const double xx = x * x;
            if (u < 1.0 - 0.0331 * xx * xx) return d * v;
            if (std::log(u) < 0.5 * xx + d * (1.0 - v + std::log(v))) return d * v;
        }
    }

    // ``random.choices(population, weights, k=1)[0]``: cumulative weights, one
    // uniform draw scaled by the total, and the first index whose running total
    // exceeds it.  Returns an index into ``weights``.
    size_t weighted_index(const std::vector<double>& weights) {
        if (weights.empty()) throw std::invalid_argument("cannot choose from an empty population");
        double total = 0.0;
        for (const double w : weights) total += w;
        if (!(total > 0.0)) throw std::invalid_argument("weights must sum to a positive value");
        const double target = uniform01() * total;
        double running = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            running += weights[i];
            if (target < running) return i;
        }
        // Only reachable through floating-point accumulation error; the last
        // index is the correct answer there, not an error.
        return weights.size() - 1;
    }

  private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t state_[4] = {0, 0, 0, 0};
};

}  // namespace soo
