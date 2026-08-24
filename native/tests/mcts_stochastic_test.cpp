// Gate E in C++: the samplers' *distributions* and the search's determinism.
//
// Section 9 of the design settles that cross-backend bit-exact RNG parity is
// not required -- CPython's gammavariate consumes a variable number of draws
// per sample, so matching its stream would mean reimplementing a rejection
// algorithm on MT19937 for reproducibility no experiment needs. What is
// required is that the distribution and the semantics match, and that each
// backend is deterministic for a given seed.
//
// That has a sharp consequence for testing: because the streams are allowed to
// differ, no comparison of sequences can catch a wrong sampler. A boost
// exponent inverted, weights normalised twice, noise applied after the first
// selection instead of before -- every one of those still produces a plausible
// stream. They are only visible in moments and frequencies, which is what this
// checks.
//
// The Python gate (tests/native/test_stochastic_parity.py) additionally
// compares the gamma CDF against CPython itself. That comparison needs CPython
// and stays in the bridge lane; everything here runs with no interpreter.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "check.hpp"
#include "golden.hpp"
#include "soo/board.hpp"
#include "soo/evaluator.hpp"
#include "soo/mcts.hpp"
#include "soo/random.hpp"

namespace {

constexpr int kSamples = 200000;

struct Moments {
    double mean = 0.0;
    double variance = 0.0;
};

Moments moments(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) sum += value;
    const double mean = sum / static_cast<double>(values.size());
    double squared = 0.0;
    for (const double value : values) squared += (value - mean) * (value - mean);
    return {mean, squared / static_cast<double>(values.size() - 1)};
}

std::vector<double> gamma_samples(double alpha, uint64_t seed, int count = kSamples) {
    soo::Rng rng(seed);
    std::vector<double> values;
    values.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) values.push_back(rng.gammavariate(alpha));
    return values;
}

// Pearson correlation, for the "neighbouring seeds decorrelate" check.
double correlation(const std::vector<double>& left, const std::vector<double>& right) {
    const Moments a = moments(left);
    const Moments b = moments(right);
    double covariance = 0.0;
    for (size_t i = 0; i < left.size(); ++i) {
        covariance += (left[i] - a.mean) * (right[i] - b.mean);
    }
    covariance /= static_cast<double>(left.size() - 1);
    return covariance / (std::sqrt(a.variance) * std::sqrt(b.variance));
}

bool threw_invalid_argument(void (*body)()) {
    try {
        body();
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 3, "usage: mcts_stochastic_test <golden-dir> <topology-dir>");
    REQUIRE(soo::load_topology_from_dir(argv[2]), "could not load the golden topology tables");

    // ---------------------------------------------------------------- gamma
    // Gamma(alpha, 1) has mean = alpha and variance = alpha. The mean alone
    // does not pin the sampler: an inverted boost exponent leaves it right and
    // the variance wrong, which is exactly the mistake worth catching. 0.3 is
    // production's dirichlet_alpha, and being below 1 it takes the boost branch.
    for (const double alpha : {0.3, 1.0, 2.5}) {
        const Moments m = moments(gamma_samples(alpha, 20260824));
        const double tolerance = 0.05 * alpha;  // ~5%, generous for 200k draws
        const std::string where = "alpha=" + std::to_string(alpha);
        if (std::fabs(m.mean - alpha) > tolerance) {
            soo_test::fail(__FILE__, __LINE__, where + ": gamma mean is " + std::to_string(m.mean));
        }
        if (std::fabs(m.variance - alpha) > tolerance) {
            soo_test::fail(__FILE__, __LINE__,
                           where + ": gamma variance is " + std::to_string(m.variance));
        }
        // Gamma is supported on (0, inf): a zero would divide by zero upstream.
        for (const double value : gamma_samples(alpha, 7, 1000)) CHECK(value > 0.0);
    }

    // Deterministic per seed, and neighbouring seeds must not correlate: lanes
    // are seeded from consecutive integers, so a generator that fails this
    // recreates the salt bug in a new place.
    CHECK(gamma_samples(0.3, 11, 500) == gamma_samples(0.3, 11, 500));
    CHECK(gamma_samples(0.3, 11, 500) != gamma_samples(0.3, 12, 500));
    const double neighbours = correlation(gamma_samples(0.3, 1000, 20000),
                                          gamma_samples(0.3, 1001, 20000));
    if (std::fabs(neighbours) > 0.05) {
        soo_test::fail(__FILE__, __LINE__,
                       "neighbouring seeds correlate: r=" + std::to_string(neighbours));
    }

    CHECK(threw_invalid_argument([] {
        soo::Rng rng(0);
        (void)rng.gammavariate(0.0);
    }));
    CHECK(threw_invalid_argument([] {
        soo::Rng rng(0);
        (void)rng.gammavariate(-1.0);
    }));

    // ------------------------------------------------------ weighted choice
    // Frequencies must follow the weights, not merely stay inside the
    // population: normalising twice, or sampling the index uniformly, both pass
    // a "returns a legal index" check.
    {
        const std::vector<double> weights = {0.1, 0.3, 0.6};
        soo::Rng rng(4242);
        std::vector<int> counts(weights.size(), 0);
        for (int i = 0; i < kSamples; ++i) ++counts[rng.weighted_index(weights)];
        for (size_t index = 0; index < weights.size(); ++index) {
            const double observed = static_cast<double>(counts[index]) / kSamples;
            if (std::fabs(observed - weights[index]) > 0.01) {
                soo_test::fail(__FILE__, __LINE__,
                               "weighted choice frequency " + std::to_string(observed) +
                                   " != weight " + std::to_string(weights[index]));
            }
        }
    }

    CHECK(threw_invalid_argument([] {
        soo::Rng rng(0);
        (void)rng.weighted_index({});
    }));
    CHECK(threw_invalid_argument([] {
        soo::Rng rng(0);
        (void)rng.weighted_index({0.0, 0.0});
    }));

    // ---------------------------------------------------- search-level knobs
    soo_test::Golden golden;
    std::string error;
    REQUIRE(soo_test::load_golden(std::string(argv[1]) + "/rules-v1.txt", golden, error),
            error.c_str());
    const soo::Match& match = golden.match(2);
    const soo::State* opening = nullptr;
    for (const soo_test::GoldenCase& entry : golden.cases) {
        if (entry.tag == "opening" && entry.player_count == 2) {
            opening = &entry.state;
            break;
        }
    }
    REQUIRE(opening != nullptr, "golden file has no 2P opening position");

    soo::DeterministicEvaluator evaluator;
    const auto search = [&](double epsilon, double temperature, uint64_t seed) {
        soo::MCTSConfig config;
        config.simulations = 64;
        config.c_puct = 1.5;
        config.dirichlet_alpha = 0.3;
        config.dirichlet_epsilon = epsilon;
        config.seed = seed;
        soo::MCTS2P mcts(match, evaluator, config);
        return mcts.run(*opening, temperature, false);
    };

    // epsilon = 0 draws nothing: the priors are the evaluator's, untouched.
    // This is what keeps the deterministic gates meaningful.
    const soo::SearchResult quiet_a = search(0.0, 0.0, 1);
    const soo::SearchResult quiet_b = search(0.0, 0.0, 999);
    CHECK(quiet_a.root_priors == quiet_b.root_priors);

    // epsilon > 0 mixes, and the mixture is still a distribution. A sampler
    // that normalised twice, or mixed before normalising, lands here.
    const soo::SearchResult noisy = search(0.25, 0.0, 1);
    CHECK(noisy.root_priors.size() == quiet_a.root_priors.size());
    double mixed_total = 0.0;
    bool moved = false;
    for (size_t index = 0; index < noisy.root_priors.size(); ++index) {
        CHECK(noisy.root_priors[index] >= 0.0);
        mixed_total += noisy.root_priors[index];
        if (noisy.root_priors[index] != quiet_a.root_priors[index]) moved = true;
    }
    CHECK(moved);
    if (std::fabs(mixed_total - 1.0) > 1e-9) {
        soo_test::fail(__FILE__, __LINE__,
                       "mixed priors sum to " + std::to_string(mixed_total));
    }

    // Every mixed prior is (1-e) * p + e * n with n >= 0, so it can never fall
    // below its own shrunk term: an inverted mixture breaks this immediately.
    for (size_t index = 0; index < noisy.root_priors.size(); ++index) {
        CHECK(noisy.root_priors[index] >= 0.75 * quiet_a.root_priors[index] - 1e-12);
    }

    // A stochastic search is reproducible from its seed, and different seeds
    // explore differently -- both halves matter, and a fixed stream that
    // ignored the seed would pass only the first.
    {
        const soo::SearchResult first = search(0.25, 1.0, 4242);
        const soo::SearchResult again = search(0.25, 1.0, 4242);
        CHECK(first.selected_action == again.selected_action);
        CHECK(first.visit_counts == again.visit_counts);
        CHECK(first.root_priors == again.root_priors);

        bool differs = false;
        for (uint64_t seed = 1; seed <= 8 && !differs; ++seed) {
            if (search(0.25, 1.0, seed).visit_counts != first.visit_counts) differs = true;
        }
        CHECK(differs);
    }

    // Temperature: at 0 the choice is the deterministic tie-break (most visits,
    // smallest action id); above 0 it samples, and over many seeds it must
    // actually visit more than one action -- otherwise self-play at
    // temperature > 0 would still play one game per lane, which is the failure
    // this knob exists to prevent.
    {
        const soo::SearchResult deterministic = search(0.0, 0.0, 5);
        size_t best = 0;
        for (size_t index = 1; index < deterministic.root_actions.size(); ++index) {
            if (deterministic.visit_counts[index] > deterministic.visit_counts[best] ||
                (deterministic.visit_counts[index] == deterministic.visit_counts[best] &&
                 deterministic.root_actions[index] < deterministic.root_actions[best])) {
                best = index;
            }
        }
        CHECK(deterministic.selected_action == deterministic.root_actions[best]);

        std::map<int32_t, int> chosen;
        for (uint64_t seed = 1; seed <= 40; ++seed) ++chosen[search(0.0, 1.0, seed).selected_action];
        CHECK(chosen.size() > 1);

        // A low temperature concentrates: it must not be *more* spread out.
        std::map<int32_t, int> cold;
        for (uint64_t seed = 1; seed <= 40; ++seed) ++cold[search(0.0, 0.15, seed).selected_action];
        CHECK(cold.size() <= chosen.size());
    }

    std::fprintf(stderr, "gamma, weighted choice, mixture and temperature checked\n");
    return soo_test::report("mcts_stochastic_test");
}
