// Gate A, in C++ and without Python: for every position in the frozen corpus
// the native engine must reproduce the Python oracle's legal actions (as an
// ordered sequence), every successor state, the canonical encoding and the
// bootstrap prior.  Expectations come from tests/golden/rules-v1.txt.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "check.hpp"
#include "golden.hpp"
#include "soo/board.hpp"
#include "soo/encoder.hpp"
#include "soo/prior.hpp"
#include "soo/rules.hpp"

namespace {

constexpr double kPriorTolerance = 1e-9;

uint64_t hash_actions(const std::vector<int32_t>& actions) {
    soo_test::Fnv digest;
    for (const int32_t action : actions) digest.i32(action);
    return digest.value();
}

void hash_state(soo_test::Fnv& digest, const soo::State& state) {
    for (const uint8_t cell : state.occupancy) digest.byte(cell);
    digest.byte(state.current_player);
    digest.byte(state.status);
    digest.byte(static_cast<uint8_t>(state.turn_number & 0xFF));
    digest.byte(static_cast<uint8_t>((state.turn_number >> 8) & 0xFF));
    digest.byte(state.finished_count);
    for (uint8_t i = 0; i < state.finished_count; ++i) digest.byte(state.finish_order[i]);
}

uint64_t hash_encoded(const soo::Encoded& encoded) {
    soo_test::Fnv digest;
    const int rows = encoded.feature_count > 0
                         ? static_cast<int>(encoded.node_features.size()) / encoded.feature_count
                         : 0;
    digest.i32(rows);
    digest.i32(encoded.feature_count);
    for (const uint8_t id : encoded.canonical_player_ids) digest.byte(id);
    for (const float value : encoded.node_features) digest.f32(value);
    return digest.value();
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 3, "usage: rules_golden_test <golden-dir> <topology-dir>");
    const std::string golden_path = std::string(argv[1]) + "/rules-v1.txt";
    REQUIRE(soo::load_topology_from_dir(argv[2]), "could not load the golden topology tables");

    soo_test::Golden golden;
    std::string error;
    REQUIRE(soo_test::load_golden(golden_path, golden, error), error.c_str());
    CHECK(golden.cases.size() >= 500);

    for (const soo_test::GoldenCase& expected : golden.cases) {
        const soo::Match& match = golden.match(expected.player_count);
        REQUIRE(match.count == expected.player_count, "golden file is missing a match line");
        const soo::State& state = expected.state;

        std::vector<int32_t> physical;
        soo::legal_action_ids(state, physical);
        std::vector<int32_t> canonical;
        soo::canonical_legal_action_ids(state, match, canonical);

        if (static_cast<int>(physical.size()) != expected.action_count ||
            hash_actions(physical) != expected.physical_fnv ||
            hash_actions(canonical) != expected.canonical_fnv) {
            soo_test::fail(__FILE__, __LINE__,
                           expected.tag + " (" + std::to_string(expected.player_count) +
                               "P): legal actions differ from Python; native=" +
                               std::to_string(physical.size()) + " python=" +
                               std::to_string(expected.action_count) +
                               (hash_actions(physical) == expected.physical_fnv ? " physical-ok"
                                                                               : " physical-differs") +
                               (hash_actions(canonical) == expected.canonical_fnv
                                    ? " canonical-ok"
                                    : " canonical-differs"));
            continue;
        }

        // The encoding must be bit-identical: the network's input is the whole
        // contract between the trainer and the shipped application.
        if (hash_encoded(soo::encode(state, match)) != expected.encoded_fnv) {
            soo_test::fail(__FILE__, __LINE__, expected.tag + ": node features differ from Python");
        }

        if (expected.state.status == soo::kFinished) continue;

        soo_test::Fnv successors;
        for (const int32_t action : physical) {
            hash_state(successors, soo::apply_action(state, match, action));
        }
        if (successors.value() != expected.successor_fnv) {
            soo_test::fail(__FILE__, __LINE__, expected.tag + ": a successor state differs");
        }

        std::vector<double> priors;
        soo::vacancy_prior(canonical, soo::canonical_self_occupancy(state, match), priors);
        if (static_cast<int>(priors.size()) != expected.prior_count) {
            soo_test::fail(__FILE__, __LINE__, expected.tag + ": prior length differs");
            continue;
        }
        double maximum = 0.0;
        double dot = 0.0;
        for (std::size_t i = 0; i < priors.size(); ++i) {
            maximum = priors[i] > maximum ? priors[i] : maximum;
            dot += static_cast<double>(i) * priors[i];
        }
        if (std::fabs(maximum - expected.prior_max) > kPriorTolerance ||
            std::fabs(dot - expected.prior_dot) > kPriorTolerance) {
            soo_test::fail(__FILE__, __LINE__, expected.tag + ": bootstrap prior differs");
        }
    }

    std::fprintf(stderr, "checked %zu golden positions\n", golden.cases.size());
    return soo_test::report("rules_golden_test");
}
