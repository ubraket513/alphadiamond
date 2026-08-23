// Gate B, in C++ and without Python: the deterministic search must reproduce
// the Python oracle's root statistics *and* its evaluator request sequence.
//
// Root visit counts alone are not enough. Two searches can arrive at the same
// counts through different traversals, so the request sequence -- which leaves
// were evaluated, in which order -- is what pins the descent.
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "golden.hpp"
#include "soo/board.hpp"
#include "soo/evaluator.hpp"
#include "soo/mcts.hpp"
#include "soo/state.hpp"

namespace {

struct Case {
    std::string tag;
    std::string evaluator;
    int simulations = 0;
    soo::State state;
    int32_t selected = 0;
    uint64_t root_fnv = 0;
    uint64_t visit_fnv = 0;
    uint64_t q_fnv = 0;
    uint64_t policy_fnv = 0;
    uint64_t trace_fnv = 0;
    uint32_t calls = 0;
    uint32_t simulations_run = 0;
};

bool load(const std::string& path, std::vector<Case>& out, std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "cannot open " + path;
        return false;
    }
    std::string line;
    Case pending;
    bool have_pending = false;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream tokens(line);
        std::string kind;
        tokens >> kind;

        if (kind == "mcts") {
            pending = Case{};
            int current = 0;
            int turn = 0;
            std::string occupancy;
            tokens >> pending.tag >> pending.evaluator >> pending.simulations >> current >> turn >>
                occupancy;
            if (occupancy.size() != static_cast<std::size_t>(soo::kBoardSize)) {
                error = "bad occupancy for " + pending.tag;
                return false;
            }
            for (int i = 0; i < soo::kBoardSize; ++i) {
                pending.state.occupancy[i] = static_cast<uint8_t>(occupancy[i] - '0');
            }
            pending.state.current_player = static_cast<uint8_t>(current);
            pending.state.turn_number = static_cast<uint16_t>(turn);
            have_pending = true;
            continue;
        }

        if (kind == "mexp") {
            if (!have_pending) {
                error = "mexp line without a preceding mcts line";
                return false;
            }
            std::string root;
            std::string visits;
            std::string q_values;
            std::string policy;
            std::string trace;
            tokens >> pending.selected >> root >> visits >> q_values >> policy >> trace >>
                pending.calls >> pending.simulations_run;
            pending.root_fnv = std::stoull(root, nullptr, 16);
            pending.visit_fnv = std::stoull(visits, nullptr, 16);
            pending.q_fnv = std::stoull(q_values, nullptr, 16);
            pending.policy_fnv = std::stoull(policy, nullptr, 16);
            pending.trace_fnv = std::stoull(trace, nullptr, 16);
            out.push_back(pending);
            have_pending = false;
            continue;
        }

        error = "unknown record: " + kind;
        return false;
    }
    if (out.empty()) {
        error = "golden file has no searches";
        return false;
    }
    return true;
}

uint64_t hash_i32(const std::vector<int32_t>& values) {
    soo_test::Fnv digest;
    for (const int32_t value : values) digest.i32(value);
    return digest.value();
}

uint64_t hash_visits(const std::vector<uint32_t>& values) {
    soo_test::Fnv digest;
    for (const uint32_t value : values) digest.i32(static_cast<int32_t>(value));
    return digest.value();
}

uint64_t hash_doubles(const std::vector<double>& values) {
    soo_test::Fnv digest;
    for (const double value : values) digest.bytes(&value, sizeof(value));
    return digest.value();
}

uint64_t hash_trace(const std::vector<soo::EvalRecord>& trace) {
    soo_test::Fnv digest;
    for (const soo::EvalRecord& record : trace) {
        digest.bytes(&record.request_hash, sizeof(record.request_hash));
        for (const int32_t action : record.legal_actions) digest.i32(action);
    }
    return digest.value();
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 3, "usage: mcts_golden_test <golden-dir> <topology-dir>");
    REQUIRE(soo::load_topology_from_dir(argv[2]), "could not load the golden topology tables");

    std::vector<Case> cases;
    std::string error;
    REQUIRE(load(std::string(argv[1]) + "/mcts-v1.txt", cases, error), error.c_str());
    CHECK(cases.size() >= 100);

    // MCTS2P is two-player only; the golden file only contains 2P positions.
    soo_test::Golden rules;
    REQUIRE(soo_test::load_golden(std::string(argv[1]) + "/rules-v1.txt", rules, error),
            error.c_str());
    const soo::Match& match = rules.match(2);
    REQUIRE(match.count == 2, "golden file is missing the 2P match line");

    soo::DeterministicEvaluator hash_evaluator;
    soo::UniformPriorEvaluator uniform_evaluator;

    for (const Case& expected : cases) {
        soo::Evaluator& evaluator = expected.evaluator == "uniform"
                                        ? static_cast<soo::Evaluator&>(uniform_evaluator)
                                        : static_cast<soo::Evaluator&>(hash_evaluator);
        soo::MCTSConfig config;
        config.simulations = expected.simulations;
        config.c_puct = 1.5;
        config.dirichlet_epsilon = 0.0;
        config.seed = 0;

        soo::MCTS2P search(match, evaluator, config);
        const soo::SearchResult result = search.run(expected.state, 0.0, true);

        const std::string where =
            expected.tag + " " + expected.evaluator + " sims=" +
            std::to_string(expected.simulations) + ": ";

        if (result.selected_action != expected.selected) {
            soo_test::fail(__FILE__, __LINE__, where + "selected action differs");
        }
        // Root action order is the expansion order, and it is observable: it is
        // the order Dirichlet noise components are assigned in.
        if (hash_i32(result.root_actions) != expected.root_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + "root action order differs");
        }
        if (hash_visits(result.visit_counts) != expected.visit_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + "visit counts differ");
        }
        // Bit-exact, not within a tolerance: the PUCT key is a double
        // comparison, so one ulp can flip a selection and change the descent.
        if (hash_doubles(result.q_values) != expected.q_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + "q values differ");
        }
        if (hash_doubles(result.policy) != expected.policy_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + "policy differs");
        }
        if (hash_trace(result.trace) != expected.trace_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + "evaluator request sequence differs");
        }
        if (result.evaluator_calls != expected.calls) {
            soo_test::fail(__FILE__, __LINE__, where + "evaluator call count differs");
        }
        if (result.simulations_run != expected.simulations_run) {
            soo_test::fail(__FILE__, __LINE__, where + "simulation count differs");
        }
    }

    std::fprintf(stderr, "checked %zu golden searches\n", cases.size());
    return soo_test::report("mcts_golden_test");
}
