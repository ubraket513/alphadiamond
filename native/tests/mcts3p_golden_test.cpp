// Min's search against the Python oracle's frozen answers, with no interpreter.
//
// The 2P gate cannot stand in for this one. Soo negates a scalar once per edge;
// Min backs a utility vector -- one component per seat -- through every ancestor
// unchanged, and two mistakes live in that difference alone: components landing
// on the wrong seats (the encoder rotates the canonical order per node, so the
// root's order is not every node's), and a node maximising somebody else's
// component. Neither crashes, and both play a different game.
//
// So the q digest is taken per seat id in ascending order rather than over a
// flattened vector: a permutation of the components changes it.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "golden.hpp"
#include "soo/board.hpp"
#include "soo/mcts3p.hpp"
#include "vector_evaluator.hpp"

namespace {

struct Case {
    std::string tag;
    int simulations = 0;
    soo::State state;
    int32_t selected = 0;
    uint64_t root_fnv = 0;
    uint64_t visit_fnv = 0;
    uint64_t policy_fnv = 0;
    uint64_t q_fnv = 0;
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

        if (kind == "mcts3p") {
            pending = Case{};
            int current = 0;
            int turn = 0;
            std::string finish;
            std::string occupancy;
            tokens >> pending.tag >> pending.simulations >> current >> turn >> finish >> occupancy;
            if (occupancy.size() != static_cast<std::size_t>(soo::kBoardSize)) {
                error = "bad occupancy for " + pending.tag;
                return false;
            }
            for (int i = 0; i < soo::kBoardSize; ++i) {
                pending.state.occupancy[i] = static_cast<uint8_t>(occupancy[i] - '0');
            }
            pending.state.current_player = static_cast<uint8_t>(current);
            pending.state.turn_number = static_cast<uint16_t>(turn);
            if (finish != "-") {
                std::istringstream ids(finish);
                std::string id;
                while (std::getline(ids, id, ',')) {
                    pending.state.finish_order[pending.state.finished_count++] =
                        static_cast<uint8_t>(std::stoi(id));
                }
            }
            have_pending = true;
            continue;
        }

        if (kind == "mexp3p") {
            if (!have_pending) {
                error = "mexp3p without a preceding mcts3p line";
                return false;
            }
            std::string root;
            std::string visits;
            std::string policy;
            std::string q_values;
            tokens >> pending.selected >> root >> visits >> policy >> q_values >> pending.calls >>
                pending.simulations_run;
            pending.root_fnv = std::stoull(root, nullptr, 16);
            pending.visit_fnv = std::stoull(visits, nullptr, 16);
            pending.policy_fnv = std::stoull(policy, nullptr, 16);
            pending.q_fnv = std::stoull(q_values, nullptr, 16);
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

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 3, "usage: mcts3p_golden_test <golden-dir> <topology-dir>");
    REQUIRE(soo::load_topology_from_dir(argv[2]), "could not load the golden topology tables");

    std::vector<Case> cases;
    std::string error;
    REQUIRE(load(std::string(argv[1]) + "/mcts3p-v1.txt", cases, error), error.c_str());
    CHECK(cases.size() >= 20);

    soo_test::Golden rules;
    REQUIRE(soo_test::load_golden(std::string(argv[1]) + "/rules-v1.txt", rules, error),
            error.c_str());
    const soo::Match& match = rules.match(3);
    REQUIRE(match.count == 3, "golden file is missing the 3P match line");

    bool saw_placed_seat = false;
    for (const Case& expected : cases) {
        if (expected.state.finished_count > 0) saw_placed_seat = true;

        soo::MCTSConfig config;
        config.simulations = expected.simulations;
        config.c_puct = 1.5;
        config.dirichlet_epsilon = 0.0;
        config.seed = 0;

        soo::SearchSession3P session(match, config);
        session.begin(expected.state, 0.0);
        while (session.advance() == soo::SearchSession3P::Status::NeedsEvaluation) {
            session.supply(
                soo_test::evaluate_vector(session.pending_features(), session.pending_actions()));
        }
        const soo::SearchResult3P& result = session.result();

        const std::string where = expected.tag + " sims=" + std::to_string(expected.simulations);

        if (result.selected_action != expected.selected) {
            soo_test::fail(__FILE__, __LINE__, where + ": selected action differs");
        }

        soo_test::Fnv root_digest;
        for (const int32_t action : result.root_actions) root_digest.i32(action);
        if (root_digest.value() != expected.root_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + ": root action order differs");
        }

        soo_test::Fnv visit_digest;
        for (const uint32_t visits : result.visit_counts) {
            visit_digest.i32(static_cast<int32_t>(visits));
        }
        if (visit_digest.value() != expected.visit_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + ": visit counts differ");
        }

        soo_test::Fnv policy_digest;
        for (const double value : result.policy) policy_digest.bytes(&value, sizeof(value));
        if (policy_digest.value() != expected.policy_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + ": policy differs");
        }

        // Per seat id in ascending order, matching the generator: hashing the
        // vector positionally would accept a permutation of the components.
        soo_test::Fnv q_digest;
        for (const soo::ValueVector& vector : result.q_vectors) {
            std::vector<std::pair<int32_t, double>> by_seat;
            for (uint8_t seat = 0; seat < match.count; ++seat) {
                by_seat.emplace_back(static_cast<int32_t>(match.players[seat].id), vector[seat]);
            }
            std::sort(by_seat.begin(), by_seat.end());
            for (const auto& [seat_id, value] : by_seat) {
                q_digest.i32(seat_id);
                q_digest.bytes(&value, sizeof(value));
            }
        }
        if (q_digest.value() != expected.q_fnv) {
            soo_test::fail(__FILE__, __LINE__, where + ": q vectors differ");
        }

        // `calls` in the golden file is the root action count, not an
        // evaluator call count; the search asks once for the root and once per
        // expansion, which the visit digest already pins.
        CHECK(result.root_actions.size() == expected.calls);
        if (result.simulations_run != expected.simulations_run) {
            soo_test::fail(__FILE__, __LINE__, where + ": simulation count differs");
        }
    }

    CHECK(saw_placed_seat);
    std::fprintf(stderr, "checked %zu golden 3P searches\n", cases.size());
    return soo_test::report("mcts3p_golden_test");
}
