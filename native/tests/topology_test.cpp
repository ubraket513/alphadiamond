// The topology tables are data exported from the Python board.  This test does
// not re-derive them -- it checks the loader refuses malformed input and that
// the loaded tables satisfy the invariants the rest of the engine assumes.
#include <cstddef>
#include <string>
#include <vector>

#include "check.hpp"
#include "soo/board.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc >= 2, "usage: topology_test <golden-topology-dir>");
    const std::string root = argv[1];

    CHECK(!soo::load_topology_from_dir(root + "/does-not-exist"));
    CHECK(!soo::mutable_topology().configured);

    REQUIRE(soo::load_topology_from_dir(root), "could not load the golden topology tables");
    const soo::Topology& topology = soo::topology();

    // Neighbours are symmetric under the opposite direction, and -1 means edge.
    for (int position = 0; position < soo::kBoardSize; ++position) {
        for (int direction = 0; direction < soo::kDirections; ++direction) {
            const int neighbour = topology.neighbour[position][direction];
            CHECK(neighbour >= -1 && neighbour < soo::kBoardSize);
            if (neighbour < 0) continue;
            const int opposite = (direction + soo::kDirections / 2) % soo::kDirections;
            CHECK_EQ(static_cast<int>(topology.neighbour[neighbour][opposite]), position);
        }
    }

    // Six camps of ten holes each. They are not disjoint: adjacent camps share
    // their corner hole, so exactly six holes belong to two camps.
    std::vector<int> camp_count(soo::kBoardSize, 0);
    for (int camp = 0; camp < soo::kCamps; ++camp) {
        std::vector<int> seen;
        for (int index = 0; index < soo::kCampSize; ++index) {
            const int position = topology.camp_positions[camp][index];
            CHECK(position >= 0 && position < soo::kBoardSize);
            for (const int other : seen) CHECK(other != position);
            seen.push_back(position);
            ++camp_count[position];
        }
    }
    int shared = 0;
    int owned = 0;
    for (const int count : camp_count) {
        CHECK(count <= 2);
        if (count >= 1) ++owned;
        if (count == 2) ++shared;
    }
    CHECK_EQ(shared, 6);
    CHECK_EQ(owned, soo::kCamps * soo::kCampSize - 6);

    // Pairwise distance is a metric on the board: zero diagonal, symmetric,
    // and finite everywhere because the board is connected.
    for (int row = 0; row < soo::kBoardSize; ++row) {
        CHECK_EQ(static_cast<int>(topology.pairwise[row][row]), 0);
        for (int column = 0; column < soo::kBoardSize; ++column) {
            CHECK_EQ(topology.pairwise[row][column], topology.pairwise[column][row]);
            if (row != column) CHECK(topology.pairwise[row][column] > 0);
        }
    }

    // The canonical rotation is a bijection per camp.
    for (int camp = 0; camp < soo::kCamps; ++camp) {
        for (int position = 0; position < soo::kBoardSize; ++position) {
            const int canonical = topology.physical_to_canonical[camp][position];
            CHECK(canonical >= 0 && canonical < soo::kBoardSize);
            CHECK_EQ(static_cast<int>(topology.canonical_to_physical[camp][canonical]), position);
        }
    }

    // Rotation preserves distance -- the encoder's canonical space would be
    // meaningless otherwise.
    for (int camp = 0; camp < soo::kCamps; ++camp) {
        for (int a = 0; a < soo::kBoardSize; ++a) {
            for (int b = 0; b < soo::kBoardSize; ++b) {
                const int ca = topology.canonical_to_physical[camp][a];
                const int cb = topology.canonical_to_physical[camp][b];
                CHECK_EQ(topology.pairwise[ca][cb],
                         topology.pairwise[topology.canonical_to_physical[soo::kCampZNeg][a]]
                                          [topology.canonical_to_physical[soo::kCampZNeg][b]]);
            }
        }
    }

    return soo_test::report("topology_test");
}
