// Generating the board tables, rather than being handed them.
//
// Every table here was previously injected from Python
// (src/diamond/contract/board.py, exported by
// src/diamond/alphazero/native/topology.py), which made the geometry the one
// part of the game the C++ core could not produce for itself: a build with no
// Python could load the tables but never derive them, so "the C++ core is the
// authority" had a hole in it exactly the size of the board.
//
// This is a port, not a transcription. Nothing below is a table of constants;
// it is the same construction the Python board performs -- the two overlapping
// triangles, the camp inequalities with their clip, the (z, x) ordering that
// fixes position ids, the neighbour lattice, breadth-first distances and the
// canonical rotation. `topology_test` requires the result to equal the frozen
// exported tables byte for byte, which is what makes the two implementations
// one contract rather than two opinions.
#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "soo/board.hpp"

namespace soo {
namespace {

// Half-width of the central hexagon: 7 holes across, 4 holes to a side.
constexpr int kHexRadius = 3;
// A hole reaches a camp when one axis hits +/- this. Equal to the hexagon's
// radius and not one past it: a Diamond camp includes the hexagon side it
// stands on, which is what makes it ten holes rather than six.
constexpr int kCampThreshold = kHexRadius;

struct Cube {
    int x = 0;
    int y = 0;
    int z = 0;
};

bool in_triangle_up(const Cube& c) {
    return c.x >= -kHexRadius && c.y >= -kHexRadius && c.z >= -kHexRadius;
}

bool in_triangle_down(const Cube& c) {
    return c.x <= kHexRadius && c.y <= kHexRadius && c.z <= kHexRadius;
}

// The six lattice directions, in the order that breaks ties in jump-path
// search. Reordering them changes the game.
constexpr std::array<Cube, kDirections> kDirectionVectors{{
    {1, -1, 0}, {1, 0, -1}, {0, 1, -1}, {-1, 1, 0}, {-1, 0, 1}, {0, -1, 1},
}};

// Camp index order matches the Python ``Camp`` enum: x+ y+ z+ x- y- z-.
// Axis 0/1/2 is x/y/z; a "+" camp is a corner of triangle "up" and is clipped
// to it, a "-" camp likewise to "down". Without that clip the far tip of a
// third star point satisfies two camp inequalities at once and two starting
// camps overlap.
int camp_index(int axis, bool positive) { return positive ? axis : axis + 3; }

std::vector<int> camps_of(const Cube& c) {
    const std::array<int, 3> axes{c.x, c.y, c.z};
    std::vector<int> found;
    for (int axis = 0; axis < 3; ++axis) {
        if (axes[static_cast<std::size_t>(axis)] >= kCampThreshold && in_triangle_up(c)) {
            found.push_back(camp_index(axis, true));
        } else if (axes[static_cast<std::size_t>(axis)] <= -kCampThreshold &&
                   in_triangle_down(c)) {
            found.push_back(camp_index(axis, false));
        }
    }
    return found;
}

// Position ids are assigned by sorting holes by (z, x) -- rendering rows top to
// bottom, then left to right. Deterministic, and persisted in saved games.
std::vector<Cube> generate_cubes() {
    const int limit = kHexRadius + kCampSize;  // generous; the predicates decide
    std::vector<Cube> cubes;
    for (int z = -limit; z <= limit; ++z) {
        for (int x = -limit; x <= limit; ++x) {
            const Cube cube{x, -x - z, z};
            if (in_triangle_up(cube) || in_triangle_down(cube)) cubes.push_back(cube);
        }
    }
    return cubes;  // already in (z, x) order by construction of the loops
}

// The rotation that puts a seat's home camp where the encoder expects it.
Cube canonical_cube(const Cube& c, int camp) {
    const int axis = camp % 3;
    const int factor = camp < 3 ? 1 : -1;
    Cube rotated;
    if (axis == 0) {
        rotated = {c.y, c.z, c.x};
    } else if (axis == 1) {
        rotated = {c.z, c.x, c.y};
    } else {
        rotated = {c.x, c.y, c.z};
    }
    return {factor * rotated.x, factor * rotated.y, factor * rotated.z};
}

}  // namespace

Topology generate_topology() {
    const std::vector<Cube> cubes = generate_cubes();
    if (cubes.size() != static_cast<std::size_t>(kBoardSize)) {
        throw std::runtime_error("board generation produced the wrong number of holes");
    }

    std::map<std::tuple<int, int, int>, int> id_of;
    for (int id = 0; id < kBoardSize; ++id) {
        const Cube& c = cubes[static_cast<std::size_t>(id)];
        id_of[{c.x, c.y, c.z}] = id;
    }

    Topology topo;

    for (int id = 0; id < kBoardSize; ++id) {
        const Cube& c = cubes[static_cast<std::size_t>(id)];
        for (int direction = 0; direction < kDirections; ++direction) {
            const Cube& d = kDirectionVectors[static_cast<std::size_t>(direction)];
            const auto found = id_of.find({c.x + d.x, c.y + d.y, c.z + d.z});
            topo.neighbour[static_cast<std::size_t>(id)][static_cast<std::size_t>(direction)] =
                found == id_of.end() ? static_cast<int8_t>(-1)
                                     : static_cast<int8_t>(found->second);
        }
    }

    std::array<std::vector<int>, kCamps> camp_members;
    for (int id = 0; id < kBoardSize; ++id) {
        for (const int camp : camps_of(cubes[static_cast<std::size_t>(id)])) {
            camp_members[static_cast<std::size_t>(camp)].push_back(id);
        }
    }
    for (int camp = 0; camp < kCamps; ++camp) {
        auto& members = camp_members[static_cast<std::size_t>(camp)];
        if (members.size() != static_cast<std::size_t>(kCampSize)) {
            throw std::runtime_error("a camp triangle came out the wrong size");
        }
        for (int slot = 0; slot < kCampSize; ++slot) {
            topo.camp_positions[static_cast<std::size_t>(camp)][static_cast<std::size_t>(slot)] =
                static_cast<uint8_t>(members[static_cast<std::size_t>(slot)]);
        }
    }

    // All-pairs neighbour-graph distance, one breadth-first sweep per origin.
    for (int origin = 0; origin < kBoardSize; ++origin) {
        std::array<int, kBoardSize> distance{};
        distance.fill(-1);
        distance[static_cast<std::size_t>(origin)] = 0;
        std::deque<int> queue{origin};
        while (!queue.empty()) {
            const int current = queue.front();
            queue.pop_front();
            for (const int8_t next : topo.neighbour[static_cast<std::size_t>(current)]) {
                if (next < 0 || distance[static_cast<std::size_t>(next)] >= 0) continue;
                distance[static_cast<std::size_t>(next)] =
                    distance[static_cast<std::size_t>(current)] + 1;
                queue.push_back(next);
            }
        }
        for (int target = 0; target < kBoardSize; ++target) {
            if (distance[static_cast<std::size_t>(target)] < 0) {
                throw std::runtime_error("board graph is disconnected; distances are undefined");
            }
            topo.pairwise[static_cast<std::size_t>(origin)][static_cast<std::size_t>(target)] =
                static_cast<uint8_t>(distance[static_cast<std::size_t>(target)]);
        }
    }

    for (int camp = 0; camp < kCamps; ++camp) {
        for (int id = 0; id < kBoardSize; ++id) {
            const Cube rotated = canonical_cube(cubes[static_cast<std::size_t>(id)], camp);
            const auto found = id_of.find({rotated.x, rotated.y, rotated.z});
            if (found == id_of.end()) {
                throw std::runtime_error("the canonical rotation left the board");
            }
            const auto canonical = static_cast<uint8_t>(found->second);
            topo.physical_to_canonical[static_cast<std::size_t>(camp)]
                                      [static_cast<std::size_t>(id)] = canonical;
            topo.canonical_to_physical[static_cast<std::size_t>(camp)]
                                      [static_cast<std::size_t>(canonical)] =
                static_cast<uint8_t>(id);
        }
    }

    topo.configured = true;
    return topo;
}

}  // namespace soo
