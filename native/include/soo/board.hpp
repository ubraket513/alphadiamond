// Fixed board topology.  The tables are generated here (src/topology_gen.cpp)
// from the same construction the Python board performs; they can also be loaded
// from exported files, which is how a deployment artifact carries its own copy.
// Nothing is transcribed as constants either way.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace soo {

inline constexpr int kBoardSize = 73;
inline constexpr int kDirections = 6;
inline constexpr int kCamps = 6;
inline constexpr int kCampSize = 10;
inline constexpr int kMaxPlayers = 3;
inline constexpr int kActionSize = kBoardSize * kBoardSize;

// Camp indices follow the Python ``Camp`` enum order: x+ y+ z+ x- y- z-.
inline constexpr int kCampZNeg = 5;

struct Topology {
    bool configured = false;
    std::array<std::array<int8_t, kDirections>, kBoardSize> neighbour{};
    std::array<std::array<uint8_t, kCampSize>, kCamps> camp_positions{};
    std::array<std::array<uint8_t, kBoardSize>, kBoardSize> pairwise{};
    std::array<std::array<uint8_t, kBoardSize>, kCamps> physical_to_canonical{};
    std::array<std::array<uint8_t, kBoardSize>, kCamps> canonical_to_physical{};
};

Topology& mutable_topology();

// Fill the topology from the five exported table files (topology_neighbour.i8,
// topology_{camp_positions,pairwise_distance,physical_to_canonical,
// canonical_to_physical}.i32) written by tools/export_deployment.py and
// tools/build_golden.py.  Returns false and leaves the topology unconfigured
// when any file is missing or the wrong size.  This is the only way a build
// without Python gets its tables: nothing in native/ transcribes them.
bool load_topology_from_dir(const std::string& root);

// Derive every table from the board's geometry: the two overlapping triangles,
// the camp inequalities, the (z, x) ordering that fixes position ids, the
// neighbour lattice, breadth-first distances and the canonical rotation. This
// is what makes the core self-sufficient in geometry -- no Python, no files.
// `topology_test` requires the result to equal the frozen exported tables.
Topology generate_topology();

// Install the generated tables, unless something already configured them.
// Idempotent, and safe to call from an entry point that does not know whether
// an artifact's tables were loaded first.
void ensure_topology_configured();

inline const Topology& topology() {
    const Topology& t = mutable_topology();
    if (!t.configured) {
        throw std::runtime_error(
            "native topology is not configured; call ensure_topology_configured(), "
            "load_topology_from_dir() or diamond_native.configure()");
    }
    return t;
}

}  // namespace soo
