// Fixed board topology.  Every table here is injected from Python
// (see src/diamond/alphazero/native/topology.py); nothing is transcribed.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

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

inline const Topology& topology() {
    const Topology& t = mutable_topology();
    if (!t.configured) {
        throw std::runtime_error(
            "native topology is not configured; call diamond_native.configure()");
    }
    return t;
}

}  // namespace soo
