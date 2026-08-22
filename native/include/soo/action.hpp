// Physical action identity: ``source * 73 + destination``.
// Mirrors diamond.alphazero.action_codec.ActionCodec.
#pragma once

#include <cstdint>
#include <stdexcept>

#include "soo/board.hpp"

namespace soo {

inline int32_t encode_action(int source, int destination) {
    return static_cast<int32_t>(source) * kBoardSize + static_cast<int32_t>(destination);
}

inline void decode_action(int32_t action, int& source, int& destination) {
    if (action < 0 || action >= kActionSize) {
        throw std::out_of_range("action id out of range");
    }
    source = action / kBoardSize;
    destination = action % kBoardSize;
}

}  // namespace soo
