#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "diamond_pipeline/replay.hpp"

namespace diamond_pipeline {

std::vector<std::byte> encode_replay_segment(const Episode& episode,
                                             const Compatibility& compatibility);
Episode decode_replay_segment(std::span<const std::byte> bytes,
                              const Compatibility& compatibility);

}  // namespace diamond_pipeline
