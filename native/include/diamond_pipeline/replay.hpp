#pragma once

#include <compare>
#include <cstdint>
#include <string>
#include <vector>

#include "diamond_training/training_sample.hpp"

namespace diamond_pipeline {

using Compatibility = diamond_training::Compatibility;
using TrainingSample = diamond_training::TrainingSample;

struct ModelKey {
    std::string model_name;
    std::string model_version;
    std::string checkpoint_sha256;

    auto operator<=>(const ModelKey&) const = default;
};

struct Episode {
    std::string game_id;
    uint64_t seed = 0;
    std::string retry_id;
    ModelKey model_key;
    Compatibility compatibility;
    std::vector<TrainingSample> samples;
    std::vector<int32_t> final_order;
    uint64_t move_count = 0;
    bool completed = true;
    std::string aborted_reason;
};

}  // namespace diamond_pipeline
