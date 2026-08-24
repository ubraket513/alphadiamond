#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace diamond_training {

struct Compatibility {
    std::string family;
    std::string model_version;

    bool operator==(const Compatibility&) const = default;
};

struct TrainingSample {
    Compatibility compatibility;
    std::vector<float> node_features;
    std::vector<std::pair<int32_t, float>> sparse_policy;
    std::vector<float> value_target;
};

}  // namespace diamond_training
